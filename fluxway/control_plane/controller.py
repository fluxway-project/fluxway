"""
Controller - Central coordination node for the Control Plane (Ray Actor).

Based on ARCHITECTURE.md Section 2.1

The Controller orchestrates the training job by coordinating:
- Monitor: Cluster state monitoring and telemetry
- Scheduler: Resource allocation via Ray Placement Groups
- SPMDManager: SPMD lifecycle management
- StateManager: Global state and checkpointing
- HeartbeatManager: Driver heartbeat management
- ControllerStateMachine: Lifecycle state management
- AsyncStepManager: SPMD async step difference control

Implemented as a Ray Actor for async operation.
"""

import asyncio
import functools
import inspect
import os
from pathlib import Path
from typing import Any, TypeVar
from collections.abc import Callable, Mapping, Sequence

from fluxway.common.ingestion import (
    CompiledSegment,
    SourceWorkAcknowledgement,
    SourceWorkLease,
    WorkerLayoutDescriptor,
)

import numpy as np
import ray

from fluxway.common.types import (
    CheckpointObjectRef,
    AsyncStepConfig,
    CheckpointCommitPermit,
    CheckpointCut,
    checkpoint_position,
    ControllerEvent,
    ControllerState,
    DriverHeartbeat,
    ExecutionFailure,
    FlowControlState,
    Graph,
    HeartbeatConfig,
    CheckpointAssembly,
    CheckpointManifest,
    SpmdCheckpoint,
    StepReference,
    CheckpointHold,
    DrainBoundary,
    DrainEvent,
    DrainEventKind,
    DrainPhase,
    reduce_drain_boundary,
    RestoreOptions,
)
from fluxway.common.utils import ensure_logging_configured
from fluxway.common.utils.logging import get_logger
from fluxway.control_plane.monitor import Monitor
from fluxway.control_plane.scheduler import Scheduler, build_topology
from fluxway.control_plane.spmd_manager import SPMDManager, SPMDHandle
from fluxway.control_plane.state_manager import StateManager
from fluxway.control_plane.heartbeat import HeartbeatManager
from fluxway.control_plane.flow_control import FlowControlManager
from fluxway.control_plane.state_machine import ControllerStateMachine
from fluxway.control_plane.lifecycle import ExecutionLifecycle
from fluxway.control_plane.async_manager import AsyncStepManager
from fluxway.control_plane.logical_work_ledger import LogicalWorkEpoch
from fluxway.control_plane.components import (
    ControlPlaneComponentRegistry,
    driver_component_id,
)
from fluxway.control_plane.ingestion import (
    IngestionLayoutPlanningComponent,
    LayoutCompileRequest,
    ReaderTopologyDescriptor,
    SourceScanBinding,
    TailLayoutMode,
)
from fluxway.control_plane.checkpointer.repository import validate_checkpoint_value
from fluxway.control_plane.errors import ExecutionFailedError
from fluxway.common.types import (
    CheckpointConfig,
    CheckpointDeadline,
    DEFAULT_SNAPSHOT_TIMEOUT_SECS,
    DEFAULT_SOURCE_MAX_IN_FLIGHT_STEPS,
    SourceControlConfig,
)
from fluxway.control_plane.source_control import SourceControlRuntime
from fluxway.control_plane.source_progress import SourceProgressSupervisor
from fluxway.common.types.drain import (
    DrainAdmission,
    DrainStrategy,
    GraphModeDrainStrategy,
)
from fluxway.rpc.protocol import DataPlaneRegistry
from fluxway.profiler import ProfilerRuntimeFactory
from fluxway.profiler.record import ProfilerSnapshot
from fluxway.placement import (
    PlacementContractError,
    PlacementError,
    PlacementRecipe,
    PlacementRuntimeConfig,
)

logger = get_logger(__name__)


async def _publish_checkpoint_output_step(
    repository_root: str, checkpoint_id: str, attempt_id: str, position: int
) -> None:
    staging_step = Path(repository_root) / "steps" / f".{checkpoint_id}.{attempt_id}"
    published_step = Path(repository_root) / "steps" / str(position)
    if published_step.exists() or published_step.is_symlink():
        raise FileExistsError(
            f"checkpoint output step already exists: {published_step}"
        )
    if not staging_step.is_dir():
        raise FileNotFoundError(
            f"checkpoint output staging root is missing: {staging_step}"
        )
    await asyncio.to_thread(published_step.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(os.replace, staging_step, published_step)


_F = TypeVar("_F", bound=Callable[..., Any])


async def _ensure_source_scan_binding(
    owner: Any,
    *,
    node_id: str,
    operator_name: str,
    config: Any,
) -> SourceScanBinding:
    source = config.operator_specs[operator_name].source
    if source is None:
        raise ValueError("source scan binding requires a source descriptor")
    build_binding = getattr(source, "build_source_scan_binding", None)
    if not callable(build_binding):
        raise TypeError("source descriptor must provide source scan binding")
    return await owner._require_ingestion_planner().open_source_binding(
        build_binding,
        spmd_id=node_id,
        operator_name=operator_name,
        config=config,
    )


def require_manager(manager_attr: str) -> Callable[[_F], _F]:
    """
    Decorator to ensure that a manager component is initialized before calling
    the method.

    A missing manager is always a lifecycle invariant violation. Dropping a
    heartbeat or control-plane event during startup would make the Controller
    state machine lose an event and wait forever, so callers receive the error
    immediately.
    """

    def decorator(func: _F) -> _F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                self = args[0]
                if getattr(self, manager_attr) is None:
                    raise RuntimeError(
                        f"{manager_attr.strip('_')} is not initialized when calling "
                        f"{func.__name__}"
                    )
                return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                self = args[0]
                if getattr(self, manager_attr) is None:
                    raise RuntimeError(
                        f"{manager_attr.strip('_')} is not initialized when calling "
                        f"{func.__name__}"
                    )
                return func(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def require_spmd_manager(func):
    return require_manager("_spmd_manager")(func)


def require_heartbeat_manager(func):
    return require_manager("_heartbeat_manager")(func)


@ray.remote
class ControllerActor:
    """
    Central Controller for the Fluxway Control Plane (Ray Actor).

    Coordinates all components for distributed training orchestration:
    - Initializes and manages the training graph
    - Allocates resources via Scheduler
    - Manages SPMD instances via SPMDManager
    - Tracks state via StateManager
    - Monitors cluster health via Monitor
    - Manages driver heartbeats via HeartbeatManager
    - Controls lifecycle via ControllerStateMachine
    - Enforces async step limits via AsyncStepManager

    Note: Resource-related operations are proxied through Scheduler.
    Controller can get SPMD view from spmd_manager but without resource info.
    """

    # ------------------------------------------------------------------
    # Properties with cloudpickle-safe guards
    #
    # Ray's metaclass calls getattr(object.__new__(cls), attr) during
    # export_actor_class.  __init__ has not run, so backing attrs are
    # absent from __dict__.  Raising AttributeError here is intentional:
    # cloudpickle's _lookup_module_and_qualname wraps the traversal in
    # `except AttributeError` and silently falls back to by-value
    # pickling, which is harmless.  RuntimeError (not a subclass of
    # AttributeError) would propagate and crash serialization.
    # ------------------------------------------------------------------

    @property
    def spmd_manager(self) -> "SPMDManager":
        if "_spmd_manager" not in self.__dict__:
            raise AttributeError("spmd_manager")
        value = self.__dict__["_spmd_manager"]
        if value is None:
            raise RuntimeError("SPMDManager is not initialized")
        return value

    @property
    def scheduler(self) -> "Scheduler":
        if "_scheduler" not in self.__dict__:
            raise AttributeError("scheduler")
        return self.__dict__["_scheduler"]

    @property
    def heartbeat_manager(self) -> "HeartbeatManager":
        if "_heartbeat_manager" not in self.__dict__:
            raise AttributeError("heartbeat_manager")
        value = self.__dict__["_heartbeat_manager"]
        if value is None:
            raise RuntimeError("HeartbeatManager is not initialized")
        return value

    @property
    def async_step_manager(self) -> "AsyncStepManager":
        if "_async_step_manager" not in self.__dict__:
            raise AttributeError("async_step_manager")
        return self.__dict__["_async_step_manager"]

    @property
    def state_manager(self) -> "StateManager":
        if "_state_manager" not in self.__dict__:
            raise AttributeError("state_manager")
        return self.__dict__["_state_manager"]

    @property
    def state_machine(self) -> "ControllerStateMachine":
        if "_state_machine" not in self.__dict__:
            raise AttributeError("state_machine")
        return self.__dict__["_state_machine"]

    @property
    def control_plane_components(self) -> ControlPlaneComponentRegistry:
        if "_control_plane_components" not in self.__dict__:
            raise AttributeError("control_plane_components")
        return self.__dict__["_control_plane_components"]

    @property
    def monitor(self) -> "Monitor":
        if "_monitor" not in self.__dict__:
            raise AttributeError("monitor")
        return self.__dict__["_monitor"]

    @property
    def source_control(self) -> SourceControlRuntime | None:
        return self._source_control

    @property
    def snapshot_timeout_s(self) -> float:
        if self._checkpoint_config is not None:
            return self._checkpoint_config.snapshot_timeout_s
        return float(
            os.environ.get(
                "FLUXWAY_SNAPSHOT_TIMEOUT_SECS",
                str(DEFAULT_SNAPSHOT_TIMEOUT_SECS),
            )
        )

    def __init__(
        self,
        graph: Graph,
        placement_recipe: PlacementRecipe,
        heartbeat_config: HeartbeatConfig | None = None,
        async_step_config: AsyncStepConfig | None = None,
        source_control: SourceControlConfig | None = None,
        checkpoint_config: CheckpointConfig | None = None,
        transport_config: dict[str, str] | None = None,
        profiler_config: Any | None = None,
        placement_runtime: PlacementRuntimeConfig | None = None,
    ):
        """
        Initialize Controller with configuration.

        Args:
            graph: The task graph defining SPMD nodes and edges
            heartbeat_config: Heartbeat configuration
            async_step_config: Async step configuration
            checkpoint_config: Configuration for checkpointer backend
            transport_config: Data plane transport backend per SPMD (e.g. {"*": "ray"})
            profiler_config: Optional profiler runtime configuration
        """
        # Ray actors run in separate worker processes; ensure INFO logs are not dropped.
        ensure_logging_configured()

        self._graph = graph
        self._placement_recipe = placement_recipe
        self._heartbeat_config = heartbeat_config or HeartbeatConfig()
        self._async_step_config = async_step_config or AsyncStepConfig()
        self._checkpoint_config = checkpoint_config
        self._transport_config = transport_config or {}
        self._profiler_config = profiler_config
        self._profiler_runtime = ProfilerRuntimeFactory.from_config(
            profiler_config,
            process_name="controller",
        )
        self._execution_sequence = 0
        self._heartbeat_observer = self._profiler_runtime.heartbeat_observer()

        self._monitor = Monitor()
        self._state_manager = StateManager()
        self._scheduler = Scheduler(
            spmd_manager=None,
            graph=graph,
            placement_recipe=placement_recipe,
            placement_runtime=placement_runtime or PlacementRuntimeConfig(),
        )
        self._state_machine = ControllerStateMachine(self)
        self._async_step_manager = AsyncStepManager(
            self._graph, self._async_step_config
        )
        self._flow_control_manager = FlowControlManager(
            self._graph, self._async_step_config
        )
        # checkpoint_id → {expected, received, metadata, event}
        self._pending_snapshots: dict[str, Any] = {}
        self._snapshot_publication_context: dict[str, tuple[str, str, int]] = {}
        if (
            checkpoint_config is not None
            and self._graph.requires_source_control_runtime
            and source_control is None
        ):
            raise ValueError(
                "checkpointed ordered-source graphs require explicit source_control"
            )
        if source_control is not None:
            resolved_source_control = source_control
        elif self._graph.requires_source_control_runtime:
            resolved_source_control = SourceControlConfig(
                max_in_flight_steps=DEFAULT_SOURCE_MAX_IN_FLIGHT_STEPS
            )
        else:
            resolved_source_control = None

        if (
            source_control is not None
            and not self._graph.requires_source_control_runtime
        ):
            raise ValueError(
                "source_control requires a graph source descriptor that declares "
                "requires_source_control_runtime"
            )

        self._source_control = (
            None
            if resolved_source_control is None
            else SourceControlRuntime(resolved_source_control)
        )
        self._control_plane_components = ControlPlaneComponentRegistry()
        self._drain_strategy: DrainStrategy | None = (
            GraphModeDrainStrategy(checkpoint_config.interval_steps)
            if checkpoint_config is not None
            else (
                GraphModeDrainStrategy(DEFAULT_SOURCE_MAX_IN_FLIGHT_STEPS)
                if self._source_control is not None
                else None
            )
        )
        self._initialized = False
        self._ingestion_planner: IngestionLayoutPlanningComponent | None = None
        if self._graph.requires_source_control_runtime:
            self.install_ingestion_layout_planning_component(
                IngestionLayoutPlanningComponent("ingestion-planning")
            )
            if self._source_control is None:
                raise RuntimeError(
                    "source-control graph requires a SourceControlRuntime"
                )
            self.control_plane_components.register(self._source_control)

        self._checkpoint_task: asyncio.Task | None = None
        self._drain_boundary = DrainBoundary()
        self._drain_boundary_lock = asyncio.Lock()
        self._source_window_activation_lock = asyncio.Lock()
        self._execution_generation = 0
        self._admission: DrainAdmission | None = None
        self._execution_lifecycle = ExecutionLifecycle(self)
        self._source_progress_supervisor = (
            None
            if self._source_control is None
            else SourceProgressSupervisor(
                self._maybe_activate_next_source_window,
                on_failure=self._on_source_progress_failure,
            )
        )

        # Track source nodes (entry points)
        self._source_nodes = (
            self._graph.source_nodes
            if self._graph.source_nodes
            else self._find_source_nodes()
        )

        self._spmd_manager: SPMDManager | None = None
        self._heartbeat_manager: HeartbeatManager | None = None
        self._checkpointer = None
        self._drained_drivers: set = set()
        self._ready_event = asyncio.Event()

        logger.debug(f"ControllerActor created with {len(self._graph.nodes)} nodes")

    def _find_source_nodes(self) -> list[str]:
        """Find nodes with no predecessors (source nodes)."""
        return list(self._graph.require_compiled_graph().source_spmd_ids)

    def get_resource_allocation_summary(self) -> dict[str, Any]:
        """Expose scheduler resource summaries for debugging and inspection."""
        if self.scheduler.allocation_report is not None:
            return self.scheduler.allocation_report.model_dump(exclude={"recipe"})
        if self.scheduler.resolved_plan is not None:
            return self.scheduler.resolved_plan.model_dump(exclude={"recipe"})
        raise PlacementContractError("resource allocation has not been resolved")

    async def initialize(self) -> bool:
        """
        Initialize the Controller (event-driven).

        Performs:
        1. Set controller handle on spmd_manager
        2. Register SPMDs for heartbeat monitoring
        3. Allocate resources
        4. Build topology data structure (no remote calls)
        5. Start SPMDs (fire-and-forget)
        6. Transition to PENDING state

        Subsequent phases are driven by heartbeat events:
        - PENDING → TOPOLOGY_REGISTERING: when all drivers report workers_ready
        - TOPOLOGY_REGISTERING → READY: when all worker runtimes are ready

        Note: Heartbeat timeout monitoring starts in RUNNING state entry,
        not here. Heartbeat *reception* works from the start.

        Returns:
            True if initialization successful
        """
        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if profiler_recorder is not None:
            span = profiler_recorder.controller_initialize(
                spmd_count=len(self._graph.nodes)
            )
            await span.__aenter__()
        else:
            span = None
        try:
            # Create sub-components locally in the controller process
            self_handle = ray.get_runtime_context().current_actor
            self._spmd_manager = SPMDManager(
                controller_handle=self_handle,
            )
            self.scheduler.set_spmd_manager(
                self.spmd_manager
            )  # Set SPMDManager on scheduler

            self._heartbeat_manager = HeartbeatManager(
                config=self._heartbeat_config,
                on_all_ready=self._on_all_drivers_ready,
                on_all_topology_ready=self._on_all_topology_ready,
                on_driver_error=self._on_driver_error,
                observer=self._heartbeat_observer,
            )

            if self._checkpoint_config is not None:
                from fluxway.control_plane.checkpointer import (
                    CheckpointComponentRegistry,
                    FluxwayCheckpointer,
                )

                component_registries = {
                    (node.spmd_config.spmd_id or node_id): CheckpointComponentRegistry(
                        tuple(node.spmd_config.checkpoint_components)
                    )
                    for node_id, node in self._graph.nodes.items()
                    if node.spmd_config is not None
                }
                self._checkpointer = await FluxwayCheckpointer.create(
                    self,
                    repository_dir=self._checkpoint_config.repository_dir,
                    interval_steps=self._checkpoint_config.interval_steps,
                    snapshot_timeout_s=self._checkpoint_config.snapshot_timeout_s,
                    inactivity_timeout_s=self._checkpoint_config.inactivity_timeout_s,
                    retention_policy=self._checkpoint_config.retention_policy,
                    retention_domains=self._checkpoint_config.retention_domains,
                    component_registries=component_registries,
                    restore_scope=self._checkpoint_config.restore_scope,
                    restore_spmd_bindings=self._checkpoint_config.restore_spmd_bindings,
                )
            # Assign specific max_steps to each node based on global config and overrides
            for node_id, node in self._graph.nodes.items():
                if node.spmd_config is not None:
                    node.spmd_config.profiler_config = self._profiler_config
                    limit = self._async_step_config.node_max_steps.get(
                        node_id, self._async_step_config.max_steps
                    )
                    if limit is not None:
                        node.spmd_config.max_steps = limit
                        logger.debug(f"Assigned max_steps={limit} to node {node_id}")

                    node.spmd_config.auto_start_source_operators = False
                    if self._drain_strategy is not None:
                        node.spmd_config.requires_drain_at_startup = True
                    if self._checkpoint_config is not None:
                        node.spmd_config.inactivity_timeout_s = (
                            self._checkpoint_config.inactivity_timeout_s
                        )

            # Register SPMDs for heartbeat monitoring
            for node_name in self._graph.nodes:
                self.heartbeat_manager.register_spmd(node_name)

            # Allocate resources
            if profiler_recorder is None:
                await self.scheduler.allocate_resources()
            else:
                async with profiler_recorder.spmd_manager_placement_group_create(
                    spmd_id="*",
                    worker_count=sum(
                        node.spmd_config.num_workers
                        for node in self._graph.nodes.values()
                        if node.spmd_config is not None
                    ),
                ):
                    await self.scheduler.allocate_resources()

            # Build and establish topology data structure
            # (adjacency lists only, no remote calls to drivers)
            if profiler_recorder is None:
                self.scheduler.establish_topology(
                    build_topology(self._graph.require_compiled_graph())
                )
            else:
                async with profiler_recorder.controller_topology_registration(
                    spmd_count=len(self._graph.nodes),
                    edge_count=len(self._graph.require_compiled_graph().dependencies),
                ):
                    self.scheduler.establish_topology(
                        build_topology(self._graph.require_compiled_graph())
                    )

            # Start SPMDs (fire-and-forget, readiness tracked via heartbeat)
            if profiler_recorder is None:
                await self.scheduler.init_spmds()
            else:
                async with profiler_recorder.spmd_manager_driver_start(
                    spmd_id="*",
                    worker_count=len(self._graph.nodes),
                ):
                    await self.scheduler.init_spmds()

            # Transition state machine: INITIALIZING → PENDING
            await self.state_machine.trigger_event(ControllerEvent.INIT_COMPLETE)

            self._initialized = True
            logger.debug(
                f"Controller initialized with {len(self._graph.nodes)} SPMDs, "
                f"waiting for drivers to report ready via heartbeat"
            )
            return True

        except PlacementError as e:
            self.scheduler.abort_initialization(e)
            logger.error(f"Controller placement initialization failed: {e}")
            raise
        except Exception as e:
            self.scheduler.abort_initialization(e)
            logger.error(f"Controller initialization failed: {e}")
            raise
        finally:
            if span is not None:
                await span.__aexit__(None, None, None)

    async def start(
        self,
        resume: str | int | None = None,
        *,
        restore_options: RestoreOptions | None = None,
    ) -> int | None:
        if self._initialized:
            raise RuntimeError("Controller is already initialized or started")

        return await self._execution_lifecycle.start(
            resume,
            restore_options=restore_options,
        )

    # =====================================================================
    # Properties (as methods for Ray Actor)
    # =====================================================================

    def get_graph(self) -> Graph:
        """Get the task graph."""
        return self._graph

    def get_state(self) -> ControllerState:
        """Get current controller state."""
        return self.state_machine.state

    async def get_status(self) -> dict[str, Any]:
        """Get detailed controller status."""
        return await self.get_cluster_status()

    @require_heartbeat_manager
    async def get_all_heartbeats(self) -> dict[str, Any]:
        """Get all latest heartbeats from all SPMDs."""
        return await self.heartbeat_manager.get_all_heartbeats()

    async def get_async_step_progress(self) -> dict[str, dict[str, int]]:
        steps = self.async_step_manager.get_all_spmd_steps()
        return {
            spmd_id: dict(spmd_step.completed_step_count)
            for spmd_id, spmd_step in steps.items()
        }

    def is_initialized(self) -> bool:
        """Check if controller is initialized."""
        return self._initialized

    # =====================================================================
    # SPMD Handle Access
    # =====================================================================

    @require_spmd_manager
    def get_spmd_handle(self, spmd_id: str) -> SPMDHandle | None:
        """
        Get handle to an SPMD instance from spmd_manager.

        Args:
            spmd_id: ID of the SPMD

        Returns:
            SPMDHandle or None if not found
        """
        return self.spmd_manager.get_spmd_handle(spmd_id)

    @require_spmd_manager
    async def dispatch_driver_component(
        self, spmd_id: str, component_id: str, request: object
    ) -> object:
        if not isinstance(spmd_id, str) or not spmd_id:
            raise ValueError("driver component dispatch requires a spmd_id")
        if not isinstance(component_id, str) or not component_id:
            raise ValueError("driver component dispatch requires a component_id")
        driver = self.spmd_manager.get_driver_handle(spmd_id)
        if driver is None:
            raise RuntimeError(
                f"driver component dispatch has no Driver for {spmd_id!r}"
            )
        return await driver.dispatch_component_request.remote(component_id, request)

    async def dispatch_control_plane_request(self, request: object) -> object:
        """Route one typed worker notice to the sole owning Controller component."""
        return await self.control_plane_components.dispatch_unaddressed_request(request)

    def _record_execution_phase(
        self,
        phase: str,
        *,
        op_name: str | None,
        step: int | None,
        epoch: int,
    ) -> None:
        if not self._profiler_runtime.enabled:
            return
        self._execution_sequence += 1
        self._profiler_runtime.recorder.controller_execution_phase(
            phase=phase,
            op_name=op_name,
            step=step,
            epoch=epoch,
            sequence=self._execution_sequence,
            execution_generation=getattr(self, "_execution_generation", 0),
        )

    async def activate_ingestion_layout(
        self,
        spmd_id: str,
        operator_name: str,
        request: LayoutCompileRequest,
        *,
        epoch: int,
        start_position: int,
    ) -> None:
        if not isinstance(spmd_id, str) or not spmd_id:
            raise ValueError("ingestion activation planning requires spmd_id")
        if not isinstance(operator_name, str) or not operator_name:
            raise ValueError("ingestion activation planning requires operator_name")
        if not isinstance(request, LayoutCompileRequest):
            raise TypeError("ingestion activation requires LayoutCompileRequest")
        planner = self._require_ingestion_planner()
        activation = await planner.plan_and_publish(
            self._require_source_control().work_ledger,
            request,
            epoch=epoch,
            start_position=start_position,
            max_units=self._require_source_control().planning_limits[0],
            max_records=self._require_source_control().planning_records,
            max_steps=self._require_source_control().planning_window_steps,
            max_descriptor_bytes=self._require_source_control().planning_limits[1],
        )
        record_execution_phase = getattr(self, "_record_execution_phase", None)
        if record_execution_phase is not None:
            record_execution_phase(
                "activation.publish",
                op_name=operator_name,
                step=activation.epoch.start_position,
                epoch=activation.epoch.epoch,
            )
        await self.dispatch_driver_component(
            spmd_id,
            driver_component_id(spmd_id, operator_name),
            activation,
        )

    async def activate_source_scan_layout_from_descriptor(self) -> None:
        async with self._source_window_activation_lock:
            node_id, operator_name, config = self._source_layout_target()
            planner = self._require_ingestion_planner()
            binding = await _ensure_source_scan_binding(
                self,
                node_id=node_id,
                operator_name=operator_name,
                config=config,
            )
            planner.bind_source_identity(binding.identity)
            adapter = binding.adapter
            topology = binding.topology
            cursor = planner.resume_cursor
            tail_layout_mode = binding.tail_layout_mode
            assignment_policy = binding.assignment_policy
            if not isinstance(topology, ReaderTopologyDescriptor):
                raise TypeError(
                    "source descriptor must return ReaderTopologyDescriptor"
                )
            if not isinstance(tail_layout_mode, TailLayoutMode):
                raise TypeError("source descriptor must return TailLayoutMode")
            planner.bind_source_scan(
                adapter,
                initial_cursor_factory=binding.initial_cursor_factory,
                budget=self._require_source_control().scan_budget,
                tail_layout_mode=tail_layout_mode,
                assignment_policy=assignment_policy,
            )
            activation = await planner.plan_source_scan(
                adapter,
                binding.initial_cursor_factory,
                self._require_source_control().work_ledger,
                cursor,
                budget=self._require_source_control().scan_budget,
                topology=topology,
                tail_layout_mode=tail_layout_mode,
                assignment_policy=assignment_policy,
                epoch=0,
                start_position=planner.resume_step,
                max_steps=self._require_source_control().planning_window_steps,
                max_descriptor_bytes=self._require_source_control().planning_limits[1],
            )
            if activation is None:
                return
            record_execution_phase = getattr(self, "_record_execution_phase", None)
            if record_execution_phase is not None:
                record_execution_phase(
                    "activation.publish",
                    op_name=operator_name,
                    step=activation.epoch.start_position,
                    epoch=activation.epoch.epoch,
                )
            await self.dispatch_driver_component(
                node_id,
                driver_component_id(node_id, operator_name),
                activation,
            )

    # =====================================================================
    # Heartbeat Interface (called by drivers)
    # =====================================================================

    @require_heartbeat_manager
    async def receive_driver_heartbeat(self, heartbeat: DriverHeartbeat) -> None:
        """
        Receive heartbeat from a driver.

        Args:
            heartbeat: Driver heartbeat data
        """
        if self._heartbeat_observer is not None:
            async with self._heartbeat_observer.receive_driver_heartbeat(heartbeat):
                await self._receive_driver_heartbeat_unprofiled(heartbeat)
            return
        await self._receive_driver_heartbeat_unprofiled(heartbeat)

    async def _receive_driver_heartbeat_unprofiled(
        self, heartbeat: DriverHeartbeat
    ) -> None:
        await self.heartbeat_manager.receive_heartbeat(heartbeat)

    # =====================================================================
    # Op Completion Interface (called by drivers)
    # =====================================================================

    @require_spmd_manager
    async def report_op_completed(
        self, spmd_id: str, op_name: str, completed_count: int
    ) -> None:
        """
        Receive op completion report from a driver.

        Args:
            spmd_id: SPMD identifier
            op_name: Operation name
            completed_count: Completed count (number of steps finished)
        """
        self.async_step_manager.report_op_completed(spmd_id, op_name, completed_count)
        windows = self._flow_control_manager.observe_op_completed(
            spmd_id, op_name, completed_count
        )
        await self._apply_flow_control_windows(windows)
        supervisor = self._source_progress_supervisor
        if supervisor is not None:
            supervisor.wake()
        await self._raise_source_progress_failure()
        if (
            self._drain_strategy is None
            and self.async_step_manager.is_all_completed()
            and self.state_machine.is_running()
        ):
            await self.state_machine.trigger_event(
                ControllerEvent.ALL_DRIVERS_COMPLETED
            )

    async def _maybe_activate_next_source_window(self) -> None:
        if self._ingestion_planner is None or self._source_control is None:
            return
        node_id, operator_name, _ = self._source_layout_target()
        progress = self.async_step_manager.get_spmd_steps(node_id)
        if progress is None:
            return
        completed_count = progress.completed_step_count.get(operator_name, 0)
        planner = self._ingestion_planner
        if not planner.windows:
            await self.activate_source_scan_layout_from_descriptor()
            return
        latest = planner.windows[-1]
        lead = self._source_control.window_lead_steps
        if latest.work_epoch.terminal or completed_count < max(
            latest.work_epoch.start_step,
            latest.work_epoch.end_step - lead,
        ):
            return
        async with self._source_window_activation_lock:
            if planner.windows[-1].work_epoch.terminal:
                return
            latest = planner.windows[-1].work_epoch
            if latest.terminal or completed_count < max(
                latest.start_step,
                latest.end_step - self._source_control.window_lead_steps,
            ):
                return
            activation = await planner.plan_next_window(
                self._source_control.work_ledger,
                max_units=self._source_control.planning_limits[0],
                max_records=self._source_control.planning_records,
                max_steps=self._source_control.planning_window_steps,
                max_descriptor_bytes=self._source_control.planning_limits[1],
            )
            if activation is None:
                return
            record_execution_phase = getattr(self, "_record_execution_phase", None)
            if record_execution_phase is not None:
                record_execution_phase(
                    "activation.publish",
                    op_name=operator_name,
                    step=activation.epoch.start_position,
                    epoch=activation.epoch.epoch,
                )
            await self.dispatch_driver_component(
                node_id,
                driver_component_id(node_id, operator_name),
                activation,
            )

    async def _on_source_progress_failure(self, error: BaseException) -> None:
        logger.error("source progress supervisor failed", exc_info=error)
        if self.state_machine.is_running():
            await self._execution_lifecycle.fail(error, "SOURCE_PROGRESS")

    async def _raise_source_progress_failure(self) -> None:
        supervisor = self._source_progress_supervisor
        if supervisor is not None:
            await supervisor.raise_if_failed()

    async def report_boundary_event(
        self, spmd_id: str, op_name: str, event: DrainEvent
    ) -> None:
        if isinstance(spmd_id, bool) or not isinstance(spmd_id, str) or not spmd_id:
            raise TypeError("boundary event SPMD ID must be a non-empty string")
        if isinstance(op_name, bool) or not isinstance(op_name, str) or not op_name:
            raise TypeError("boundary event operator name must be a non-empty string")
        if not isinstance(event, DrainEvent):
            raise TypeError("boundary event must be a DrainEvent")
        if event.kind is not DrainEventKind.TERMINAL_OBSERVATION:
            raise ValueError(
                "worker boundary reports only terminal observations, "
                f"got {event.kind.value}"
            )
        compiled_graph = self._graph.require_compiled_graph()
        try:
            operator = compiled_graph.spmd(spmd_id).operator(op_name)
        except KeyError as error:
            raise ValueError(
                f"boundary event references unknown operator {spmd_id}.{op_name}"
            ) from error
        if not operator.spec.is_source:
            raise ValueError(
                f"boundary event requires a source operator, got {spmd_id}.{op_name}"
            )
        await self._reduce_drain_event(event)

    def _require_source_control(self) -> SourceControlRuntime:
        if self._source_control is None:
            raise RuntimeError("ordered control streams require source_control runtime")
        return self._source_control

    async def publish_source_work_epoch(
        self, epoch: LogicalWorkEpoch, bindings: Sequence[WorkerLayoutDescriptor]
    ) -> None:
        await self._require_source_control().publish_work_epoch(epoch, bindings)

    async def acquire_source_work(
        self, epoch: int, segments: Sequence[CompiledSegment]
    ) -> tuple[SourceWorkLease, ...]:
        return await self._require_source_control().acquire_work(epoch, segments)

    async def acknowledge_source_work(
        self, acknowledgements: list[SourceWorkAcknowledgement]
    ) -> None:
        await self._require_source_control().acknowledge_work(acknowledgements)
        supervisor = self._source_progress_supervisor
        if supervisor is not None:
            supervisor.wake()
        await self._raise_source_progress_failure()

    async def candidate_source_work_cut(self) -> int:
        return await self._require_source_control().candidate_work_cut()

    async def wait_for_source_work_cut(self, target: int) -> int:
        return await self._require_source_control().wait_for_work_cut(target)

    async def _reduce_drain_event(self, event: DrainEvent) -> None:
        cancelled_task = None
        admission = None
        async with self._drain_boundary_lock:
            previous = self._drain_boundary
            transition = reduce_drain_boundary(previous, event)
            self._drain_boundary = transition.boundary
            target = transition.dispatch_target
            if target is not None:
                max_steps = self.async_step_manager._config.max_steps
                if max_steps is not None:
                    target = min(target, max_steps)
                admission = DrainAdmission(
                    transition.boundary.revision,
                    target,
                    transition.boundary.observed_cut,
                )
                self._admission = admission
                self._drained_drivers.clear()
                if (
                    previous.phase is DrainPhase.SNAPSHOTTING
                    and previous.admitted_cut is not None
                    and target < previous.admitted_cut
                ):
                    task = self._checkpoint_task
                    if (
                        task is not None
                        and task is not asyncio.current_task()
                        and not task.done()
                    ):
                        task.cancel()
                        cancelled_task = task
                        self._checkpoint_task = None
        if cancelled_task is not None:
            try:
                await cancelled_task
            except asyncio.CancelledError:
                pass
        if admission is not None:
            await self._execution_lifecycle.deliver_current()

    def _source_spmd_ids(self) -> tuple[str, ...]:
        return self._graph.source_control_spmd_ids

    async def _notify_control_plane_components(self, state: ControllerState) -> None:
        supervisor = self._source_progress_supervisor
        operations = [self.control_plane_components.notify_state(state)]
        if state is ControllerState.RUNNING:
            if supervisor is not None:
                operations.append(supervisor.start())
        elif state in (
            ControllerState.COMPLETING,
            ControllerState.FAILING,
            ControllerState.CANCELLING,
        ):
            if supervisor is not None:
                operations.append(supervisor.stop())
        await asyncio.gather(*operations)

    def install_ingestion_layout_planning_component(
        self, component: IngestionLayoutPlanningComponent
    ) -> None:
        if not isinstance(component, IngestionLayoutPlanningComponent):
            raise TypeError("Controller requires an IngestionLayoutPlanningComponent")
        if self._initialized:
            raise RuntimeError(
                "ingestion layout planning component must be installed before "
                "Controller initialization"
            )
        if self._ingestion_planner is not None:
            raise RuntimeError("Controller already has an ingestion layout planner")
        self.control_plane_components.register(component)
        self._ingestion_planner = component

    def _require_ingestion_planner(self) -> IngestionLayoutPlanningComponent:
        if self._ingestion_planner is None:
            raise RuntimeError("source control requires an ingestion planner")
        return self._ingestion_planner

    def _source_layout_target(self):
        targets = []
        for node_id, node in self._graph.nodes.items():
            config = node.spmd_config
            if config is None:
                continue
            for operator_name, spec in config.operator_specs.items():
                if not (spec.is_source and spec.requires_source_control_runtime):
                    continue
                targets.append((node_id, operator_name, config))
        if len(targets) != 1:
            raise ValueError(
                "a logical source clock requires exactly one controlled source operator"
            )
        return targets[0]

    async def activate_source_layout_from_descriptor(self) -> None:
        node_id, operator_name, _ = self._source_layout_target()
        planner = self._require_ingestion_planner()
        if not planner.windows:
            await self.activate_source_scan_layout_from_descriptor()
            return
        for compiled in planner.windows:
            epoch = compiled.work_epoch
            if (
                epoch.end_step
                <= self._require_source_control().work_ledger.committed_cut
            ):
                continue
            await self.dispatch_driver_component(
                node_id,
                driver_component_id(node_id, operator_name),
                compiled.to_source_layout_activation(
                    epoch=epoch.epoch,
                    start_position=epoch.start_step,
                    end_position=epoch.end_step,
                    terminal=epoch.terminal,
                ),
            )

    def control_stream_cuts(
        self, spmd_payload: Mapping[str, SpmdCheckpoint]
    ) -> dict[str, int]:
        if not isinstance(spmd_payload, Mapping):
            raise TypeError("SPMD checkpoint payload must be a mapping")
        if any(
            not isinstance(value, SpmdCheckpoint) for value in spmd_payload.values()
        ):
            raise TypeError("SPMD checkpoint entries must be SpmdCheckpoint values")
        return {}

    def checkpoint_state(
        self, *, cuts: Mapping[str, int], cut: CheckpointCut
    ) -> dict[str, Any]:
        """Return all control-plane progress needed by a global checkpoint."""
        if not isinstance(cut, CheckpointCut):
            raise TypeError(
                f"controller checkpoint requires CheckpointCut, got {type(cut).__name__}"
            )
        snapshot_boundary = self._drain_boundary
        if (
            snapshot_boundary.admitted_cut is not None
            and snapshot_boundary.drained_cut is not None
            and snapshot_boundary.drained_cut >= snapshot_boundary.admitted_cut
        ):
            snapshot_boundary = DrainBoundary(
                phase=(
                    DrainPhase.COMPLETED
                    if snapshot_boundary.observed_cut is not None
                    else DrainPhase.OPEN
                ),
                observed_cut=snapshot_boundary.observed_cut,
                admitted_cut=snapshot_boundary.admitted_cut,
                drained_cut=snapshot_boundary.drained_cut,
                committed_cut=snapshot_boundary.admitted_cut,
                revision=snapshot_boundary.revision + 1,
            )
        state = {
            "components": self.control_plane_components.checkpoint_state(cut),
            "drain_boundary": snapshot_boundary.to_dict(),
        }
        validate_checkpoint_value(state)
        return state

    async def restore_checkpoint_state(
        self, state: dict[str, Any], cut: CheckpointCut
    ) -> None:
        """Restore control-plane progress before any SPMD source is resumed."""
        if not isinstance(cut, CheckpointCut):
            raise TypeError(
                f"controller restore requires CheckpointCut, got {type(cut).__name__}"
            )
        if not isinstance(state, dict):
            raise TypeError("checkpoint control state must be a dictionary")
        expected_fields = {"components", "drain_boundary"}
        if set(state) != expected_fields:
            raise ValueError(
                "checkpoint control state fields do not match protocol: "
                f"expected={sorted(expected_fields)} actual={sorted(state)}"
            )
        if self._source_control is not None:
            node_id, operator_name, config = self._source_layout_target()
            planner = self._require_ingestion_planner()
            binding = await _ensure_source_scan_binding(
                self,
                node_id=node_id,
                operator_name=operator_name,
                config=config,
            )
            planner.bind_source_identity(binding.identity)
            if not isinstance(binding.topology, ReaderTopologyDescriptor):
                raise TypeError(
                    "source descriptor must return ReaderTopologyDescriptor"
                )
            planner.set_reader_topology(binding.topology)
        restored_boundary = DrainBoundary.from_mapping(state["drain_boundary"])
        if restored_boundary.committed_cut is None:
            raise ValueError(
                "checkpoint control state drain_boundary requires committed_cut"
            )
        self._drain_boundary = DrainBoundary.from_committed_cut(
            restored_boundary.committed_cut
        )
        components = state.get("components")
        if not isinstance(components, Mapping):
            raise ValueError("checkpoint control state is missing components")
        self.control_plane_components.restore_state(components, cut)

    def prepare_checkpoint_transaction(
        self,
        checkpoint_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        control_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        components = control_state.get("components")
        if not isinstance(components, Mapping):
            raise ValueError("checkpoint control state is missing components")
        try:
            return {
                "components": self.control_plane_components.prepare_checkpoint_transaction(
                    checkpoint_id, permit, cut, components
                )
            }
        except BaseException:
            self.control_plane_components.abort_checkpoint_transaction(
                checkpoint_id, permit
            )
            raise

    def apply_prepared_checkpoint_transaction(
        self,
        checkpoint_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        control_state: Mapping[str, Any],
        prepared_state: Mapping[str, Any],
    ) -> None:
        if cut.position is None:
            raise ValueError("checkpoint transaction requires a positioned cut")
        components = control_state.get("components")
        if not isinstance(components, Mapping):
            raise ValueError("checkpoint control state is missing components")
        self.control_plane_components.apply_prepared_checkpoint_transaction(
            checkpoint_id, permit, cut, components, prepared_state
        )
        if self._source_control is not None:
            self._require_ingestion_planner().reclaim_before(cut.position)

    def abort_checkpoint_transaction(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        self.control_plane_components.abort_checkpoint_transaction(
            checkpoint_id, permit
        )

    @require_spmd_manager
    async def publish_snapshot(
        self,
        checkpoint_id: str,
        permit: CheckpointCommitPermit,
        deadline: CheckpointDeadline | None = None,
    ) -> None:
        tasks = []
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is None:
                raise RuntimeError(
                    f"snapshot checkpoint_object commit is missing Driver {spmd_id!r}"
                )
            tasks.append(driver.publish_snapshot.remote(checkpoint_id, permit))
        if deadline is not None:
            results = await deadline.run_phase(asyncio.gather(*tasks))
        else:
            results = await asyncio.gather(*tasks)
        context = self._snapshot_publication_context.get(checkpoint_id)
        if not any(result is True for result in results):
            self._snapshot_publication_context.pop(checkpoint_id, None)
            return
        if context is None:
            raise RuntimeError(
                f"checkpoint output publication context is missing for {checkpoint_id!r}"
            )
        repository_root, attempt_id, position = context
        await _publish_checkpoint_output_step(
            repository_root, checkpoint_id, attempt_id, position
        )
        self._snapshot_publication_context.pop(checkpoint_id, None)

    @require_spmd_manager
    async def rollback_snapshot(
        self, checkpoint_id: str, permit: CheckpointCommitPermit
    ) -> None:
        tasks = []
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is None:
                raise RuntimeError(
                    f"snapshot checkpoint_object rollback is missing Driver {spmd_id!r}"
                )
            tasks.append(driver.rollback_snapshot.remote(checkpoint_id, permit))
        await asyncio.gather(*tasks)

    abort_snapshot = rollback_snapshot

    async def report_flow_control_state(self, state: FlowControlState) -> None:
        self._flow_control_manager.report_state(state)

    @require_spmd_manager
    async def _apply_flow_control_windows(
        self, windows_by_spmd_id: dict[str, tuple[Any, ...]]
    ) -> None:
        apply_tasks = []
        for spmd_id, windows in windows_by_spmd_id.items():
            if not windows:
                continue
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is None:
                raise RuntimeError(
                    f"Missing driver handle for SPMD {spmd_id!r} "
                    "while applying flow-control windows."
                )
            apply_tasks.append(driver.apply_credit_window.remote(windows))
        if apply_tasks:
            await asyncio.gather(*apply_tasks)

    async def report_driver_drained(
        self, spmd_id: str, admission: DrainAdmission
    ) -> None:
        completed = False
        async with self._drain_boundary_lock:
            participants = set(self.spmd_manager.get_all_spmd_ids())
            if spmd_id not in participants:
                raise ValueError(f"unregistered Driver {spmd_id!r}")
            if admission != self._admission:
                raise ValueError("Driver completion belongs to a stale admission")
            if self._drain_boundary.phase is not DrainPhase.DRAINING:
                if spmd_id in self._drained_drivers:
                    return
                raise ValueError("Driver completion requires an active drain")
            self._drained_drivers.add(spmd_id)
            if self._drained_drivers == participants:
                self._drain_boundary = reduce_drain_boundary(
                    self._drain_boundary,
                    DrainEvent(DrainEventKind.DRAIN_COMPLETE, admission.target),
                ).boundary
                completed = True
        if completed:
            await self._on_all_drivers_drained()

    async def _issue_initial_drain(self) -> None:
        """Issues the very first drain target to unlock workers at step 0."""
        if self._drain_strategy is not None:
            initial_target = self._drain_strategy.get_initial_drain_target(
                self._drain_boundary.committed_cut or 0
            )
            if initial_target is not None:
                max_steps = getattr(self.async_step_manager._config, "max_steps", None)
                if isinstance(max_steps, int):
                    initial_target = min(initial_target, max_steps)
                logger.info(f"Issuing initial drain target: {initial_target}")
                await self._schedule_drain(initial_target)

    async def _schedule_drain(self, target: int) -> None:
        """Broadcast a new drain target to all drivers."""
        max_steps = getattr(self.async_step_manager._config, "max_steps", None)
        if isinstance(max_steps, int):
            target = min(target, max_steps)

        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.controller_drain(
                target_step=target,
                spmd_count=len(self._graph.nodes),
            ):
                await self._reduce_drain_event(
                    DrainEvent(DrainEventKind.PERIODIC_REQUEST, target)
                )
            return
        await self._reduce_drain_event(
            DrainEvent(DrainEventKind.PERIODIC_REQUEST, target)
        )

    async def _on_all_drivers_drained(self) -> None:
        await self._execution_lifecycle.drained()

    async def _resume_and_schedule_next_drain(self) -> None:
        await self._execution_lifecycle.resume()

    @require_spmd_manager
    async def _lock_spmd_op(self, spmd_id: str, op_name: str) -> None:
        """Lock an SPMD's operation."""
        driver = self.spmd_manager.get_driver_handle(spmd_id)
        if driver is not None:
            try:
                await driver.block_op.remote(op_name)
                # logger.debug(f"Locked {spmd_id} {op_name}")
            except Exception as e:
                logger.error(f"Failed to lock {spmd_id} {op_name}: {e}")

    @require_spmd_manager
    async def _unlock_spmd_op(self, spmd_id: str, op_name: str) -> None:
        """Unlock an SPMD's operation."""
        driver = self.spmd_manager.get_driver_handle(spmd_id)
        if driver is not None:
            try:
                await driver.unblock_op.remote(op_name)
                # logger.debug(f"Unlocked {spmd_id} {op_name}")
            except Exception as e:
                logger.error(f"Failed to unlock {spmd_id} {op_name}: {e}")

    # =====================================================================
    # State Argument Transport
    # =====================================================================

    @require_spmd_manager
    async def receive_control_argument(
        self,
        source_spmd_id: str,
        target_spmd_id: str,
        target_op: str,
        names: list[str],
        step: int,
        values: list[Any],
    ) -> None:
        """
        Receive control_arguments and forward to target SPMD.

        Transport path: rank0-worker -> source-driver -> controller
                       -> target-driver -> all-workers

        Args:
            source_spmd_id: Source SPMD ID
            target_spmd_id: Target SPMD ID
            target_op: Target operator type on the receiving side
            names: State variable names
            step: Current step number
            values: State values
        """
        target_driver = self.spmd_manager.get_driver_handle(target_spmd_id)
        if target_driver is None:
            logger.error(
                f"Target SPMD {target_spmd_id} not found for control_arguments"
            )
            return

        try:
            await target_driver.broadcast_control_argument.remote(
                target_op, names, step, values
            )
            logger.debug(
                f"Forwarded control_arguments {names} from {source_spmd_id} "
                f"to {target_spmd_id}, target_op={target_op}, step={step}"
            )
        except Exception as e:
            logger.error(f"Failed to forward control_arguments: {e}")

    # =====================================================================
    # Global Snapshotting
    # =====================================================================

    @require_spmd_manager
    async def request_snapshot(
        self,
        checkpoint_id: str,
        attempt_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
        repository_root: str,
        deadline: CheckpointDeadline | None = None,
    ) -> CheckpointAssembly:
        logger.info(
            "Requesting global snapshot '%s' at cut %s",
            checkpoint_id,
            cut.cut_id,
        )
        all_spmd_ids = self.spmd_manager.get_all_spmd_ids()
        if not all_spmd_ids:
            raise RuntimeError("global snapshot requires at least one SPMD")

        drivers: dict[str, Any] = {}
        for spmd_id in all_spmd_ids:
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is None:
                raise RuntimeError(
                    f"request_snapshot: driver for spmd_id '{spmd_id}' not found. "
                    "Cannot perform partial snapshot."
                )
            drivers[spmd_id] = driver

        if deadline is None:
            deadline = CheckpointDeadline.start(self.snapshot_timeout_s)

        done_event = asyncio.Event()
        self._pending_snapshots[checkpoint_id] = {
            "expected": len(all_spmd_ids),
            "expected_spmd_ids": frozenset(all_spmd_ids),
            "received": 0,
            "reported": set(),
            "metadata": {},
            "errors": [],
            "repository_root": repository_root,
            "event": done_event,
            "permit": permit,
            "deadline": deadline,
        }
        for driver in drivers.values():
            driver.prepare_snapshot.remote(
                checkpoint_id, attempt_id, permit, cut, repository_root
            )

        try:
            try:
                await deadline.run_phase(done_event.wait())
            except TimeoutError as error:
                await self._fail_snapshot_drivers(drivers, checkpoint_id)
                raise RuntimeError(
                    f"Snapshot '{checkpoint_id}' timed out after {deadline.total_timeout_s:.1f}s"
                ) from error
            except BaseException:
                await self._fail_snapshot_drivers(drivers, checkpoint_id)
                raise

            snap_info = self._pending_snapshots[checkpoint_id]
            if snap_info["errors"]:
                await self._fail_snapshot_drivers(drivers, checkpoint_id)
                raise RuntimeError(
                    f"Snapshot '{checkpoint_id}' failed for SPMDs: {snap_info['errors']}"
                )
            prepared_spmds = dict(snap_info["metadata"])
            stream_cuts = self.control_stream_cuts(prepared_spmds)
            resolved_cut = CheckpointCut(
                cut_id=cut.cut_id,
                position=cut.position,
                stream_cuts=stream_cuts,
                sealed_ends={},
                control_epoch=cut.control_epoch,
            )
            try:
                finalized = await deadline.run_phase(
                    asyncio.gather(
                        *(
                            driver.capture_driver_components.remote(
                                checkpoint_id, resolved_cut
                            )
                            for driver in drivers.values()
                        )
                    )
                )
            except BaseException:
                await self._fail_snapshot_drivers(drivers, checkpoint_id)
                raise
            spmds = dict(zip(drivers, finalized, strict=True))
            if any(
                not isinstance(snapshot, SpmdCheckpoint) for snapshot in spmds.values()
            ):
                raise TypeError("Driver component capture must return SpmdCheckpoint")
            if resolved_cut.position is not None:
                self._snapshot_publication_context[checkpoint_id] = (
                    repository_root,
                    attempt_id,
                    resolved_cut.position,
                )
            return CheckpointAssembly(
                cut=resolved_cut,
                framework_state=self.checkpoint_state(
                    cuts=stream_cuts, cut=resolved_cut
                ),
                spmds=spmds,
            )
        finally:
            self._pending_snapshots.pop(checkpoint_id, None)

    @staticmethod
    async def _fail_snapshot_drivers(
        drivers: Mapping[str, Any], checkpoint_id: str
    ) -> None:
        await asyncio.gather(
            *(
                driver.fail_snapshot.remote(checkpoint_id)
                for driver in drivers.values()
            ),
            return_exceptions=True,
        )

    @require_spmd_manager
    async def receive_snapshot_metadata(
        self,
        spmd_id: str,
        checkpoint_id: str,
        snapshot: SpmdCheckpoint,
        permit: CheckpointCommitPermit,
    ) -> None:
        snap_info = self._pending_snapshots.get(checkpoint_id)
        if snap_info is None:
            logger.warning("Received metadata for unknown snapshot %r", checkpoint_id)
            return
        if permit != snap_info["permit"]:
            raise RuntimeError(
                f"fencing violation: snapshot permit mismatch for checkpoint {checkpoint_id}. "
                f"Expected {snap_info['permit']}, got {permit}."
            )
        if spmd_id not in snap_info["expected_spmd_ids"]:
            raise RuntimeError(
                f"unexpected participant {spmd_id} for checkpoint {checkpoint_id}; "
                f"expected one of {snap_info['expected_spmd_ids']}"
            )
        if spmd_id in snap_info["reported"]:
            logger.warning(
                "Duplicate snapshot metadata spmd_id=%s checkpoint_id=%s",
                spmd_id,
                checkpoint_id,
            )
            return
        if not isinstance(snapshot, SpmdCheckpoint):
            raise TypeError("Driver snapshot metadata must be SpmdCheckpoint")
        snap_info["reported"].add(spmd_id)
        snap_info["metadata"][spmd_id] = snapshot
        snap_info["received"] += 1
        if snap_info["received"] == snap_info["expected"]:
            snap_info["event"].set()

    async def report_snapshot_failed(
        self,
        spmd_id: str,
        checkpoint_id: str,
        permit: CheckpointCommitPermit,
    ) -> None:
        """Unblock request_snapshot when a Driver triggers SNAPSHOT_FAILED.

        Called by a Driver that encountered an unrecoverable error during
        persistence and could not (or failed to) deliver metadata via
        receive_snapshot_metadata.  Advances the received counter so the
        Controller's done_event is eventually set rather than waiting forever.
        """
        snap_info = self._pending_snapshots.get(checkpoint_id)
        if snap_info is None:
            logger.warning(
                f"report_snapshot_failed: unknown snapshot '{checkpoint_id}' from {spmd_id}"
            )
            return
        if permit != snap_info["permit"]:
            raise RuntimeError(
                f"fencing violation: snapshot permit mismatch for checkpoint {checkpoint_id}. "
                f"Expected {snap_info['permit']}, got {permit}."
            )
        if spmd_id not in snap_info["expected_spmd_ids"]:
            raise RuntimeError(
                f"unexpected participant {spmd_id} for checkpoint {checkpoint_id}; "
                f"expected one of {snap_info['expected_spmd_ids']}"
            )

        # Same idempotency guard as receive_snapshot_metadata.
        if spmd_id in snap_info["reported"]:
            logger.warning(
                f"report_snapshot_failed: duplicate report from '{spmd_id}' "
                f"for snapshot '{checkpoint_id}' — ignoring"
            )
            return
        snap_info["reported"].add(spmd_id)

        logger.error(
            f"Snapshot '{checkpoint_id}' failed for SPMD '{spmd_id}' — marking as error"
        )
        snap_info["errors"].append(spmd_id)
        snap_info["received"] += 1
        if snap_info["received"] == snap_info["expected"]:
            snap_info["event"].set()

    @require_spmd_manager
    async def restore_snapshot(
        self,
        checkpoint_id: str,
        manifest: CheckpointManifest,
        repository_root: str,
        restore_options: RestoreOptions | None = None,
    ) -> None:
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.controller_restore(
                checkpoint_id=checkpoint_id,
            ):
                await self._restore_snapshot_unprofiled(
                    checkpoint_id,
                    manifest,
                    repository_root,
                    restore_options,
                )
            return
        await self._restore_snapshot_unprofiled(
            checkpoint_id,
            manifest,
            repository_root,
            restore_options,
        )

    async def _restore_snapshot_unprofiled(
        self,
        checkpoint_id: str,
        manifest: CheckpointManifest,
        repository_root: str,
        restore_options: RestoreOptions | None = None,
    ) -> None:
        if self.state_machine.state not in (
            ControllerState.READY,
            ControllerState.ACTIVATING,
        ):
            raise RuntimeError("checkpoint restore requires pre-execution readiness")
        logger.info(
            "Controller restoring snapshot %s (cut %s)",
            checkpoint_id,
            manifest.cut.cut_id,
        )
        if manifest.checkpoint_id != checkpoint_id:
            raise ValueError("manifest checkpoint ID does not match restore request")
        if restore_options is not None:
            unknown_spmds = set(restore_options).difference(manifest.spmds)
            if unknown_spmds:
                raise ValueError(
                    f"restore_options contain unknown SPMDs: {sorted(unknown_spmds)!r}"
                )
        await self.restore_checkpoint_state(
            dict(manifest.framework_state), manifest.cut
        )
        position = manifest.cut.position
        if position is None:
            raise RuntimeError(
                "current Controller restore requires a durable checkpoint position"
            )
        if self.heartbeat_manager is not None:
            restore_timeout = (
                self._checkpoint_config.snapshot_timeout_s
                if self._checkpoint_config is not None
                else int(os.environ.get("FLUXWAY_SNAPSHOT_TIMEOUT_SECS", "600"))
            )
            ready = await self.heartbeat_manager.wait_until_all_drivers_ready(
                timeout=restore_timeout
            )
            if not ready:
                raise RuntimeError(
                    "timed out waiting for drivers to become restore-ready"
                )
        # Reset runtime boundary to OPEN at the durable position for current runtime execution
        self._drain_boundary = DrainBoundary.from_committed_cut(position)
        runtime_snapshot = {
            spmd_id: {0: dict(snapshot.runtime_state)}
            for spmd_id, snapshot in manifest.spmds.items()
        }
        self.state_manager.save_snapshot(checkpoint_id, runtime_snapshot)
        flow_control_updates: dict[str, tuple[Any, ...]] = {}
        for spmd_id, snapshot in manifest.spmds.items():
            count = checkpoint_position(snapshot.runtime_state)
            graph_view = self._graph.require_compiled_graph().spmd(spmd_id)
            for op_name in graph_view.operator_ids:
                self.async_step_manager.report_op_completed(spmd_id, op_name, count)
                flow_control_updates.update(
                    self._flow_control_manager.observe_op_completed(
                        spmd_id, op_name, count
                    )
                )
        await self._apply_flow_control_windows(flow_control_updates)
        tasks = []
        for spmd_id, snapshot in manifest.spmds.items():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is None:
                raise RuntimeError(
                    f"restore_snapshot: driver for spmd_id {spmd_id!r} not found"
                )
            tasks.append(
                driver.restore_from_manifest.remote(
                    checkpoint_id,
                    manifest.cut,
                    snapshot,
                    repository_root,
                    None if restore_options is None else restore_options.get(spmd_id),
                )
            )
        if tasks:
            await asyncio.gather(*tasks)

        logger.info("Snapshot restored successfully")

    async def restore_checkpoint(
        self,
        step: int | None = None,
        *,
        restore_options: RestoreOptions | None = None,
    ) -> int | None:
        """
        User-facing API to trigger checkpoint restoration.
        """
        if self._checkpointer is None:
            raise RuntimeError(
                "Checkpoint config was not provided during controller initialization"
            )

        reference = "latest" if step is None else StepReference(step)
        return await self._execution_lifecycle.restore(
            reference,
            restore_options=restore_options,
        )

    async def acquire_checkpoint_lease(
        self,
        spmd_id: str,
        checkpoint_object: CheckpointObjectRef,
        lease_holder_id: str,
        ttl_seconds: float,
        request_id: str,
    ) -> CheckpointHold:
        if self._checkpointer is None:
            raise RuntimeError(
                "Checkpoint config was not provided during controller initialization"
            )
        return await self._checkpointer.hold_checkpoint_object(
            spmd_id, checkpoint_object, lease_holder_id, ttl_seconds, request_id
        )

    async def renew_checkpoint_lease(
        self,
        spmd_id: str,
        lease: CheckpointHold,
        ttl_seconds: float,
    ) -> CheckpointHold:
        if self._checkpointer is None:
            raise RuntimeError(
                "Checkpoint config was not provided during controller initialization"
            )
        return await self._checkpointer.renew_checkpoint_hold(
            spmd_id, lease, ttl_seconds
        )

    async def release_checkpoint_lease(
        self, spmd_id: str, lease: CheckpointHold
    ) -> None:
        if self._checkpointer is None:
            raise RuntimeError(
                "Checkpoint config was not provided during controller initialization"
            )
        await self._checkpointer.release_checkpoint_hold(spmd_id, lease)

    async def stop_checkpointer(self) -> None:
        task = self._checkpoint_task
        if task is not None and not task.done():
            current_task = asyncio.current_task()
            if task is not current_task:
                await task
        self._checkpoint_task = None

    @require_spmd_manager
    async def inspect_operator_state(self) -> dict[str, Any]:
        tasks: dict[str, Any] = {}
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                tasks[spmd_id] = driver.inspect_operator_state.remote()
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results, strict=True))

    @require_spmd_manager
    async def capture_profiler_snapshots(
        self,
    ) -> dict[str, dict[int, ProfilerSnapshot | None]]:
        tasks: dict[str, Any] = {}
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                tasks[spmd_id] = driver.export_profiler_snapshots.remote()
        if not tasks:
            raise RuntimeError("No SPMD drivers are available for profiler export.")

        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results, strict=True))

    async def capture_profiler_snapshots_if_enabled(
        self,
    ) -> dict[str, dict[int, ProfilerSnapshot | None]] | None:
        if self._profiler_config is None:
            return None
        return await self.capture_profiler_snapshots()

    @require_spmd_manager
    async def export_profiler_trace(self, path: str) -> str:
        snapshots_by_spmd = await self.capture_profiler_snapshots()
        return await self.write_profiler_trace(path, snapshots_by_spmd)

    async def write_profiler_trace(
        self,
        path: str,
        snapshots_by_spmd: dict[str, dict[int, ProfilerSnapshot | None]],
    ) -> str:
        records = []
        names_by_id: dict[int, str] = {}
        payload_fields_by_name_id: dict[int, tuple[str, ...]] = {}

        for rank_snapshots in snapshots_by_spmd.values():
            for snapshot in rank_snapshots.values():
                if snapshot is None:
                    continue
                records_array = snapshot["records"].copy()
                records_array[:, 1] += snapshot["monotonic_to_epoch_offset_ns"]
                if len(records_array) > 0:
                    records.append(records_array)
                metadata = snapshot["metadata"]
                names_by_id.update(metadata.names_by_id)
                payload_fields_by_name_id.update(metadata.payload_fields_by_name_id)

        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.controller_profiler_export(
                spmd_count=len(snapshots_by_spmd),
                record_count=sum(len(record_array) for record_array in records),
            ):
                pass
            controller_snapshot = self._profiler_runtime.snapshot()
            if controller_snapshot is not None:
                from fluxway.profiler.record import TraceMetadata

                records_obj = controller_snapshot["records"]
                assert isinstance(records_obj, np.ndarray)
                records_array = records_obj.copy()

                offset_obj = controller_snapshot["monotonic_to_epoch_offset_ns"]
                assert isinstance(offset_obj, int)
                records_array[:, 1] += offset_obj

                if len(records_array) > 0:
                    records.append(records_array)

                metadata = controller_snapshot["metadata"]
                assert isinstance(metadata, TraceMetadata)
                names_by_id.update(metadata.names_by_id)
                payload_fields_by_name_id.update(metadata.payload_fields_by_name_id)

        if not records:
            raise RuntimeError(
                "Profiler is enabled but no trace records were captured."
            )

        from fluxway.profiler.perfetto_writer import PerfettoTraceWriter
        from fluxway.profiler.record import TraceMetadata
        from fluxway.profiler.filtering import filter_steady_state_records
        from fluxway.profiler.graph_symbols import GraphTraceSymbols

        merged = np.vstack(records)
        metadata = TraceMetadata(
            names_by_id=names_by_id,
            payload_fields_by_name_id=payload_fields_by_name_id,
        )
        if bool(getattr(self._profiler_config, "steady_state_only", False)):
            merged = filter_steady_state_records(merged, metadata)
            if len(merged) == 0:
                raise RuntimeError(
                    "Profiler steady-state filtering removed all trace records."
                )
        output = Path(path)
        PerfettoTraceWriter().write(
            merged,
            output,
            metadata=metadata,
            graph_symbols=GraphTraceSymbols.from_compiled_graph(
                self._graph.require_compiled_graph()
            ),
        )
        return str(output)

    async def write_profiler_trace_if_enabled(
        self, snapshots_by_spmd: dict[str, dict[int, ProfilerSnapshot | None]] | None
    ) -> str | None:
        if self._profiler_config is None:
            return None
        if snapshots_by_spmd is None:
            snapshots_by_spmd = await self.capture_profiler_snapshots()
            assert snapshots_by_spmd is not None
        output_path = getattr(self._profiler_config, "output_path", None)
        if output_path is None:
            output_path = (
                Path(self._profiler_config.trace_dir)
                / str(self._profiler_config.run_id)
                / "trace.pftrace"
            )
        return await self.write_profiler_trace(str(output_path), snapshots_by_spmd)

    async def export_profiler_trace_if_enabled(self) -> str | None:
        if self._profiler_config is None:
            return None
        output_path = getattr(self._profiler_config, "output_path", None)
        if output_path is None:
            output_path = (
                Path(self._profiler_config.trace_dir)
                / str(self._profiler_config.run_id)
                / "trace.pftrace"
            )
        return await self.export_profiler_trace(str(output_path))

    # =====================================================================
    # SPMD Error Handling
    # =====================================================================

    async def report_spmd_error(self, spmd_id: str, failure: ExecutionFailure) -> None:
        """
        Receive error report from a driver.

        Args:
            spmd_id: SPMD identifier
            failure: Structured execution failure
        """
        logger.error(f"SPMD {spmd_id} reported error: {failure}")
        await self.state_machine.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id=spmd_id, failure=failure
        )

    @require_spmd_manager
    @require_heartbeat_manager
    async def report_spmd_completed(self, spmd_id: str) -> None:
        """
        Receive completion report from a driver.

        Args:
            spmd_id: SPMD identifier
        """
        logger.debug(f"SPMD {spmd_id} reported completion")

        # Check if all SPMDs are completed
        all_completed = True
        for sid in self.spmd_manager.get_all_spmd_ids():
            heartbeat = await self.heartbeat_manager.get_driver_heartbeat(sid)
            if heartbeat is None:
                all_completed = False
                break

        if all_completed:
            await self.state_machine.trigger_event(
                ControllerEvent.ALL_DRIVERS_COMPLETED
            )

    # =====================================================================
    # Blocking Operations
    # =====================================================================

    @require_spmd_manager
    async def block_all_spmds(self) -> None:
        """Block all SPMD operations (called during error handling)."""
        logger.warning("Blocking all SPMDs")

        tasks = []
        block_errors: list[Exception] = []
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                try:
                    tasks.append(driver.block_all_ops.remote())
                except Exception as error:
                    block_errors.append(error)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            block_errors.extend(
                result for result in results if isinstance(result, Exception)
            )
        if len(block_errors) == 1:
            raise block_errors[0]
        if block_errors:
            raise ExceptionGroup("block SPMDs failed", block_errors)

    @require_spmd_manager
    async def unblock_all_spmds(self) -> None:
        """Unblock all SPMD operations (called during recovery)."""
        logger.debug("Unblocking all SPMDs")

        tasks = []
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                tasks.append(driver.unblock_all_ops.remote())

        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error unblocking SPMDs: {e}")

    # =====================================================================
    # Heartbeat Callbacks (async)
    # =====================================================================

    async def _on_all_drivers_ready(self) -> None:
        """Callback when all drivers report workers_ready=True.

        Triggers: PENDING → TOPOLOGY_REGISTERING → _initiate_topology_registration()
        """
        logger.debug(
            "All drivers are ready (workers_ready=True), triggering topology registration"
        )
        await self.state_machine.trigger_event(ControllerEvent.ALL_DRIVERS_READY)

    async def _on_all_topology_ready(self) -> None:
        await self.state_machine.trigger_event(ControllerEvent.ALL_SPMDS_TOPOLOGY_READY)

    async def _on_driver_error(self, spmd_id: str, failure: ExecutionFailure) -> None:
        """Callback when a driver has an error."""
        await self.report_spmd_error(spmd_id, failure)

    # =====================================================================
    # Internal Methods for State Machine
    # =====================================================================

    @require_spmd_manager
    async def _initiate_topology_registration(self) -> None:
        """
        Initiate topology registration on all SPMDs.

        This method:
        1. Collects ALL worker handles from ALL drivers upfront to avoid actor deadlock.
        2. For each SPMD, wraps handles with DataPlaneCallable and calls register_topology.
        """
        all_spmd_ids = self.spmd_manager.get_all_spmd_ids()
        if not all_spmd_ids:
            logger.warning("No SPMDs found, cannot initiate topology registration")
            return

        # Collect all worker handles upfront to avoid actor re-entrancy deadlocks during registration
        # {spmd_id: {rank: worker_handle}}
        all_worker_handles: dict[str, dict[int, Any]] = {}

        fetch_tasks = []
        spmd_ids_to_fetch = []

        for spmd_id in all_spmd_ids:
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                fetch_tasks.append(driver.get_worker_handles.remote())
                spmd_ids_to_fetch.append(spmd_id)

        if fetch_tasks:
            try:
                results = await asyncio.gather(*fetch_tasks)
                for spmd_id, worker_map in zip(spmd_ids_to_fetch, results, strict=True):
                    all_worker_handles[spmd_id] = worker_map
                logger.debug(
                    f"Collected worker handles for: {list(all_worker_handles.keys())}"
                )
                for sid, handles in all_worker_handles.items():
                    logger.debug(f"  - {sid}: {len(handles)} workers")
            except Exception as e:
                raise RuntimeError(
                    "Failed to collect worker handles for topology; "
                    f"spmd_ids={spmd_ids_to_fetch!r}"
                ) from e

        def _get_transport_type(checkpoint_object_spmd_id: str) -> str:
            if checkpoint_object_spmd_id in self._transport_config:
                return self._transport_config[checkpoint_object_spmd_id]
            if "*" in self._transport_config:
                return self._transport_config["*"]
            node = self._graph.nodes.get(checkpoint_object_spmd_id)
            if node and node.spmd_config:
                return getattr(node.spmd_config, "transport_type", "ray")
            return "ray"

        # Wrap raw handles with DataPlaneCallable after predecessor/successor
        # direction is known, so each sender-side adapter can bind sender_id
        # once instead of carrying it in every hot-path data frame.

        # Initiate topology registration on each driver
        spmd_code_by_id = {
            spmd_id: code for code, spmd_id in enumerate(sorted(all_spmd_ids))
        }
        spmd_id_by_code = {code: spmd_id for spmd_id, code in spmd_code_by_id.items()}
        driver_handles_by_id: dict[str, Any] = {}
        for driver_spmd_id in all_spmd_ids:
            driver_handle = self.spmd_manager.get_driver_handle(driver_spmd_id)
            if driver_handle is None:
                raise RuntimeError(
                    f"Missing driver handle for SPMD {driver_spmd_id!r} "
                    "during credit topology registration."
                )
            driver_handles_by_id[driver_spmd_id] = driver_handle

        for spmd_id in all_spmd_ids:
            driver = driver_handles_by_id[spmd_id]
            if driver is None:
                continue

            predecessors = self.spmd_manager.get_predecessors(spmd_id)
            successors = self.spmd_manager.get_successors(spmd_id)

            logger.debug(
                f"Driver {spmd_id}: predecessors={predecessors}, successors={successors}"
            )

            # Build predecessor/successor worker handle maps: {spmd_id: {rank: handle}}
            succ_handlers = {
                s: {
                    rank: DataPlaneRegistry.create(
                        _get_transport_type(s),
                        handle,
                        sender_spmd_code=spmd_code_by_id[spmd_id],
                    )
                    for rank, handle in all_worker_handles[s].items()
                }
                for s in successors
                if s in all_worker_handles
            }
            pred_handlers = {
                p: {
                    rank: DataPlaneRegistry.create(
                        _get_transport_type(p),
                        handle,
                        sender_spmd_code=spmd_code_by_id[p],
                    )
                    for rank, handle in all_worker_handles[p].items()
                }
                for p in predecessors
                if p in all_worker_handles
            }

            logger.debug(f"  - pred_handlers keys: {list(pred_handlers.keys())}")
            logger.debug(f"  - succ_handlers keys: {list(succ_handlers.keys())}")

            try:
                upstream_edges, downstream_edges = (
                    self._flow_control_manager.topology_for_driver(spmd_id)
                )
                upstream_driver_handles = {
                    edge.source_spmd_id: driver_handles_by_id[edge.source_spmd_id]
                    for edge in downstream_edges
                }
                await driver.register_credit_topology.remote(
                    upstream_driver_handles,
                    upstream_edges,
                    downstream_edges,
                    self._flow_control_manager.epoch,
                )
                initial_windows = self._flow_control_manager.initial_windows_for_driver(
                    spmd_id
                )
                if initial_windows:
                    await driver.apply_credit_window.remote(initial_windows)
                await driver.register_topology.remote(
                    pred_handlers,
                    succ_handlers,
                    len(predecessors),
                    len(successors),
                    spmd_code_by_id[spmd_id],
                    spmd_id_by_code,
                )
                logger.debug(f"Completed topology registration for SPMD {spmd_id}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to register topology for SPMD {spmd_id!r}"
                ) from e

    @require_spmd_manager
    @require_heartbeat_manager
    async def _reclaim_all_spmds(self) -> None:
        """
        Cleanup all SPMD resources.

        Called when entering COMPLETING or FAILING state.
        """
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.controller_cleanup(
                spmd_count=len(self.spmd_manager.get_all_spmd_ids()),
            ):
                await self._reclaim_all_spmds_unprofiled()
            return
        await self._reclaim_all_spmds_unprofiled()

    async def _reclaim_all_spmds_unprofiled(self) -> None:
        logger.debug("Cleaning up all SPMDs")

        cleanup_errors: list[Exception] = []
        try:
            await self.heartbeat_manager.stop_monitoring_async()
        except Exception as error:
            cleanup_errors.append(error)

        spmd_cleanup_errors = await asyncio.gather(
            *(
                self._stop_and_unregister_spmd(spmd_id)
                for spmd_id in self.spmd_manager.get_all_spmd_ids()
            )
        )
        for errors in spmd_cleanup_errors:
            cleanup_errors.extend(errors)

        if self.scheduler.pg_manager.has_job_pg:
            try:
                self.scheduler.release_resources()
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise ExceptionGroup("SPMD reclaim failed", cleanup_errors)

    async def _stop_and_unregister_spmd(self, spmd_id: str) -> tuple[Exception, ...]:
        errors: list[Exception] = []
        try:
            stop_spmd = functools.partial(
                self.spmd_manager.stop_spmd,
                spmd_id,
                save_checkpoint=False,
                release_driver_handle=False,
            )
            if self._profiler_runtime.enabled:
                async with self._profiler_runtime.recorder.spmd_manager_driver_stop(
                    spmd_id=spmd_id,
                ):
                    await stop_spmd()
            else:
                await stop_spmd()
        except Exception as error:
            logger.error(f"Error stopping SPMD {spmd_id}: {error}")
            errors.append(error)
        finally:
            try:
                self.heartbeat_manager.unregister_spmd(spmd_id)
            except Exception as error:
                errors.append(error)
        return tuple(errors)

    async def _finalize_runtime(self, *, export_profiler_trace: bool) -> None:
        cleanup_errors: list[Exception] = []
        profiler_snapshots = None
        profiler_capture_succeeded = False
        try:
            await self.stop_checkpointer()
        except Exception as error:
            cleanup_errors.append(error)
        if self._checkpointer is not None:
            checkpointer = self._checkpointer
            self._checkpointer = None
            try:
                await checkpointer.close()
            except Exception as error:
                cleanup_errors.append(error)
        if export_profiler_trace:
            try:
                profiler_snapshots = await self.capture_profiler_snapshots_if_enabled()
                profiler_capture_succeeded = True
            except Exception as error:
                cleanup_errors.append(error)
        try:
            await self._reclaim_all_spmds()
        except Exception as error:
            cleanup_errors.append(error)
        if self._ingestion_planner is not None:
            try:
                await self._ingestion_planner.close()
            except Exception as error:
                cleanup_errors.append(error)
        if export_profiler_trace and profiler_capture_succeeded:
            try:
                await self.write_profiler_trace_if_enabled(profiler_snapshots)
            except Exception as error:
                cleanup_errors.append(error)
        try:
            self._release_stopped_spmd_handles()
        except Exception as error:
            cleanup_errors.append(error)
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise ExceptionGroup("terminal reclaim failed", cleanup_errors)

    def _release_stopped_spmd_handles(self) -> None:
        """Release driver handles after terminal cleanup has fully completed."""
        self.spmd_manager.release_all_driver_handles()

    # =====================================================================
    # Lifecycle Control
    # =====================================================================

    async def stop(self, save_checkpoint: bool = False) -> None:
        if save_checkpoint:
            raise ValueError(
                "stop cannot create a checkpoint; configure checkpoint cuts before execution"
            )
        await self._execution_lifecycle.cancel()

    @require_heartbeat_manager
    async def get_cluster_status(self) -> dict[str, str]:
        """
        Get status of all SPMD nodes.

        Returns:
            Mapping of node name to status
        """
        status = await self.heartbeat_manager.get_all_driver_status()
        return {spmd_id: s.value for spmd_id, s in status.items()}

    async def wait_for_completion(self, timeout: float | None = None) -> bool:
        """
        Wait for training to complete.

        Args:
            timeout: Maximum time to wait (None for infinite)

        Returns:
            True if completed successfully, False if timeout elapsed before terminal state.

        Raises:
            ExecutionFailedError: The execution reached FAILED.
            asyncio.CancelledError: The execution was cancelled.
        """
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.controller_completion(
                spmd_count=len(self._graph.nodes),
            ):
                return await self._wait_for_completion_unprofiled(timeout)
        return await self._wait_for_completion_unprofiled(timeout)

    async def _wait_for_completion_unprofiled(
        self, timeout: float | None = None
    ) -> bool:
        try:
            state = await self.state_machine.wait_until_terminal(timeout)
        except TimeoutError:
            return False
        if state == ControllerState.COMPLETED:
            return True
        if state == ControllerState.CANCELLED:
            raise asyncio.CancelledError("execution was cancelled")
        if state == ControllerState.FAILED:
            raise ExecutionFailedError(self.state_machine.get_failures())
        raise RuntimeError(f"Unexpected terminal state: {state}")


async def create_controller(
    graph: Graph,
    *,
    placement_recipe: PlacementRecipe,
    placement_runtime: PlacementRuntimeConfig | None = None,
    heartbeat_config: HeartbeatConfig | None = None,
    async_step_config: AsyncStepConfig | None = None,
    checkpoint_config: CheckpointConfig | None = None,
    runtime_env: dict[str, Any] | None = None,
    transport_config: dict[str, str] | None = None,
    profiler_config: Any | None = None,
) -> Any:
    """
    Create and initialize a Controller actor.

    This is the main entry point for creating a Controller.

    Args:
        graph: The static execution Graph.
        placement_recipe: Resource placement constraints and objectives.
        heartbeat_config: Optional heartbeat configuration
        async_step_config: Optional async step configuration
        checkpoint_config: Optional checkpoint configuration
        runtime_env: Optional Ray runtime environment
        transport_config: Optional data plane transport backend per SPMD (e.g. {"*": "ray"})
        profiler_config: Optional profiler runtime configuration

    Returns:
        Controller actor handle

    Raises:
        RuntimeError: If initialization fails
    """
    if not isinstance(graph, Graph):
        raise TypeError("create_controller requires a Graph")
    if not isinstance(placement_recipe, PlacementRecipe):
        raise TypeError("create_controller requires a PlacementRecipe")
    if placement_runtime is None:
        placement_runtime = PlacementRuntimeConfig()

    # Create the actor
    controller = ControllerActor.options(runtime_env=runtime_env).remote(  # type: ignore[attr-defined]
        graph=graph,
        placement_recipe=placement_recipe,
        heartbeat_config=heartbeat_config,
        async_step_config=async_step_config,
        checkpoint_config=checkpoint_config,
        transport_config=transport_config,
        profiler_config=profiler_config,
        placement_runtime=placement_runtime,
    )

    # Initialize the actor
    success = await controller.initialize.remote()  # type: ignore[attr-defined]
    if not success:
        raise RuntimeError("Failed to initialize controller")

    return controller


Controller = ControllerActor
