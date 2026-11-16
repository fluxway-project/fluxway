"""
SPMD Driver: Master controller for an SPMD instance.

Based on ARCHITECTURE.md Section 3.1

The SPMD Driver coordinates all workers within the SPMD:
- Starts and manages worker actors using placement group
- Handles worker heartbeats and monitors health
- Synchronizes worker steps within the SPMD
- Reports SPMD-level progress to Controller
- Handles control_argument broadcasting
- Manages op-level blocking
"""

import asyncio
from collections.abc import Mapping
import time
from typing import Any

import ray
from fluxway.common.types.drain import DrainAdmission
from fluxway.common.types import (
    CheckpointCut,
    DriverEvent,
    DriverState,
    ExecutionFailure,
    FlowControlState,
    HeartbeatConfig,
    SPMDConfig,
    SpmdCheckpoint,
    WorkerHeartbeat,
    DrainEvent,
    RestoreOptionsByRank,
)
from fluxway.common.utils import ensure_logging_configured
from fluxway.common.driver_control import DriverComponentCapabilities
from fluxway.common.effect_scheduler import (
    BoundedLifecycleEffectScheduler,
    LifecycleEffectExecutionMode,
    LifecycleEffectSubmission,
)
from fluxway.rpc.protocol import DataPlaneCallable
from fluxway.spmd.driver.credit import CreditUpdate, CreditWindow
from fluxway.spmd.driver.state_machine import DriverStateMachine
from fluxway.spmd.driver.synchronizer import (
    MaxRankStepSkewSyncPolicy,
    SPMDSynchronizer,
)
from fluxway.spmd.driver.monitor import SPMDMonitor
from fluxway.spmd.driver.executor.ray_executor import RayWorkerExecutor
from fluxway.spmd.driver.snapshot.orchestrator import SnapshotOrchestrator
from fluxway.spmd.driver.heartbeat import HeartbeatManager
from fluxway.spmd.driver.credit import CreditManager
from fluxway.control_plane.components import (
    DriverComponentRegistry,
    build_driver_component_registry,
)
from fluxway.profiler import ProfilerRuntimeFactory
from fluxway.profiler.recorder import DRIVER_REGISTER_TOPOLOGY
from fluxway.common.utils.logging import get_logger

logger = get_logger(__name__)


@ray.remote  # type: ignore[reportPossiblyUnboundVariable]
class SPMDDriver:  # pyright: ignore[reportRedeclaration]
    """
    SPMD Driver (Ray Actor).

    Coordinates workers within a single SPMD instance.
    Started by SPMDManager, manages workers using placement group.
    """

    def __init__(
        self,
        spmd_id: str,
        config: SPMDConfig,
        placement_group: Any | None,
        worker_bundle_indices: list[int],
        controller_handle: Any | None,
        graph_topology: dict[str, Any],
        worker_resource: Mapping[str, Any],
        expected_node_ids: list[str],
    ):
        """
        Initialize SPMD Driver.
        """
        ensure_logging_configured()

        self.spmd_id = spmd_id
        self.config = config
        self.controller_handle = controller_handle
        self.graph_topology = graph_topology
        self.expected_node_ids = expected_node_ids

        # State machine
        self.state_machine = DriverStateMachine(self)
        self._effect_scheduler = BoundedLifecycleEffectScheduler(
            queue_size=64, worker_count=4
        )
        self._driver_effect_outcomes: set[asyncio.Future] = set()

        self.spmd_synchronizer = SPMDSynchronizer(
            spmd_id=spmd_id,
            num_workers=config.num_workers,
            sync_policy=MaxRankStepSkewSyncPolicy(config.max_rank_step_skew),
        )
        self.spmd_synchronizer.register_expected_ops(
            config.require_graph_view().operator_ids
        )
        self.monitor = SPMDMonitor(spmd_id)

        # Worker readiness aggregation
        self._workers_ready: bool = False
        self._topology_ready: bool = False
        self._shutdown_task: asyncio.Task[None] | None = None

        # Heartbeat tracking for HeartbeatManager to read
        self._worker_heartbeats: dict[int, WorkerHeartbeat] = {}
        self._worker_heartbeat_times: dict[int, float] = {}

        # Backpressure / Pause control
        self._is_paused = False

        profiler_config = getattr(config, "profiler_config", None)
        self._profiler_runtime = ProfilerRuntimeFactory.from_config(
            profiler_config,
            process_name=f"driver/{self.spmd_id}",
        )
        self._profile_heartbeat = self._profiler_runtime.heartbeat_diagnostics_enabled()

        # Initialize Components
        self.executor = RayWorkerExecutor(
            config=config,
            placement_group=placement_group,
            worker_bundle_indices=worker_bundle_indices,
            worker_resource=worker_resource,
            expected_node_ids=expected_node_ids,
        )
        self._driver_components: DriverComponentRegistry = (
            build_driver_component_registry(
                config,
                spmd_id=spmd_id,
                controller_handle=controller_handle,
                report_failure=self._report_component_failure,
                capabilities=DriverComponentCapabilities(worker_control=self.executor),
            )
        )
        self._driver_component_notification_tasks: set[asyncio.Task[None]] = set()

        self.snapshot_orchestrator = SnapshotOrchestrator(
            spmd_id=self.spmd_id,
            executor=self.executor,
            synchronizer=self.spmd_synchronizer,
            controller_handle=self.controller_handle,
            state_machine=self.state_machine,
            profiler_runtime=self._profiler_runtime,
            component_registry=self._driver_components,
            notify_component_state=self._notify_driver_components,
        )

        self.heartbeat_manager = HeartbeatManager(
            spmd_id=self.spmd_id,
            num_workers=self.config.num_workers,
            heartbeat_config=HeartbeatConfig(
                heartbeat_interval=self.config.heartbeat_interval_s,
                heartbeat_timeout=self.config.heartbeat_timeout_s,
            ),
            profile_heartbeat=self._profile_heartbeat,
            get_driver_state=lambda: self.state_machine.state,
            get_driver_failure=lambda: self.state_machine.failure,
            get_workers_ready=lambda: self._workers_ready,
            get_topology_ready=lambda: self._topology_ready,
            get_worker_heartbeat_times=lambda: self._worker_heartbeat_times,
            controller_handle=self.controller_handle,
            trigger_state_event=lambda e, failure: self.state_machine.trigger(
                e, failure=failure
            ),
            receive_worker_failure=self.receive_worker_failure,
            profiler_runtime=self._profiler_runtime,
            get_worker_heartbeats=lambda: self._worker_heartbeats,
            inactivity_timeout_s=self.config.inactivity_timeout_s,
        )

        self.credit_manager = CreditManager(
            spmd_id=self.spmd_id,
            synchronizer=self.spmd_synchronizer,
        )

        logger.debug(
            f"Driver initialized spmd_id={spmd_id} num_workers={config.num_workers}"
        )

    # =====================================================================
    # Initialization
    # =====================================================================

    async def initialize(self, checkpoint_id: str | None = None) -> None:
        await self._effect_scheduler.start()
        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if profiler_recorder is not None:
            span = profiler_recorder.driver_initialize(
                worker_count=self.config.num_workers
            )
            await span.__aenter__()
        else:
            span = None
        try:
            logger.debug(f"Initializing SPMD spmd_id={self.spmd_id}")

            self.spmd_synchronizer.set_callbacks(
                on_spmd_op_completed=self._on_spmd_op_completed,
                block_worker_op=self.executor.block_worker_op,
                unblock_worker_op=self.executor.unblock_worker_op,
            )

            try:
                self_handle = ray.get_runtime_context().current_actor
            except Exception:
                self_handle = None

            await self.executor.start_workers(
                self.spmd_id, self_handle, self.controller_handle
            )

            if checkpoint_id is not None:
                raise NotImplementedError(
                    "Checkpoint restore must go through Controller.restore_snapshot(). "
                    f"checkpoint_id={checkpoint_id} spmd_id={self.spmd_id}"
                )

            # Record init start time and start heartbeat
            init_start_time = time.time()
            self.heartbeat_manager.start(init_start_time)

            logger.debug(
                f"Workers started waiting for heartbeats spmd_id={self.spmd_id}"
            )
        except Exception as e:
            logger.error(f"Initialization failed error={e} spmd_id={self.spmd_id}")
            self.state_machine.trigger(
                DriverEvent.INIT_FAILED,
                failure=ExecutionFailure.from_exception(
                    spmd_id=self.spmd_id,
                    rank=None,
                    phase="DRIVER_INITIALIZATION",
                    exception=e,
                ),
            )
            await self._wait_for_driver_effects()
            await self._effect_scheduler.close()
            raise
        finally:
            if span is not None:
                await span.__aexit__(None, None, None)

    def get_attested_node_ids(self) -> tuple[str, ...]:
        return self.executor.get_attested_node_ids()

    def get_worker_handles(self) -> dict[int, Any]:
        return self.executor.get_worker_handles()

    async def inspect_operator_state(self) -> dict[int, Any]:
        return await self.executor.inspect_operator_state()

    async def export_profiler_snapshots(self) -> dict[Any, Any]:
        snapshots = await self.executor.export_profiler_snapshots()
        if self._profiler_runtime.enabled:
            snapshots["driver"] = self._profiler_runtime.snapshot()
        return snapshots

    def register_topology(
        self,
        predecessor_handlers: dict[str, dict[int, "DataPlaneCallable"]],
        successor_handlers: dict[str, dict[int, "DataPlaneCallable"]],
        predecessor_count: int,
        successor_count: int,
        local_spmd_code: int,
        spmd_id_by_code: dict[int, str],
    ) -> None:
        logger.debug(
            f"Topology registration starting "
            f"predecessor_count={predecessor_count} successor_count={successor_count} "
            f"spmd_id={self.spmd_id}"
        )
        if self._profiler_runtime.enabled:
            self._profiler_runtime.recorder.worker_scheduler_instant(
                spec=DRIVER_REGISTER_TOPOLOGY,
                payload=(predecessor_count, successor_count),
            )

        self.executor.register_topology(
            predecessor_handlers,
            successor_handlers,
            predecessor_count,
            successor_count,
            local_spmd_code,
            spmd_id_by_code,
        )

        logger.debug(f"Topology registration completed spmd_id={self.spmd_id}")

    def register_credit_topology(
        self,
        upstream_driver_handles: dict[str, Any],
        upstream_edges: tuple[Any, ...],
        downstream_edges: tuple[Any, ...],
        epoch: int = 0,
    ) -> None:
        self.credit_manager.register_credit_topology(
            upstream_driver_handles, upstream_edges, downstream_edges, epoch
        )

    def apply_credit_window(self, windows: tuple[CreditWindow, ...]) -> None:
        self.credit_manager.apply_credit_window(windows)

    def receive_credit_update(
        self, downstream_id: str, edge_id: str, update: CreditUpdate
    ) -> None:
        self.credit_manager.receive_credit_update(downstream_id, edge_id, update)

    def _block_all_workers(self) -> None:
        """Block all workers (internal method)."""
        ops = tuple(self.spmd_synchronizer.get_spmd_steps())
        for op_name in ops:
            self.spmd_synchronizer.block_spmd_op(op_name)

    def _submit_driver_effect(
        self,
        operation: Any,
        *,
        args: tuple[Any, ...] = (),
        effect_name: str,
        state_revision: int,
    ) -> None:
        if not self._effect_scheduler.started:
            raise RuntimeError("Driver effect scheduler is not started")
        submission = LifecycleEffectSubmission(
            operation=operation,
            args=args,
            ordering_key=f"driver:{self.spmd_id}",
            state_revision=state_revision,
            mode=LifecycleEffectExecutionMode.THREAD,
        )
        outcome = self._effect_scheduler.submit(submission)
        self._driver_effect_outcomes.add(outcome)
        outcome.add_done_callback(
            lambda completed: self._driver_effect_finished(
                completed, effect_name=effect_name
            )
        )

    def _driver_effect_finished(
        self,
        outcome: asyncio.Future,
        *,
        effect_name: str,
    ) -> None:
        self._driver_effect_outcomes.discard(outcome)
        if outcome.cancelled():
            return
        result = outcome.result()
        if result.error is not None:
            if result.state_revision != self.state_machine.state_revision:
                logger.error(
                    "Stale Driver effect failure state_revision=%s current=%s effect=%s: %s",
                    result.state_revision,
                    self.state_machine.state_revision,
                    effect_name,
                    result.error,
                    exc_info=result.error,
                )
                return
            self._report_component_failure(
                RuntimeError(f"driver effect {effect_name} failed: {result.error}")
            )

    async def _wait_for_driver_effects(self) -> None:
        outcomes = tuple(self._driver_effect_outcomes)
        if outcomes:
            await asyncio.gather(*outcomes)

    @property
    def _stopping(self) -> bool:
        return self.state_machine.state in (DriverState.STOPPING, DriverState.STOPPED)

    async def stop(self, save_checkpoint: bool = False) -> None:
        if save_checkpoint:
            raise ValueError(
                "Driver stop requires checkpoint coordination before shutdown"
            )
        if self._shutdown_task is None:
            self.state_machine.require_transition(DriverEvent.STOP)
            self._shutdown_task = asyncio.create_task(self._shutdown())
            self._shutdown_task.add_done_callback(self._shutdown_finished)
        await asyncio.shield(self._shutdown_task)

    def _shutdown_finished(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _shutdown(self) -> None:
        logger.debug(f"Stopping spmd_id={self.spmd_id}")
        shutdown_errors: list[Exception] = []
        try:
            await self._wait_for_driver_component_notifications()
        except Exception as error:
            shutdown_errors.append(error)
        try:
            await self._notify_driver_components(DriverState.STOPPING)
        except Exception as error:
            shutdown_errors.append(error)
        try:
            await self.heartbeat_manager.stop()
        except Exception as error:
            shutdown_errors.append(error)

        try:
            shutdown_errors.extend(await self.executor.stop_workers())
        except Exception as error:
            shutdown_errors.append(error)
        try:
            await self._wait_for_driver_effects()
            await self._effect_scheduler.close()
        except Exception as error:
            shutdown_errors.append(error)
        self.state_machine.require_transition(DriverEvent.STOPPED)

        if shutdown_errors:
            raise ExceptionGroup(
                f"SPMD {self.spmd_id!r} shutdown failed",
                shutdown_errors,
            )

    # =====================================================================
    # Snapshot Orchestration
    # =====================================================================

    async def request_drain(self, admission: DrainAdmission) -> None:
        await self.snapshot_orchestrator.request_drain(admission)

    async def dispatch_component_request(
        self, component_id: str, request: object
    ) -> object:
        return await self._driver_components.dispatch_request(component_id, request)

    async def forward_control_plane_request(self, request: object) -> object:
        if self.controller_handle is None:
            raise RuntimeError(
                f"SPMD {self.spmd_id!r} cannot forward a control request without Controller"
            )
        return await self.controller_handle.dispatch_control_plane_request.remote(
            request
        )

    async def prepare_snapshot(
        self,
        checkpoint_id: str,
        attempt_id: str,
        permit: Any,
        cut: CheckpointCut,
        repository_root: str,
    ) -> None:
        await self.snapshot_orchestrator.prepare_snapshot(
            checkpoint_id, attempt_id, permit, cut, repository_root
        )

    async def capture_driver_components(
        self, checkpoint_id: str, cut: CheckpointCut
    ) -> SpmdCheckpoint:
        return self.snapshot_orchestrator.capture_driver_components(checkpoint_id, cut)

    async def resume(self) -> None:
        await self.snapshot_orchestrator.resume()

    async def publish_snapshot(self, checkpoint_id: str, permit: Any) -> bool:
        return await self.snapshot_orchestrator.publish_save(checkpoint_id, permit)

    async def rollback_snapshot(self, checkpoint_id: str, permit: Any) -> None:
        await self.snapshot_orchestrator.rollback_save(checkpoint_id, permit)

    async def fail_snapshot(self, checkpoint_id: str) -> None:
        await self.snapshot_orchestrator.fail_snapshot(checkpoint_id)

    async def restore_from_manifest(
        self,
        checkpoint_id: str,
        cut: CheckpointCut,
        snapshot: SpmdCheckpoint,
        repository_root: str,
        restore_options_by_rank: RestoreOptionsByRank | None = None,
    ) -> None:
        await self.snapshot_orchestrator.restore_from_manifest(
            checkpoint_id,
            cut,
            snapshot,
            repository_root,
            restore_options_by_rank,
        )

    # =====================================================================
    # Worker Heartbeat Handling
    # =====================================================================

    def receive_worker_heartbeat(self, rank: int, heartbeat: WorkerHeartbeat) -> None:
        if self._stopping:
            logger.debug(
                f"Ignoring worker heartbeat during shutdown rank={rank} spmd_id={self.spmd_id}"
            )
            return

        self._worker_heartbeats[rank] = heartbeat
        self._worker_heartbeat_times[rank] = time.time()
        if self._profile_heartbeat:
            self._profiler_runtime.recorder.driver_worker_heartbeat(
                rank=rank,
                status=heartbeat.status.name,
            )
        logger.heartbeat(
            f"Received worker heartbeat rank={rank} "
            f"status={heartbeat.status.name} spmd_id={self.spmd_id}"
        )

        if heartbeat.failure is not None:
            self.receive_worker_failure(rank, heartbeat.failure)
            return

        if not self._workers_ready:
            if len(self._worker_heartbeat_times) >= self.config.num_workers:
                self._workers_ready = True
                logger.heartbeat(
                    f"All workers ready worker_count={self.config.num_workers} spmd_id={self.spmd_id}"
                )

                if self.state_machine.state == DriverState.INITIALIZING:
                    self.state_machine.trigger(DriverEvent.INIT_COMPLETE)
                    logger.heartbeat(f"INIT_COMPLETE triggered spmd_id={self.spmd_id}")

        if not self._topology_ready and self._workers_ready:
            from fluxway.common.types import WorkerStatus

            all_ready = all(
                hb.status
                in (
                    WorkerStatus.READY,
                    WorkerStatus.RESTORING,
                    WorkerStatus.RUNNING,
                    WorkerStatus.BLOCKED,
                )
                for hb in self._worker_heartbeats.values()
            )
            if all_ready and len(self._worker_heartbeats) >= self.config.num_workers:
                self._topology_ready = True
                logger.heartbeat(f"All worker runtimes ready spmd_id={self.spmd_id}")

    def receive_worker_failure(self, rank: int, failure: ExecutionFailure) -> None:
        if self._stopping:
            logger.debug(
                "Ignoring worker failure during shutdown rank=%s spmd_id=%s",
                rank,
                self.spmd_id,
            )
            return
        if not isinstance(failure, ExecutionFailure):
            raise TypeError(
                "receive_worker_failure requires ExecutionFailure, "
                f"got {type(failure).__name__}."
            )
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError(f"Worker failure rank must be non-negative, got {rank!r}.")
        if failure.spmd_id != self.spmd_id:
            raise ValueError(
                f"Worker failure SPMD {failure.spmd_id!r} does not match "
                f"driver {self.spmd_id!r}."
            )
        if failure.rank != rank:
            raise ValueError(
                f"Worker failure rank {failure.rank!r} does not match "
                f"reported rank {rank}."
            )
        logger.error("Worker failure rank=%s failure=%s", rank, failure)
        self.state_machine.trigger(DriverEvent.WORKER_ERROR, failure=failure)

    # =====================================================================
    # Worker Op Completion Handling
    # =====================================================================

    def report_worker_op_completed(self, rank: int, op_name: str, step: int) -> None:
        if self._stopping:
            logger.debug(
                f"Ignoring worker op completion during shutdown rank={rank} "
                f"op={op_name} step={step} spmd_id={self.spmd_id}"
            )
            return

        if self._profiler_runtime.enabled:
            self._profiler_runtime.recorder.driver_op_completion(
                rank=rank,
                op_name=op_name,
                step=step,
            )
        self.spmd_synchronizer.report_worker_op_completed(rank, op_name, step)

    def report_boundary_event(self, op_name: str, event: DrainEvent) -> None:
        if not isinstance(event, DrainEvent):
            raise TypeError("boundary event must be a DrainEvent")
        if self.controller_handle is None:
            raise RuntimeError(
                f"Driver {self.spmd_id!r} cannot report a boundary event without "
                "a controller handle."
            )
        previous = self.spmd_synchronizer.drain_boundary
        self.spmd_synchronizer.apply_boundary_event(event)
        boundary = self.spmd_synchronizer.drain_boundary
        if boundary.revision == previous.revision:
            return
        self.controller_handle.report_boundary_event.remote(
            self.spmd_id, op_name, event
        )

    def _on_spmd_op_completed(self, spmd_id: str, op_name: str, step: int) -> None:
        if self.controller_handle is None:
            raise RuntimeError(
                f"Driver {self.spmd_id!r} cannot report op completion without "
                "a controller handle."
            )
        self.controller_handle.report_op_completed.remote(spmd_id, op_name, step)
        self.credit_manager.send_credit_updates(op_name, step)

    # =====================================================================
    # Op Blocking
    # =====================================================================

    def block_op(self, op_name: str) -> None:
        self.spmd_synchronizer.block_spmd_op(op_name)

    def unblock_op(self, op_name: str) -> None:
        self.spmd_synchronizer.unblock_spmd_op(op_name)

    def block_all_ops(self) -> None:
        self.spmd_synchronizer.block_all_ops()
        self.state_machine.trigger(DriverEvent.BLOCK_REQUESTED)

    def unblock_all_ops(self) -> None:
        self.spmd_synchronizer.unblock_all_ops()
        self.state_machine.trigger(DriverEvent.UNBLOCK_REQUESTED)

    # =====================================================================
    # State Argument Broadcasting
    # =====================================================================

    def broadcast_control_argument(
        self, target_op: str, names: list[str], step: int, values: list[Any]
    ) -> None:
        self.executor.broadcast_control_argument(target_op, names, step, values)
        logger.debug(
            f"Broadcast control_arguments to all workers names={names} step={step} "
            f"spmd_id={self.spmd_id}"
        )

    def routing_control_argument_to_controller(
        self,
        target_spmd_id: str,
        target_op: str,
        names: list[str],
        step: int,
        values: list[Any],
    ) -> None:
        if self._stopping:
            logger.debug(
                f"Ignoring control_argument routing during shutdown target_spmd_id={target_spmd_id} "
                f"step={step} spmd_id={self.spmd_id}"
            )
            return

        if self.controller_handle is not None:
            try:
                self.controller_handle.receive_control_argument.remote(
                    self.spmd_id, target_spmd_id, target_op, names, step, values
                )
                logger.debug(
                    f"Routed control_arguments to controller names={names} step={step} "
                    f"spmd_id={self.spmd_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to route control_arguments to controller error={e} "
                    f"spmd_id={self.spmd_id}"
                )

    # =====================================================================
    # Controller Reporting
    # =====================================================================

    def _report_completion(self) -> None:
        if self.controller_handle is not None:
            self.controller_handle.report_spmd_completed.remote(self.spmd_id)

    def _report_error(self, failure: ExecutionFailure) -> None:
        if self.controller_handle is not None:
            self.controller_handle.report_spmd_error.remote(self.spmd_id, failure)

    # =====================================================================
    # Topology Update
    # =====================================================================

    def update_topology(self, predecessors: list[str], successors: list[str]) -> None:
        self.graph_topology["predecessors"] = predecessors
        self.graph_topology["successors"] = successors
        logger.debug(
            f"Topology updated predecessors={predecessors} successors={successors} "
            f"spmd_id={self.spmd_id}"
        )

    # =====================================================================
    # Status Query
    # =====================================================================

    def get_state(self) -> DriverState:
        return self.state_machine.state

    def get_status(self) -> dict[str, Any]:
        return {
            "spmd_id": self.spmd_id,
            "state": self.state_machine.state.value,
            "node_type": "spmd",
            "num_workers": self.config.num_workers,
            "completed_step_count": self.spmd_synchronizer.get_spmd_steps(),
            "locked_ops": self.spmd_synchronizer.get_locked_ops(),
            "failure": self.state_machine.failure,
            "worker_failures": self.state_machine.get_worker_failures(),
        }

    # =====================================================================
    # Source Node Control
    # =====================================================================

    def pause_sources(self) -> None:
        if self._is_paused:
            return

        self._is_paused = True
        self._report_flow_control_state()
        logger.debug(f"Sources paused due to backpressure spmd_id={self.spmd_id}")
        self.executor.pause_sources()

    def resume_sources(self) -> None:
        if not self._is_paused:
            return

        self._is_paused = False
        self._report_flow_control_state()
        logger.debug(f"Sources resumed spmd_id={self.spmd_id}")
        self.executor.resume_sources()

    async def activate_execution(self) -> None:
        self.state_machine.require_transition(DriverEvent.START)
        await self.executor.activate_execution()
        self.state_machine.require_transition(DriverEvent.ACTIVATION_COMPLETE)
        self.spmd_synchronizer.unblock_all_ops()
        await self._notify_driver_components(DriverState.RUNNING)

    async def start_source_operators(self) -> None:
        if self.state_machine.state not in {
            DriverState.RUNNING,
            DriverState.DRAINING,
            DriverState.DRAINED,
        }:
            raise RuntimeError(
                "source operators require a running Driver, got "
                f"{self.state_machine.state.name} spmd_id={self.spmd_id}"
            )
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_source_start(
                worker_count=len(self.executor.get_worker_handles()),
            ):
                self._report_flow_control_state()
                await self.executor.start_source_operators()
            return

        self._report_flow_control_state()
        await self.executor.start_source_operators()

    def _report_flow_control_state(self) -> None:
        if self.controller_handle is None:
            raise RuntimeError(
                f"Driver {self.spmd_id!r} cannot report flow-control state "
                "without a controller handle."
            )
        self.controller_handle.report_flow_control_state.remote(
            FlowControlState(
                spmd_id=self.spmd_id,
                timestamp=time.time(),
                paused=self._is_paused,
                epoch=self.credit_manager.get_credit_epoch(),
            )
        )

    @property
    def driver_components(self) -> DriverComponentRegistry:
        return self._driver_components

    def _schedule_driver_component_state(self, state: DriverState) -> None:
        if not self._driver_components.component_ids:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "Driver component state notification requires an active event loop"
            ) from None
        task = loop.create_task(self._notify_driver_components(state))
        self._driver_component_notification_tasks.add(task)

    async def _wait_for_driver_component_notifications(self) -> None:
        tasks = tuple(self._driver_component_notification_tasks)
        self._driver_component_notification_tasks.difference_update(tasks)
        if tasks:
            await asyncio.gather(*tasks)

    async def _notify_driver_components(self, state: DriverState) -> None:
        try:
            await self._driver_components.notify_state(state)
        except Exception as error:
            failure = ExecutionFailure.from_exception(
                spmd_id=self.spmd_id,
                rank=None,
                phase="DRIVER_COMPONENT",
                exception=error,
            )
            if self.state_machine.state is not DriverState.ERROR:
                self.state_machine.trigger(
                    DriverEvent.COMPONENT_ERROR,
                    failure=failure,
                )
            raise

    def _report_component_failure(self, error: BaseException) -> None:
        failure = ExecutionFailure.from_exception(
            spmd_id=self.spmd_id,
            rank=None,
            phase="DRIVER_COMPONENT",
            exception=error,
        )
        if self.state_machine.state is not DriverState.ERROR:
            self.state_machine.trigger(
                DriverEvent.COMPONENT_ERROR,
                failure=failure,
            )
