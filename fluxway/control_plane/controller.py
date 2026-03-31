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
import os
import time
from typing import Any, Dict, List, Optional

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None  # type: ignore
from fluxway.common.types import (
    AsyncStepConfig,
    ControllerEvent,
    ControllerState,
    DriverHeartbeat,
    Graph,
    HeartbeatConfig,
)
from fluxway.common.utils import ensure_logging_configured
from fluxway.common.utils.logging import get_logger
from fluxway.control_plane.monitor import Monitor
from fluxway.control_plane.scheduler import Scheduler
from fluxway.control_plane.spmd_manager import SPMDManager, SPMDHandle
from fluxway.control_plane.state_manager import StateManager
from fluxway.control_plane.heartbeat import HeartbeatManager
from fluxway.control_plane.state_machine import ControllerStateMachine
from fluxway.control_plane.async_manager import AsyncStepManager
from fluxway.common.types import CheckpointConfig

logger = get_logger(__name__)


def require_manager(manager_attr: str):
    """
    Decorator to ensure that a manager component is initialized before calling
    the method.

    Two cases when the manager attribute is None:
    - Controller not yet initialized (_initialized=False): transient startup race,
      the caller (driver) will retry on the next heartbeat cycle. Log a warning
      and return without raising so the fire-and-forget ObjectRef is not lost silently.
    - Controller already initialized (_initialized=True): this is a post-init bug;
      raise RuntimeError unconditionally.
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                if getattr(self, manager_attr) is None:
                    if not self._initialized:
                        logger.warning(
                            f"{func.__name__} called before controller initialized, dropping"
                        )
                        return
                    raise RuntimeError(
                        f"{manager_attr.strip('_')} is None after initialization "
                        f"when calling {func.__name__}"
                    )
                return await func(self, *args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                if getattr(self, manager_attr) is None:
                    if not self._initialized:
                        logger.warning(
                            f"{func.__name__} called before controller initialized, dropping"
                        )
                        return
                    raise RuntimeError(
                        f"{manager_attr.strip('_')} is None after initialization "
                        f"when calling {func.__name__}"
                    )
                return func(self, *args, **kwargs)

            return sync_wrapper

    return decorator


def require_spmd_manager(func):
    return require_manager("_spmd_manager")(func)


def require_heartbeat_manager(func):
    return require_manager("_heartbeat_manager")(func)


if RAY_AVAILABLE:

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

        def __init__(
            self,
            graph: Graph,
            heartbeat_config: Optional[HeartbeatConfig] = None,
            async_step_config: Optional[AsyncStepConfig] = None,
            checkpoint_config: Optional[CheckpointConfig] = None,
        ):
            """
            Initialize Controller with configuration.

            Args:
                graph: The task graph defining SPMD nodes and edges
                heartbeat_config: Heartbeat configuration
                async_step_config: Async step configuration
                checkpoint_config: Configuration for checkpointer backend
            """
            # Ray actors run in separate worker processes; ensure INFO logs are not dropped.
            ensure_logging_configured()

            self._graph = graph
            self._heartbeat_config = heartbeat_config or HeartbeatConfig()
            self._async_step_config = async_step_config or AsyncStepConfig()
            self._checkpoint_config = checkpoint_config

            self._monitor = Monitor()
            self._state_manager = StateManager()
            self._scheduler = Scheduler(spmd_manager=None)
            self._state_machine = ControllerStateMachine(self)
            self._async_step_manager = AsyncStepManager(graph, self._async_step_config)
            # snapshot_id → {expected, received, metadata, event}
            self._pending_snapshots: Dict[str, Any] = {}

            self._checkpoint_task: Optional[asyncio.Task] = None
            self._last_checkpointed_step: int = 0

            # Track source nodes (entry points)
            self._source_nodes = (
                graph.source_nodes if graph.source_nodes else self._find_source_nodes()
            )

            # Initialization flag
            self._initialized = False
            self._spmd_manager: Optional[SPMDManager] = None
            self._heartbeat_manager: Optional[HeartbeatManager] = None
            self._checkpointer = None

            # Backpressure tracking
            self._busy_spmds: Dict[str, float] = {}  # {spmd_id: busy_worker_ratio}
            self._backpressure_threshold: float = (
                0.5  # 50% workers busy triggers backpressure
            )
            self._paused_readers: set = set()  # Set of paused reader IDs

            logger.debug(f"ControllerActor created with {len(graph.nodes)} nodes")

        def _find_source_nodes(self) -> List[str]:
            """Find nodes with no predecessors (source nodes)."""
            all_targets = set()
            for edge in self._graph.edges:
                all_targets.add(edge.target_node)

            source_nodes = []
            for node_name in self._graph.nodes:
                if node_name not in all_targets:
                    source_nodes.append(node_name)

            return source_nodes

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
            - TOPOLOGY_REGISTERING → RUNNING: when all workers are running

            Note: Heartbeat timeout monitoring starts in RUNNING state entry,
            not here. Heartbeat *reception* works from the start.

            Returns:
                True if initialization successful
            """
            try:
                # Create sub-components locally in the controller process
                self_handle = ray.runtime_context.get_runtime_context().current_actor
                self._spmd_manager = SPMDManager(controller_handle=self_handle)
                self._scheduler.set_spmd_manager(
                    self._spmd_manager
                )  # Set SPMDManager on scheduler

                self._heartbeat_manager = HeartbeatManager(
                    config=self._heartbeat_config,
                    on_all_ready=self._on_all_drivers_ready,
                    on_all_workers_running=self._on_all_workers_running,
                    on_driver_error=self._on_driver_error,
                    on_driver_recovered=self._on_driver_recovered,
                )

                if self._checkpoint_config is not None:
                    from fluxway.control_plane.checkpointer import (
                        FluxwayCheckpointer,
                        WorkerCheckpointBackend,
                    )

                    self._checkpointer = FluxwayCheckpointer(
                        self,
                        ckpt_dir=self._checkpoint_config.ckpt_dir,
                        interval_steps=self._checkpoint_config.interval_steps,
                        force_write=self._checkpoint_config.force_write,
                    )
                    self._spmd_manager.set_checkpoint_backend(
                        WorkerCheckpointBackend.create(
                            self._checkpoint_config.backend,
                            force_write=self._checkpoint_config.force_write,
                        )
                    )

                # Assign specific max_steps to each node based on global config and overrides
                for node_id, node in self._graph.nodes.items():
                    limit = self._async_step_config.node_max_steps.get(
                        node_id, self._async_step_config.max_steps
                    )
                    if limit is not None:
                        node.spmd_config.max_steps = limit
                        logger.debug(f"Assigned max_steps={limit} to node {node_id}")

                # Register SPMDs for heartbeat monitoring
                for node_name in self._graph.nodes:
                    self._heartbeat_manager.register_spmd(node_name)

                # Allocate resources
                if not await self._scheduler.allocate_resources(self._graph):
                    logger.error("Failed to allocate resources")
                    return False

                # Build and establish topology data structure
                # (adjacency lists only, no remote calls to drivers)
                self._scheduler.build_and_establish_topology(self._graph)

                # Start SPMDs (fire-and-forget, readiness tracked via heartbeat)
                if not await self._scheduler.init_spmds(self._graph):
                    logger.error("Failed to start SPMDs")
                    return False

                # Transition state machine: INITIALIZING → PENDING
                await self._state_machine.trigger_event(ControllerEvent.INIT_COMPLETE)

                self._initialized = True
                logger.debug(
                    f"Controller initialized with {len(self._graph.nodes)} SPMDs, "
                    f"waiting for drivers to report ready via heartbeat"
                )
                return True

            except Exception as e:
                logger.error(f"Controller initialization failed: {e}")
                return False

        # =====================================================================
        # Properties (as methods for Ray Actor)
        # =====================================================================

        def get_graph(self) -> Graph:
            """Get the task graph."""
            return self._graph

        def get_state(self) -> ControllerState:
            """Get current controller state."""
            return self._state_machine.state

        async def get_status(self) -> Dict[str, Any]:
            """Get detailed controller status."""
            return await self.get_cluster_status()

        @require_heartbeat_manager
        async def get_all_heartbeats(self) -> Dict[str, Any]:
            """Get all latest heartbeats from all SPMDs."""
            return await self._heartbeat_manager.get_all_heartbeats()

        def is_initialized(self) -> bool:
            """Check if controller is initialized."""
            return self._initialized

        # =====================================================================
        # SPMD Handle Access
        # =====================================================================

        @require_spmd_manager
        def get_spmd_handle(self, spmd_id: str) -> Optional[SPMDHandle]:
            """
            Get handle to an SPMD instance from spmd_manager.

            Args:
                spmd_id: ID of the SPMD

            Returns:
                SPMDHandle or None if not found
            """
            return self._spmd_manager.get_spmd_handle(spmd_id)

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
            # Route to heartbeat manager
            await self._heartbeat_manager.receive_heartbeat(heartbeat)

            # Update async step manager with completed count
            if heartbeat.completed_step_count:
                self._async_step_manager.report_completed_count(
                    heartbeat.spmd_id, heartbeat.completed_step_count
                )

            # Check and enforce step limits
            await self._check_and_enforce_step_limits()

            # Update busy status for backpressure
            self._busy_spmds[heartbeat.spmd_id] = heartbeat.busy_worker_ratio

            # Check and enforce backpressure for readers
            await self._check_and_enforce_backpressure()

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
            self._async_step_manager.report_op_completed(
                spmd_id, op_name, completed_count
            )
            await self._check_and_enforce_step_limits()
            await self._maybe_trigger_checkpoint(completed_count)

        async def _check_and_enforce_step_limits(self) -> None:
            """Check step differences and apply lock/unlock actions."""
            if self._async_step_manager.is_all_completed():
                logger.debug("All operators reached max_steps, triggering completion")
                await self._state_machine.trigger_event(
                    ControllerEvent.ALL_DRIVERS_COMPLETED
                )
                return

            for (
                spmd_id,
                op_name,
                allowed_step,
            ) in self._async_step_manager.check_and_enforce_limits():
                await self._set_spmd_allowed_step(spmd_id, op_name, allowed_step)

        async def _maybe_trigger_checkpoint(self, completed_count: int) -> None:
            """
            Event-driven proactive checkpoint trigger.

            A checkpoint is scheduled only when every SPMD's every op has
            completed at least `next_ckpt` steps, ensuring the snapshot is globally
            consistent with drain_to_step_count mechanism.
            """
            # Prevent fast-starting operators from triggering global checkpoints before
            # all drivers are fully compiled and initialized across the cluster.
            if not self._state_machine.is_running():
                return
            if self._checkpointer is None:
                return
            if self._checkpoint_task is not None and not self._checkpoint_task.done():
                return

            interval = self._checkpointer.interval_steps
            next_ckpt = self._last_checkpointed_step + interval
            if completed_count < next_ckpt:
                return

            # Advance the watermark before launching the task so that a
            # persistent failure (e.g. disk full) skips this boundary and
            # moves on to the next, rather than retrying the same boundary on
            # every subsequent report_op_completed call.
            self._last_checkpointed_step = next_ckpt

            async def do_ckpt(boundary: int) -> None:
                logger.info(
                    f"Proactively scheduled checkpoint boundary at count {boundary}"
                )
                try:
                    await self._checkpointer.save(boundary)
                except Exception as e:
                    logger.error(
                        f"Checkpoint at step {boundary} failed and will be skipped: {e}"
                    )

            self._checkpoint_task = asyncio.create_task(do_ckpt(next_ckpt))

        @require_spmd_manager
        async def _set_spmd_allowed_step(
            self, spmd_id: str, op_name: str, allowed_step: int
        ) -> None:
            """Propagate global watermark to an SPMD Driver."""
            driver = self._spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                try:
                    await driver.set_allowed_step.remote(op_name, allowed_step)
                except Exception as e:
                    logger.error(
                        f"Failed to set allowed_step {allowed_step} for {spmd_id}.{op_name}: {e}"
                    )

        @require_spmd_manager
        async def _lock_spmd_op(self, spmd_id: str, op_name: str) -> None:
            """Lock an SPMD's operation."""
            driver = self._spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                try:
                    await driver.block_op.remote(op_name)
                    # logger.debug(f"Locked {spmd_id} {op_name}")
                except Exception as e:
                    logger.error(f"Failed to lock {spmd_id} {op_name}: {e}")

        @require_spmd_manager
        async def _unlock_spmd_op(self, spmd_id: str, op_name: str) -> None:
            """Unlock an SPMD's operation."""
            driver = self._spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                try:
                    await driver.unblock_op.remote(op_name)
                    # logger.debug(f"Unlocked {spmd_id} {op_name}")
                except Exception as e:
                    logger.error(f"Failed to unlock {spmd_id} {op_name}: {e}")

        # =====================================================================
        # Backpressure Management for Reader Nodes
        # =====================================================================

        @require_spmd_manager
        async def _check_and_enforce_backpressure(self) -> None:
            """
            Check downstream SPMD busy status and apply backpressure to readers.

            Logic:
            - For each Reader node, check if any of its downstream SPMDs are busy
            - If downstream is busy (busy_worker_ratio > threshold), pause the reader
            - If downstream is no longer busy, resume the reader
            """
            # Source nodes are candidates for backpressure (e.g., readers)
            reader_nodes = self._source_nodes

            for reader_id in reader_nodes:
                # Get downstream SPMDs for this reader
                downstream_spmds = self._spmd_manager.get_successors(reader_id)

                # Check if any downstream SPMD is busy
                is_downstream_busy = False
                for downstream_id in downstream_spmds:
                    busy_ratio = self._busy_spmds.get(downstream_id, 0.0)
                    if busy_ratio >= self._backpressure_threshold:
                        is_downstream_busy = True
                        break

                # Apply backpressure actions
                if is_downstream_busy and reader_id not in self._paused_readers:
                    # Pause reader
                    await self._pause_reader(reader_id)
                elif not is_downstream_busy and reader_id in self._paused_readers:
                    # Resume reader
                    await self._resume_reader(reader_id)

        @require_spmd_manager
        async def _pause_reader(self, reader_id: str) -> None:
            """
            Pause a reader due to backpressure.

            Args:
                reader_id: Reader SPMD ID to pause
            """
            driver = self._spmd_manager.get_driver_handle(reader_id)
            if driver is not None:
                try:
                    await driver.pause_sources.remote()
                    self._paused_readers.add(reader_id)
                    logger.debug(f"Paused reader {reader_id} due to backpressure")
                except Exception as e:
                    logger.error(f"Failed to pause reader {reader_id}: {e}")

        @require_spmd_manager
        async def _resume_reader(self, reader_id: str) -> None:
            """
            Resume a paused reader.

            Args:
                reader_id: Reader SPMD ID to resume
            """
            driver = self._spmd_manager.get_driver_handle(reader_id)
            if driver is not None:
                try:
                    await driver.resume_sources.remote()
                    self._paused_readers.discard(reader_id)
                    logger.debug(f"Resumed reader {reader_id}")
                except Exception as e:
                    logger.error(f"Failed to resume reader {reader_id}: {e}")

        def get_paused_readers(self) -> List[str]:
            """Get list of currently paused readers."""
            return list(self._paused_readers)

        # =====================================================================
        # State Argument Transport
        # =====================================================================

        @require_spmd_manager
        async def receive_control_argument(
            self,
            source_spmd_id: str,
            target_spmd_id: str,
            target_op: str,
            names: List[str],
            step: int,
            values: List[Any],
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
            target_driver = self._spmd_manager.get_driver_handle(target_spmd_id)
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
            self, snapshot_id: str, target_count: int, ckpt_dir: str
        ) -> Dict[str, Any]:
            """
            Trigger a global snapshot and block until all Workers have persisted
            their state and all Drivers have reported metadata back.

            Returns the assembled manifest directly — no polling required.
            """
            logger.info(
                f"Requesting global snapshot '{snapshot_id}' at step {target_count}"
            )
            all_spmd_ids = self._spmd_manager.get_all_spmd_ids()
            if not all_spmd_ids:
                return {}

            # validate all drivers exist BEFORE registering the snapshot entry
            # and before dispatching any RPCs.
            drivers: Dict[str, Any] = {}
            for spmd_id in all_spmd_ids:
                driver = self._spmd_manager.get_driver_handle(spmd_id)
                if driver is None:
                    raise RuntimeError(
                        f"request_snapshot: driver for spmd_id '{spmd_id}' not found. "
                        f"Cannot perform partial snapshot — aborting to prevent corrupt manifest."
                    )
                drivers[spmd_id] = driver

            # Register the accounting entry, then dispatch — all drivers confirmed.
            done_event = asyncio.Event()
            self._pending_snapshots[snapshot_id] = {
                "expected": len(all_spmd_ids),
                "received": 0,
                "reported": set(),  # idempotent guard: each spmd_id counted at most once
                "metadata": {},
                "errors": [],
                "ckpt_dir": ckpt_dir,
                "event": done_event,
            }
            for spmd_id, driver in drivers.items():
                driver.request_snapshot.remote(snapshot_id, target_count, ckpt_dir)

            snapshot_timeout = int(
                os.environ.get("FLUXWAY_SNAPSHOT_TIMEOUT_SECS", "600")
            )
            try:
                await asyncio.wait_for(done_event.wait(), timeout=snapshot_timeout)
            except asyncio.TimeoutError as err:
                self._pending_snapshots.pop(snapshot_id, None)
                raise RuntimeError(
                    f"Snapshot '{snapshot_id}' timed out after {snapshot_timeout}s. "
                    f"A Driver may have crashed without calling report_snapshot_failed. "
                    f"Set FLUXWAY_SNAPSHOT_TIMEOUT_SECS to adjust."
                ) from err

            snap_info = self._pending_snapshots.pop(snapshot_id)
            if snap_info["errors"]:
                raise RuntimeError(
                    f"Snapshot '{snapshot_id}' failed for SPMDs: {snap_info['errors']}. "
                    f"Refusing to persist incomplete manifest."
                )
            manifest = {
                sid: {str(rank): m for rank, m in per_rank.items()}
                for sid, per_rank in snap_info["metadata"].items()
            }
            return manifest

        @require_spmd_manager
        async def receive_snapshot_metadata(
            self, spmd_id: str, snapshot_id: str, metadata_futures: Dict[int, Any]
        ) -> None:
            snap_info = self._pending_snapshots.get(snapshot_id)
            if snap_info is None:
                logger.warning(
                    f"Received metadata for unknown snapshot '{snapshot_id}'"
                )
                return

            # Idempotency guard: a Driver may call both receive_snapshot_metadata
            # (which fails partway) and report_snapshot_failed.  Only the first
            # call for a given spmd_id is counted to prevent received > expected.
            if spmd_id in snap_info["reported"]:
                logger.warning(
                    f"receive_snapshot_metadata: duplicate report from '{spmd_id}' "
                    f"for snapshot '{snapshot_id}' — ignoring"
                )
                return
            snap_info["reported"].add(spmd_id)

            try:
                ranks = list(metadata_futures.keys())
                refs = list(metadata_futures.values())
                results = await asyncio.gather(*refs)
                resolved = dict(zip(ranks, results, strict=True))
            except Exception as e:
                logger.error(f"Failed to resolve snapshot metadata for {spmd_id}: {e}")
                snap_info["errors"].append(spmd_id)
                snap_info["received"] += 1
                if snap_info["received"] == snap_info["expected"]:
                    snap_info["event"].set()
                return

            snap_info["metadata"][spmd_id] = resolved
            snap_info["received"] += 1

            if snap_info["received"] == snap_info["expected"]:
                logger.info(f"Snapshot '{snapshot_id}' complete — resuming all Drivers")
                for s_id in self._spmd_manager.get_all_spmd_ids():
                    driver = self._spmd_manager.get_driver_handle(s_id)
                    if driver is not None:
                        driver.resume.remote()
                snap_info["event"].set()

        async def report_snapshot_failed(self, spmd_id: str, snapshot_id: str) -> None:
            """Unblock request_snapshot when a Driver triggers SNAPSHOT_FAILED.

            Called by a Driver that encountered an unrecoverable error during
            persistence and could not (or failed to) deliver metadata via
            receive_snapshot_metadata.  Advances the received counter so the
            Controller's done_event is eventually set rather than waiting forever.
            """
            snap_info = self._pending_snapshots.get(snapshot_id)
            if snap_info is None:
                logger.warning(
                    f"report_snapshot_failed: unknown snapshot '{snapshot_id}' from {spmd_id}"
                )
                return

            # Same idempotency guard as receive_snapshot_metadata.
            if spmd_id in snap_info["reported"]:
                logger.warning(
                    f"report_snapshot_failed: duplicate report from '{spmd_id}' "
                    f"for snapshot '{snapshot_id}' — ignoring"
                )
                return
            snap_info["reported"].add(spmd_id)

            logger.error(
                f"Snapshot '{snapshot_id}' failed for SPMD '{spmd_id}' — marking as error"
            )
            snap_info["errors"].append(spmd_id)
            snap_info["received"] += 1
            if snap_info["received"] == snap_info["expected"]:
                snap_info["event"].set()

        @require_spmd_manager
        async def restore_snapshot(
            self, snapshot_id: str, step: int, manifest: Dict[str, Any]
        ) -> None:
            """
            Restore a snapshot: each Driver loads its Workers' tensors from disk directly.

            Args:
                snapshot_id: Snapshot identifier for bookkeeping.
                step:        The logical step count of this snapshot.
                manifest:    {spmd_id: {str(rank): metadata_dict}} where metadata_dict
                             contains ``backend`` and backend-specific path fields.
            """
            logger.info(f"Controller restoring snapshot {snapshot_id} (step {step})")

            # Important: update the baseline *before* waking up workers,
            # so their `report_op_completed` pulses don't trigger a nested checkpoint.
            self._last_checkpointed_step = step

            self._state_manager.save_snapshot(snapshot_id, manifest)

            await self.block_all_spmds()

            try:
                restore_tasks = []
                for spmd_id, rank_metadata_map_str in manifest.items():
                    driver = self._spmd_manager.get_driver_handle(spmd_id)
                    if driver is None:
                        raise RuntimeError(
                            f"restore_snapshot: driver for spmd_id '{spmd_id}' not found. "
                            f"Cannot perform partial restore — aborting to prevent split-brain."
                        )
                    rank_metadata_map = {
                        int(k): v for k, v in rank_metadata_map_str.items()
                    }
                    restore_tasks.append(
                        driver.restore_from_disk.remote(rank_metadata_map)
                    )
                if restore_tasks:
                    await asyncio.gather(*restore_tasks)
            finally:
                await self.unblock_all_spmds()

            logger.info("Snapshot restored successfully")

        async def restore_checkpoint(self, step: Optional[int] = None) -> Optional[int]:
            """
            User-facing API to trigger checkpoint restoration.
            """
            if self._checkpointer is None:
                raise RuntimeError(
                    "Checkpoint config was not provided during controller initialization"
                )

            restored_step = await self._checkpointer.restore(step)
            return restored_step

        async def stop_checkpointer(self) -> None:
            """Await any in-flight checkpoint to completion before shutdown.

            Waiting (rather than cancelling) guarantees the last auto-checkpoint
            boundary is persisted.  do_ckpt() catches all exceptions internally,
            so this await never raises.  The upper wait bound is
            FLUXWAY_SNAPSHOT_TIMEOUT_SECS (default 600 s).
            """
            if self._checkpoint_task is not None and not self._checkpoint_task.done():
                await self._checkpoint_task
            self._checkpoint_task = None

        # =====================================================================
        # SPMD Error Handling
        # =====================================================================

        async def report_spmd_error(self, spmd_id: str, error: str) -> None:
            """
            Receive error report from a driver.

            Args:
                spmd_id: SPMD identifier
                error: Error message
            """
            logger.error(f"SPMD {spmd_id} reported error: {error}")
            await self._state_machine.trigger_event(
                ControllerEvent.DRIVER_ERROR, spmd_id=spmd_id, error=error
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
            for sid in self._spmd_manager.get_all_spmd_ids():
                heartbeat = await self._heartbeat_manager.get_driver_heartbeat(sid)
                if heartbeat is None:
                    all_completed = False
                    break

            if all_completed:
                await self._state_machine.trigger_event(
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
            for spmd_id in self._spmd_manager.get_all_spmd_ids():
                driver = self._spmd_manager.get_driver_handle(spmd_id)
                if driver is not None:
                    tasks.append(driver.block_all_ops.remote())

            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    logger.error(f"Error blocking SPMDs: {e}")

        @require_spmd_manager
        async def unblock_all_spmds(self) -> None:
            """Unblock all SPMD operations (called during recovery)."""
            logger.debug("Unblocking all SPMDs")

            tasks = []
            for spmd_id in self._spmd_manager.get_all_spmd_ids():
                driver = self._spmd_manager.get_driver_handle(spmd_id)
                if driver is not None:
                    tasks.append(driver.unblock_all_ops.remote())

            if tasks:
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    logger.error(f"Error unblocking SPMDs: {e}")

        # =====================================================================
        # SPMD Restart (for error recovery)
        # =====================================================================

        @require_spmd_manager
        async def restart_spmd(self, spmd_id: str) -> bool:
            """
            Restart an SPMD (for error recovery).

            Args:
                spmd_id: SPMD to restart

            Returns:
                True if restart successful

            Note: Checkpoint save/restore is a placeholder for now.
            """
            logger.warning(f"Restarting SPMD {spmd_id} (checkpoint not implemented)")

            # Get config from graph
            node = self._graph.nodes.get(spmd_id)
            if node is None:
                logger.error(f"Node {spmd_id} not found in graph")
                return False

            # Stop current SPMD
            await self._spmd_manager.stop_spmd(spmd_id, save_checkpoint=True)

            # Get PG info from scheduler
            pg_info = self._scheduler.pg_manager.get_pg_info(spmd_id)

            # Restart SPMD
            handle = self._spmd_manager.restore_spmd(
                spmd_id=spmd_id,
                config=node.spmd_config,
                placement_group=pg_info.get("pg") if pg_info else None,
                worker_bundle_indices=pg_info.get("worker_bundle_indices")
                if pg_info
                else None,
                checkpoint_id=None,  # TODO: implement checkpoint
            )

            return handle is not None

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
            await self._state_machine.trigger_event(ControllerEvent.ALL_DRIVERS_READY)

        async def _on_all_workers_running(self) -> None:
            """Callback when all drivers report workers_running=True.

            Triggers: TOPOLOGY_REGISTERING → RUNNING → start heartbeat monitoring
            """
            logger.debug("All workers are running after topology registration")
            await self._state_machine.trigger_event(
                ControllerEvent.ALL_SPMDS_TOPOLOGY_READY
            )

        async def _on_driver_error(self, spmd_id: str, error: str) -> None:
            """Callback when a driver has an error."""
            logger.error(f"Driver {spmd_id} error: {error}")
            await self._state_machine.trigger_event(
                ControllerEvent.DRIVER_ERROR, spmd_id=spmd_id, error=error
            )

        async def _on_driver_recovered(self, spmd_id: str) -> None:
            """Callback when a driver recovers."""
            logger.debug(f"Driver {spmd_id} recovered")

            # Check if all error drivers have recovered
            error_spmds = self._state_machine.get_error_spmds()
            if not error_spmds:
                await self._state_machine.trigger_event(
                    ControllerEvent.DRIVER_RECOVERED, spmd_id=spmd_id
                )

        # =====================================================================
        # Internal Methods for State Machine
        # =====================================================================

        @require_spmd_manager
        async def _initiate_topology_registration(self) -> None:
            """
            Initiate topology registration on all SPMDs.

            This method:
            1. Collects ALL worker handles from ALL drivers upfront to avoid actor deadlock.
            2. For each SPMD, calls register_topology with actual worker handles.
            """
            all_spmd_ids = self._spmd_manager.get_all_spmd_ids()
            if not all_spmd_ids:
                logger.warning("No SPMDs found, cannot initiate topology registration")
                return

            # Collect all worker handles upfront to avoid actor re-entrancy deadlocks during registration
            # {spmd_id: {rank: worker_handle}}
            all_worker_handles: Dict[str, Dict[int, Any]] = {}

            fetch_tasks = []
            spmd_ids_to_fetch = []

            for spmd_id in all_spmd_ids:
                driver = self._spmd_manager.get_driver_handle(spmd_id)
                if driver is not None:
                    fetch_tasks.append(driver.get_worker_handles.remote())
                    spmd_ids_to_fetch.append(spmd_id)

            if fetch_tasks:
                try:
                    results = await asyncio.gather(*fetch_tasks)
                    for spmd_id, worker_map in zip(  # type: ignore
                        spmd_ids_to_fetch, results, strict=True
                    ):
                        all_worker_handles[spmd_id] = worker_map
                    logger.debug(
                        f"Collected worker handles for: {list(all_worker_handles.keys())}"
                    )
                    for sid, handles in all_worker_handles.items():
                        logger.debug(f"  - {sid}: {len(handles)} workers")
                except Exception as e:
                    logger.error(f"Failed to collect worker handles for topology: {e}")
                    return

            # Initiate topology registration on each driver
            for spmd_id in all_spmd_ids:
                driver = self._spmd_manager.get_driver_handle(spmd_id)
                if driver is None:
                    continue

                predecessors = self._spmd_manager.get_predecessors(spmd_id)
                successors = self._spmd_manager.get_successors(spmd_id)

                logger.debug(
                    f"Driver {spmd_id}: predecessors={predecessors}, successors={successors}"
                )

                # Build predecessor/successor worker handle maps: {spmd_id: {rank: handle}}
                pred_handlers = {
                    p: all_worker_handles[p]
                    for p in predecessors
                    if p in all_worker_handles
                }
                succ_handlers = {
                    s: all_worker_handles[s]
                    for s in successors
                    if s in all_worker_handles
                }

                logger.debug(f"  - pred_handlers keys: {list(pred_handlers.keys())}")
                logger.debug(f"  - succ_handlers keys: {list(succ_handlers.keys())}")

                try:
                    # Now call register_topology with ACTUAL handles, bypassing the need for drivers to fetch them
                    driver.register_topology.remote(
                        pred_handlers, succ_handlers, len(predecessors), len(successors)
                    )
                    logger.debug(f"Initiated topology registration for SPMD {spmd_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to initiate topology registration for {spmd_id}: {e}"
                    )

        @require_spmd_manager
        @require_heartbeat_manager
        async def _cleanup_all_spmds(self) -> None:
            """
            Cleanup all SPMD resources.

            Called when entering COMPLETED or FAILED state.
            """
            logger.debug("Cleaning up all SPMDs")

            # Stop heartbeat monitoring
            await self._heartbeat_manager.stop_monitoring_async()

            # Stop all SPMDs
            for spmd_id in self._spmd_manager.get_all_spmd_ids():
                try:
                    await self._spmd_manager.stop_spmd(spmd_id, save_checkpoint=False)
                except Exception as e:
                    logger.error(f"Error stopping SPMD {spmd_id}: {e}")

            # Release all placement groups
            for spmd_id in list(self._scheduler.pg_manager.get_all_spmd_ids()):
                try:
                    self._scheduler.release_resources(spmd_id)
                except Exception as e:
                    logger.error(f"Error releasing resources for {spmd_id}: {e}")

        # =====================================================================
        # Lifecycle Control
        # =====================================================================

        async def stop(self, save_checkpoint: bool = True) -> None:
            """
            Stop training execution.

            Args:
                save_checkpoint: Whether to save checkpoint before stopping
            """
            logger.debug("Stopping controller")

            # Block all SPMDs first
            await self.block_all_spmds()

            # Save checkpoint if requested (placeholder)
            if save_checkpoint:
                logger.warning("Checkpoint save not implemented")

            # Cleanup
            await self._cleanup_all_spmds()

            logger.debug("Controller stopped")

        @require_heartbeat_manager
        async def get_cluster_status(self) -> Dict[str, str]:
            """
            Get status of all SPMD nodes.

            Returns:
                Mapping of node name to status
            """
            status = await self._heartbeat_manager.get_all_driver_status()
            return {spmd_id: s.value for spmd_id, s in status.items()}

        async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
            """
            Wait for training to complete.

            Args:
                timeout: Maximum time to wait (None for infinite)

            Returns:
                True if completed successfully, False if failed or timeout
            """
            start_time = time.time()

            while True:
                state = self._state_machine.state

                if state == ControllerState.COMPLETED:
                    return True

                if state == ControllerState.FAILED:
                    return False

                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        logger.warning("Timeout waiting for completion")
                        return False

                await asyncio.sleep(1.0)


async def create_controller(
    graph: Graph,
    heartbeat_config: Optional[HeartbeatConfig] = None,
    async_step_config: Optional[AsyncStepConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None,
    runtime_env: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create and initialize a Controller actor.

    This is the main entry point for creating a Controller.

    Args:
        graph: The task graph built using Builder
        heartbeat_config: Optional heartbeat configuration
        async_step_config: Optional async step configuration
        checkpoint_config: Optional checkpoint configuration
        runtime_env: Optional Ray runtime environment

    Returns:
        Controller actor handle

    Raises:
        RuntimeError: If Ray is not available or initialization fails
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not available")

    # Create the actor
    controller = ControllerActor.options(runtime_env=runtime_env).remote(
        graph=graph,
        heartbeat_config=heartbeat_config,
        async_step_config=async_step_config,
        checkpoint_config=checkpoint_config,
    )

    # Initialize the actor
    success = await controller.initialize.remote()
    if not success:
        raise RuntimeError("Failed to initialize controller")

    return controller


# Alias for backwards compatibility
if RAY_AVAILABLE:
    Controller = ControllerActor
else:
    # Mock for when Ray is not available
    class Controller:  # type: ignore
        """Mock Controller when Ray is not available."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError("Ray is not available")
