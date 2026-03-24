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

            # Initialize components (some are initialized in initialize() now)
            self._monitor = Monitor()
            self._state_manager = StateManager()
            self._scheduler = Scheduler(
                spmd_manager=None
            )  # SPMDManager set in initialize()

            # Initialize state machine
            self._state_machine = ControllerStateMachine(self)

            # Initialize async step manager
            self._async_step_manager = AsyncStepManager(graph, self._async_step_config)

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
                    if self._checkpoint_config.backend == "orbax":
                        from fluxway.control_plane.checkpointer.orbax import (
                            OrbaxCheckpointer,
                        )

                        self._checkpointer = OrbaxCheckpointer(
                            self,
                            ckpt_dir=self._checkpoint_config.ckpt_dir,
                            interval_steps=self._checkpoint_config.interval_steps,
                        )
                    elif self._checkpoint_config.backend == "safetensors":
                        from fluxway.control_plane.checkpointer.safetensors import (
                            SafeTensorCheckpointer,
                        )

                        self._checkpointer = SafeTensorCheckpointer(
                            self,
                            ckpt_dir=self._checkpoint_config.ckpt_dir,
                            interval_steps=self._checkpoint_config.interval_steps,
                        )
                    else:
                        raise ValueError(
                            f"Unknown checkpoint backend: {self._checkpoint_config.backend}"
                        )

                    self._checkpointer.start()

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

        async def get_all_heartbeats(self) -> Dict[str, Any]:
            """Get all latest heartbeats from all SPMDs."""
            if self._heartbeat_manager is None:
                return {}
            return await self._heartbeat_manager.get_all_heartbeats()

        def is_initialized(self) -> bool:
            """Check if controller is initialized."""
            return self._initialized

        # =====================================================================
        # SPMD Handle Access
        # =====================================================================

        def get_spmd_handle(self, spmd_id: str) -> Optional[SPMDHandle]:
            """
            Get handle to an SPMD instance from spmd_manager.

            Args:
                spmd_id: ID of the SPMD

            Returns:
                SPMDHandle or None if not found
            """
            if self._spmd_manager is None:
                return None
            return self._spmd_manager.get_spmd_handle(spmd_id)

        # =====================================================================
        # Heartbeat Interface (called by drivers)
        # =====================================================================

        async def receive_driver_heartbeat(self, heartbeat: DriverHeartbeat) -> None:
            """
            Receive heartbeat from a driver.

            Args:
                heartbeat: Driver heartbeat data
            """
            if self._heartbeat_manager is None:
                logger.warning("Heartbeat manager not initialized, dropping heartbeat.")
                return

            # Route to heartbeat manager
            await self._heartbeat_manager.receive_heartbeat(heartbeat)

            # Update async step manager with op steps
            if heartbeat.op_steps:
                self._async_step_manager.report_op_steps(
                    heartbeat.spmd_id, heartbeat.op_steps
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

        async def report_op_completed(
            self, spmd_id: str, op_name: str, step: int
        ) -> None:
            """
            Receive op completion report from a driver.

            Args:
                spmd_id: SPMD identifier
                op_name: Operation name
                step: Completed step number
            """
            self._async_step_manager.report_op_completed(spmd_id, op_name, step)
            await self._check_and_enforce_step_limits()

        async def _check_and_enforce_step_limits(self) -> None:
            """Check step differences and apply lock/unlock actions."""
            if self._async_step_manager.is_all_completed():
                logger.debug("All operators reached max_steps, triggering completion")
                await self._state_machine.trigger_event(
                    ControllerEvent.ALL_DRIVERS_COMPLETED
                )
                return

            # Enforce step difference limits via global watermarks
            for (
                spmd_id,
                op_name,
                allowed_step,
            ) in self._async_step_manager.check_and_enforce_limits():
                await self._set_spmd_allowed_step(spmd_id, op_name, allowed_step)

        async def _set_spmd_allowed_step(
            self, spmd_id: str, op_name: str, allowed_step: int
        ) -> None:
            """Propagate global watermark to an SPMD Driver."""
            if self._spmd_manager is None:
                logger.error(
                    f"SPMD manager not initialized, cannot set allowed_step for {spmd_id}.{op_name}"
                )
                return
            driver = self._spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                try:
                    await driver.set_allowed_step.remote(op_name, allowed_step)
                except Exception as e:
                    logger.error(
                        f"Failed to set allowed_step {allowed_step} for {spmd_id}.{op_name}: {e}"
                    )

        async def _lock_spmd_op(self, spmd_id: str, op_name: str) -> None:
            """Lock an SPMD's operation."""
            if self._spmd_manager is None:
                logger.error(
                    f"SPMD manager not initialized, cannot lock {spmd_id}.{op_name}"
                )
                return
            driver = self._spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                try:
                    await driver.block_op.remote(op_name)
                    # logger.debug(f"Locked {spmd_id} {op_name}")
                except Exception as e:
                    logger.error(f"Failed to lock {spmd_id} {op_name}: {e}")

        async def _unlock_spmd_op(self, spmd_id: str, op_name: str) -> None:
            """Unlock an SPMD's operation."""
            if self._spmd_manager is None:
                logger.error(
                    f"SPMD manager not initialized, cannot unlock {spmd_id}.{op_name}"
                )
                return
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

        async def _check_and_enforce_backpressure(self) -> None:
            """
            Check downstream SPMD busy status and apply backpressure to readers.

            Logic:
            - For each Reader node, check if any of its downstream SPMDs are busy
            - If downstream is busy (busy_worker_ratio > threshold), pause the reader
            - If downstream is no longer busy, resume the reader
            """
            if self._spmd_manager is None:
                logger.error(
                    "SPMD manager not initialized, cannot enforce backpressure."
                )
                return

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

        async def _pause_reader(self, reader_id: str) -> None:
            """
            Pause a reader due to backpressure.

            Args:
                reader_id: Reader SPMD ID to pause
            """
            if self._spmd_manager is None:
                logger.error(
                    f"SPMD manager not initialized, cannot pause reader {reader_id}."
                )
                return
            driver = self._spmd_manager.get_driver_handle(reader_id)
            if driver is not None:
                try:
                    await driver.pause_sources.remote()
                    self._paused_readers.add(reader_id)
                    logger.debug(f"Paused reader {reader_id} due to backpressure")
                except Exception as e:
                    logger.error(f"Failed to pause reader {reader_id}: {e}")

        async def _resume_reader(self, reader_id: str) -> None:
            """
            Resume a paused reader.

            Args:
                reader_id: Reader SPMD ID to resume
            """
            if self._spmd_manager is None:
                logger.error(
                    f"SPMD manager not initialized, cannot resume reader {reader_id}."
                )
                return
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
            if self._spmd_manager is None:
                logger.error(
                    f"SPMD manager not initialized, cannot receive control argument for {target_spmd_id}."
                )
                return
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

        async def request_snapshot(self, snapshot_id: str, target_step: int) -> None:
            """
            Trigger a global snapshot across all SPMDs.
            """
            logger.info(
                f"Controller requesting global snapshot {snapshot_id} at step {target_step}"
            )
            self._pending_snapshots = getattr(self, "_pending_snapshots", {})
            if self._spmd_manager is None:
                logger.error("SPMD manager not initialized, cannot request snapshot.")
                return
            all_spmd_ids = self._spmd_manager.get_all_spmd_ids()
            self._pending_snapshots[snapshot_id] = {
                "expected": len(all_spmd_ids),
                "received": 0,
                "states": {},
            }

            for spmd_id in all_spmd_ids:
                driver = self._spmd_manager.get_driver_handle(spmd_id)
                if driver is not None:
                    driver.request_snapshot.remote(snapshot_id, target_step)

        async def receive_snapshot_states(
            self, spmd_id: str, snapshot_id: str, futures: Dict[int, Any]
        ) -> None:
            """
            Receive snapshot states from a driver.
            """
            logger.info(
                f"Controller received snapshot {snapshot_id} states from SPMD {spmd_id}"
            )
            self._pending_snapshots = getattr(self, "_pending_snapshots", {})
            if snapshot_id not in self._pending_snapshots:
                logger.warning(
                    f"Received unknown snapshot {snapshot_id} from {spmd_id}"
                )
                return

            try:
                resolved_states = {}
                ranks = list(futures.keys())
                refs = list(futures.values())
                results = await asyncio.gather(
                    *[
                        asyncio.wrap_future(r.future()) if hasattr(r, "future") else r
                        for r in refs
                    ]
                )
                for rank, res in zip(ranks, results, strict=True):
                    resolved_states[rank] = res
            except Exception as e:
                logger.error(f"Failed to resolve snapshot futures for {spmd_id}: {e}")
                return

            snap_info = self._pending_snapshots[snapshot_id]
            snap_info["states"][spmd_id] = resolved_states
            snap_info["received"] += 1

            if snap_info["received"] == snap_info["expected"]:
                logger.info(
                    f"Global snapshot {snapshot_id} complete. Aggregating to StateManager."
                )
                self._state_manager.save_snapshot(snapshot_id, snap_info["states"])

                # Resume all drivers
                if self._spmd_manager is not None:
                    for s_id in self._spmd_manager.get_all_spmd_ids():
                        driver = self._spmd_manager.get_driver_handle(s_id)
                        if driver is not None:
                            driver.resume.remote()

        async def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
            """Retrieve an aggregated snapshot, or None if not yet ready."""
            try:
                return self._state_manager.load_snapshot(snapshot_id)
            except KeyError:
                return None

        async def restore_snapshot(
            self, snapshot_id: str, states: Dict[str, Any]
        ) -> None:
            """
            Restore a snapshot by interacting with drivers directly.
            """
            logger.info(f"Controller restoring snapshot {snapshot_id}")
            self._state_manager.save_snapshot(snapshot_id, states)

            # Pause everything
            await self.block_all_spmds()

            # Send state to workers
            if self._spmd_manager is not None:
                for spmd_id, rank_states in states.items():
                    driver = self._spmd_manager.get_driver_handle(spmd_id)
                    if driver is None:
                        continue
                    worker_handles = await driver.get_worker_handles.remote()
                    tasks = []
                    for rank, w_state in rank_states.items():
                        rank_int = int(rank)
                        if rank_int in worker_handles:
                            tasks.append(
                                worker_handles[rank_int].set_state.remote(w_state)
                            )
                    if tasks:
                        await asyncio.gather(
                            *[
                                asyncio.wrap_future(t.future())
                                if hasattr(t, "future")
                                else t
                                for t in tasks
                            ]
                        )

            # Unpause
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
            return await self._checkpointer.restore(step)

        async def stop_checkpointer(self) -> None:
            """Gracefully stop background checkpointer loop."""
            if self._checkpointer is not None:
                await self._checkpointer.wait_until_finished()

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
                    await asyncio.gather(
                        *[asyncio.wrap_future(t.future()) for t in tasks],
                        return_exceptions=True,
                    )
                except Exception as e:
                    logger.error(f"Error blocking SPMDs: {e}")

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
                    await asyncio.gather(
                        *[asyncio.wrap_future(t.future()) for t in tasks],
                        return_exceptions=True,
                    )
                except Exception as e:
                    logger.error(f"Error unblocking SPMDs: {e}")

        # =====================================================================
        # SPMD Restart (for error recovery)
        # =====================================================================

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
                    results = await asyncio.gather(
                        *[asyncio.wrap_future(t.future()) for t in fetch_tasks]
                    )
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
