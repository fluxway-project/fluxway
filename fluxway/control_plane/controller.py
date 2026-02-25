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
import logging
import time
from typing import Any, Dict, List, Optional

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None  # type: ignore

from ..common.types import (
    AsyncStepConfig,
    ControllerEvent,
    ControllerState,
    DriverHeartbeat,
    Graph,
    HeartbeatConfig,
    OpType,
    NodeType,
)
from ..common.utils import ensure_logging_configured
from .monitor import Monitor
from .scheduler import Scheduler
from .spmd_manager import SPMDManager, SPMDHandle
from .state_manager import StateManager
from .heartbeat import HeartbeatManager
from .state_machine import ControllerStateMachine
from .async_manager import AsyncStepManager

logger = logging.getLogger(__name__)


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
        ):
            """
            Initialize Controller with configuration.

            Args:
                graph: The task graph defining SPMD nodes and edges
                heartbeat_config: Heartbeat configuration
                async_step_config: Async step configuration
            """
            # Ray actors run in separate worker processes; ensure INFO logs are not dropped.
            ensure_logging_configured()

            self._graph = graph
            self._heartbeat_config = heartbeat_config or HeartbeatConfig()
            self._async_step_config = async_step_config or AsyncStepConfig()

            # Initialize components
            self._monitor = Monitor()
            self._state_manager = StateManager()
            self._spmd_manager = SPMDManager(controller_handle=None)
            self._scheduler = Scheduler(spmd_manager=self._spmd_manager)

            # Initialize heartbeat manager with async callbacks
            self._heartbeat_manager = HeartbeatManager(
                config=self._heartbeat_config,
                on_all_ready=self._on_all_drivers_ready,
                on_all_workers_running=self._on_all_workers_running,
                on_driver_error=self._on_driver_error,
                on_driver_recovered=self._on_driver_recovered,
            )

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

            # Backpressure tracking
            self._busy_spmds: Dict[str, float] = {}  # {spmd_id: busy_worker_ratio}
            self._backpressure_threshold: float = (
                0.5  # 50% workers busy triggers backpressure
            )
            self._paused_readers: set = set()  # Set of paused reader IDs

            logger.info(f"ControllerActor created with {len(graph.nodes)} nodes")

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
                # Get self actor handle
                self_handle = ray.runtime_context.get_runtime_context().current_actor
                self._spmd_manager.set_controller_handle(self_handle)

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
                logger.info(
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
            # Forward to heartbeat manager
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
            self, spmd_id: str, op_type: OpType, step: int
        ) -> None:
            """
            Receive op completion report from a driver.

            Args:
                spmd_id: SPMD identifier
                op_type: Operation type
                step: Completed step number
            """
            self._async_step_manager.report_op_completed(spmd_id, op_type, step)
            await self._check_and_enforce_step_limits()

        async def _check_and_enforce_step_limits(self) -> None:
            """Check step differences and apply lock/unlock actions."""
            actions = self._async_step_manager.check_and_enforce_limits()

            for spmd_id, op_type, is_lock in actions:
                if is_lock:
                    await self._lock_spmd_op(spmd_id, op_type)
                else:
                    await self._unlock_spmd_op(spmd_id, op_type)

        async def _lock_spmd_op(self, spmd_id: str, op_type: OpType) -> None:
            """Lock an SPMD's operation."""
            driver = self._spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                try:
                    await driver.block_op.remote(op_type)
                    logger.info(f"Locked {spmd_id} {op_type.name}")
                except Exception as e:
                    logger.error(f"Failed to lock {spmd_id} {op_type.name}: {e}")

        async def _unlock_spmd_op(self, spmd_id: str, op_type: OpType) -> None:
            """Unlock an SPMD's operation."""
            driver = self._spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                try:
                    await driver.unblock_op.remote(op_type)
                    logger.info(f"Unlocked {spmd_id} {op_type.name}")
                except Exception as e:
                    logger.error(f"Failed to unlock {spmd_id} {op_type.name}: {e}")

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
            # Get all reader nodes
            reader_nodes = [
                node_name
                for node_name, node in self._graph.nodes.items()
                if node.node_type == NodeType.READER
            ]

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
            driver = self._spmd_manager.get_driver_handle(reader_id)
            if driver is not None:
                try:
                    await driver.pause_reading.remote()
                    self._paused_readers.add(reader_id)
                    logger.info(f"Paused reader {reader_id} due to backpressure")
                except Exception as e:
                    logger.error(f"Failed to pause reader {reader_id}: {e}")

        async def _resume_reader(self, reader_id: str) -> None:
            """
            Resume a paused reader.

            Args:
                reader_id: Reader SPMD ID to resume
            """
            driver = self._spmd_manager.get_driver_handle(reader_id)
            if driver is not None:
                try:
                    await driver.resume_reading.remote()
                    self._paused_readers.discard(reader_id)
                    logger.info(f"Resumed reader {reader_id}")
                except Exception as e:
                    logger.error(f"Failed to resume reader {reader_id}: {e}")

        def get_paused_readers(self) -> List[str]:
            """Get list of currently paused readers."""
            return list(self._paused_readers)

        # =====================================================================
        # State Argument Transport
        # =====================================================================

        async def receive_state_argument(
            self,
            source_spmd_id: str,
            target_spmd_id: str,
            target_op: "OpType",
            names: List[str],
            step: int,
            values: List[Any],
        ) -> None:
            """
            Receive state_arguments and forward to target SPMD.

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
                    f"Target SPMD {target_spmd_id} not found for state_arguments"
                )
                return

            try:
                await target_driver.broadcast_state_argument.remote(
                    target_op, names, step, values
                )
                logger.debug(
                    f"Forwarded state_arguments {names} from {source_spmd_id} "
                    f"to {target_spmd_id}, target_op={target_op}, step={step}"
                )
            except Exception as e:
                logger.error(f"Failed to forward state_arguments: {e}")

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
            logger.info(f"SPMD {spmd_id} reported completion")

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
            logger.info("Unblocking all SPMDs")

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
            logger.info(
                "All drivers are ready (workers_ready=True), triggering topology registration"
            )
            await self._state_machine.trigger_event(ControllerEvent.ALL_DRIVERS_READY)

        async def _on_all_workers_running(self) -> None:
            """Callback when all drivers report workers_running=True.

            Triggers: TOPOLOGY_REGISTERING → RUNNING → start heartbeat monitoring
            """
            logger.info("All workers are running after topology registration")
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
            logger.info(f"Driver {spmd_id} recovered")

            # Check if all error drivers have recovered
            error_spmds = self._state_machine.get_error_spmds()
            if not error_spmds:
                await self._state_machine.trigger_event(
                    ControllerEvent.DRIVER_RECOVERED, spmd_id=spmd_id
                )

        # =====================================================================
        # Internal Methods for State Machine
        # =====================================================================

        def _initiate_topology_registration(self) -> None:
            """
            Initiate topology registration on all SPMDs (fire-and-forget).

            This method:
            1. Collects driver handles (not worker handles)
            2. For each SPMD, calls register_topology with driver handles

            All calls are async fire-and-forget. Workers will auto-start after
            handler registration. State changes are observed via heartbeat.
            """
            all_spmd_ids = self._spmd_manager.get_all_spmd_ids()
            if not all_spmd_ids:
                logger.warning("No SPMDs found, cannot initiate topology registration")
                return

            # Collect all driver handles: {spmd_id: driver_handle}
            all_driver_handles: Dict[str, Any] = {}
            for spmd_id in all_spmd_ids:
                driver = self._spmd_manager.get_driver_handle(spmd_id)
                if driver is not None:
                    all_driver_handles[spmd_id] = driver

            # For each SPMD, initiate topology registration (fire-and-forget)
            for spmd_id in all_spmd_ids:
                driver = self._spmd_manager.get_driver_handle(spmd_id)
                if driver is None:
                    continue

                predecessors = self._spmd_manager.get_predecessors(spmd_id)
                successors = self._spmd_manager.get_successors(spmd_id)

                # Build predecessor/successor driver handles
                pred_drivers = {
                    p: all_driver_handles[p]
                    for p in predecessors
                    if p in all_driver_handles
                }
                succ_drivers = {
                    s: all_driver_handles[s]
                    for s in successors
                    if s in all_driver_handles
                }

                try:
                    # Fire-and-forget: don't wait for return
                    driver.register_topology.remote(
                        pred_drivers, succ_drivers, len(predecessors), len(successors)
                    )
                    logger.info(f"Initiated topology registration for SPMD {spmd_id}")
                except Exception as e:
                    logger.error(
                        f"Failed to initiate topology registration for {spmd_id}: {e}"
                    )

        async def _cleanup_all_spmds(self) -> None:
            """
            Cleanup all SPMD resources.

            Called when entering COMPLETED or FAILED state.
            """
            logger.info("Cleaning up all SPMDs")

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
            logger.info("Stopping controller")

            # Block all SPMDs first
            await self.block_all_spmds()

            # Save checkpoint if requested (placeholder)
            if save_checkpoint:
                logger.warning("Checkpoint save not implemented")

            # Cleanup
            await self._cleanup_all_spmds()

            logger.info("Controller stopped")

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
) -> Any:
    """
    Create and initialize a Controller actor.

    This is the main entry point for creating a Controller.

    Args:
        graph: The task graph built using Builder
        heartbeat_config: Optional heartbeat configuration
        async_step_config: Optional async step configuration

    Returns:
        Controller actor handle

    Raises:
        RuntimeError: If Ray is not available or initialization fails
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray is not available")

    # Create the actor
    controller = ControllerActor.remote(
        graph=graph,
        heartbeat_config=heartbeat_config,
        async_step_config=async_step_config,
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
