"""
SPMD Driver: Master controller for an SPMD instance.

Based on ARCHITECTURE.md Section 3.1

The SPMD Driver coordinates all workers within the SPMD:
- Starts and manages worker actors using placement group
- Handles worker heartbeats and monitors health
- Synchronizes worker steps within the SPMD
- Reports SPMD-level progress to Controller
- Handles state_argument broadcasting
- Manages op-level blocking
"""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from ...common.types import (
    DriverEvent,
    DriverHeartbeat,
    DriverState,
    DriverStatus,
    HeartbeatConfig,
    SPMDConfig,
    WorkerHeartbeat,
    NodeType,
    ReaderConfig,
    ReaderState,
)
from ...common.utils import ensure_logging_configured
from .state_machine import DriverStateMachine
from .synchronizer import SPMDSynchronizer
from .synchronizer.worker_sync import WorkerSynchronizer
from .stateful_executor import StatefulExecutor
from .monitor import SPMDMonitor
from .worker_registrar import WorkerRegistrar
from .function_signatures import FunctionSignatures
from .partition_manager import PartitionManager

logger = logging.getLogger(__name__)


if RAY_AVAILABLE:

    @ray.remote
    class SPMDDriver:
        """
        SPMD Driver (Ray Actor).

        Coordinates workers within a single SPMD instance.
        Started by SPMDManager, manages workers using placement group.
        """

        def __init__(
            self,
            spmd_id: str,
            config: SPMDConfig,
            placement_group: Optional[Any],
            worker_bundle_indices: List[int],
            controller_handle: Optional[Any],
            graph_topology: Dict[str, Any],
            node_type: NodeType = NodeType.SPMD,
            reader_config: Optional[ReaderConfig] = None,
        ):
            """
            Initialize SPMD Driver.

            Args:
                spmd_id: SPMD identifier
                config: SPMD configuration
                placement_group: Ray PlacementGroup for worker allocation
                worker_bundle_indices: Bundle indices for workers [1, 2, ..., n]
                controller_handle: Reference to Controller (for callbacks)
                graph_topology: Dict with 'predecessors' and 'successors' lists
                node_type: Type of node (SPMD or READER)
                reader_config: Reader configuration (required for READER nodes)
            """
            # Ray actors run in separate worker processes; ensure INFO logs are not dropped.
            ensure_logging_configured()

            self.spmd_id = spmd_id
            self.config = config
            self.placement_group = placement_group
            self.worker_bundle_indices = worker_bundle_indices
            self.controller_handle = controller_handle
            self.graph_topology = graph_topology
            self.node_type = node_type
            self.reader_config = reader_config

            # State machine
            self.state_machine = DriverStateMachine(self)

            # Worker synchronizer
            self.worker_synchronizer = WorkerSynchronizer(
                spmd_id=spmd_id,
                worker_count=config.worker_count,
                max_worker_step_diff=config.max_async_steps,
            )

            # Existing components
            self.synchronizer = SPMDSynchronizer(spmd_id)
            self.executor = StatefulExecutor(spmd_id)
            self.monitor = SPMDMonitor(spmd_id)
            self.registrar = WorkerRegistrar(spmd_id)
            self.function_signatures = FunctionSignatures()
            self.partition_manager = PartitionManager()

            # Worker handles
            self._worker_handles: Dict[int, Any] = {}

            # Heartbeat configuration
            self._heartbeat_config = HeartbeatConfig()
            self._heartbeat_task: Optional[asyncio.Task] = None
            self._heartbeat_running = False

            # Worker heartbeat tracking
            self._worker_heartbeats: Dict[int, WorkerHeartbeat] = {}
            self._worker_heartbeat_times: Dict[int, float] = {}

            # Recovery tracking
            self._recovery_attempts: Dict[int, int] = {}
            self._max_recovery_attempts = 3

            # Workers ready/running state tracking
            self._workers_ready: bool = False
            self._workers_running: bool = False

            # Init timeout (seconds) — if workers don't report within this time,
            # INIT_FAILED is triggered
            self._init_timeout: float = 120.0
            self._init_start_time: float = 0.0

            # Reader-specific state
            self._reader_state: ReaderState = ReaderState.INITIALIZING
            self._is_paused: bool = False  # Backpressure pause flag
            self._file_tasks: List[Any] = []  # File read tasks for Reader mode
            self._busy_worker_ratio: float = 0.0  # Ratio of workers with full mailbox

            node_type_str = "Reader" if node_type == NodeType.READER else "SPMD"
            logger.info(
                f"[spmd_id={spmd_id}] {node_type_str}Driver initialized: spmd_id={spmd_id}, "
                f"worker_count={config.worker_count}"
            )

        # =====================================================================
        # Initialization
        # =====================================================================

        def initialize(self, checkpoint_id: Optional[str] = None) -> bool:
            """
            Initialize the driver and start all workers.

            Event-driven design:
            - Workers are started but INIT_COMPLETE is NOT triggered here.
            - INIT_COMPLETE is triggered when all workers report heartbeat
              (detected in receive_worker_heartbeat).
            - Heartbeat to controller starts immediately so controller
              can track this driver's state.
            - An init timeout is scheduled: if workers don't report within
              the timeout, INIT_FAILED is triggered.

            Args:
                checkpoint_id: Optional checkpoint to restore from

            Returns:
                True if workers were started successfully
            """
            try:
                node_type_str = (
                    "Reader" if self.node_type == NodeType.READER else "SPMD"
                )
                logger.info(
                    f"[spmd_id={self.spmd_id}] Initializing {node_type_str} {self.spmd_id}"
                )

                # Set up worker synchronizer callbacks
                self.worker_synchronizer.set_callbacks(
                    on_spmd_op_completed=self._on_spmd_op_completed,
                    block_worker_op=self._block_worker_op,
                    unblock_worker_op=self._unblock_worker_op,
                )

                # Reader-specific initialization
                if self.node_type == NodeType.READER:
                    self._initialize_reader()

                # Start all worker actors
                self._start_workers()

                # Restore from checkpoint if provided
                if checkpoint_id is not None:
                    self._restore_from_checkpoint(checkpoint_id)

                # Record init start time for timeout detection
                self._init_start_time = time.time()

                # Start heartbeat to controller immediately (state=INITIALIZING)
                # so controller can see this driver exists
                self._start_heartbeat()

                # NOTE: INIT_COMPLETE is NOT triggered here.
                # It will be triggered in receive_worker_heartbeat() when
                # all workers have reported their first heartbeat.

                logger.info(
                    f"[spmd_id={self.spmd_id}] {node_type_str} {self.spmd_id} workers started, "
                    f"waiting for worker heartbeats to confirm readiness"
                )
                return True

            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id}] SPMD {self.spmd_id} initialization failed: {e}"
                )
                self.state_machine.trigger(DriverEvent.INIT_FAILED, error=str(e))
                return False

        def _initialize_reader(self) -> None:
            """Initialize Reader mode by discovering files and creating read tasks."""
            if self.reader_config is None:
                raise ValueError("ReaderConfig is required for Reader node")

            from ...reader.datasource import create_datasource

            # Create datasource and get file tasks
            datasource = create_datasource(
                data_type=self.reader_config.data_type,
                src_dir=self.reader_config.src_dir,
                file_pattern=self.reader_config.file_pattern,
                encoding=self.reader_config.encoding,
            )

            # Get file read tasks (partitioned by worker count)
            self._file_tasks = datasource.get_read_tasks(
                parallelism=self.reader_config.worker_count
            )

            logger.info(
                f"[spmd_id={self.spmd_id}] Reader {self.spmd_id} initialized with {len(self._file_tasks)} file tasks"
            )

            self._reader_state = ReaderState.READING

        def _start_workers(self) -> None:
            """Start all worker actors."""
            from ..worker import SPMDWorker

            # Get self handle for registering with workers
            self_handle = ray.get_runtime_context().current_actor

            # Determine worker name prefix
            worker_prefix = (
                "reader_worker" if self.node_type == NodeType.READER else "spmd_worker"
            )

            for rank, bundle_idx in enumerate(self.worker_bundle_indices):
                # Prepare reader task for this worker (if Reader mode)
                reader_task = None
                if self.node_type == NodeType.READER and self._file_tasks:
                    if rank < len(self._file_tasks):
                        reader_task = self._file_tasks[rank]

                if self.placement_group is not None:
                    worker = SPMDWorker.options(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=bundle_idx,
                        name=f"{worker_prefix}_{self.spmd_id}_{rank}",
                    ).remote(
                        config=self.config,
                        rank=rank,
                        spmd_id=self.spmd_id,
                        driver_handle=self_handle,
                        node_type=self.node_type,
                        reader_task=reader_task,
                        reader_config=self.reader_config,
                    )
                else:
                    worker = SPMDWorker.remote(
                        config=self.config,
                        rank=rank,
                        spmd_id=self.spmd_id,
                        driver_handle=self_handle,
                        node_type=self.node_type,
                        reader_task=reader_task,
                        reader_config=self.reader_config,
                    )

                self._worker_handles[rank] = worker
                self._recovery_attempts[rank] = 0
                logger.debug(
                    f"[spmd_id={self.spmd_id}] Started worker {rank} for SPMD {self.spmd_id}"
                )

            node_type_str = "Reader" if self.node_type == NodeType.READER else "SPMD"
            logger.info(
                f"[spmd_id={self.spmd_id}] Started {len(self._worker_handles)} workers for {node_type_str} {self.spmd_id}"
            )

        def get_worker_handles(self) -> Dict[int, Any]:
            """Get all worker handles for topology registration."""
            return dict(self._worker_handles)

        async def register_topology(
            self,
            predecessor_drivers: Dict[str, Any],  # {spmd_id: driver_handle}
            successor_drivers: Dict[str, Any],  # {spmd_id: driver_handle}
            predecessor_count: int,
            successor_count: int,
        ) -> None:
            """
            Register topology by fetching worker handles from predecessor/successor drivers.

            Called by Controller when all SPMDs are ready.
            This is a fire-and-forget call - workers will auto-start after registration.

            Args:
                predecessor_drivers: {spmd_id: driver_handle} for predecessors
                successor_drivers: {spmd_id: driver_handle} for successors
                predecessor_count: expected predecessor count for workers
                successor_count: expected successor count for workers
            """
            logger.info(
                f"[spmd_id={self.spmd_id}] SPMD {self.spmd_id}: Starting topology registration with "
                f"{predecessor_count} predecessors, {successor_count} successors"
            )

            # Step 1: Notify workers of expected topology counts
            for worker in self._worker_handles.values():
                worker.set_expected_topology.remote(predecessor_count, successor_count)

            # Step 2: Fetch worker handles from predecessor drivers and register
            for spmd_id, driver_handle in predecessor_drivers.items():
                try:
                    # Fetch worker handles from the predecessor driver
                    worker_handles = await driver_handle.get_worker_handles.remote()
                    # Register to all local workers
                    for worker in self._worker_handles.values():
                        worker.register_predecessor_handlers.remote(
                            spmd_id, worker_handles
                        )
                    logger.debug(
                        f"[spmd_id={self.spmd_id}] Registered predecessor {spmd_id} with {len(worker_handles)} workers"
                    )
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to register predecessor {spmd_id}: {e}"
                    )

            # Step 3: Fetch worker handles from successor drivers and register
            for spmd_id, driver_handle in successor_drivers.items():
                try:
                    # Fetch worker handles from the successor driver
                    worker_handles = await driver_handle.get_worker_handles.remote()
                    # Register to all local workers
                    for worker in self._worker_handles.values():
                        worker.register_successor_handlers.remote(
                            spmd_id, worker_handles
                        )
                    logger.debug(
                        f"[spmd_id={self.spmd_id}] Registered successor {spmd_id} with {len(worker_handles)} workers"
                    )
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to register successor {spmd_id}: {e}"
                    )

            logger.info(
                f"[spmd_id={self.spmd_id}] SPMD {self.spmd_id}: Topology registration completed"
            )

        def _restore_from_checkpoint(self, checkpoint_id: str) -> None:
            """Restore state from checkpoint (placeholder)."""
            logger.warning(
                f"[spmd_id={self.spmd_id}] Checkpoint restore not implemented: {checkpoint_id}"
            )

        # =====================================================================
        # Lifecycle Control
        # =====================================================================

        def stop(self, save_checkpoint: bool = True) -> None:
            """Stop execution."""
            logger.info(f"[spmd_id={self.spmd_id}] Stopping SPMD {self.spmd_id}")

            # Stop heartbeat
            self._stop_heartbeat()

            # Save checkpoint if requested
            if save_checkpoint:
                self._save_checkpoint()

            # Stop all workers
            for rank, worker in self._worker_handles.items():
                try:
                    worker.stop.remote()
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to stop worker {rank}: {e}"
                    )

        def _save_checkpoint(self) -> None:
            """Save checkpoint (placeholder)."""
            logger.warning(
                f"[spmd_id={self.spmd_id}] Checkpoint save not implemented for SPMD {self.spmd_id}"
            )

        # =====================================================================
        # Worker Heartbeat Handling
        # =====================================================================

        def receive_worker_heartbeat(
            self, rank: int, heartbeat: WorkerHeartbeat
        ) -> None:
            """
            Receive heartbeat from a worker.

            Args:
                rank: Worker rank
                heartbeat: Worker heartbeat data
            """
            self._worker_heartbeats[rank] = heartbeat
            self._worker_heartbeat_times[rank] = time.time()
            logger.debug(
                f"[spmd_id={self.spmd_id}] Driver {self.spmd_id} received worker {rank} heartbeat: "
                f"status={heartbeat.status.name}, op_steps={heartbeat.op_steps}, "
                f"mailbox_full={heartbeat.mailbox_status.is_full if heartbeat.mailbox_status else False}"
            )

            # Check for worker error
            if heartbeat.error_message:
                self._handle_worker_error(rank, heartbeat.error_message)

            # Update monitor
            if heartbeat.telemetry:
                self.monitor.report_single_worker_telemetry(
                    str(rank), heartbeat.telemetry
                )

            # Check if all workers have reported heartbeat (workers_ready)
            if not self._workers_ready:
                if len(self._worker_heartbeat_times) >= self.config.worker_count:
                    self._workers_ready = True
                    logger.info(
                        f"[spmd_id={self.spmd_id}] All {self.config.worker_count} workers are ready for SPMD {self.spmd_id}"
                    )

                    # Event-driven: trigger INIT_COMPLETE now that all workers are alive
                    if self.state_machine.state == DriverState.INITIALIZING:
                        self.state_machine.trigger(DriverEvent.INIT_COMPLETE)
                        logger.info(
                            f"[spmd_id={self.spmd_id}] INIT_COMPLETE triggered by worker heartbeats "
                            f"for SPMD {self.spmd_id}"
                        )

            # Check if all workers are running (workers_running)
            if not self._workers_running and self._workers_ready:
                from ...common.types import WorkerStatus

                all_running = all(
                    hb.status == WorkerStatus.RUNNING
                    for hb in self._worker_heartbeats.values()
                )
                if (
                    all_running
                    and len(self._worker_heartbeats) >= self.config.worker_count
                ):
                    self._workers_running = True
                    logger.info(
                        f"[spmd_id={self.spmd_id}] All workers are running for SPMD {self.spmd_id} "
                        f"(topology registration complete)"
                    )
                    # Trigger READY → RUNNING transition
                    if self.state_machine.state == DriverState.READY:
                        self.state_machine.trigger(DriverEvent.START)
                        logger.info(
                            f"[spmd_id={self.spmd_id}] START triggered, "
                            f"driver state: READY → RUNNING"
                        )

        def _handle_worker_error(self, rank: int, error: str) -> None:
            """Handle worker error."""
            logger.error(
                f"[spmd_id={self.spmd_id}] Worker {rank} in SPMD {self.spmd_id} reported error: {error}"
            )

            attempts = self._recovery_attempts.get(rank, 0)

            if attempts < self._max_recovery_attempts:
                # Try worker-level recovery
                self.state_machine.trigger(
                    DriverEvent.WORKER_ERROR, rank=rank, error=error
                )
            else:
                # Too many attempts, report to controller
                self._report_error(
                    f"Worker {rank} recovery failed after {attempts} attempts: {error}"
                )

        # =====================================================================
        # Worker Op Completion Handling
        # =====================================================================

        def report_worker_op_completed(
            self, rank: int, op_name: str, step: int
        ) -> None:
            """
            Receive op completion report from a worker.

            Args:
                rank: Worker rank
                op_name: Operation name
                step: Completed step number
            """
            self.worker_synchronizer.report_worker_op_completed(rank, op_name, step)

        def _on_spmd_op_completed(self, spmd_id: str, op_name: str, step: int) -> None:
            """
            Callback when SPMD completes an op step.

            Reports to Controller.
            """
            if self.controller_handle is not None:
                try:
                    self.controller_handle.report_op_completed.remote(
                        spmd_id, op_name, step
                    )
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to report op completion to controller: {e}"
                    )

        # =====================================================================
        # Op Blocking
        # =====================================================================

        def block_op(self, op_name: str) -> None:
            """Block an op for all workers (called by Controller)."""
            self.worker_synchronizer.block_spmd_op(op_name)

        def unblock_op(self, op_name: str) -> None:
            """Unblock an op for all workers (called by Controller)."""
            self.worker_synchronizer.unblock_spmd_op(op_name)

        def block_all_ops(self) -> None:
            """Block all ops for all workers."""
            self.worker_synchronizer.block_all_ops()
            self.state_machine.trigger(DriverEvent.BLOCK_REQUESTED)

        def unblock_all_ops(self) -> None:
            """Unblock all ops for all workers."""
            self.worker_synchronizer.unblock_all_ops()
            self.state_machine.trigger(DriverEvent.UNBLOCK_REQUESTED)

        def _block_worker_op(self, rank: int, op_name: str) -> None:
            """Block a specific worker's op."""
            worker = self._worker_handles.get(rank)
            if worker is not None:
                try:
                    worker.block_op.remote(op_name)
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to block worker {rank} op: {e}"
                    )

        def _unblock_worker_op(self, rank: int, op_name: str) -> None:
            """Unblock a specific worker's op."""
            worker = self._worker_handles.get(rank)
            if worker is not None:
                try:
                    worker.unblock_op.remote(op_name)
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to unblock worker {rank} op: {e}"
                    )

        def _block_all_workers(self) -> None:
            """Block all workers (internal method)."""
            for rank, worker in self._worker_handles.items():
                try:
                    ops = list(self.worker_synchronizer.get_spmd_steps().keys())
                    for op_name in ops:
                        worker.block_op.remote(op_name)
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to block worker {rank}: {e}"
                    )

        # =====================================================================
        # State Argument Broadcasting
        # =====================================================================

        def broadcast_state_argument(
            self, target_op: str, names: List[str], step: int, values: List[Any]
        ) -> None:
            """
            Broadcast state_arguments to all workers via receive_states.

            Args:
                target_op: Target operator type on the receiving workers
                names: List of state variable names
                step: Current step number
                values: List of state values
            """
            for rank, worker in self._worker_handles.items():
                try:
                    worker.receive_states.remote(target_op, step, names, values)
                    logger.debug(
                        f"[spmd_id={self.spmd_id}] Broadcasted step {step} state_arguments {names} to worker {rank}"
                    )
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to send state_arguments to worker: {e}"
                    )

            logger.debug(
                f"[spmd_id={self.spmd_id}] Broadcasted step {step} state_arguments {names} to all workers"
            )

        def forward_state_argument_to_controller(
            self,
            target_spmd_id: str,
            target_op: str,
            names: List[str],
            step: int,
            values: List[Any],
        ) -> None:
            """
            Forward state_arguments to controller (called by rank0 worker).

            Args:
                target_spmd_id: Target SPMD ID
                target_op: Target operator type on the receiving side
                names: State variable names
                step: Current step number
                values: State values
            """
            if self.controller_handle is not None:
                try:
                    self.controller_handle.receive_state_argument.remote(
                        self.spmd_id, target_spmd_id, target_op, names, step, values
                    )
                    logger.debug(
                        f"[spmd_id={self.spmd_id}] Forwarded step {step} state_arguments {names} to controller"
                    )
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to forward state_arguments to controller: {e}"
                    )

        # =====================================================================
        # Heartbeat Reporting to Controller
        # =====================================================================

        def _start_heartbeat(self) -> None:
            """Start heartbeat reporting to controller."""
            if self._heartbeat_running:
                return

            self._heartbeat_running = True
            # Use a thread for synchronous heartbeat loop
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, daemon=True
            )
            self._heartbeat_thread.start()

        def _stop_heartbeat(self) -> None:
            """Stop heartbeat reporting."""
            self._heartbeat_running = False

        def _heartbeat_loop(self) -> None:
            """Heartbeat reporting loop.

            Also checks for init timeout: if the driver is still in
            INITIALIZING state after _init_timeout seconds, triggers INIT_FAILED.
            """
            while self._heartbeat_running:
                try:
                    self._send_heartbeat()

                    # Check init timeout
                    if (
                        self.state_machine.state == DriverState.INITIALIZING
                        and self._init_start_time > 0
                    ):
                        elapsed = time.time() - self._init_start_time
                        if elapsed > self._init_timeout:
                            logger.error(
                                f"[spmd_id={self.spmd_id}] SPMD {self.spmd_id} worker init "
                                f"timeout after {elapsed:.1f}s "
                                f"(received heartbeat from "
                                f"{len(self._worker_heartbeat_times)}/{self.config.worker_count} workers)"
                            )
                            self.state_machine.trigger(
                                DriverEvent.INIT_FAILED,
                                error=f"Worker init timeout after {elapsed:.1f}s",
                            )
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Error sending heartbeat: {e}"
                    )

                time.sleep(self._heartbeat_config.heartbeat_interval)

        def _send_heartbeat(self) -> None:
            """Send heartbeat to controller."""
            if self.controller_handle is None:
                return

            # Count healthy workers
            healthy_count = 0
            current_time = time.time()
            for rank in range(self.config.worker_count):
                last_time = self._worker_heartbeat_times.get(rank, 0)
                if current_time - last_time < self._heartbeat_config.heartbeat_timeout:
                    healthy_count += 1

            # Determine status
            if self.state_machine.is_error():
                status = DriverStatus.ERROR
            elif healthy_count < self.config.worker_count:
                status = DriverStatus.WARNING
            else:
                status = DriverStatus.HEALTHY

            # Calculate busy worker ratio for backpressure
            busy_count = 0
            for hb in self._worker_heartbeats.values():
                if hb.mailbox_status is not None and hb.mailbox_status.is_full:
                    busy_count += 1
            self._busy_worker_ratio = (
                busy_count / self.config.worker_count
                if self.config.worker_count > 0
                else 0.0
            )

            heartbeat = DriverHeartbeat(
                spmd_id=self.spmd_id,
                timestamp=current_time,
                status=status,
                state=self.state_machine.state,
                error_message=self.state_machine.get_error_message(),
                op_steps=self.worker_synchronizer.get_spmd_steps(),
                worker_count=self.config.worker_count,
                healthy_worker_count=healthy_count,
                blocked_ops=self.worker_synchronizer.get_locked_ops(),
                workers_ready=self._workers_ready,
                workers_running=self._workers_running,
                is_downstream_busy=self._is_paused,
                reader_state=self._reader_state
                if self.node_type == NodeType.READER
                else None,
                busy_worker_ratio=self._busy_worker_ratio,
            )

            try:
                self.controller_handle.receive_driver_heartbeat.remote(heartbeat)
                logger.debug(
                    f"[spmd_id={self.spmd_id}] Driver {self.spmd_id} sent heartbeat: status={status.name}, "
                    f"state={heartbeat.state.name}, healthy_workers={healthy_count}/{self.config.worker_count}, "
                    f"op_steps={heartbeat.op_steps}, busy_ratio={self._busy_worker_ratio:.2f}"
                )
            except Exception as e:
                logger.error(f"[spmd_id={self.spmd_id}] Failed to send heartbeat: {e}")

        # =====================================================================
        # Worker Recovery
        # =====================================================================

        def _recover_worker(self, rank: int) -> bool:
            """
            Attempt to recover a failed worker.

            Args:
                rank: Worker rank

            Returns:
                True if recovery successful
            """
            logger.info(
                f"[spmd_id={self.spmd_id}] Attempting to recover worker {rank} in SPMD {self.spmd_id}"
            )

            self._recovery_attempts[rank] = self._recovery_attempts.get(rank, 0) + 1

            try:
                # Kill old worker if exists
                old_worker = self._worker_handles.get(rank)
                if old_worker is not None:
                    try:
                        ray.kill(old_worker)
                    except Exception:
                        pass

                # Start new worker
                bundle_idx = self.worker_bundle_indices[rank]
                self_handle = ray.get_runtime_context().current_actor

                from ..worker import SPMDWorker

                if self.placement_group is not None:
                    worker = SPMDWorker.options(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=bundle_idx,
                        name=f"spmd_worker_{self.spmd_id}_{rank}_recovered",
                    ).remote(
                        config=self.config,
                        rank=rank,
                        spmd_id=self.spmd_id,
                        driver_handle=self_handle,
                    )
                else:
                    worker = SPMDWorker.remote(
                        config=self.config,
                        rank=rank,
                        spmd_id=self.spmd_id,
                        driver_handle=self_handle,
                    )

                self._worker_handles[rank] = worker

                # Start worker execution if we're in RUNNING state
                if self.state_machine.is_running():
                    worker.start.remote()

                logger.info(
                    f"[spmd_id={self.spmd_id}] Successfully recovered worker {rank}"
                )
                return True

            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id}] Failed to recover worker {rank}: {e}"
                )
                return False

        # =====================================================================
        # Controller Reporting
        # =====================================================================

        def _report_completion(self) -> None:
            """Report completion to controller."""
            if self.controller_handle is not None:
                try:
                    self.controller_handle.report_spmd_completed.remote(self.spmd_id)
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to report completion: {e}"
                    )

        def _report_error(self, error: str) -> None:
            """Report error to controller."""
            if self.controller_handle is not None:
                try:
                    self.controller_handle.report_spmd_error.remote(self.spmd_id, error)
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to report error: {e}"
                    )

        # =====================================================================
        # Topology Update
        # =====================================================================

        def update_topology(
            self, predecessors: List[str], successors: List[str]
        ) -> None:
            """
            Update topology information.

            Args:
                predecessors: List of predecessor SPMD IDs
                successors: List of successor SPMD IDs
            """
            self.graph_topology["predecessors"] = predecessors
            self.graph_topology["successors"] = successors
            logger.debug(
                f"[spmd_id={self.spmd_id}] Updated topology for SPMD {self.spmd_id}: "
                f"pred={predecessors}, succ={successors}"
            )

        # =====================================================================
        # Status Query
        # =====================================================================

        def get_state(self) -> DriverState:
            """Get current driver state."""
            return self.state_machine.state

        def get_status(self) -> Dict[str, Any]:
            """Get detailed driver status."""
            status = {
                "spmd_id": self.spmd_id,
                "state": self.state_machine.state.value,
                "node_type": self.node_type.value,
                "worker_count": self.config.worker_count,
                "op_steps": self.worker_synchronizer.get_spmd_steps(),
                "locked_ops": self.worker_synchronizer.get_locked_ops(),
                "error_workers": self.state_machine.get_error_workers(),
            }

            # Add Reader-specific status
            if self.node_type == NodeType.READER:
                status.update(
                    {
                        "reader_state": self._reader_state.value,
                        "is_paused": self._is_paused,
                        "file_tasks_count": len(self._file_tasks),
                    }
                )

            return status

        # =====================================================================
        # Reader Backpressure Control
        # =====================================================================

        def pause_reading(self) -> None:
            """
            Pause data reading (called by Controller due to backpressure).

            This method is called when downstream SPMDs report busy status.
            """
            if self.node_type != NodeType.READER:
                logger.warning(
                    f"[spmd_id={self.spmd_id}] pause_reading called on non-Reader node {self.spmd_id}"
                )
                return

            if self._is_paused:
                return  # Already paused

            self._is_paused = True
            self._reader_state = ReaderState.PAUSED
            logger.info(
                f"[spmd_id={self.spmd_id}] Reader {self.spmd_id} paused due to backpressure"
            )

            # Notify all workers to pause reading
            for worker in self._worker_handles.values():
                try:
                    worker.pause_reading.remote()
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to pause worker: {e}"
                    )

        def resume_reading(self) -> None:
            """
            Resume data reading (called by Controller when backpressure is relieved).
            """
            if self.node_type != NodeType.READER:
                logger.warning(
                    f"[spmd_id={self.spmd_id}] resume_reading called on non-Reader node {self.spmd_id}"
                )
                return

            if not self._is_paused:
                return  # Not paused

            self._is_paused = False
            self._reader_state = ReaderState.READING
            logger.info(
                f"[spmd_id={self.spmd_id}] Reader {self.spmd_id} resumed reading"
            )

            # Notify all workers to resume reading
            for worker in self._worker_handles.values():
                try:
                    worker.resume_reading.remote()
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id}] Failed to resume worker: {e}"
                    )

        def is_reader(self) -> bool:
            """Check if this driver is a Reader."""
            return self.node_type == NodeType.READER

        def get_reader_state(self) -> Optional[ReaderState]:
            """Get current Reader state."""
            if self.node_type == NodeType.READER:
                return self._reader_state
            return None

        def get_busy_worker_ratio(self) -> float:
            """Get ratio of workers with full mailbox."""
            return self._busy_worker_ratio

else:
    # Mock class when Ray is not available
    class SPMDDriver:
        """Mock SPMD Driver for testing without Ray."""

        def __init__(self, spmd_id: str, **kwargs):
            self.spmd_id = spmd_id
            logger.warning(
                f"[spmd_id={spmd_id}] Created mock SPMDDriver for {spmd_id} (Ray not available)"
            )

        def initialize(self, checkpoint_id: Optional[str] = None) -> bool:
            return True

        def start(self) -> None:
            pass

        def stop(self, save_checkpoint: bool = True) -> None:
            pass
