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
from collections import defaultdict
import threading
import time
import pathlib
from typing import Any, Dict, List, Optional

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from fluxway.common.types import (
    DriverEvent,
    DriverHeartbeat,
    DriverState,
    DriverStatus,
    HeartbeatConfig,
    ReaderState,
    SPMDConfig,
    WorkerHeartbeat,
)
from fluxway.common.utils import ensure_logging_configured
from fluxway.spmd.driver.state_machine import DriverStateMachine
from fluxway.spmd.driver.synchronizer import SPMDSynchronizer, MaxSkewSyncPolicy
from fluxway.spmd.driver.monitor import SPMDMonitor
from fluxway.spmd.worker import SPMDWorker

from fluxway.common.utils.logging import get_logger

logger = get_logger(__name__)


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
            checkpoint_backend: Optional[Any] = None,
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
                checkpoint_backend: Per-worker checkpoint backend instance (WorkerCheckpointBackend)
            """
            # Ray actors run in separate worker processes; ensure INFO logs are not dropped.
            ensure_logging_configured()

            self.spmd_id = spmd_id
            self.config = config
            self.placement_group = placement_group
            self.worker_bundle_indices = worker_bundle_indices
            self.controller_handle = controller_handle
            self.graph_topology = graph_topology
            self._checkpoint_backend = checkpoint_backend

            # State machine
            self.state_machine = DriverStateMachine(self)

            self.spmd_synchronizer = SPMDSynchronizer(
                spmd_id=spmd_id,
                num_workers=config.num_workers,
                sync_policy=MaxSkewSyncPolicy(config.max_worker_step_skew),
            )
            # Add known ops dynamically from config
            if hasattr(config, "operator_metas") and config.operator_metas:
                self.spmd_synchronizer.register_expected_ops(
                    list(config.operator_metas.keys())
                )
            elif hasattr(config, "operators") and config.operators:
                self.spmd_synchronizer.register_expected_ops(
                    list(config.operators.keys())
                )

            self.monitor = SPMDMonitor(spmd_id)

            # Worker handles
            self._worker_handles: Dict[int, Any] = {}

            # Heartbeat configuration
            self._heartbeat_config = HeartbeatConfig()
            self._heartbeat_task: Optional[asyncio.Task] = None
            self._heartbeat_running = False
            self._heartbeat_thread: Optional[threading.Thread] = None

            # Worker heartbeat tracking
            self._worker_heartbeats: Dict[int, WorkerHeartbeat] = {}
            self._worker_heartbeat_times: Dict[int, float] = {}

            # Recovery tracking
            self._recovery_attempts: Dict[int, int] = {}
            self._max_recovery_attempts = 3

            # Workers ready/running state tracking
            self._workers_ready: bool = False
            self._workers_running: bool = False
            self._stopping: bool = False

            # Init timeout (seconds) — if workers don't report within this time,
            # INIT_FAILED is triggered
            self._init_timeout: float = 120.0
            self._init_start_time: float = 0.0

            self._busy_worker_ratio: float = 0.0  # Ratio of workers with full mailbox

            # Backpressure / Pause control
            self._is_paused = False
            self._source_state: Optional[ReaderState] = ReaderState.INITIALIZING

            # Pending snapshot state
            self._pending_snapshot_id: Optional[str] = None
            self._pending_target_step_count: Optional[int] = None
            self._pending_ckpt_dir: Optional[str] = None

            logger.debug(
                f"Driver initialized spmd_id={spmd_id} num_workers={config.num_workers}"
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
                logger.debug(f"Initializing SPMD spmd_id={self.spmd_id}")

                # Set up worker synchronizer callbacks
                self.spmd_synchronizer.set_callbacks(
                    on_spmd_op_completed=self._on_spmd_op_completed,
                    block_worker_op=self._block_worker_op,
                    unblock_worker_op=self._unblock_worker_op,
                )

                # Start all worker actors
                self._start_workers()

                if checkpoint_id is not None:
                    raise NotImplementedError(
                        "Checkpoint restore must go through Controller.restore_snapshot(). "
                        f"checkpoint_id={checkpoint_id} spmd_id={self.spmd_id}"
                    )

                # Record init start time for timeout detection
                self._init_start_time = time.time()

                # Start heartbeat to controller immediately (state=INITIALIZING)
                # so controller can see this driver exists
                self._start_heartbeat()

                # NOTE: INIT_COMPLETE is NOT triggered here.
                # It will be triggered in receive_worker_heartbeat() when
                # all workers have reported their first heartbeat.

                logger.debug(
                    f"Workers started waiting for heartbeats spmd_id={self.spmd_id}"
                )
                return True

            except Exception as e:
                logger.error(f"Initialization failed error={e} spmd_id={self.spmd_id}")
                self.state_machine.trigger(DriverEvent.INIT_FAILED, error=str(e))
                return False

        def _start_workers(self) -> None:
            """Start all worker actors."""

            # Get self handle for registering with workers
            self_handle = ray.get_runtime_context().current_actor

            for rank, bundle_idx in enumerate(self.worker_bundle_indices):
                if self.placement_group is not None:
                    worker = SPMDWorker.options(  # type: ignore[attr-defined]
                        placement_group=self.placement_group,
                        placement_group_bundle_index=bundle_idx,
                        name=f"spmd_worker_{self.spmd_id}_{rank}",
                    ).remote(
                        config=self.config,
                        rank=rank,
                        spmd_id=self.spmd_id,
                        driver_handle=self_handle,
                        checkpoint_backend=self._checkpoint_backend,
                    )
                else:
                    worker = SPMDWorker.remote(
                        config=self.config,
                        rank=rank,
                        spmd_id=self.spmd_id,
                        driver_handle=self_handle,
                        checkpoint_backend=self._checkpoint_backend,
                    )

                self._worker_handles[rank] = worker
                self._recovery_attempts[rank] = 0
                logger.debug(f"Started worker rank={rank} spmd_id={self.spmd_id}")

            node_id_refs = {
                rank: worker.get_node_id.remote()
                for rank, worker in self._worker_handles.items()
            }
            node_id_values = ray.get(list(node_id_refs.values()))
            node_ids = dict(zip(node_id_refs.keys(), node_id_values, strict=True))
            rank_info = self._compute_rank_info(node_ids)

            for rank, worker in self._worker_handles.items():
                worker.setup_and_start.remote(**rank_info[rank])

            logger.debug(
                f"All workers started worker_count={len(self._worker_handles)} spmd_id={self.spmd_id}"
            )

        def _compute_rank_info(
            self, node_ids: Dict[int, str]
        ) -> Dict[int, Dict[str, int]]:
            node_groups: Dict[str, List[int]] = defaultdict(list)
            for rank, node_id in node_ids.items():
                node_groups[node_id].append(rank)

            for ranks in node_groups.values():
                ranks.sort()

            sorted_nodes = sorted(node_groups, key=lambda node_id: node_groups[node_id][0])
            node_to_rank = {
                node_id: node_rank for node_rank, node_id in enumerate(sorted_nodes)
            }

            return {
                rank: {
                    "local_rank": local_rank,
                    "local_world_size": len(ranks),
                    "node_rank": node_to_rank[node_id],
                    "num_nodes": len(sorted_nodes),
                }
                for node_id, ranks in node_groups.items()
                for local_rank, rank in enumerate(ranks)
            }

        def get_worker_handles(self) -> Dict[int, Any]:
            """Get all worker handles for topology registration."""
            return dict(self._worker_handles)

        def register_topology(
            self,
            predecessor_handlers: Dict[
                str, Dict[int, Any]
            ],  # {spmd_id: {rank: handle}}
            successor_handlers: Dict[str, Dict[int, Any]],  # {spmd_id: {rank: handle}}
            predecessor_count: int,
            successor_count: int,
        ) -> None:
            """
            Register topology with provided worker handles.

            Called by Controller when all SPMDs are ready.
            Worker handles are provided directly to avoid actor re-entrancy deadlocks.

            Args:
                predecessor_handlers: {spmd_id: {rank: worker_handle}}
                successor_handlers: {spmd_id: {rank: worker_handle}}
                predecessor_count: expected predecessor count for workers
                successor_count: expected successor count for workers
            """
            logger.debug(
                f"Topology registration starting "
                f"predecessor_count={predecessor_count} successor_count={successor_count} "
                f"spmd_id={self.spmd_id}"
            )

            # Step 1: Notify workers of expected topology counts
            for worker in self._worker_handles.values():
                worker.set_expected_topology.remote(predecessor_count, successor_count)

            # Step 2: Register predecessor worker handles
            for spmd_id, handlers in predecessor_handlers.items():
                for worker in self._worker_handles.values():
                    worker.register_predecessor_handlers.remote(spmd_id, handlers)
                logger.debug(
                    f"Registered predecessor predecessor_spmd_id={spmd_id} spmd_id={self.spmd_id}"
                )

            # Step 3: Register successor worker handles
            for spmd_id, handlers in successor_handlers.items():
                for worker in self._worker_handles.values():
                    worker.register_successor_handlers.remote(spmd_id, handlers)
                logger.debug(
                    f"Registered successor successor_spmd_id={spmd_id} spmd_id={self.spmd_id}"
                )

            logger.debug(f"Topology registration completed spmd_id={self.spmd_id}")

        async def stop(self, save_checkpoint: bool = True) -> None:
            """Stop execution."""
            logger.debug(f"Stopping spmd_id={self.spmd_id}")
            self._stopping = True

            # Stop heartbeat
            self._stop_heartbeat()

            # Stop all workers
            stop_refs: List[tuple[int, Any]] = []
            for rank, worker in self._worker_handles.items():
                try:
                    stop_refs.append((rank, worker.stop.remote()))
                except Exception as e:
                    logger.error(
                        f"Failed to stop worker rank={rank} error={e} spmd_id={self.spmd_id}"
                    )

            worker_handles = list(self._worker_handles.items())

            if stop_refs:
                results = await asyncio.gather(
                    *(ref for _, ref in stop_refs),
                    return_exceptions=True,
                )
                for (rank, _), result in zip(stop_refs, results, strict=True):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Worker stop RPC failed rank={rank} error={result} "
                            f"spmd_id={self.spmd_id}"
                        )

            detach_refs: List[tuple[int, Any]] = []
            for rank, worker in worker_handles:
                try:
                    detach_refs.append((rank, worker.set_driver_handle.remote(None)))
                except Exception as e:
                    logger.error(
                        f"Failed to detach worker rank={rank} error={e} "
                        f"spmd_id={self.spmd_id}"
                    )

            if detach_refs:
                detach_results = await asyncio.gather(
                    *(ref for _, ref in detach_refs),
                    return_exceptions=True,
                )
                for (rank, _), result in zip(
                    detach_refs, detach_results, strict=True
                ):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Worker detach RPC failed rank={rank} error={result} "
                            f"spmd_id={self.spmd_id}"
                        )

            self._worker_handles.clear()

        # =====================================================================
        # Snapshot Orchestration (Drain → Snapshot → Resume)
        # =====================================================================

        async def request_drain(self, target_count: int, reason: str = "") -> None:
            if self.state_machine.state not in (DriverState.RUNNING, DriverState.READY):
                raise RuntimeError(
                    f"request_drain requires RUNNING or READY state, "
                    f"got {self.state_machine.state} spmd_id={self.spmd_id}"
                )
            self._pending_target_step_count = target_count
            self.state_machine.trigger(DriverEvent.DRAIN_REQUESTED)
            await asyncio.gather(
                *[
                    worker.set_drain_target_count.remote(target_count)
                    for worker in self._worker_handles.values()
                ]
            )
            self.spmd_synchronizer.drain_to_step_count(
                target_count=target_count, on_drained=self._on_drain_complete
            )

        async def request_snapshot(
            self, snapshot_id: str, target_count: int, ckpt_dir: str
        ) -> None:
            if self.state_machine.state != DriverState.DRAINED:
                raise RuntimeError(
                    f"request_snapshot requires DRAINED state, "
                    f"got {self.state_machine.state} spmd_id={self.spmd_id}"
                )
            self.state_machine.trigger(DriverEvent.SNAPSHOT_REQUESTED)
            self._pending_snapshot_id = snapshot_id
            self._pending_target_step_count = target_count
            self._pending_ckpt_dir = ckpt_dir
            self._persist_worker_states()

        def _on_drain_complete(self) -> None:
            if self.state_machine.state != DriverState.DRAINING:
                raise RuntimeError(
                    f"_on_drain_complete requires DRAINING state, "
                    f"got {self.state_machine.state} spmd_id={self.spmd_id}"
                )
            self.state_machine.trigger(DriverEvent.DRAIN_COMPLETE)
            if self.controller_handle is not None:
                self.controller_handle.report_driver_drained.remote(self.spmd_id)

        def _persist_worker_states(self) -> None:
            if (
                self._pending_ckpt_dir is None
                or self._pending_target_step_count is None
            ):
                logger.error(
                    f"Failed to persist worker states: pending_ckpt_dir or pending_target_step_count is not set. spmd_id={self.spmd_id}"
                )
                self._clear_drain_and_fail()
                return
            spmd_dir = (
                pathlib.Path(self._pending_ckpt_dir)
                / str(self._pending_target_step_count)
                / self.spmd_id
            )
            try:
                force_write = getattr(self._checkpoint_backend, "_force_write", False)
                if spmd_dir.exists() and any(spmd_dir.iterdir()):
                    if not force_write:
                        raise FileExistsError(
                            f"Checkpoint directory {spmd_dir} already exists and is not empty. "
                            "Set force_write=True in CheckpointConfig to overwrite."
                        )
                    else:
                        import shutil

                        shutil.rmtree(spmd_dir, ignore_errors=True)

                spmd_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(
                    f"Failed to create checkpoint directory {spmd_dir} error={e} spmd_id={self.spmd_id}"
                )
                self._clear_drain_and_fail()
                return

            metadata_futures = {
                rank: worker.save_state_to_disk.remote(
                    self._pending_ckpt_dir, self._pending_target_step_count
                )
                for rank, worker in self._worker_handles.items()
            }
            # Submit snapshot metadata to controller.
            # This is a fire-and-forget remote call. In Ray, `.remote()` does not block
            # and will not raise network/actor-dead errors synchronously.
            # Synchronous exceptions here only indicate local programmatic bugs (e.g. TypeError)
            # and should fail-fast directly rather than being swallowed by generic exception handlers.
            if self.controller_handle is None:
                logger.error(
                    f"controller_handle is None, cannot report snapshot metadata "
                    f"spmd_id={self.spmd_id}"
                )
                self._clear_drain_and_fail()
                return

            self.controller_handle.receive_snapshot_metadata.remote(
                self.spmd_id, self._pending_snapshot_id, metadata_futures
            )

        def _clear_drain_and_fail(self) -> None:
            """Unblock all drained Workers, then mark this Driver as failed.

            Workers self-block in _execute_operator by await-ing _drain_resume_event.
            If SNAPSHOT_FAILED fires without first clearing the drain, all Workers
            remain permanently suspended because resume() requires SNAPSHOTTING state
            and will raise in ERROR state.
            """
            self.spmd_synchronizer.clear_drain()
            for worker in self._worker_handles.values():
                worker.clear_drain_target_count.remote()
            if self.controller_handle and self._pending_snapshot_id:
                self.controller_handle.report_snapshot_failed.remote(
                    self.spmd_id, self._pending_snapshot_id
                )
            self.state_machine.trigger(DriverEvent.SNAPSHOT_FAILED)

        def resume(self) -> None:
            """
            Phase 3 of snapshot: clear drain ceiling and resume training.

            Transitions: SNAPSHOTTING → RUNNING.
            Must be called by the controller after snapshot data is consumed.

            Order:
              1. Synchronizer clears the ACK fence.
              2. Fan-out clear_drain_target_count to all Workers so they resume
                 executing steps that were deferred behind the target.
            """
            if self.state_machine.state not in (
                DriverState.SNAPSHOTTING,
                DriverState.DRAINED,
            ):
                raise RuntimeError(
                    f"resume requires SNAPSHOTTING or DRAINED state, "
                    f"got {self.state_machine.state} spmd_id={self.spmd_id}"
                )
            self.spmd_synchronizer.clear_drain()
            for worker in self._worker_handles.values():
                worker.clear_drain_target_count.remote()
            self.state_machine.trigger(DriverEvent.SNAPSHOT_COMPLETE)

        async def restore_from_disk(
            self, rank_metadata_map: Dict[int, Dict[str, Any]]
        ) -> None:
            """
            Fan out checkpoint restore to all workers in parallel.

            Each Worker reads its own tensors directly from the shared filesystem.

            Args:
                rank_metadata_map: {rank: metadata_dict} where metadata_dict contains
                                   ``backend`` and backend-specific path fields.
            """
            self.spmd_synchronizer.load_step_counts(rank_metadata_map)

            tasks = []
            for rank, metadata in rank_metadata_map.items():
                worker = self._worker_handles.get(rank)
                if worker is None:
                    raise RuntimeError(
                        f"Worker rank {rank} not found spmd_id={self.spmd_id}"
                    )
                tasks.append(worker.load_state_from_disk.remote(metadata))
            await asyncio.gather(*tasks)

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
            if self._stopping:
                logger.debug(
                    f"Ignoring worker heartbeat during shutdown rank={rank} spmd_id={self.spmd_id}"
                )
                return

            self._worker_heartbeats[rank] = heartbeat
            self._worker_heartbeat_times[rank] = time.time()
            logger.heartbeat(
                f"Received worker heartbeat rank={rank} "
                f"status={heartbeat.status.name} completed_count={heartbeat.completed_step_count} "
                f"mailbox_full={heartbeat.mailbox_status.is_full if heartbeat.mailbox_status else False} "
                f"spmd_id={self.spmd_id}"
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
                if len(self._worker_heartbeat_times) >= self.config.num_workers:
                    self._workers_ready = True
                    logger.heartbeat(
                        f"All workers ready worker_count={self.config.num_workers} spmd_id={self.spmd_id}"
                    )

                    # Event-driven: trigger INIT_COMPLETE now that all workers are alive
                    if self.state_machine.state == DriverState.INITIALIZING:
                        self.state_machine.trigger(DriverEvent.INIT_COMPLETE)
                        logger.heartbeat(
                            f"INIT_COMPLETE triggered spmd_id={self.spmd_id}"
                        )

            # Check if all workers are running (workers_running)
            if not self._workers_running and self._workers_ready:
                from fluxway.common.types import WorkerStatus

                all_running = all(
                    hb.status in (WorkerStatus.RUNNING, WorkerStatus.BLOCKED)
                    for hb in self._worker_heartbeats.values()
                )
                if (
                    all_running
                    and len(self._worker_heartbeats) >= self.config.num_workers
                ):
                    self._workers_running = True
                    logger.heartbeat(
                        f"All workers running triggering START spmd_id={self.spmd_id}"
                    )
                    # Trigger READY → RUNNING transition
                    if self.state_machine.state == DriverState.READY:
                        self.state_machine.trigger(DriverEvent.START)

        def _handle_worker_error(self, rank: int, error: str) -> None:
            """Handle worker error."""
            logger.error(
                f"Worker error rank={rank} error={error} spmd_id={self.spmd_id}"
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
            if self._stopping:
                logger.debug(
                    f"Ignoring worker op completion during shutdown rank={rank} "
                    f"op={op_name} step={step} spmd_id={self.spmd_id}"
                )
                return

            self.spmd_synchronizer.report_worker_op_completed(rank, op_name, step)

        def _on_spmd_op_completed(self, spmd_id: str, op_name: str, step: int) -> None:
            """
            Callback when SPMD completes an op step.

            Reports to Controller.
            """
            if self.controller_handle is not None:
                self.controller_handle.report_op_completed.remote(
                    spmd_id, op_name, step
                )

        # =====================================================================
        # Op Blocking
        # =====================================================================

        def block_op(self, op_name: str) -> None:
            """Block an op for all workers (called by Controller)."""
            self.spmd_synchronizer.block_spmd_op(op_name)

        def unblock_op(self, op_name: str) -> None:
            """Unblock an op for all workers (called by Controller)."""
            self.spmd_synchronizer.unblock_spmd_op(op_name)

        def set_allowed_step(self, op_name: str, step: int) -> None:
            """Set global watermark for an op (called by Controller)."""
            self.spmd_synchronizer.update_allowed_step(op_name, step)

        def block_all_ops(self) -> None:
            """Block all ops for all workers."""
            self.spmd_synchronizer.block_all_ops()
            self.state_machine.trigger(DriverEvent.BLOCK_REQUESTED)

        def unblock_all_ops(self) -> None:
            """Unblock all ops for all workers."""
            self.spmd_synchronizer.unblock_all_ops()
            self.state_machine.trigger(DriverEvent.UNBLOCK_REQUESTED)

        def _block_worker_op(self, rank: int, op_name: str) -> None:
            """Block a specific worker's op."""
            worker = self._worker_handles.get(rank)
            if worker is not None:
                try:
                    worker.block_op.remote(op_name)
                except Exception as e:
                    logger.error(
                        f"Failed to block worker op rank={rank} error={e} spmd_id={self.spmd_id}"
                    )

        def _unblock_worker_op(self, rank: int, op_name: str) -> None:
            """Unblock a specific worker's op."""
            worker = self._worker_handles.get(rank)
            if worker is not None:
                try:
                    worker.unblock_op.remote(op_name)
                except Exception as e:
                    logger.error(
                        f"Failed to unblock worker op rank={rank} error={e} spmd_id={self.spmd_id}"
                    )

        def _block_all_workers(self) -> None:
            """Block all workers (internal method)."""
            for rank, worker in self._worker_handles.items():
                try:
                    ops = list(self.spmd_synchronizer.get_spmd_steps().keys())
                    for op_name in ops:
                        worker.block_op.remote(op_name)
                except Exception as e:
                    logger.error(
                        f"Failed to block worker rank={rank} error={e} spmd_id={self.spmd_id}"
                    )

        # =====================================================================
        # State Argument Broadcasting
        # =====================================================================

        def broadcast_control_argument(
            self, target_op: str, names: List[str], step: int, values: List[Any]
        ) -> None:
            """
            Broadcast control_arguments to all workers via receive_control.

            Args:
                target_op: Target operator type on the receiving workers
                names: List of state variable names
                step: Current step number
                values: List of state values
            """
            for rank, worker in self._worker_handles.items():
                try:
                    worker.receive_control.remote(target_op, step, names, values)
                    logger.debug(
                        f"Broadcast control_arguments names={names} step={step} "
                        f"target_rank={rank} spmd_id={self.spmd_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to send control_arguments to worker rank={rank} error={e} "
                        f"spmd_id={self.spmd_id}"
                    )

            logger.debug(
                f"Broadcast control_arguments to all workers names={names} step={step} "
                f"spmd_id={self.spmd_id}"
            )

        def routing_control_argument_to_controller(
            self,
            target_spmd_id: str,
            target_op: str,
            names: List[str],
            step: int,
            values: List[Any],
        ) -> None:
            """
            Route control_arguments to controller (called by rank0 worker).

            Args:
                target_spmd_id: Target SPMD ID
                target_op: Target operator type on the receiving side
                names: State variable names
                step: Current step number
                values: State values
            """
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
            heartbeat_thread = self._heartbeat_thread
            if (
                heartbeat_thread is not None
                and heartbeat_thread.is_alive()
                and heartbeat_thread is not threading.current_thread()
            ):
                heartbeat_thread.join(
                    timeout=self._heartbeat_config.heartbeat_interval + 1.0
                )
            self._heartbeat_thread = None

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
                                f"Worker init timeout spmd_id={self.spmd_id} "
                                f"elapsed={elapsed:.1f}s "
                                f"reported_workers={len(self._worker_heartbeat_times)}/{self.config.num_workers}"
                            )
                            self.state_machine.trigger(
                                DriverEvent.INIT_FAILED,
                                error=f"Worker init timeout after {elapsed:.1f}s",
                            )
                except Exception as e:
                    logger.error(
                        f"Error sending heartbeat error={e} spmd_id={self.spmd_id}"
                    )

                time.sleep(self._heartbeat_config.heartbeat_interval)

        def _send_heartbeat(self) -> None:
            """Send heartbeat to controller."""
            if self.controller_handle is None:
                return

            # Count healthy workers
            healthy_count = 0
            current_time = time.time()
            for rank in range(self.config.num_workers):
                last_time = self._worker_heartbeat_times.get(rank, 0)
                if current_time - last_time < self._heartbeat_config.heartbeat_timeout:
                    healthy_count += 1

            # Determine status
            if self.state_machine.is_error():
                status = DriverStatus.ERROR
            elif healthy_count < self.config.num_workers:
                status = DriverStatus.WARNING
            else:
                status = DriverStatus.HEALTHY

            # Calculate busy worker ratio for backpressure
            busy_count = 0
            for hb in self._worker_heartbeats.values():
                if hb.mailbox_status is not None and hb.mailbox_status.is_full:
                    busy_count += 1
            self._busy_worker_ratio = (
                busy_count / self.config.num_workers
                if self.config.num_workers > 0
                else 0.0
            )

            heartbeat = DriverHeartbeat(
                spmd_id=self.spmd_id,
                timestamp=time.time(),
                status=status,
                state=self.state_machine.state,
                error_message=self.state_machine.get_error_message(),
                completed_step_count=self.spmd_synchronizer.get_spmd_steps(),
                num_workers=self.config.num_workers,
                healthy_num_workers=healthy_count,
                blocked_ops=self.spmd_synchronizer.get_locked_ops(),
                workers_ready=self._workers_ready,
                workers_running=self._workers_running,
                is_downstream_busy=self._is_paused,
                source_state=self._source_state,
                busy_worker_ratio=self._busy_worker_ratio,
            )

            try:
                self.controller_handle.receive_driver_heartbeat.remote(heartbeat)
                logger.heartbeat(
                    f"Driver heartbeat sent status={status.name} "
                    f"state={heartbeat.state.name} "
                    f"healthy_workers={healthy_count}/{self.config.num_workers} "
                    f"completed_count={heartbeat.completed_step_count} "
                    f"busy_ratio={self._busy_worker_ratio:.2f} "
                    f"spmd_id={self.spmd_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to send heartbeat error={e} spmd_id={self.spmd_id}"
                )

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
            logger.debug(
                f"Attempting to recover worker rank={rank} spmd_id={self.spmd_id}"
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

                if self.placement_group is not None:
                    worker = SPMDWorker.options(  # type: ignore[attr-defined]
                        placement_group=self.placement_group,
                        placement_group_bundle_index=bundle_idx,
                        name=f"spmd_worker_{self.spmd_id}_{rank}_recovered",
                    ).remote(
                        config=self.config,
                        rank=rank,
                        spmd_id=self.spmd_id,
                        driver_handle=self_handle,
                        checkpoint_backend=self._checkpoint_backend,
                    )
                else:
                    worker = SPMDWorker.remote(
                        config=self.config,
                        rank=rank,
                        spmd_id=self.spmd_id,
                        driver_handle=self_handle,
                        checkpoint_backend=self._checkpoint_backend,
                    )

                self._worker_handles[rank] = worker

                # Start worker execution if we're in RUNNING state
                if self.state_machine.is_running():
                    worker.start.remote()

                logger.debug(
                    f"Worker recovery successful rank={rank} spmd_id={self.spmd_id}"
                )
                return True

            except Exception as e:
                logger.error(
                    f"Failed to recover worker rank={rank} error={e} spmd_id={self.spmd_id}"
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
                        f"Failed to report completion error={e} spmd_id={self.spmd_id}"
                    )

        def _report_error(self, error: str) -> None:
            """Report error to controller."""
            if self.controller_handle is not None:
                try:
                    self.controller_handle.report_spmd_error.remote(self.spmd_id, error)
                except Exception as e:
                    logger.error(
                        f"Failed to report error error={e} spmd_id={self.spmd_id}"
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
                f"Topology updated predecessors={predecessors} successors={successors} "
                f"spmd_id={self.spmd_id}"
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
                "node_type": "spmd",  # Generic node type
                "num_workers": self.config.num_workers,
                "completed_step_count": self.spmd_synchronizer.get_spmd_steps(),
                "locked_ops": self.spmd_synchronizer.get_locked_ops(),
                "error_workers": self.state_machine.get_error_workers(),
            }

            # Add Reader-specific status
            # Source nodes (like readers) are now generic SPMDs, so no special reader status
            pass

            return status

        # =====================================================================
        # Source Node Control
        # =====================================================================

        def pause_sources(self) -> None:
            """
            Pause all source operators (called by Controller due to backpressure).
            """
            if self._is_paused:
                return

            self._is_paused = True
            self._source_state = ReaderState.PAUSED
            logger.debug(f"Sources paused due to backpressure spmd_id={self.spmd_id}")

            for worker in self._worker_handles.values():
                try:
                    worker.pause_sources.remote()
                except Exception as e:
                    logger.error(
                        f"Failed to pause worker error={e} spmd_id={self.spmd_id}"
                    )

        def resume_sources(self) -> None:
            """
            Resume all source operators (called by Controller when backpressure relieved).
            """
            if not self._is_paused:
                return

            self._is_paused = False
            self._source_state = ReaderState.READING
            logger.debug(f"Sources resumed spmd_id={self.spmd_id}")

            for worker in self._worker_handles.values():
                try:
                    worker.resume_sources.remote()
                except Exception as e:
                    logger.error(
                        f"Failed to resume worker error={e} spmd_id={self.spmd_id}"
                    )

        def get_source_state(self) -> Optional[ReaderState]:
            """Get current source state."""
            return self._source_state

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
                f"Created mock SPMDDriver Ray not available spmd_id={spmd_id}"
            )

        @classmethod
        def options(cls, **kwargs) -> Any:
            """Mock Ray options."""
            return cls

        @classmethod
        def remote(cls, *args, **kwargs) -> Any:
            """Mock Ray remote call."""
            return cls(*args, **kwargs)

        def initialize(self, checkpoint_id: Optional[str] = None) -> bool:
            return True

        def start(self) -> None:
            pass

        def stop(self, save_checkpoint: bool = True) -> None:
            pass
