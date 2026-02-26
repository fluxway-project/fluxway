"""
SPMD Worker: Ray Actor for computation execution.

Based on ARCHITECTURE.md Section 3.2

Each SPMD Worker runs user-defined forward/backward/update logic
and implements dual-stream convergence with mailbox-based execution.

Key responsibilities:
1. Provide handler registration interface (handlers only, topology from config)
2. Provide data reception interface
3. Execute operators from priority queue
4. Send outputs to successors using computed routing
5. Provide op-level blocking (block one op without affecting others)
6. Report op completion to driver
7. Handle state_argument reception and propagate to successors

This implementation uses Ray's async actor API for concurrency.
See: https://docs.ray.io/en/latest/ray-core/actors/async_api.html
"""

from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.sharding import ShardingSpec, PartitionSpec
from typing import Any, Dict, List, Optional, Tuple
from fluxway.common.backend import get_backend
import asyncio
import logging
import time

import ray

from ...common.types import SPMDConfig, WorkerHeartbeat, WorkerStatus, WorkerTelemetry
from ...common.types.configs import NodeType, ReaderConfig, MailboxStatus
from ...common.types.tensor import TransportTensor
from ...common.utils import ensure_logging_configured
from ...rpc.protocol import DataPlaneCallable
from .mailbox import WorkerMailbox
from .operator_manager import OperatorManager


logger = logging.getLogger(__name__)


@ray.remote
class SPMDWorker:
    """
    SPMD Worker (Ray Async Actor).

    Implements dual-stream convergence with priority queue execution:
    - Data is received by WorkerMailbox (async-safe)
    - Executes operators based on priority metadata
    - Supports op-level blocking without affecting other ops
    - Reports op completion to driver
    - Handles state_argument for cross-SPMD state transfer

    Uses asyncio for concurrency instead of threading.
    Ray will automatically detect this as an async actor due to async method definitions.
    """

    def __init__(
        self,
        config: SPMDConfig,
        rank: int,
        spmd_id: str = "",
        driver_handle: Optional[Any] = None,
        node_type: NodeType = NodeType.SPMD,
        reader_task: Optional[Any] = None,
        reader_config: Optional[ReaderConfig] = None,
    ):
        """
        Initialize worker.

        Args:
            config: SPMD configuration with user functions
            rank: This worker's rank in the SPMD
            spmd_id: ID of this SPMD (for metadata tracking)
            driver_handle: Reference to the SPMD driver for callbacks (Native Ray)
            node_type: Type of node (SPMD or READER)
            reader_task: File read task (for READER nodes)
            reader_config: Reader configuration (for READER nodes)
        """
        # Ray actors run in separate worker processes; ensure they respect
        # FLUXWAY_LOG_LEVEL / FLUXWAY_LOG_FORCE so DEBUG logs are not dropped.
        ensure_logging_configured()
        self.config = config
        self.rank = rank
        self.spmd_id = spmd_id
        self._running = False

        # Driver handle for callbacks (Control Plane - Native Ray)
        self._driver_handle: Optional[Any] = driver_handle

        # Node type
        self.node_type = node_type
        self.reader_task = reader_task
        self.reader_config = reader_config

        # Mesh from config
        self._mesh = config.mesh
        self._total_rank = config.num_workers  # Total number of workers in this SPMD

        # Operator Manager (topology parsed from config) - only for SPMD nodes
        if node_type == NodeType.SPMD:
            self._op_manager = OperatorManager(config, rank, spmd_id)
        else:
            self._op_manager = None

        # Execution step counters {op_name: step} - tracks which step is being/was executed
        self._current_steps: Dict[str, int] = dict.fromkeys(
            self.config.operators if self.config else {}, 0
        )

        # Op-level blocking: asyncio.Event (set=runnable, clear=blocked)
        self._op_runnable: Dict[str, asyncio.Event] = {
            op_name: asyncio.Event()
            for op_name in (self.config.operators if self.config else {})
        }
        for event in self._op_runnable.values():
            event.set()  # Initially all ops are runnable

        # Pending tasks when blocked (to avoid priority queue reentry loop)
        self._pending_blocked: Dict[str, List[int]] = {
            op_name: [] for op_name in (self.config.operators if self.config else {})
        }

        # Pending tasks for future steps (arrived early, waiting for current step to complete)
        # Structure: {op_name: List[step]} - steps waiting to be executed
        self._pending_future_steps: Dict[str, List[int]] = {
            op_name: [] for op_name in (self.config.operators if self.config else {})
        }

        self._pending_lock = asyncio.Lock()

        # Priority queue: asyncio.PriorityQueue with (priority, step, op_name)
        # We can extract priority from OperatorMeta
        self._execution_queue: asyncio.PriorityQueue[Tuple[int, int, str]] = (
            asyncio.PriorityQueue()
        )

        # --- Backpressure components ---
        self._max_in_flight = 50
        self._send_semaphore = asyncio.Semaphore(self._max_in_flight)
        self._mailbox_capacity = 100  # Default value
        self._high_watermark = int(self._mailbox_capacity * 0.8)

        # Internal Mailbox (async-safe), wired to auto-enqueue when a step is ready
        self._mailbox = WorkerMailbox()
        self._mailbox.set_ready_callback(self._on_step_ready)

        # Handler storage (Data Plane - wrapped in DataPlaneCallable)
        # {spmd_id: {rank: DataPlaneCallable}}
        self._predecessor_handlers: Dict[str, Dict[int, DataPlaneCallable]] = {}
        self._successor_handlers: Dict[str, Dict[int, DataPlaneCallable]] = {}

        # Heartbeat configuration
        self._heartbeat_interval: float = 5.0
        self._heartbeat_running: bool = False
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Error tracking
        self._last_error: Optional[str] = None

        # Run loop synchronization
        self._start_event = asyncio.Event()  # Signals when to start execution
        self._run_task: Optional[asyncio.Task] = None

        # Expected topology counts for auto-start detection
        self._expected_predecessor_count: int = 0
        self._expected_successor_count: int = 0

        # Cached assembled input tensors: {op_name: {tensor_name: TransportTensor}}
        # Updated by _assemble_input, used by _send_output for pass-through metadata
        self._last_assembled_tensors: Dict[
            str, Dict[str, Optional[TransportTensor]]
        ] = {op_name: {} for op_name in (self.config.operators if self.config else {})}

        # Tracks which (op_name, step) barriers have been registered to avoid double-declaring.
        self._barriers_declared: Dict[Tuple[str, int], bool] = {}

        # Reader-specific state
        self._is_reading_paused: bool = False  # Backpressure pause flag
        self._reader_pause_event = asyncio.Event()
        self._reader_pause_event.set()  # Initially not paused
        self._reader_completed: bool = False  # Flag to indicate reading completion
        self._batches_sent: int = 0  # Counter for batches sent

        node_type_str = "Reader" if node_type == NodeType.READER else "SPMD"
        logger.info(
            f"{node_type_str}Worker initialized: spmd_id={spmd_id}, rank={rank}"
        )

        # Start heartbeat if driver_handle is provided
        if self._driver_handle is not None:
            self._start_heartbeat()

        # Start the run loop in background (will block until start() is called)
        self._start_run_loop()

    # =========================================================================
    # Driver Handle Management
    # =========================================================================

    def set_driver_handle(self, driver_handle: Any) -> None:
        """
        Set the driver handle for callbacks.

        Args:
            driver_handle: Reference to the SPMD driver
        """
        self._driver_handle = driver_handle

    def get_driver_handle(self) -> Optional[Any]:
        """Get the driver handle."""
        return self._driver_handle

    def _resolve_output_step(
        self, *, is_periodic: bool, current_step: int, input_step: int
    ) -> int:
        """
        Compute output step number based on SPMDConfig and operator mode.

        Spec:
        - use_global_step_indexing=False (default): output_step == input_step
        - use_global_step_indexing=True:
            - periodic_barrier=True: output_step = total_rank * current_step + local_rank
            - standard (non-periodic): output_step = current_step
        """
        use_global_step_indexing = getattr(
            self.config, "use_global_step_indexing", False
        )
        if not use_global_step_indexing:
            return input_step
        if is_periodic:
            return self._total_rank * current_step + self.rank
        return current_step

    # =========================================================================
    # Handler Registration (handlers only, topology from config)
    # =========================================================================

    def register_predecessor_handlers(
        self, spmd_id: str, handlers: Dict[int, Any]
    ) -> None:
        """
        Register handlers for a predecessor SPMD.

        Note: Topology info is already in config/OperatorMeta.
        Only handlers need runtime registration.

        Args:
            spmd_id: Predecessor SPMD ID
            handlers: {rank: ray.ActorHandle}
        """
        self._predecessor_handlers[spmd_id] = {
            rank: DataPlaneCallable(h) for rank, h in handlers.items()
        }
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Registered predecessor handlers: {spmd_id}, ranks={list(handlers.keys())}"
        )

        # Check if all handlers are registered for auto-start
        self._check_and_auto_start()

    def register_successor_handlers(
        self, spmd_id: str, handlers: Dict[int, Any]
    ) -> None:
        """
        Register handlers for a successor SPMD.

        Note: Topology info is already in config/OperatorMeta.
        Only handlers need runtime registration.

        Args:
            spmd_id: Successor SPMD ID
            handlers: {rank: ray.ActorHandle}
        """
        self._successor_handlers[spmd_id] = {
            rank: DataPlaneCallable(h) for rank, h in handlers.items()
        }
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Registered successor handlers: {spmd_id}, ranks={list(handlers.keys())}"
        )

        # Check if all handlers are registered for auto-start
        self._check_and_auto_start()

    def set_expected_topology(
        self, predecessor_count: int, successor_count: int
    ) -> None:
        """
        Set expected predecessor/successor counts for auto-start detection.

        Called by driver before topology registration begins.

        Args:
            predecessor_count: Expected number of predecessor SPMDs
            successor_count: Expected number of successor SPMDs
        """
        self._expected_predecessor_count = predecessor_count
        self._expected_successor_count = successor_count
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} expects {predecessor_count} predecessors, "
            f"{successor_count} successors"
        )

        # If both are 0 (source node with no successors), auto-start immediately
        if predecessor_count == 0 and successor_count == 0:
            self._auto_start()
        else:
            # Check if handlers are already registered
            self._check_and_auto_start()

    def _check_and_auto_start(self) -> None:
        """
        Check if all expected handlers are registered and auto-start if ready.
        """
        if self._running:
            return  # Already running

        # Check if all expected handlers are registered
        pred_registered = (
            len(self._predecessor_handlers) >= self._expected_predecessor_count
        )
        succ_registered = (
            len(self._successor_handlers) >= self._expected_successor_count
        )

        if pred_registered and succ_registered:
            logger.info(
                f"[spmd_id={self.spmd_id}] Worker {self.rank}: All handlers registered "
                f"(pred={len(self._predecessor_handlers)}/{self._expected_predecessor_count}, "
                f"succ={len(self._successor_handlers)}/{self._expected_successor_count}), "
                f"auto-starting execution"
            )
            self._auto_start()

    def _auto_start(self) -> None:
        """Internal auto-start method."""
        if self._running:
            return

        self._running = True
        self._start_event.set()
        logger.info(f"[spmd_id={self.spmd_id}] Worker {self.rank} auto-started")

    # =========================================================================
    # Op-Level Blocking
    # =========================================================================

    def block_op(self, op_name: str) -> None:
        """
        Block an operator's execution.

        When blocked, tasks of this op_name will be held in pending queue
        instead of being executed. Other ops continue normally.
        """
        self._op_runnable[op_name].clear()
        logger.info(f"[spmd_id={self.spmd_id}] Worker {self.rank}: {op_name} blocked")

    async def unblock_op(self, op_name: str) -> None:
        """
        Unblock an operator and re-enqueue pending tasks.
        """
        self._op_runnable[op_name].set()

        # Re-enqueue pending tasks
        async with self._pending_lock:
            for step in self._pending_blocked[op_name]:
                await self._enqueue(op_name, step)
            self._pending_blocked[op_name].clear()

        logger.info(f"[spmd_id={self.spmd_id}] Worker {self.rank}: {op_name} unblocked")

    def is_op_blocked(self, op_name: str) -> bool:
        """Check if an operator is blocked."""
        return not self._op_runnable[op_name].is_set()

    # =========================================================================
    # Data Reception
    # =========================================================================

    async def receive_states(
        self, op_name: str, step: int, names: List[str], values: List[Any]
    ) -> None:
        """
        Receive state data from Driver (control plane).

        Args:
            op_name: Operator name
            step: Step number
            names: State variable names
            values: State data values
        """
        # Normal mode: step-ordered execution
        exec_step = self._current_steps.get(op_name, 0)

        # Drop stale data (data for steps already executed)
        # Note: periodic ops don't have stale steps as they use virtual sequence
        is_periodic = (
            self._op_manager.is_periodic_trigger(op_name) if self._op_manager else False
        )
        if not is_periodic and step < exec_step:
            logger.warning(
                f"[spmd_id={self.spmd_id}] Dropping stale state data: op={op_name}, names={names}, "
                f"step={step} < exec_step={exec_step}"
            )
            return

        # Lazily declare the step barrier on first data arrival
        self._ensure_step_declared(op_name, step)

        # Cache in mailbox — barrier fires on_ready automatically
        await self._mailbox.deposit_states(
            op_name, step=step, names=names, values=values
        )
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} received states: op={op_name}, names={names}, step={step}"
        )

    async def receive_shards(
        self,
        op_name: str,
        name: str,
        step: int,
        sender_rank: int,
        payloads: List[Any],
        counts: List[int],
        is_ragged: bool = False,
        has_indices: bool = False,
    ) -> None:
        """
        Receive tensor shard data from upstream worker (data plane).

        Args:
            op_name: Operator name
            name: Tensor variable name
            step: Step number
            sender_rank: Sender's rank
            payloads: Raw data arrays for the shards
            counts: Number of values for each payload
            is_ragged: Indicates if ragged lengths are included
            has_indices: Indicates if routing indices are included
        """
        # Normal mode: step-ordered execution
        exec_step = self._current_steps.get(op_name, 0)

        # Drop stale data (data for steps already executed)
        # Note: periodic ops don't have stale steps as they use virtual sequence
        is_periodic = (
            self._op_manager.is_periodic_trigger(op_name) if self._op_manager else False
        )
        if not is_periodic and step < exec_step:
            logger.warning(
                f"[spmd_id={self.spmd_id}] Dropping stale tensor data: op={op_name}, name={name}, "
                f"step={step} < exec_step={exec_step}, rank={sender_rank}"
            )
            return

        # Lazily declare the step barrier on first data arrival
        self._ensure_step_declared(op_name, step)

        # Cache in mailbox — barrier fires on_ready automatically
        await self._mailbox.deposit_shards(
            op_name, name, step, sender_rank, payloads, counts, is_ragged, has_indices
        )
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} received shards: op={op_name}, name={name}, "
            f"step={step}, sender_rank={sender_rank}"
        )

    def _ensure_step_declared(self, op_name: str, step: int) -> None:
        """
        Lazily register a StepBarrier for (op_name, step) on first data arrival.

        Expected counts are read from OperatorManager so the barrier knows
        exactly how many shards/states to wait for before firing.
        """
        if self._mailbox.is_step_ready(op_name, step):
            return  # Already done (or re-fired from a previous run)
        if self._barriers_declared.get((op_name, step)):
            return  # Already declared

        expected_tensors = (
            self._op_manager.get_expected_tensor_shards(op_name)
            if self._op_manager
            else {}
        )
        expected_states = (
            self._op_manager.get_expected_state_names(op_name)
            if self._op_manager
            else []
        )
        meta = (
            self._op_manager.get_operator_metadata(op_name)
            if self._op_manager
            else None
        )
        self._mailbox.declare_step(
            op_name,
            step,
            expected_tensors,
            expected_states,
            barrier_config=meta.barrier_config if meta else None,
        )
        self._barriers_declared[(op_name, step)] = True

    def _on_step_ready(self, op_name: str, step: int) -> None:
        """
        Called by WorkerMailbox when a step's StepBarrier fires.

        Schedules _enqueue as an asyncio task so the lock in deposit_shards/
        deposit_states can return immediately without blocking on queue insertion.
        """
        logger.info(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Step ready: op={op_name}, step={step}"
        )
        asyncio.create_task(self._enqueue(op_name, step))

    async def _enqueue(self, op_name: str, step: int) -> None:
        """Enqueue a task for execution."""
        from ...common.types.configs import OperatorMeta

        meta = self._op_manager._get_operator_metadata(op_name)
        priority = meta.priority if isinstance(meta, OperatorMeta) else 0
        await self._execution_queue.put((-priority, step, op_name))
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Task enqueued: op={op_name}, step={step}"
        )

    # =========================================================================
    # Execution Logic
    # =========================================================================

    def _start_run_loop(self) -> None:
        """
        Start the run loop as an asyncio task.

        The loop will block until start() is called.
        """
        # Create and schedule the run task
        # Note: In Ray async actors, we schedule the task to run in the actor's event loop
        self._run_task = asyncio.get_event_loop().create_task(self._run())
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} run loop task started (waiting for start signal)"
        )

    async def _run(self) -> None:
        """
        Internal main execution loop (async).

        Blocks on PriorityQueue until tasks are available.
        Respects op-level blocking.

        For Reader nodes, runs a different loop that actively reads data.
        """
        # Wait for start signal from driver
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} waiting for start signal..."
        )
        await self._start_event.wait()

        logger.info(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} execution loop started"
        )

        # Debug: log node_type value
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} node_type={self.node_type} "
            f"(type={type(self.node_type).__name__}), "
            f"NodeType.READER={NodeType.READER} (type={type(NodeType.READER).__name__}), "
            f"comparison={self.node_type == NodeType.READER}, "
            f"str_comparison={str(self.node_type) == str(NodeType.READER)}, "
            f"value_comparison={self.node_type.value == NodeType.READER.value if hasattr(self.node_type, 'value') else 'N/A'}"
        )

        # Reader mode: run reader loop
        # Use both enum comparison and string comparison for safety
        is_reader = (
            self.node_type == NodeType.READER
            or str(self.node_type) == str(NodeType.READER)
            or (
                hasattr(self.node_type, "value")
                and self.node_type.value == NodeType.READER.value
            )
        )

        if is_reader:
            logger.debug(
                f"[spmd_id={self.spmd_id}] Worker {self.rank} entering READER mode, calling _reader_loop()"
            )
            await self._reader_loop()
            logger.debug(
                f"[spmd_id={self.spmd_id}] Worker {self.rank} _reader_loop() returned"
            )
            return
        else:
            logger.debug(
                f"[spmd_id={self.spmd_id}] Worker {self.rank} NOT entering READER mode, continuing with SPMD mode"
            )

        # Normal SPMD mode: data-driven execution loop
        while self._running:
            try:
                # Blocking wait for next task (async)
                priority, step, op_name = await self._execution_queue.get()

                # Check if we should stop
                if not self._running:
                    self._execution_queue.task_done()
                    break

                # Check if this op is blocked
                if not self._op_runnable[op_name].is_set():
                    # Hold in pending queue instead of re-enqueueing
                    async with self._pending_lock:
                        self._pending_blocked[op_name].append(step)
                    logger.debug(
                        f"[spmd_id={self.spmd_id}] Task held (op blocked): op={op_name}, step={step}"
                    )
                    continue

                # Check if this op is in periodic_trigger mode
                is_periodic = (
                    self._op_manager.is_periodic_trigger(op_name)
                    if self._op_manager
                    else False
                )

                if not is_periodic:
                    # Normal mode: verify step matches current expected execution step
                    exec_step = self._current_steps[op_name]
                    if step < exec_step:
                        # Truly stale task (already executed), drop it
                        logger.warning(
                            f"[spmd_id={self.spmd_id}] Stale task dropped: op={op_name}, "
                            f"task_step={step}, exec_step={exec_step}"
                        )
                        continue
                    elif step > exec_step:
                        # Future task (arrived early), hold for later execution
                        async with self._pending_lock:
                            self._pending_future_steps[op_name].append(step)
                        logger.debug(
                            f"[spmd_id={self.spmd_id}] Future task held: op={op_name}, "
                            f"task_step={step}, exec_step={exec_step}"
                        )
                        continue

                # Execute (for periodic mode, execute any task that comes)
                await self._execute_operator(op_name, step)

                # Advance execution step counter
                await self._advance_step(op_name)

            except asyncio.CancelledError:
                logger.info(
                    f"[spmd_id={self.spmd_id}] Worker {self.rank} run loop cancelled"
                )
                break
            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Error in execution loop: {e}"
                )

    async def _reader_loop(self) -> None:
        """
        Reader data reading loop (async).

        Actively reads data from assigned files and sends to downstream workers.
        Respects backpressure by pausing when signaled.
        """
        logger.debug(
            f"[spmd_id={self.spmd_id}] Reader worker {self.rank} _reader_loop started"
        )

        if self.reader_task is None:
            logger.warning(
                f"[spmd_id={self.spmd_id}] Reader worker {self.rank} has no file task, exiting"
            )
            self._reader_completed = True
            return

        if self.reader_config is None:
            logger.error(
                f"[spmd_id={self.spmd_id}] Reader worker {self.rank} has no reader config"
            )
            self._reader_completed = True
            return

        from ...reader.datasource import create_datasource

        try:
            logger.debug(
                f"[spmd_id={self.spmd_id}] Reader worker {self.rank} creating datasource: "
                f"data_type={self.reader_config.data_type}, src_dir={self.reader_config.src_dir}, "
                f"file_pattern={self.reader_config.file_pattern}"
            )

            # Create datasource with tensor type conversion config
            # tensor_name is not passed to datasource, it's only used when sending data
            datasource = create_datasource(
                data_type=self.reader_config.data_type,
                src_dir=self.reader_config.src_dir,
                file_pattern=self.reader_config.file_pattern,
                encoding=self.reader_config.encoding,
                tensor_type=self.reader_config.tensor_type,
            )

            logger.debug(
                f"[spmd_id={self.spmd_id}] Reader worker {self.rank} datasource created successfully"
            )

            batch_size = self.reader_config.batch_size
            step = 0

            # Check if reader_task has files attribute
            if not hasattr(self.reader_task, "files"):
                logger.error(
                    f"[spmd_id={self.spmd_id}] Reader worker {self.rank} reader_task has no 'files' attribute"
                )
                self._reader_completed = True
                return

            logger.info(
                f"[spmd_id={self.spmd_id}] Reader worker {self.rank} starting to read {len(self.reader_task.files)} files"
            )

            # Read and send batches
            for batch in datasource.read_batch(self.reader_task, batch_size):
                if not self._running:
                    break

                # Wait for backpressure to clear (async)
                while self._is_reading_paused and self._running:
                    logger.debug(
                        f"[spmd_id={self.spmd_id}] Reader worker {self.rank} paused due to backpressure"
                    )
                    try:
                        await asyncio.wait_for(
                            self._reader_pause_event.wait(), timeout=1.0
                        )
                        if self._reader_pause_event.is_set():
                            break
                    except asyncio.TimeoutError:
                        continue

                if not self._running:
                    break

                # Send batch to downstream workers
                await self._send_reader_batch(batch, step)
                step += 1
                self._batches_sent += 1

                logger.debug(
                    f"[spmd_id={self.spmd_id}] Reader worker {self.rank} sent batch {step}"
                )

            logger.info(
                f"[spmd_id={self.spmd_id}] Reader worker {self.rank} completed reading, "
                f"sent {self._batches_sent} batches"
            )
            self._reader_completed = True

        except Exception as e:
            logger.error(
                f"[spmd_id={self.spmd_id}] Reader worker {self.rank} error: {e}"
            )
            self._last_error = str(e)
            self._reader_completed = True

    async def _send_reader_batch(self, batch: Any, step: int) -> None:
        """
        Send a batch of data to downstream SPMD workers.

        For periodic mode (reader's downstream), there should be exactly one tensor arg.
        The batch is already converted to the target type by datasource.

        Args:
            batch: Batch of data to send (already converted by datasource, e.g., np.ndarray)
            step: Current step number
        """
        if not self._successor_handlers:
            logger.warning(
                f"[spmd_id={self.spmd_id}] Reader worker {self.rank} has no successor handlers"
            )
            return

        # Get tensor name from reader_config (parsed during build)
        if not self.reader_config or not self.reader_config.tensor_name:
            raise ValueError(
                f"[spmd_id={self.spmd_id}] Reader worker {self.rank} has no tensor_name configured. "
                f"Reader must have output_data_type specified when building the graph."
            )
        tensor_name = self.reader_config.tensor_name

        # Batch is already converted by datasource to the target type (e.g., np.ndarray)
        data_to_send = batch

        # Wrap in TransportTensor
        transport_tensor = TransportTensor.from_tensor(data_to_send)

        # Send to all downstream SPMDs (round-robin to workers within each SPMD)
        for spmd_id, handlers in self._successor_handlers.items():
            if not handlers:
                continue

            # Determine target worker (round-robin based on step)
            target_ranks = list(handlers.keys())
            target_rank = target_ranks[step % len(target_ranks)]
            handler = handlers[target_rank]

            try:
                # Send as target op data
                await handler.receive_shards(
                    self.reader_config.target_op,
                    tensor_name,
                    step,
                    self.rank,
                    [transport_tensor.flat_values],
                    [int(transport_tensor.shape[0])],
                    transport_tensor.is_ragged,
                    transport_tensor.global_indices is not None,
                )
            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Failed to send batch to {spmd_id}:{target_rank}: {e}"
                )

    async def stop(self) -> None:
        """Stop the execution loop gracefully."""
        self._running = False
        # Put a sentinel to unblock the queue
        await self._execution_queue.put((999, 0, "__stop__"))

        # Cancel tasks
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

        logger.info(f"[spmd_id={self.spmd_id}] Worker {self.rank} stopping")

    async def _check_all_empty(self, op_name: str, step: int) -> bool:
        """Check if all received shards for a step are empty packets."""
        expected_tensor_names = self._op_manager.get_expected_tensor_names(op_name)
        if not expected_tensor_names:
            return False

        all_tensor_shards = await self._mailbox.get_all_tensor_shards(op_name, step)

        for tensor_name in expected_tensor_names:
            if tensor_name not in all_tensor_shards:
                return False
            shards = all_tensor_shards[tensor_name]
            if not shards:
                return False
            for rank, tensor in shards.items():
                backend = get_backend()
                shape = backend.shape(tensor.flat_values)
                if shape and shape[0] > 0:
                    return False
        return True

    async def _propagate_empty_packets(self, op_name: str, step: int) -> None:
        """Propagate empty packets to successor SPMDs to make sure the transmission steps are complete in all workers."""
        if self._op_manager is None:
            return

        output_sharding = self._op_manager.get_output_sharding(op_name)
        successor_ids = self._op_manager.get_successor_ids(op_name)
        if not successor_ids:
            return

        is_periodic = self._op_manager.is_periodic_trigger(op_name)
        current_step = self._current_steps[op_name]
        output_step = self.compute_output_step(
            is_periodic=is_periodic, current_step=current_step, input_step=step
        )

        for spmd_id in successor_ids:
            succ_info = self._op_manager.get_successor_info(op_name, spmd_id)
            if succ_info is None:
                continue

            target_op = succ_info.target_op
            for tensor_name in output_sharding:
                # Create an empty TransportTensor
                backend = get_backend()
                empty_tensor = TransportTensor.from_tensor(
                    backend.zeros((0,), dtype=backend.float32())
                )

                if succ_info.is_self_loop:
                    await self._self_receive_tensor(
                        target_op, tensor_name, output_step, empty_tensor
                    )
                else:
                    handlers = self._successor_handlers.get(spmd_id, {})
                    for target_rank in range(succ_info.total_rank):
                        handler = handlers.get(target_rank)
                        if handler is not None:
                            await handler.receive_shards(
                                target_op,
                                tensor_name,
                                output_step,
                                self.rank,
                                [empty_tensor.flat_values],
                                [0],
                                empty_tensor.is_ragged,
                                empty_tensor.global_indices is not None,
                            )

    async def _execute_operator(self, op_name: str, step: int) -> None:
        """Execute an operator: assemble data, execute, send outputs."""
        try:
            # --- Early check: all received shards are empty packets ---
            all_empty = await self._check_all_empty(op_name, step)
            if all_empty:
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] All received shards are empty for "
                    f"op={op_name}, step={step}. Propagating per sparse routing rules."
                )
                await self._propagate_empty_packets(op_name, step)
                await self._mailbox.clear_step(op_name, step)
                return

            # Assemble input data
            input_data = await self._assemble_input(op_name, step)

            # Execute operator
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Executing: op={op_name}, step={step}"
            )
            result = self._op_manager.execute(op_name, input_data)

            if result is None:
                logger.warning(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Operator {op_name} returned None, skipping output send"
                )
                return

            # Determine output step
            is_periodic = (
                self._op_manager.is_periodic_trigger(op_name)
                if self._op_manager
                else False
            )
            current_step = self._current_steps[op_name]
            output_step = self._resolve_output_step(
                is_periodic=is_periodic, current_step=current_step, input_step=step
            )
            if is_periodic:
                logger.debug(
                    f"[spmd_id={self.spmd_id}] periodic_trigger=True: input_step={step}, current_step={current_step}, "
                    f"output_step={output_step}, rank={self.rank}, total_rank={self._total_rank}, "
                    f"use_global_step_indexing={getattr(self.config, 'use_global_step_indexing', False)}"
                )

            # Send output
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] About to send output: op={op_name}, "
                f"output_step={output_step}, result_type={type(result).__name__}"
            )
            await self._send_output(op_name, output_step, result, exec_step=step)

            # Clear mailbox
            await self._mailbox.clear_step(op_name, step)

        except Exception as e:
            logger.error(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Operator {op_name} execution failed: {e}"
            )
            raise

    async def _assemble_input(self, op_name: str, step: int) -> Any:
        """
        Assemble input data from mailbox and construct data_type object.

        This method completes 5 tasks:
        1. Get static_args from op_manager
        2. Get state_args from mailbox based on expected_state_names
        3. Assemble tensor_args from mailbox based on expected_tensor_names and sharding
        4. Assemble tensor meta
        5. Construct the final data_type object expected by the operator

        Args:
            op_name: Operator name
            step: Execution step

        Returns:
            Constructed data_type object ready for operator execution
        """
        # Task 1: Get static arguments
        static_args = self._op_manager.get_static_args()

        # Task 2: Get state arguments
        expected_state_names = self._op_manager.get_expected_state_names(op_name)
        all_states = await self._mailbox.get_all_states(op_name, step)
        state_args = {
            name: all_states[name]
            for name in expected_state_names
            if name in all_states
        }

        # Task 3 & 4: Assemble tensor arguments and meta
        expected_tensor_names = self._op_manager.get_expected_tensor_names(op_name)
        all_tensor_shards = await self._mailbox.get_all_tensor_shards(op_name, step)

        # Check if periodic_trigger mode
        is_periodic = self._op_manager.is_periodic_trigger(op_name)

        tensor_args = {}
        for tensor_name in expected_tensor_names:
            if tensor_name not in all_tensor_shards:
                raise ValueError(
                    f"Missing tensor '{tensor_name}' for operator {op_name} at step {step}"
                )

            shards = all_tensor_shards[tensor_name]

            if is_periodic:
                # Periodic mode: use first shard directly, no assembly needed
                first_rank = next(iter(shards.keys()))
                tensor_args[tensor_name] = shards[first_rank]
            else:
                # Normal mode: assemble tensor based on sharding
                assembled_tensor = await self._assemble_tensor(
                    op_name, tensor_name, shards
                )
                tensor_args[tensor_name] = assembled_tensor

        # Cache assembled tensors (they carry their own indices/meta)
        self._last_assembled_tensors[op_name] = tensor_args

        for tn, tensor in tensor_args.items():
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] _assemble_input cached tensor: op={op_name}, step={step}, "
                f"tensor={tn}, has_indices={tensor.global_indices is not None}"
            )

        # Task 5: Construct final data_type object
        input_type = self._op_manager.get_operator_input_type(op_name)
        if input_type is None:
            raise ValueError(
                f"Cannot determine input type for operator {op_name}. "
                f"Make sure the operator function has type annotations."
            )

        # Combine all arguments
        kwargs = {}
        kwargs.update(static_args)
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Update to kwargs: static arguments: {static_args}"
        )
        kwargs.update(state_args)
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Update to kwargs: state arguments: {state_args}"
        )
        kwargs.update(tensor_args)
        _tensor_log_str = f"[spmd_id={self.spmd_id} rank={self.rank}] Update to kwargs: tensor arguments: {tensor_args}"
        logger.debug(
            _tensor_log_str[:100] + "..."
            if len(_tensor_log_str) > 100
            else _tensor_log_str
        )

        # Construct the data_type object
        input_obj = input_type(**kwargs)
        _input_log_str = f"[spmd_id={self.spmd_id} rank={self.rank}] Constructed input object: {input_obj}"
        logger.debug(
            _input_log_str[:1000] + "..."
            if len(_input_log_str) > 1000
            else _input_log_str
        )

        return input_obj

    async def _assemble_tensor(
        self, op_name: str, tensor_name: str, shards: Dict[int, TransportTensor]
    ) -> TransportTensor:
        """Assemble tensor shards into a complete TransportTensor."""
        input_sharding = self._op_manager.get_input_sharding(op_name)
        input_spec = input_sharding.get(tensor_name)

        if input_spec is None:
            # Fallback: merge all shards
            return self._assemble_to_dynamic_tensor(tensor_name, shards)

        if not input_spec.is_content_addressable:
            # Check for dynamic predecessors
            predecessor_topology = self._op_manager.get_predecessor_topology(op_name)
            has_dynamic_predecessor = any(
                p.tensor_sharding.get(tensor_name).is_content_addressable
                if p.tensor_sharding.get(tensor_name)
                else False
                for p in predecessor_topology.values()
            )

            if has_dynamic_predecessor:
                return self._assemble_dynamic_to_static_tensor(
                    tensor_name, shards, input_spec
                )
            else:
                return await self._assemble_static_to_static_tensor(
                    op_name, tensor_name, shards, input_spec, predecessor_topology
                )
        else:
            return self._assemble_to_dynamic_tensor(tensor_name, shards)

    async def _assemble_static_to_static_tensor(
        self,
        op_name: str,
        tensor_name: str,
        shards: Dict[int, TransportTensor],
        input_spec: ShardingSpec,
        predecessor_topology: Dict[str, Any],
    ) -> TransportTensor:
        """Assemble static-to-static tensor using shards."""
        backend = get_backend()
        shard_list = []
        for pre_spmd_id, pre_info in predecessor_topology.items():
            strategy = self._op_manager.get_input_routing_strategy(
                op_name, pre_spmd_id, tensor_name
            )
            if strategy:
                for rank in strategy.dependent_ranks:
                    if rank in shards:
                        shard_list.append(shards[rank])
            else:
                for rank in range(pre_info.total_rank):
                    if rank in shards:
                        shard_list.append(shards[rank])

        if not shard_list:
            raise ValueError(f"No shards for {tensor_name}")

        if len(shard_list) == 1:
            return shard_list[0]

        # Concatenate shards using backend
        flat_values = [s.flat_values for s in shard_list]
        merged_values = backend.cat(flat_values, dim=0)

        merged_indices = None
        if all(s.global_indices is not None for s in shard_list):
            merged_indices = backend.cat([s.global_indices for s in shard_list], dim=0)

        # Propagate global_shape from shards
        global_shape = shard_list[0].global_shape

        return TransportTensor(
            merged_values, indices=merged_indices, global_shape=global_shape
        )

    def _assemble_dynamic_to_static_tensor(
        self,
        tensor_name: str,
        shards: Dict[int, TransportTensor],
        input_spec: ShardingSpec,
    ) -> TransportTensor:
        """Assemble dynamic-to-static (Sparse-to-Dense) tensor using global_indices."""
        backend = get_backend()

        # Merge all non-empty shards
        valid_shards = [
            s for s in shards.values() if backend.shape(s.flat_values)[0] > 0
        ]
        if not valid_shards:
            return TransportTensor.from_tensor(
                backend.zeros((0,), dtype=backend.float32())
            )

        # Simple 1D path for now (Sparse-to-Dense typically assumes 1D indices for scattering)
        all_values = backend.cat([s.flat_values for s in valid_shards], dim=0)
        # Materialize indices if needed
        all_indices_list = []
        for s in valid_shards:
            idx = s.global_indices
            if idx is None:
                # Should not happen for sparse, fallback to arange if absolutely needed
                num = backend.shape(s.flat_values)[0]
                idx = backend.arange(0, num, dtype=backend.int32())
            all_indices_list.append(idx)
        all_indices = backend.cat(all_indices_list, dim=0)

        # Infer global length to compute local range
        max_idx = int(backend.item(backend.max(all_indices)))
        global_shape = (max_idx + 1,)

        internal_layout = input_spec.lower(global_shape, self.rank)
        local_shape = internal_layout.get_local_shape()
        dense_slices = internal_layout.dense_slices

        slice_start = dense_slices[0].start
        slice_end = dense_slices[0].end

        mask = (all_indices >= slice_start) & (all_indices < slice_end)
        local_indices = all_indices[mask] - slice_start
        local_values = all_values[mask]

        assembled_data = backend.zeros(local_shape, dtype=all_values.dtype)
        # Reconstruct (scatter)
        # Backend should support indexed assignment or scatter
        # For simplicity, use a generic scatter-like logic
        # assembled_data[local_indices] = local_values
        # Note: index_update is a common functional pattern
        try:
            assembled_data = backend.index_update(
                assembled_data, local_indices, local_values
            )
        except AttributeError:
            # Fallback to numpy if backend lacks index_update (periodic/fallback)
            arr = backend.to_numpy(assembled_data)
            arr[backend.to_numpy(local_indices)] = backend.to_numpy(local_values)
            assembled_data = backend.tensor(arr)

        return TransportTensor.from_tensor(assembled_data, global_shape=global_shape)

    def _assemble_to_dynamic_tensor(
        self, tensor_name: str, shards: Dict[int, TransportTensor]
    ) -> TransportTensor:
        """Assemble to-dynamic (Any -> Sparse) tensor by merging shards."""
        backend = get_backend()
        shard_list = sorted(shards.items())
        valid_tensors = [
            t for r, t in shard_list if backend.shape(t.flat_values)[0] > 0
        ]

        if not valid_tensors:
            return TransportTensor.from_tensor(
                backend.zeros((0,), dtype=backend.float32())
            )

        merged_values = backend.cat([t.flat_values for t in valid_tensors], dim=0)
        merged_indices = None
        if all(t.global_indices is not None for t in valid_tensors):
            merged_indices = backend.cat(
                [t.global_indices for t in valid_tensors], dim=0
            )

        return TransportTensor(merged_values, indices=merged_indices)

    async def _advance_step(self, op_name: str) -> None:
        """Advance execution step counter for an operator."""
        self._current_steps[op_name] += 1
        new_step = self._current_steps[op_name]

        # Report op completion to driver
        self._report_op_completed(op_name, new_step)

        # Check if this op is in periodic_trigger mode
        is_periodic = (
            self._op_manager.is_periodic_trigger(op_name) if self._op_manager else False
        )

        if not is_periodic:
            # Normal mode: re-enqueue pending future tasks that match the new step
            async with self._pending_lock:
                pending = self._pending_future_steps[op_name]
                to_enqueue = [s for s in pending if s == new_step]
                self._pending_future_steps[op_name] = [
                    s for s in pending if s != new_step
                ]

            for s in to_enqueue:
                await self._enqueue(op_name, s)

    def _report_op_completed(self, op_name: str, step: int) -> None:
        """
        Report op completion to driver.

        Args:
            op_name: Operation name
            step: Completed step number
        """
        if self._driver_handle is not None:
            try:
                self._driver_handle.report_worker_op_completed.remote(
                    self.rank, op_name, step
                )
            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Failed to report op completion: {e}"
                )

    # =========================================================================
    # Data Sending
    # =========================================================================

    def _should_propagate_indices(
        self, op_name: str, spmd_id: str, tensor_name: str
    ) -> bool:
        """
        Determines whether global indices need to be propagated to the target SPMD.

        If the target relies on content-addressable routing (e.g., Hash, Dynamic)
        or explicit sparse point-to-point routing, it needs the indices.
        If it uses simple Dense Chunk routing, it does not.
        """
        succ_info = self._op_manager.get_successor_info(op_name, spmd_id)
        if succ_info is None or succ_info.tensor_sharding is None:
            return False

        target_spec = succ_info.tensor_sharding.get(tensor_name)
        if target_spec is None:
            return False

        # ShardingSpecs like HASH and DYNAMIC require explicit indices
        partition_type_name = getattr(
            getattr(target_spec, "partition_type", None), "name", ""
        )
        if partition_type_name in ("HASH", "DYNAMIC"):
            return True

        return False

    async def _send_output(
        self, op_name: str, step: int, result: Any, *, exec_step: Optional[int] = None
    ) -> None:
        """Send output to successor SPMDs using TransportTensor."""
        if exec_step is None:
            exec_step = step

        output_sharding = self._op_manager.get_output_sharding(op_name)
        tensors_to_send = self._extract_tensors(result, output_sharding)
        successor_ids = self._op_manager.get_successor_ids(op_name)
        input_tensors = self._last_assembled_tensors.get(op_name, {})

        for spmd_id in successor_ids:
            succ_info = self._op_manager.get_successor_info(op_name, spmd_id)
            if succ_info is None:
                continue

            target_op = succ_info.target_op

            for tensor_name, tensor_data in tensors_to_send.items():
                # 1. Check if we need to propagate indices to this specific target
                propagate_indices = self._should_propagate_indices(
                    op_name, spmd_id, tensor_name
                )

                # 2. Extract indices if pass-through is possible
                input_tensor = input_tensors.get(tensor_name)
                indices = None

                if input_tensor is not None:
                    if propagate_indices:
                        indices = input_tensor.global_indices
                    else:
                        # Optimization: zero-cost descriptor instead of full tensor
                        indices = (
                            input_tensor.index_descriptor
                            if input_tensor.index_descriptor is not None
                            else input_tensor.global_indices
                        )

                # Wrap in TransportTensor
                transport_tensor = TransportTensor.from_tensor(
                    tensor_data, indices=indices
                )

                if succ_info.is_self_loop:
                    await self._self_receive_tensor(
                        target_op, tensor_name, step, transport_tensor
                    )
                    continue

                handlers = self._successor_handlers.get(spmd_id, {})
                if not handlers:
                    continue

                from fluxway.transport.communication import AsyncDataDistributor

                distributor = AsyncDataDistributor(self.rank)

                source_spec = output_sharding.get(tensor_name)
                if source_spec is None:
                    # Default: replicated on self mesh
                    source_mesh = self._mesh
                    if source_mesh is None:
                        source_mesh = DeviceMesh(
                            device_shape=(1,), axis_names=("data",)
                        )
                    source_spec = ShardingSpec.chunk(source_mesh, PartitionSpec())

                target_spec = succ_info.tensor_sharding.get(tensor_name)
                if target_spec is None:
                    # Default: replicated on target mesh (or self mesh if target mesh unknown)
                    target_mesh = (
                        succ_info.mesh if succ_info.mesh is not None else self._mesh
                    )
                    if target_mesh is None:
                        target_mesh = DeviceMesh(
                            device_shape=(1,), axis_names=("data",)
                        )
                    target_spec = ShardingSpec.chunk(target_mesh, PartitionSpec())

                async def _send_fn(
                    target_rank: int,
                    tensors: list,
                    counts: list,
                    *,
                    _handlers=handlers,
                    _target_op=target_op,
                    _tensor_name=tensor_name,
                    _step=step,
                    _is_ragged=transport_tensor.is_ragged,
                    _has_indices=transport_tensor.global_indices is not None,
                ) -> bool:
                    handler = _handlers.get(target_rank)
                    if handler is None:
                        return False

                    await self._send_semaphore.acquire()

                    count = int(counts[0]) if counts else 0
                    if count == 0:
                        backend = get_backend()
                        empty_tensor = backend.zeros((0,), dtype=backend.float32())
                        ref_task = handler.receive_shards(
                            _target_op,
                            _tensor_name,
                            _step,
                            self.rank,
                            [empty_tensor],
                            [0],
                            _is_ragged,
                            _has_indices,
                        )
                    else:
                        ref_task = handler.receive_shards(
                            _target_op,
                            _tensor_name,
                            _step,
                            self.rank,
                            tensors,
                            counts,
                            _is_ragged,
                            _has_indices,
                        )

                    async def _wait_and_release(t=ref_task):
                        try:
                            # Await the call to get the downstream Mailbox watermark
                            watermark = await t
                            if (
                                isinstance(watermark, int)
                                and watermark > self._high_watermark
                            ):
                                # Apply adaptive throttling backoff
                                await asyncio.sleep(0.05)
                        except Exception as e:
                            logger.debug(
                                f"[spmd_id={self.spmd_id}] Error tracing ACK for backpressure: {e}"
                            )
                        finally:
                            self._send_semaphore.release()

                    asyncio.create_task(_wait_and_release())
                    return True

                await distributor.distribute(
                    transport_tensor,
                    source=source_spec,
                    target=target_spec,
                    send_fn=_send_fn,
                )

        # --- Send state_argument via control plane ---
        # State arguments are sent through: rank0-worker -> driver -> controller -> target-driver -> all-workers
        output_state_names = self._op_manager.get_output_state_names(op_name)
        if output_state_names:
            state_data = self._extract_states(result, output_state_names)

            for spmd_id in successor_ids:
                succ_info = self._op_manager.get_successor_info(op_name, spmd_id)
                if succ_info is None:
                    continue

                # Check condition
                if succ_info.condition_fn is not None:
                    if not succ_info.condition_fn(result):
                        continue

                target_op = succ_info.target_op

                # Collect all state names and values for this target
                names_to_send = []
                values_to_send = []
                for state_name, state_value in state_data.items():
                    if succ_info.is_self_loop:
                        # Self-loop: directly set as static arg
                        await self._self_receive_state(
                            target_op, state_name, step, state_value
                        )
                    elif self.rank == 0:
                        names_to_send.append(state_name)
                        values_to_send.append(state_value)

                if names_to_send:
                    logger.debug(
                        f"[spmd_id={self.spmd_id}] Worker {self.rank} sending state_arguments: names={names_to_send}, "
                        f"step={step} to {spmd_id}"
                    )
                    if self._driver_handle is not None:
                        try:
                            self._driver_handle.routing_state_argument_to_controller.remote(
                                spmd_id, target_op, names_to_send, step, values_to_send
                            )
                        except Exception as e:
                            logger.error(
                                f"[spmd_id={self.spmd_id}] Worker {self.rank} failed to send state_arguments "
                                f"via driver: {e}"
                            )

    def _extract_tensors(
        self, data: Any, output_sharding: Dict[str, ShardingSpec]
    ) -> Dict[str, Any]:
        """Extract named tensors from output data."""
        tensors: Dict[str, Any] = {}

        if isinstance(data, dict):
            for name in output_sharding.keys():
                if name in data:
                    tensors[name] = data[name]
        elif hasattr(data, "__dict__"):
            for name in output_sharding.keys():
                if hasattr(data, name):
                    tensors[name] = getattr(data, name)

        # If no specific tensors found, treat entire data as single tensor
        if not tensors and output_sharding:
            first_name = next(iter(output_sharding.keys()))
            tensors[first_name] = data

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] _extract_tensors: output_sharding keys={list(output_sharding.keys())}, "
            f"extracted tensors={list(tensors.keys())}, data_type={type(data).__name__}"
        )

        return tensors

    # =========================================================================
    # Backward Reverse Routing (using forward shard origins)
    # =========================================================================

    async def _self_receive_tensor(
        self, target_op: str, name: str, step: int, tensor: TransportTensor
    ) -> None:
        """Self-loop data passing - directly call data receive interface."""
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Self-loop data passing: {name} step={step} "
            f"-> {target_op}"
        )
        await self.receive_shards(
            target_op,
            name,
            step,
            self.rank,
            [tensor.flat_values],
            [int(tensor.shape[0])],
            tensor.is_ragged,
            tensor.global_indices is not None,
        )

    async def _self_receive_state(
        self, target_op: str, name: str, step: int, data: Any
    ) -> None:
        """
        Self-loop state passing - directly call data receive interface.

        For same-node edges (e.g., forward -> backward), this method
        maintains the data-driven execution model by calling receive_states
        on self rather than sending via control plane.
        """
        # Directly call receive_states to trigger target op's data readiness check
        await self.receive_states(
            op_name=target_op, step=step, names=[name], values=[data]
        )

    def _extract_states(self, data: Any, state_names: List[str]) -> Dict[str, Any]:
        """
        Extract named state values from output data.

        Args:
            data: Output data from operator execution
            state_names: List of state variable names to extract

        Returns:
            Dictionary of {state_name: state_value}
        """
        states: Dict[str, Any] = {}

        if isinstance(data, dict):
            for name in state_names:
                if name in data:
                    states[name] = data[name]
        elif hasattr(data, "__dict__"):
            for name in state_names:
                if hasattr(data, name):
                    states[name] = getattr(data, name)

        return states

    # =========================================================================
    # Static Arguments
    # =========================================================================

    def set_static_arg(self, name: str, value: Any) -> None:
        """Set a static argument value."""
        self._op_manager.set_static_arg(name, value)

    def get_static_arg(self, name: str) -> Optional[Any]:
        """Get a static argument value."""
        return self._op_manager.get_static_arg(name)

    # =========================================================================
    # State Argument Handling
    # =========================================================================

    def receive_state_argument(self, state_name: str, state_value: Any) -> None:
        """
        Receive state_argument from driver (broadcast from Controller).

        State arguments are set as static args for operator execution.

        Args:
            state_name: State variable name
            state_value: State value
        """
        self.set_static_arg(state_name, state_value)
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} received state_argument: {state_name}"
        )

    def send_state_argument(
        self,
        target_spmd_id: str,
        target_op: str,
        state_name: str,
        step: int,
        state_value: Any,
    ) -> None:
        """
        Send state_argument to target SPMD.

        Only rank 0 worker should call this method.
        Path: rank0-worker -> driver -> controller -> target-driver -> all-workers

        Args:
            target_spmd_id: Target SPMD ID
            target_op: Target operator type on the receiving side
            state_name: State variable name
            step: Current step number
            state_value: State value

        Raises:
            RuntimeError: If called from non-rank-0 worker
        """
        if self.rank != 0:
            raise RuntimeError("Only rank 0 worker can send state_argument")

        if self._driver_handle is None:
            logger.error(
                f"[spmd_id={self.spmd_id} rank={self.rank}] No driver handle set, cannot send state_argument"
            )
            return

        try:
            self._driver_handle.routing_state_argument_to_controller.remote(
                target_spmd_id, target_op, state_name, step, state_value
            )
            logger.debug(
                f"[spmd_id={self.spmd_id}] Worker {self.rank} sent state_argument '{state_name}' "
                f"to SPMD {target_spmd_id}, target_op={target_op}, step={step}"
            )
        except Exception as e:
            logger.error(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Failed to send state_argument: {e}"
            )

    # =========================================================================
    # Heartbeat
    # =========================================================================

    def _start_heartbeat(self) -> None:
        """Start heartbeat reporting to driver as an asyncio task."""
        if self._heartbeat_running:
            return

        self._heartbeat_running = True
        self._heartbeat_task = asyncio.get_event_loop().create_task(
            self._heartbeat_loop()
        )
        logger.debug(f"[spmd_id={self.spmd_id}] Worker {self.rank} heartbeat started")

    async def stop_heartbeat(self) -> None:
        """Stop heartbeat reporting."""
        self._heartbeat_running = False
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Heartbeat reporting loop (async)."""
        while self._heartbeat_running:
            try:
                await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Error sending heartbeat: {e}"
                )

            await asyncio.sleep(self._heartbeat_interval)

    async def _send_heartbeat(self) -> None:
        """Send heartbeat to driver."""
        if self._driver_handle is None:
            return

        # Determine status
        if self._last_error:
            status = WorkerStatus.ERROR
        elif any(not event.is_set() for event in self._op_runnable.values()):
            status = WorkerStatus.BLOCKED
        elif self._running:
            status = WorkerStatus.RUNNING
        else:
            status = WorkerStatus.INITIALIZING

        # Get mailbox status for backpressure monitoring
        mailbox_status = await self._get_mailbox_status()

        heartbeat = WorkerHeartbeat(
            rank=self.rank,
            timestamp=time.time(),
            op_steps=dict(self._current_steps),
            status=status,
            error_message=self._last_error,
            telemetry=self._collect_telemetry(),
            mailbox_status=mailbox_status,
        )

        try:
            self._driver_handle.receive_worker_heartbeat.remote(self.rank, heartbeat)
            logger.debug(
                f"[spmd_id={self.spmd_id}] Worker {self.rank} sent heartbeat: status={status.name}, "
                f"op_steps={heartbeat.op_steps}, mailbox_full={mailbox_status.is_full}"
            )
        except Exception as e:
            logger.error(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Failed to send heartbeat: {e}"
            )

    async def _get_mailbox_status(self) -> MailboxStatus:
        """Get mailbox status for backpressure monitoring."""
        current_size = await self._mailbox.get_size()
        capacity = await self._mailbox.get_capacity()
        full_ratio = 0.8  # Consider full when 80% capacity reached
        is_full = current_size >= capacity * full_ratio

        return MailboxStatus(
            is_full=is_full,
            capacity=capacity,
            current_size=current_size,
            full_ratio=full_ratio,
        )

    def _collect_telemetry(self) -> Optional[WorkerTelemetry]:
        """Collect worker telemetry data."""
        # Placeholder - implement actual telemetry collection
        return WorkerTelemetry(worker_id=f"{self.spmd_id}_{self.rank}")

    # =========================================================================
    # State Management
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """Get worker state for checkpointing."""
        return {
            "spmd_id": self.spmd_id,
            "rank": self.rank,
            "current_steps": dict(self._current_steps),
            "static_args": self._op_manager.get_static_args(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore worker state from checkpoint."""
        if "current_steps" in state:
            for op_name, step in state["current_steps"].items():
                self._current_steps[op_name] = step
        if "static_args" in state:
            for name, value in state["static_args"].items():
                self._op_manager.set_static_arg(name, value)

    # =========================================================================
    # Utility
    # =========================================================================

    def get_current_step(self, op_name: str) -> int:
        """Get current step for an operator name."""
        return self._current_steps[op_name]

    def get_rank(self) -> int:
        """Get this worker's rank."""
        return self.rank

    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    # =========================================================================
    # Reader Backpressure Control
    # =========================================================================

    def pause_reading(self) -> None:
        """
        Pause data reading (for Reader nodes).

        Called by driver when downstream is busy.
        """
        if self.node_type != NodeType.READER:
            return

        self._is_reading_paused = True
        self._reader_pause_event.clear()
        logger.debug(f"[spmd_id={self.spmd_id}] Reader worker {self.rank} paused")

    def resume_reading(self) -> None:
        """
        Resume data reading (for Reader nodes).

        Called by driver when downstream backpressure is relieved.
        """
        if self.node_type != NodeType.READER:
            return

        self._is_reading_paused = False
        self._reader_pause_event.set()
        logger.debug(f"[spmd_id={self.spmd_id}] Reader worker {self.rank} resumed")

    def is_reader_completed(self) -> bool:
        """Check if reader has completed reading all data."""
        return self._reader_completed

    def get_batches_sent(self) -> int:
        """Get number of batches sent by this reader."""
        return self._batches_sent
