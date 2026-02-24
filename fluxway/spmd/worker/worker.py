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
7. Handle state_argument reception and forwarding

This implementation uses Ray's async actor API for concurrency.
See: https://docs.ray.io/en/latest/ray-core/actors/async_api.html
"""

from fluxway.common.types.sharding import ShardingSpec
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import logging
import time
import numpy as np

import ray

from ...common.types import SPMDConfig, WorkerHeartbeat, WorkerStatus, WorkerTelemetry
from ...common.types.data_types import (
    OpType,
    TensorMeta,
    NodeType,
    ReaderConfig,
    MailboxStatus,
    ShardOrigin,
    StepTensorMeta,
)
from ...common.utils import ensure_logging_configured
from .mailbox import WorkerMailbox
from .operator_manager import OperatorManager


logger = logging.getLogger(__name__)


@ray.remote
class SPMDWorker:
    """
    SPMD Worker (Ray Async Actor).

    Implements dual-stream convergence with priority queue execution:
    - Data is received by WorkerMailbox (async-safe)
    - Executes operators in priority order: update > backward > forward
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
            driver_handle: Reference to the SPMD driver for callbacks
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

        # Driver handle for callbacks
        self._driver_handle: Optional[Any] = driver_handle

        # Node type
        self.node_type = node_type
        self.reader_task = reader_task
        self.reader_config = reader_config

        # Mesh from config
        self._mesh = config.mesh
        self._total_rank = config.worker_count  # Total number of workers in this SPMD

        # Operator Manager (topology parsed from config) - only for SPMD nodes
        if node_type == NodeType.SPMD:
            self._op_manager = OperatorManager(config, rank, spmd_id)
        else:
            self._op_manager = None

        # Execution step counters {op_type: step} - tracks which step is being/was executed
        self._current_steps: Dict[OpType, int] = {
            OpType.FORWARD: 0,
            OpType.BACKWARD: 0,
            OpType.UPDATE: 0,
        }

        # Input step counters for eager mode {op_type: step}
        # Used to assign sequential step numbers to incoming data for eager_trigger operators
        # Separate from _current_steps because eager mode doesn't require step matching
        self._eager_input_steps: Dict[OpType, int] = {
            OpType.FORWARD: 0,
            OpType.BACKWARD: 0,
            OpType.UPDATE: 0,
        }

        # Op-level blocking: asyncio.Event (set=runnable, clear=blocked)
        self._op_runnable: Dict[OpType, asyncio.Event] = {
            OpType.FORWARD: asyncio.Event(),
            OpType.BACKWARD: asyncio.Event(),
            OpType.UPDATE: asyncio.Event(),
        }
        for event in self._op_runnable.values():
            event.set()  # Initially all ops are runnable

        # Pending tasks when blocked (to avoid priority queue reentry loop)
        self._pending_blocked: Dict[OpType, List[int]] = {
            OpType.FORWARD: [],
            OpType.BACKWARD: [],
            OpType.UPDATE: [],
        }

        # Pending tasks for future steps (arrived early, waiting for current step to complete)
        # Structure: {op_type: List[step]} - steps waiting to be executed
        self._pending_future_steps: Dict[OpType, List[int]] = {
            OpType.FORWARD: [],
            OpType.BACKWARD: [],
            OpType.UPDATE: [],
        }

        self._pending_lock = asyncio.Lock()

        # Priority queue: asyncio.PriorityQueue with (priority, step, op_type)
        self._execution_queue: asyncio.PriorityQueue[Tuple[int, int, OpType]] = (
            asyncio.PriorityQueue()
        )

        # Internal Mailbox (async-safe)
        self._mailbox = WorkerMailbox()

        # Handler storage (worker-level, not in OperatorManager)
        # {spmd_id: {rank: handler}}
        self._predecessor_handlers: Dict[str, Dict[int, Any]] = {}
        self._successor_handlers: Dict[str, Dict[int, Any]] = {}

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

        # Cached assembled input metas: {op_type: {tensor_name: TensorMeta}}
        # Updated by _assemble_input, used by _send_output for global_shape / global_indices
        self._last_assembled_metas: Dict[OpType, Dict[str, Optional[TensorMeta]]] = {
            OpType.FORWARD: {},
            OpType.BACKWARD: {},
            OpType.UPDATE: {},
        }

        # Per-step meta history for backward reverse routing
        # Structure: {op_type: {step: {tensor_name: StepTensorMeta}}}
        # Each StepTensorMeta preserves both the assembled global view AND
        # per-shard origin info (source_spmd_id, source_rank, global_indices,
        # row range) so that backward can reverse-route without a route_fn.
        self._meta_history: Dict[OpType, Dict[int, Dict[str, StepTensorMeta]]] = {
            OpType.FORWARD: {},
            OpType.BACKWARD: {},
            OpType.UPDATE: {},
        }

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

    def compute_output_step(
        self, *, is_eager: bool, current_step: int, input_step: int
    ) -> int:
        """
        Compute output step number based on SPMDConfig and operator mode.

        Spec:
        - enable_output_step_remap=False (default): output_step == input_step
        - enable_output_step_remap=True:
          - eager_trigger=True: output_step = total_rank * current_step + local_rank
          - eager_trigger=False: output_step = current_step
        """
        enable_output_step_remap = getattr(
            self.config, "enable_output_step_remap", False
        )
        if not enable_output_step_remap:
            return input_step
        if is_eager:
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
        self._predecessor_handlers[spmd_id] = handlers
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
        self._successor_handlers[spmd_id] = handlers
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

    def block_op(self, op_type: OpType) -> None:
        """
        Block an operator's execution.

        When blocked, tasks of this op_type will be held in pending queue
        instead of being executed. Other ops continue normally.
        """
        self._op_runnable[op_type].clear()
        logger.info(
            f"[spmd_id={self.spmd_id}] Worker {self.rank}: {op_type.name} blocked"
        )

    async def unblock_op(self, op_type: OpType) -> None:
        """
        Unblock an operator and re-enqueue pending tasks.
        """
        self._op_runnable[op_type].set()

        # Re-enqueue pending tasks
        async with self._pending_lock:
            for step in self._pending_blocked[op_type]:
                await self._enqueue(op_type, step)
            self._pending_blocked[op_type].clear()

        logger.info(
            f"[spmd_id={self.spmd_id}] Worker {self.rank}: {op_type.name} unblocked"
        )

    def is_op_blocked(self, op_type: OpType) -> bool:
        """Check if an operator is blocked."""
        return not self._op_runnable[op_type].is_set()

    # =========================================================================
    # Data Reception
    # =========================================================================

    async def receive_state_data(
        self, op_type: OpType, name: str, step: int, data: Any
    ) -> None:
        """
        Receive state data from Driver (control plane).

        Args:
            op_type: Operator type
            name: State variable name
            step: Step number (ignored for eager_trigger operators)
            data: State data
        """
        # Check if this op is in eager_trigger mode
        is_eager = (
            self._op_manager.is_eager_trigger(op_type) if self._op_manager else False
        )

        if is_eager:
            # Eager mode: use _eager_input_steps for input labeling, ignore incoming step
            # Each incoming data gets a unique input_step, then enqueue immediately
            input_step = self._eager_input_steps[op_type]
            self._eager_input_steps[op_type] += 1  # Increment for next input

            await self._mailbox.put_state(op_type, name, input_step, data)
            logger.debug(
                f"[spmd_id={self.spmd_id}] Eager receive state: op={op_type.name}, name={name}, "
                f"input_step={input_step}"
            )
            await self._enqueue(op_type, input_step)
            return

        # Normal mode: step-ordered execution
        exec_step = self._current_steps[op_type]

        # Drop stale data (data for steps already executed)
        if step < exec_step:
            logger.warning(
                f"[spmd_id={self.spmd_id}] Dropping stale state data: op={op_type.name}, name={name}, "
                f"step={step} < exec_step={exec_step}"
            )
            return

        # Cache in mailbox
        await self._mailbox.put_state(op_type, name, step, data)
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} received state: op={op_type.name}, name={name}, step={step}"
        )

        # Check and enqueue if data is for current execution step
        await self._check_and_enqueue(op_type, step)

    async def receive_tensor_data(
        self,
        op_type: OpType,
        name: str,
        step: int,
        sender_rank: int,
        data: Any,
        meta: Optional[TensorMeta] = None,
    ) -> None:
        """
        Receive tensor data from upstream worker (data plane).

        Args:
            op_type: Operator type
            name: Tensor variable name
            step: Step number (ignored for eager_trigger operators)
            sender_rank: Sender's rank
            data: Tensor shard
            meta: Optional tensor metadata (for routing and backward tracking)
        """
        # Check if this op is in eager_trigger mode
        is_eager = (
            self._op_manager.is_eager_trigger(op_type) if self._op_manager else False
        )

        if is_eager:
            # Eager mode: use _eager_input_steps for input labeling, ignore incoming step
            # Each incoming data gets a unique input_step, then enqueue immediately
            input_step = self._eager_input_steps[op_type]
            self._eager_input_steps[op_type] += 1  # Increment for next input

            await self._mailbox.put_tensor(
                op_type, name, input_step, sender_rank, data, meta
            )
            logger.debug(
                f"[spmd_id={self.spmd_id}] Eager receive tensor: op={op_type.name}, name={name}, "
                f"sender_rank={sender_rank}, input_step={input_step}"
            )
            await self._enqueue(op_type, input_step)
            return

        # Normal mode: step-ordered execution
        exec_step = self._current_steps[op_type]

        # Drop stale data (data for steps already executed)
        if step < exec_step:
            logger.warning(
                f"[spmd_id={self.spmd_id}] Dropping stale tensor data: op={op_type.name}, name={name}, "
                f"step={step} < exec_step={exec_step}, rank={sender_rank}"
            )
            return

        # Cache in mailbox with metadata
        await self._mailbox.put_tensor(op_type, name, step, sender_rank, data, meta)
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} received tensor: op={op_type.name}, name={name}, "
            f"step={step}, sender_rank={sender_rank}, has_meta={meta is not None}, data={data}"
        )

        # Check and enqueue if data might be ready for execution
        await self._check_and_enqueue(op_type, step)

    async def _check_and_enqueue(self, op_type: OpType, step: int) -> None:
        """Check data availability and enqueue if ready."""
        # Get expected data from OperatorManager
        expected_states = self._op_manager.get_expected_state_names(op_type)
        expected_tensors = self._op_manager.get_expected_tensor_shards(op_type)

        # Check state data
        if expected_states:
            if not await self._mailbox.has_all_states(op_type, step, expected_states):
                logger.debug(
                    f"[spmd_id={self.spmd_id}] Waiting for state data: op={op_type.name}, step={step}, "
                    f"expected={expected_states}"
                )
                return

        # Check tensor shards
        if expected_tensors:
            if not await self._mailbox.has_all_tensors(op_type, step, expected_tensors):
                logger.debug(
                    f"[spmd_id={self.spmd_id}] Waiting for tensor shards: op={op_type.name}, step={step}, "
                    f"expected={expected_tensors}"
                )
                return

        # All conditions met, enqueue
        logger.info(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Enqueuing task: op={op_type.name}, step={step}"
        )
        await self._enqueue(op_type, step)

    async def _enqueue(self, op_type: OpType, step: int) -> None:
        """Enqueue a task for execution."""
        await self._execution_queue.put((op_type.value, step, op_type))
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Task enqueued: op={op_type.name}, step={step}"
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
                priority, step, op_type = await self._execution_queue.get()

                # Check if we should stop
                if not self._running:
                    self._execution_queue.task_done()
                    break

                # Check if this op is blocked
                if not self._op_runnable[op_type].is_set():
                    # Hold in pending queue instead of re-enqueueing
                    async with self._pending_lock:
                        self._pending_blocked[op_type].append(step)
                    logger.debug(
                        f"[spmd_id={self.spmd_id}] Task held (op blocked): op={op_type.name}, step={step}"
                    )
                    continue

                # Check if this op is in eager_trigger mode
                is_eager = (
                    self._op_manager.is_eager_trigger(op_type)
                    if self._op_manager
                    else False
                )

                if not is_eager:
                    # Normal mode: verify step matches current expected execution step
                    exec_step = self._current_steps[op_type]
                    if step < exec_step:
                        # Truly stale task (already executed), drop it
                        logger.warning(
                            f"[spmd_id={self.spmd_id}] Stale task dropped: op={op_type.name}, "
                            f"task_step={step}, exec_step={exec_step}"
                        )
                        continue
                    elif step > exec_step:
                        # Future task (arrived early), hold for later execution
                        async with self._pending_lock:
                            self._pending_future_steps[op_type].append(step)
                        logger.debug(
                            f"[spmd_id={self.spmd_id}] Future task held: op={op_type.name}, "
                            f"task_step={step}, exec_step={exec_step}"
                        )
                        continue

                # Execute (for eager mode, execute any task that comes)
                await self._execute_operator(op_type, step)

                # Advance execution step counter
                await self._advance_step(op_type)

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

        For eager mode (reader's downstream), there should be exactly one tensor arg.
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

        # Build tensor metadata
        shape = ()
        dtype = "object"
        if isinstance(data_to_send, np.ndarray):
            shape = data_to_send.shape
            dtype = str(data_to_send.dtype)

        meta = TensorMeta(
            tensor_name=tensor_name,
            shape=shape,
            dtype=dtype,
            source_spmd_id=self.spmd_id,
            source_rank=self.rank,
            forward_step=step,
            is_sparse=False,
        )

        # Send to all downstream SPMDs (round-robin to workers within each SPMD)
        for spmd_id, handlers in self._successor_handlers.items():
            if not handlers:
                continue

            # Determine target worker (round-robin based on step)
            target_ranks = list(handlers.keys())
            target_rank = target_ranks[step % len(target_ranks)]
            handler = handlers[target_rank]

            try:
                # Send as forward op data
                # data_to_send is already converted to the target type by datasource
                handler.receive_tensor_data.remote(
                    OpType.FORWARD, tensor_name, step, self.rank, data_to_send, meta
                )
            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Failed to send batch to {spmd_id}:{target_rank}: {e}"
                )

    async def stop(self) -> None:
        """Stop the execution loop gracefully."""
        self._running = False
        # Put a sentinel to unblock the queue
        await self._execution_queue.put((999, 0, OpType.FORWARD))

        # Cancel tasks
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

        logger.info(f"[spmd_id={self.spmd_id}] Worker {self.rank} stopping")

    async def _check_all_empty(self, op_type: OpType, step: int) -> bool:
        """
        Check if ALL received tensor shards for this step are empty packets.

        Returns True only when every shard of every expected tensor carries
        ``is_empty=True`` in its meta — meaning no real data was routed to
        this worker for this step.
        """
        if self._op_manager is None:
            return False
        expected_tensor_names = self._op_manager.get_expected_tensor_names(op_type)
        if not expected_tensor_names:
            return False

        all_tensor_shards = await self._mailbox.get_all_tensor_shards(op_type, step)

        for tensor_name in expected_tensor_names:
            if tensor_name not in all_tensor_shards:
                return False  # Missing tensor is not "all empty"
            shards = all_tensor_shards[tensor_name]
            if not shards:
                return False  # No shards at all
            for rank, (data, meta) in shards.items():
                if meta is None or not meta.is_empty:
                    return False  # At least one non-empty shard
        return True

    async def _propagate_empty_packets(self, op_type: OpType, step: int) -> None:
        """
        Forward empty packets to successor SPMDs when sparse routing is involved.

        Called when this worker received only empty packets for all tensors at
        a given step.  The rule is:

        - Dense-to-Dense (output_spec=Static, post_spec=Static): ignore, no
          forwarding needed because the downstream dense assembler doesn't
          require a packet from every rank.
        - Any sparse involvement (output_spec=Dynamic OR post_spec=Dynamic):
          forward an empty packet to **every** target rank so that the
          downstream sparse assembler (which waits for a packet from every
          source) doesn't hang.
        """
        if self._op_manager is None:
            return

        # enable_output_step_remap=True means multiple upstream ranks map to
        # different output steps; the downstream assembler never expects a
        # packet from every rank at the same step, so empty packets are
        # unnecessary and would cause step-number confusion.
        if getattr(self.config, "enable_output_step_remap", False):
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] _propagate_empty: skipped because "
                f"enable_output_step_remap=True (op={op_type.name}, step={step})"
            )
            return

        output_sharding = self._op_manager.get_output_sharding(op_type)
        successor_ids = self._op_manager.get_successor_ids(op_type)

        if not successor_ids:
            return

        # Collect base metas from received empty packets (for global_shape propagation)
        expected_tensor_names = self._op_manager.get_expected_tensor_names(op_type)
        all_tensor_shards = await self._mailbox.get_all_tensor_shards(op_type, step)
        received_metas: Dict[str, TensorMeta] = {}
        for tensor_name in expected_tensor_names:
            shards = all_tensor_shards.get(tensor_name, {})
            for rank, (data, meta) in shards.items():
                if meta is not None:
                    received_metas[tensor_name] = meta
                    break

        # Compute output_step consistently with the normal execution path
        is_eager = self._op_manager.is_eager_trigger(op_type)
        current_step = self._current_steps[op_type]
        output_step = self.compute_output_step(
            is_eager=is_eager, current_step=current_step, input_step=step
        )

        for spmd_id in successor_ids:
            succ_info = self._op_manager.get_successor_info(op_type, spmd_id)
            if succ_info is None:
                continue

            target_op = succ_info.target_op

            for tensor_name in output_sharding:
                output_spec = output_sharding.get(tensor_name)
                post_spec = succ_info.tensor_sharding.get(tensor_name)

                is_sparse_routing = getattr(
                    output_spec, "is_content_addressable", False
                ) or getattr(post_spec, "is_content_addressable", False)

                # For self_loop, always send empty packets even for Dense-to-Dense
                # For non-self_loop, skip Dense-to-Dense (no forwarding needed)
                if not is_sparse_routing and not succ_info.is_self_loop:
                    # Dense-to-dense (non-self_loop): ignore empty packets, no forwarding needed
                    logger.debug(
                        f"[spmd_id={self.spmd_id} rank={self.rank}] _propagate_empty: dense-to-dense for tensor "
                        f"{tensor_name} to successor {spmd_id}, skipping (step={step})"
                    )
                    continue

                # Build base_meta from received empty packets for global_shape propagation
                base_meta = received_metas.get(tensor_name)
                if base_meta is None:
                    # Fallback: create minimal meta
                    base_meta = TensorMeta(
                        tensor_name=tensor_name,
                        forward_step=output_step,
                        is_sparse=getattr(output_spec, "is_content_addressable", False),
                    )

                if succ_info.is_self_loop:
                    logger.debug(
                        f"[spmd_id={self.spmd_id} rank={self.rank}] _propagate_empty: self_loop for tensor "
                        f"{tensor_name} to successor {spmd_id}, forwarding empty packets "
                        f"(step={step}, output_step={output_step}, "
                        f"global_shape={base_meta.global_shape})"
                    )
                else:
                    logger.debug(
                        f"[spmd_id={self.spmd_id} rank={self.rank}] _propagate_empty: sparse routing for tensor "
                        f"{tensor_name} to successor {spmd_id}, forwarding empty packets "
                        f"(step={step}, output_step={output_step}, "
                        f"global_shape={base_meta.global_shape})"
                    )

                if succ_info.is_self_loop:
                    empty_meta = TensorMeta(
                        tensor_name=base_meta.tensor_name,
                        shape=(0,),
                        dtype=base_meta.dtype,
                        global_shape=base_meta.global_shape,
                        source_spmd_id=self.spmd_id,
                        source_rank=self.rank,
                        forward_step=output_step,
                        is_sparse=base_meta.is_sparse,
                        is_empty=True,
                    )
                    self._self_receive_tensor(
                        target_op, tensor_name, output_step, np.array([]), empty_meta
                    )
                else:
                    handlers = self._successor_handlers.get(spmd_id, {})
                    for target_rank in range(succ_info.total_rank):
                        handler = handlers.get(target_rank)
                        if handler is not None:
                            self._send_empty_packet(
                                handler,
                                target_op,
                                tensor_name,
                                output_step,
                                base_meta,
                                spmd_id,
                                target_rank,
                            )

    async def _execute_operator(self, op_type: OpType, step: int) -> None:
        """Execute an operator: assemble data, execute, send outputs."""
        try:
            # --- Early check: all received shards are empty packets ---
            all_empty = await self._check_all_empty(op_type, step)
            if all_empty:
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] All received shards are empty for "
                    f"op={op_type.name}, step={step}. Propagating per sparse routing rules."
                )
                await self._propagate_empty_packets(op_type, step)
                await self._mailbox.clear_step(op_type, step)
                return

            # Assemble input data
            input_data = await self._assemble_input(op_type, step)

            # Execute operator
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Executing: op={op_type.name}, step={step}"
            )
            result = self._op_manager.execute(op_type, input_data)

            if result is None:
                logger.warning(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Operator {op_type.name} returned None, skipping output send"
                )
                return

            # Determine output step
            is_eager = (
                self._op_manager.is_eager_trigger(op_type)
                if self._op_manager
                else False
            )
            current_step = self._current_steps[op_type]
            output_step = self.compute_output_step(
                is_eager=is_eager, current_step=current_step, input_step=step
            )
            if is_eager:
                logger.debug(
                    f"[spmd_id={self.spmd_id}] eager_trigger=True: input_step={step}, current_step={current_step}, "
                    f"output_step={output_step}, rank={self.rank}, total_rank={self._total_rank}, "
                    f"enable_output_step_remap={getattr(self.config, 'enable_output_step_remap', False)}"
                )

            # Send output
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] About to send output: op={op_type.name}, "
                f"output_step={output_step}, result_type={type(result).__name__}"
            )
            await self._send_output(op_type, output_step, result, exec_step=step)

            # Clear mailbox
            await self._mailbox.clear_step(op_type, step)

        except Exception as e:
            logger.error(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Operator {op_type.name} execution failed: {e}"
            )
            raise

    async def _assemble_input(self, op_type: OpType, step: int) -> Any:
        """
        Assemble input data from mailbox and construct data_type object.

        This method completes 5 tasks:
        1. Get static_args from op_manager
        2. Get state_args from mailbox based on expected_state_names
        3. Assemble tensor_args from mailbox based on expected_tensor_names and sharding
        4. Assemble tensor meta
        5. Construct the final data_type object expected by the operator

        Args:
            op_type: Operator type
            step: Execution step

        Returns:
            Constructed data_type object ready for operator execution
        """
        # Task 1: Get static arguments
        static_args = self._op_manager.get_static_args()

        # Task 2: Get state arguments
        expected_state_names = self._op_manager.get_expected_state_names(op_type)
        all_states = await self._mailbox.get_all_states(op_type, step)
        state_args = {
            name: all_states[name]
            for name in expected_state_names
            if name in all_states
        }

        # Task 3 & 4: Assemble tensor arguments and meta
        expected_tensor_names = self._op_manager.get_expected_tensor_names(op_type)
        all_tensor_shards = await self._mailbox.get_all_tensor_shards(op_type, step)

        # Check if eager_trigger mode
        is_eager = self._op_manager.is_eager_trigger(op_type)

        tensor_args = {}
        tensor_metas: Dict[str, Optional[TensorMeta]] = {}
        tensor_shard_origins: Dict[str, List[ShardOrigin]] = {}

        for tensor_name in expected_tensor_names:
            if tensor_name not in all_tensor_shards:
                raise ValueError(
                    f"Missing tensor '{tensor_name}' for operator {op_type.name} at step {step}"
                )

            shards = all_tensor_shards[tensor_name]

            if is_eager:
                # Eager mode: use first shard directly, no assembly needed
                first_rank = next(iter(shards.keys()))
                shard_data, shard_meta = shards[first_rank]
                tensor_args[tensor_name] = shard_data
                tensor_metas[tensor_name] = shard_meta
                # Record single shard origin for eager mode.
                # Eager 下没有装配/重排序，整个 tensor 视为一个 shard，
                # local_indices 即 [0, 1, ..., num_rows-1]。
                shard_shape = (
                    tuple(shard_data.shape) if hasattr(shard_data, "shape") else ()
                )
                num_rows = shard_shape[0] if shard_shape else 0
                local_indices = (
                    np.arange(num_rows, dtype=np.int64) if num_rows > 0 else None
                )
                tensor_shard_origins[tensor_name] = [
                    ShardOrigin(
                        source_spmd_id=shard_meta.source_spmd_id if shard_meta else "",
                        source_rank=shard_meta.source_rank
                        if shard_meta
                        else first_rank,
                        global_indices=shard_meta.global_indices
                        if shard_meta
                        else None,
                        shard_shape=shard_shape,
                        local_indices=local_indices,
                    )
                ]
            else:
                # Normal mode: assemble tensor based on sharding
                (
                    assembled_tensor,
                    assembled_meta,
                    shard_origins,
                ) = await self._assemble_tensor(op_type, tensor_name, shards)
                tensor_args[tensor_name] = assembled_tensor
                tensor_metas[tensor_name] = assembled_meta
                tensor_shard_origins[tensor_name] = shard_origins

        # Cache assembled metas for use in _send_output (global_shape / global_indices)
        self._last_assembled_metas[op_type] = tensor_metas
        # Store per-step history with shard origins for backward reverse routing
        step_meta_history: Dict[str, StepTensorMeta] = {}
        for tn in tensor_metas:
            step_meta_history[tn] = StepTensorMeta(
                assembled_meta=tensor_metas[tn],
                shard_origins=tensor_shard_origins.get(tn, []),
            )
        self._meta_history[op_type][step] = step_meta_history

        for tn, tm in tensor_metas.items():
            if tm is not None:
                origins = tensor_shard_origins.get(tn, [])
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] _assemble_input cached meta: op={op_type.name}, step={step}, "
                    f"tensor={tn}, shape={tm.shape}, global_shape={tm.global_shape}, "
                    f"global_indices_len={len(tm.global_indices) if tm.global_indices is not None else None}, "
                    f"is_sparse={tm.is_sparse}, shard_origins_count={len(origins)}"
                )

        # Task 5: Construct final data_type object
        input_type = self._op_manager.get_operator_input_type(op_type)
        if input_type is None:
            raise ValueError(
                f"Cannot determine input type for operator {op_type.name}. "
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
        self,
        op_type: OpType,
        tensor_name: str,
        shards: Dict[int, Tuple[Any, Optional[TensorMeta]]],
    ) -> Tuple[Any, Optional[TensorMeta], List[ShardOrigin]]:
        """
        Assemble tensor shards into a complete tensor based on sharding specs.

        Handles three cases:
        1. static -> static (Dense-to-Dense): Use DenseShardingStrategy
        2. dynamic -> static (Sparse-to-Dense): Reconstruct from global_indices
        3. to dynamic (Any -> Sparse): Merge all shards

        Args:
            op_type: Operator type
            tensor_name: Tensor name
            shards: Dict of {rank: (data, meta)}

        Returns:
            Tuple of (assembled_tensor, assembled_meta, shard_origins)
        """
        input_sharding = self._op_manager.get_input_sharding(op_type)
        input_spec = input_sharding.get(tensor_name)

        if input_spec is None:
            # No input sharding specified, treat as to dynamic (merge all)
            return self._assemble_to_dynamic_tensor(tensor_name, shards)

        if not input_spec.is_content_addressable:
            # Check if any predecessor is dynamic (sparse)
            predecessor_topology = self._op_manager.get_predecessor_topology(op_type)
            has_dynamic_predecessor = False

            for pre_spmd_id, pre_info in predecessor_topology.items():
                pre_sharding = pre_info.tensor_sharding.get(tensor_name)
                if pre_sharding is not None and pre_sharding.is_content_addressable:
                    has_dynamic_predecessor = True
                    break

            if has_dynamic_predecessor:
                # Case 2: dynamic -> static
                return self._assemble_dynamic_to_static_tensor(
                    tensor_name, shards, input_spec
                )
            else:
                # Case 1: static -> static
                return await self._assemble_static_to_static_tensor(
                    op_type, tensor_name, shards, input_spec, predecessor_topology
                )
        else:
            # Case 3: to dynamic
            return self._assemble_to_dynamic_tensor(tensor_name, shards)

    async def _assemble_static_to_static_tensor(
        self,
        op_type: OpType,
        tensor_name: str,
        shards: Dict[int, Tuple[Any, Optional[TensorMeta]]],
        input_spec: ShardingSpec,
        predecessor_topology: Dict[str, Any],
    ) -> Tuple[Any, Optional[TensorMeta], List[ShardOrigin]]:
        """
        Assemble static-to-static (Dense-to-Dense) tensor.

        Uses DenseShardingStrategy to determine which shards to use and how to assemble them.
        """
        # Collect shards from all predecessors based on routing strategies
        all_shards_to_assemble: List[
            Tuple[int, Tuple[Any, Optional[TensorMeta]], str]
        ] = []

        for pre_spmd_id, pre_info in predecessor_topology.items():
            pre_sharding = pre_info.tensor_sharding.get(tensor_name)
            if pre_sharding is None or getattr(
                pre_sharding, "is_content_addressable", False
            ):
                continue

            # Get input routing strategy
            strategy = self._op_manager.get_input_routing_strategy(
                op_type, pre_spmd_id, tensor_name
            )

            if strategy is None:
                # Fallback: use all shards from this predecessor
                logger.warning(
                    f"No routing strategy for {pre_spmd_id}:{tensor_name}, "
                    f"using all shards as fallback"
                )
                for rank in range(pre_info.total_rank):
                    if rank in shards:
                        all_shards_to_assemble.append((rank, shards[rank], pre_spmd_id))
            else:
                # Use dependent_ranks from strategy
                for rank in strategy.dependent_ranks:
                    if rank in shards:
                        all_shards_to_assemble.append((rank, shards[rank], pre_spmd_id))

        if not all_shards_to_assemble:
            raise ValueError(
                f"No shards found for tensor '{tensor_name}' in static-to-static assembly"
            )

        # Extract data and meta
        shard_data = [item[1][0] for item in all_shards_to_assemble]
        shard_metas = [
            item[1][1] for item in all_shards_to_assemble if item[1][1] is not None
        ]

        # Build shard origins (track local row indices in assembled tensor).
        # Static-to-static 装配没有重排序，所有 shard 只是简单按轴 0 拼接，
        # 因此每个 shard 在 assembled tensor 中对应一个连续区间：
        #   [row_offset, row_offset + num_rows)
        shard_origins: List[ShardOrigin] = []
        row_offset = 0
        for rank, (data, meta), pre_spmd_id in all_shards_to_assemble:
            s_shape = tuple(data.shape) if hasattr(data, "shape") else ()
            num_rows = s_shape[0] if s_shape else 0
            local_indices = (
                np.arange(row_offset, row_offset + num_rows, dtype=np.int64)
                if num_rows > 0
                else None
            )
            shard_origins.append(
                ShardOrigin(
                    source_spmd_id=meta.source_spmd_id if meta else pre_spmd_id,
                    source_rank=meta.source_rank if meta else rank,
                    global_indices=meta.global_indices if meta else None,
                    shard_shape=s_shape,
                    local_indices=local_indices,
                )
            )
            row_offset += num_rows

        # For now, simple concatenation along first axis
        # TODO: Implement proper assembly based on dim_infos and sharding spec
        # This is a simplified version - full implementation would need to:
        # 1. Determine assembly order based on dim_infos
        # 2. Handle reshape/transpose if needed
        # 3. Concatenate/stack along appropriate axes

        if len(shard_data) == 1:
            assembled_tensor = shard_data[0]
        else:
            # Simple concatenation (assumes shards can be concatenated)
            try:
                assembled_tensor = np.concatenate(shard_data, axis=0)
            except ValueError:
                # If concatenation fails, try stacking
                assembled_tensor = np.stack(shard_data, axis=0)

        # Build assembled meta with correct shape and global_shape
        first_meta = shard_metas[0] if shard_metas else None
        assembled_shape = (
            tuple(assembled_tensor.shape) if hasattr(assembled_tensor, "shape") else ()
        )
        assembled_dtype = (
            str(assembled_tensor.dtype)
            if hasattr(assembled_tensor, "dtype")
            else "unknown"
        )

        # Infer global_shape from input_spec (assembled local shape → global shape)
        global_shape: Optional[Tuple[int, ...]] = None
        if not input_spec.is_content_addressable and assembled_shape:
            # We don't have infer_global_shape on ShardingSpec right now.
            # We'll use the assembled shape directly, assuming no other axes are sharded.
            global_shape = assembled_shape
        elif first_meta is not None:
            global_shape = first_meta.global_shape

        assembled_meta = TensorMeta(
            tensor_name=tensor_name,
            shape=assembled_shape,
            dtype=assembled_dtype,
            global_shape=global_shape,
            source_spmd_id=first_meta.source_spmd_id if first_meta else "",
            source_rank=first_meta.source_rank if first_meta else -1,
            forward_step=first_meta.forward_step if first_meta else -1,
            is_sparse=False,
        )

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] _assemble_static_to_static: tensor={tensor_name}, "
            f"num_shards={len(shard_data)}, assembled_shape={assembled_shape}, "
            f"global_shape={global_shape}, input_spec={type(input_spec).__name__ if input_spec else None}, "
            f"shard_origins={len(shard_origins)}"
        )

        return assembled_tensor, assembled_meta, shard_origins

    def _assemble_dynamic_to_static_tensor(
        self,
        tensor_name: str,
        shards: Dict[int, Tuple[Any, Optional[TensorMeta]]],
        input_spec: ShardingSpec,
    ) -> Tuple[Any, Optional[TensorMeta], List[ShardOrigin]]:
        """
        Assemble dynamic-to-static (Sparse-to-Dense) tensor.

        Reconstructs LOCAL dense tensor from sparse shards.
        The received sparse data has already been filtered by routing function,
        so we only need to convert global_indices to local_indices and reconstruct.

        Key insight: We need global_shape to compute local range, then convert
        global_indices to local_indices for local dense tensor reconstruction.
        """
        # Collect all sparse shards, filtering out empty packets.
        #
        # 对于 dynamic→static，我们需要：
        #   1) 保留业务逻辑：按 global_indices 排序重建本地 dense tensor。
        #   2) 为每个 shard 记录 **重排序之后** 在本地 tensor 中的行下标
        #      (local_indices)，以便 backward 通过 local_indices 做精确切片。
        all_values: List[np.ndarray] = []
        all_indices: List[np.ndarray] = []
        all_metas: List[Optional[TensorMeta]] = []
        shard_datas: List[np.ndarray] = []
        shard_metas: List[Optional[TensorMeta]] = []

        for rank, (data, meta) in shards.items():
            # Filter empty packets
            if meta is not None and meta.is_empty:
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] "
                    f"Filtered empty packet from sender_rank={rank}"
                )
                continue

            s_shape = tuple(data.shape) if hasattr(data, "shape") else ()
            num_rows = s_shape[0] if s_shape else 0
            if num_rows == 0:
                continue

            if meta is None or meta.global_indices is None:
                # No global_indices — still include the data, and generate
                # sequential fallback indices so that dense reconstruction
                # and local_indices 推导可以继续。
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Missing global_indices for "
                    f"sparse tensor '{tensor_name}' from rank {rank}, generating "
                    f"fallback indices. This should never happen, please check the "
                    f"sharding spec"
                )
                if all_indices:
                    max_prev = int(np.max(np.concatenate(all_indices))) + 1
                else:
                    max_prev = 0
                indices = np.arange(max_prev, max_prev + num_rows, dtype=np.int64)
            else:
                indices = meta.global_indices.astype(np.int64, copy=False)

            shard_datas.append(data)
            shard_metas.append(meta)
            all_values.append(data)
            all_indices.append(indices)
            all_metas.append(meta)

        if not all_values:
            # All shards were empty — return empty result with propagated meta
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] _assemble_dynamic_to_static: all shards empty "
                f"for tensor '{tensor_name}', returning empty result"
            )
            # Propagate global_shape from any received empty meta
            global_shape = None
            for rank, (data, meta) in shards.items():
                if meta is not None and meta.global_shape is not None:
                    global_shape = meta.global_shape
                    break
            empty_meta = TensorMeta(
                tensor_name=tensor_name,
                shape=(0,),
                dtype="float32",
                global_shape=global_shape,
                is_sparse=False,
                is_empty=True,
            )
            # 对于全空的情况，不存在任何 shard，返回空的 shard_origins。
            return np.array([]), empty_meta, []

        # Merge all indices and values, **and按 global_indices 排序**。
        cat_indices = (
            np.concatenate(all_indices, axis=0)
            if len(all_indices) > 1
            else all_indices[0]
        )
        cat_values = (
            np.concatenate(all_values, axis=0) if len(all_values) > 1 else all_values[0]
        )

        # 为了在排序后还能知道每一行属于哪个 shard，我们并行维护一个
        # owner_id 向量：owner_ids[i] = 第 i 行来源的 shard 下标。
        owner_id_chunks: List[np.ndarray] = []
        shard_row_counts: List[int] = []
        for shard_idx, data in enumerate(shard_datas):
            s_shape = tuple(data.shape) if hasattr(data, "shape") else ()
            num_rows = s_shape[0] if s_shape else 0
            shard_row_counts.append(num_rows)
            owner_id_chunks.append(np.full((num_rows,), shard_idx, dtype=np.int64))
        owner_ids = (
            np.concatenate(owner_id_chunks, axis=0)
            if len(owner_id_chunks) > 1
            else owner_id_chunks[0]
        )

        # 按 global_indices 排序，保证业务上的全局顺序。
        if len(cat_indices.shape) == 1:
            sort_order = np.argsort(cat_indices)
        else:
            # Multi-dimensional indices: lexicographic sort
            sort_order = np.lexsort(cat_indices.T[::-1])

        merged_indices = cat_indices[sort_order]
        merged_values = cat_values[sort_order]
        sorted_owner_ids = owner_ids[sort_order]

        # Get global_shape from meta (required to compute local range)
        global_shape = None
        for meta in all_metas:
            if meta.global_shape is not None:
                global_shape = meta.global_shape
                break

        # If global_shape not in meta, try to infer from global_indices
        if global_shape is None:
            if merged_indices is not None and len(merged_indices.shape) > 0:
                if merged_indices.shape[0] > 0:
                    # For 1D indices, global_shape = [max_index + 1]
                    if len(merged_indices.shape) == 1:
                        max_idx = int(np.max(merged_indices))
                        global_shape = (max_idx + 1,)
                    else:
                        # For multi-dimensional indices, take max along each dimension
                        max_indices = np.max(merged_indices, axis=0)
                        global_shape = tuple(int(x) + 1 for x in max_indices)

        if global_shape is None:
            # Cannot infer global_shape – 这里仍然返回 dense 形式的数据，但
            # global_shape 为空，仅用于本地计算，不影响后续利用 local_indices
            # 做反向切片。
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] _assemble_dynamic_to_static: "
                f"cannot infer global_shape for tensor '{tensor_name}', using "
                f"merged_values.shape as local dense shape"
            )
            global_shape = merged_values.shape

        # Compute local range using input_spec.lower(global_shape, rank)
        # This gives us the slices for each dimension
        internal_layout = input_spec.lower(global_shape, self.rank)
        local_shape = internal_layout.get_local_shape()
        dense_slices = internal_layout.dense_slices

        if dense_slices is None:
            raise ValueError(
                f"Cannot compute local range for tensor '{tensor_name}': "
                f"input_spec.lower() did not return dense_slices"
            )

        # Filter indices that fall into this worker's local range (axis-0).
        if len(merged_indices.shape) == 1:
            slice_start = dense_slices[0].start
            slice_end = dense_slices[0].end
            mask = (merged_indices >= slice_start) & (merged_indices < slice_end)
        else:
            # Multi-dimensional: 仍旧只用第 0 维来决定本地 batch 范围。
            slice_start = dense_slices[0].start
            slice_end = dense_slices[0].end
            dim0 = merged_indices[:, 0]
            mask = (dim0 >= slice_start) & (dim0 < slice_end)

        local_indices = merged_indices[mask] - slice_start
        local_values = merged_values[mask]

        # Reconstruct LOCAL dense tensor from sparse data
        assembled_tensor = np.zeros(local_shape, dtype=merged_values.dtype)
        assembled_tensor[local_indices] = local_values

        # Build shard origins with **local_indices**: 对于排序后的 local dense
        # tensor，统计每个 shard 在 axis-0 上对应的行坐标。
        shard_origins: List[ShardOrigin] = []
        num_shards = len(shard_datas)
        # 对于每个 shard，下标 j 的行属于该 shard 的位置即：
        #   np.nonzero(sorted_owner_ids[mask] == j)[0]
        for shard_idx in range(num_shards):
            data = shard_datas[shard_idx]
            meta = shard_metas[shard_idx]
            s_shape = tuple(data.shape) if hasattr(data, "shape") else ()
            if s_shape and s_shape[0] > 0:
                local_pos = np.nonzero(sorted_owner_ids[mask] == shard_idx)[0]
                local_indices_for_shard: Optional[np.ndarray]
                if local_pos.size > 0:
                    local_indices_for_shard = local_pos.astype(np.int64, copy=False)
                else:
                    local_indices_for_shard = None
            else:
                local_indices_for_shard = None

            shard_origins.append(
                ShardOrigin(
                    source_spmd_id=meta.source_spmd_id if meta else "",
                    source_rank=meta.source_rank if meta else -1,
                    global_indices=(
                        meta.global_indices.copy()
                        if (meta is not None and meta.global_indices is not None)
                        else None
                    ),
                    shard_shape=s_shape,
                    local_indices=local_indices_for_shard,
                )
            )

        # Create new meta for local dense tensor
        assembled_meta = TensorMeta(
            tensor_name=tensor_name,
            shape=assembled_tensor.shape,
            dtype=str(assembled_tensor.dtype),
            global_shape=global_shape,  # Keep global_shape for reference
            is_sparse=False,
        )

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] _assemble_dynamic_to_static: tensor={tensor_name}, "
            f"num_shards={len(all_values)}, global_shape={global_shape}, "
            f"local_shape={assembled_tensor.shape}, "
            f"merged_indices_len={len(merged_indices) if merged_indices is not None else 0}, "
            f"shard_origins={len(shard_origins)}"
        )

        return assembled_tensor, assembled_meta, shard_origins

    def _assemble_to_dynamic_tensor(
        self, tensor_name: str, shards: Dict[int, Tuple[Any, Optional[TensorMeta]]]
    ) -> Tuple[Any, TensorMeta, List[ShardOrigin]]:
        """
        Assemble to-dynamic (Any -> Sparse) tensor.

        Simply merges all shards and their metadata.
        Also collects per-shard origins for backward reverse routing.
        """
        # Collect all shards, filtering out empty packets
        all_data = []
        all_metas = []
        shard_origins: List[ShardOrigin] = []
        row_offset = 0

        for rank in sorted(shards.keys()):
            data, meta = shards[rank]
            # Filter empty packets
            if meta is not None and meta.is_empty:
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] "
                    f"Filtered empty packet from sender_rank={rank}"
                )
                continue
            s_shape = tuple(data.shape) if hasattr(data, "shape") else ()
            num_rows = s_shape[0] if s_shape else 0
            local_indices = (
                np.arange(row_offset, row_offset + num_rows, dtype=np.int64)
                if num_rows > 0
                else None
            )
            shard_origins.append(
                ShardOrigin(
                    source_spmd_id=meta.source_spmd_id if meta else "",
                    source_rank=meta.source_rank if meta else rank,
                    global_indices=(
                        meta.global_indices.copy()
                        if (meta and meta.global_indices is not None)
                        else None
                    ),
                    shard_shape=s_shape,
                    local_indices=local_indices,
                )
            )
            row_offset += num_rows
            all_data.append(data)
            if meta is not None:
                all_metas.append(meta)

        if not all_data:
            # All shards were empty — return empty result with propagated meta
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] _assemble_to_dynamic: all shards empty "
                f"for tensor '{tensor_name}', returning empty result"
            )
            global_shape = None
            for rank, (data, meta) in shards.items():
                if meta is not None and meta.global_shape is not None:
                    global_shape = meta.global_shape
                    break
            empty_meta = TensorMeta(
                tensor_name=tensor_name,
                shape=(0,),
                dtype="float32",
                global_shape=global_shape,
                is_sparse=True,
                is_empty=True,
            )
            return np.array([]), empty_meta, shard_origins

        # Merge data
        if len(all_data) == 1:
            merged_data = all_data[0]
        else:
            merged_data = np.concatenate(all_data, axis=0)

        # Merge metadata
        if all_metas:
            # Merge global_indices
            all_indices = []
            for meta in all_metas:
                if meta.global_indices is not None:
                    all_indices.append(meta.global_indices)

            merged_indices = (
                np.concatenate(all_indices, axis=0)
                if len(all_indices) > 1
                else all_indices[0]
                if all_indices
                else None
            )

            # Use first meta as base, update with merged info, propagate global_shape
            base_meta = all_metas[0]
            # Propagate global_shape from any shard that carries it
            global_shape = None
            for meta in all_metas:
                if meta.global_shape is not None:
                    global_shape = meta.global_shape
                    break
            merged_meta = TensorMeta(
                tensor_name=tensor_name,
                shape=merged_data.shape,
                dtype=str(merged_data.dtype),
                global_indices=merged_indices,
                global_shape=global_shape,
                source_spmd_id=base_meta.source_spmd_id,
                source_rank=base_meta.source_rank,
                forward_step=base_meta.forward_step,
                is_sparse=True,
            )
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] _assemble_to_dynamic: tensor={tensor_name}, "
                f"num_shards={len(all_data)}, merged_shape={merged_data.shape}, "
                f"global_shape={global_shape}, "
                f"merged_indices_len={len(merged_indices) if merged_indices is not None else None}, "
                f"shard_origins={len(shard_origins)}"
            )
        else:
            # Create new meta if none available
            merged_meta = TensorMeta(
                tensor_name=tensor_name,
                shape=merged_data.shape,
                dtype=str(merged_data.dtype),
                is_sparse=True,
            )
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] _assemble_to_dynamic: tensor={tensor_name}, "
                f"num_shards={len(all_data)}, merged_shape={merged_data.shape}, no shard metas"
            )

        return merged_data, merged_meta, shard_origins

    async def _advance_step(self, op_type: OpType) -> None:
        """Advance execution step counter for an operator."""
        self._current_steps[op_type] += 1
        new_step = self._current_steps[op_type]

        # Report op completion to driver
        self._report_op_completed(op_type, new_step)

        # Check if this op is in eager_trigger mode
        is_eager = (
            self._op_manager.is_eager_trigger(op_type) if self._op_manager else False
        )

        if not is_eager:
            # Normal mode: re-enqueue pending future tasks that match the new step
            async with self._pending_lock:
                pending = self._pending_future_steps[op_type]
                to_enqueue = [s for s in pending if s == new_step]
                self._pending_future_steps[op_type] = [
                    s for s in pending if s != new_step
                ]

            for s in to_enqueue:
                await self._enqueue(op_type, s)

            # Also check if we have pending data for the new execution step
            await self._check_and_enqueue(op_type, new_step)

    def _report_op_completed(self, op_type: OpType, step: int) -> None:
        """
        Report op completion to driver.

        Args:
            op_type: Operation type
            step: Completed step number
        """
        if self._driver_handle is not None:
            try:
                self._driver_handle.report_worker_op_completed.remote(
                    self.rank, op_type, step
                )
            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Failed to report op completion: {e}"
                )

    # =========================================================================
    # Data Sending
    # =========================================================================

    async def _send_output(
        self,
        op_type: OpType,
        step: int,
        result: Any,
        *,
        exec_step: Optional[int] = None,
    ) -> None:
        """
        Send output to successor SPMDs.

        Uses OperatorManager.get_successor_target_route() for routing.
        Handles self-loops (same node, different op) by calling _self_receive_tensor.

        Routing strategy (refined for backward ops):

        1. **Dense-to-dense** — always use sharding-based geometric routing.
        2. **Sparse involved**:
           2.1. If the relevant ``DynamicShardingSpec`` carries a user-provided
                ``fn`` → delegate to ``get_successor_target_route`` (calls fn).
           2.2. If **no** route fn → reverse-route using the forward step's
                per-shard origin metadata stored in ``_meta_history``.

        Meta lifecycle in this method:
        1. Build one ``routing_meta`` per tensor (full output description)
        2. Pass ``routing_meta`` to routing function for route computation
        3. Derive ``shard_meta`` = ``_build_shard_meta(routing_meta, data_mask)`` per route

        Args:
            op_type: Current operator type
            step: Output step number (may differ from execution step in eager mode)
            result: Operator output
            exec_step: The actual execution step (for forward meta lookup).
                       Defaults to ``step`` when not provided.
        """
        if exec_step is None:
            exec_step = step

        output_sharding = self._op_manager.get_output_sharding(op_type)

        # Extract tensors from result
        tensors_to_send = self._extract_tensors(result, output_sharding)

        # Get successor IDs from OperatorManager
        successor_ids = self._op_manager.get_successor_ids(op_type)

        # Cached input metas for global_shape / global_indices computation
        input_metas = self._last_assembled_metas.get(op_type, {})

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] _send_output: op={op_type.name}, step={step}, "
            f"exec_step={exec_step}, tensors={list(tensors_to_send.keys())}, successors={successor_ids}"
        )

        # --- Phase 1: Build routing metas (once per tensor, independent of successor) ---
        # Use output-to-input tensor name mapping so that when an operator's
        # input and output tensor names differ (e.g. input="input", output="tensor"),
        # the correct input meta is found for global_shape / global_indices propagation.
        output_to_input_mapping = self._op_manager.get_output_to_input_tensor_mapping(
            op_type
        )

        # Determine whether this operator is an "origin tensor" source:
        # eager_trigger + enable_output_step_remap means the SPMD produces
        # initial tensors whose shape IS the global_shape and indices are un-sharded.
        is_eager = (
            self._op_manager.is_eager_trigger(op_type) if self._op_manager else False
        )
        enable_remap = getattr(self.config, "enable_output_step_remap", False)
        is_origin_tensor = is_eager and enable_remap

        routing_metas: Dict[str, TensorMeta] = {}
        for tensor_name, tensor_data in tensors_to_send.items():
            # Look up the corresponding input tensor name via the mapping
            input_tensor_name = output_to_input_mapping.get(tensor_name, tensor_name)
            input_meta = input_metas.get(input_tensor_name)
            output_spec = output_sharding.get(tensor_name)
            routing_metas[tensor_name] = self._build_routing_meta(
                tensor_name,
                tensor_data,
                step,
                output_spec,
                input_meta,
                is_origin_tensor=is_origin_tensor,
            )

        # --- Phase 2: For each successor, route and send ---
        for spmd_id in successor_ids:
            # Get successor topology info (contains target_op and is_self_loop)
            succ_info = self._op_manager.get_successor_info(op_type, spmd_id)
            if succ_info is None:
                logger.warning(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] No topology info for successor {spmd_id}"
                )
                continue

            # Check condition if exists
            if succ_info.condition_fn is not None:
                if not succ_info.condition_fn(result):
                    logger.debug(
                        f"[spmd_id={self.spmd_id} rank={self.rank}] Condition not met for successor {spmd_id}, skipping"
                    )
                    continue

            # Get target op type from topology
            target_op = succ_info.target_op

            # Send each tensor
            for tensor_name, tensor_data in tensors_to_send.items():
                routing_meta = routing_metas[tensor_name]
                output_spec = output_sharding.get(tensor_name)
                post_spec = succ_info.tensor_sharding.get(tensor_name)

                # Handle self-loop: directly call data receive interface
                if succ_info.is_self_loop:
                    shard_meta = self._build_shard_meta(routing_meta, None)
                    self._self_receive_tensor(
                        target_op, tensor_name, step, tensor_data, shard_meta
                    )
                    continue

                # Cross-node: send via handlers
                handlers = self._successor_handlers.get(spmd_id, {})
                if not handlers:
                    logger.warning(
                        f"[spmd_id={self.spmd_id} rank={self.rank}] No handlers for successor {spmd_id}"
                    )
                    continue

                is_sparse_routing = (
                    output_spec.is_content_addressable if output_spec else False
                ) or (post_spec.is_content_addressable if post_spec else False)

                # For backward with sparse routing and NO route fn, use
                # forward-step shard origins for reverse routing.
                use_reverse_routing = False
                if is_sparse_routing and op_type == OpType.BACKWARD:
                    has_fn = (
                        getattr(post_spec.strategy if post_spec else None, "fn", None)
                        is not None
                        or getattr(
                            output_spec.strategy if output_spec else None, "fn", None
                        )
                        is not None
                    )
                    if not has_fn:
                        use_reverse_routing = True

                if use_reverse_routing:
                    await self._send_backward_via_reverse_route(
                        forward_step=exec_step,
                        tensor_name=tensor_name,
                        tensor_data=tensor_data,
                        routing_meta=routing_meta,
                        target_op=target_op,
                        output_step=step,
                        successor_spmd_id=spmd_id,
                        handlers=handlers,
                        succ_total_rank=succ_info.total_rank,
                    )
                else:
                    from fluxway.transport.communication import AsyncDataDistributor
                    from fluxway.transport.tensor import TransportTensor
                    import functools

                    # Wrap single tensor into TransportTensor
                    transport_tensor = TransportTensor(
                        data=tensor_data, indices=routing_meta.global_indices
                    )

                    distributor = AsyncDataDistributor(succ_info.total_rank)

                    # Adapter function for sending packed parts to handlers
                    send_fn = functools.partial(
                        self._send_packaged_tensors,
                        handlers=handlers,
                        target_op=target_op,
                        tensor_name=tensor_name,
                        step=step,
                        routing_meta=routing_meta,
                        successor_spmd_id=spmd_id,
                    )

                    distributor.distribute(
                        local_data=transport_tensor,
                        src_spec=output_spec,
                        dst_spec=post_spec,
                        src_rank=self.rank,
                        send_fn=send_fn,
                    )

        # --- Send state_argument via control plane ---
        # State arguments are sent through: rank0-worker -> driver -> controller -> target-driver -> all-workers
        output_state_names = self._op_manager.get_output_state_names(op_type)
        if output_state_names:
            state_data = self._extract_states(result, output_state_names)

            for spmd_id in successor_ids:
                succ_info = self._op_manager.get_successor_info(op_type, spmd_id)
                if succ_info is None:
                    continue

                # Check condition
                if succ_info.condition_fn is not None:
                    if not succ_info.condition_fn(result):
                        continue

                target_op = succ_info.target_op

                for state_name, state_value in state_data.items():
                    if succ_info.is_self_loop:
                        # Self-loop: directly set as static arg
                        await self._self_receive_state(
                            target_op, state_name, step, state_value
                        )
                    elif self.rank == 0:
                        # Cross-node: send via control plane (only rank 0)
                        logger.debug(
                            f"[spmd_id={self.spmd_id}] Worker {self.rank} sending state_argument: name={state_name}, "
                            f"target_spmd={spmd_id}, step={step}"
                        )
                        self.send_state_argument(
                            spmd_id, target_op, state_name, step, state_value
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

    def _build_routing_meta(
        self,
        tensor_name: str,
        tensor_data: Any,
        step: int,
        output_spec: Optional[ShardingSpec],
        input_meta: Optional[TensorMeta],
        *,
        is_origin_tensor: bool = False,
    ) -> TensorMeta:
        """
        Build **routing meta** — describes the full output tensor before routing/slicing.

        This meta is passed to routing functions for sparse routing decisions.
        For dense-to-dense routing, only ``shape`` is actually used.

        Args:
            tensor_name: Tensor name
            tensor_data: Operator output tensor (may be ``(data, indices)`` tuple for sparse)
            step: Execution step
            output_spec: Output sharding spec of current operator
            input_meta: Cached assembled input meta (from ``_last_assembled_metas``)
            is_origin_tensor: If True, this tensor is the origin of the data pipeline
                (eager + enable_output_step_remap).  Its shape IS the global_shape
                and its indices cover the full un-sharded range.

        Returns:
            TensorMeta describing the full (pre-routing) output tensor
        """
        # --- Unwrap sparse tuple convention: (data, indices) ---
        actual_data = tensor_data
        output_global_indices: Optional[np.ndarray] = None
        if isinstance(tensor_data, tuple) and len(tensor_data) == 2:
            data_part, indices_part = tensor_data
            if isinstance(indices_part, np.ndarray):
                actual_data = data_part
                output_global_indices = indices_part

        # --- shape / dtype from actual data ---
        shape = tuple(actual_data.shape) if hasattr(actual_data, "shape") else ()
        dtype = str(actual_data.dtype) if hasattr(actual_data, "dtype") else "unknown"

        # --- global_shape ---
        # Dense output: infer from output_spec;  Sparse output: propagate from input
        global_shape: Optional[Tuple[int, ...]] = None
        if output_spec is not None and not output_spec.is_content_addressable and shape:
            # We don't have infer_global_shape on ShardingSpec right now.
            # We'll use the local shape directly, assuming no other axes are sharded.
            global_shape = shape
        elif input_meta is not None and input_meta.global_shape is not None:
            global_shape = input_meta.global_shape

        # --- global_indices ---
        # Priority: operator-produced (sparse output) > input meta > dense spec offset
        global_indices: Optional[np.ndarray] = None
        if output_global_indices is not None:
            # Sparse output: operator explicitly produced indices
            global_indices = output_global_indices
        elif input_meta is not None and input_meta.global_indices is not None:
            # Propagate from input (e.g., sparse-input → dense-output pass-through)
            global_indices = input_meta.global_indices
        elif (
            output_spec is not None and not output_spec.is_content_addressable and shape
        ):
            # Dense output: compute from spec's axis-0 global position
            axis0_offset = self._compute_axis0_global_offset(output_spec, actual_data)
            if shape:
                global_indices = np.arange(
                    axis0_offset, axis0_offset + shape[0], dtype=np.int64
                )

        is_sparse = output_spec is not None and output_spec.is_content_addressable

        # Origin tensor: when the SPMD is eager + enable_output_step_remap, the
        # output tensor is the initial tensor in the pipeline.  Its shape IS the
        # global_shape and its indices cover the full un-sharded range.
        if is_origin_tensor and shape and shape[0] > 0:
            if global_shape is None:
                global_shape = shape
            if global_indices is None:
                global_indices = np.arange(shape[0], dtype=np.int64)
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] _build_routing_meta: "
                f"origin tensor '{tensor_name}': global_shape={global_shape}, "
                f"global_indices_len={len(global_indices)}"
            )

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] _build_routing_meta: tensor={tensor_name}, step={step}, "
            f"shape={shape}, dtype={dtype}, global_shape={global_shape}, "
            f"global_indices_len={len(global_indices) if global_indices is not None else None}, "
            f"is_sparse={is_sparse}, "
            f"output_spec={type(output_spec).__name__ if output_spec else None}, "
            f"has_input_meta={input_meta is not None}"
        )

        return TensorMeta(
            tensor_name=tensor_name,
            shape=shape,
            dtype=dtype,
            global_indices=global_indices,
            global_shape=global_shape,
            source_spmd_id=self.spmd_id,
            source_rank=self.rank,
            forward_step=step,
            is_sparse=is_sparse,
        )

    def _build_shard_meta(
        self,
        routing_meta: TensorMeta,
        # NOTE:
        # ``data_mask`` is conceptually a tuple of slices selecting a shard of the
        # local tensor.  For dynamic sparse routing we also allow numpy / sequence
        # indices on axis 0 (e.g. ``(np.ndarray,)``) for fancy indexing.
        # The implementation below supports both.
        data_mask: Optional[Tuple[Any, ...]],
    ) -> TensorMeta:
        """
        Build **shard meta** — describes one sliced shard actually sent to a receiver.

        Derived from ``routing_meta`` (full output tensor meta) by applying the
        route's ``data_mask`` (tuple of slices).

        - ``shape``: computed from data_mask
        - ``global_shape``: preserved from routing_meta
        - ``global_indices``: axis-0 slice of routing_meta's global_indices

        Args:
            routing_meta: Pre-routing meta from ``_build_routing_meta``
            data_mask: Tuple of slices from routing (``None`` = full tensor, no slicing)

        Returns:
            TensorMeta for the shard
        """
        if data_mask is None:
            # No slicing — send the full tensor
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] _build_shard_meta: tensor={routing_meta.tensor_name}, "
                f"no data_mask, returning routing_meta as-is (shape={routing_meta.shape})"
            )
            return routing_meta

        # --- Sliced shape ---
        # Handle several cases:
        # - slice(None) (i.e. [:]) → use routing_meta.shape for extent
        # - slice(a, b) → b - a
        # - np.ndarray / sequence of indices → len(indices)
        # - scalar index (int) → 1
        def _slice_extent(s: Any, dim_idx: int) -> int:
            import numpy as np

            if isinstance(s, slice):
                start = s.start if s.start is not None else 0
                if s.stop is not None:
                    return s.stop - start
                # stop is None → full dimension extent
                if routing_meta.shape and dim_idx < len(routing_meta.shape):
                    return routing_meta.shape[dim_idx] - start
                return 1  # fallback

            # Fancy indexing with an integer array / list of indices
            if isinstance(s, np.ndarray):
                return int(s.shape[0])
            if isinstance(s, (list, tuple)):
                return len(s)

            # Scalar index
            return 1

        sliced_shape = tuple(_slice_extent(s, i) for i, s in enumerate(data_mask))

        # --- Slice global_indices on axis-0 ---
        global_indices: Optional[np.ndarray] = None
        if routing_meta.global_indices is not None and len(data_mask) > 0:
            axis0 = data_mask[0]
            if isinstance(axis0, slice):
                start = axis0.start if axis0.start is not None else 0
                end = axis0.stop  # routing always sets explicit stop
                global_indices = routing_meta.global_indices[start:end]
            else:
                # Support fancy indexing masks such as np.ndarray or list / tuple
                global_indices = routing_meta.global_indices[axis0]

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] _build_shard_meta: tensor={routing_meta.tensor_name}, "
            f"data_mask={data_mask}, routing_shape={routing_meta.shape} -> shard_shape={sliced_shape}, "
            f"global_indices_len={len(global_indices) if global_indices is not None else None}, "
            f"global_shape={routing_meta.global_shape}"
        )

        return TensorMeta(
            tensor_name=routing_meta.tensor_name,
            shape=sliced_shape,
            dtype=routing_meta.dtype,
            global_indices=global_indices,
            global_shape=routing_meta.global_shape,
            source_spmd_id=routing_meta.source_spmd_id,
            source_rank=routing_meta.source_rank,
            forward_step=routing_meta.forward_step,
            is_sparse=routing_meta.is_sparse,
        )

    def _compute_axis0_global_offset(
        self, output_spec: Optional[ShardingSpec], tensor_data: Any
    ) -> int:
        """
        Compute the global offset for axis 0 based on output sharding spec.

        For a Dense ShardingSpec with axis 0 sharded, computes where this rank's
        local data starts in the global tensor on axis 0.

        Args:
            output_spec: Output sharding spec
            tensor_data: The tensor data (needed for local shape)

        Returns:
            Global offset for axis 0 (0 if axis 0 is replicated or spec is None)
        """
        if output_spec is None or output_spec.is_content_addressable:
            return 0

        # We assume the only partitioned axis is axis 0 for this legacy function.
        if "axis_0" not in output_spec.partition:
            # Axis 0 is not sharded (replicated)
            return 0

        if not hasattr(tensor_data, "shape") or len(tensor_data.shape) == 0:
            return 0

        local_axis0 = tensor_data.shape[0]
        num_devices_axis0 = output_spec.mesh.axis_shapes[0]
        local_axis0 * num_devices_axis0

        # WARNING: This assumes a simple chunk sharding on axis 0.
        # This will be entirely removed once AsyncDataDistributor fully takes over.
        return self.rank * local_axis0

    # =========================================================================
    # Meta History (for backward reverse routing)
    # =========================================================================

    def get_forward_step_meta(
        self, forward_step: int, tensor_name: str
    ) -> Optional[StepTensorMeta]:
        """
        Look up forward step's metadata (assembled meta + shard origins)
        for backward reverse routing.

        When backward encounters sparse routing without a route fn, it uses
        the shard_origins recorded here to reverse-route gradients back to
        the original source workers.

        Args:
            forward_step: The forward step whose meta is needed
            tensor_name: Tensor name

        Returns:
            StepTensorMeta with assembled meta and shard origins, or None
        """
        step_metas = self._meta_history[OpType.FORWARD].get(forward_step, {})
        stm = step_metas.get(tensor_name)
        if stm is not None:
            meta = stm.assembled_meta
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] get_forward_step_meta: forward_step={forward_step}, "
                f"tensor={tensor_name}, found=True, shard_origins={len(stm.shard_origins)}"
                + (
                    f", shape={meta.shape}, global_shape={meta.global_shape}"
                    if meta
                    else ""
                )
            )
        else:
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] get_forward_step_meta: forward_step={forward_step}, "
                f"tensor={tensor_name}, found=False"
            )
        return stm

    def get_step_meta(
        self, op_type: OpType, step: int, tensor_name: str
    ) -> Optional[StepTensorMeta]:
        """
        Look up metadata for any (op_type, step).

        Generalised version of ``get_forward_step_meta``.
        """
        step_metas = self._meta_history[op_type].get(step, {})
        stm = step_metas.get(tensor_name)
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] get_step_meta: op={op_type.name}, step={step}, "
            f"tensor={tensor_name}, found={stm is not None}"
        )
        return stm

    def clear_meta_history(
        self, op_type: OpType, up_to_step: Optional[int] = None
    ) -> None:
        """
        Free meta history that is no longer needed.

        Args:
            op_type: Operator type whose history to clear
            up_to_step: If given, only clear steps <= this value.
                        If None, clear all history for this op_type.
        """
        if up_to_step is None:
            num_cleared = len(self._meta_history[op_type])
            self._meta_history[op_type].clear()
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] clear_meta_history: op={op_type.name}, "
                f"cleared all ({num_cleared} steps)"
            )
        else:
            steps_to_remove = [
                s for s in self._meta_history[op_type] if s <= up_to_step
            ]
            for s in steps_to_remove:
                del self._meta_history[op_type][s]
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] clear_meta_history: op={op_type.name}, "
                f"up_to_step={up_to_step}, cleared {len(steps_to_remove)} steps, "
                f"remaining={list(self._meta_history[op_type].keys())}"
            )

    # =========================================================================
    # Backward Reverse Routing (using forward shard origins)
    # =========================================================================

    async def _send_backward_via_reverse_route(
        self,
        *,
        forward_step: int,
        tensor_name: str,
        tensor_data: Any,
        routing_meta: TensorMeta,
        target_op: OpType,
        output_step: int,
        successor_spmd_id: str,
        handlers: Dict[int, Any],
        succ_total_rank: int,
    ) -> None:
        """
        Send backward output using forward-step shard origins (reverse routing).

        When the backward operator's sparse edge has **no route fn**, we look up
        the per-shard origin records saved during the corresponding forward
        assembly.  For each shard origin whose ``source_spmd_id`` matches the
        target successor, we slice the backward output tensor at the same row
        range (for dynamic-assembled tensors) or gather using the original
        ``global_indices`` (for dense-assembled tensors), then send the slice
        to the original ``source_rank``.

        Target ranks that are NOT present in the shard origins receive an empty
        packet so that sparse synchronisation is maintained.

        Args:
            forward_step: The forward execution step whose shard origins to use.
            tensor_name: Tensor name.
            tensor_data: The full backward output tensor.
            routing_meta: Pre-routing meta for the backward output.
            target_op: Successor's target operator type.
            output_step: The step number to stamp on outgoing packets.
            successor_spmd_id: SPMD id of the backward successor.
            handlers: ``{rank: handler}`` for the successor SPMD.
            succ_total_rank: Total number of ranks in the successor SPMD.
        """
        # Look up forward shard origins
        step_tensor_meta = self.get_forward_step_meta(forward_step, tensor_name)
        if step_tensor_meta is None or not step_tensor_meta.shard_origins:
            logger.warning(
                f"[spmd_id={self.spmd_id} rank={self.rank}] _send_backward_via_reverse_route: "
                f"no forward shard origins for step={forward_step}, tensor={tensor_name}. "
                f"Falling back to broadcast (sending full tensor to all successor ranks)."
            )
            # Fallback: broadcast full tensor to every successor rank
            for target_rank in range(succ_total_rank):
                handler = handlers.get(target_rank)
                if handler is None:
                    continue
                shard_meta = self._build_shard_meta(routing_meta, None)
                self._send_tensor(
                    handler,
                    target_op,
                    tensor_name,
                    output_step,
                    tensor_data,
                    (),
                    shard_meta,
                    successor_spmd_id,
                    target_rank,
                )
            return

        # Filter shard origins belonging to the target successor SPMD
        relevant_origins = [
            origin
            for origin in step_tensor_meta.shard_origins
            if origin.source_spmd_id == successor_spmd_id
        ]

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] _send_backward_via_reverse_route: "
            f"forward_step={forward_step}, tensor={tensor_name}, successor={successor_spmd_id}, "
            f"total_origins={len(step_tensor_meta.shard_origins)}, "
            f"relevant_origins={len(relevant_origins)}"
        )

        # Build a set of target ranks that will receive actual data
        routed_ranks: set = set()

        for origin in relevant_origins:
            target_rank = origin.source_rank
            routed_ranks.add(target_rank)
            handler = handlers.get(target_rank)
            if handler is None:
                logger.warning(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] reverse_route: no handler "
                    f"for target_rank={target_rank} in {successor_spmd_id}"
                )
                continue

            # Determine how to slice the backward output
            # Determine how to slice the backward output.
            if origin.local_indices is not None:
                # 使用 forward 记录的 local_indices 在当前 backward 输出 tensor
                # 上进行精确 gather。这里直接先 gather 出一个子 tensor，再以
                # “整体发送”的形式交给 _send_tensor（slices=()）。
                try:
                    ndim = len(routing_meta.shape) if routing_meta.shape else 1
                    index_tuple = (origin.local_indices,) + tuple(
                        slice(None) for _ in range(ndim - 1)
                    )
                    gathered_tensor = tensor_data[index_tuple]
                except Exception as e:
                    logger.warning(
                        f"[spmd_id={self.spmd_id} rank={self.rank}] reverse_route: "
                        f"failed to gather by local_indices for tensor={tensor_name}, "
                        f"falling back to full tensor. error={e}"
                    )
                    gathered_tensor = tensor_data
                    index_tuple = ()

                # 基于 routing_meta 构造 shard_meta，并覆盖 shape/global_indices。
                shard_meta = self._build_shard_meta(
                    routing_meta,
                    None,  # 我们已经在 data 里做了 gather，这里不再依赖 slice mask
                )
                shard_meta = TensorMeta(
                    tensor_name=shard_meta.tensor_name,
                    shape=getattr(gathered_tensor, "shape", shard_meta.shape),
                    dtype=shard_meta.dtype,
                    global_indices=origin.global_indices,
                    global_shape=shard_meta.global_shape,
                    source_spmd_id=self.spmd_id,
                    source_rank=self.rank,
                    forward_step=forward_step,
                    is_sparse=False,
                )

                self._send_tensor(
                    handler,
                    target_op,
                    tensor_name,
                    output_step,
                    gathered_tensor,
                    (),
                    shard_meta,
                    successor_spmd_id,
                    target_rank,
                )
            elif origin.global_indices is not None:
                # 兼容旧逻辑：没有 local_indices 但有 global_indices 的情况，
                # 仍然退化为发送完整 tensor，由接收端自行根据 global_indices 过滤。
                data_mask: Tuple[slice, ...] = (
                    (slice(0, routing_meta.shape[0]),) if routing_meta.shape else ()
                )
                shard_meta = self._build_shard_meta(
                    routing_meta, data_mask if data_mask else None
                )
                shard_meta = TensorMeta(
                    tensor_name=shard_meta.tensor_name,
                    shape=shard_meta.shape,
                    dtype=shard_meta.dtype,
                    global_indices=origin.global_indices,
                    global_shape=shard_meta.global_shape,
                    source_spmd_id=self.spmd_id,
                    source_rank=self.rank,
                    forward_step=forward_step,
                    is_sparse=shard_meta.is_sparse,
                )
                self._send_tensor(
                    handler,
                    target_op,
                    tensor_name,
                    output_step,
                    tensor_data,
                    data_mask,
                    shard_meta,
                    successor_spmd_id,
                    target_rank,
                )
            else:
                # 既没有 local_indices 也没有 global_indices，退化为广播。
                data_mask = ()
                shard_meta = self._build_shard_meta(routing_meta, None)
                self._send_tensor(
                    handler,
                    target_op,
                    tensor_name,
                    output_step,
                    tensor_data,
                    data_mask,
                    shard_meta,
                    successor_spmd_id,
                    target_rank,
                )

        # Send empty packets to successor ranks NOT covered by shard_origins
        for target_rank in range(succ_total_rank):
            if target_rank in routed_ranks:
                continue
            handler = handlers.get(target_rank)
            if handler is not None:
                self._send_empty_packet(
                    handler,
                    target_op,
                    tensor_name,
                    output_step,
                    routing_meta,
                    successor_spmd_id,
                    target_rank,
                )

    def _get_successor_op_type(self, current_op: OpType) -> OpType:
        """Map current op type to successor's op type (deprecated - use topology target_op)."""
        return current_op

    def _self_receive_tensor(
        self, target_op: OpType, name: str, step: int, data: Any, meta: TensorMeta
    ) -> None:
        """
        Self-loop data passing - directly call data receive interface.

        For same-node edges (e.g., forward -> backward), this method
        maintains the data-driven execution model by calling receive_tensor_data
        on self rather than sending via handlers.

        Args:
            target_op: Target operator type (e.g., BACKWARD for forward->backward edge)
            name: Tensor name
            step: Current step number
            data: Tensor data to pass
            meta: Tensor metadata
        """
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Self-loop data passing: {name} step={step} "
            f"-> {target_op.name}"
        )

        # Directly call receive_tensor_data to trigger target op's data readiness check
        asyncio.create_task(
            self.receive_tensor_data(
                op_type=target_op,
                name=name,
                step=step,
                sender_rank=self.rank,
                data=data,
                meta=meta,
            )
        )

    def _send_tensor(
        self,
        handler: Any,
        op_type: OpType,
        name: str,
        step: int,
        data: Any,
        slices: Tuple[slice, ...],
        meta: TensorMeta,
        target_spmd_id: str,
        target_rank: int,
    ) -> None:
        """
        Send tensor data to a worker handler.

        First slices the tensor according to the slice tuple, then sends the sliced data.

        Args:
            handler: Target worker handler
            op_type: Operator type
            name: Tensor name
            step: Step number
            data: Full tensor data
            slices: Tuple of slice objects for each dimension (can be used directly for indexing)
            meta: Tensor metadata
        """
        if handler is None:
            return

        # Slice tensor directly using the slice tuple
        try:
            sliced_data = data[slices] if slices else data
        except (TypeError, IndexError):
            # If slicing fails (e.g., non-array type), use original data
            logger.warning(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Failed to slice tensor with slices {slices}, sending original data"
            )
            sliced_data = data

        if hasattr(handler, "receive_tensor_data"):
            handler.receive_tensor_data.remote(
                op_type, name, step, self.rank, sliced_data, meta
            )
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Worker sending tensor: "
                f"op={op_type.name}, name={name}, step={step}, "
                f"target_spmd_id={target_spmd_id}, target_rank={target_rank}, "
                f"slices={slices}, data_shape={getattr(sliced_data, 'shape', 'N/A')}, "
                f"meta.shape={meta.shape}, meta.global_shape={meta.global_shape}, "
                f"meta.global_indices_len={len(meta.global_indices) if meta.global_indices is not None else None}, "
                f"meta.is_sparse={meta.is_sparse}"
            )
        else:
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Would send tensor {name} to handler: "
                f"step={step}, target_spmd_id={target_spmd_id}, target_rank={target_rank}"
            )

    def _send_empty_packet(
        self,
        handler: Any,
        op_type: OpType,
        name: str,
        step: int,
        base_meta: TensorMeta,
        target_spmd_id: str,
        target_rank: int,
    ) -> None:
        """
        Send empty packet to target worker for sparse routing.

        For sparse routing, all target ranks must receive either actual data or
        an empty packet to ensure proper synchronization.

        Args:
            handler: Target worker handler
            op_type: Operator type
            name: Tensor name
            step: Step number
            base_meta: Base tensor metadata to copy from
        """
        if handler is None:
            return

        empty_meta = TensorMeta(
            tensor_name=base_meta.tensor_name,
            shape=(0,),
            dtype=base_meta.dtype,
            global_shape=base_meta.global_shape,
            source_spmd_id=self.spmd_id,
            source_rank=self.rank,
            forward_step=base_meta.forward_step,
            is_sparse=base_meta.is_sparse,
            is_empty=True,
        )
        empty_data = np.array([])

        if hasattr(handler, "receive_tensor_data"):
            handler.receive_tensor_data.remote(
                op_type, name, step, self.rank, empty_data, empty_meta
            )
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Worker sent empty packet: "
                f"op={op_type.name}, name={name}, step={step}, "
                f"target_spmd_id={target_spmd_id}, target_rank={target_rank}, "
                f"global_shape={empty_meta.global_shape}, is_sparse={empty_meta.is_sparse}"
            )

    async def _self_receive_state(
        self, target_op: OpType, name: str, step: int, data: Any
    ) -> None:
        """
        Self-loop state passing - directly call data receive interface.

        For same-node edges (e.g., forward -> backward), this method
        maintains the data-driven execution model by calling receive_state_data
        on self rather than sending via control plane.

        Args:
            target_op: Target operator type
            name: State variable name
            step: Current step number
            data: State data to pass
        """
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Self-loop state passing: {name} step={step} "
            f"-> {target_op.name}"
        )

        # Directly call receive_state_data to trigger target op's data readiness check
        await self.receive_state_data(
            op_type=target_op, name=name, step=step, data=data
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
        target_op: OpType,
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
            self._driver_handle.forward_state_argument_to_controller.remote(
                target_spmd_id, target_op, state_name, step, state_value
            )
            logger.debug(
                f"[spmd_id={self.spmd_id}] Worker {self.rank} sent state_argument '{state_name}' "
                f"to SPMD {target_spmd_id}, target_op={target_op.name}, step={step}"
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
        elif not self._op_runnable[OpType.FORWARD].is_set():
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
            op_steps={op.name: step for op, step in self._current_steps.items()},
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
            for op_type, step in state["current_steps"].items():
                self._current_steps[op_type] = step
        if "static_args" in state:
            for name, value in state["static_args"].items():
                self._op_manager.set_static_arg(name, value)

    # =========================================================================
    # Utility
    # =========================================================================

    def get_current_step(self, op_type: OpType) -> int:
        """Get current step for an operator type."""
        return self._current_steps[op_type]

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
