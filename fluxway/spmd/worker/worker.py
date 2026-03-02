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
7. Handle control_argument reception and propagate to successors

This implementation uses Ray's async actor API for concurrency.
See: https://docs.ray.io/en/latest/ray-core/actors/async_api.html
"""

import asyncio
from collections import defaultdict
import logging
import time

import ray

from fluxway.common.types import (
    SPMDConfig,
    WorkerHeartbeat,
    WorkerStatus,
    WorkerTelemetry,
)
from fluxway.common.index_descriptor import IndexDescriptor
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.sharding import ShardingSpec, PartitionSpec
from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_type_hints
from fluxway.common.backend import get_backend
from fluxway.common.types.configs import MailboxStatus
from fluxway.common.utils import ensure_logging_configured
from fluxway.rpc.protocol import DataPlaneCallable
from .mailbox import WorkerMailbox
from .operator_manager import OperatorManager
from fluxway.transport.communication import AsyncDataDistributor


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
    - Handles control_argument for cross-SPMD state transfer

    Uses asyncio for concurrency instead of threading.
    Ray will automatically detect this as an async actor due to async method definitions.
    """

    def __init__(
        self,
        config: SPMDConfig,
        rank: int,
        spmd_id: str = "",
        driver_handle: Optional[Any] = None,
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

        # Mesh from config
        self._mesh = config.mesh
        self._world_size = config.num_workers

        self._op_manager = OperatorManager(config, rank, spmd_id)

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

        # Source operator pause control
        self._source_pause_events: Dict[str, asyncio.Event] = {
            op_name: asyncio.Event()
            for op_name in (self.config.operators if self.config else {})
            if self._op_manager.is_source_operator(op_name)
        }
        for event in self._source_pause_events.values():
            event.set()  # Initially all sources are running

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

        logger.debug(f"SPMD Worker initialized: spmd_id={spmd_id}, rank={rank}")

        # Start heartbeat if driver_handle is provided
        if self._driver_handle is not None:
            self._start_heartbeat()

        # Start the run loop in background (will block until start() is called)
        self._start_run_loop()

        # Inject operator context for this thread
        from fluxway.common.decorators import operator_context

        operator_context(self.config.static_args, self.rank, self._world_size)

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
        use_global_step_indexing = self.config.use_global_step_indexing
        if not use_global_step_indexing:
            return input_step
        if is_periodic:
            return self._world_size * current_step + self.rank

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

        logger.debug(
            f"[{self.spmd_id} Rank {self.rank}] _check_and_auto_start: "
            f"pred={len(self._predecessor_handlers)}/{self._expected_predecessor_count} "
            f"succ={len(self._successor_handlers)}/{self._expected_successor_count}"
        )

        if pred_registered and succ_registered:
            logger.debug(
                f"[spmd_id={self.spmd_id}] Worker {self.rank}: All handlers registered "
                f"(pred={len(self._predecessor_handlers)}/{self._expected_predecessor_count}, "
                f"succ={len(self._successor_handlers)}/{self._expected_successor_count}), "
                f"auto-starting execution"
            )
            self._auto_start()

    def _auto_start(self) -> None:
        """Internal auto-start method."""
        logger.debug(
            f"[{self.spmd_id} Rank {self.rank}] _auto_start CALLED. _running={self._running}"
        )
        if self._running:
            return

        self._running = True
        self._start_event.set()
        logger.debug(f"[spmd_id={self.spmd_id}] Worker {self.rank} auto-started")

        # Trigger source operators (operators with 0 input tensors)
        asyncio.create_task(self._trigger_source_operators())

    async def _trigger_source_operators(self) -> None:
        """
        Trigger initial steps for source operators (those with 0 input tensors).
        """
        source_ops = self._op_manager.get_source_operators()
        for op_name in source_ops:
            logger.debug(
                f"[{self.spmd_id} Rank {self.rank}] Enqueuing source op: {op_name}"
            )
            await self._enqueue(op_name, 0)

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
        # logger.debug(f"[spmd_id={self.spmd_id}] Worker {self.rank}: {op_name} blocked")

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

        # logger.debug(
        #     f"[spmd_id={self.spmd_id}] Worker {self.rank}: {op_name} unblocked"
        # )

    def is_op_blocked(self, op_name: str) -> bool:
        """Check if an operator is blocked."""
        return not self._op_runnable[op_name].is_set()

    # =========================================================================
    # Data Reception
    # =========================================================================

    async def receive_control(
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

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] RECEIVED STATES: op={op_name}, step={step}, names={names}"
        )

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
        names: List[str],
        step: int,
        sender_id: str,
        sender_rank: int,
        payload_groups: List[List[Any]],
        counts_groups: List[List[int]],
        is_ragged: Optional[List[bool]] = None,
        has_spatial_indices: Optional[List[bool]] = None,
        has_origin_indices: Optional[List[bool]] = None,
    ) -> int:
        """Receive multiple tensor shards from upstream worker (data plane).

        Args:
            op_name: Target operator name
            names: List of tensor names in the bundle
            step: Execution step
            sender_id: Source SPMD ID
            sender_rank: Source worker rank
            payload_groups: List of payload lists (one per tensor)
            counts_groups: List of count lists (one per tensor)
            is_ragged: Flags for ragged layout per tensor (default False)
            has_spatial_indices: Flags for spatial indices per tensor (default False)
            has_origin_indices: Flags for logical assembly indices per tensor (default False)
        """
        # Ensure metadata lists are provided, or default to False
        num = len(names)
        is_ragged = is_ragged if is_ragged is not None else [False] * num
        has_spatial_indices = (
            has_spatial_indices if has_spatial_indices is not None else [False] * num
        )
        has_origin_indices = (
            has_origin_indices if has_origin_indices is not None else [False] * num
        )
        """
        Receive multiple tensor shards from upstream worker (data plane).

        Args:
            op_name: Operator name
            names: Tensor variable names
            step: Step number
            sender_rank: Sender's rank
            is_ragged: Indicates if ragged lengths are included for each tensor
            has_spatial_indices: Indicates if routing indices are included for each tensor
            has_origin_indices: Indicates if assembly (logical) indices are included for each tensor
        """
        if len(names) != len(set(names)):
            raise ValueError(f"[receive_shards] Duplicate names in bundle: {names}")

        # Lazily declare the step barrier on first data arrival
        self._ensure_step_declared(op_name, step)

        # Cache in mailbox — barrier fires on_ready automatically
        watermark = await self._mailbox.deposit_shards(
            op_name,
            names,
            step,
            sender_id,
            sender_rank,
            payload_groups,
            counts_groups,
            is_ragged,
            has_spatial_indices,
            has_origin_indices,
        )
        return watermark

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
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Step ready: op={op_name}, step={step}"
        )
        asyncio.create_task(self._enqueue(op_name, step))

    async def _enqueue(self, op_name: str, step: int) -> None:
        """Enqueue a task for execution."""
        from fluxway.common.types.configs import OperatorMeta

        logger.debug(
            f"[Worker {self.spmd_id} Rank {self.rank}] _enqueue CALLED: op={op_name}, step={step}"
        )
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
        """
        # Wait for start signal from driver
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} waiting for start signal..."
        )
        await self._start_event.wait()

        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} execution loop started"
        )

        # Normal SPMD mode: data-driven execution loop
        while self._running:
            try:
                # Blocking wait for next task (async)
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Execution loop waiting for task in _execution_queue..."
                )
                priority, step, op_name = await self._execution_queue.get()
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Execution loop picked up task: {op_name} step={step} priority={priority}"
                )

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
                        f"[spmd_id={self.spmd_id}] OP BLOCKED in queue: op={op_name}, step={step}"
                    )
                    continue

                logger.debug(
                    f"[{self.spmd_id}] Task processing: op={op_name} step={step}"
                )
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
                result = await self._execute_operator(op_name, step)

                # Advance execution step counter
                await self._advance_step(op_name, result=result)

            except asyncio.CancelledError:
                logger.debug(
                    f"[spmd_id={self.spmd_id}] Worker {self.rank} run loop cancelled"
                )
                break
            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Error in execution loop: {e}"
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

        logger.debug(f"[spmd_id={self.spmd_id}] Worker {self.rank} stopping")

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
            for key, tensor_list in shards.items():
                backend = get_backend()
                for tensor in tensor_list:
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
        # Deduplicate to avoid redundant transmissions
        successor_ids = sorted(set(successor_ids))
        if not successor_ids:
            return

        is_periodic = self._op_manager.is_periodic_trigger(op_name)
        current_step = self._current_steps[op_name]
        output_step = self._resolve_output_step(
            is_periodic=is_periodic, current_step=current_step, input_step=step
        )

        for spmd_id in successor_ids:
            succ_info = self._op_manager.get_successor_info(op_name, spmd_id)
            if succ_info is None:
                continue

            target_op = succ_info.target_op or op_name
            for tensor_name in output_sharding:
                if succ_info.tensor_map:
                    if tensor_name not in succ_info.tensor_map:
                        continue
                    target_tensor_name = succ_info.tensor_map[tensor_name]
                else:
                    target_tensor_name = tensor_name

                # Create an empty TransportTensor
                backend = get_backend()
                empty_tensor = TransportTensor.from_tensor(
                    backend.zeros((0,), dtype=backend.float32())
                )

                if succ_info.is_self_loop:
                    await self._self_receive_data(
                        target_op, target_tensor_name, output_step, empty_tensor
                    )
                else:
                    handlers = self._successor_handlers.get(spmd_id, {})
                    # Find actual consumers for this tensor to avoid unnecessary all-rank broadcasts
                    source_spec = output_sharding.get(tensor_name)
                    target_spec = succ_info.tensor_sharding.get(tensor_name)

                    if source_spec and target_spec:
                        consumers = source_spec.resolve_consumers(
                            target_spec, self.rank
                        )
                    else:
                        # Fallback to broadcast if specs missing
                        consumers = range(succ_info.world_size)

                    for target_rank in consumers:
                        handler = handlers.get(target_rank)
                        if handler is not None:
                            await handler.receive_shards(
                                target_op,
                                [target_tensor_name],
                                output_step,
                                self.spmd_id,
                                self.rank,
                                [[empty_tensor.flat_values]],
                                [[0]],
                                [False],
                                [False],
                                [False],
                            )

    async def _execute_operator(self, op_name: str, step: int) -> Any:
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
                return None

            # Assemble input data
            try:
                input_data = await self._assemble_input(op_name, step)
            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Failed to assemble input data for "
                    f"op={op_name}, step={step}: {e}"
                )
                raise e

            # Execute operator in a thread to avoid blocking the worker event loop
            if isinstance(input_data, dict):
                result = await asyncio.to_thread(
                    self._op_manager.execute, op_name, **input_data
                )
            else:
                result = await asyncio.to_thread(
                    self._op_manager.execute, op_name, input_data
                )
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Execution finished: op={op_name}, step={step}"
            )

            if result is None:
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Operator {op_name} returned None. "
                    f"Continuing to advance step."
                )

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
                    f"output_step={output_step}, rank={self.rank}, world_size={self._world_size}, "
                    f"use_global_step_indexing={self.config.use_global_step_indexing}"
                )

            # Send output
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] About to send output: op={op_name}, "
                f"output_step={output_step}, result_type={type(result).__name__}"
            )
            await self._send_output(op_name, output_step, result, exec_step=step)

            # Clear mailbox
            await self._mailbox.clear_step(op_name, step)

            return result

        except Exception as e:
            import traceback

            logger.error(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Operator {op_name} execution failed: {e}, traceback: {traceback.format_exc()}"
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

        # Arival maybe a list of tensors or a single tensor
        tensor_args: Dict[str, Optional[TransportTensor]] = {}
        for tensor_name in expected_tensor_names:
            if tensor_name not in all_tensor_shards:
                raise ValueError(
                    f"Missing tensor '{tensor_name}' for operator {op_name} at step {step}"
                )

            shards = all_tensor_shards[tensor_name]

            if is_periodic:
                # Periodic mode: merge all shards arrived for this periodic trigger
                # We use _assemble_to_dynamic_tensor which simply concatenates all shards.
                merged_tensor = self._assemble_to_dynamic_tensor(tensor_name, shards)
                tensor_args[tensor_name] = merged_tensor
            else:
                # Normal mode: assemble tensor based on sharding
                assembled_tensor = await self._assemble_tensor(
                    op_name, tensor_name, shards
                )
                tensor_args[tensor_name] = assembled_tensor

        # Cache assembled tensors (they carry their own indices/meta)
        self._last_assembled_tensors[op_name] = tensor_args

        # Task 5: Construct final keyword arguments
        input_types = self._op_manager.get_operator_input_types(op_name)
        if not input_types and not self._op_manager.is_source_operator(op_name):
            raise ValueError(
                f"Cannot determine input types for operator {op_name}. "
                f"Make sure the operator function has type annotations."
            )

        # Combine all arguments (including flattened tensor paths)
        all_args = {}
        all_args.update(static_args)
        all_args.update(state_args)
        all_args.update(tensor_args)

        input_kwargs = {}

        from fluxway.common.decorators import _is_fluxway_data_type

        data_type_count = sum(
            1 for tp in input_types.values() if _is_fluxway_data_type(tp)
        )
        use_prefix = data_type_count > 1

        for p_name, p_type in input_types.items():
            if _is_fluxway_data_type(p_type):
                try:
                    prefix = p_name if use_prefix else ""
                    input_kwargs[p_name] = self._reconstruct_data_type(
                        p_type, all_args, prefix=prefix
                    )
                except Exception as e:
                    logger.error(
                        f"[spmd_id={self.spmd_id} rank={self.rank}] Failed to reconstruct "
                        f"parameter {p_name} for op={op_name}: {e}"
                    )
            else:
                if p_name in all_args:
                    input_kwargs[p_name] = all_args[p_name]

        return input_kwargs

    def _get_values(self, obj: Any) -> Any:
        """Helper to get raw values from either TransportTensor or raw array."""
        if isinstance(obj, TransportTensor):
            return obj.flat_values
        return obj

    def _get_spatial_indices(self, obj: Any) -> Optional[Any]:
        """Helper to get global indices from TransportTensor."""
        if isinstance(obj, TransportTensor):
            return obj.spatial_indices
        return None

    def _flatten_shards(
        self, shards: Dict[Any, List[TransportTensor]], index: int = 0
    ) -> List[TransportTensor]:
        """Flatten shards from a specific index in the payload lists across all senders."""
        flat = []
        for key in sorted(shards.keys()):
            tensor_list = shards[key]
            if index < len(tensor_list):
                flat.append(tensor_list[index])
        return flat

    def _flatten_shards_with_ranks(
        self, shards: Dict[Any, List[TransportTensor]], index: int = 0
    ) -> List[Tuple[Any, TransportTensor]]:
        """Flatten shards and include their sender keys (ranks)."""
        flat = []
        for key in sorted(shards.keys()):
            tensor_list = shards[key]
            if index < len(tensor_list):
                flat.append((key, tensor_list[index]))
        return flat

    async def _assemble_tensor(
        self, op_name: str, tensor_name: str, shards: Dict[Any, List[TransportTensor]]
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
                (spec := p.tensor_sharding.get(tensor_name)) is not None
                and spec.is_content_addressable
                for p in predecessor_topology.values()
            )

            if has_dynamic_predecessor:
                return self._assemble_dynamic_to_static_tensor(
                    op_name, tensor_name, shards, input_spec
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
        shards: Dict[Any, List[TransportTensor]],
        input_spec: ShardingSpec,
        predecessor_topology: Dict[str, Any],
    ) -> TransportTensor:
        """Assemble static-to-static tensor using exact physical shards."""
        backend = get_backend()
        shard_list = []
        for pre_spmd_id, pre_info in predecessor_topology.items():
            source_spec = self._op_manager.get_tensor_input_sharding_spec(
                op_name, pre_spmd_id, tensor_name
            )
            if source_spec:
                deps = source_spec.resolve_producers(input_spec, self.rank)
                for rank in deps:
                    key = (pre_spmd_id, rank)
                    if key in shards:
                        shard_list.extend(
                            self._flatten_shards({key: shards[key]}, index=0)
                        )
            else:
                for rank in range(pre_info.world_size):
                    key = (pre_spmd_id, rank)
                    if key in shards:
                        shard_list.extend(
                            self._flatten_shards({key: shards[key]}, index=0)
                        )

        if not shard_list:
            raise ValueError(f"No shards for {tensor_name}")

        if len(shard_list) == 1:
            return shard_list[0]

        # Concatenate shards using backend
        flat_values = [self._get_values(s) for s in shard_list]
        merged_values = backend.cat(flat_values, dim=0)

        merged_indices = None
        if all(s.index_descriptor is not None for s in shard_list):
            descriptors = [s.index_descriptor for s in shard_list]
            merged_indices = IndexDescriptor.concatenate(descriptors)

        # Propagate global_shape from shards
        first_shard = shard_list[0]
        global_shape = (
            first_shard.global_shape if hasattr(first_shard, "global_shape") else None
        )

        return TransportTensor(
            merged_values, spatial_indices=merged_indices, global_shape=global_shape
        )

    def _assemble_dynamic_to_static_tensor(
        self,
        op_name: str,
        tensor_name: str,
        shards: Dict[Any, List[TransportTensor]],
        input_spec: ShardingSpec,
    ) -> TransportTensor:
        """Assemble dynamic-to-static (Sparse-to-Dense) tensor using spatial_indices."""
        backend = get_backend()

        # Consistent flattening of multi-tensor parameters
        valid_shards = [
            s
            for s in self._flatten_shards(shards)
            if backend.shape(self._get_values(s))[0] > 0
        ]
        if not valid_shards:
            return TransportTensor.from_tensor(
                backend.zeros((0,), dtype=backend.float32())
            )

        # Simple 1D path for now (Sparse-to-Dense typically assumes 1D indices for scattering)
        all_values = backend.cat([self._get_values(s) for s in valid_shards], dim=0)
        # Materialize indices if needed
        all_indices_list = []
        for s in valid_shards:
            # For assembly into a static tensor, we prioritize origin_indices (logical offsets)
            idx = s.origin_indices
            if idx is None:
                # Fallback to spatial_indices (global keys) - works if they represent global addresses
                idx = self._get_spatial_indices(s)

            if idx is None:
                # Fallback to local arange
                num = backend.shape(self._get_values(s))[0]
                idx = backend.arange(0, num, dtype=backend.int32())
            all_indices_list.append(idx)
        all_indices = backend.cat(all_indices_list, dim=0)

        # Infer global length for the routed dimension (usually batch axis)
        max_idx = int(backend.item(backend.max(all_indices)))
        inferred_dim_size = max_idx + 1

        # Multi-dimensional GSPMD support: Combine inferred dim size with static feature dimensions.
        val_shape = list(backend.shape(all_values))
        routing_dim = getattr(input_spec, "routing_dim", 0)
        if isinstance(routing_dim, (list, tuple)):
            routing_dim = routing_dim[
                0
            ]  # TODO(He Jia): Only support 1D routing dim for now

        global_shape_list = val_shape
        global_shape_list[routing_dim] = inferred_dim_size
        global_shape = tuple(global_shape_list)

        layout = input_spec._to_layout(global_shape)
        local_shape = layout.get_shard_shape(self.rank)
        slice_start, slice_end = layout.get_shard_bounds(self.rank, 0)

        mask = (all_indices >= slice_start) & (all_indices < slice_end)
        local_indices = all_indices[mask] - slice_start
        local_values = all_values[mask]

        assembled_data = backend.zeros(local_shape, dtype=all_values.dtype)
        # Backend now supports indexed update natively
        assembled_data = backend.index_update(
            assembled_data, local_indices, local_values
        )

        return TransportTensor.from_tensor(assembled_data, global_shape=global_shape)

    def _assemble_to_dynamic_tensor(
        self, tensor_name: str, shards: Dict[Any, List[TransportTensor]]
    ) -> TransportTensor:
        """Assemble to-dynamic (Any -> Dynamic) tensor by merging shards."""
        backend = get_backend()
        shard_list = self._flatten_shards_with_ranks(shards, index=0)
        valid_tensors = [
            (r, t) for r, t in shard_list if backend.shape(self._get_values(t))[0] > 0
        ]

        if not valid_tensors:
            # Derive dtype from received shards (even if empty) to maintain consistency
            dtype = shard_list[0][1].dtype if shard_list else backend.float32()
            return TransportTensor.from_tensor(backend.zeros((0,), dtype=dtype))

        merged_values = backend.cat(
            [self._get_values(t) for r, t in valid_tensors], dim=0
        )

        merged_indices = None
        if all(self._get_spatial_indices(t) is not None for r, t in valid_tensors):
            merged_indices = backend.cat(
                [self._get_spatial_indices(t) for r, t in valid_tensors], dim=0
            )

        merged_origin_indices = None
        if all(
            hasattr(t, "origin_indices") and t.origin_indices is not None
            for r, t in valid_tensors
        ):
            merged_origin_indices = backend.cat(
                [t.origin_indices for r, t in valid_tensors], dim=0
            )

        # DETERMINISTIC SORTING: Restore original order using origin_indices
        # This is critical for sparse-to-dense flows where the downstream expects
        # a specific sequence (e.g., for inv_map indexing).
        if merged_origin_indices is not None:
            # Sort all merged buffers based on origin_indices
            sort_idx = backend.argsort(merged_origin_indices)

            merged_values = backend.take(merged_values, sort_idx)
            if merged_indices is not None:
                merged_indices = backend.take(merged_indices, sort_idx)

            # Update origin_indices to be sorted (ascending)
            merged_origin_indices = backend.take(merged_origin_indices, sort_idx)

        global_shape = backend.shape(merged_values)

        return TransportTensor(
            merged_values,
            spatial_indices=merged_indices,
            global_shape=global_shape,
            origin_indices=merged_origin_indices,
        )

    async def _advance_step(self, op_name: str, result: Any = "not_none") -> None:
        """Advance execution step counter for an operator."""
        old_step = self._current_steps[op_name]
        self._current_steps[op_name] += 1
        new_step = self._current_steps[op_name]
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] _advance_step: op={op_name}, {old_step} -> {new_step}"
        )

        # If it's a source operator, automatically trigger the next step
        if self._op_manager.is_source_operator(op_name):
            # Check for global step limit (max_steps)
            max_steps = self.config.max_steps
            if max_steps is not None and new_step >= max_steps:
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Source operator '{op_name}' "
                    f"reached max_steps ({max_steps}). Try to be terminated by driver."
                )
                # Do NOT enqueue new_step + 1, effectively stopping the pulse
            elif result is not None:
                # Wait if paused by backpressure
                if op_name in self._source_pause_events:
                    await self._source_pause_events[op_name].wait()

                asyncio.create_task(
                    self._enqueue(op_name, new_step)
                )  # Use new_step for next step
            else:
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Source operator {op_name} finished (returned None)"
                )

        # Report op completion to driver (here new_step means water level)
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
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Reporting op completion: {op_name}, step={step}"
                )
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
        return target_spec.is_content_addressable

    async def _send_output(
        self, op_name: str, step: int, result: Any, *, exec_step: Optional[int] = None
    ) -> None:
        """Send output to successor SPMDs using TransportTensor."""
        if exec_step is None:
            exec_step = step

        output_sharding = self._op_manager.get_output_sharding(op_name)
        tensors_to_send = self._extract_tensors(result, output_sharding)
        successor_ids = sorted(set(self._op_manager.get_successor_ids(op_name)))
        input_tensors = self._last_assembled_tensors.get(op_name, {})

        for spmd_id in successor_ids:
            succ_info = self._op_manager.get_successor_info(op_name, spmd_id)
            if succ_info is None:
                continue

            target_op = succ_info.target_op or op_name

            # batches: Groups tensors by their sharding transformation (source -> target).
            # This allows AsyncDataDistributor to optimize routing for multiple tensors at once.
            # Key: (source_spec, target_spec)
            # Value: list of (tensor_name, target_tensor_name, transport_tensor, propagate_indices)
            batches = defaultdict(list)

            for tensor_name, tensor_data in tensors_to_send.items():
                # 1. Map output tensor name to successor's expected input name
                if succ_info.tensor_map is not None:
                    if tensor_name not in succ_info.tensor_map:
                        continue
                    target_tensor_name = succ_info.tensor_map[tensor_name]
                else:
                    target_tensor_name = tensor_name

                # 2. Decide if we need to send global indices (e.g., for Hash/Dynamic routing)
                propagate_indices = self._should_propagate_indices(
                    op_name, spmd_id, tensor_name
                )

                # 3. Try to retrieve indices from original input for passthrough/metadata
                input_tensor = input_tensors.get(tensor_name)
                indices = None

                if input_tensor is not None:
                    if propagate_indices:
                        indices = input_tensor.spatial_indices
                    else:
                        indices = (
                            input_tensor.index_descriptor
                            if input_tensor.index_descriptor is not None
                            else input_tensor.spatial_indices
                        )

                # 4. Wrap data into TransportTensor for the communication layer
                if isinstance(tensor_data, TransportTensor):
                    transport_tensor = tensor_data
                    # Attach indices if the current TransportTensor is missing them
                    if indices is not None and transport_tensor.spatial_indices is None:
                        transport_tensor = TransportTensor(
                            transport_tensor.flat_values,
                            nested_splits=transport_tensor.nested_splits,
                            spatial_indices=indices,
                            global_shape=transport_tensor.global_shape,
                            origin_indices=transport_tensor.origin_indices,
                        )
                else:
                    transport_tensor = TransportTensor.from_tensor(
                        tensor_data, spatial_indices=indices
                    )

                # 5. Fast path for self-loops: bypass network and send to self
                if succ_info.is_self_loop:
                    await self._self_receive_data(
                        target_op, target_tensor_name, step, transport_tensor
                    )
                    continue

                # 6. Resolve ShardingSpecs for both source and target sides
                source_spec = output_sharding.get(tensor_name)
                if source_spec is None:
                    source_mesh = self._mesh
                    if source_mesh is None:
                        source_mesh = DeviceMesh(
                            device_shape=(1,), axis_names=("data",)
                        )
                    source_spec = ShardingSpec.chunk(source_mesh, PartitionSpec())
                else:
                    # Infer global shape for static specs to avoid geometry collapse in the transport layer
                    if (
                        transport_tensor.global_shape is None
                        and not source_spec.is_content_addressable
                    ):
                        from fluxway.common.backend import get_backend

                        backend = get_backend()
                        if transport_tensor.is_ragged:
                            dim0 = (
                                backend.shape(transport_tensor.nested_splits[0])[0] - 1
                            )
                            inferred_shape = [int(dim0)]
                        else:
                            inferred_shape = list(
                                backend.shape(transport_tensor.flat_values)
                            )

                        for dim, axes in source_spec.partitions.items():
                            if dim < len(inferred_shape):
                                dim_multiplier = 1
                                for ax in axes:
                                    src_mesh = source_spec.mesh
                                    if src_mesh is None:
                                        raise ValueError(
                                            f"Mesh of {source_spec.name} is None. Please provide a mesh for {source_spec.name} with the @spmd decorator."
                                        )
                                    ax_idx = src_mesh.axis_names.index(ax)
                                    dim_multiplier *= src_mesh.shape[ax_idx]
                                inferred_shape[dim] = int(
                                    inferred_shape[dim] * dim_multiplier
                                )

                        transport_tensor = TransportTensor(
                            transport_tensor.flat_values,
                            nested_splits=transport_tensor.nested_splits,
                            spatial_indices=transport_tensor.spatial_indices,
                            global_shape=tuple(inferred_shape),
                            origin_indices=transport_tensor.origin_indices,
                        )
                # Use local tensor name for successor spec lookup (as mapped by Builder)
                target_spec = succ_info.tensor_sharding.get(tensor_name)
                if target_spec is None:
                    raise KeyError(
                        f"Sharding spec for local tensor '{tensor_name}' (target '{target_tensor_name}') "
                        f"not found in successor '{spmd_id}' topology info. "
                        f"Available tensors: {list(succ_info.tensor_sharding.keys())}"
                    )

                # 7. Collect into batches for efficient bulk distribution
                batches[(source_spec, target_spec)].append(
                    (
                        tensor_name,
                        target_tensor_name,
                        transport_tensor,
                        propagate_indices,
                    )
                )

            handlers = self._successor_handlers.get(spmd_id, {})
            if not handlers:
                continue

            distributor = AsyncDataDistributor(self.rank)

            for (source_spec, target_spec), batch in batches.items():
                batch_tensor_names = []
                batch_target_names = []
                batch_transport_tensors = []
                batch_is_ragged = []

                for t_name, t_target_name, t_tensor, _ in batch:
                    batch_tensor_names.append(t_name)
                    batch_target_names.append(t_target_name)
                    batch_transport_tensors.append(t_tensor)
                    batch_is_ragged.append(t_tensor.is_ragged)

                async def _send_fn(
                    target_rank: int,
                    tensors: list,
                    counts: list,
                    has_spatial_indices_list: Optional[list] = None,
                    is_ragged_list: Optional[list] = None,
                    has_origin_indices_list: Optional[list] = None,
                    *,
                    _handlers=handlers,
                    _target_op=target_op,
                    _target_names=batch_target_names,
                    _step=step,
                    _is_ragged=batch_is_ragged,
                    _batch_tensors=batch_transport_tensors,
                ) -> bool:
                    handler = _handlers.get(target_rank)
                    if handler is None:
                        logger.warning(
                            f"[Worker {_step}] handler for {self.spmd_id} target_rank={target_rank} is None"
                        )
                        return False

                    # Send bundle to target rank
                    await self._send_semaphore.acquire()

                    count = int(counts[0]) if counts else 0
                    if count == 0:
                        from fluxway.common.backend import get_backend

                        backend = get_backend()
                        # Use actual batch_tensors metadata to create correctly typed empty shards
                        empty_shards = [
                            [backend.zeros((0,), dtype=tt.dtype)]
                            for tt in _batch_tensors
                        ]
                        ref_task = handler.receive_shards(
                            _target_op,
                            _target_names,
                            _step,
                            self.spmd_id,
                            self.rank,
                            empty_shards,
                            [[0] for _ in _target_names],
                            is_ragged=[tt.is_ragged for tt in _batch_tensors],
                            has_spatial_indices=[
                                tt.spatial_indices is not None for tt in _batch_tensors
                            ],
                            has_origin_indices=[
                                tt.origin_indices is not None for tt in _batch_tensors
                            ],
                        )
                    else:
                        # AsyncDataDistributor yields a flat list of tensors.
                        # Since we packed multiple TransportTensors, we need to unflatten them
                        # back into groups per tensor to send to the worker.
                        # Each tt yields (data, optional lengths, optional indices)
                        # The layout in `tensors` is deterministic based on `payload_tensors.append`
                        payload_groups: List[List[Any]] = []
                        counts_groups: List[List[int]] = []
                        cursor = 0
                        c_cursor = 0

                        # Global Routing indices (appended ONCE for the whole bundle at the end in communication.py)
                        # We extract it now to insert it into the correct tensor's group during unflattening.
                        global_indices_tensor = None
                        global_indices_count = None
                        if (
                            has_spatial_indices_list
                            and any(has_spatial_indices_list)
                            and len(tensors) > cursor
                        ):
                            # The last tensor in the flat list is the global routing indices
                            global_indices_tensor = tensors[-1]
                            global_indices_count = counts[-1]

                        for i, (ragged, has_asmb) in enumerate(
                            zip(_is_ragged, has_origin_indices_list, strict=True)  # type: ignore
                        ):
                            group_payloads = []
                            group_counts = []

                            # Primary data
                            if cursor < len(tensors) and c_cursor < len(counts):
                                group_payloads.append(tensors[cursor])
                                group_counts.append(counts[c_cursor])
                                cursor += 1
                                c_cursor += 1
                            else:
                                raise RuntimeError(
                                    f"Unexpected end of tensors/counts while unflattening primary data for {op_name}"
                                )

                            # Ragged lengths
                            if ragged:
                                if cursor < len(tensors) and c_cursor < len(counts):
                                    group_payloads.append(tensors[cursor])
                                    group_counts.append(counts[c_cursor])
                                    cursor += 1
                                    c_cursor += 1
                                else:
                                    raise RuntimeError(
                                        f"Unexpected end of tensors/counts while unflattening lengths for ragged tensor in {op_name}"
                                    )

                            # Spatial Indices (Routing)
                            # Shared spatial indices are appended to the first payload group if flagged
                            if (
                                has_spatial_indices_list
                                and has_spatial_indices_list[i]
                                and global_indices_tensor is not None
                            ):
                                group_payloads.append(global_indices_tensor)
                                group_counts.append(global_indices_count)

                            # Assembly indices (Origin)
                            if has_asmb:
                                if cursor < len(tensors) and c_cursor < len(counts):
                                    group_payloads.append(tensors[cursor])
                                    group_counts.append(counts[c_cursor])
                                    cursor += 1
                                    c_cursor += 1
                                else:
                                    raise RuntimeError(
                                        f"Unexpected end of tensors/counts while unflattening assembly indices for {op_name}"
                                    )

                            payload_groups.append(group_payloads)
                            counts_groups.append(group_counts)

                        # Skip the global indices in tensors list if we consumed them
                        if global_indices_tensor is not None:
                            cursor += 1
                            c_cursor += 1

                        ref_task = handler.receive_shards(
                            _target_op,
                            _target_names,
                            _step,
                            self.spmd_id,
                            self.rank,
                            payload_groups,
                            counts_groups,
                            is_ragged=is_ragged_list or _is_ragged,
                            has_spatial_indices=has_spatial_indices_list,
                            has_origin_indices=has_origin_indices_list,
                        )

                    async def _wait_and_release(t=ref_task, target_r=target_rank):
                        try:
                            watermark = await t
                            if (
                                isinstance(watermark, int)
                                and watermark > self._high_watermark
                            ):
                                await asyncio.sleep(0.05)
                        except Exception as e:
                            logger.error(
                                f"[spmd_id={self.spmd_id}] Error tracing ACK for backpressure: {e}",
                                exc_info=True,
                            )
                        finally:
                            self._send_semaphore.release()

                    asyncio.create_task(_wait_and_release())
                    return True

                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Distributing batch for target SPMD {spmd_id}: "
                    f"tensor_names={batch_tensor_names}, target_names={batch_target_names}, "
                    f"source_spec={source_spec}, target_spec={target_spec}"
                )
                await distributor.distribute(
                    batch_transport_tensors,
                    source=source_spec,
                    target=target_spec,
                    send_fn=_send_fn,
                )

        # --- Send control_argument via control plane ---
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

                target_op = succ_info.target_op or op_name

                # Collect all state names and values for this target
                names_to_send = []
                values_to_send = []
                for state_name, state_value in state_data.items():
                    target_state_name = succ_info.tensor_map.get(state_name, state_name)
                    if succ_info.is_self_loop:
                        # Self-loop: directly set as static arg
                        await self._self_receive_state(
                            target_op, target_state_name, step, state_value
                        )
                    elif self.rank == 0:
                        names_to_send.append(target_state_name)
                        values_to_send.append(state_value)

                if names_to_send:
                    logger.debug(
                        f"[spmd_id={self.spmd_id}] Worker {self.rank} sending control_arguments: names={names_to_send}, "
                        f"step={step} to {spmd_id}"
                    )
                    if self._driver_handle is not None:
                        try:
                            self._driver_handle.routing_control_argument_to_controller.remote(
                                spmd_id, target_op, names_to_send, step, values_to_send
                            )
                        except Exception as e:
                            logger.error(
                                f"[spmd_id={self.spmd_id}] Worker {self.rank} failed to send control_arguments "
                                f"via driver: {e}"
                            )

    def _extract_tensors(
        self, data: Any, output_sharding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract named tensors from output data using generic attribute access.
        Supports recursive traversal for nested @data_type objects.
        """
        tensors: Dict[str, Any] = {}

        for name in output_sharding.keys():
            # 'name' can be a flattened path like 'active.user_id'
            parts = name.split(".")
            val = data
            for part in parts:
                if val is None:
                    break
                # Try attribute first (dataclass / @data_type)
                if hasattr(val, part):
                    val = getattr(val, part)
                elif isinstance(val, dict) and part in val:
                    # Fallback to dict access
                    val = val[part]
                else:
                    val = None
                    break

            if val is not None:
                tensors[name] = val

        # Handle legacy or direct return case (entire result is a single tensor)
        if not tensors and output_sharding and not isinstance(data, (dict, list)):
            first_name = next(iter(output_sharding.keys()))
            tensors[first_name] = data

        # Ensure all extracted values are TransportTensors
        for name in list(tensors.keys()):
            val = tensors[name]
            if not isinstance(val, TransportTensor):
                val = TransportTensor.from_tensor(val)
                tensors[name] = val

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] _extract_tensors: "
            f"extracted={list(tensors.keys())}, data_type={type(data).__name__}"
        )
        return tensors

    def _reconstruct_data_type(
        self, cls: Type, flat_data: Dict[str, Any], prefix: str = ""
    ) -> Any:
        """
        Recursively reconstruct a @data_type object from flattened fields.

        Args:
            cls: The @data_type class to reconstruct
            flat_data: Mapping of full paths to TransportTensors/States
            prefix: Current path prefix
        """
        from fluxway.common.decorators import _is_fluxway_data_type

        # Use library-agnostic reconstruction if possible
        if not _is_fluxway_data_type(cls):
            return flat_data.get(prefix)

        # Get fields and their type hints
        try:
            hints = get_type_hints(cls, include_extras=True)
        except Exception:
            # Fallback if hints can't be resolved
            return None

        kwargs = {}
        for name, hint in hints.items():
            if name.startswith("_"):
                continue

            full_name = f"{prefix}.{name}" if prefix else name

            # If hint is another @data_type, recurse
            if _is_fluxway_data_type(hint):
                # Resolve nested type (handle Optional/Union)
                actual_cls = hint
                origin = getattr(hint, "__origin__", None)
                if origin is Union:
                    args = getattr(hint, "__args__", [])
                    for arg in args:
                        if arg is not type(None) and _is_fluxway_data_type(arg):
                            actual_cls = arg
                            break
                kwargs[name] = self._reconstruct_data_type(
                    actual_cls, flat_data, prefix=full_name
                )
            else:
                # Direct field (Tensor or State)
                if full_name in flat_data:
                    kwargs[name] = flat_data[full_name]

        # Instantiate the @data_type object (usually a dataclass)
        try:
            return cls(**kwargs)
        except Exception as e:
            logger.error(f"Failed to reconstruct {cls.__name__} at {prefix}: {e}")
            return None

    # =========================================================================
    # Backward Reverse Routing (using forward shard origins)
    # =========================================================================

    async def _self_receive_data(
        self, target_op: str, name: str, step: int, tensor: TransportTensor
    ) -> None:
        """Self-loop data passing - directly call data receive interface."""
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Self-loop data passing: {name} step={step} "
            f"-> {target_op}"
        )
        # Construct group per payload (data, lengths, indices)
        group = [tensor.flat_values]
        if tensor.is_ragged:
            # Reconstruct lengths from splits
            splits = tensor.nested_splits[0]
            lengths = splits[1:] - splits[:-1]
            group.append(lengths)
        if tensor.spatial_indices is not None:
            group.append(tensor.spatial_indices)
        if tensor.origin_indices is not None:
            group.append(tensor.origin_indices)

        # Scalar handling: use [1] for count if shape is empty
        row_count = int(tensor.shape[0]) if tensor.shape else 1
        await self.receive_shards(
            target_op,
            [name],
            step,
            self.spmd_id,
            self.rank,
            [group],
            [[row_count]],
            is_ragged=[tensor.is_ragged],
            has_spatial_indices=[tensor.spatial_indices is not None],
            has_origin_indices=[tensor.origin_indices is not None],
        )

    async def _self_receive_state(
        self, target_op: str, name: str, step: int, data: Any
    ) -> None:
        """
        Self-loop state passing - directly call data receive interface.

        For same-node edges (e.g., forward -> backward), this method
        maintains the data-driven execution model by calling receive_control
        on self rather than sending via control plane.
        """
        # Directly call receive_control to trigger target op's data readiness check
        await self.receive_control(
            op_name=target_op, step=step, names=[name], values=[data]
        )

    def _extract_states(self, data: Any, state_names: List[str]) -> Dict[str, Any]:
        """
        Extract named state values from output data using generic attribute access.

        Args:
            data: Output data from operator execution
            state_names: List of state variable names to extract

        Returns:
            Dictionary of {state_name: state_value}
        """
        states: Dict[str, Any] = {}

        # 1. Try getattr protocol
        for name in state_names:
            val = getattr(data, name, None)
            if val is not None:
                states[name] = val

        # 2. Fallback to dict access
        if len(states) < len(state_names) and isinstance(data, dict):
            for name in state_names:
                if name not in states and name in data:
                    states[name] = data[name]

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

    def receive_control_argument(self, state_name: str, state_value: Any) -> None:
        """
        Receive control_argument from driver (broadcast from Controller).

        State arguments are set as static args for operator execution.

        Args:
            state_name: State variable name
            state_value: State value
        """
        self.set_static_arg(state_name, state_value)
        logger.debug(
            f"[spmd_id={self.spmd_id}] Worker {self.rank} received control_argument: {state_name}"
        )

    def send_control_argument(
        self,
        target_spmd_id: str,
        target_op: str,
        state_name: str,
        step: int,
        state_value: Any,
    ) -> None:
        """
        Send control_argument to target SPMD.

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
            raise RuntimeError("Only rank 0 worker can send control_argument")

        if self._driver_handle is None:
            logger.error(
                f"[spmd_id={self.spmd_id} rank={self.rank}] No driver handle set, cannot send control_argument"
            )
            return

        try:
            self._driver_handle.routing_control_argument_to_controller.remote(
                target_spmd_id, target_op, state_name, step, state_value
            )
            logger.debug(
                f"[spmd_id={self.spmd_id}] Worker {self.rank} sent control_argument '{state_name}' "
                f"to SPMD {target_spmd_id}, target_op={target_op}, step={step}"
            )
        except Exception as e:
            logger.error(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Failed to send control_argument: {e}"
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
            # logger.log(5,  # HEARTBEAT
            #     f"[spmd_id={self.spmd_id}] Worker {self.rank} sent heartbeat: status={status.name}, "
            #     f"op_steps={heartbeat.op_steps}, mailbox_full={mailbox_status.is_full}"
            # )
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
    # Source Node Control
    # =========================================================================

    def pause_sources(self) -> None:
        """
        Pause all source operators (called by driver when downstream is busy).
        """
        for event in self._source_pause_events.values():
            event.clear()
        logger.debug(f"[spmd_id={self.spmd_id}] Sources paused on worker {self.rank}")

    def resume_sources(self) -> None:
        """
        Resume all source operators (called by driver when backpressure relieved).
        """
        for event in self._source_pause_events.values():
            event.set()
        logger.debug(f"[spmd_id={self.spmd_id}] Sources resumed on worker {self.rank}")
