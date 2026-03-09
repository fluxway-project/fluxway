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
from fluxway.common.types.sharding import ShardingSpec
from fluxway.common.utils.tree import tree_flatten, tree_unflatten
from typing import Any, Dict, List, Optional, Tuple, Type, get_type_hints
from fluxway.common.backend import get_backend
from fluxway.common.types.configs import MailboxStatus
from fluxway.common.types.tensor import SymmetryReceipt
from fluxway.common.utils import ensure_logging_configured
from fluxway.rpc.protocol import DataPlaneCallable
from .mailbox import WorkerMailbox
from .operator_manager import OperatorManager
from .operator_manager.dispatcher import SKIPPED
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
        self.spmd_id = spmd_id or ""
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
        # Initialize to None to strictly wait for registration from driver
        self._expected_predecessor_count: Optional[int] = None
        self._expected_successor_count: Optional[int] = None

        # Cached assembled input tensors: {op_name: {tuuid: TransportTensor}}
        # Updated by _assemble_input, used by _send_output for pass-through metadata
        self._last_assembled_tensors: Dict[
            str, Dict[str, Optional[TransportTensor]]
        ] = {op_name: {} for op_name in (self.config.operators if self.config else {})}

        # Metadata for "Mirror Sharding" (Symmetry Routing)
        # Structure: {step_id: {op_name.tuuid: TransportTensor}}
        self._step_input_cache: Dict[int, Dict[str, TransportTensor]] = {}

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

        if (
            self._expected_predecessor_count is None
            or self._expected_successor_count is None
        ):
            logger.debug(
                f"[{self.spmd_id} Rank {self.rank}] _check_and_auto_start: Waiting for topology counts..."
            )
            return

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
        logger.debug(
            f"[{self.spmd_id} Rank {self.rank}] Identified source operators: {source_ops}"
        )
        for op_name in source_ops:
            logger.info(
                f"[{self.spmd_id} Rank {self.rank}] AUTO-TRIGGERING source operator '{op_name}' for step 0"
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
        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] BLOCK_OP applied for {op_name}"
        )
        self._op_runnable[op_name].clear()

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

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] UNBLOCK_OP applied for {op_name}"
        )

    def is_op_blocked(self, op_name: str) -> bool:
        """Check if an operator is blocked."""
        return not self._op_runnable[op_name].is_set()

    # =========================================================================
    # Eager Mode Support
    # =========================================================================

    async def prepare_eager_op(
        self,
        op_name: str,
        step: int,
        tuuid_sharding: Dict[int, "ShardingSpec"],
        expected_counts: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Prepare for an Eager operator execution by registering its input TUUIDs.
        Called by the Driver BEFORE the task is triggered on all workers.
        """
        if self._op_manager:
            self._op_manager.register_eager_inputs(
                op_name, tuuid_sharding, expected_counts
            )

        # Ensure the step is declared in the mailbox so it can start receiving data
        self._ensure_step_declared(op_name, step)

        logger.debug(
            f"[spmd_id={self.spmd_id} rank={self.rank}] Prepared eager op {op_name} "
            f"for step {step} with {len(tuuid_sharding)} inputs."
        )

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
        tuuids: List[int],
        step: int,
        sender_id: str,
        sender_rank: int,
        payloads: List[Any],
        counts: List[int],
        global_shapes: Optional[List[Optional[Tuple[int, ...]]]] = None,
        meta_tensor_state_list: Optional[List[int]] = None,
        slice_compact_desc_params: Optional[List[Optional[List[int]]]] = None,
    ) -> int:
        """Receive multiple tensor shards and dispatch to local operators by TUUID.

        Args:
            op_name: Operation type
            tuuids: List of TUUIDs
            step: Execution step
            sender_id: ID of the sender
            rank: Sender rank
            sender_world_size: World size of the sender mesh
            payloads: Flat list of raw data arrays for all tensors in the bundle
            counts: Flat list of counts for each element in the payloads
            global_shapes: Optional global shapes per TUUID
            meta_tensor_state_list: List of ints representing the bitmask state of each tensor
            slice_compact_desc_params: Parameters for compact index reconstruction
        """
        num = len(tuuids)
        meta_tensor_state_list = (
            meta_tensor_state_list if meta_tensor_state_list is not None else [0] * num
        )
        global_shapes = global_shapes if global_shapes is not None else [None] * num

        # Group by operator using node-local TUUID dispatch table
        op_to_indices = {}  # {op_name: [idx_in_bundle, ...]}
        for i, u in enumerate(tuuids):
            ops = self._op_manager.get_operators_for_tuuid(u)
            for op_name in ops:
                if op_name not in op_to_indices:
                    op_to_indices[op_name] = []
                op_to_indices[op_name].append(i)

        watermark = 0
        # Pre-parse the flat RPC list into per-TUUID bundles based on meta_state
        # This allows us to correctly pick shards for different operators.
        per_tuuid_payloads = []
        per_tuuid_counts = []
        cursor = 0
        for i, u in enumerate(tuuids):
            meta_state = meta_tensor_state_list[i]
            n_elements = 1
            if meta_state & 1:  # Ragged
                n_elements += 1
            if meta_state & 2:  # Materialized Index
                n_elements += 1

            per_tuuid_payloads.append(payloads[cursor : cursor + n_elements])
            per_tuuid_counts.append(counts[cursor : cursor + n_elements])
            cursor += n_elements

        for op_name, indices in op_to_indices.items():
            self._ensure_step_declared(op_name, step)

            # Extract subsets for this operator
            sub_uuids = [tuuids[i] for i in indices]
            sub_shapes = [global_shapes[i] for i in indices]
            sub_meta = [meta_tensor_state_list[i] for i in indices]

            # Re-flatten the payloads/counts for this specific operator's shards
            sub_payloads = []
            sub_counts = []
            for i in indices:
                sub_payloads.extend(per_tuuid_payloads[i])
                sub_counts.extend(per_tuuid_counts[i])

            # Automatic lookup for sender world size based on topology or eager mesh
            # Assume first TUUID in subset is representative of the sender's world size for this edge
            sender_world_size = self._op_manager.get_sender_world_size(
                op_name, sender_id, sub_uuids[0]
            )
            if sender_world_size is None:
                raise ValueError(
                    f"Could not determine sender world size for op={op_name}, "
                    f"sender_id={sender_id}, tuuid={sub_uuids[0]}"
                )

            watermark = await self._mailbox.deposit_shards(
                op_name,
                sub_uuids,
                step,
                sender_id,
                sender_rank,
                sender_world_size,
                sub_payloads,
                sub_counts,
                sub_shapes,
                meta_tensor_state_list=sub_meta,
                slice_compact_desc_params=[
                    slice_compact_desc_params[i] for i in indices
                ]
                if slice_compact_desc_params
                else None,
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
        expected_tuuids = self._op_manager.get_expected_tuuids(op_name)
        if not expected_tuuids:
            return False

        all_tensor_shards = await self._mailbox.get_all_tensor_shards(op_name, step)

        for tuuid in expected_tuuids:
            if tuuid not in all_tensor_shards:
                return False
            shards = all_tensor_shards[tuuid]
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
        """Propagate empty packets to successor SPMDs for skipped/empty steps."""
        if self._op_manager is None:
            return

        successor_ids = self._op_manager.get_successor_ids(op_name)
        if not successor_ids:
            return

        backend = get_backend()
        empty_tensor = TransportTensor.from_tensor(
            backend.zeros((0,), dtype=backend.int32())
        )

        for spmd_id in successor_ids:
            succ_info = self._op_manager.get_successor_info(op_name, spmd_id)
            if not succ_info:
                continue

            # In empty propagation, we send for ALL UUIDs mapped to this successor
            for target_name, tuuid in succ_info.tuuid_map.items():
                # # TODO(He Jia): Find a better way to avoid send empty packets to all consumers
                # target_spec = succ_info.tuuid_sharding.get(tuuid)
                # # Note: For empty packets, we might not have a source_spec if we skipped execution.
                # # However, the OperatorManager's output_sharding contains it.
                # output_sharding = self._op_manager.get_output_sharding(op_name)
                # # Find the source spec for this TUUID (searching reverse mapping isn't great but works here)
                # # Actually, OperatorManager should have a get_output_sharding_by_uuid.
                # # For now, let's just find it.
                # source_spec = None
                # for uuid_val in self._op_manager.get_output_tuuids(op_name):
                #     if uuid_val == tuuid:
                #         source_spec = output_sharding.get(uuid_val)
                #         break

                # if source_spec and target_spec:
                #     effective_rank = self.rank
                #     dispatcher = self._op_manager.get_dispatcher(op_name)
                #     if getattr(dispatcher, "interleave_unit_size", None) is not None:
                #         effective_rank = self.rank % dispatcher.interleave_unit_size

                #     consumers = source_spec.resolve_consumers(
                #         target_spec, effective_rank
                #     )
                # else:
                #     # If sharding is missing, it must be replicated or non-standard.
                #     # As per user request: explicit error if ambiguous.
                #     if source_spec or target_spec or succ_info.is_interleaved:
                #         raise ValueError(
                #             f"[SPMDWorker] Ambiguous empty propagation for TUUID {tuuid} "
                #             f"to {spmd_id}. Specs: src={bool(source_spec)}, dst={bool(target_spec)}"
                #         )

                # TODO(He Jia): Find a better way to avoid send empty packets to all consumers
                # To prevent deadlocks, empty packets MUST be broadcast to all potential consumers
                # because the receiver's StepBarrier expects a fixed number of shards (mesh size)
                # regardless of whether the static routing overlap exists.
                consumers = range(succ_info.world_size)

                for target_rank in consumers:
                    handler = self._successor_handlers.get(spmd_id, {}).get(target_rank)
                    if handler:
                        # Send single-tensor payload for this TUUID
                        await handler.receive_shards(
                            tuuids=[tuuid],
                            step=step,
                            sender_id=self.spmd_id,
                            sender_rank=self.rank,
                            payloads=[empty_tensor.flat_values],
                            counts=[0],
                            global_shapes=[None],
                            meta_tensor_state_list=[empty_tensor.meta_state],
                            slice_compact_desc_params=[None],
                        )

    async def _execute_operator(self, op_name: str, step: int) -> Any:
        """Execute an operator: assemble data, execute, send outputs."""
        try:
            # --- Dispatcher Check: Should this rank execute this step? ---
            dispatcher = self._op_manager.get_dispatcher(op_name)
            if not dispatcher.should_execute(self.rank, self._world_size, step):
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Dispatcher logic skipped execution for op={op_name}, step={step}. Silence inactive unit."
                )
                await self._mailbox.clear_step(op_name, step)
                self._barriers_declared.pop((op_name, step), None)
                return SKIPPED

            # --- Early check: all received shards are empty packets ---
            all_empty = await self._check_all_empty(op_name, step)
            if all_empty:
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] All received shards are empty for "
                    f"op={op_name}, step={step}. Propagating per sparse routing rules."
                )
                await self._propagate_empty_packets(op_name, step)
                await self._mailbox.clear_step(op_name, step)
                self._barriers_declared.pop((op_name, step), None)
                return []

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
                    self._op_manager.execute, op_name, step, **input_data
                )
            else:
                result = await asyncio.to_thread(
                    self._op_manager.execute, op_name, step, input_data
                )
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Execution finished: op={op_name}, step={step}"
            )

            if result is None:
                logger.debug(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Operator {op_name} returned None. "
                    f"Continuing to advance step."
                )

            # Send output
            logger.debug(
                f"[spmd_id={self.spmd_id} rank={self.rank}] About to send output: op={op_name}, "
                f"at step={step}, result_type={type(result).__name__}"
            )
            await self._send_output(op_name, result, exec_step=step)

            # Clear mailbox
            await self._mailbox.clear_step(op_name, step)
            self._barriers_declared.pop((op_name, step), None)

            return result

        except Exception as e:
            import traceback

            logger.error(
                f"[spmd_id={self.spmd_id} rank={self.rank}] Operator {op_name} execution failed: {e}, traceback: {traceback.format_exc()}"
            )
            raise

    async def _assemble_input(self, op_name: str, step: int) -> Any:
        static_args = self._op_manager.get_static_args()
        expected_state_names = self._op_manager.get_expected_state_names(op_name)
        all_states = await self._mailbox.get_all_states(op_name, step)
        state_args = {
            name: all_states[name]
            for name in expected_state_names
            if name in all_states
        }

        meta = self._op_manager.get_operator_metadata(op_name)
        input_types = self._op_manager.get_operator_input_types(op_name)
        pre_topo = self._op_manager.get_predecessor_topology(op_name)

        all_tuuids = []
        p_to_tuuids = {}
        for p_name in input_types:
            if p_name in static_args:
                continue

            p_paths = meta.input_param_map.get(p_name, [p_name])
            p_tuuids = []
            for path in p_paths:
                tuuid = None
                for info in pre_topo.values():
                    if path in info.tuuid_map:
                        tuuid = info.tuuid_map[path]
                        break
                p_tuuids.append(tuuid)

            p_to_tuuids[p_name] = p_tuuids
            all_tuuids.extend([u for u in p_tuuids if u is not None])

        unique_tuuids = list(set(all_tuuids))

        shard_tasks = [
            self._mailbox.get_tensor_shards(op_name, u, step) for u in unique_tuuids
        ]
        all_shards_list = await asyncio.gather(*shard_tasks)
        tuuid_to_shards = dict(zip(unique_tuuids, all_shards_list, strict=True))  # type: ignore

        assemble_tasks = [
            self._assemble_tensor(op_name, u, step, tuuid_to_shards[u])
            for u in unique_tuuids
        ]
        assembled_list = await asyncio.gather(*assemble_tasks)
        tuuid_to_assembled = dict(zip(unique_tuuids, assembled_list, strict=True))  # type: ignore

        input_kwargs = {}
        for p_name, p_tuuids in p_to_tuuids.items():
            if not p_tuuids or p_tuuids[0] is None:
                continue

            p_leaves = [tuuid_to_assembled[u] for u in p_tuuids]
            tdef = meta.input_treedefs.get(p_name)
            if tdef:
                input_kwargs[p_name] = tree_unflatten(tdef, p_leaves)
            else:
                input_kwargs[p_name] = p_leaves[0]

        final_args = {}
        final_args.update(static_args)
        final_args.update(state_args)
        final_args.update(input_kwargs)
        return final_args

    def _get_values(self, obj: Any) -> Any:
        if isinstance(obj, TransportTensor):
            return obj.flat_values
        return obj

    def _get_routing_indices(self, obj: Any) -> Optional[Any]:
        if isinstance(obj, TransportTensor):
            return obj.routing_indices_materialized
        return None

    def _flatten_shards(
        self, shards: Dict[Tuple[str, int], List[TransportTensor]], index: int = 0
    ) -> List[TransportTensor]:
        """
        Extract the N-th tensor from each multi-tensor packet.
        """
        tensors = []
        for sender_res in shards.values():
            if index < len(sender_res):
                tensors.append(sender_res[index])
        return tensors

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
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards: Dict[Any, List[TransportTensor]],
    ) -> TransportTensor:
        """Assemble tensor shards into a complete TransportTensor."""
        # Note: Input spec is now retrieved by TUUID
        input_sharding = self._op_manager.get_input_sharding(op_name)
        input_spec = input_sharding.get(tuuid)
        predecessor_topology = self._op_manager.get_predecessor_topology(op_name)

        if input_spec is None or input_spec.is_content_addressable:
            # Fallback or dynamic: merge all shards
            return self._assemble_to_dynamic_tensor(
                op_name, tuuid, step, shards, predecessor_topology
            )

        # Check for dynamic predecessors
        has_dynamic_predecessor = any(
            (spec := p.tuuid_sharding.get(tuuid)) is not None
            and spec.is_content_addressable
            for p in predecessor_topology.values()
        )

        if has_dynamic_predecessor:
            return self._assemble_dynamic_to_static_tensor(
                op_name, tuuid, step, shards, input_spec, predecessor_topology
            )
        else:
            return await self._assemble_static_to_static_tensor(
                op_name, tuuid, step, shards, input_spec, predecessor_topology
            )

    def _filter_shards_by_routing(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards_dict: Dict[Any, List[TransportTensor]],
        predecessor_topology: Dict[str, Any],
    ) -> List[TransportTensor]:
        """
        Filters shards from mailbox based on analytical or fallback routing.
        """
        filtered_shards = []
        for pre_spmd_id, pre_info in predecessor_topology.items():
            # Only query predecessors that actually provide this TUUID.
            if tuuid not in pre_info.tuuid_sharding:
                continue

            # Unified resolution (handles interleave, etc.)
            producer_ranks = self._op_manager.resolve_tensor_producers(
                op_name, tuuid, pre_spmd_id, self.rank, step=step
            )

            for rank in producer_ranks:
                key = (pre_spmd_id, rank)
                if key in shards_dict:
                    # Flattening multi-tensor packets if any
                    filtered_shards.extend(
                        self._flatten_shards({key: shards_dict[key]}, index=0)
                    )
        return filtered_shards

    def _normalize_and_cat_shards(self, tensor_list: List[Any], dim: int = 0) -> Any:
        """
        Robustly concatenate high-dimensional shards, healing rank/dtype mismatches.
        """
        backend = get_backend()
        count = len(tensor_list)
        if count == 0:
            return backend.zeros((0,), dtype=backend.float32())

        # Fast Path: Single tensor is already the result (Zero-copy)
        if count == 1:
            return tensor_list[0]

        # Find the archetype (first non-empty tensor with proper rank)
        prototype = None
        for t in tensor_list:
            if backend.size(t) > 0:
                prototype = t
                break

        # All empty fallback: use the most "structured" one
        if prototype is None:
            prototype = max(tensor_list, key=lambda t: len(backend.shape(t)))

        # Check for mismatch against prototype to decide if we need the slow-path
        needs_normalization = False
        proto_shape = backend.shape(prototype)
        proto_dtype = backend.dtype(prototype)
        for t in tensor_list:
            if backend.shape(t) != proto_shape or backend.dtype(t) != proto_dtype:
                needs_normalization = True
                break

        # Homogeneous Fast Path: If no mismatches found, skip alignment loop
        if not needs_normalization:
            return backend.cat(tensor_list, dim=dim)

        proto_shape = backend.shape(prototype)
        proto_dtype = backend.dtype(prototype)
        proto_ndims = len(proto_shape)

        # Slow Path: Align and Repair empty tensors shape
        normalized = []
        for t in tensor_list:
            t_shape = backend.shape(t)
            if len(t_shape) != proto_ndims:
                # Common case: (0,) being concatenated with (N, D)
                new_shape = list(proto_shape)
                new_shape[dim] = 0  # Ensure it's empty along the cat dimension
                normalized.append(backend.zeros(tuple(new_shape), dtype=proto_dtype))
            elif backend.dtype(t) != proto_dtype:
                normalized.append(backend.cast(t, proto_dtype))
            else:
                normalized.append(t)

        return backend.cat(normalized, dim=dim)

    # =========================================================================
    # Shard Assembly Helpers (shared by all _assemble_* functions)
    # =========================================================================

    def _merge_index_descriptors(
        self, shard_list: List["TransportTensor"], property: str = "origin_indices"
    ) -> Optional["IndexDescriptor"]:
        """Merge IndexDescriptors from shards only when ALL shards carry one."""
        descriptors: List[IndexDescriptor] = [
            getattr(s, property) for s in shard_list if getattr(s, property) is not None
        ]
        if len(descriptors) == len(shard_list) and shard_list:
            return IndexDescriptor.concatenate(descriptors)
        return None

    def _merge_materialized_index_descriptors(
        self,
        shards: List["TransportTensor"],
        property: str = "origin_indices_materialized",
    ) -> Optional[Any]:
        """Merge materialized indices from shards, falling back to spatial / arange."""
        backend = get_backend()
        if not any(
            hasattr(s, property) and getattr(s, property) is not None for s in shards
        ):
            return None
        index_list = []
        for s in shards:
            idx = getattr(s, property, None)
            if idx is None and len(s) > 0:
                raise ValueError(
                    f"Shard {s} is missing materialized index descriptor {property}"
                )
            if idx is None:
                # Fallback to routing if origin is missing, then to arange
                idx = self._get_routing_indices(s)
            if idx is None:
                num = backend.shape(self._get_values(s))[0]
                idx = backend.arange(0, num, dtype=backend.int32())
            index_list.append(idx)
        return self._normalize_and_cat_shards(index_list, dim=0)

    def _maybe_merge_receipts(
        self, shards: List["TransportTensor"], has_mirror: bool
    ) -> Optional["SymmetryReceipt"]:
        """Merge SymmetryReceipts only when the edge is declared as mirror (symmetric_to).

        For non-mirror edges the mailbox never constructs a receipt, so this is
        a pure no-op (zero overhead).
        """
        if not has_mirror:
            return None
        receipts = [s.receipt for s in shards]
        if all(r is not None for r in receipts):
            return SymmetryReceipt.concatenate(receipts)  # type: ignore[arg-type]
        return None

    def _heal_empty_shape(
        self, input_spec: "ShardingSpec", reference_values: Optional[Any] = None
    ) -> "TransportTensor":
        """Return a correctly-shaped empty TransportTensor when no data arrived.

        Falls back to (0,) only when dimension information is completely unavailable.
        """
        backend = get_backend()
        dtype = backend.int32()
        # Attempt to infer feature dimension from global_shape in the spec
        feature_dim: Optional[int] = None
        if hasattr(input_spec, "global_shape") and input_spec.global_shape:
            gs = list(input_spec.global_shape)
            if len(gs) >= 2:
                feature_dim = gs[1]  # assume axis-0 is the batch/routing dim
        # Fallback: peek at a non-empty reference tensor
        if feature_dim is None and reference_values is not None:
            ref_shape = backend.shape(reference_values)
            if len(ref_shape) >= 2:
                feature_dim = int(ref_shape[1])
        if feature_dim is not None:
            return TransportTensor.from_tensor(
                backend.zeros((0, feature_dim), dtype=dtype)
            )
        return TransportTensor.from_tensor(backend.zeros((0,), dtype=dtype))

    # =========================================================================
    # Tensor Assembly
    # =========================================================================

    async def _assemble_static_to_static_tensor(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards: Dict[Any, List["TransportTensor"]],
        input_spec: "ShardingSpec",
        predecessor_topology: Dict[str, Any],
    ) -> "TransportTensor":
        """Assemble static-to-static tensor using exact physical shards."""
        shard_list = self._filter_shards_by_routing(
            op_name, tuuid, step, shards, predecessor_topology
        )

        if not shard_list:
            raise ValueError(
                f"No shards for TUUID {tuuid} at step {step} (SPMD {self.spmd_id} Rank {self.rank})"
            )

        if len(shard_list) == 1:
            return shard_list[0]

        flat_values = [self._get_values(s) for s in shard_list]
        merged_values = self._normalize_and_cat_shards(flat_values, dim=0)
        merged_routing_indices = self._merge_index_descriptors(
            shard_list, "routing_indices"
        )
        merged_origin_indices = self._merge_index_descriptors(
            shard_list, "origin_indices"
        )

        first_shard = shard_list[0]
        global_shape = (
            first_shard.global_shape if hasattr(first_shard, "global_shape") else None
        )

        # If the assembled result is empty but global_shape says data exists, this is a
        # configuration error that slipped past the Communication-layer check (e.g. the
        # global_shape was attached after distribution).
        backend = get_backend()
        assembled_rows = backend.shape(merged_values)[0]
        if assembled_rows == 0 and global_shape is not None and global_shape[0] > 0:
            raise ValueError(
                f"\n[Fluxway Static Sharding Error] Static assembly for '{tuuid}' "
                f"at {self.spmd_id} Rank {self.rank} produced 0 rows, "
                f"but global_shape={global_shape} indicates data exists.\n"
                f"  This means the upstream data size is smaller than this SPMD's mesh size.\n"
                f"  Fix: Increase the data size to cover all {self._world_size} workers, "
                f"or reduce the target mesh."
            )

        # Mirror receipt: only when this edge is declared symmetric
        has_mirror = any(
            info.symmetric_to is not None for info in predecessor_topology.values()
        )
        merged_receipt = self._maybe_merge_receipts(shard_list, has_mirror)

        return TransportTensor(
            merged_values,
            routing_indices=merged_routing_indices,
            origin_indices=merged_origin_indices,
            global_shape=global_shape,
            receipt=merged_receipt,
        )

    def _assemble_dynamic_to_static_tensor(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards: Dict[Any, List["TransportTensor"]],
        input_spec: "ShardingSpec",
        predecessor_topology: Dict[str, Any],
    ) -> "TransportTensor":
        """Assemble dynamic-to-static (Sparse-to-Dense) tensor using origin_indices."""
        backend = get_backend()

        filtered_shards = self._filter_shards_by_routing(
            op_name, tuuid, step, shards, predecessor_topology
        )

        valid_shards = [
            s for s in filtered_shards if backend.shape(self._get_values(s))[0] > 0
        ]
        if not valid_shards:
            return self._heal_empty_shape(input_spec)

        # Mirror receipt: only when this edge is declared symmetric
        has_mirror = any(
            info.symmetric_to is not None for info in predecessor_topology.values()
        )
        all_receipt = self._maybe_merge_receipts(valid_shards, has_mirror)

        routing_dim = getattr(input_spec, "routing_dim", 0)
        if isinstance(routing_dim, (list, tuple)):
            routing_dim = routing_dim[0]

        all_values = backend.cat(
            [self._get_values(s) for s in valid_shards], dim=routing_dim
        )

        # Materialize indices (origin_indices preferred for logical ordering)
        all_indices_list = []
        for s in valid_shards:
            idx = s.origin_indices
            if idx is None:
                idx = self._get_routing_indices(s)
            if idx is None:
                num = backend.shape(self._get_values(s))[0]
                idx = backend.arange(0, num, dtype=backend.int32())
            all_indices_list.append(idx)
        all_indices = backend.cat(all_indices_list, dim=0)

        max_idx = int(backend.item(backend.max(all_indices)))
        inferred_dim_size = max_idx + 1

        val_shape = list(backend.shape(all_values))
        global_shape_list = val_shape
        global_shape_list[routing_dim] = inferred_dim_size
        global_shape = tuple(global_shape_list)

        local_shape = input_spec.get_shard_shape(self.rank, data_shape=global_shape)
        slice_start, slice_end = input_spec.get_shard_bounds(
            self.rank, routing_dim, data_shape=global_shape
        )

        mask = (all_indices >= slice_start) & (all_indices < slice_end)
        local_indices = all_indices[mask] - slice_start

        if routing_dim == 0:
            local_values = all_values[mask]
        else:
            indices_to_take = backend.nonzero(mask).flatten()
            idx_tuple = [slice(None)] * len(val_shape)
            idx_tuple[routing_dim] = indices_to_take
            local_values = all_values[tuple(idx_tuple)]

        assembled_data = backend.zeros(local_shape, dtype=all_values.dtype)
        indices_tuple = [slice(None)] * len(local_shape)
        indices_tuple[routing_dim] = local_indices
        assembled_data = backend.index_update(
            assembled_data, tuple(indices_tuple), local_values
        )

        # Assemble mirror receipt into the static layout if present
        assembled_receipt = None
        if all_receipt is not None:
            final_ranks = backend.full(
                (local_shape[routing_dim],), -1, dtype=backend.int32()
            )
            local_ranks = all_receipt.sender_rank_tags[mask]
            final_ranks = backend.index_update(final_ranks, local_indices, local_ranks)
            assembled_receipt = SymmetryReceipt(all_receipt.world_size, final_ranks)

        return TransportTensor(
            assembled_data,
            routing_indices=None,
            global_shape=global_shape,
            receipt=assembled_receipt,
        )

    def _assemble_to_dynamic_tensor(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards: Dict[Any, List["TransportTensor"]],
        predecessor_topology: Dict[str, Any],
    ) -> "TransportTensor":
        """
        Assemble to-dynamic (Sparse/Dense -> Sparse) tensor by merging shards.
        Handles both static_to_dynamic and dynamic_to_dynamic scenarios.
        """
        backend = get_backend()

        filtered_shards = self._filter_shards_by_routing(
            op_name, tuuid, step, shards, predecessor_topology
        )

        if not filtered_shards:
            return TransportTensor.from_tensor(
                backend.zeros((0,), dtype=backend.int32())
            )

        all_values_list = [self._get_values(s) for s in filtered_shards]
        merged_values = self._normalize_and_cat_shards(all_values_list, dim=0)

        # Spatial Indices (global keys)
        merged_routing_indices = None
        if any(self._get_routing_indices(s) is not None for s in filtered_shards):
            indices_list = []
            for s in filtered_shards:
                idx = self._get_routing_indices(s)
                if idx is None:
                    num = backend.shape(self._get_values(s))[0]
                    idx = backend.arange(0, num, dtype=backend.int32())
                indices_list.append(idx)
            merged_routing_indices = self._normalize_and_cat_shards(indices_list, dim=0)

        # Origin Indices (logical offsets for deterministic sorting)
        merged_origin_indices = self._merge_materialized_index_descriptors(
            filtered_shards
        )

        # Mirror receipt: only when this edge is declared symmetric
        has_mirror = any(
            info.symmetric_to is not None for info in predecessor_topology.values()
        )
        merged_receipt = self._maybe_merge_receipts(filtered_shards, has_mirror)

        # DETERMINISTIC SORTING:
        if merged_origin_indices is not None:
            sort_idx = backend.argsort(merged_origin_indices)
            merged_values = backend.take(merged_values, sort_idx)
            if merged_routing_indices is not None:
                merged_routing_indices = backend.take(merged_routing_indices, sort_idx)
            if merged_receipt is not None:
                merged_receipt = merged_receipt[sort_idx]
            merged_origin_indices = backend.take(merged_origin_indices, sort_idx)

        return TransportTensor(
            merged_values,
            routing_indices=merged_routing_indices,
            global_shape=backend.shape(merged_values),
            receipt=merged_receipt,
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
            elif result is not None or result is SKIPPED:
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

        # Cleanup Mirror Sharding Cache for very old steps
        old_steps = [s for s in self._step_input_cache.keys() if s < step - 5]
        for s in old_steps:
            del self._step_input_cache[s]

    # =========================================================================
    # Data Sending
    # =========================================================================

    async def _send_output(self, op_name: str, result: Any, *, exec_step: int) -> None:
        """Send output to successor SPMDs using TransportTensor."""
        if result is None or result is SKIPPED:
            # If the result is None (e.g., end of stream or no data Produced),
            # we must propagate empty packets to all successors to prevent deadlocks.
            await self._propagate_empty_packets(op_name, exec_step)
            return

        output_uuids = self._op_manager.get_output_tuuids(op_name)
        # Flatten the result using guided tree_flatten
        meta = self._op_manager.get_operator_metadata(op_name)
        flat_results, treedef = tree_flatten(
            result, treedef_template=meta.output_treedef if meta else None
        )

        successor_ids = sorted(set(self._op_manager.get_successor_ids(op_name)))

        # Extract state arguments
        state_data = {}
        expected_state_names = self._op_manager.get_output_state_names(op_name)
        if expected_state_names:
            # Use tree_unflatten to reconstruct the original structure for state extraction
            # This is a temporary measure until state is properly integrated into the graph
            reconstructed_result = tree_unflatten(treedef, flat_results)
            for name in expected_state_names:
                val = getattr(reconstructed_result, name, None)
                if val is not None:
                    state_data[name] = val

        for spmd_id in successor_ids:
            succ_info = self._op_manager.get_successor_info(op_name, spmd_id)
            if succ_info is None:
                continue

            output_step = exec_step
            batches = defaultdict(list)

            # Map flattened results to UUIDs for this successor using the canonical ordered list
            for i, val in enumerate(flat_results):
                if val is None or i >= len(output_uuids):
                    continue

                u = output_uuids[i]

                # Check if this successor's tuuid_map (or sharding) contains this u
                if u not in succ_info.tuuid_sharding:
                    continue

                # Convert to TransportTensor if not already
                if not isinstance(val, TransportTensor):
                    val = TransportTensor.from_tensor(val)

                # Resolve ShardingSpecs
                source_spec = self._op_manager.get_output_sharding(op_name).get(u)
                target_spec = succ_info.tuuid_sharding.get(u)

                if source_spec is None or target_spec is None:
                    continue

                # INTERLEAVED SOURCE ABSTRACTION:
                # For interleaved units, override the source mesh to represent the active unit.
                dispatcher = self._op_manager.get_dispatcher(op_name)
                unit_size = getattr(dispatcher, "interleave_unit_size", None)
                if isinstance(unit_size, int):
                    unit_idx = self.rank // unit_size
                    unit_ranks = list(
                        range(unit_idx * unit_size, (unit_idx + 1) * unit_size)
                    )
                    if source_spec.mesh is not None:
                        base_mesh = source_spec.mesh
                        other_dims_size = 1
                        for s in base_mesh.shape[1:]:
                            other_dims_size *= s
                        if unit_size % other_dims_size == 0:
                            first_axis_size = unit_size // other_dims_size
                            new_shape = (first_axis_size,) + base_mesh.shape[1:]
                            from fluxway.common.mesh import DeviceMesh

                            sub_mesh = DeviceMesh(
                                device_shape=new_shape,
                                axis_names=base_mesh.axis_names,
                                device_ids=unit_ranks,
                            )
                            source_spec = source_spec.with_mesh(sub_mesh)

                # Automate Mirror Injection:
                from fluxway.common.types.sharding import MirrorPartition

                if isinstance(source_spec, ShardingSpec) and isinstance(
                    source_spec.strategy, MirrorPartition
                ):
                    source_node_arg = source_spec.strategy.source
                    if exec_step in self._step_input_cache:
                        cached_input = self._step_input_cache[exec_step].get(
                            source_node_arg
                        )
                        if cached_input and cached_input.receipt:
                            val = TransportTensor(
                                val.flat_values,
                                nested_splits=val.nested_splits,
                                routing_indices=val.routing_indices,
                                global_shape=val.global_shape,
                                origin_indices=val.origin_indices,
                                receipt=cached_input.receipt,
                            )

                # Group by "Routing Consistency"
                routing_id = (
                    id(val.routing_indices) if val.routing_indices is not None else 0
                )
                batches[(source_spec, target_spec, routing_id)].append((u, val))

            handlers = self._successor_handlers.get(spmd_id, {})
            if not handlers:
                continue

            effective_rank = self.rank
            if succ_info.is_interleaved:
                effective_rank = self.rank % succ_info.interleave_unit_size

            distributor = AsyncDataDistributor(effective_rank)

            # Build TUUID -> Name mapping for debugging and reporting (cached in OpManager)
            tuuid_to_name = self._op_manager.get_output_tuuid_to_name(op_name)

            for (source_spec, target_spec, _), batch in batches.items():
                batch_target_uuids = [t[0] for t in batch]
                batch_transport_tensors = [t[1] for t in batch]

                if logger.level <= logging.DEBUG:
                    for u, val in batch:
                        name = (
                            tuuid_to_name.get(u, "unknown")
                            if tuuid_to_name
                            else "unknown"
                        )
                        logger.debug(
                            f"[Distributor Plan] [SPMD {self.spmd_id}] [Rank {self.rank}] Name={name} TUUID={u} global_shape={val.global_shape} tensor={val.flat_values} origin_indices={val.origin_indices} routing_indices={val.routing_indices}"
                        )

                # Accrue to distributor's internal plan
                distributor.plan_and_shard(
                    batch_transport_tensors,
                    source_spec,
                    target_spec,
                    tuuid=batch_target_uuids,
                    tuuid_to_name=tuuid_to_name,
                )

            async def _send_fn(
                target_rank: int,
                tuuids: list,
                tensors: list,
                counts: list,
                global_shapes_list: Optional[list] = None,
                hash_meta_tensor_state_list: Optional[list] = None,
                slice_compact_desc_params: Optional[list] = None,
                *,
                _handlers=handlers,
                _step=output_step,
            ) -> bool:
                handler = _handlers.get(target_rank)
                if handler is None:
                    return False

                await self._send_semaphore.acquire()

                ref_task = handler.receive_shards(
                    tuuids=tuuids,
                    step=_step,
                    sender_id=self.spmd_id,
                    sender_rank=self.rank,
                    payloads=tensors,
                    counts=counts,
                    global_shapes=global_shapes_list,
                    meta_tensor_state_list=hash_meta_tensor_state_list,
                    slice_compact_desc_params=slice_compact_desc_params,
                )

                async def _wait_and_release():
                    try:
                        await ref_task
                    finally:
                        self._send_semaphore.release()

                asyncio.create_task(_wait_and_release())
                return True

            await distributor.distribute(send_fn=_send_fn)

            # --- Send control_argument via control plane ---
            if state_data:
                # Check condition
                if succ_info.condition_fn is not None:
                    if not succ_info.condition_fn(result):
                        continue

                # Collect all state names and values for this target
                names_to_send = []
                values_to_send = []
                # Leader of the active group/SPMD sends to avoid duplicates
                is_leader = self.rank % succ_info.interleave_unit_size == 0

                for state_name, state_value in state_data.items():
                    # Control states are mapped by name (str), not TUUID (int).
                    # TODO(He Jia): Currently we assume names match across the edge unless a state-specific map is added.
                    target_state_name = state_name
                    if is_leader:
                        names_to_send.append(target_state_name)
                        values_to_send.append(state_value)

                if names_to_send:
                    logger.debug(
                        f"[spmd_id={self.spmd_id}] Worker {self.rank} sending control_arguments: names={names_to_send}, "
                        f"output_step={output_step} to {spmd_id}"
                    )
                    if self._driver_handle is not None:
                        try:
                            self._driver_handle.routing_control_argument_to_controller.remote(
                                spmd_id,
                                op_name,
                                names_to_send,
                                output_step,
                                values_to_send,
                            )
                        except Exception as e:
                            logger.error(
                                f"[spmd_id={self.spmd_id}] Worker {self.rank} failed to send control_arguments "
                                f"via driver: {e}"
                            )

    def _reconstruct_data_type_from_dict(self, cls: Type, data: Dict[str, Any]) -> Any:
        """Reconstruct a @data_type or dataclass from a nested dictionary of fields."""
        from fluxway.common.decorators import _is_fluxway_data_type

        if not _is_fluxway_data_type(cls):
            # If not a data_type, it might be a simple type or a raw dict
            return data

        # Get type hints for constructor
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}

        kwargs = {}
        for name, hint in hints.items():
            if name in data:
                val = data[name]
                if isinstance(val, dict) and _is_fluxway_data_type(hint):
                    kwargs[name] = self._reconstruct_data_type_from_dict(hint, val)
                else:
                    kwargs[name] = val

        return cls(**kwargs)

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

        # Try getattr protocol
        for name in state_names:
            val = getattr(data, name, None)
            if val is not None:
                states[name] = val

        # Fallback to dict access
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
        target_op_name: str,
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
            target_op_name: Target operator type on the receiving side
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
                target_spmd_id, target_op_name, state_name, step, state_value
            )
            logger.debug(
                f"[spmd_id={self.spmd_id}] Worker {self.rank} sent control_argument '{state_name}' "
                f"to SPMD {target_spmd_id}, target_op={target_op_name}, step={step}"
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
