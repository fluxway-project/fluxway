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

import ray

from ...common.types import SPMDConfig, WorkerHeartbeat, WorkerStatus, WorkerTelemetry
from ...common.types.data_types import OpType, TensorMeta, NodeType, ReaderConfig, MailboxStatus
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
        self._execution_queue: asyncio.PriorityQueue[Tuple[int, int, OpType]] = asyncio.PriorityQueue()
        
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
        
        # Reader-specific state
        self._is_reading_paused: bool = False  # Backpressure pause flag
        self._reader_pause_event = asyncio.Event()
        self._reader_pause_event.set()  # Initially not paused
        self._reader_completed: bool = False  # Flag to indicate reading completion
        self._batches_sent: int = 0  # Counter for batches sent
        
        node_type_str = "Reader" if node_type == NodeType.READER else "SPMD"
        logger.info(f"{node_type_str}Worker initialized: spmd_id={spmd_id}, rank={rank}")
        
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
    
    # =========================================================================
    # Handler Registration (handlers only, topology from config)
    # =========================================================================
    
    def register_predecessor_handlers(
        self,
        spmd_id: str,
        handlers: Dict[int, Any],
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
        logger.debug(f"Registered predecessor handlers: {spmd_id}, ranks={list(handlers.keys())}")
        
        # Check if all handlers are registered for auto-start
        self._check_and_auto_start()
    
    def register_successor_handlers(
        self,
        spmd_id: str,
        handlers: Dict[int, Any],
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
        logger.debug(f"Registered successor handlers: {spmd_id}, ranks={list(handlers.keys())}")
        
        # Check if all handlers are registered for auto-start
        self._check_and_auto_start()
    
    def set_expected_topology(
        self,
        predecessor_count: int,
        successor_count: int,
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
            f"Worker {self.rank} expects {predecessor_count} predecessors, "
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
        pred_registered = len(self._predecessor_handlers) >= self._expected_predecessor_count
        succ_registered = len(self._successor_handlers) >= self._expected_successor_count
        
        if pred_registered and succ_registered:
            logger.info(
                f"Worker {self.rank}: All handlers registered "
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
        logger.info(f"Worker {self.rank} auto-started")
    
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
        logger.info(f"Worker {self.rank}: {op_type.name} blocked")
    
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
        
        logger.info(f"Worker {self.rank}: {op_type.name} unblocked")
    
    def is_op_blocked(self, op_type: OpType) -> bool:
        """Check if an operator is blocked."""
        return not self._op_runnable[op_type].is_set()
    
    # =========================================================================
    # Data Reception
    # =========================================================================
    
    async def receive_state_data(
        self, 
        op_type: OpType, 
        name: str, 
        step: int, 
        data: Any
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
        is_eager = self._op_manager.is_eager_trigger(op_type) if self._op_manager else False
        
        if is_eager:
            # Eager mode: use _eager_input_steps for input labeling, ignore incoming step
            # Each incoming data gets a unique input_step, then enqueue immediately
            input_step = self._eager_input_steps[op_type]
            self._eager_input_steps[op_type] += 1  # Increment for next input
            
            await self._mailbox.put_state(op_type, name, input_step, data)
            logger.debug(
                f"Eager receive state: op={op_type.name}, name={name}, "
                f"input_step={input_step}"
            )
            await self._enqueue(op_type, input_step)
            return
        
        # Normal mode: step-ordered execution
        exec_step = self._current_steps[op_type]
        
        # Drop stale data (data for steps already executed)
        if step < exec_step:
            logger.warning(
                f"Dropping stale state data: op={op_type.name}, name={name}, "
                f"step={step} < exec_step={exec_step}"
            )
            return
        
        # Cache in mailbox
        await self._mailbox.put_state(op_type, name, step, data)
        
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
        is_eager = self._op_manager.is_eager_trigger(op_type) if self._op_manager else False
        
        if is_eager:
            # Eager mode: use _eager_input_steps for input labeling, ignore incoming step
            # Each incoming data gets a unique input_step, then enqueue immediately
            input_step = self._eager_input_steps[op_type]
            self._eager_input_steps[op_type] += 1  # Increment for next input
            
            await self._mailbox.put_tensor(op_type, name, input_step, sender_rank, data, meta)
            logger.debug(
                f"Eager receive tensor: op={op_type.name}, name={name}, "
                f"sender_rank={sender_rank}, input_step={input_step}"
            )
            await self._enqueue(op_type, input_step)
            return
        
        # Normal mode: step-ordered execution
        exec_step = self._current_steps[op_type]
        
        # Drop stale data (data for steps already executed)
        if step < exec_step:
            logger.warning(
                f"Dropping stale tensor data: op={op_type.name}, name={name}, "
                f"step={step} < exec_step={exec_step}, rank={sender_rank}"
            )
            return
        
        # Cache in mailbox with metadata
        await self._mailbox.put_tensor(op_type, name, step, sender_rank, data, meta)
        
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
                    f"Waiting for state data: op={op_type.name}, step={step}, "
                    f"expected={expected_states}"
                )
                return
        
        # Check tensor shards
        if expected_tensors:
            if not await self._mailbox.has_all_tensors(op_type, step, expected_tensors):
                logger.debug(
                    f"Waiting for tensor shards: op={op_type.name}, step={step}, "
                    f"expected={expected_tensors}"
                )
                return
        
        # All conditions met, enqueue
        logger.info(f"Enqueuing task: op={op_type.name}, step={step}")
        await self._enqueue(op_type, step)
    
    async def _enqueue(self, op_type: OpType, step: int) -> None:
        """Enqueue a task for execution."""
        await self._execution_queue.put((op_type.value, step, op_type))
        logger.debug(f"Task enqueued: op={op_type.name}, step={step}")
    
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
        logger.debug(f"Worker {self.rank} run loop task started (waiting for start signal)")
    
    async def _run(self) -> None:
        """
        Internal main execution loop (async).
        
        Blocks on PriorityQueue until tasks are available.
        Respects op-level blocking.
        
        For Reader nodes, runs a different loop that actively reads data.
        """
        # Wait for start signal from driver
        logger.debug(f"Worker {self.rank} waiting for start signal...")
        await self._start_event.wait()
        
        logger.info(f"Worker {self.rank} execution loop started")
        
        # Reader mode: run reader loop
        if self.node_type == NodeType.READER:
            await self._reader_loop()
            return
        
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
                    logger.debug(f"Task held (op blocked): op={op_type.name}, step={step}")
                    continue
                
                # Check if this op is in eager_trigger mode
                is_eager = self._op_manager.is_eager_trigger(op_type) if self._op_manager else False
                
                if not is_eager:
                    # Normal mode: verify step matches current expected execution step
                    exec_step = self._current_steps[op_type]
                    if step < exec_step:
                        # Truly stale task (already executed), drop it
                        logger.warning(
                            f"Stale task dropped: op={op_type.name}, "
                            f"task_step={step}, exec_step={exec_step}"
                        )
                        continue
                    elif step > exec_step:
                        # Future task (arrived early), hold for later execution
                        async with self._pending_lock:
                            self._pending_future_steps[op_type].append(step)
                        logger.debug(
                            f"Future task held: op={op_type.name}, "
                            f"task_step={step}, exec_step={exec_step}"
                        )
                        continue
                
                # Execute (for eager mode, execute any task that comes)
                await self._execute_operator(op_type, step)
                
                # Advance execution step counter
                await self._advance_step(op_type)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {self.rank} run loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
    
    async def _reader_loop(self) -> None:
        """
        Reader data reading loop (async).
        
        Actively reads data from assigned files and sends to downstream workers.
        Respects backpressure by pausing when signaled.
        """
        if self.reader_task is None:
            logger.warning(f"Reader worker {self.rank} has no file task, exiting")
            self._reader_completed = True
            return
        
        if self.reader_config is None:
            logger.error(f"Reader worker {self.rank} has no reader config")
            self._reader_completed = True
            return
        
        from ...reader.datasource import create_datasource
        
        try:
            # Create datasource
            datasource = create_datasource(
                data_type=self.reader_config.data_type,
                src_dir=self.reader_config.src_dir,
                file_pattern=self.reader_config.file_pattern,
                encoding=self.reader_config.encoding,
            )
            
            batch_size = self.reader_config.batch_size
            step = 0
            
            logger.info(
                f"Reader worker {self.rank} starting to read {len(self.reader_task.files)} files"
            )
            
            # Read and send batches
            for batch in datasource.read_batch(self.reader_task, batch_size):
                if not self._running:
                    break
                
                # Wait for backpressure to clear (async)
                while self._is_reading_paused and self._running:
                    logger.debug(f"Reader worker {self.rank} paused due to backpressure")
                    try:
                        await asyncio.wait_for(self._reader_pause_event.wait(), timeout=1.0)
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
                
                logger.debug(f"Reader worker {self.rank} sent batch {step}")
            
            logger.info(
                f"Reader worker {self.rank} completed reading, "
                f"sent {self._batches_sent} batches"
            )
            self._reader_completed = True
            
        except Exception as e:
            logger.error(f"Reader worker {self.rank} error: {e}")
            self._last_error = str(e)
            self._reader_completed = True
    
    async def _send_reader_batch(self, batch: Any, step: int) -> None:
        """
        Send a batch of data to downstream SPMD workers.
        
        Uses round-robin distribution to all downstream workers.
        
        Args:
            batch: Batch of data to send
            step: Current step number
        """
        if not self._successor_handlers:
            logger.warning(f"Reader worker {self.rank} has no successor handlers")
            return
        
        # Build tensor metadata
        meta = TensorMeta(
            tensor_name="data",
            shape=(),  # Will be determined by actual data
            dtype="object",
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
                handler.receive_tensor_data.remote(
                    OpType.FORWARD,
                    "data",
                    step,
                    self.rank,
                    batch,
                    meta,
                )
            except Exception as e:
                logger.error(f"Failed to send batch to {spmd_id}:{target_rank}: {e}")
    
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
        
        logger.info(f"Worker {self.rank} stopping")
    
    async def _execute_operator(self, op_type: OpType, step: int) -> None:
        """Execute an operator: assemble data, execute, send outputs."""
        try:
            # Assemble input data
            input_data = await self._assemble_input(op_type, step)
            
            # Execute operator
            logger.debug(f"Executing: op={op_type.name}, step={step}")
            result = self._op_manager.execute(op_type, input_data)
            
            if result is None:
                return
            
            # Determine output step
            # For eager mode: use global step = local_step * total_rank + rank
            # This ensures unique step numbers across all workers
            is_eager = self._op_manager.is_eager_trigger(op_type) if self._op_manager else False
            if is_eager:
                output_step = step * self._total_rank + self.rank
            else:
                output_step = step
            
            # Send output
            await self._send_output(op_type, output_step, result)
            
            # Clear mailbox
            await self._mailbox.clear_step(op_type, step)
            
        except Exception as e:
            logger.error(f"Operator execution failed: {e}")
            raise
    
    async def _assemble_input(self, op_type: OpType, step: int) -> Dict[str, Any]:
        """Assemble input data from mailbox and static args."""
        result: Dict[str, Any] = {}
        
        # Add static arguments
        result.update(self._op_manager.get_static_args())
        
        # Get all state data (keyed by name)
        states = await self._mailbox.get_all_states(op_type, step)
        if states:
            result["states"] = states
        
        # Get all tensor shards with metadata
        tensor_shards = await self._mailbox.get_all_tensor_shards(op_type, step)
        if tensor_shards:
            # Extract data only for operator execution
            result["tensor_shards"] = {
                name: {rank: item[0] for rank, item in rank_data.items()}
                for name, rank_data in tensor_shards.items()
            }
            # Also provide metadata for backward tracking
            result["tensor_metas"] = {
                name: {rank: item[1] for rank, item in rank_data.items()}
                for name, rank_data in tensor_shards.items()
            }
        
        return result
    
    async def _advance_step(self, op_type: OpType) -> None:
        """Advance execution step counter for an operator."""
        self._current_steps[op_type] += 1
        new_step = self._current_steps[op_type]
        
        # Report op completion to driver
        self._report_op_completed(op_type, new_step)
        
        # Check if this op is in eager_trigger mode
        is_eager = self._op_manager.is_eager_trigger(op_type) if self._op_manager else False
        
        if not is_eager:
            # Normal mode: re-enqueue pending future tasks that match the new step
            async with self._pending_lock:
                pending = self._pending_future_steps[op_type]
                to_enqueue = [s for s in pending if s == new_step]
                self._pending_future_steps[op_type] = [s for s in pending if s != new_step]
            
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
                    self.rank,
                    op_type,
                    step,
                )
            except Exception as e:
                logger.error(f"Failed to report op completion: {e}")
    
    # =========================================================================
    # Data Sending
    # =========================================================================
    
    async def _send_output(self, op_type: OpType, step: int, result: Any) -> None:
        """
        Send output to successor SPMDs.
        
        Uses OperatorManager.get_successor_target_route() for routing.
        Handles self-loops (same node, different op) by calling _self_receive_tensor.
        """
        output_sharding = self._op_manager.get_output_sharding(op_type)
        
        # Extract tensors from result
        tensors_to_send = self._extract_tensors(result, output_sharding)
        
        # Get successor IDs from OperatorManager
        successor_ids = self._op_manager.get_successor_ids(op_type)
        
        for spmd_id in successor_ids:
            # Get successor topology info (contains target_op and is_self_loop)
            succ_info = self._op_manager.get_successor_info(op_type, spmd_id)
            if succ_info is None:
                logger.warning(f"No topology info for successor {spmd_id}")
                continue
            
            # Check condition if exists
            if succ_info.condition_fn is not None:
                if not succ_info.condition_fn(result):
                    logger.debug(f"Condition not met for successor {spmd_id}, skipping")
                    continue
            
            # Get target op type from topology (key change!)
            target_op = succ_info.target_op
            
            # Send each tensor
            for tensor_name, tensor_data in tensors_to_send.items():
                # Build metadata
                meta = self._build_tensor_meta(tensor_name, tensor_data, step)
                
                # Handle self-loop: directly call data receive interface
                if succ_info.is_self_loop:
                    await self._self_receive_tensor(target_op, tensor_name, step, tensor_data, meta)
                    continue
                
                # Cross-node: send via handlers
                handlers = self._successor_handlers.get(spmd_id, {})
                if not handlers:
                    logger.warning(f"No handlers for successor {spmd_id}")
                    continue
                
                # Get routing table from OperatorManager
                routes = self._op_manager.get_successor_target_route(
                    current_rank=self.rank,
                    target_spmd_id=spmd_id,
                    op_type=op_type,
                    output_tensor_name=tensor_name,
                    output_tensor=tensor_data,
                    output_tensor_meta=meta,
                )
                
                # Send according to routing
                for data_mask, target_rank in routes:
                    handler = handlers.get(target_rank)
                    if handler is not None:
                        self._send_tensor(
                            handler, target_op, tensor_name, step, tensor_data, data_mask, meta
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
                        await self._self_receive_state(target_op, state_name, step, state_value)
                    elif self.rank == 0:
                        # Cross-node: send via control plane (only rank 0)
                        self.send_state_argument(spmd_id, state_name, state_value)
    
    def _extract_tensors(self, data: Any, output_sharding: Dict[str, ShardingSpec]) -> Dict[str, Any]:
        """Extract named tensors from output data."""
        tensors: Dict[str, Any] = {}
        
        if isinstance(data, dict):
            for name in output_sharding.keys():
                if name in data:
                    tensors[name] = data[name]
        elif hasattr(data, '__dict__'):
            for name in output_sharding.keys():
                if hasattr(data, name):
                    tensors[name] = getattr(data, name)
        
        # If no specific tensors found, treat entire data as single tensor
        if not tensors and output_sharding:
            first_name = next(iter(output_sharding.keys()))
            tensors[first_name] = data
        
        return tensors
    
    def _build_tensor_meta(self, tensor_name: str, tensor_data: Any, step: int) -> TensorMeta:
        """Build tensor metadata for sending."""
        import numpy as np
        
        shape = ()
        dtype = "unknown"
        is_sparse = False
        global_indices = None
        
        if hasattr(tensor_data, 'shape'):
            shape = tuple(tensor_data.shape)
        if hasattr(tensor_data, 'dtype'):
            dtype = str(tensor_data.dtype)
        
        # Check if sparse (tuple of data and indices)
        if isinstance(tensor_data, tuple) and len(tensor_data) == 2:
            data_part, indices_part = tensor_data
            if isinstance(indices_part, np.ndarray):
                is_sparse = True
                global_indices = indices_part
                if hasattr(data_part, 'shape'):
                    shape = tuple(data_part.shape)
                if hasattr(data_part, 'dtype'):
                    dtype = str(data_part.dtype)
        
        return TensorMeta(
            tensor_name=tensor_name,
            shape=shape,
            dtype=dtype,
            global_indices=global_indices,
            source_spmd_id=self.spmd_id,
            source_rank=self.rank,
            forward_step=step,
            is_sparse=is_sparse,
        )
    
    def _get_successor_op_type(self, current_op: OpType) -> OpType:
        """Map current op type to successor's op type (deprecated - use topology target_op)."""
        return current_op
    
    async def _self_receive_tensor(
        self,
        target_op: OpType,
        name: str,
        step: int,
        data: Any,
        meta: TensorMeta,
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
            f"Self-loop data passing: {name} step={step} "
            f"-> {target_op.name}"
        )
        
        # Directly call receive_tensor_data to trigger target op's data readiness check
        await self.receive_tensor_data(
            op_type=target_op,
            name=name,
            step=step,
            sender_rank=self.rank,
            data=data,
            meta=meta,
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
            logger.warning(f"Failed to slice tensor with slices {slices}, sending original data")
            sliced_data = data
        
        if hasattr(handler, 'receive_tensor_data'):
            handler.receive_tensor_data.remote(op_type, name, step, self.rank, sliced_data, meta)
        else:
            logger.debug(f"Would send tensor {name} to handler: step={step}")
    
    async def _self_receive_state(
        self,
        target_op: OpType,
        name: str,
        step: int,
        data: Any,
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
            f"Self-loop state passing: {name} step={step} "
            f"-> {target_op.name}"
        )
        
        # Directly call receive_state_data to trigger target op's data readiness check
        await self.receive_state_data(
            op_type=target_op,
            name=name,
            step=step,
            data=data,
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
        elif hasattr(data, '__dict__'):
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
        logger.debug(f"Worker {self.rank} received state_argument: {state_name}")
    
    def send_state_argument(
        self,
        target_spmd_id: str,
        state_name: str,
        state_value: Any,
    ) -> None:
        """
        Send state_argument to target SPMD.
        
        Only rank 0 worker should call this method.
        Path: rank0-worker -> driver -> controller -> target-driver -> all-workers
        
        Args:
            target_spmd_id: Target SPMD ID
            state_name: State variable name
            state_value: State value
        
        Raises:
            RuntimeError: If called from non-rank-0 worker
        """
        if self.rank != 0:
            raise RuntimeError("Only rank 0 worker can send state_argument")
        
        if self._driver_handle is None:
            logger.error("No driver handle set, cannot send state_argument")
            return
        
        try:
            self._driver_handle.forward_state_argument_to_controller.remote(
                target_spmd_id,
                state_name,
                state_value,
            )
            logger.debug(
                f"Worker {self.rank} sent state_argument '{state_name}' "
                f"to SPMD {target_spmd_id}"
            )
        except Exception as e:
            logger.error(f"Failed to send state_argument: {e}")
    
    # =========================================================================
    # Heartbeat
    # =========================================================================
    
    def _start_heartbeat(self) -> None:
        """Start heartbeat reporting to driver as an asyncio task."""
        if self._heartbeat_running:
            return
        
        self._heartbeat_running = True
        self._heartbeat_task = asyncio.get_event_loop().create_task(self._heartbeat_loop())
        logger.debug(f"Worker {self.rank} heartbeat started")
    
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
                logger.error(f"Error sending heartbeat: {e}")
            
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
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
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
        logger.debug(f"Reader worker {self.rank} paused")
    
    def resume_reading(self) -> None:
        """
        Resume data reading (for Reader nodes).
        
        Called by driver when downstream backpressure is relieved.
        """
        if self.node_type != NodeType.READER:
            return
        
        self._is_reading_paused = False
        self._reader_pause_event.set()
        logger.debug(f"Reader worker {self.rank} resumed")
    
    def is_reader_completed(self) -> bool:
        """Check if reader has completed reading all data."""
        return self._reader_completed
    
    def get_batches_sent(self) -> int:
        """Get number of batches sent by this reader."""
        return self._batches_sent
