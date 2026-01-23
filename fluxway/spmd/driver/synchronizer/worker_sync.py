"""
WorkerSynchronizer for managing worker step synchronization within an SPMD.

Responsibilities:
1. Control maximum step difference between workers within an SPMD
2. Aggregate worker op completion, update SPMD-level step count
3. Report op completion to Controller
4. Provide SPMD-level op blocking interface (called by Controller)
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Set

from ....common.types import OpType

logger = logging.getLogger(__name__)


class WorkerSynchronizer:
    """
    Manages worker step synchronization within a single SPMD.
    
    Tracks individual worker steps, enforces maximum step difference,
    and reports SPMD-level progress to controller.
    """
    
    def __init__(
        self, 
        spmd_id: str, 
        worker_count: int,
        max_worker_step_diff: int = 1,
    ):
        """
        Initialize WorkerSynchronizer.
        
        Args:
            spmd_id: SPMD identifier
            worker_count: Number of workers in this SPMD
            max_worker_step_diff: Maximum allowed step difference between workers
        """
        self.spmd_id = spmd_id
        self.worker_count = worker_count
        self.max_worker_step_diff = max_worker_step_diff
        
        # Worker-level step tracking: rank -> {op_type -> step}
        self._worker_steps: Dict[int, Dict[OpType, int]] = {
            rank: {op: 0 for op in OpType}
            for rank in range(worker_count)
        }
        
        # SPMD-level steps (minimum completed by all workers)
        self._spmd_steps: Dict[OpType, int] = {op: 0 for op in OpType}
        
        # Op lock states (controller-requested)
        self._op_locked: Dict[OpType, bool] = {op: False for op in OpType}
        
        # Worker lock states (for step difference enforcement)
        self._worker_op_locked: Dict[int, Dict[OpType, bool]] = {
            rank: {op: False for op in OpType}
            for rank in range(worker_count)
        }
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_spmd_op_completed: Optional[Callable[[str, OpType, int], None]] = None
        self._block_worker_op: Optional[Callable[[int, OpType], None]] = None
        self._unblock_worker_op: Optional[Callable[[int, OpType], None]] = None
    
    def set_callbacks(
        self,
        on_spmd_op_completed: Optional[Callable[[str, OpType, int], None]] = None,
        block_worker_op: Optional[Callable[[int, OpType], None]] = None,
        unblock_worker_op: Optional[Callable[[int, OpType], None]] = None,
    ) -> None:
        """
        Set callback functions.
        
        Args:
            on_spmd_op_completed: Called when SPMD completes an op step (spmd_id, op_type, step)
            block_worker_op: Called to block a worker's op (rank, op_type)
            unblock_worker_op: Called to unblock a worker's op (rank, op_type)
        """
        self._on_spmd_op_completed = on_spmd_op_completed
        self._block_worker_op = block_worker_op
        self._unblock_worker_op = unblock_worker_op
    
    def report_worker_op_completed(
        self, 
        rank: int, 
        op_type: OpType, 
        step: int,
    ) -> None:
        """
        Report that a worker completed an op step.
        
        This method:
        1. Updates worker step count
        2. Checks worker step difference, enforces limit
        3. If all workers completed the step, updates SPMD step and reports to controller
        
        Args:
            rank: Worker rank
            op_type: Operation type
            step: Completed step number
        """
        with self._lock:
            if rank not in self._worker_steps:
                logger.warning(f"Unknown worker rank {rank} reported op completion")
                return
            
            # Update worker step
            old_step = self._worker_steps[rank][op_type]
            self._worker_steps[rank][op_type] = max(old_step, step)
            
            logger.debug(
                f"Worker {rank} completed {op_type.name} step {step} "
                f"in SPMD {self.spmd_id}"
            )
            
            # Check step difference and enforce limits
            self._check_worker_step_diff_internal(op_type)
            
            # Check if all workers completed this step
            if self._all_workers_completed_step_internal(op_type, step):
                # Update SPMD-level step
                old_spmd_step = self._spmd_steps[op_type]
                self._spmd_steps[op_type] = step
                
                logger.info(
                    f"SPMD {self.spmd_id} completed {op_type.name} step {step}"
                )
                
                # Report to controller (outside lock)
                should_report = step > old_spmd_step
        
        # Report completion outside lock
        if should_report and self._on_spmd_op_completed:
            try:
                self._on_spmd_op_completed(self.spmd_id, op_type, step)
            except Exception as e:
                logger.error(f"Error reporting op completion: {e}")
    
    def _check_worker_step_diff_internal(self, op_type: OpType) -> None:
        """
        Check worker step difference and enforce limits.
        
        Must hold lock when calling.
        """
        steps = [self._worker_steps[r][op_type] for r in range(self.worker_count)]
        max_step = max(steps)
        min_step = min(steps)
        diff = max_step - min_step
        
        if diff > self.max_worker_step_diff:
            # Block workers that are too far ahead
            for rank in range(self.worker_count):
                worker_step = self._worker_steps[rank][op_type]
                if worker_step >= max_step and not self._worker_op_locked[rank][op_type]:
                    self._worker_op_locked[rank][op_type] = True
                    logger.info(
                        f"Blocking worker {rank} {op_type.name} in SPMD {self.spmd_id} "
                        f"(step {worker_step}, min={min_step})"
                    )
                    if self._block_worker_op:
                        try:
                            self._block_worker_op(rank, op_type)
                        except Exception as e:
                            logger.error(f"Error blocking worker {rank}: {e}")
        
        elif diff <= self.max_worker_step_diff // 2:
            # Unblock workers when difference is small enough
            for rank in range(self.worker_count):
                if self._worker_op_locked[rank][op_type]:
                    self._worker_op_locked[rank][op_type] = False
                    logger.info(
                        f"Unblocking worker {rank} {op_type.name} in SPMD {self.spmd_id}"
                    )
                    if self._unblock_worker_op:
                        try:
                            self._unblock_worker_op(rank, op_type)
                        except Exception as e:
                            logger.error(f"Error unblocking worker {rank}: {e}")
    
    def _all_workers_completed_step_internal(self, op_type: OpType, step: int) -> bool:
        """Check if all workers completed the given step. Must hold lock."""
        return all(
            self._worker_steps[r][op_type] >= step
            for r in range(self.worker_count)
        )
    
    # =========================================================================
    # SPMD-level Op Blocking (called by Controller)
    # =========================================================================
    
    def block_spmd_op(self, op_type: OpType) -> None:
        """
        Block an SPMD's op (called by Controller).
        
        This blocks the op for all workers.
        
        Args:
            op_type: Operation type to block
        """
        with self._lock:
            self._op_locked[op_type] = True
            logger.info(f"SPMD {self.spmd_id} {op_type.name} blocked by controller")
        
        # Block all workers
        if self._block_worker_op:
            for rank in range(self.worker_count):
                try:
                    self._block_worker_op(rank, op_type)
                except Exception as e:
                    logger.error(f"Error blocking worker {rank} op: {e}")
    
    def unblock_spmd_op(self, op_type: OpType) -> None:
        """
        Unblock an SPMD's op (called by Controller).
        
        This unblocks the op for all workers that aren't blocked
        due to step difference limits.
        
        Args:
            op_type: Operation type to unblock
        """
        with self._lock:
            self._op_locked[op_type] = False
            logger.info(f"SPMD {self.spmd_id} {op_type.name} unblocked by controller")
            
            # Unblock workers (unless they're blocked by step diff)
            workers_to_unblock = [
                rank for rank in range(self.worker_count)
                if not self._worker_op_locked[rank][op_type]
            ]
        
        if self._unblock_worker_op:
            for rank in workers_to_unblock:
                try:
                    self._unblock_worker_op(rank, op_type)
                except Exception as e:
                    logger.error(f"Error unblocking worker {rank} op: {e}")
    
    def block_all_ops(self) -> None:
        """Block all ops for all workers."""
        for op_type in OpType:
            self.block_spmd_op(op_type)
    
    def unblock_all_ops(self) -> None:
        """Unblock all ops for all workers."""
        for op_type in OpType:
            self.unblock_spmd_op(op_type)
    
    def is_op_locked(self, op_type: OpType) -> bool:
        """Check if an op is locked by controller."""
        with self._lock:
            return self._op_locked[op_type]
    
    # =========================================================================
    # Step Query
    # =========================================================================
    
    def get_spmd_steps(self) -> Dict[str, int]:
        """Get SPMD-level steps (as string keys for serialization)."""
        with self._lock:
            return {op.name: step for op, step in self._spmd_steps.items()}
    
    def get_spmd_step(self, op_type: OpType) -> int:
        """Get SPMD-level step for a specific op."""
        with self._lock:
            return self._spmd_steps[op_type]
    
    def get_worker_step(self, rank: int, op_type: OpType) -> int:
        """Get worker's step for a specific op."""
        with self._lock:
            return self._worker_steps.get(rank, {}).get(op_type, 0)
    
    def get_all_worker_steps(self) -> Dict[int, Dict[str, int]]:
        """Get all worker steps."""
        with self._lock:
            return {
                rank: {op.name: step for op, step in ops.items()}
                for rank, ops in self._worker_steps.items()
            }
    
    def get_locked_ops(self) -> List[str]:
        """Get list of locked op names."""
        with self._lock:
            return [op.name for op, locked in self._op_locked.items() if locked]
    
    # =========================================================================
    # Reset
    # =========================================================================
    
    def reset(self) -> None:
        """Reset all step tracking and locks."""
        with self._lock:
            for rank in range(self.worker_count):
                for op in OpType:
                    self._worker_steps[rank][op] = 0
                    self._worker_op_locked[rank][op] = False
            
            for op in OpType:
                self._spmd_steps[op] = 0
                self._op_locked[op] = False
