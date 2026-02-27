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
from typing import Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


class WorkerSynchronizer:
    """
    Manages worker step synchronization within a single SPMD.

    Tracks individual worker steps, enforces maximum step difference,
    and reports SPMD-level progress to controller.
    """

    def __init__(self, spmd_id: str, num_workers: int, max_worker_step_skew: int = 1):
        """
        Initialize WorkerSynchronizer.

        Args:
            spmd_id: SPMD identifier
            num_workers: Number of workers in this SPMD
            max_worker_step_skew: Maximum allowed step difference (skew) between workers
        """
        self.spmd_id = spmd_id
        self.num_workers = num_workers
        self.max_worker_step_skew = max_worker_step_skew

        # Worker-level step tracking: rank -> {op_name -> step}
        self._worker_steps: Dict[int, Dict[str, int]] = {
            rank: {} for rank in range(num_workers)
        }

        # SPMD-level steps (minimum completed by all workers)
        self._spmd_steps: Dict[str, int] = {}

        # Op lock states (controller-requested)
        self._op_locked: Dict[str, bool] = {}

        # Worker lock states (for step difference enforcement)
        self._worker_op_locked: Dict[int, Dict[str, bool]] = {
            rank: {} for rank in range(num_workers)
        }

        # Lock for thread safety
        self._lock = threading.Lock()

        # Callbacks
        self._block_worker_op: Optional[Callable[[int, str], None]] = None
        self._unblock_worker_op: Optional[Callable[[int, str], None]] = None

    def set_callbacks(
        self,
        on_spmd_op_completed: Optional[Callable[[str, str, int], None]] = None,
        block_worker_op: Optional[Callable[[int, str], None]] = None,
        unblock_worker_op: Optional[Callable[[int, str], None]] = None,
    ) -> None:
        """
        Set callback functions.

        Args:
            on_spmd_op_completed: Called when SPMD completes an op step (spmd_id, op_name, step)
            block_worker_op: Called to block a worker's op (rank, op_name)
            unblock_worker_op: Called to unblock a worker's op (rank, op_name)
        """
        self._on_spmd_op_completed = on_spmd_op_completed
        self._block_worker_op = block_worker_op
        self._unblock_worker_op = unblock_worker_op

    def report_worker_op_completed(self, rank: int, op_name: str, step: int) -> None:
        """
        Report that a worker completed an op step.

        This method:
        1. Updates worker step count
        2. Checks worker step difference, enforces limit
        3. If all workers completed the step, updates SPMD step and reports to controller

        Args:
            rank: Worker rank
            op_name: Operation name
            step: Completed step number
        """
        should_report = False
        with self._lock:
            if rank not in self._worker_steps:
                logger.warning(f"Unknown worker rank {rank} reported op completion")
                return

            # Update worker step
            old_step = self._worker_steps[rank].get(op_name, 0)
            self._worker_steps[rank][op_name] = max(old_step, step)

            logger.debug(
                f"Worker {rank} completed {op_name} step {step} in SPMD {self.spmd_id}"
            )

            # Check step difference and enforce limits
            self._check_worker_step_diff_internal(op_name)

            # Check if all workers completed this step
            if self._all_workers_completed_step_internal(op_name, step):
                # Update SPMD-level step
                old_spmd_step = self._spmd_steps.get(op_name, 0)
                self._spmd_steps[op_name] = step

                logger.info(f"SPMD {self.spmd_id} completed {op_name} step {step}")

                # Report to controller (outside lock)
                should_report = step > old_spmd_step

        # Report completion outside lock
        if should_report and self._on_spmd_op_completed:
            try:
                self._on_spmd_op_completed(self.spmd_id, op_name, step)
            except Exception as e:
                logger.error(f"Error reporting op completion: {e}")

    def _check_worker_step_diff_internal(self, op_name: str) -> None:
        """
        Check worker step difference and enforce limits.

        Must hold lock when calling.
        """
        steps = [self._worker_steps[r].get(op_name, 0) for r in range(self.num_workers)]
        max_step = max(steps)
        min_step = min(steps)
        diff = max_step - min_step

        if diff > self.max_worker_step_skew:
            # Block workers that are too far ahead
            for rank in range(self.num_workers):
                worker_step = self._worker_steps[rank].get(op_name, 0)
                if worker_step >= max_step and not self._worker_op_locked[rank].get(
                    op_name, False
                ):
                    self._worker_op_locked[rank][op_name] = True
                    logger.info(
                        f"Blocking worker {rank} {op_name} in SPMD {self.spmd_id} "
                        f"(step {worker_step}, min={min_step})"
                    )
                    if self._block_worker_op:
                        try:
                            self._block_worker_op(rank, op_name)
                        except Exception as e:
                            logger.error(f"Error blocking worker {rank}: {e}")

        elif diff <= self.max_worker_step_skew // 2:
            # Unblock workers when difference is small enough
            for rank in range(self.num_workers):
                if self._worker_op_locked[rank].get(op_name, False):
                    self._worker_op_locked[rank][op_name] = False
                    logger.info(
                        f"Unblocking worker {rank} {op_name} in SPMD {self.spmd_id}"
                    )
                    if self._unblock_worker_op:
                        try:
                            self._unblock_worker_op(rank, op_name)
                        except Exception as e:
                            logger.error(f"Error unblocking worker {rank}: {e}")

    def _all_workers_completed_step_internal(self, op_name: str, step: int) -> bool:
        """Check if all workers completed the given step. Must hold lock."""
        return all(
            self._worker_steps[r].get(op_name, 0) >= step
            for r in range(self.num_workers)
        )

    # =========================================================================
    # SPMD-level Op Blocking (called by Controller)
    # =========================================================================

    def block_spmd_op(self, op_name: str) -> None:
        """
        Block an SPMD's op (called by Controller).

        This blocks the op for all workers.

        Args:
            op_name: Operation name to block
        """
        with self._lock:
            self._op_locked[op_name] = True
            logger.info(f"SPMD {self.spmd_id} {op_name} blocked by controller")

        # Block all workers
        if self._block_worker_op:
            for rank in range(self.num_workers):
                try:
                    self._block_worker_op(rank, op_name)
                except Exception as e:
                    logger.error(f"Error blocking worker {rank} op: {e}")

    def unblock_spmd_op(self, op_name: str) -> None:
        """
        Unblock an SPMD's op (called by Controller).

        This unblocks the op for all workers that aren't blocked
        due to step difference limits.

        Args:
            op_name: Operation name to unblock
        """
        with self._lock:
            self._op_locked[op_name] = False
            logger.info(f"SPMD {self.spmd_id} {op_name} unblocked by controller")

            # Unblock workers (unless they're blocked by step diff)
            workers_to_unblock = [
                rank
                for rank in range(self.num_workers)
                if not self._worker_op_locked[rank].get(op_name, False)
            ]

        if self._unblock_worker_op:
            for rank in workers_to_unblock:
                try:
                    self._unblock_worker_op(rank, op_name)
                except Exception as e:
                    logger.error(f"Error unblocking worker {rank} op: {e}")

    def block_all_ops(self) -> None:
        """Block all ops for all workers."""
        with self._lock:
            ops = list(self._spmd_steps.keys())
        for op_name in ops:
            self.block_spmd_op(op_name)

    def unblock_all_ops(self) -> None:
        """Unblock all ops for all workers."""
        with self._lock:
            ops = list(self._spmd_steps.keys())
        for op_name in ops:
            self.unblock_spmd_op(op_name)

    def is_op_locked(self, op_name: str) -> bool:
        """Check if an op is locked by controller."""
        with self._lock:
            return self._op_locked.get(op_name, False)

    # =========================================================================
    # Step Query
    # =========================================================================

    def get_spmd_steps(self) -> Dict[str, int]:
        """Get SPMD-level steps (as string keys for serialization)."""
        with self._lock:
            return dict(self._spmd_steps)

    def get_spmd_step(self, op_name: str) -> int:
        """Get SPMD-level step for a specific op."""
        with self._lock:
            return self._spmd_steps.get(op_name, 0)

    def get_worker_step(self, rank: int, op_name: str) -> int:
        """Get worker's step for a specific op."""
        with self._lock:
            return self._worker_steps.get(rank, {}).get(op_name, 0)

    def get_all_worker_steps(self) -> Dict[int, Dict[str, int]]:
        """Get all worker steps."""
        with self._lock:
            return {rank: dict(ops) for rank, ops in self._worker_steps.items()}

    def get_locked_ops(self) -> List[str]:
        """Get list of locked op names."""
        with self._lock:
            return [op_name for op_name, locked in self._op_locked.items() if locked]

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:
        """Reset all step tracking and locks."""
        with self._lock:
            for rank in range(self.num_workers):
                self._worker_steps[rank].clear()
                self._worker_op_locked[rank].clear()

            self._spmd_steps.clear()
            self._op_locked.clear()
