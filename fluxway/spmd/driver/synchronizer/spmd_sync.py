"""
SPMDSynchronizer for managing worker step synchronization within an SPMD.

Responsibilities:
1. Control maximum step difference between workers within an SPMD (via SyncPolicy)
2. Aggregate worker op completion, update SPMD-level step count
3. Report op completion to Controller
4. Provide SPMD-level op blocking interface (called by Controller)
"""

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class SyncPolicy(ABC):
    """Abstract interface for operator dependency synchronization logic."""

    @abstractmethod
    def evaluate(
        self,
        worker_steps: dict[int, dict[str, int]],
        num_workers: int,
        rank: int,
        op_name: str,
        max_step: int,
        min_step: int,
    ) -> bool:
        """
        Evaluate if a worker should be blocked based on the synchronization policy.

        Returns True if the worker should be blocked.
        """
        pass


class MaxSkewSyncPolicy(SyncPolicy):
    """Blocks a worker if its step is too far ahead of the slowest peer."""

    def __init__(self, max_skew: int = 1):
        self.max_skew = max_skew

    def evaluate(
        self,
        worker_steps: dict[int, dict[str, int]],
        num_workers: int,
        rank: int,
        op_name: str,
        max_step: int,
        min_step: int,
    ) -> bool:
        skew_diff = max_step - min_step
        worker_step = worker_steps[rank].get(op_name, 0)
        return skew_diff > self.max_skew and worker_step >= max_step


class SPMDSynchronizer:
    """
    Manages worker step synchronization within a single SPMD.

    Tracks individual worker steps, enforces step constraints via a SyncPolicy,
    and reports SPMD-level progress to controller.
    """

    def __init__(
        self, spmd_id: str, num_workers: int, sync_policy: SyncPolicy | None = None
    ):
        self.spmd_id = spmd_id
        self.num_workers = num_workers
        self.sync_policy = sync_policy or MaxSkewSyncPolicy()

        # Worker-level completed count tracking: rank -> {op_name -> count}
        self._worker_completed_count: dict[int, dict[str, int]] = {
            rank: {} for rank in range(num_workers)
        }

        # SPMD-level completed count (minimum count reached by all workers)
        self._spmd_completed_count: dict[str, int] = {}

        # Op lock states (controller-requested)
        self._op_locked: dict[str, bool] = {}

        # Worker lock states (for step difference enforcement)
        self._worker_op_locked: dict[int, dict[str, bool]] = {
            rank: {} for rank in range(num_workers)
        }

        # Global flow control: op_name -> max allowed step (watermark from Controller)
        self._allowed_steps: dict[str, int] = {}

        self._lock = threading.Lock()

        self._on_spmd_op_completed: Callable[[str, str, int], None] | None = None
        self._block_worker_op: Callable[[int, str], None] | None = None
        self._unblock_worker_op: Callable[[int, str], None] | None = None

        self._expected_ops: set = set()

        # Drain parameters
        self._drain_target_count: int | None = None
        self._drain_on_drained: Callable[[], None] | None = None
        self._drain_acked_ops: set = set()  # ops where all workers reached ceiling

    def load_spmd_state(self, spmd_state: dict[str, Any]) -> None:
        """Load step counts for all workers uniformly from SPMD-level logical state.

        Since ``completed_step_count`` in ``spmd_state`` is topology-independent
        (all ranks share the same logical progress), this method broadcasts the
        counts to every rank slot, ensuring no rank starts at step 0 after restore.
        """
        with self._lock:
            for op_name, count in spmd_state.get("completed_step_count", {}).items():
                for rank in range(self.num_workers):
                    self._worker_completed_count[rank][op_name] = count
                self._spmd_completed_count[op_name] = count
                if (
                    op_name not in self._allowed_steps
                    or self._allowed_steps[op_name] < count
                ):
                    self._allowed_steps[op_name] = count

            logger.debug(
                "SPMD %s loaded spmd_state steps=%s for %d workers",
                self.spmd_id,
                spmd_state.get("completed_step_count", {}),
                self.num_workers,
            )

    def register_expected_ops(self, ops: list[str]) -> None:
        """Register statically known operators to properly anchor drain ceilings.
        Can be called additively to prepare for Eager Mode dynamic scheduling.
        """
        with self._lock:
            self._expected_ops.update(ops)

    def register_expected_op(self, op: str) -> None:
        """Register a single dynamically dispatched operator for Eager Mode."""
        with self._lock:
            self._expected_ops.add(op)

    def _active_ops_for_drain(self) -> set:
        """Drain denominator: expected ops that have actually been executed.

        An op present in ``_expected_ops`` but never reported (e.g. skipped by
        Dispatcher for every rank, or a conditional Eager op that never ran on
        this SPMD) must not block drain completion.  Only ops that have reached
        ``_spmd_completed_count`` -- meaning at least one worker completed at
        least one step -- are included.

        Must be called while holding ``_lock``.
        """
        if self._expected_ops:
            return self._expected_ops & set(self._spmd_completed_count.keys())
        return set(self._spmd_completed_count.keys())

    def set_callbacks(
        self,
        on_spmd_op_completed: Callable[[str, str, int], None] | None = None,
        block_worker_op: Callable[[int, str], None] | None = None,
        unblock_worker_op: Callable[[int, str], None] | None = None,
    ) -> None:
        self._on_spmd_op_completed = on_spmd_op_completed
        self._block_worker_op = block_worker_op
        self._unblock_worker_op = unblock_worker_op

    def report_worker_op_completed(
        self, rank: int, op_name: str, completed_count: int
    ) -> None:
        should_report = False
        drain_complete = False
        callback = None
        with self._lock:
            if rank not in self._worker_completed_count:
                logger.warning(f"Unknown worker rank {rank} reported op completion")
                return

            old_count = self._worker_completed_count[rank].get(op_name, 0)
            self._worker_completed_count[rank][op_name] = max(
                old_count, completed_count
            )

            logger.debug(
                f"Worker {rank} reached {op_name} total_count {completed_count} in SPMD {self.spmd_id}"
            )

            self._check_worker_skew_internal(op_name)

            if self._all_workers_reached_count_internal(op_name, completed_count):
                old_spmd_count = self._spmd_completed_count.get(op_name, 0)
                self._spmd_completed_count[op_name] = completed_count
                logger.debug(
                    f"SPMD {self.spmd_id} reached {op_name} total_count {completed_count}"
                )
                should_report = completed_count > old_spmd_count

            # Drain ACK: All workers have fully completed the requested number of steps.
            # They are now idle at the START of step index == _drain_target_count.
            if (
                self._drain_target_count is not None
                and op_name not in self._drain_acked_ops
                and self._all_workers_reached_count_internal(
                    op_name, self._drain_target_count
                )
            ):
                self._drain_acked_ops.add(op_name)
                known_ops = self._active_ops_for_drain()
                if self._drain_acked_ops >= known_ops and known_ops:
                    drain_complete = True
                    # Clear immediately so subsequent completions cannot re-fire the callback.
                    callback = self._drain_on_drained
                    self._drain_on_drained = None

        if should_report and self._on_spmd_op_completed:
            try:
                self._on_spmd_op_completed(self.spmd_id, op_name, completed_count)
            except Exception as e:
                logger.error(f"Error reporting op completion: {e}")

        if drain_complete and callback:
            callback()

    def _check_worker_skew_internal(self, op_name: str) -> None:
        """Check worker count skew and enforce limits. Must hold lock when calling."""
        counts = [
            self._worker_completed_count[r].get(op_name, 0)
            for r in range(self.num_workers)
        ]
        max_count = max(counts)
        min_count = min(counts)

        allowed_limit = self._allowed_steps.get(op_name)

        for rank in range(self.num_workers):
            worker_count = self._worker_completed_count[rank].get(op_name, 0)

            should_block_policy = self.sync_policy.evaluate(
                self._worker_completed_count,
                self.num_workers,
                rank,
                op_name,
                max_count,
                min_count,
            )
            should_block_global = (
                allowed_limit is not None and worker_count > allowed_limit
            )

            if (should_block_policy or should_block_global) or self._op_locked.get(
                op_name, False
            ):
                if not self._worker_op_locked[rank].get(op_name, False):
                    self._worker_op_locked[rank][op_name] = True
                    reason = (
                        "policy"
                        if should_block_policy
                        else ("global" if should_block_global else "controller")
                    )
                    logger.debug(
                        f"Blocking rank {rank} {op_name} (count {worker_count}, reason={reason})"
                    )
                    if self._block_worker_op:
                        self._block_worker_op(rank, op_name)
            else:
                if self._worker_op_locked[rank].get(op_name, False):
                    self._worker_op_locked[rank][op_name] = False
                    logger.debug(f"Unblocking rank {rank} {op_name}")
                    if self._unblock_worker_op:
                        self._unblock_worker_op(rank, op_name)

    def _all_workers_reached_count_internal(self, op_name: str, count: int) -> bool:
        """Check if all workers have reached the given count. Must hold lock."""
        return all(
            self._worker_completed_count[r].get(op_name, 0) >= count
            for r in range(self.num_workers)
        )

    # =========================================================================
    # Drain-to-Step Control
    # =========================================================================

    @property
    def is_draining(self) -> bool:
        """True if a drain ceiling is active."""
        with self._lock:
            return self._drain_target_count is not None

    def drain_to_step_count(
        self, target_count: int, on_drained: Callable[[], None]
    ) -> None:
        """
        Request the SPMD to drain until every worker has COMPLETED exactly 'target_count' steps.
        """
        drain_complete = False
        with self._lock:
            if self._drain_target_count is not None:
                raise RuntimeError(
                    f"SPMD {self.spmd_id} is already draining to count "
                    f"{self._drain_target_count}; call clear_drain() first"
                )
            self._drain_target_count = target_count
            self._drain_on_drained = on_drained
            self._drain_acked_ops = set()
            logger.debug(f"Drain target set to {target_count} spmd_id={self.spmd_id}")

            # Instantly check if we are ALREADY drained
            known_ops = self._active_ops_for_drain()
            for op_name in known_ops:
                if self._all_workers_reached_count_internal(op_name, target_count):
                    self._drain_acked_ops.add(op_name)

            # Require at least one op to have been registered before declaring
            # drain complete.  An empty known_ops means no work has started at
            # all; calling on_drained at that point would produce a premature
            # empty snapshot.  This is consistent with the guard in
            # report_worker_op_completed.
            if self._drain_acked_ops >= known_ops and known_ops:
                drain_complete = True
                self._drain_on_drained = None

        if drain_complete:
            on_drained()

    def clear_drain(self) -> None:
        """Remove the drain target and unblock workers so training resumes."""
        with self._lock:
            if self._drain_target_count is None:
                return
            self._drain_target_count = None
            self._drain_on_drained = None
            self._drain_acked_ops = set()
            logger.debug(f"Drain cleared spmd_id={self.spmd_id}")

    # =========================================================================
    # SPMD-level Op Blocking (called by Controller)
    # =========================================================================

    def block_spmd_op(self, op_name: str) -> None:
        with self._lock:
            self._op_locked[op_name] = True

        if self._block_worker_op:
            for rank in range(self.num_workers):
                try:
                    self._block_worker_op(rank, op_name)
                except Exception as e:
                    logger.error(f"Error blocking worker {rank} op: {e}")

    def unblock_spmd_op(self, op_name: str) -> None:
        with self._lock:
            self._op_locked[op_name] = False
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
        with self._lock:
            ops = list(self._spmd_completed_count.keys())
        for op_name in ops:
            self.block_spmd_op(op_name)

    def unblock_all_ops(self) -> None:
        with self._lock:
            ops = list(self._spmd_completed_count.keys())
        for op_name in ops:
            self.unblock_spmd_op(op_name)

    def is_op_locked(self, op_name: str) -> bool:
        with self._lock:
            return self._op_locked.get(op_name, False)

    def update_allowed_step(self, op_name: str, step: int) -> None:
        with self._lock:
            old_step = self._allowed_steps.get(op_name, -1)
            if step > old_step:
                self._allowed_steps[op_name] = step
                logger.debug(
                    f"SPMD {self.spmd_id} {op_name} allowed_step updated: {old_step} -> {step}"
                )
                self._check_worker_skew_internal(op_name)

    # =========================================================================
    # Step Query
    # =========================================================================

    def get_spmd_steps(self) -> dict[str, int]:
        with self._lock:
            return dict(self._spmd_completed_count)

    def get_spmd_step(self, op_name: str) -> int:
        with self._lock:
            return self._spmd_completed_count.get(op_name, 0)

    def get_worker_step(self, rank: int, op_name: str) -> int:
        with self._lock:
            return self._worker_completed_count.get(rank, {}).get(op_name, 0)

    def get_all_worker_steps(self) -> dict[int, dict[str, int]]:
        with self._lock:
            return {
                rank: dict(ops) for rank, ops in self._worker_completed_count.items()
            }

    def get_locked_ops(self) -> list[str]:
        with self._lock:
            return [op_name for op_name, locked in self._op_locked.items() if locked]

    # =========================================================================
    # Reset
    # =========================================================================

    def reset(self) -> None:
        with self._lock:
            for rank in range(self.num_workers):
                self._worker_completed_count[rank].clear()
                self._worker_op_locked[rank].clear()
            self._spmd_completed_count.clear()
            self._op_locked.clear()
