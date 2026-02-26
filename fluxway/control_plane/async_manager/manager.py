"""
AsyncStepManager for controlling maximum async step difference between SPMDs.

Responsibilities:
1. Maintain op-level step counts for each SPMD
2. Detect step difference between connected SPMDs exceeds threshold
3. Lock the ahead SPMD's corresponding op when threshold exceeded
4. Unlock when step difference shrinks within threshold

Key logic:
- Check step difference for all functional edges defined in the graph
- Has minimum lock duration to prevent frequent lock/unlock
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from ...common.types import AsyncStepConfig, Graph, SPMDOpStep, OpLockState

logger = logging.getLogger(__name__)


class AsyncStepManager:
    """
    Manages async step differences between SPMDs.

    Enforces maximum step difference by locking/unlocking ops.
    """

    def __init__(self, graph: Graph, config: Optional[AsyncStepConfig] = None):
        """
        Initialize AsyncStepManager.

        Args:
            graph: The task graph defining SPMD connections
            config: Async step configuration
        """
        self._graph = graph
        self._config = config or AsyncStepConfig()

        # SPMD step tracking: spmd_id -> SPMDOpStep
        self._spmd_steps: Dict[str, SPMDOpStep] = {}

        # Initialize step tracking for all nodes
        for node_name in graph.nodes:
            self._spmd_steps[node_name] = SPMDOpStep(spmd_id=node_name)

        # Op lock states: (spmd_id, op_type) -> OpLockState
        self._lock_states: Dict[Tuple[str, str], OpLockState] = {}

        # Lock for thread safety
        self._lock = threading.Lock()

    def report_op_completed(self, spmd_id: str, op_name: str, step: int) -> None:
        """
        Report that an SPMD completed an op step.

        Args:
            spmd_id: SPMD identifier
            op_name: Operation name
            step: Completed step number
        """
        with self._lock:
            if spmd_id not in self._spmd_steps:
                logger.warning(f"Unknown SPMD {spmd_id} reported op completion")
                return

            spmd_step = self._spmd_steps[spmd_id]
            spmd_step.op_steps[op_name] = max(spmd_step.op_steps.get(op_name, 0), step)

            logger.debug(
                f"SPMD {spmd_id} completed {op_name} step {step}, "
                f"current steps: {spmd_step.op_steps}"
            )

    def report_op_steps(self, spmd_id: str, op_steps: Dict[str, int]) -> None:
        """
        Report all op steps for an SPMD (from heartbeat).

        Args:
            spmd_id: SPMD identifier
            op_steps: Dict mapping op_name to step number
        """
        with self._lock:
            if spmd_id not in self._spmd_steps:
                logger.warning(f"Unknown SPMD {spmd_id} reported op steps")
                return

            spmd_step = self._spmd_steps[spmd_id]

            for op_name, step in op_steps.items():
                spmd_step.op_steps[op_name] = max(
                    spmd_step.op_steps.get(op_name, 0), step
                )

    def check_and_enforce_limits(self) -> List[Tuple[str, str, bool]]:
        """
        Check step differences and determine lock/unlock actions.

        Returns:
            List of (spmd_id, op_name, is_lock) tuples indicating required actions.
            is_lock=True means lock the op, is_lock=False means unlock.
        """
        actions: List[Tuple[str, str, bool]] = []
        current_time = time.time()

        with self._lock:
            for edge in self._graph.edges:
                if not edge.source_op or not edge.target_op:
                    continue

                source = edge.source_node
                target = edge.target_node
                source_op = edge.source_op
                target_op = edge.target_op

                if source not in self._spmd_steps or target not in self._spmd_steps:
                    continue

                source_step = self._spmd_steps[source].op_steps.get(source_op, 0)
                target_step = self._spmd_steps[target].op_steps.get(target_op, 0)

                diff = source_step - target_step

                # Check if need to lock source's op
                if diff > self._config.max_step_diff:
                    if not self._is_locked(source, source_op):
                        actions.append((source, source_op, True))
                        self._set_locked(
                            source,
                            source_op,
                            current_time,
                            f"Step diff {diff} > {self._config.max_step_diff} "
                            f"(edge: {source}.{source_op} -> {target}.{target_op})",
                        )

                # Check if can unlock (diff within half of max)
                elif diff <= self._config.max_step_diff // 2:
                    if self._is_locked(source, source_op):
                        if self._can_unlock(source, source_op, current_time):
                            actions.append((source, source_op, False))
                            self._clear_locked(source, source_op)

        return actions

    def _is_locked(self, spmd_id: str, op_name: str) -> bool:
        """Check if an op is locked (must hold lock)."""
        key = (spmd_id, op_name)
        return key in self._lock_states

    def _set_locked(
        self, spmd_id: str, op_name: str, timestamp: float, reason: str
    ) -> None:
        """Set op as locked (must hold lock)."""
        key = (spmd_id, op_name)
        self._lock_states[key] = OpLockState(
            spmd_id=spmd_id, op_name=op_name, locked_at=timestamp, reason=reason
        )
        logger.info(f"Locked {spmd_id} {op_name}: {reason}")

    def _clear_locked(self, spmd_id: str, op_name: str) -> None:
        """Clear op lock (must hold lock)."""
        key = (spmd_id, op_name)
        self._lock_states.pop(key, None)
        logger.info(f"Unlocked {spmd_id} {op_name}")

    def _can_unlock(self, spmd_id: str, op_name: str, current_time: float) -> bool:
        """Check if an op can be unlocked (must hold lock)."""
        key = (spmd_id, op_name)
        lock_state = self._lock_states.get(key)

        if lock_state is None:
            return True

        # Check minimum lock duration
        time_locked = current_time - lock_state.locked_at
        if time_locked < self._config.min_lock_duration:
            return False

        return True

    def get_locked_ops(self) -> List[OpLockState]:
        """Get all currently locked ops."""
        with self._lock:
            return list(self._lock_states.values())

    def get_spmd_steps(self, spmd_id: str) -> Optional[SPMDOpStep]:
        """Get current steps for an SPMD."""
        with self._lock:
            return self._spmd_steps.get(spmd_id)

    def get_all_spmd_steps(self) -> Dict[str, SPMDOpStep]:
        """Get current steps for all SPMDs."""
        with self._lock:
            return {k: SPMDOpStep(**v.as_dict()) for k, v in self._spmd_steps.items()}

    def is_op_locked(self, spmd_id: str, op_name: str) -> bool:
        """Check if a specific op is locked."""
        with self._lock:
            return self._is_locked(spmd_id, op_name)

    def force_lock_op(self, spmd_id: str, op_name: str, reason: str) -> None:
        """Forcefully lock an op (external request)."""
        with self._lock:
            if not self._is_locked(spmd_id, op_name):
                self._set_locked(spmd_id, op_name, time.time(), reason)

    def force_unlock_op(self, spmd_id: str, op_name: str) -> None:
        """Forcefully unlock an op (external request)."""
        with self._lock:
            self._clear_locked(spmd_id, op_name)

    def reset(self) -> None:
        """Reset all step tracking and locks."""
        with self._lock:
            for spmd_step in self._spmd_steps.values():
                spmd_step.op_steps.clear()
            self._lock_states.clear()
