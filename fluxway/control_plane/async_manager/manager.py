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

from fluxway.common.types import AsyncStepConfig, Graph, SPMDOpStep, OpLockState

logger = logging.getLogger(__name__)


class AsyncStepManager:
    """
    Manages async step differences between SPMDs.

    Enforces maximum step difference by locking/unlocking ops.
    """

    def __init__(self, graph: Graph, config: AsyncStepConfig | None = None):
        """
        Initialize AsyncStepManager.

        Args:
            graph: The task graph defining SPMD connections
            config: Async step configuration
        """
        self._graph = graph
        self._config = config or AsyncStepConfig()

        # SPMD step tracking: spmd_id -> SPMDOpStep
        self._spmd_steps: dict[str, SPMDOpStep] = {}

        # Initialize step tracking for all nodes
        for node_name in graph.nodes:
            self._spmd_steps[node_name] = SPMDOpStep(spmd_id=node_name)

        # Op lock states: (spmd_id, op_type) -> OpLockState
        self._lock_states: dict[tuple[str, str], OpLockState] = {}

        # Cache for last issued allowed steps to optimize Controller RPCs
        self._last_allowed_steps: dict[tuple[str, str], int] = {}

        # Lock for thread safety
        self._lock = threading.Lock()

    def report_op_completed(
        self, spmd_id: str, op_name: str, completed_count: int
    ) -> None:
        """
        Report SPMD-level min-count completion.
        """
        with self._lock:
            if spmd_id not in self._spmd_steps:
                self._spmd_steps[spmd_id] = SPMDOpStep(spmd_id=spmd_id)

            spmd_progress = self._spmd_steps[spmd_id]
            spmd_progress.completed_step_count[op_name] = max(
                spmd_progress.completed_step_count.get(op_name, 0), completed_count
            )

    def check_and_enforce_limits(self) -> list[tuple[str, str, int]]:
        """
        Calculate the allowed step for each operator based on topology and limits.

        Returns:
            List of (spmd_id, op_name, allowed_step)
        """
        # TODO(He Jia): Optimize self._graph.edges and self._graph.nodes traversing.
        # No need to call every op with driver.set_allowed_step.remote(...)
        # TODO(He Jia): Maintain a self._last_allowed_steps cache inside AsyncStepManager. Only when the newly calculated allowed_step is greater than the value issued last time will it be placed in the updates list and returned to the Controller.
        updates = []
        with self._lock:
            # Topological limits from edges
            op_topo_limits = {}  # (source_id, source_op) -> min_allowed_step

            for edge in self._graph.edges:
                if not edge.source_op or not edge.target_op:
                    continue

                source_id = edge.source_node
                target_id = edge.target_node
                source_op = edge.source_op
                target_op = edge.target_op

                # Get trailing progress of target
                target_data = self._spmd_steps.get(target_id)
                target_min_step = (
                    target_data.completed_step_count.get(target_op, 0)
                    if target_data
                    else 0
                )

                limit = self._config.default_max_inflight_steps
                target_node = self._graph.nodes.get(target_id)
                if target_node:
                    meta = target_node.spmd_config.operator_metas.get(target_op)
                    if meta:
                        limit = getattr(meta, "max_inflight_steps", limit)

                allowed_step = target_min_step + limit
                key = (source_id, source_op)
                if key not in op_topo_limits or allowed_step < op_topo_limits[key]:
                    op_topo_limits[key] = allowed_step

            for node_id, node in self._graph.nodes.items():
                if node.spmd_config is None:
                    continue
                # Use both to ensure we find all defined ops even in mock/partial configs
                all_ops = set(node.spmd_config.operators.keys()) | set(
                    node.spmd_config.operator_metas.keys()
                )
                for op_name in all_ops:
                    # Start with topological limit if any
                    allowed_step = op_topo_limits.get(
                        (node_id, op_name), 999999999
                    )  # Default "infinity"

                    # Global limit
                    if self._config.max_steps is not None:
                        allowed_step = min(allowed_step, self._config.max_steps)

                    cache_key = (node_id, op_name)
                    if (
                        cache_key not in self._last_allowed_steps
                        or allowed_step > self._last_allowed_steps[cache_key]
                    ):
                        self._last_allowed_steps[cache_key] = allowed_step
                        updates.append((node_id, op_name, allowed_step))

        return updates

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
        # logger.debug(f"Locked {spmd_id} {op_name}: {reason}")

    def _clear_locked(self, spmd_id: str, op_name: str) -> None:
        """Clear op lock (must hold lock)."""
        key = (spmd_id, op_name)
        self._lock_states.pop(key, None)
        # logger.debug(f"Unlocked {spmd_id} {op_name}")

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

    def get_locked_ops(self) -> list[OpLockState]:
        """Get all currently locked ops."""
        with self._lock:
            return list(self._lock_states.values())

    def get_spmd_steps(self, spmd_id: str) -> SPMDOpStep | None:
        """Get current steps for an SPMD."""
        with self._lock:
            return self._spmd_steps.get(spmd_id)

    def get_all_spmd_steps(self) -> dict[str, SPMDOpStep]:
        """Get current steps for all SPMDs."""
        with self._lock:
            return {k: SPMDOpStep(**v.as_dict()) for k, v in self._spmd_steps.items()}

    def is_all_completed(self) -> bool:
        """Return True when every tracked operator has reached its step limit."""
        with self._lock:
            global_max_steps = self._config.max_steps

            for node_id, node in self._graph.nodes.items():
                if node.spmd_config is None:
                    continue

                node_max_steps = node.spmd_config.max_steps
                if node_max_steps is None:
                    node_max_steps = global_max_steps
                if node_max_steps is None:
                    return False

                all_ops = (
                    node.spmd_config.operators.keys()
                    | node.spmd_config.operator_metas.keys()
                )
                if not all_ops:
                    continue

                spmd_progress = self._spmd_steps.get(node_id)
                if spmd_progress is None:
                    return False

                for op_name in all_ops:
                    if (
                        spmd_progress.completed_step_count.get(op_name, 0)
                        < node_max_steps
                    ):
                        return False

            return True

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
                spmd_step.completed_step_count.clear()
            self._lock_states.clear()
