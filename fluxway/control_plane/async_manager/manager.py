"""
AsyncStepManager for controlling maximum async step difference between SPMDs.

Responsibilities:
1. Maintain op-level step counts for each SPMD
2. Detect step difference between connected SPMDs exceeds threshold
3. Lock the ahead SPMD's corresponding op when threshold exceeded
4. Unlock when step difference shrinks within threshold

Key logic:
- Forward edges check forward op step difference
- Backward edges check backward op step difference
- Has minimum lock duration to prevent frequent lock/unlock
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ...common.types import (
    AsyncStepConfig,
    Graph,
    OpType,
    SPMDOpStep,
    OpLockState,
)

logger = logging.getLogger(__name__)


class AsyncStepManager:
    """
    Manages async step differences between SPMDs.
    
    Enforces maximum step difference by locking/unlocking ops.
    """
    
    def __init__(
        self, 
        graph: Graph, 
        config: Optional[AsyncStepConfig] = None,
    ):
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
        
        # Build edge map for quick lookup
        self._forward_edges = self._build_edge_map()
        self._backward_edges = self._build_reverse_edge_map()
    
    def _build_edge_map(self) -> Dict[str, List[str]]:
        """Build forward edge map: source -> [targets]"""
        edge_map: Dict[str, List[str]] = {name: [] for name in self._graph.nodes}
        for edge in self._graph.edges:
            edge_map[edge.source_node].append(edge.target_node)
        return edge_map
    
    def _build_reverse_edge_map(self) -> Dict[str, List[str]]:
        """Build backward edge map: target -> [sources]"""
        edge_map: Dict[str, List[str]] = {name: [] for name in self._graph.nodes}
        for edge in self._graph.edges:
            edge_map[edge.target_node].append(edge.source_node)
        return edge_map
    
    def report_op_completed(
        self, 
        spmd_id: str, 
        op_type: OpType, 
        step: int
    ) -> None:
        """
        Report that an SPMD completed an op step.
        
        Args:
            spmd_id: SPMD identifier
            op_type: Operation type
            step: Completed step number
        """
        with self._lock:
            if spmd_id not in self._spmd_steps:
                logger.warning(f"Unknown SPMD {spmd_id} reported op completion")
                return
            
            spmd_step = self._spmd_steps[spmd_id]
            
            if op_type == OpType.FORWARD:
                spmd_step.forward_step = max(spmd_step.forward_step, step)
            elif op_type == OpType.BACKWARD:
                spmd_step.backward_step = max(spmd_step.backward_step, step)
            elif op_type == OpType.UPDATE:
                spmd_step.update_step = max(spmd_step.update_step, step)
            
            logger.debug(
                f"SPMD {spmd_id} completed {op_type.name} step {step}, "
                f"current steps: fwd={spmd_step.forward_step}, "
                f"bwd={spmd_step.backward_step}, upd={spmd_step.update_step}"
            )
    
    def report_op_steps(self, spmd_id: str, op_steps: Dict[str, int]) -> None:
        """
        Report all op steps for an SPMD (from heartbeat).
        
        Args:
            spmd_id: SPMD identifier
            op_steps: Dict mapping op_type name to step number
        """
        with self._lock:
            if spmd_id not in self._spmd_steps:
                logger.warning(f"Unknown SPMD {spmd_id} reported op steps")
                return
            
            spmd_step = self._spmd_steps[spmd_id]
            
            for op_name, step in op_steps.items():
                if op_name == OpType.FORWARD.name:
                    spmd_step.forward_step = max(spmd_step.forward_step, step)
                elif op_name == OpType.BACKWARD.name:
                    spmd_step.backward_step = max(spmd_step.backward_step, step)
                elif op_name == OpType.UPDATE.name:
                    spmd_step.update_step = max(spmd_step.update_step, step)
    
    def check_and_enforce_limits(self) -> List[Tuple[str, OpType, bool]]:
        """
        Check step differences and determine lock/unlock actions.
        
        Returns:
            List of (spmd_id, op_type, is_lock) tuples indicating required actions.
            is_lock=True means lock the op, is_lock=False means unlock.
        """
        actions: List[Tuple[str, OpType, bool]] = []
        current_time = time.time()
        
        with self._lock:
            # Check forward edges for forward step difference
            for source, targets in self._forward_edges.items():
                source_step = self._spmd_steps[source].forward_step
                
                for target in targets:
                    target_step = self._spmd_steps[target].forward_step
                    diff = source_step - target_step
                    
                    # Check if need to lock source's forward
                    if diff > self._config.max_forward_step_diff:
                        if not self._is_locked(source, OpType.FORWARD):
                            actions.append((source, OpType.FORWARD, True))
                            self._set_locked(
                                source, OpType.FORWARD, current_time,
                                f"Step diff {diff} > {self._config.max_forward_step_diff} "
                                f"(target: {target})"
                            )
                    
                    # Check if can unlock (diff within half of max)
                    elif diff <= self._config.max_forward_step_diff // 2:
                        if self._is_locked(source, OpType.FORWARD):
                            if self._can_unlock(source, OpType.FORWARD, current_time):
                                actions.append((source, OpType.FORWARD, False))
                                self._clear_locked(source, OpType.FORWARD)
            
            # Check backward edges for backward step difference
            # For backward, the gradient flows from target back to source
            for target, sources in self._backward_edges.items():
                target_step = self._spmd_steps[target].backward_step
                
                for source in sources:
                    source_step = self._spmd_steps[source].backward_step
                    # For backward, target should be ahead (gradient comes from target)
                    diff = target_step - source_step
                    
                    if diff > self._config.max_backward_step_diff:
                        if not self._is_locked(target, OpType.BACKWARD):
                            actions.append((target, OpType.BACKWARD, True))
                            self._set_locked(
                                target, OpType.BACKWARD, current_time,
                                f"Backward step diff {diff} > "
                                f"{self._config.max_backward_step_diff} (source: {source})"
                            )
                    
                    elif diff <= self._config.max_backward_step_diff // 2:
                        if self._is_locked(target, OpType.BACKWARD):
                            if self._can_unlock(target, OpType.BACKWARD, current_time):
                                actions.append((target, OpType.BACKWARD, False))
                                self._clear_locked(target, OpType.BACKWARD)
        
        return actions
    
    def _is_locked(self, spmd_id: str, op_type: OpType) -> bool:
        """Check if an op is locked (must hold lock)."""
        key = (spmd_id, op_type.name)
        return key in self._lock_states
    
    def _set_locked(
        self, 
        spmd_id: str, 
        op_type: OpType, 
        timestamp: float,
        reason: str
    ) -> None:
        """Set op as locked (must hold lock)."""
        key = (spmd_id, op_type.name)
        self._lock_states[key] = OpLockState(
            spmd_id=spmd_id,
            op_type=op_type.name,
            locked_at=timestamp,
            reason=reason,
        )
        logger.info(f"Locked {spmd_id} {op_type.name}: {reason}")
    
    def _clear_locked(self, spmd_id: str, op_type: OpType) -> None:
        """Clear op lock (must hold lock)."""
        key = (spmd_id, op_type.name)
        self._lock_states.pop(key, None)
        logger.info(f"Unlocked {spmd_id} {op_type.name}")
    
    def _can_unlock(
        self, 
        spmd_id: str, 
        op_type: OpType, 
        current_time: float
    ) -> bool:
        """Check if an op can be unlocked (must hold lock)."""
        key = (spmd_id, op_type.name)
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
            return {k: SPMDOpStep(**v.model_dump()) for k, v in self._spmd_steps.items()}
    
    def is_op_locked(self, spmd_id: str, op_type: OpType) -> bool:
        """Check if a specific op is locked."""
        with self._lock:
            return self._is_locked(spmd_id, op_type)
    
    def force_lock_op(self, spmd_id: str, op_type: OpType, reason: str) -> None:
        """Forcefully lock an op (external request)."""
        with self._lock:
            if not self._is_locked(spmd_id, op_type):
                self._set_locked(spmd_id, op_type, time.time(), reason)
    
    def force_unlock_op(self, spmd_id: str, op_type: OpType) -> None:
        """Forcefully unlock an op (external request)."""
        with self._lock:
            self._clear_locked(spmd_id, op_type)
    
    def reset(self) -> None:
        """Reset all step tracking and locks."""
        with self._lock:
            for spmd_step in self._spmd_steps.values():
                spmd_step.forward_step = 0
                spmd_step.backward_step = 0
                spmd_step.update_step = 0
            self._lock_states.clear()
