"""
Mailbox module for dual-stream convergence.

Based on ARCHITECTURE.md Section 3.2.2

Implements dual-stream data caching:
- WorkerMailbox: Data plane (tensor data from upstream workers)
- ControlMailbox: Control plane (state data from Driver)

Key technical points:
1. Unique Alignment ID: Data and control converge via batch_id + global_step
2. Credit-based Flow Control: Prevent speed mismatch between Controller and data
3. Step-based synchronization: Track current step per operator type
4. Named variables: Support multiple named tensors/states per operator
5. Tensor metadata: Store (data, meta) tuple for routing and backward tracking
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import asyncio

from fluxway.common.types.data_types import OpType

if TYPE_CHECKING:
    from fluxway.common.types.data_types import TensorMeta


class WorkerMailbox:
    """
    Mailbox for caching incoming data in SPMD Worker.
    
    Supports dual-stream convergence with named variables:
    - State data from Driver (control plane): {op_type: {name: {step: data}}}
    - Tensor data from upstream workers (data plane): {op_type: {name: {step: {rank: (data, meta)}}}}
    
    Async-safe for concurrent access using asyncio.Lock.
    """
    
    def __init__(self):
        # {op_type: {name: {step: data}}} - State data from control plane
        self._state_data: Dict[OpType, Dict[str, Dict[int, Any]]] = {
            OpType.FORWARD: {},
            OpType.BACKWARD: {},
            OpType.UPDATE: {},
        }
        # {op_type: {name: {step: {rank: (data, meta)}}}} - Tensor data with metadata
        self._tensor_data: Dict[OpType, Dict[str, Dict[int, Dict[int, Tuple[Any, Optional["TensorMeta"]]]]]] = {
            OpType.FORWARD: {},
            OpType.BACKWARD: {},
            OpType.UPDATE: {},
        }
        self._lock = asyncio.Lock()
    
    # =========================================================================
    # State Data Operations (async versions)
    # =========================================================================
    
    async def put_state(self, op_type: OpType, name: str, step: int, data: Any) -> None:
        """
        Cache state data from control plane.
        
        Args:
            op_type: Operator type this data is for
            name: State variable name
            step: Step number
            data: State data
        """
        async with self._lock:
            if name not in self._state_data[op_type]:
                self._state_data[op_type][name] = {}
            self._state_data[op_type][name][step] = data
    
    async def get_state(self, op_type: OpType, name: str, step: int) -> Optional[Any]:
        """Get state data if available."""
        async with self._lock:
            name_data = self._state_data[op_type].get(name, {})
            return name_data.get(step)
    
    async def get_all_states(self, op_type: OpType, step: int) -> Dict[str, Any]:
        """Get all state data for a step, keyed by name."""
        async with self._lock:
            result = {}
            for name, step_data in self._state_data[op_type].items():
                if step in step_data:
                    result[name] = step_data[step]
            return result
    
    async def has_state(self, op_type: OpType, name: str, step: int) -> bool:
        """Check if state data exists for a specific name."""
        async with self._lock:
            name_data = self._state_data[op_type].get(name, {})
            return step in name_data
    
    async def has_all_states(self, op_type: OpType, step: int, expected_names: List[str]) -> bool:
        """Check if all expected state variables have arrived."""
        async with self._lock:
            for name in expected_names:
                name_data = self._state_data[op_type].get(name, {})
                if step not in name_data:
                    return False
            return True
    
    # =========================================================================
    # Tensor Data Operations (with metadata support, async versions)
    # =========================================================================
    
    async def put_tensor(
        self,
        op_type: OpType,
        name: str,
        step: int,
        rank: int,
        data: Any,
        meta: Optional["TensorMeta"] = None,
    ) -> None:
        """
        Cache tensor data from upstream worker with optional metadata.
        
        Args:
            op_type: Operator type this data is for
            name: Tensor variable name
            step: Step number
            rank: Sender's rank
            data: Tensor shard
            meta: Optional tensor metadata (for routing and backward tracking)
        """
        async with self._lock:
            if name not in self._tensor_data[op_type]:
                self._tensor_data[op_type][name] = {}
            if step not in self._tensor_data[op_type][name]:
                self._tensor_data[op_type][name][step] = {}
            self._tensor_data[op_type][name][step][rank] = (data, meta)
    
    async def get_tensor_shard(
        self,
        op_type: OpType,
        name: str,
        step: int,
        rank: int,
    ) -> Tuple[Optional[Any], Optional["TensorMeta"]]:
        """
        Get tensor shard and its metadata for a specific rank.
        
        Returns:
            Tuple of (data, meta), or (None, None) if not found
        """
        async with self._lock:
            name_data = self._tensor_data[op_type].get(name, {})
            step_data = name_data.get(step, {})
            return step_data.get(rank, (None, None))
    
    async def get_tensor_shards(
        self,
        op_type: OpType,
        name: str,
        step: int,
    ) -> Dict[int, Tuple[Any, Optional["TensorMeta"]]]:
        """
        Get all tensor shards (with metadata) for a step and name.
        
        Returns:
            Dict of {rank: (data, meta)}
        """
        async with self._lock:
            name_data = self._tensor_data[op_type].get(name, {})
            return dict(name_data.get(step, {}))
    
    async def get_tensor_shards_data_only(
        self,
        op_type: OpType,
        name: str,
        step: int,
    ) -> Dict[int, Any]:
        """
        Get all tensor shards (data only, no metadata) for a step and name.
        
        Returns:
            Dict of {rank: data}
        """
        async with self._lock:
            name_data = self._tensor_data[op_type].get(name, {})
            step_data = name_data.get(step, {})
            return {rank: item[0] for rank, item in step_data.items()}
    
    async def get_all_tensor_shards(
        self,
        op_type: OpType,
        step: int,
    ) -> Dict[str, Dict[int, Tuple[Any, Optional["TensorMeta"]]]]:
        """
        Get all tensor shards (with metadata) for a step, keyed by name.
        
        Returns:
            Dict of {name: {rank: (data, meta)}}
        """
        async with self._lock:
            result = {}
            for name, step_data in self._tensor_data[op_type].items():
                if step in step_data:
                    result[name] = dict(step_data[step])
            return result
    
    async def get_all_tensor_shards_data_only(
        self,
        op_type: OpType,
        step: int,
    ) -> Dict[str, Dict[int, Any]]:
        """
        Get all tensor shards (data only) for a step, keyed by name.
        
        Returns:
            Dict of {name: {rank: data}}
        """
        async with self._lock:
            result = {}
            for name, step_data in self._tensor_data[op_type].items():
                if step in step_data:
                    result[name] = {rank: item[0] for rank, item in step_data[step].items()}
            return result
    
    async def get_tensor_shard_count(self, op_type: OpType, name: str, step: int) -> int:
        """Get number of received tensor shards for a step and name."""
        async with self._lock:
            name_data = self._tensor_data[op_type].get(name, {})
            return len(name_data.get(step, {}))
    
    async def has_all_tensors(
        self, 
        op_type: OpType, 
        step: int, 
        expected_counts: Dict[str, int]
    ) -> bool:
        """
        Check if all expected tensor shards have arrived.
        
        Args:
            op_type: Operator type
            step: Step number
            expected_counts: {tensor_name: expected_shard_count}
        """
        async with self._lock:
            for name, expected in expected_counts.items():
                name_data = self._tensor_data[op_type].get(name, {})
                if len(name_data.get(step, {})) < expected:
                    return False
            return True
    
    # =========================================================================
    # Cleanup Operations (async versions)
    # =========================================================================
    
    async def clear_step(self, op_type: OpType, step: int) -> None:
        """Clear data for completed step."""
        async with self._lock:
            # Clear state data for this step across all names
            for name in list(self._state_data[op_type].keys()):
                self._state_data[op_type][name].pop(step, None)
                # Clean up empty name entries
                if not self._state_data[op_type][name]:
                    del self._state_data[op_type][name]
            
            # Clear tensor data for this step across all names
            for name in list(self._tensor_data[op_type].keys()):
                self._tensor_data[op_type][name].pop(step, None)
                # Clean up empty name entries
                if not self._tensor_data[op_type][name]:
                    del self._tensor_data[op_type][name]
    
    async def clear_all(self) -> None:
        """Clear all cached data."""
        async with self._lock:
            for op_type in OpType:
                self._state_data[op_type].clear()
                self._tensor_data[op_type].clear()
    
    # =========================================================================
    # Size and Capacity (for backpressure monitoring, async versions)
    # =========================================================================
    
    async def get_size(self) -> int:
        """
        Get current total number of items in the mailbox.
        
        Counts all state entries and tensor entries.
        """
        async with self._lock:
            size = 0
            # Count state entries
            for op_type in OpType:
                for name_data in self._state_data[op_type].values():
                    size += len(name_data)
            
            # Count tensor entries
            for op_type in OpType:
                for name_data in self._tensor_data[op_type].values():
                    for step_data in name_data.values():
                        size += len(step_data)
            
            return size
    
    async def get_capacity(self) -> int:
        """
        Get mailbox capacity for backpressure calculation.
        
        This is a configurable limit on how many items can be stored.
        Default is 100 items.
        """
        # TODO: Make this configurable
        return 100
