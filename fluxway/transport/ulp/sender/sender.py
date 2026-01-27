"""
Sharded Sender for ULP protocol.

Based on ARCHITECTURE.md Section 4.4

Sender workflow principle: Computation complexity handled by C++/CUDA,
Python only responsible for launching kernels.

Workflow:
1. Route Compute: Calculate target rank for each element using vectorized ops
2. Group & Sort: Reorder data by target rank for contiguous grouping
3. Empty Packet Generation: Generate len=0 packets for ranks with no data
4. RPC Launch: Async send data shards to target ranks
"""

from typing import Any, Dict, List, Optional
import numpy as np

from .fluxway.common.types import LayoutSpec, PartitionOp


class ShardedSender:
    """
    Sharded Sender for ULP protocol.

    Handles data partitioning and distribution based on LayoutSpec.
    All routing computation uses vectorized operations.
    """

    def __init__(self, layout_spec: LayoutSpec):
        self._layout_spec = layout_spec
        self._world_size: int = 0

    def set_world_size(self, world_size: int) -> None:
        """Set the world size (number of target ranks)."""
        self._world_size = world_size

    def compute_routes(self, data: np.ndarray) -> np.ndarray:
        """
        Compute target rank for each element.

        Uses vectorized operators based on op_type:
        - DIV_MOD: PID = (val // params[0]) % params[1]
        - HASH_MOD: PID = hash(val) % params[0]
        - LOOKUP: PID = params[val]
        - REPLICATE: PID = all target ranks

        Args:
            data: Input data array

        Returns:
            Array of target rank IDs for each element

        TODO: Implement vectorized routing for each op_type
        TODO: Apply topology_map for logical->physical mapping
        TODO: Use jnp.argsort, torch.gather, etc. for implementation
        """
        pass

    def group_and_sort(self, data: np.ndarray, routes: np.ndarray) -> tuple:
        """
        Group and sort data by target rank.

        Args:
            data: Input data array
            routes: Target rank for each element

        Returns:
            (sorted_data, sorted_indices, split_offsets)

        TODO: Execute argsort(routes) to reorder indices
        TODO: Execute unique_consecutive or bucketize for split points
        TODO: Return ordered DataView and IndexView
        """
        pass

    def generate_payloads(
        self,
        batch_id: str,
        sorted_data: np.ndarray,
        sorted_indices: np.ndarray,
        split_offsets: List[int],
    ) -> Dict[int, Any]:
        """
        Generate payloads for each target rank.

        Critical: Must generate len=0 payloads for ranks with no data
        to prevent receiver deadlock.

        Args:
            batch_id: Unique batch identifier
            sorted_data: Data sorted by target rank
            sorted_indices: Original indices for restoration
            split_offsets: Split points for each rank

        Returns:
            Mapping of rank_id to RPCPayload

        TODO: Split data by offsets
        TODO: Create RPCPayload for each rank
        TODO: Generate empty payloads for missing ranks
        """
        pass

    async def send_async(self, payloads: Dict[int, Any], targets: Dict[int, Any]) -> None:
        """
        Async send payloads to target ranks.

        Args:
            payloads: Mapping of rank_id to payload
            targets: Mapping of rank_id to target handle

        TODO: Initiate async RPC to each target
        TODO: Don't wait for response (pipeline)
        TODO: Support batch sending for reduced overhead
        """
        pass

    async def send(self, batch_id: str, data: np.ndarray, targets: Dict[int, Any]) -> None:
        """
        Complete send workflow.

        TODO: Compute routes
        TODO: Group and sort
        TODO: Generate payloads
        TODO: Send async
        """
        pass
