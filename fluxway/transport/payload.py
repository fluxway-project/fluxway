from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class TransportPayload:
    """
    Payload for transport layer RPC calls.
    Back-end agnostic.
    """

    # 1. Start with metadata (Global Offset)
    global_offset: int

    # 2. DataShard: FlatValues slice
    # Stored as generic tensor (Torch Tensor, JAX Array, etc.)
    data: Any

    # 3. StructureMeta: NestedSplits for this shard
    nested_splits: List[Any]

    # 4. RoutingMeta (Optional)
    routing_meta: Optional[Any] = None

    # Serialization helper if needed
    def __repr__(self):
        # We can't access .shape directly on Any unless we know it's a tensor-like object
        # But usually string representation is for debugging.
        shape = getattr(self.data, "shape", "unknown")
        return f"TransportPayload(offset={self.global_offset}, shape={shape}, splits_len={len(self.nested_splits)})"
