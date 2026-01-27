"""
RPC Payload for ULP protocol.

Based on ARCHITECTURE.md Section 4.6

Design principle: Keep transmission extremely lightweight, only essential data.

RPCPayload structure:
- batch_id: Unique batch identifier
- data: Data shard (Tensor)
- global_indices: Original global indices for Sparse->Dense restoration

Design notes:
- No sender_id or complex metadata needed
- Minimize metadata to reduce transmission overhead
- global_indices supports order restoration for sparse data
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RPCPayload:
    """
    RPC Payload for data transmission.

    Kept minimal for reduced transmission overhead.
    Receiver doesn't need sender_id - decentralized design.
    """

    batch_id: str
    data: Optional[np.ndarray] = None
    global_indices: Optional[np.ndarray] = None

    @property
    def is_empty(self) -> bool:
        """Check if payload has no data."""
        return self.data is None or len(self.data) == 0

    @property
    def size(self) -> int:
        """Get payload data size."""
        if self.data is None:
            return 0
        return self.data.size


def create_payload(
    batch_id: str, data: np.ndarray, global_indices: Optional[np.ndarray] = None
) -> RPCPayload:
    """
    Create an RPC payload.

    Args:
        batch_id: Unique batch identifier
        data: Data array to send
        global_indices: Optional original indices for order restoration

    Returns:
        RPCPayload object

    TODO: Validate data
    TODO: Consider compression for sparse data
    """
    return RPCPayload(batch_id=batch_id, data=data, global_indices=global_indices)


def create_empty_payload(batch_id: str) -> RPCPayload:
    """
    Create an empty payload for keep-alive.

    Critical for preventing receiver deadlock.
    Even if sender has no data for a rank, must send empty payload
    so receiver's ticket count can reach limit.

    Args:
        batch_id: Unique batch identifier

    Returns:
        Empty RPCPayload object
    """
    return RPCPayload(batch_id=batch_id, data=None, global_indices=None)
