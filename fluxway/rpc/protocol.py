"""
Fluxway Data Plane RPC Protocol.
Used for high-performance peer-to-peer data transmission (Tensors).
Control plane operations (scheduling, heartbeats) still use native Ray.
"""

from typing import Any


class DataPlaneCallable:
    """
    Wraps a handle to provide a decoupled interface for data transmission.
    Allows replacing Ray's .remote() with other RPC libraries without changing worker logic.
    """

    def __init__(self, handle: Any):
        self._handle = handle

    async def receive_shards(self, *args, **kwargs) -> Any:
        """
        Data Plane: Primary method for tensor shard transmission.
        """
        # Current implementation uses Ray, but this is the only place we'd need to change
        # if we move to a different high-performance transport (e.g., gRPC/UCX).
        return await self._handle.receive_shards.remote(*args, **kwargs)

    async def receive_states(self, *args, **kwargs) -> Any:
        """
        Data Plane: Method for state data transmission.
        """
        return await self._handle.receive_states.remote(*args, **kwargs)

    def __repr__(self):
        return f"DataPlaneCallable(handle={self._handle})"
