"""
Fluxway Data Plane RPC Protocol.
Used for high-performance peer-to-peer data transmission (Tensors).
Control plane operations (scheduling, heartbeats) still use native Ray.
"""

from typing import Any, List, Optional, Tuple


class DataPlaneCallable:
    """
    Wraps a handle to provide a decoupled interface for data transmission.
    Allows replacing Ray's .remote() with other RPC libraries without changing worker logic.
    """

    def __init__(self, handle: Any):
        self._handle = handle

    async def receive_shards(
        self,
        tuuids: List[int],
        step: int,
        sender_id: str,
        sender_rank: int,
        payloads: List[Any],
        counts: List[int],
        global_shapes: Optional[List[Optional[Tuple[int, ...]]]] = None,
        meta_tensor_state_list: Optional[List[int]] = None,
        slice_compact_desc_params: Optional[List[Optional[List[int]]]] = None,
    ) -> Any:
        """
        Data Plane: Primary method for tensor shard transmission.
        """
        return await self._handle.receive_shards.remote(
            tuuids=tuuids,
            step=step,
            sender_id=sender_id,
            sender_rank=sender_rank,
            payloads=payloads,
            counts=counts,
            global_shapes=global_shapes,
            meta_tensor_state_list=meta_tensor_state_list,
            slice_compact_desc_params=slice_compact_desc_params,
        )

    async def receive_control(
        self, op_name: str, step: int, names: List[str], values: List[Any]
    ) -> Any:
        """
        Data Plane: Method for state data transmission.
        """
        return await self._handle.receive_control.remote(op_name, step, names, values)

    def __repr__(self):
        return f"DataPlaneCallable(handle={self._handle})"
