"""Communication layer for scatter-send and fan-out operations."""

from __future__ import annotations

from typing import Any, Callable, Coroutine, List, Tuple

from fluxway.transport.backend import get_backend
from fluxway.transport.payload import TransportPayload
from fluxway.transport.planner import TransportPlanner
from fluxway.transport.sharding import ShardingSpec
from fluxway.transport.tensor import TransportTensor


import asyncio
from typing import Union

from fluxway.transport.sharding import Layout


class BaseDataDistributor:
    """Base Data distributor logic using TransportPlanner."""

    def __init__(self, local_rank: int):
        self.local_rank = local_rank

    def _ensure_layout(
        self, spec_or_layout: Union[ShardingSpec, Layout], data_shape: Tuple[int, ...]
    ) -> Layout:
        if isinstance(spec_or_layout, Layout):
            return spec_or_layout
        if isinstance(spec_or_layout, ShardingSpec):
            return spec_or_layout._to_layout(data_shape)
        raise ValueError(f"Invalid type for source/target: {type(spec_or_layout)}")

    def _plan_and_shard(
        self,
        local_data: TransportTensor,
        source: Union[ShardingSpec, Layout],
        target: Union[ShardingSpec, Layout],
        global_shape: Any = None,
    ):
        """Generates (targets, payload) pairs for distribution."""
        backend = get_backend()

        # Derive context from TransportTensor (self-describing).
        topology = local_data.topology
        routing_indices = local_data.global_indices

        # Local shape from data.
        if local_data.is_ragged:
            top_split = local_data.nested_splits[0]
            dim0 = backend.shape(top_split)[0] - 1
            local_shape = (dim0,)
        else:
            local_shape = tuple(backend.shape(local_data.flat_values))

        # Routing indices or implicit inference.
        src_indices = routing_indices
        if (
            src_indices is None
            and isinstance(source, ShardingSpec)
            and not source.is_content_addressable
        ):
            dim_size = local_shape[source.dim] if source.dim < len(local_shape) else 0
            src_indices = source.get_expected_local_indices(self.local_rank, dim_size)

        src_shape_arg = (
            None
            if (isinstance(source, ShardingSpec) and source.is_content_addressable)
            else local_shape
        )
        src_layout = self._ensure_layout(source, src_shape_arg)

        # For dst layout, prefer explicit global_shape, then src_layout.data_shape.
        dst_shape_arg = (
            global_shape if global_shape is not None else src_layout.data_shape
        )
        if isinstance(target, ShardingSpec) and target.is_content_addressable:
            dst_shape_arg = None

        if isinstance(target, ShardingSpec):
            dst_layout = target._to_layout(dst_shape_arg, topology, routing_indices)
        else:
            dst_layout = target

        # Compute plan: Diff(L_src, L_dst)
        plan = TransportPlanner.compute_plan(
            src_layout=src_layout,
            dst_layout=dst_layout,
            local_rank=self.local_rank,
            src_indices=src_indices,
            data_values=local_data.flat_values
            if dst_layout.is_content_addressable
            else None,  # Optimization: only pass if needed
        )

        # Reorder data by destination
        reordered = local_data.gather(plan.sort_indices)

        offsets = plan.send_offsets
        counts = plan.send_counts_tensor

        # Batch-fetch counts to host to filter empty sends without N syncs.
        counts_host = backend.to_numpy(counts).flatten()

        if plan.is_replicated:
            # Fan-out: Use explicit destination list from plan
            sorted_dest = backend.take(plan.dest_devices, plan.sort_indices)

            current_offset = 0
            for device_id, count in enumerate(counts_host):
                count = int(count)
                if count == 0:
                    continue

                # Use tensor logic for data slicing (zero-sync data path)
                offset_tensor = backend.select(offsets, 0, device_id)
                count_tensor = backend.select(counts, 0, device_id)

                shard = reordered.dynamic_slice(offset_tensor, count_tensor)

                # Prepare payload
                global_offset = 0
                payload = TransportPayload(
                    global_offset=global_offset,
                    data=shard.flat_values,
                    nested_splits=shard.nested_splits,
                    routing_meta=plan.global_indices,
                )

                # Fetch target group (dynamic slice from tensor to keep metadata flow clean)
                target_sample = backend.dynamic_slice(
                    sorted_dest, [offset_tensor, 0], [1, sorted_dest.shape[1]]
                )
                targets = backend.to_numpy(target_sample).flatten().tolist()

                yield targets, payload

                current_offset += count

        else:
            # Scatter-send: prefer self-describing shard indices, fall back to plan
            for device_id, count in enumerate(counts_host):
                count = int(count)
                if count == 0:
                    continue

                offset_tensor = backend.select(offsets, 0, device_id)
                count_tensor = backend.select(counts, 0, device_id)

                shard = reordered.dynamic_slice(offset_tensor, count_tensor)

                # Prefer shard's propagated indices; fall back to plan's indices
                shard_indices = shard.global_indices
                if shard_indices is None and plan.global_indices is not None:
                    sorted_globals = backend.take(
                        plan.global_indices, plan.sort_indices
                    )
                    shard_indices = backend.dynamic_slice(
                        sorted_globals, [offset_tensor], [count_tensor]
                    )

                if shard_indices is not None:
                    idx_tensor = backend.reshape(
                        backend.take(
                            shard_indices, backend.tensor(0, dtype=backend.int32())
                        ),
                        (),
                    )
                    global_offset = idx_tensor
                else:
                    global_offset = backend.zeros((), dtype=backend.int32())

                payload = TransportPayload(
                    global_offset=global_offset,
                    data=shard.flat_values,
                    nested_splits=shard.nested_splits,
                    routing_meta=shard_indices,
                )

                yield [device_id], payload


class SyncDataDistributor(BaseDataDistributor):
    """Synchronous Data Distributor."""

    def distribute(
        self,
        local_data: TransportTensor,
        source: Union[ShardingSpec, Layout],
        target: Union[ShardingSpec, Layout],
        send_fn: Callable[[int, TransportPayload], Any],
        global_shape: Any = None,
    ) -> List[Any]:
        """Performs synchronous distribution."""
        results = []
        for targets, payload in self._plan_and_shard(
            local_data, source, target, global_shape
        ):
            for target_id in targets:
                res = send_fn(target_id, payload)
                results.append(res)
        return results


class AsyncDataDistributor(BaseDataDistributor):
    """Asynchronous Data Distributor (using asyncio)."""

    async def distribute(
        self,
        local_data: TransportTensor,
        source: Union[ShardingSpec, Layout],
        target: Union[ShardingSpec, Layout],
        send_fn: Callable[[int, TransportPayload], Coroutine[Any, Any, Any]],
        global_shape: Any = None,
    ) -> List[Any]:
        """Performs asynchronous distribution."""
        tasks = []
        for targets, payload in self._plan_and_shard(
            local_data, source, target, global_shape
        ):
            for target_id in targets:
                tasks.append(send_fn(target_id, payload))

        if not tasks:
            return []

        return await asyncio.gather(*tasks)


# compatibility
DataDistributor = SyncDataDistributor


class DataReceiverMixin:
    """Receiver Mixin. Provides buffering capability."""

    def __init__(self):
        self._rx_buffer: List[TransportPayload] = []

    def receive_shard(self, source_rank: int, payload: TransportPayload) -> bool:
        self._rx_buffer.append(payload)
        return True

    def get_received_shards(self) -> List[TransportPayload]:
        shards = self._rx_buffer
        self._rx_buffer = []
        return shards
