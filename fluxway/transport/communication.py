"""Communication layer for scatter-send and fan-out operations."""

from __future__ import annotations

from typing import Any, Callable, Coroutine, List, Tuple

from fluxway.transport.backend import get_backend
from fluxway.transport.payload import TransportPayload
from fluxway.transport.planner import TransportPlanner
from fluxway.transport.sharding import ShardingSpec
from fluxway.transport.tensor import TransportTensor


import asyncio
import functools
import operator
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
            world_size = 1
            if spec_or_layout.mesh:
                world_size = functools.reduce(
                    operator.mul, spec_or_layout.mesh.shape, 1
                )

            return spec_or_layout.compute_layout(data_shape, world_size)

        raise ValueError(f"Invalid type for source/target: {type(spec_or_layout)}")

    def _plan_and_shard(
        self,
        local_data: TransportTensor,
        source: Union[ShardingSpec, Layout],
        target: Union[ShardingSpec, Layout],
        topology_meta: Any = None,
        dependency: Any = None,
    ):
        """Generates (targets, payload) pairs for distribution."""
        backend = get_backend()

        # Determine data_shape from local_data
        # If ShardingSpec is used, we infer layout based on local data shape.
        # Ideally, ShardingSpec + Implicit Inference allows recovering global context.

        if local_data.is_ragged:
            top_split = local_data.nested_splits[0]
            dim0 = backend.shape(top_split)[0] - 1
            data_shape = (dim0,)
        else:
            data_shape = tuple(backend.shape(local_data.flat_values))

        # Handle dependency or implicit inference
        src_indices = dependency
        if (
            src_indices is None
            and isinstance(source, ShardingSpec)
            and not source.is_content_addressable
        ):
            dim_size = (
                data_shape[source.partition_dim]
                if source.partition_dim < len(data_shape)
                else 0
            )
            src_indices = source.get_expected_local_indices(self.local_rank, dim_size)

        src_shape_arg = (
            None
            if (isinstance(source, ShardingSpec) and source.is_content_addressable)
            else data_shape
        )
        src_layout = self._ensure_layout(source, src_shape_arg)

        dst_shape_arg = src_layout.data_shape
        if isinstance(target, ShardingSpec) and target.is_content_addressable:
            dst_shape_arg = None

        if isinstance(target, ShardingSpec):
            w_size = 1
            if target.mesh:
                w_size = functools.reduce(operator.mul, target.mesh.shape, 1)
            dst_layout = target.compute_layout(
                dst_shape_arg, w_size, topology_meta, dependency
            )
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
            # Scatter-send
            for device_id, count in enumerate(counts_host):
                count = int(count)
                if count == 0:
                    continue

                offset_tensor = backend.select(offsets, 0, device_id)
                count_tensor = backend.select(counts, 0, device_id)

                shard = reordered.dynamic_slice(offset_tensor, count_tensor)

                global_offset = 0
                if plan.global_indices is not None:
                    sorted_globals = backend.take(
                        plan.global_indices, plan.sort_indices
                    )
                    idx_tensor = backend.reshape(offset_tensor, (1,))
                    global_offset = backend.take(sorted_globals, idx_tensor)
                    if backend.shape(global_offset) == (1,):
                        global_offset = backend.reshape(global_offset, ())
                else:
                    global_offset = backend.zeros((), dtype=backend.int32())

                # Include indices if Planner says so (e.g. Combine)
                shard_indices = None
                if plan.global_indices is not None:
                    sorted_globals = backend.take(
                        plan.global_indices, plan.sort_indices
                    )
                    shard_indices = backend.dynamic_slice(
                        sorted_globals, [offset_tensor], [count_tensor]
                    )

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
        topology_meta: Any = None,
        dependency: Any = None,
    ) -> List[Any]:
        """Performs synchronous distribution."""
        results = []
        for targets, payload in self._plan_and_shard(
            local_data, source, target, topology_meta, dependency
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
        topology_meta: Any = None,
        dependency: Any = None,
    ) -> List[Any]:
        """Performs asynchronous distribution."""
        tasks = []
        for targets, payload in self._plan_and_shard(
            local_data, source, target, topology_meta, dependency
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
