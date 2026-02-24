"""Communication layer for scatter-send and fan-out operations."""

from __future__ import annotations

from typing import Any, Callable, Coroutine, List, Tuple

from fluxway.transport.backend import get_backend
from fluxway.transport.planner import TransportPlanner
from fluxway.transport.sharding import ShardingSpec
from fluxway.transport.tensor import TransportTensor
from fluxway.transport.execution import compile


import asyncio
from typing import Union, Optional

from fluxway.transport.sharding import Layout


class BaseDataDistributor:
    """Base class for data distribution using TransportPlanner.

    Provides core logic for planning and sharding tensors across a distributed
    mesh network, handling both dense and ragged tensor layouts.
    """

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
        local_data: Union[TransportTensor, List[TransportTensor]],
        source: Union[ShardingSpec, Layout],
        target: Union[ShardingSpec, Layout],
        global_shape: Any = None,
        capacity: Optional[Union[int, List[int]]] = None,
    ):
        """Generates plans and padded sequences for distribution."""
        backend = get_backend()

        # Pack all tensors and lengths
        # local_data can be a TransportTensor or List[TransportTensor]
        tensors_to_pack = (
            [local_data] if isinstance(local_data, TransportTensor) else local_data
        )

        # Primary tensor drives routing
        primary = tensors_to_pack[0]

        # Local shape from primary data.
        if primary.is_ragged:
            top_split = primary.nested_splits[0]
            dim0 = backend.shape(top_split)[0] - 1
            local_shape = (dim0,)
        else:
            local_shape = tuple(backend.shape(primary.flat_values))

        # Routing indices or implicit inference.
        src_indices = primary.global_indices
        if (
            src_indices is None
            and isinstance(source, ShardingSpec)
            and not source.is_content_addressable
        ):
            source_routing_dim = (
                source.routing_dim if source.routing_dim is not None else 0
            )
            dim_size = (
                local_shape[source_routing_dim]
                if source_routing_dim < len(local_shape)
                else 0
            )
            src_indices = source.get_expected_local_indices(
                self.local_rank, dim_size, source_routing_dim
            )

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
            dst_layout = target._to_layout(dst_shape_arg, topology=None)
        else:
            dst_layout = target

        routing_dim = dst_layout.routing_dim
        if routing_dim is None:
            if dst_layout.partitions:
                routing_dim = next(iter(dst_layout.partitions.keys()))
            else:
                routing_dim = 0

        # Set capacity bucket
        actual_req = local_shape[0] if local_shape else 1
        if isinstance(capacity, (list, tuple)):
            selected_capacity = None
            for cap in sorted(capacity):
                if cap >= actual_req:
                    selected_capacity = cap
                    break
            if selected_capacity is None:
                selected_capacity = actual_req
            capacity = selected_capacity
        elif capacity is None:
            capacity = actual_req

        # Prepare payload tuples for JIT
        # format: tuple of (flat_values, None or lengths_tensor)
        payloads = []
        for tt in tensors_to_pack:
            lengths = None
            if tt.is_ragged:
                # Calculate lengths from splits (splits[1:] - splits[:-1])
                splits = tt.nested_splits[0]
                lengths = splits[1:] - splits[:-1]
            payloads.append((tt.flat_values, lengths))

        payloads_tuple = tuple(payloads)

        dst_partitions_tuple = (
            tuple(sorted(dst_layout.partitions.items()))
            if dst_layout.partitions
            else None
        )

        routing_dim = dst_layout.routing_dim

        # Compute plan and sequence packing (JIT compiled)
        (
            padded_buffers_lists,  # List of (padded_data, padded_lengths)
            valid_counts,
            dest_devices,
            replication_factor,
            plan_global_indices,
            padded_routing,
            sort_indices,
        ) = _jit_plan_and_pack(
            payloads_tuple,
            src_indices,
            dst_layout.topology,
            routing_dim,
            dst_layout.data_shape,
            dst_layout.strategy,
            dst_layout.mesh,
            dst_partitions_tuple,
            src_layout.get_local_indices,
            self.local_rank,
            primary.flat_values if dst_layout.is_content_addressable else None,
            capacity,
        )

        counts_host = backend.to_numpy(valid_counts).flatten()
        is_replicated = replication_factor > 1

        if is_replicated:
            # Fan-out
            sorted_dest_host = backend.to_numpy(dest_devices)
            for device_id, count in enumerate(counts_host):
                count = int(count)
                if count == 0:
                    continue

                # Fetch target group (from host memory)
                targets = sorted_dest_host[device_id].flatten().tolist()

                # Dynamic Payload Assembly (filter out None)
                payload_tensors = []
                for p_data, p_len in padded_buffers_lists:
                    payload_tensors.append(p_data[device_id])
                    if p_len is not None:
                        payload_tensors.append(p_len[device_id])

                if padded_routing is not None:
                    payload_tensors.append(padded_routing[device_id])

                yield targets, payload_tensors, [count], sort_indices
        else:
            # Scatter-send
            for device_id, count in enumerate(counts_host):
                count = int(count)
                if count == 0:
                    continue

                # Dynamic Payload Assembly (filter out None)
                payload_tensors = []
                for p_data, p_len in padded_buffers_lists:
                    payload_tensors.append(p_data[device_id])
                    if p_len is not None:
                        payload_tensors.append(p_len[device_id])

                if padded_routing is not None:
                    payload_tensors.append(padded_routing[device_id])

                yield [device_id], payload_tensors, [count], sort_indices


class SyncDataDistributor(BaseDataDistributor):
    """Synchronous Data Distributor."""

    def distribute(
        self,
        local_data: Union[TransportTensor, List[TransportTensor]],
        source: Union[ShardingSpec, Layout],
        target: Union[ShardingSpec, Layout],
        send_fn: Callable[[int, List[Any], List[int]], Any],
        global_shape: Any = None,
        capacity: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[List[Any], Any]:
        """Performs synchronous distribution."""
        results = []
        meta_sort_indices = None
        for targets, tensors, counts, sort_idx in self._plan_and_shard(
            local_data, source, target, global_shape, capacity
        ):
            if meta_sort_indices is None:
                meta_sort_indices = sort_idx
            for target_id in targets:
                res = send_fn(target_id, tensors, counts)
                results.append(res)
        return results, meta_sort_indices


class AsyncDataDistributor(BaseDataDistributor):
    """Asynchronous Data Distributor (using asyncio)."""

    async def distribute(
        self,
        local_data: Union[TransportTensor, List[TransportTensor]],
        source: Union[ShardingSpec, Layout],
        target: Union[ShardingSpec, Layout],
        send_fn: Callable[[int, List[Any], List[int]], Coroutine[Any, Any, Any]],
        global_shape: Any = None,
        capacity: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[List[Any], Any]:
        """Performs asynchronous distribution."""
        tasks = []
        meta_sort_indices = None
        for targets, tensors, counts, sort_idx in self._plan_and_shard(
            local_data, source, target, global_shape, capacity
        ):
            if meta_sort_indices is None:
                meta_sort_indices = sort_idx
            for target_id in targets:
                tasks.append(send_fn(target_id, tensors, counts))

        if not tasks:
            return [], meta_sort_indices

        res = await asyncio.gather(*tasks)
        return res, meta_sort_indices


# compatibility
DataDistributor = SyncDataDistributor


class TransportReceiver:
    """Unified configurable receiver for zero-copy or buffered scenarios.

    Waits for `wait_for_n_slices` slices to arrive, upon which `on_ready_callback` is fired.
    If `wait_for_n_slices=1`, it behaves as an asynchronous stream.
    If `target_buffers` are registered, received data is written directly into them
    via zero-copy mechanisms. Otherwise, it behaves as a buffer collector storing
    slices in an internal queue.
    """

    def __init__(
        self, wait_for_n_slices: int = 1, on_ready_callback: Optional[Callable] = None
    ):
        self.wait_for_n_slices = wait_for_n_slices
        self.on_ready_callback = on_ready_callback
        self.target_buffers: Optional[List[Any]] = None
        self.expect_offsets = {}  # src_rank -> List[offsets]
        self._rx_buffers = []  # For buffering mode
        self._received_count = 0

    def register_zero_copy_buffers(
        self, target_buffers: List[Any], expect_offsets: dict
    ):
        """Registers destination buffers for Zero Copy writing."""
        self.target_buffers = target_buffers
        self.expect_offsets = expect_offsets
        self._received_count = 0

    def receive_slices(
        self, source_rank: int, tensors: List[Any], counts: List[int]
    ) -> bool:
        """Called by network layer when a payload array arrives."""

        # Zero-Copy Mode
        if self.target_buffers is not None:
            if source_rank in self.expect_offsets:
                offsets = self.expect_offsets[source_rank]
                if not isinstance(offsets, (list, tuple)):
                    offsets = [offsets]
                backend = get_backend()

                # Write to target_buffers directly using slice views/updates
                for i in range(len(tensors)):
                    if tensors[i] is None:
                        continue
                    if i >= len(self.target_buffers):
                        continue
                    count = counts[i] if i < len(counts) else counts[0]
                    offset = offsets[i] if i < len(offsets) else offsets[0]

                    # Zero-Copy bounds check mapped for Backend transparency
                    try:
                        actual_buffer_len = backend.shape(self.target_buffers[i])[0]
                        if offset + count > actual_buffer_len:
                            raise RuntimeError(
                                f"Zero-copy receiver buffer [{i}] overflow: offset {offset} + count {count} > {actual_buffer_len}"
                            )
                    except Exception:
                        pass  # Ignore shape-check failures for completely opaque tensors

                    # Zero-Copy Slice Retrieval
                    _ = backend.get_dlpack_slice(
                        self.target_buffers[i], slice(offset, offset + count)
                    )

                    # Functional Fallback Update
                    if backend.name() == "torch":
                        self.target_buffers[i][offset : offset + count] = tensors[i]
                    elif backend.name() == "jax":
                        # using dynamic scale update
                        if hasattr(
                            tensors[i], "shape"
                        ):  # skip flat list like structures if any
                            padded_update = tensors[i]
                            # we only want to write `count` items, but tensors[i] might be padded
                            # we trim it to `count` before dynamic update if needed in JAX
                            trim_size = (count,) + backend.shape(padded_update)[1:]
                            trimmed = backend.dynamic_slice(
                                padded_update, (0,) * len(trim_size), trim_size
                            )
                            self.target_buffers[i] = backend.dynamic_slice_update(
                                self.target_buffers[i],
                                trimmed,
                                (offset,) * len(trim_size),
                            )

            self._received_count += 1

        # Buffered Mode
        else:
            self._rx_buffers.append((tensors, counts))
            self._received_count += 1

        # Check wait criteria
        if (
            self.wait_for_n_slices > 0
            and self._received_count >= self.wait_for_n_slices
        ):
            if self.on_ready_callback:
                self.on_ready_callback(self)

        return True

    def get_buffered_slices(self):
        """Consume and return buffered slices."""
        ret = self._rx_buffers
        self._rx_buffers = []
        return ret


@compile(static_argnums=(3, 4, 5, 7, 8, 9, 11))
def _jit_plan_and_pack(
    payloads_tuple: Tuple[Tuple[Any, Optional[Any]], ...],
    src_indices: Any,
    dst_topology: Any,
    dst_routing_dim: Optional[Union[int, Tuple[int, ...]]],
    dst_data_shape: Tuple[int, ...],
    dst_strategy: Callable,
    dst_mesh: Any,
    dst_partitions: Any,
    src_layout_local_indices_fn: Callable,
    local_rank: int,
    dst_routing_values: Any,
    capacity: int,
):
    """JIT-compiled kernel for planning and sequence packing.

    Returns padded and packed data structures ready for transmission.
    This function leverages backend primitives to trace distribution logic
    statically, reducing runtime overhead for tensor routing and padding.
    """
    backend = get_backend()

    # 1. Compute Plan
    (
        sort_indices,
        dest_devices,
        send_counts,
        global_indices,
        target_world_size,
        replica_size,
    ) = TransportPlanner._compute_plan_internal(
        src_indices=src_indices,
        data_values=dst_routing_values,
        dst_topology=dst_topology,
        dst_routing_dim=dst_routing_dim,
        dst_data_shape=dst_data_shape,
        dst_strategy=dst_strategy,
        dst_mesh=dst_mesh,
        dst_partitions=dict(dst_partitions) if dst_partitions else None,
        src_layout_local_indices_fn=src_layout_local_indices_fn,
        local_rank=local_rank,
    )

    # Determine sending offsets by cumulative sum
    cs = backend.cumsum(send_counts, dim=0)
    zero = backend.zeros((1,), dtype=backend.int32())
    send_offsets = backend.cat([zero, cs[:-1]], dim=0)

    # 2. Flatten and Gather Masked Sequences for each payload
    padded_buffers_lists = []

    for data, lengths in payloads_tuple:
        # Sort data
        reordered_data = backend.take(data, sort_indices)

        # Pad reordered_data to avoid dynamic_slice out of bounds when size < capacity
        dtype_data = getattr(reordered_data, "dtype", None)
        pad_shape_data = (capacity,) + backend.shape(reordered_data)[1:]
        padded_reordered = backend.cat(
            [reordered_data, backend.zeros(pad_shape_data, dtype=dtype_data)], dim=0
        )

        # Vectorized / vmap-friendly packing for data
        def pack_data_for_rank(offset, count, padded_reordered=padded_reordered):
            ndim = len(backend.shape(padded_reordered))
            start_indices = (offset,) + (0,) * (ndim - 1)
            slice_sizes = (capacity,) + backend.shape(padded_reordered)[1:]
            seq = backend.dynamic_slice(padded_reordered, start_indices, slice_sizes)
            return seq

        padded_data = backend.vmap(pack_data_for_rank)(send_offsets, send_counts)

        padded_lengths = None
        if lengths is not None:
            # Sort lengths
            reordered_lengths = backend.take(lengths, sort_indices)
            dtype_len = getattr(reordered_lengths, "dtype", None)
            pad_shape_len = (capacity,) + backend.shape(reordered_lengths)[1:]
            padded_reordered_len = backend.cat(
                [reordered_lengths, backend.zeros(pad_shape_len, dtype=dtype_len)],
                dim=0,
            )

            def pack_len_for_rank(
                offset, count, padded_reordered_len=padded_reordered_len
            ):
                ndim = len(backend.shape(padded_reordered_len))
                start_indices = (offset,) + (0,) * (ndim - 1)
                slice_sizes = (capacity,) + backend.shape(padded_reordered_len)[1:]
                return backend.dynamic_slice(
                    padded_reordered_len, start_indices, slice_sizes
                )

            padded_lengths = backend.vmap(pack_len_for_rank)(send_offsets, send_counts)

        padded_buffers_lists.append((padded_data, padded_lengths))

    # Pack routing indices similarly
    padded_routing = None
    if global_indices is not None:
        sorted_globals = backend.take(global_indices, sort_indices)
        dtype_route = getattr(sorted_globals, "dtype", None)
        pad_shape_route = (capacity,) + backend.shape(sorted_globals)[1:]
        padded_sorted_globals = backend.cat(
            [sorted_globals, backend.zeros(pad_shape_route, dtype=dtype_route)], dim=0
        )

        def pack_routing(offset, count):
            return backend.dynamic_slice(padded_sorted_globals, (offset,), (capacity,))

        padded_routing = backend.vmap(pack_routing)(send_offsets, send_counts)

    return (
        padded_buffers_lists,
        send_counts,
        dest_devices,
        replica_size,
        global_indices,
        padded_routing,
        sort_indices,
    )
