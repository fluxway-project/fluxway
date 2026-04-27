"""Communication layer for scatter-send and fan-out operations."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union, Optional
import logging

import asyncio
import numpy as np

from fluxway.common.backend import get_backend
from fluxway.transport.planner import TransportPlanner
from fluxway.common.types.sharding import PhysicalLayout
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.execution import compile
from fluxway.common.index_descriptor import ContinuousIndex, IndexDescriptor

logger = logging.getLogger(__name__)


class BaseDataDistributor:
    """Base class for data distribution using TransportPlanner.

    Provides core logic for planning and sharding tensors across a distributed
    mesh network, handling both dense and ragged tensor layouts.
    """

    @staticmethod
    def _format_tuuids(tuuid, tuuid_to_name):
        def _fmt(u):
            if tuuid_to_name and u in tuuid_to_name:
                return f"{u} ({tuuid_to_name[u]})"
            return str(u)

        if isinstance(tuuid, (list, tuple)):
            return "[" + ", ".join(_fmt(u) for u in tuuid) + "]"
        return _fmt(tuuid)

    def __init__(self, local_rank: int):
        self.local_rank = local_rank
        from collections import defaultdict

        # Internal aggregation for "Late Aggregation"
        # {target_rank: [(tensors, counts, global_shapes, has_meta_tensor_state_list)]}
        self._dest_agg: Dict[int, List[Tuple]] = defaultdict(list)
        self._meta_sort_indices = None
        self._current_source_spec = None
        self._current_target_spec = None

    def _plan_mirror_group(
        self,
        receipt,
        payloads,
        has_meta_tensor_state_list,
        compact_desc_info,
        global_shapes_list,
    ):
        """Internal logic for Mirror Backflow (using physical receipts)."""
        backend = get_backend()
        mirror_target_world_size = receipt.world_size
        sender_rank_tags = receipt.sender_rank_tags

        # Scalar fast-path: all data goes to one destination.
        # Handle before entering the JIT-compiled _stitch_with_prime to avoid
        # ConcretizationTypeError (int() on traced JAX values).
        is_scalar = (
            isinstance(sender_rank_tags, (int, np.integer))
            or (hasattr(sender_rank_tags, "shape") and np.asarray(sender_rank_tags).ndim == 0)
        )
        if is_scalar:
            target_rank = int(sender_rank_tags)
            counts_np = np.zeros(mirror_target_world_size, dtype=np.int32)
            offsets_np = np.zeros(mirror_target_world_size, dtype=np.int32)
            if 0 <= target_rank < mirror_target_world_size:
                flat_leaves, _ = backend.tree_flatten(payloads)
                if flat_leaves:
                    counts_np[target_rank] = backend.shape(flat_leaves[0])[0]
            stitched_tensors = payloads
            tensors_size = counts_np
            tensor_offsets = offsets_np
        else:
            sender_rank_tags = backend.as_tensor(sender_rank_tags)
            stitched_tensors, tensors_size, tensor_offsets = _stitch_with_prime(
                sender_rank_tags, payloads, mirror_target_world_size
            )

        counts_np = np.array(tensors_size)
        offsets_np = np.array(tensor_offsets)

        for i in range(mirror_target_world_size):
            count = int(counts_np[i])
            rank_flat_tensors = _get_rank_data_from_tree(
                i, stitched_tensors, counts_np, offsets_np
            )

            if count > 0:
                rank_flat_tensors = [t[:count] for t in rank_flat_tensors]
            rank_flat_counts = [count] * len(rank_flat_tensors)

            slice_params = []
            for meta_state, desc_obj in zip(  # type: ignore
                has_meta_tensor_state_list, compact_desc_info, strict=True
            ):
                if meta_state & 4:
                    # TODO(He Jia): Support Cyclic mapping if needed
                    offset = int(offsets_np[i])
                    slice_params.append(
                        desc_obj.slice(offset, offset + count).to_compact_params()
                    )
                else:
                    slice_params.append(None)

            yield (
                [i],
                rank_flat_tensors,
                rank_flat_counts,
                None,
                global_shapes_list,
                has_meta_tensor_state_list,
                slice_params,
            )

    def _plan_single_tensor_group(
        self,
        local_data: Union[TransportTensor, List[TransportTensor]],
        source: PhysicalLayout,
        target: PhysicalLayout,
        capacity: Optional[Union[int, List[int]]] = None,
        tuuid: Optional[Union[int, List[int]]] = None,
        tuuid_to_name: Optional[Dict[int, str]] = None,
    ):
        """Generates plans and padded sequences for a single distribution group.

        A group is defined as one or more tensors that SHARE the same routing indices
        (e.g. follow-sharding or explicitly aligned tensors).
        """

        # Pack all tensors and lengths
        # local_data can be a TransportTensor or List[TransportTensor]
        tensors_to_pack = (
            [local_data] if isinstance(local_data, TransportTensor) else local_data
        )

        # Primary tensor drives routing
        primary = tensors_to_pack[0]

        # Logical shape for routing (respecting Ragged/Padded boundaries)
        local_shape = primary.logical_shape

        src_shape = (
            primary.global_shape if primary.global_shape is not None else local_shape
        )

        # Signal-driven Routing Detection (Referential vs Algebraic)
        is_referential = (source and source.is_referential) or (
            target and target.is_referential
        )
        is_mirror = is_referential

        # 1. Determine Addressability
        # Mirror is never content-addressable (it uses physical receipts)
        if is_referential:
            dst_is_content_addressable = False
        else:
            dst_is_content_addressable = getattr(
                target, "is_content_addressable", False
            )

        # 2. Pack Tensors and Metadata (Required for all paths)
        payloads = []
        has_meta_tensor_state_list = []
        nested_length_list = []
        compact_desc_info: List[Optional[IndexDescriptor]] = []

        for tt in tensors_to_pack:
            payloads.append(tt.flat_values)
            meta_tensor_state = 0

            if tt.is_ragged:
                meta_tensor_state |= 1
                splits = tt.nested_splits[0]
                lengths = splits[1:] - splits[:-1]
                nested_length_list.append(lengths)
                if is_mirror:
                    payloads.append(lengths)
                    nested_length_list.append(None)
            else:
                nested_length_list.append(None)

            if tt.origin_indices is not None:
                if isinstance(tt.origin_indices, ContinuousIndex):
                    meta_tensor_state |= 4
                    compact_desc_info.append(tt.origin_indices)
                else:
                    meta_tensor_state |= 2
                    idx = (
                        tt.origin_indices.materialize()
                        if hasattr(tt.origin_indices, "materialize")
                        else tt.origin_indices
                    )
                    payloads.append(idx)
                    nested_length_list.append(None)
                    compact_desc_info.append(None)
            else:
                compact_desc_info.append(None)

            has_meta_tensor_state_list.append(meta_tensor_state)

        # 3. Explicit Mirror Early-Exit
        if is_referential and primary.receipt is not None:
            global_shapes_list = [tt.global_shape for tt in tensors_to_pack]
            yield from self._plan_mirror_group(
                primary.receipt,
                payloads=payloads,
                has_meta_tensor_state_list=has_meta_tensor_state_list,
                compact_desc_info=compact_desc_info,
                global_shapes_list=global_shapes_list,
            )
            return

        # 1. Extract Initial Routing Signal from Primary Tensor
        src_geometry_indices = primary.origin_indices
        if src_geometry_indices is None:
            src_geometry_indices = primary.routing_indices

        # 2. Resolve Final Source Signal
        if src_geometry_indices is not None:
            # Case A: Explicitly provided indices (e.g. from a previous OP)
            if hasattr(src_geometry_indices, "materialize"):
                src_geometry_indices = src_geometry_indices.materialize()
            dst_content_routing_values = None
        elif source and source.is_content_addressable:
            # Case B: Content-addressable (Dynamic) routing
            src_geometry_indices = primary.flat_values
            dst_content_routing_values = None
        elif source:
            # Case C: Rank-based (Static) routing
            source_routing_dim = source.routing_dim
            if isinstance(source_routing_dim, tuple):
                source_routing_dim = source_routing_dim[0]

            dim_size = (
                local_shape[source_routing_dim]
                if source_routing_dim < len(local_shape)
                else 0
            )
            effective_rank = source.get_shard_rank(self.local_rank, source_routing_dim)
            start = effective_rank * dim_size
            src_geometry_indices = np.arange(start, start + dim_size, dtype=np.int32)
            dst_content_routing_values = None
        else:
            raise ValueError(
                f"[Fluxway Distributor Error] No sharding source available for TUUID(s): {self._format_tuuids(tuuid, tuuid_to_name)}.\n"
                f"  Problem: The algebraic path requires either a valid 'source' layout or explicit 'routing_indices'.\n"
                f"  Fix: Ensure the TransportTensor is initialized with a source PhysicalLayout."
            )

        # Resolve Destination Signal (Content-based Hash)
        dst_is_content_addressable = target.is_content_addressable if target else False
        if dst_is_content_addressable and not is_referential:
            dst_content_routing_values = (
                primary.routing_indices
                if primary.routing_indices is not None
                else primary.flat_values
            )
            if hasattr(dst_content_routing_values, "materialize"):
                dst_content_routing_values = dst_content_routing_values.materialize()
        else:
            dst_content_routing_values = None

        # For dst layout, prefer explicit global_shape, then tensor's global_shape, then src_layout.data_shape.
        dst_shape = (
            primary.global_shape if primary.global_shape is not None else src_shape
        )

        if (
            not is_referential
            and target
            and source
            and not target.is_content_addressable
            and not source.is_content_addressable
            and primary.shape != ()
        ):
            is_fallback_shape = (dst_shape == local_shape) or (
                primary.global_shape is None
            )
            src_mesh_size = source.mesh.size if source.mesh else 1
            # axis=[] → source.partitions == {} → fully replicated; local_shape == global_shape is correct
            is_src_replicated = not source.partitions
            if dst_shape is None or (
                is_fallback_shape and src_mesh_size > 1 and not is_src_replicated
            ):
                raise ValueError(
                    f"[Distributor] Static-to-Static routing requires a valid 'global_shape' to determine shard boundaries. "
                    f"TUUID: {self._format_tuuids(tuuid, tuuid_to_name)}. "
                    f"Please ensure the source TransportTensor has 'global_shape' set explicitly."
                )

        if (
            not is_referential
            and target
            and source
            and not target.is_content_addressable
            and not source.is_content_addressable
            and dst_shape is not None
            and target.partitions
        ):
            for d, axes in target.partitions.items():
                if d >= len(dst_shape):
                    continue
                global_dim = dst_shape[d]
                sub_mesh_size = target.shard_size(d)
                if global_dim > 0 and global_dim < sub_mesh_size:
                    axis_names = list(axes)
                    raise ValueError(
                        f"\n[Fluxway Static Sharding Error] Data dimension {d} has size {global_dim}, "
                        f"but the target mesh needs {sub_mesh_size} elements (axes={axis_names}) "
                        f"to give every worker at least one element.\n"
                        f"  TUUID: {self._format_tuuids(tuuid, tuuid_to_name)}\n"
                        f"  global_shape={dst_shape}, target mesh axes={axis_names}, "
                        f"sub_mesh_size={sub_mesh_size}\n"
                        f"  Fix: Increase the data size along dimension {d} to >= {sub_mesh_size}, "
                        f"or reduce the target mesh size."
                    )

        if source and target:
            logger.debug(
                f"[Distributor] Planning: src_mesh={source.mesh.size if source.mesh else 1}, "
                f"dst_mesh={target.mesh.size if target.mesh else 1}, "
                f"local_rank={self.local_rank}, src_indices={src_geometry_indices}"
            )

        routing_dim = target.routing_dim if target else 0
        if routing_dim is None:
            if target and target.partitions:
                routing_dim = next(iter(target.partitions.keys()))
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

        global_shapes_list = [tt.global_shape for tt in tensors_to_pack]

        is_src_replicated = source is not None and not source.partitions
        if primary.shape == () or is_src_replicated:
            target_world_size = target.mesh.size if target and target.mesh else 1
            src_world_size = source.mesh.size if source and source.mesh else 1

            # Replicated dispatch: responsible_src = floor(j * N / M)
            # Each target j receives exactly one copy from a designated source.
            # N < M → broadcast (contiguous block per source)
            # N > M → selection (only sources 0..M-1 send, others idle)
            # N == M → 1:1 bijection
            targets_to_send = []
            for j in range(target_world_size):
                responsible_src = (j * src_world_size) // target_world_size
                if responsible_src == self.local_rank:
                    targets_to_send.append(j)

            if not targets_to_send:
                return

            # Scalar payload has exactly 1 element; non-scalar carries actual length.
            counts = (
                [1] * len(payloads)
                if primary.shape == ()
                else [len(p) if hasattr(p, "__len__") else 1 for p in payloads]
            )
            yield (
                targets_to_send,
                payloads,
                counts,
                None,
                global_shapes_list,
                has_meta_tensor_state_list,
                [None] * len(has_meta_tensor_state_list),
            )
            return

        if is_mirror and primary.receipt is not None:
            yield from self._plan_mirror_group(
                primary.receipt,
                payloads,
                has_meta_tensor_state_list,
                compact_desc_info,
                global_shapes_list,
            )
            return

        payloads_tuple = tuple(payloads)

        dst_partitions_tuple = (
            tuple(sorted(target.partitions.items())) if target.partitions else None
        )

        target_routing_dim = target.routing_dim

        # Prepare static shapes for JIT padding
        # static_shapes: (item_cap, payload0_flat_cap, payload1_flat_cap, ...)
        static_shapes = [capacity]
        for p_data, p_len in zip(payloads, nested_length_list, strict=True):  # type: ignore
            if p_len is not None:
                static_shapes.append(int(p_data.shape[0]))
            else:
                static_shapes.append(capacity)
        static_shapes_tuple = tuple(static_shapes)

        (
            flat_buffers_lists,
            counts_per_payload,
            offsets_per_payload,
            send_counts,
            send_offsets,
            dest_devices,
            replication_factor,
            sort_indices,
        ) = _jit_plan_and_pack(
            payloads_tuple=payloads_tuple,
            nested_length_list=tuple(nested_length_list),
            src_local_rank=self.local_rank,
            src_data_shape=src_shape,
            src_routing_indices=src_geometry_indices,
            src_routing_based_values=dst_content_routing_values,
            dst_index_desc=target.index_descriptor,
            dst_routing_dim=target_routing_dim,
            dst_strategy=target.strategy,
            dst_mesh=target.mesh,
            dst_partitions=dst_partitions_tuple,
            dst_replica_axes=target.replica_axes,
            dst_replica_size=target.replica_size,
            static_shapes=static_shapes_tuple,
            src_spec=source,
            src_index_desc=primary.topology,
        )

        # Materialise per-device send counts/offsets on the host once.  These are the
        # minimal scalars needed to slice the flat buffers; the heavy tensor data stays
        # on-device until the Python slice below.
        counts_host = np.array(send_counts).flatten()
        offsets_host = np.array(send_offsets).flatten()
        counts_per_payload_host = [np.array(c).flatten() for c in counts_per_payload]
        offsets_per_payload_host = [np.array(o).flatten() for o in offsets_per_payload]

        def _get_slice_compact_desc_params(device_id):
            slice_compact_desc_params: List[Optional[List[int]]] = []
            for meta_tensor_state, desc_obj in zip(  # type: ignore
                has_meta_tensor_state_list,
                compact_desc_info,
                strict=True,  # type: ignore
            ):  # type: ignore
                if meta_tensor_state & 4:
                    # TODO(He Jia): Support Cyclic
                    offset = int(offsets_host[device_id])
                    count = int(counts_host[device_id])
                    slice_compact_desc_params.append(
                        desc_obj.slice(offset, offset + count).to_compact_params()
                    )
                else:
                    slice_compact_desc_params.append(None)
            return slice_compact_desc_params

        backend = get_backend()

        def _get_flat_payloads_for_rank(device_id):
            """Slice this device's segment out of each O(N) flat buffer.

            flat_buffers_lists holds globally-reordered 1-D arrays.  Each device's
            data occupies a contiguous window [offset, offset+count), so a single
            backend.get_contiguous_view call extracts the view.

            On backends with DMA support (NCCL/UCX), the returned view exposes
            __cuda_array_interface__ without any data copy; on Ray the default
            implementation falls back to a plain Python slice.
            """
            p_flat = []
            c_flat = []
            p_idx = 0
            c_idx = 0
            for meta_state in has_meta_tensor_state_list:
                p_data, p_len = flat_buffers_lists[p_idx]
                offset = int(offsets_per_payload_host[c_idx][device_id])
                count = int(counts_per_payload_host[c_idx][device_id])
                p_flat.append(backend.get_contiguous_view(p_data, offset, count))
                c_flat.append(count)

                if p_len is not None:  # ragged: also slice the item-length array
                    len_offset = int(offsets_per_payload_host[c_idx + 1][device_id])
                    count_len = int(counts_per_payload_host[c_idx + 1][device_id])
                    p_flat.append(
                        backend.get_contiguous_view(p_len, len_offset, count_len)
                    )
                    c_flat.append(count_len)
                    c_idx += 1

                p_idx += 1
                c_idx += 1

                if meta_state & 2:  # materialized origin indices
                    p_data, _ = flat_buffers_lists[p_idx]
                    offset = int(offsets_per_payload_host[c_idx][device_id])
                    count = int(counts_per_payload_host[c_idx][device_id])
                    p_flat.append(backend.get_contiguous_view(p_data, offset, count))
                    c_flat.append(count)
                    p_idx += 1
                    c_idx += 1
            return p_flat, c_flat

        is_replicated = replication_factor > 1

        if is_replicated:
            # Fan-out
            sorted_dest_host = np.array(dest_devices)

            for device_id, count in enumerate(counts_host):
                targets = sorted_dest_host[device_id].flatten().tolist()
                p_flat, c_flat = _get_flat_payloads_for_rank(device_id)

                yield (
                    targets,
                    p_flat,
                    c_flat,
                    None,
                    global_shapes_list,
                    has_meta_tensor_state_list,
                    _get_slice_compact_desc_params(device_id),
                )
        else:
            # Scatter-send
            for device_id, count in enumerate(counts_host):
                targets = [device_id]
                p_flat, c_flat = _get_flat_payloads_for_rank(device_id)

                # Optimization: In Static-to-Static sharding, do not send empty bundles.
                is_static = (source.is_static if source else False) and (
                    target.is_static if target else False
                )
                if is_static and sum(c_flat) == 0:
                    continue

                yield (
                    targets,
                    p_flat,
                    c_flat,
                    sort_indices,
                    global_shapes_list,
                    has_meta_tensor_state_list,
                    _get_slice_compact_desc_params(device_id),
                )

    def plan_and_shard(
        self,
        local_data: Union[TransportTensor, List[TransportTensor]],
        source: Optional[PhysicalLayout] = None,
        target: Optional[PhysicalLayout] = None,
        capacity: Optional[Union[int, List[int]]] = None,
        tuuid: Optional[Union[int, List[int]]] = None,
        tuuid_to_name: Optional[Dict[int, str]] = None,
    ) -> None:
        """Computes sharding plan and buffers data for all destinations."""

        # PLAN ISOLATION:
        # We MUST NOT reuse plans across separate calls to plan_and_shard.
        # Each call represents a distinct group of tensors that might have
        # different routing requirements (e.g., different values/indices).
        self._meta_sort_indices = None
        self._current_source_spec = source
        self._current_target_spec = target

        for (
            targets,
            tensors,
            counts,
            sort_idx,
            global_shapes_list,
            has_meta_tensor_state_list,
            slice_compact_desc_params,
        ) in self._plan_single_tensor_group(
            local_data, source, target, capacity, tuuid, tuuid_to_name
        ):
            logger.debug(
                f"[Distributor Plan] [Rank {self.local_rank}] targets={targets} tensors={tensors} counts={counts} sort_idx={sort_idx} global_shapes_list={global_shapes_list} has_meta_tensor_state_list={has_meta_tensor_state_list} slice_compact_desc_params={slice_compact_desc_params}"
            )
            if self._meta_sort_indices is None:
                self._meta_sort_indices = sort_idx
            for target_id in targets:
                # Store tuuid along with data for correct reconstruction in send_fn
                self._dest_agg[target_id].append(
                    (
                        tuuid,
                        tensors,
                        counts,
                        global_shapes_list,
                        has_meta_tensor_state_list,
                        slice_compact_desc_params,
                    )
                )


class SyncDataDistributor(BaseDataDistributor):
    """Synchronous Data Distributor."""

    def distribute(
        self,
        local_data: Optional[Union[TransportTensor, List[TransportTensor]]] = None,
        source: Optional[PhysicalLayout] = None,
        target: Optional[PhysicalLayout] = None,
        send_fn: Optional[Callable] = None,
        capacity: Optional[Union[int, List[int]]] = None,
        tuuid: Optional[Union[int, List[int]]] = None,
        tuuid_to_name: Optional[Dict[int, str]] = None,
    ) -> Tuple[List[Any], Any]:
        """Performs synchronous distribution."""
        if local_data is not None:
            self.plan_and_shard(
                local_data, source, target, capacity, tuuid, tuuid_to_name
            )

        if send_fn is None:
            return [], self._meta_sort_indices

        results = []
        for target_id in sorted(self._dest_agg.keys()):
            groups = self._dest_agg[target_id]
            merged_tuuids = []
            merged_tensors = []
            merged_counts = []
            merged_g_shapes = []
            merged_h_states = []
            merged_c_params = []
            for tuuids, tensors, counts, g_shapes, h_states, c_params in groups:
                if isinstance(tuuids, (list, tuple)):
                    merged_tuuids.extend(tuuids)
                else:
                    merged_tuuids.append(tuuids)
                merged_tensors.extend(tensors)
                merged_counts.extend(counts)
                merged_g_shapes.extend(g_shapes)
                merged_h_states.extend(h_states)
                merged_c_params.extend(c_params)

            res = send_fn(
                target_id,
                merged_tuuids,
                merged_tensors,
                merged_counts,
                merged_g_shapes,
                merged_h_states,
                merged_c_params,
            )

            results.append(res)

        self._dest_agg.clear()
        sort_indices = self._meta_sort_indices
        self._meta_sort_indices = None
        return results, sort_indices


class AsyncDataDistributor(BaseDataDistributor):
    """Asynchronous Data Distributor (using asyncio)."""

    async def distribute(
        self,
        local_data: Optional[Union[TransportTensor, List[TransportTensor]]] = None,
        source: Optional[PhysicalLayout] = None,
        target: Optional[PhysicalLayout] = None,
        send_fn: Optional[Callable] = None,
        capacity: Optional[Union[int, List[int]]] = None,
        tuuid: Optional[Union[int, List[int]]] = None,
        tuuid_to_name: Optional[Dict[int, str]] = None,
    ) -> Tuple[List[Any], Any]:
        """Performs asynchronous distribution."""
        if local_data is not None:
            self.plan_and_shard(
                local_data, source, target, capacity, tuuid, tuuid_to_name
            )

        if send_fn is None:
            return [], self._meta_sort_indices

        tasks = []
        for target_id in sorted(self._dest_agg.keys()):
            groups = self._dest_agg[target_id]
            merged_tuuids = []
            merged_tensors = []
            merged_counts = []
            merged_g_shapes = []
            merged_h_states = []
            merged_c_params = []
            for tuuids, tensors, counts, g_shapes, h_states, c_params in groups:
                if isinstance(tuuids, (list, tuple)):
                    merged_tuuids.extend(tuuids)
                else:
                    merged_tuuids.append(tuuids)
                merged_tensors.extend(tensors)
                merged_counts.extend(counts)
                merged_g_shapes.extend(g_shapes)
                merged_h_states.extend(h_states)
                merged_c_params.extend(c_params)

            tasks.append(
                send_fn(
                    target_id,
                    merged_tuuids,
                    merged_tensors,
                    merged_counts,
                    merged_g_shapes,
                    merged_h_states,
                    merged_c_params,
                )
            )

        res = await asyncio.gather(*tasks) if tasks else []
        self._dest_agg.clear()
        sort_indices = self._meta_sort_indices
        self._meta_sort_indices = None
        return res, sort_indices


# compatibility
DataDistributor = SyncDataDistributor


# ---------------------------------------------------------------------------
# Module-level JIT-compiled zero-copy write kernels.
#
# These are defined at module level so the @compile decorator traces them
# exactly once per unique (count, ndim) combination across all receivers,
# eliminating JAX eager-mode overhead on every receive_slices() call.
#
# JAX constraint: dynamic_slice requires slice_sizes to be statically known
# at trace time, so `count` is a static argnum.  The destination offset is
# passed as a runtime (jnp.int32) so it can vary per sender without
# triggering re-compilation.
# ---------------------------------------------------------------------------


@compile(static_argnums=(2,))
def _jit_write_at_offset(src, dst, count, offset):
    """Write `count` rows of `src` (which may be capacity-padded) into `dst`
    starting at row `offset`.  JIT-compiled; `count` is static."""
    backend = get_backend()
    ndim = len(backend.shape(src))
    start = (0,) * ndim
    sizes = (count,) + backend.shape(src)[1:]
    trimmed = backend.dynamic_slice(src, start, sizes)
    dst_start = (offset,) + (0,) * (ndim - 1)
    return backend.dynamic_slice_update(dst, trimmed, dst_start)


def _get_rank_data_from_tree(rank_id, stitched_tensors_tree, counts, offsets):
    """
    rank_id: target rank id
    stitched_tensors_tree: Pytree sorted by rank
    counts: counts of each rank
    offsets: offsets of each rank
    """
    backend = get_backend()

    start = offsets[rank_id]
    size = counts[rank_id]

    def slice_leaf(array):
        # assume all slaves are in the 0-th dimension (Batch dimension)
        return backend.dynamic_slice_in_dim(array, start=start, size=size, axis=0)

    # 3. 使用 tree_map 一次性处理 List 中的所有 Tensor
    return backend.tree_map(slice_leaf, stitched_tensors_tree)


@compile(static_argnums=(2,))
def _jit_write_at_zero(src, dst, count):
    """Write `count` rows of `src` (which may be capacity-padded) into `dst`
    starting at row 0.  JIT-compiled; `count` is static."""
    backend = get_backend()
    ndim = len(backend.shape(src))
    start = (0,) * ndim
    sizes = (count,) + backend.shape(src)[1:]
    trimmed = backend.dynamic_slice(src, start, sizes)
    return backend.dynamic_slice_update(dst, trimmed, (0,) * ndim)


class TransportReceiver:
    """Configurable receiver for zero-copy or buffered transport scenarios.

    **Modes**:

    Buffered (default):

        Example::
            receiver.register_per_sender_buffers({
                0: [np.zeros((256, 64))],
                1: [np.zeros((128, 64))],
            })

    **Actual written lengths**:
        After receive_slices(), call get_written_counts() to retrieve
        {source_rank: [count_payload_0, count_payload_1, ...]}.
        This is the number of valid rows written (the tensor may arrive
        capacity-padded, so the buffer holds more rows than were written).

    **Completion**:

    One-shot (default):  await receiver.wait()
    Periodic:            async for count in receiver: ...  +  receiver.close()
    """

    def __init__(self, wait_for_n_slices: int = 1, periodic: bool = False):
        from fluxway.common.sync import ArrivalBarrier

        self._barrier = ArrivalBarrier(wait_for_n_slices, periodic=periodic)
        self._periodic = periodic

        # Shared-buffer zero-copy (offset mode)
        self.target_buffers: Optional[List[Any]] = None
        self.expect_offsets: dict = {}

        # Per-sender zero-copy (no-offset mode)
        self._per_sender_buffers: Optional[dict] = None  # {rank: [buf, ...]}

        # Buffered mode
        self._rx_buffers: list = []

        # Actual written row counts: {source_rank: [count_per_payload_index]}
        self._written_counts: dict = {}
        self.backend_name = get_backend().name()

    # ------------------------------------------------------------------
    # Zero-copy registration
    # ------------------------------------------------------------------

    def register_zero_copy_buffers(
        self, target_buffers: List[Any], expect_offsets: dict
    ) -> None:
        """
        Offset mode: all senders write into a single shared buffer set.

        Args:
            target_buffers: List of destination tensors (one per payload index).
            expect_offsets: {source_rank: row_offset_or_list_of_offsets}
        """
        self.target_buffers = target_buffers
        self.expect_offsets = expect_offsets
        self._per_sender_buffers = None
        self._written_counts = {}
        self._barrier.reset()

    def register_per_sender_buffers(self, per_sender_buffers: dict) -> None:
        """
        No-offset mode: each sender has its own pre-allocated buffer list.

        Data is written at row 0; actual written rows are tracked in
        get_written_counts().

        Args:
            per_sender_buffers: {source_rank: [buf0, buf1, ...]}
        """
        self._per_sender_buffers = per_sender_buffers
        self.target_buffers = None
        self.expect_offsets = {}
        self._written_counts = {}

    # ------------------------------------------------------------------
    # Slice reception
    # ------------------------------------------------------------------

    def receive_slices(
        self, source_rank: int, tensors: List[Any], counts: List[int]
    ) -> bool:
        """Called by the network layer when a payload arrives.

        All JAX write operations go through the module-level @compile kernels
        (_jit_write_at_offset / _jit_write_at_zero) to avoid eager execution.
        """
        written = []

        if self._per_sender_buffers is not None:
            # No-offset zero-copy
            sender_bufs = self._per_sender_buffers.get(source_rank)
            if sender_bufs is not None:
                for i, tensor in enumerate(tensors):
                    if tensor is None or i >= len(sender_bufs):
                        written.append(0)
                        continue
                    if i < len(counts):
                        count = int(counts[i])
                    else:
                        raise RuntimeError(
                            f"Missing shard count for bundle element {i} from rank {source_rank}"
                        )
                    if self.backend_name == "torch":
                        sender_bufs[i][:count] = tensor[:count]
                    elif self.backend_name == "jax" and get_backend().is_tensor(tensor):
                        sender_bufs[i] = _jit_write_at_zero(
                            tensor, sender_bufs[i], count
                        )
                    else:
                        sender_bufs[i][:count] = tensor[:count]
                    written.append(count)

        elif self.target_buffers is not None:
            # Offset zero-copy
            if source_rank in self.expect_offsets:
                offsets = self.expect_offsets[source_rank]
                if not isinstance(offsets, (list, tuple)):
                    offsets = [offsets]

                for i, tensor in enumerate(tensors):
                    if tensor is None or i >= len(self.target_buffers):
                        written.append(0)
                        continue
                    count = int(counts[i] if i < len(counts) else counts[0])
                    offset = int(offsets[i] if i < len(offsets) else offsets[0])

                    # Bounds check (host-side, before JIT dispatch)
                    try:
                        buf_rows = np.shape(self.target_buffers[i])[0]
                        if offset + count > buf_rows:
                            raise RuntimeError(
                                f"Zero-copy buffer [{i}] overflow: "
                                f"offset={offset} + count={count} > buf_rows={buf_rows}"
                            )
                    except (TypeError, AttributeError):
                        pass  # opaque buffer, skip check

                    if self.backend_name == "torch":
                        self.target_buffers[i][offset : offset + count] = tensor[:count]
                    elif self.backend_name == "jax" and get_backend().is_tensor(tensor):
                        self.target_buffers[i] = _jit_write_at_offset(
                            tensor, self.target_buffers[i], count, offset
                        )
                    else:
                        self.target_buffers[i][offset : offset + count] = tensor[:count]
                    written.append(count)

        else:
            # Buffered mode: accumulate for later consumption.
            self._rx_buffers.append((tensors, counts))
            written = [int(c) for c in counts]

        if written:
            self._written_counts[source_rank] = written
        self._barrier.arrive()
        return True

    # ------------------------------------------------------------------
    # Completion interface
    # ------------------------------------------------------------------

    async def wait(self) -> None:
        """One-shot mode: block until all expected slices have arrived."""
        await self._barrier.wait()

    async def wait_batch(self) -> int:
        """Periodic mode: block until the next batch fires."""
        return await self._barrier.wait_batch()

    def close(self) -> bool:
        """Periodic mode: flush remainder batch."""
        return self._barrier.close()

    def __aiter__(self):
        return self._barrier.__aiter__()

    def is_done(self) -> bool:
        """Non-blocking: True if the one-shot barrier has fired."""
        return self._barrier.is_done()

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def get_buffered_slices(self) -> list:
        """Consume and return all pending (tensors, counts) tuples."""
        ret = self._rx_buffers
        self._rx_buffers = []
        return ret

    def get_per_sender_buffers(self) -> Optional[dict]:
        """Return the per-sender buffer dict (no-offset zero-copy) or None."""
        return self._per_sender_buffers

    def get_written_counts(self) -> dict:
        """Return {source_rank: [actual_row_count_per_payload_index]}.

        Available after receive_slices() for both zero-copy modes.
        Useful when tensors arrive capacity-padded: only the first
        `count` rows of each buffer are valid data.
        """
        return dict(self._written_counts)


@compile(static_argnums=(2,))
def _stitch_with_prime(prime_rank_tags, slave_tensors, world_size):
    """
    prime_rank_tags: (N,) or scalar - origin rang tags decide where to go
    slave_tensors: pytree (list/dict), with (N, ...extra_dims) tensors
    world_size: scalar
    """
    backend = get_backend()

    # Fast path: If prime_rank_tags is a scalar, everyone goes to one destination
    if (
        isinstance(prime_rank_tags, (int, np.integer))
        or len(backend.shape(prime_rank_tags)) == 0
    ):
        target_rank = int(prime_rank_tags)
        counts = backend.zeros((world_size,), dtype=backend.int32())

        if target_rank >= 0 and target_rank < world_size:
            first_tensor = backend.tree_flatten(slave_tensors)[0][0]
            rows = backend.shape(first_tensor)[0]
            counts = backend.index_update(counts, (target_rank,), rows)

        offsets = backend.zeros((world_size,), dtype=backend.int32())
        return slave_tensors, counts, offsets

    # Sentinel Protocol:
    # Valid tags: 0 <= tag < world_size
    # Invalid tags (Padding): tag == world_size
    # We use minlength=world_size + 1 to naturally collect garbage in the last bin.

    sort_idx = backend.argsort(prime_rank_tags)

    counts_with_garbage = backend.bincount(prime_rank_tags, minlength=world_size + 1)
    counts = counts_with_garbage[:world_size]

    offsets_with_garbage = backend.cumsum(counts_with_garbage) - counts_with_garbage
    offsets = offsets_with_garbage[:world_size]

    def reorder_fn(x):
        return x[sort_idx]

    stitched_tensors = backend.tree_map(reorder_fn, slave_tensors)
    return stitched_tensors, counts, offsets


@compile(static_argnums=(2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
def _jit_plan_and_pack(
    payloads_tuple: Tuple[Any, ...],
    nested_length_list: Tuple[Optional[Any], ...],
    src_local_rank: int,
    src_data_shape: Tuple[int, ...],
    src_routing_indices: Optional[Any],
    src_routing_based_values: Any,
    dst_index_desc: Any,
    dst_routing_dim: Optional[Union[int, Tuple[int, ...]]],
    dst_strategy: Callable,
    dst_mesh: Any,
    dst_partitions: Any,
    dst_replica_axes: Any,
    dst_replica_size: int,
    static_shapes: Tuple[int, ...],
    src_spec: Optional[Any] = None,
    src_index_desc: Optional[Any] = None,
):
    """JIT-compiled kernel: compute routing plan and reorder payloads.

    Returns O(N) flat reordered arrays together with per-device (offsets, counts)
    vectors.  The caller slices each device's window directly from the flat buffer,
    avoiding the O(W*N) vmap-padded intermediate that the previous implementation
    produced.
    """
    backend = get_backend()

    # 1. Compute Plan
    sort_indices, dest_devices, send_counts, target_num_workers, replica_size = (
        TransportPlanner._compute_plan_internal(
            src_local_rank=src_local_rank,
            src_data_shape=src_data_shape,
            src_routing_indices=src_routing_indices,
            src_routing_based_values=src_routing_based_values,
            dst_index_desc=dst_index_desc,
            dst_routing_dim=dst_routing_dim,
            dst_strategy=dst_strategy,
            dst_mesh=dst_mesh,
            dst_partitions=dict(dst_partitions) if dst_partitions else None,
            dst_replica_axes=dst_replica_axes,
            dst_replica_size=dst_replica_size,
            src_spec=src_spec,
            src_index_desc=src_index_desc,
        )
    )

    cs = backend.cumsum(send_counts, dim=0)
    zero = backend.zeros((1,), dtype=backend.int32())
    send_offsets = backend.cat([zero, cs[:-1]], dim=0)

    # 2. Reorder data per payload — O(N) flat reorder, no vmap padding
    flat_buffers_lists = []
    counts_per_payload = []
    offsets_per_payload = []

    for i in range(len(payloads_tuple)):
        data = payloads_tuple[i]
        lengths = nested_length_list[i]
        is_ragged = lengths is not None

        if not is_ragged:
            # --- Dense Path ---
            # A single gather reorders all rows into send order.  No padding needed:
            # the caller extracts each device's slice via [offset:offset+count].
            reordered_data = backend.take(data, sort_indices)
            flat_buffers_lists.append((reordered_data, None))
            counts_per_payload.append(send_counts)
            offsets_per_payload.append(send_offsets)

        else:
            # --- Ragged Path ---
            # flat_cap is a static upper bound on total flat elements (required by
            # repeat_interleave so XLA knows the output size at trace time).
            flat_cap = static_shapes[i + 1]
            item_offsets_orig = backend.cat([zero, backend.cumsum(lengths)[:-1]], dim=0)
            reordered_item_starts = backend.take(item_offsets_orig, sort_indices)
            reordered_item_lengths = backend.take(lengths, sort_indices)

            reordered_flat_offsets = backend.cat(
                [zero, backend.cumsum(reordered_item_lengths)], dim=0
            )

            num_items = backend.shape(reordered_item_lengths)[0]
            item_ids = backend.repeat_interleave(
                backend.arange(0, num_items, dtype=backend.int32()),
                reordered_item_lengths,
                total_repeat_length=flat_cap,
            )

            iota = backend.arange(0, flat_cap, dtype=backend.int32())
            reordered_flat_starts = backend.take(reordered_flat_offsets, item_ids)
            inner_offsets = iota - reordered_flat_starts

            orig_item_starts = backend.take(reordered_item_starts, item_ids)
            flat_indices = orig_item_starts + inner_offsets

            reordered_data = backend.take(data, flat_indices)
            send_flat_offsets = backend.take(reordered_flat_offsets, send_offsets)

            flat_cs = backend.cat([zero, backend.cumsum(reordered_item_lengths)], dim=0)
            send_flat_counts = backend.take(
                flat_cs, send_offsets + send_counts
            ) - backend.take(flat_cs, send_offsets)

            flat_buffers_lists.append((reordered_data, reordered_item_lengths))
            counts_per_payload.append(send_flat_counts)
            offsets_per_payload.append(send_flat_offsets)
            counts_per_payload.append(send_counts)
            offsets_per_payload.append(send_offsets)

    return (
        flat_buffers_lists,
        counts_per_payload,
        offsets_per_payload,
        send_counts,
        send_offsets,
        dest_devices,
        replica_size,
        sort_indices,
    )
