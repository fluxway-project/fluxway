"""Transport Planner — Diff(L_src, L_dst) operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

from fluxway.common.backend import get_backend
from fluxway.common.types.sharding import ShardingSpec, Layout


# ============================================================
# TransportPlan — Result of Diff(L_src, L_dst)
# ============================================================


@dataclass
class TransportPlan:
    """Transport plan: pure tensor description."""

    sort_indices: Any
    dest_devices: Any
    send_counts: Any
    target_num_workers: int
    replication_factor: int = 1
    spatial_indices: Optional[Any] = None

    @property
    def is_replicated(self) -> bool:
        return self.replication_factor > 1

    @property
    def send_offsets(self) -> Any:
        backend = get_backend()
        cs = backend.cumsum(self.send_counts, dim=0)
        zero = backend.zeros((1,), dtype=backend.int32())
        return backend.cat([zero, cs[:-1]], dim=0)

    @property
    def send_counts_tensor(self) -> Any:
        return self.send_counts

    def iter_device_slices(self):
        """Iterate by device: (device_id, offset, count)."""
        backend = get_backend()
        offset = 0
        for device_id in range(self.target_num_workers):
            count = int(backend.item(self.send_counts[device_id]))
            if count > 0:
                yield device_id, offset, count
            offset += count


# ============================================================
# TransportPlanner — Core Diff Operator
# ============================================================


class TransportPlanner:
    """Computes the permutation from source layout to target layout."""

    _PLAN_CACHE = {}

    @classmethod
    def clear_cache(cls):
        cls._PLAN_CACHE.clear()

    @staticmethod
    def compute_plan(
        src_layout: Layout,
        dst_layout: Layout,
        local_rank: int,
        src_indices: Optional[Any] = None,
        data_values: Optional[Any] = None,
    ) -> TransportPlan:
        """Diff(L_src, L_dst) using pre-computed Layouts."""
        # N-D Native logic extracts the correct dimensionalities implicitly.
        # Default destination dimension 0 is passed to respect returning a legacy single
        # dimension for 1D layout serialization/inspection in tests.
        routing_dim = dst_layout.routing_dim
        if routing_dim is None:
            if dst_layout.partitions:
                routing_dim = next(iter(dst_layout.partitions.keys()))
            else:
                routing_dim = 0

        # Call internal static method (JIT-friendly)
        (
            sort_indices,
            dest_devices,
            send_counts,
            spatial_indices,
            target_num_workers,
            replica_size,
        ) = TransportPlanner._compute_plan_internal(
            src_indices=src_indices,
            data_values=data_values,
            dst_index_desc=dst_layout.index_desc,
            dst_routing_dim=routing_dim,
            dst_data_shape=dst_layout.data_shape,
            dst_strategy=dst_layout.strategy,
            dst_mesh=dst_layout.mesh,
            dst_partitions=dst_layout.partitions,
            src_shard_resolver=src_layout.get_shard_indices,
            local_rank=local_rank,
        )

        return TransportPlan(
            sort_indices=sort_indices,
            dest_devices=dest_devices,
            send_counts=send_counts,
            target_num_workers=target_num_workers,
            replication_factor=replica_size,
            spatial_indices=spatial_indices,
        )

    @staticmethod
    def _compute_plan_internal(
        src_indices: Optional[Any],
        data_values: Optional[Any],
        dst_index_desc: Any,
        dst_routing_dim: Optional[Union[int, Tuple[int, ...]]],
        dst_data_shape: Tuple[int, ...],
        dst_strategy: Optional[Callable],
        dst_mesh: Optional[Any],
        dst_partitions: Optional[dict],
        src_shard_resolver: Callable[[int, int], Any],
        local_rank: int,
    ) -> Tuple[Any, Any, Any, Any, int, int]:
        """Internal static logical core for JIT compilation."""
        backend = get_backend()

        dst_num_workers = dst_mesh.size if dst_mesh else 1
        print(
            f"[Planner] Compute plan: local_rank={local_rank}, src_indices={src_indices}, dst_num_workers={dst_num_workers}, dst_data_shape={dst_data_shape}",
            flush=True,
        )

        dst_is_content_addressable = (
            getattr(dst_strategy, "routing_paradigm", None) is not None
            and dst_strategy.routing_paradigm.name == "DYNAMIC"
        )

        dst_replica_axes = None
        dst_replica_size = 1
        if dst_mesh and dst_partitions is not None:
            used = set()
            for axes in dst_partitions.values():
                used.update(axes)
            all_axes = set(dst_mesh.axis_names)
            repl = sorted(all_axes - used, key=lambda x: dst_mesh.axis_names.index(x))
            if repl:
                dst_replica_axes = repl
                for ax in repl:
                    dst_replica_size *= dst_mesh.shape[dst_mesh.axis_names.index(ax)]

        # Step 1: Recover global indices held by this rank
        # Standardize routing_dims to tuple for multi-axis dynamic routing
        # (Where certain dimensions route by value, and others locally by chunk)
        routing_dims = (
            (dst_routing_dim,)
            if isinstance(dst_routing_dim, int)
            else (dst_routing_dim or (0,))
        )

        # Fallback for explicit src_indices overriding the primary dimension.
        primary_dim = routing_dims[0] if routing_dims else 0
        if src_indices is not None:
            primary_src_spatial_indices = src_indices
        else:
            primary_src_spatial_indices = src_shard_resolver(local_rank, primary_dim)

        # =====================================================================
        # [ N-D Cross-Mesh Routing & Cartesian Broadcasting ]
        #
        # This section determines which physical device (rank) each piece of data
        # should be sent to. Instead of writing slow Python `for` loops to calculate
        # this for every item, we use mathematical "Cartesian Products" via JAX's
        # Array Broadcasting. This makes the compilation O(1) complexity.
        #
        # Let's look at a beginner-friendly example:
        # Imagine a 2D DeviceMesh with shape [2, 4] (axis names: 'x', 'y').
        # Our data tensor has 2 dimensions: [Batch, Sequence].
        #
        # Dim 0 (Batch): We want to route this dynamically (e.g., Hash routing based on values).
        #   -> The Strategy calculates destination indices for the 'x' axis: [N_items]
        # Dim 1 (Sequence): We want to route this statically (e.g., Chunk slicing).
        #   -> The Strategy calculates destination indices for the 'y' axis: [M_items]
        #
        # How JAX Broadcasting (Cartesian Product) works here:
        # 1. We reshape the Dim 0 result from [N] to [N, 1].
        # 2. We reshape the Dim 1 result from [M] to [1, M].
        # 3. We let JAX combine them! -> [N, M].
        # Every pair (n, m) now perfectly describes the 2D coordinate on the Mesh.
        # =====================================================================

        ndim = len(dst_data_shape)
        mesh_coords = [None] * len(dst_mesh.shape) if dst_mesh else []

        from fluxway.common.types.sharding import ChunkPartition

        _CHUNK = ChunkPartition()

        # Step 2: Compute target devices across all topological constraints (Cartesian Mapping)
        for d in range(ndim):
            if not dst_mesh or not dst_partitions or d not in dst_partitions:
                continue

            is_dynamic_dim = dst_is_content_addressable and (d in routing_dims)
            base_strategy = dst_strategy if dst_strategy is not None else _CHUNK
            strategy = (
                base_strategy
                if is_dynamic_dim
                else (base_strategy if not dst_is_content_addressable else _CHUNK)
            )

            if d == primary_dim and src_indices is not None:
                indices_for_routing = src_indices
            elif is_dynamic_dim and data_values is not None:
                indices_for_routing = data_values
            else:
                indices_for_routing = src_shard_resolver(local_rank, d)

            # Identify sub-mesh assigned to this tensor dimension
            axes = dst_partitions[d]
            axis_indices = [dst_mesh.axis_names.index(ax) for ax in axes]
            sub_mesh_shape = tuple(dst_mesh.shape[idx] for idx in axis_indices)
            sub_num_workers = 1
            for s in sub_mesh_shape:
                sub_num_workers *= s

            logical_ranks_d = strategy(
                indices_for_routing,
                dim_size=dst_data_shape[d],
                num_workers=sub_num_workers,
                index_desc=dst_index_desc if is_dynamic_dim else None,
            )

            unraveled = backend.unravel_index(logical_ranks_d, sub_mesh_shape)
            if not isinstance(unraveled, tuple):
                unraveled = (unraveled,)

            for ax_idx, coord_tensor in zip(axis_indices, unraveled, strict=False):  # type: ignore
                # When a tensor has multiple dimensions (ndim > 1) and we partition different dimensions onto different mesh axes,
                # we need to combine these 1D coordinate transformations into a single N-D mesh coordinate system.
                # JAX Array Broadcasting achieves this: if we reshape the coord_tensor for dimension `d` to have size `N` along
                # dimension `d` and size `1` everywhere else, JAX will automatically broadcast (Cartesian product) them together later.
                if len(dst_partitions) == 1:
                    mesh_coords[ax_idx] = coord_tensor
                else:
                    if len(backend.shape(coord_tensor)) == 1:
                        new_shape = [1] * ndim
                        new_shape[d] = -1
                        mesh_coords[ax_idx] = backend.reshape(
                            coord_tensor, tuple(new_shape)
                        )
                    else:
                        mesh_coords[ax_idx] = coord_tensor

        for i in range(len(mesh_coords)):
            if mesh_coords[i] is None:
                mesh_coords[i] = backend.zeros((1,), dtype=backend.int32())

        if dst_mesh:
            primary_dest = dst_mesh.coordinate_to_device(tuple(mesh_coords))
        else:
            primary_dest = backend.zeros((1,), dtype=backend.int32())

        dest_devices = primary_dest

        # Step 3: Handle Replication (Cartesian Product / Broadcast)
        #
        # If the Layout requires replicas (e.g., repeating the same data across multiple devices),
        # we use the exact same Cartesian broadcasting trick!
        #
        # Example:
        # Imagine we have 1 Device computed above, but need 2 replicas across the 'z' axis.
        # We take our existing coordinate array [N, M] and expand it to [N, M, 1].
        # We generate a replica index array [0, 1] and reshape it to [1, 1, 2].
        # JAX broadcasts them together, instantly generating actual device coordinates
        # formatted as [N, M, 2]!
        #
        num_replicas = dst_replica_size

        if dst_replica_axes and dst_mesh:
            mesh = dst_mesh
            primary_coords = mesh.unravel_index(dest_devices)

            expanded_coords = [backend.reshape(c, (-1, 1)) for c in primary_coords]

            rep_axis_idx = [mesh.axis_names.index(ax) for ax in dst_replica_axes]
            rep_shape = [mesh.shape[idx] for idx in rep_axis_idx]

            rep_flat_indices = backend.arange(0, num_replicas, dtype=backend.int32())

            if len(dst_replica_axes) == 1:
                expanded_coords[rep_axis_idx[0]] = backend.reshape(
                    rep_flat_indices, (1, num_replicas)
                )
            else:
                rep_unraveled = backend.unravel_index(
                    rep_flat_indices, tuple(rep_shape)
                )
                if not isinstance(rep_unraveled, tuple):
                    rep_unraveled = (rep_unraveled,)
                for ax_id, r_coord in zip(rep_axis_idx, rep_unraveled, strict=False):  # type: ignore
                    expanded_coords[ax_id] = backend.reshape(r_coord, (1, num_replicas))

            dest_devices = mesh.coordinate_to_device(tuple(expanded_coords))
        else:
            dest_devices = backend.reshape(dest_devices, (-1, 1))

        # Step 4: Sort by target device
        primary_dest_flat = (
            backend.reshape(dest_devices, (-1,))
            if num_replicas == 1
            else backend.reshape(backend.select(dest_devices, dim=1, index=0), (-1,))
        )

        sort_indices = backend.argsort(primary_dest_flat, dim=0, stable=True)

        # Step 5: Compute send counts
        # Generate the count of elements destined for each device using bincount.
        # We use the physical target world size to ensure the counts array is correctly sized.

        send_counts = _bincount(backend, primary_dest_flat, minlength=dst_num_workers)

        return (
            sort_indices,
            dest_devices,
            send_counts,
            primary_src_spatial_indices,
            dst_num_workers,
            num_replicas,
        )

    @staticmethod
    def plan(
        src_spec: ShardingSpec,
        dst_spec: ShardingSpec,
        data_shape: Tuple[int, ...],
        local_rank: int,
        index_desc: Any = None,
    ) -> TransportPlan:
        """High-level Diff API."""
        src_layout = src_spec._to_layout(data_shape, index_desc)
        dst_layout = dst_spec._to_layout(data_shape, index_desc)
        return TransportPlanner.compute_plan(src_layout, dst_layout, local_rank)


# ============================================================
# Internal helpers
# ============================================================


def _bincount(backend, tensor: Any, minlength: int = 0) -> Any:
    """Cross-backend bincount."""
    return backend.bincount(tensor, minlength=minlength)
