"""Transport Planner — Diff(L_src, L_dst) operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable
import logging

import numpy as np

from fluxway.common.backend import get_backend
from fluxway.common.types.sharding import PhysicalLayout

logger = logging.getLogger(__name__)

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

    @property
    def is_replicated(self) -> bool:
        return self.replication_factor > 1

    @property
    def send_offsets(self) -> Any:
        counts = np.array(self.send_counts)
        cs = np.cumsum(counts)
        return np.concatenate([[0], cs[:-1]]).astype(np.int32)

    @property
    def send_counts_tensor(self) -> Any:
        return self.send_counts

    def iter_device_slices(self):
        """Iterate by device: (device_id, offset, count)."""
        import numpy as np

        counts = np.array(self.send_counts)
        offset = 0
        for device_id in range(self.target_num_workers):
            count = int(counts[device_id])
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
        src_local_rank: int,
        src_data_shape: tuple[int, ...],
        src_spec: PhysicalLayout,
        dst_spec: PhysicalLayout,
        src_index_desc: Any | None = None,
        dst_index_desc: Any | None = None,
        src_routing_shard_resolver: Callable[[int, int], Any] | None = None,
        src_routing_indices: Any | None = None,
        src_routing_based_values: Any | None = None,
    ) -> TransportPlan:
        """Diff(L_src, L_dst) using physical layouts."""
        # N-D Native logic extracts the correct dimensionalities implicitly.
        # Default destination dimension 0 is passed to respect returning a legacy single
        # dimension for 1D layout serialization/inspection in tests.
        routing_dim = dst_spec.routing_dim
        if routing_dim is None:
            if dst_spec.partitions:
                routing_dim = next(iter(dst_spec.partitions.keys()))
            else:
                routing_dim = 0

        # Create a partial resolver bound to the data_shape and src_index_desc
        if src_routing_shard_resolver:
            src_shard_resolver = src_routing_shard_resolver
        else:

            def src_shard_resolver(rank, dim):
                return src_spec.get_shard_indices(
                    rank,
                    dim,
                    dim_size=(
                        src_data_shape[dim]
                        if src_data_shape and dim < len(src_data_shape)
                        else 0
                    ),
                    index_desc=src_index_desc,
                )

        # Call internal static method (JIT-friendly)
        sort_indices, dest_devices, send_counts, target_num_workers, replica_size = (
            TransportPlanner._compute_plan_internal(
                src_local_rank=src_local_rank,
                src_data_shape=src_data_shape,
                src_routing_indices=src_routing_indices,
                src_routing_based_values=src_routing_based_values,
                dst_index_desc=dst_index_desc,
                dst_routing_dim=routing_dim,
                dst_strategy=dst_spec.strategy,
                dst_mesh=dst_spec.mesh,
                dst_partitions=dst_spec.partitions,
                dst_replica_axes=dst_spec.replica_axes,
                dst_replica_size=dst_spec.replica_size,
                src_spec=src_spec,
                src_index_desc=src_index_desc,
            )
        )

        return TransportPlan(
            sort_indices=sort_indices,
            dest_devices=dest_devices,
            send_counts=send_counts,
            target_num_workers=target_num_workers,
            replication_factor=replica_size,
        )

    @staticmethod
    def _compute_plan_internal(
        src_local_rank: int,
        src_data_shape: tuple[int, ...],
        src_routing_indices: Any | None,
        src_routing_based_values: Any | None,
        dst_index_desc: Any,
        dst_routing_dim: int | tuple[int, ...] | None,
        dst_strategy: Callable | None,
        dst_mesh: Any | None,
        dst_partitions: dict | None,
        dst_replica_axes: Any | None,
        dst_replica_size: int,
        src_spec: Any | None = None,
        src_index_desc: Any | None = None,
    ) -> tuple[Any, Any, Any, int, int]:
        """Internal static logical core for JIT compilation."""
        backend = get_backend()

        dst_num_workers = dst_mesh.size if dst_mesh else 1
        logger.debug(
            f"[Planner] Compute plan: local_rank={src_local_rank}, src_indices={src_routing_indices}, dst_num_workers={dst_num_workers}, src_data_shape={src_data_shape}"
        )

        dst_is_content_addressable = (
            getattr(dst_strategy, "routing_paradigm", None) is not None
            and dst_strategy.routing_paradigm.name == "DYNAMIC"
        )

        # Step 1: Recover global indices held by this rank
        # Standardize routing_dims to tuple for multi-axis dynamic routing
        # (Where certain dimensions route by value, and others locally by chunk)
        routing_dims = (
            (dst_routing_dim,)
            if isinstance(dst_routing_dim, int)
            else (dst_routing_dim or (0,))
        )
        primary_dim = routing_dims[0]

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

        # The planner operates on the logical global domain.
        # src_data_shape is provided by the distributor as the global shape.
        global_shape = src_data_shape
        ndim = len(global_shape)

        mesh_coords: list[Any | None] = [None] * len(dst_mesh.shape) if dst_mesh else []
        combined_valid_mask = backend.ones((1,), dtype=backend.int32())

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

            if is_dynamic_dim and src_routing_based_values is not None:
                input_for_routing = src_routing_based_values
            elif d == primary_dim and src_routing_indices is not None:
                input_for_routing = src_routing_indices
            else:
                if src_spec is not None:
                    input_for_routing = src_spec.get_shard_indices(
                        src_local_rank,
                        d,
                        dim_size=global_shape[d],
                        index_desc=src_index_desc,
                    )
                else:
                    # Fallback for anonymous shards (1:1 mapping)
                    input_for_routing = backend.arange(0, src_data_shape[d])

            # Identify sub-mesh assigned to this tensor dimension
            axes = dst_partitions[d]
            axis_indices = [dst_mesh.axis_names.index(ax) for ax in axes]
            sub_mesh_shape = tuple(dst_mesh.shape[idx] for idx in axis_indices)
            sub_num_workers = 1
            for s in sub_mesh_shape:
                sub_num_workers *= s

            logical_ranks_d = strategy(
                input_for_routing,
                dim_size=global_shape[d],
                num_workers=sub_num_workers,
                index_desc=dst_index_desc if is_dynamic_dim else None,
            )

            # Compute valid mask for this dimension and broadcast it
            is_valid_d = backend.cast(input_for_routing >= 0, backend.int32())
            if len(dst_partitions) == 1:
                combined_valid_mask = is_valid_d
            else:
                new_shape = [1] * ndim
                new_shape[d] = -1
                combined_valid_mask = combined_valid_mask * backend.reshape(
                    is_valid_d, tuple(new_shape)
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
        # Filter out sentinels (-1) from padded JIT operations
        valid_mask_flat = backend.reshape(combined_valid_mask, (-1,))

        # We ensure primary_dest_flat is safe for bincount (no -1)
        # and use weights to count only valid entries.
        safe_primary_dest = backend.where(valid_mask_flat > 0, primary_dest_flat, 0)
        send_counts = _bincount(
            backend,
            safe_primary_dest,
            weights=backend.cast(valid_mask_flat, backend.float32()),
            minlength=dst_num_workers,
        )
        send_counts = backend.cast(send_counts, backend.int32())

        return (sort_indices, dest_devices, send_counts, dst_num_workers, num_replicas)

    @staticmethod
    def plan(
        src_local_rank: int,
        src_spec: PhysicalLayout,
        dst_spec: PhysicalLayout,
        src_data_shape: tuple[int, ...],
        src_index_desc: Any = None,
        dst_index_desc: Any = None,
    ) -> TransportPlan:
        """High-level Diff API."""
        return TransportPlanner.compute_plan(
            src_local_rank,
            src_data_shape,
            src_spec,
            dst_spec,
            src_index_desc=src_index_desc,
            dst_index_desc=dst_index_desc,
        )


# ============================================================
# Internal helpers
# ============================================================


def _bincount(
    backend, tensor: Any, weights: Any | None = None, minlength: int = 0
) -> Any:
    """Cross-backend bincount."""
    return backend.bincount(tensor, weights=weights, minlength=minlength)
