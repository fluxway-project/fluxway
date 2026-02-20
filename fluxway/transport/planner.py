"""Transport Planner — Diff(L_src, L_dst) operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from fluxway.transport.backend import get_backend
from fluxway.transport.sharding import ShardingSpec, Layout


# ============================================================
# TransportPlan — Result of Diff(L_src, L_dst)
# ============================================================


@dataclass
class TransportPlan:
    """Transport plan: pure tensor description."""

    sort_indices: Any
    dest_devices: Any
    send_counts: Any
    dim: int
    target_world_size: int
    replication_factor: int = 1
    global_indices: Optional[Any] = None

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
        for device_id in range(self.target_world_size):
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
        # Call internal static method (JIT-friendly)
        (
            sort_indices,
            dest_devices,
            send_counts,
            global_indices,
            dim,
            target_world_size,
            replication_factor,
        ) = TransportPlanner._compute_plan_internal(
            src_indices=src_indices,
            data_values=data_values,
            dst_topology=dst_layout.topology,
            dst_dim=dst_layout.dim,
            dst_data_shape=dst_layout.data_shape,
            dst_world_size=dst_layout.world_size,
            dst_is_content_addressable=dst_layout.is_content_addressable,
            dst_strategy=dst_layout.strategy,
            dst_replication_factor=dst_layout.replication_factor,
            dst_replication_axes=dst_layout.replication_axes,
            dst_mesh=dst_layout.mesh,
            src_layout_local_indices_fn=src_layout.get_local_indices,
            local_rank=local_rank,
        )

        return TransportPlan(
            sort_indices=sort_indices,
            dest_devices=dest_devices,
            send_counts=send_counts,
            dim=dim,
            target_world_size=target_world_size,
            replication_factor=replication_factor,
            global_indices=global_indices,
        )

    @staticmethod
    def _compute_plan_internal(
        src_indices: Optional[Any],
        data_values: Optional[Any],
        dst_topology: Any,
        dst_dim: int,
        dst_data_shape: Tuple[int, ...],
        dst_world_size: int,
        dst_is_content_addressable: bool,
        dst_strategy: Optional[Callable],
        dst_replication_factor: int,
        dst_replication_axes: Optional[Tuple[str, ...]],
        dst_mesh: Optional[Any],
        src_layout_local_indices_fn: Callable[[int], Any],
        local_rank: int,
    ) -> Tuple[Any, Any, Any, Any, int, int, int]:
        """Internal static logical core for JIT compilation."""
        backend = get_backend()

        # Step 1: Recover global indices held by this rank
        if src_indices is not None:
            src_global_indices = src_indices
        else:
            # Note: calling src_layout.get_local_indices inside JIT might be tricky
            # if it's not a Pytree or simple function.
            # Assuming get_local_indices returns a tensor or we passed it in.
            # If src_layout is complex, this callback might fail JIT.
            # Ideally src_indices should be passed pre-computed if dynamic.
            src_global_indices = src_layout_local_indices_fn(local_rank)

        # Step 2: Compute target devices

        if dst_is_content_addressable and data_values is not None:
            if dst_strategy is None:
                raise ValueError("Content-addressable layout must have strategy")

            indices_for_routing = data_values
            if src_indices is not None:
                indices_for_routing = src_indices
            if indices_for_routing is None:
                raise ValueError("Content routing requires indices or values")

            primary_dest = dst_strategy(
                indices_for_routing,
                dim_size=backend.shape(indices_for_routing)[dst_dim],
                world_size=dst_world_size,
                topology=dst_topology,
            )

            # Note: logical_to_physical mapping is assumed to be identity or handled by strategy
            # for content-addressable layouts in the current JIT-compiled path.
            dest_devices = primary_dest

        else:
            dim_size = dst_data_shape[dst_dim]
            primary_dest = dst_strategy(
                src_global_indices,
                dim_size=dim_size,
                world_size=dst_world_size,
                topology=dst_topology,
            )
            dest_devices = primary_dest

        # Note: We assume `dest_devices` contains physical rank IDs.
        # If a mesh with non-identity logical-to-physical mapping is used,
        # the strategy or a pre-processing step must handle the translation
        # before this point, as `Layout` objects are not fully JIT-traced yet.

        # Step 3: Handle Replication
        num_replicas = dst_replication_factor

        if dst_replication_axes and dst_mesh:
            mesh = dst_mesh
            primary_coords = mesh.unravel_index(dest_devices)

            rep_axis = dst_replication_axes[0]
            rep_axis_idx = mesh.axis_names.index(rep_axis)
            rep_size = mesh.shape[rep_axis_idx]

            replica_cols = []
            for r in range(rep_size):
                current_coords = list(primary_coords)
                current_coords[rep_axis_idx] = backend.full(
                    backend.shape(dest_devices), r, dtype=backend.int32()
                )
                col_ids = mesh.coordinate_to_device(tuple(current_coords))
                replica_cols.append(col_ids)

            dest_devices = backend.stack(replica_cols, dim=1)
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

        send_counts = _bincount(backend, primary_dest_flat, minlength=dst_world_size)

        return (
            sort_indices,
            dest_devices,
            send_counts,
            src_global_indices,
            dst_dim,
            dst_world_size,
            num_replicas,
        )

    @staticmethod
    def plan(
        src_spec: ShardingSpec,
        dst_spec: ShardingSpec,
        data_shape: Tuple[int, ...],
        local_rank: int,
        topology: Any = None,
        routing_indices: Any = None,
    ) -> TransportPlan:
        """High-level Diff API."""
        src_layout = src_spec._to_layout(data_shape, topology, routing_indices)
        dst_layout = dst_spec._to_layout(data_shape, topology, routing_indices)
        return TransportPlanner.compute_plan(src_layout, dst_layout, local_rank)


# ============================================================
# Internal helpers
# ============================================================


def _bincount(backend, tensor: Any, minlength: int = 0) -> Any:
    """Cross-backend bincount."""
    return backend.bincount(tensor, minlength=minlength)
