"""Transport Planner — Diff(L_src, L_dst) operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

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
        backend = get_backend()

        # Step 1: Recover global indices held by this rank
        if src_indices is not None:
            src_global_indices = src_indices
        else:
            src_global_indices = src_layout.get_local_indices(local_rank)

        # Step 2: Compute target devices
        ews = dst_layout.effective_world_size

        if dst_layout.is_content_addressable and data_values is not None:
            if dst_layout.strategy is None:
                raise ValueError("Content-addressable layout must have strategy")

            indices_for_routing = data_values
            if src_indices is not None:
                indices_for_routing = src_indices
            if indices_for_routing is None:
                raise ValueError("Content routing requires indices or values")

            primary_dest = dst_layout.strategy(
                indices_for_routing,
                dim_size=backend.shape(indices_for_routing)[dst_layout.dim],
                world_size=ews,
                topology=dst_layout.topology,
            )
            dest_devices = dst_layout.logical_to_physical(primary_dest)

        else:
            dim_size = dst_layout.data_shape[dst_layout.dim]
            primary_dest = dst_layout.strategy(
                src_global_indices,
                dim_size=dim_size,
                world_size=ews,
                topology=dst_layout.topology,
            )
            dest_devices = dst_layout.logical_to_physical(primary_dest)

        # Step 3: Handle Replication
        num_replicas = dst_layout.replication_factor

        if dst_layout.replication_axes and dst_layout.mesh:
            mesh = dst_layout.mesh
            primary_coords = mesh.unravel_index(dest_devices)

            rep_axis = dst_layout.replication_axes[0]
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
        primary_dest = (
            dest_devices
            if num_replicas == 1
            else backend.select(dest_devices, dim=1, index=0)
        )
        primary_dest_flat = backend.reshape(primary_dest, (-1,))
        sort_indices = backend.argsort(primary_dest_flat, dim=0, stable=True)

        # Step 5: Send counts
        send_counts = _bincount(
            backend, primary_dest_flat, minlength=dst_layout.world_size
        )

        return TransportPlan(
            sort_indices=sort_indices,
            dest_devices=dest_devices,
            send_counts=send_counts,
            dim=dst_layout.dim,
            target_world_size=dst_layout.world_size,
            replication_factor=num_replicas,
            global_indices=src_global_indices,
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
