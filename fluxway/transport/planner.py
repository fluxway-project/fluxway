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

    sort_indices: Any  # Tensor[int32] shape=(local_dim_size,) sorting indices
    dest_devices: (
        Any  # Tensor[int32] shape=(local_dim_size, num_replicas) target mesh group
    )
    send_counts: (
        Any  # Tensor[int32] shape=(target_world_size,) send volume per PRIMARY device
    )
    partition_dim: int  # along which dimension to partition
    target_world_size: int
    replication_factor: int = 1  # >1 = fan-out
    global_indices: Optional[Any] = None  # routing metadata

    @property
    def is_replicated(self) -> bool:
        return self.replication_factor > 1

    @property
    def send_offsets(self) -> Any:
        """Returns tensor of send offsets (exclusive scan of send_counts)."""
        backend = get_backend()
        # cumsum of counts.
        # shifts by 1 to get starts.
        # [c0, c1, c2] -> [c0, c0+c1, ...]
        # starts: [0, c0, c0+c1]
        cs = backend.cumsum(self.send_counts, dim=0)
        zero = backend.zeros((1,), dtype=backend.int32())
        # prepend 0, drop last
        return backend.cat([zero, cs[:-1]], dim=0)

    @property
    def send_counts_tensor(self) -> Any:
        """Returns tensor of send counts. Alias for self.send_counts."""
        return self.send_counts

    def iter_device_slices(self):
        """Iterate by device (device_id, offset, count).

        Used for scatter-send in the communication layer.
        """
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

        # --- Step 1: Recover global indices held by this rank ---
        if src_indices is not None:
            src_global_indices = src_indices
        else:
            # Recover global indices held by this rank
            src_global_indices = src_layout.get_local_indices(local_rank)

        # --- Step 2: Compute target devices ---

        if dst_layout.is_content_addressable and data_values is not None:
            # Content-based routing (Hash/Ragged Dispatch)
            # Use partition_fn provided by Layout to route values
            if dst_layout.partition_fn is None:
                raise ValueError("Content-addressable layout must have partition_fn")

            # Calculate effective world size
            effective_world_size = dst_layout.world_size
            if dst_layout.mesh and dst_layout.sharding_axis:
                axis_name = (
                    dst_layout.sharding_axis
                    if isinstance(dst_layout.sharding_axis, str)
                    else dst_layout.sharding_axis[0]
                )
                if axis_name in dst_layout.mesh.axis_names:
                    idx = dst_layout.mesh.axis_names.index(axis_name)
                    effective_world_size = dst_layout.mesh.shape[idx]

            # Select routing source (indices vs values)
            indices_for_routing = data_values
            if src_indices is not None:
                indices_for_routing = src_indices

            if indices_for_routing is None:
                raise ValueError("Content routing requires indices or values")

            # Route
            primary_dest = dst_layout.partition_fn(
                indices_for_routing,
                dim_size=backend.shape(indices_for_routing)[dst_layout.partition_dim],
                world_size=effective_world_size,
                topology_meta=dst_layout.topology_meta,
            )

            # Map logical ranks to physical ranks if using mesh
            if dst_layout.mesh and dst_layout.sharding_axis:
                coords = [
                    backend.zeros(backend.shape(primary_dest), dtype=backend.int32())
                    for _ in dst_layout.mesh.shape
                ]
                axis_idx = dst_layout.mesh.axis_names.index(axis_name)
                coords[axis_idx] = primary_dest

                # Map using mesh device IDs (vectorized lookup)
                mesh_device_ids = backend.tensor(
                    dst_layout.mesh.device_ids, dtype=backend.int32()
                )
                # device_ids[c1, c2, ...]
                dest_devices = mesh_device_ids[tuple(coords)]
            else:
                dest_devices = primary_dest

        else:
            # Layout-based routing (Coordinate dispatch)
            # We have src_global_indices. We want to know where they go in DST.

            # 1. Compute Destination Logical Coordinates
            effective_world_size = dst_layout.world_size
            axis_name = None
            if dst_layout.mesh and dst_layout.sharding_axis:
                axis_name = (
                    dst_layout.sharding_axis
                    if isinstance(dst_layout.sharding_axis, str)
                    else dst_layout.sharding_axis[0]
                )
                if axis_name in dst_layout.mesh.axis_names:
                    idx = dst_layout.mesh.axis_names.index(axis_name)
                    effective_world_size = dst_layout.mesh.shape[idx]

            dim_size = dst_layout.data_shape[dst_layout.partition_dim]

            # Call partition_fn on INDICES
            primary_dest = dst_layout.partition_fn(
                src_global_indices,
                dim_size=dim_size,
                world_size=effective_world_size,
                topology_meta=dst_layout.topology_meta,
            )

            # 2. Map to Physical Devices
            if dst_layout.mesh and dst_layout.sharding_axis:
                coords = [
                    backend.zeros(backend.shape(primary_dest), dtype=backend.int32())
                    for _ in dst_layout.mesh.shape
                ]
                axis_idx = dst_layout.mesh.axis_names.index(axis_name)
                coords[axis_idx] = primary_dest

                mesh_device_ids = backend.tensor(
                    dst_layout.mesh.device_ids, dtype=backend.int32()
                )
                # Use vectorized indexing
                dest_devices = mesh_device_ids[tuple(coords)]
            else:
                dest_devices = primary_dest

        # --- Handle Replication (Expansion) ---
        num_replicas = dst_layout.replication_factor

        if dst_layout.replication_axes and dst_layout.mesh:
            # Expand primary destination devices to include all replicas
            mesh = dst_layout.mesh
            primary_coords = mesh.unravel_index(dest_devices)

            # TODO: Support multi-axis replication
            rep_axes = dst_layout.replication_axes
            rep_axis = rep_axes[0]
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
            # No replication: reshape to (N, 1)
            dest_devices = backend.reshape(dest_devices, (-1, 1))

        # --- Step 3: Sort by target device ---
        primary_dest = (
            dest_devices
            if num_replicas == 1
            else backend.select(dest_devices, dim=1, index=0)
        )

        primary_dest_flat = backend.reshape(primary_dest, (-1,))
        sort_indices = backend.argsort(primary_dest_flat, dim=0, stable=True)

        # --- Step 4: Send volume ---
        # Need world size for bincount minlength.
        # Layout stores world_size
        send_counts = _bincount(
            backend, primary_dest_flat, minlength=dst_layout.world_size
        )

        return TransportPlan(
            sort_indices=sort_indices,
            dest_devices=dest_devices,
            send_counts=send_counts,
            partition_dim=dst_layout.partition_dim,
            target_world_size=dst_layout.world_size,
            replication_factor=num_replicas,
            global_indices=src_global_indices,  # Always include for offset tracking
        )

    @staticmethod
    def plan(
        src_spec: ShardingSpec,
        dst_spec: ShardingSpec,
        data_shape: Tuple[int, ...],
        local_rank: int,
        src_world_size: int,
        dst_world_size: int,
        topology_meta: Any = None,
        dependency: Any = None,
    ) -> TransportPlan:
        """Legacy Diff API."""
        # Compute Global Layouts
        src_layout = src_spec.compute_layout(
            data_shape, src_world_size, topology_meta, dependency
        )
        dst_layout = dst_spec.compute_layout(
            data_shape, dst_world_size, topology_meta, dependency
        )

        return TransportPlanner.compute_plan(src_layout, dst_layout, local_rank)


# ============================================================
# Internal helpers
# ============================================================


def _where_true(backend, mask: Any) -> Any:
    """Returns indices where mask is True (1D).

    Uses backend boolean indexing (supported by both PyTorch and JAX).
    """
    n = backend.shape(mask)[0]
    # Returns (N, 1) usually for 1D mask. flatten to 1D.
    nz = backend.nonzero(mask)
    if backend.shape(nz)[1] > 0:
        return backend.reshape(nz, (-1,))
    return backend.reshape(nz, (0,))


def _bincount(backend, tensor: Any, minlength: int = 0) -> Any:
    """Cross-backend bincount using backend primitive."""
    return backend.bincount(tensor, minlength=minlength)
