"""Algebraic Sharding — Layout as Function."""

from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

from fluxway.transport.backend import get_backend
from fluxway.transport.mesh import DeviceMesh


# ============================================================
# Layout — Data Distribution Description (Data Distribution Histogram)
# ============================================================


@dataclass
class Layout:
    """Data distribution description supporting dense, ragged, and sparse topologies."""

    partition_dim: int  # along which dimension to partition
    data_shape: Tuple[int, ...]  # Full shape
    world_size: int  # Total mesh group count (deprecated in favor of mesh)
    topology_kind: str = "dense"  # "dense" | "ragged" | "sparse"
    topology_meta: Optional[Any] = None
    dependency: Optional[Any] = None  # Dependency indices/values for dynamic layouts
    is_static: bool = True

    # Mesh Info
    mesh: Optional[DeviceMesh] = None  # DeviceMesh topology
    sharding_axis: Optional[Union[str, Tuple[str, ...]]] = None
    partition_fn: Optional[Callable] = None

    # Legacy compatibility (to be removed)
    replication_axes: Optional[List[str]] = None
    replication_factor: int = 1

    def get_local_indices(self, rank: int) -> Any:
        """Returns the global indices held by this rank."""
        backend = get_backend()

        # Determine effective rank and world size if using mesh
        effective_rank = rank
        effective_world_size = self.world_size

        if self.mesh and self.sharding_axis:
            # Normalize to list
            axes = (
                [self.sharding_axis]
                if isinstance(self.sharding_axis, str)
                else list(self.sharding_axis)
            )
            if all(ax in self.mesh.axis_names for ax in axes):
                try:
                    coords = self.mesh.get_coordinate(rank)
                    sub_coords = []
                    sub_sizes = []
                    for ax in axes:
                        axis_idx = self.mesh.axis_names.index(ax)
                        sub_coords.append(coords[axis_idx])
                        sub_sizes.append(self.mesh.shape[axis_idx])
                    # Flatten multi-axis coordinate to single effective rank
                    import numpy as _np

                    effective_rank = int(_np.ravel_multi_index(sub_coords, sub_sizes))
                    effective_world_size = 1
                    for s in sub_sizes:
                        effective_world_size *= s
                except ValueError:
                    pass

        dim_size = self.data_shape[self.partition_dim]

        # Check standard partition functions
        fn = self.partition_fn
        if fn == chunk_partition:
            # Chunk: [rank * chunk_size, min((rank+1)*chunk_size, dim_size))
            chunk_size = (dim_size + effective_world_size - 1) // effective_world_size
            start = effective_rank * chunk_size
            end = min(start + chunk_size, dim_size)
            if start >= end:
                return backend.arange(0, 0, dtype=backend.int32())
            return backend.arange(start, end, dtype=backend.int32())

        elif fn == cyclic_partition:
            # Cyclic: range(rank, dim_size, world_size)
            return backend.arange(
                effective_rank,
                dim_size,
                step=effective_world_size,
                dtype=backend.int32(),
            )

        # Fallback: Materialize and filter (O(N) cost)
        indices = backend.arange(0, dim_size, dtype=backend.int32())
        # Re-run partition function
        target_ranks = fn(
            indices,
            dim_size,
            effective_world_size,
            topology_meta=self.topology_meta,
            dependency=self.dependency,
        )

        # Filter for current rank
        mask = target_ranks == effective_rank
        nz = backend.nonzero(mask)
        return backend.reshape(nz, (-1,))

    @property
    def is_content_addressable(self) -> bool:
        """True if layout is determined by content (hash/ragged) rather than coordinates."""
        return self.topology_kind in ("ragged", "sparse", "hash")


# ============================================================
# Partition Functions — Composable and Invertible Algebraic Mappings
# ============================================================

# PartitionFn signature:
#   (indices, dim_size, world_size, topology_meta?, dependency?) -> Tensor[int32]
PartitionFn = Callable[..., Any]


def chunk_partition(
    indices: Any,
    dim_size: int,
    world_size: int,
    topology_meta: Any = None,
    dependency: Any = None,
) -> Any:
    """Chunk partitioning: idx → idx // chunk_size"""
    backend = get_backend()
    chunk_size = max((dim_size + world_size - 1) // world_size, 1)
    ranks = indices // chunk_size
    # Clamp to [0, world_size - 1]
    return backend.where(
        ranks >= world_size,
        backend.full(backend.shape(ranks), world_size - 1, dtype=backend.int32()),
        ranks,
    )


def cyclic_partition(
    indices: Any,
    dim_size: int,
    world_size: int,
    topology_meta: Any = None,
    dependency: Any = None,
) -> Any:
    """Cyclic partitioning: idx → idx % world_size"""
    backend = get_backend()
    return backend.mod(indices, world_size)


def hash_partition(
    indices: Any,
    dim_size: int,
    world_size: int,
    topology_meta: Any = None,
    dependency: Any = None,
) -> Any:
    """Hash partitioning with support for custom hash functions.

    Priority: dependency > custom hash_fn > topology splits > default
    """
    backend = get_backend()

    # Priority 1: Use dependency (externally computed indices)
    if dependency is not None:
        values = dependency
    # Priority 2: Use custom hash function from topology_meta
    elif topology_meta is not None and isinstance(topology_meta, dict):
        hash_fn = topology_meta.get("hash_fn")
        if hash_fn is not None:
            values = hash_fn(indices)
        else:
            # Dict without hash_fn, fall back to default
            values = indices
    # Priority 3: Use ragged splits as hash seed
    elif topology_meta is not None:
        if isinstance(topology_meta, list):
            # Ragged splits: use split starts as seed
            starts = topology_meta[0][:-1]
            values = backend.take(starts, indices)
        else:
            # Direct tensor
            starts = topology_meta
            values = backend.take(starts, indices)
    else:
        # Priority 4: Default - hash the indices themselves
        values = indices

    return backend.mod(values, world_size)


def greedy_partition(
    indices: Any,
    dim_size: int,
    world_size: int,
    topology_meta: Any = None,
    dependency: Any = None,
) -> Any:
    """Greedy load balancing partitioning.

    Requires topology_meta to provide sizes (e.g., ragged splits or length arrays).
    """
    backend = get_backend()

    # Extract lengths from topology_meta
    if topology_meta is not None:
        if isinstance(topology_meta, list):
            # Ragged splits: lengths = splits[1:] - splits[:-1]
            splits = topology_meta[0]
            lengths = splits[1:] - splits[:-1]
        else:
            lengths = topology_meta
        # Convert to Python list for iteration
        lengths_list = backend.to_numpy(lengths).tolist()
    else:
        # No size info — fall back to uniform
        lengths_list = [1] * dim_size

    # Greedy algorithm using pure Python (inherently sequential)
    rank_loads = [0] * world_size
    assigned = []

    for length in lengths_list:
        # Find rank with minimum load
        best = min(range(world_size), key=lambda i: rank_loads[i])
        assigned.append(best)
        rank_loads[best] += int(length)

    # print(f"DEBUG: ws={world_size} lengths={lengths_list} loads={rank_loads} assigned={assigned}")

    return backend.tensor(assigned, dtype=backend.int32())


# ============================================================
# ShardingSpec — Algebraic Partitioning Description + User-friendly API
# ============================================================


class ShardingSpec:
    """Algebraic partitioning description. Immutable value object."""

    def __init__(
        self,
        partition_fn: PartitionFn,
        partition_dim: int = 0,
        mesh: Optional[DeviceMesh] = None,
        sharding_axis: Optional[Union[str, Tuple[str, ...]]] = None,
        is_content_addressable: bool = False,
    ):
        self.partition_fn = partition_fn
        self.partition_dim = partition_dim
        self.mesh = mesh
        self.sharding_axis = sharding_axis
        self._is_content_addressable = is_content_addressable

        # Validation
        if self.mesh:
            # Check implicit replication
            pass

    @property
    def is_content_addressable(self) -> bool:
        return self._is_content_addressable

    @property
    def replication_axis(self) -> Optional[List[str]]:
        """Inferred replication axes (Mesh axes NOT used for sharding)."""
        if not self.mesh:
            return None

        all_axes = set(self.mesh.axis_names)

        if self.sharding_axis:
            shard_axes = (
                [self.sharding_axis]
                if isinstance(self.sharding_axis, str)
                else list(self.sharding_axis)
            )
            used_axes = set(shard_axes)
        else:
            used_axes = set()

        # Replicated = All - Used
        repl_axes = list(all_axes - used_axes)
        # Sort for deterministic order
        repl_axes.sort(key=lambda x: self.mesh.axis_names.index(x))
        return repl_axes if repl_axes else None

    # ---- Core ----

    def compute_layout(
        self,
        data_shape: Optional[Tuple[int, ...]],
        world_size: int,
        topology_meta: Any = None,
        dependency: Any = None,
    ) -> Layout:
        """Evaluate partition function → Layout."""

        if data_shape is None:
            if not self.is_content_addressable:
                raise ValueError(
                    "data_shape is required for static sharding (chunk/cyclic)"
                )
            # For content addressable, use dummy shape for partition_fn
            data_shape = (-1,) * (self.partition_dim + 1)

        # Determine effective world size for the partition function
        effective_world_size = world_size
        if self.mesh and self.sharding_axis:
            # Compute effective world size across all sharding axes
            axes = (
                [self.sharding_axis]
                if isinstance(self.sharding_axis, str)
                else list(self.sharding_axis)
            )
            effective_world_size = 1
            for ax in axes:
                if ax in self.mesh.axis_names:
                    idx = self.mesh.axis_names.index(ax)
                    effective_world_size *= self.mesh.shape[idx]

        # Infer topology_kind
        topology_kind = "dense"
        if self.is_content_addressable:
            topology_kind = "hash"

        if topology_meta is not None:
            if isinstance(topology_meta, list):
                topology_kind = "ragged"
            elif isinstance(topology_meta, tuple):
                topology_kind = "sparse"

        # Calculate replication factor for legacy/downstream usage
        rep_factor = 1
        # Calculate replication factor for legacy/downstream usage
        rep_factor = 1
        repl_axes = self.replication_axis
        if self.mesh and repl_axes:
            for ax in repl_axes:
                if ax in self.mesh.axis_names:
                    rep_factor *= self.mesh.shape[self.mesh.axis_names.index(ax)]

        return Layout(
            partition_dim=self.partition_dim,
            data_shape=data_shape,
            world_size=world_size,
            topology_kind=topology_kind,
            topology_meta=topology_meta,
            dependency=dependency,
            is_static=(topology_meta is None and dependency is None),
            mesh=self.mesh,
            sharding_axis=self.sharding_axis,
            partition_fn=self.partition_fn,
            replication_axes=repl_axes,
            replication_factor=rep_factor,  # Computed
        )

    # ---- Replication ----

    def replicated(self, axis: str) -> ShardingSpec:
        """Returns a new spec, replicated on the specified mesh axis."""
        raise NotImplementedError(
            "replicated() deprecated with implicit mesh inference. Manage sharding_axis to control replication."
        )

    # ---- Helper for Implicit Inference ----

    def get_expected_local_indices(
        self, rank: int, local_dim_size: int
    ) -> Optional[Any]:
        """Returns the global indices expected to be held by rank, if deducible (e.g. Chunk).

        Used by DataDistributor for implicit mesh inference when global shape is unknown.
        Assumes standard partitioning (no uneven chunks unless logic handled).
        """
        backend = get_backend()

        # Only support Chunk for now
        if self.partition_fn == chunk_partition:
            # Assume uniform chunks for implicit inference
            effective_rank = rank
            if self.mesh and self.sharding_axis:
                # Map physical rank to logical coordinate on sharding axes
                coords = self.mesh.get_coordinate(rank)
                axes = (
                    [self.sharding_axis]
                    if isinstance(self.sharding_axis, str)
                    else list(self.sharding_axis)
                )
                start_offset = 0
                if all(ax in self.mesh.axis_names for ax in axes):
                    import numpy as _np

                    sub_coords = []
                    sub_sizes = []
                    for ax in axes:
                        axis_idx = self.mesh.axis_names.index(ax)
                        sub_coords.append(coords[axis_idx])
                        sub_sizes.append(self.mesh.shape[axis_idx])
                    effective_rank = int(_np.ravel_multi_index(sub_coords, sub_sizes))
                    start_offset = effective_rank * local_dim_size

                return backend.arange(
                    start_offset, start_offset + local_dim_size, dtype=backend.int32()
                )

        return None

    # ---- User-friendly Factory Methods ----

    @staticmethod
    def chunk(
        dim: int = 0,
        mesh: Optional[DeviceMesh] = None,
        axis: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> ShardingSpec:
        """Chunk partitioning: idx → idx // chunk_size"""
        return ShardingSpec(
            chunk_partition, partition_dim=dim, mesh=mesh, sharding_axis=axis
        )

    @staticmethod
    def cyclic(
        dim: int = 0,
        mesh: Optional[DeviceMesh] = None,
        axis: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> ShardingSpec:
        """Cyclic partitioning: idx → idx % world_size"""
        return ShardingSpec(
            cyclic_partition, partition_dim=dim, mesh=mesh, sharding_axis=axis
        )

    @staticmethod
    def hash(
        dim: int = 0,
        mesh: Optional[DeviceMesh] = None,
        axis: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> ShardingSpec:
        """Hash partitioning (requires topology_meta or dependency)"""
        return ShardingSpec(
            hash_partition,
            partition_dim=dim,
            mesh=mesh,
            sharding_axis=axis,
            is_content_addressable=True,
        )

    @staticmethod
    def greedy(
        dim: int = 0,
        mesh: Optional[DeviceMesh] = None,
        axis: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> ShardingSpec:
        """Greedy load balancing (requires topology_meta providing sizes)"""
        return ShardingSpec(
            greedy_partition, partition_dim=dim, mesh=mesh, sharding_axis=axis
        )

    # ---- Dependency ----

    @staticmethod
    def follow(source_layout: Layout, dim: int = 0) -> ShardingSpec:
        """Follow another Layout.

        Creates a spec that reuses the partition function and mesh of the source layout.
        """
        if source_layout.partition_fn is None:
            raise ValueError("Cannot follow a Layout without partition_fn")

        # Create a proxy function that bakes in the source topology_meta if present
        captured_meta = source_layout.topology_meta
        captured_fn = source_layout.partition_fn

        def _follow_fn(
            indices, dim_size, world_size, topology_meta=None, dependency=None
        ):
            # Use captured meta if original layout had it (e.g. Ragged splits)
            # Otherwise use passed meta (e.g. if just following strategy)
            meta = captured_meta if captured_meta is not None else topology_meta
            return captured_fn(
                indices, dim_size, world_size, topology_meta=meta, dependency=dependency
            )

        return ShardingSpec(
            _follow_fn,
            partition_dim=dim,
            mesh=source_layout.mesh,
            sharding_axis=source_layout.sharding_axis,
        )

    @staticmethod
    def derive_from(
        driver_spec: ShardingSpec,
        driver_data_shape: Tuple[int, ...],
        world_size: int,
        driver_topology: Any = None,
        follow_dim: int = 0,
    ) -> ShardingSpec:
        """One-liner for dependency distribution.

        Computes the layout of the driver tensor and returns a spec that follows it.
        """
        driver_layout = driver_spec.compute_layout(
            driver_data_shape, world_size, topology_meta=driver_topology
        )
        return ShardingSpec.follow(driver_layout, dim=follow_dim)

    def __repr__(self) -> str:
        fn_name = getattr(self.partition_fn, "__name__", str(self.partition_fn))
        parts = [f"fn={fn_name}", f"dim={self.partition_dim}"]
        if self.mesh:
            parts.append(f"mesh={self.mesh.shape}")
        if self.sharding_axis:
            parts.append(f"shard={self.sharding_axis}")
        # Show inferred replication
        if self.mesh and self.replication_axis:
            parts.append(f"repl(implicit)={self.replication_axis}")
        return f"ShardingSpec({', '.join(parts)})"
