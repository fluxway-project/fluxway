"""Algebraic Sharding — Layout as Function."""

from __future__ import annotations

import enum
import logging
from typing import Any, List, Optional, Tuple, Union, Dict
from pydantic import Field, model_validator

from fluxway.common.backend import get_backend
from fluxway.common.mesh import DeviceMesh
from fluxway.common.index_descriptor import (
    IndexDescriptor,
    ContinuousIndex,
    ExplicitIndex,
    RaggedIndex,
    FunctionalIndex,
)
from fluxway.common.types.configs import FluxwayBaseSchema, ConfigDict

logger = logging.getLogger(__name__)

# ============================================================
# Enums — Routing Semantics
# ============================================================


class RoutingParadigm(str, enum.Enum):
    """Semantics of routing: static (rank-based) or dynamic (content-based)."""

    STATIC = "static"
    DYNAMIC = "dynamic"


# ============================================================
# PartitionSpec — JAX-style Partition Description
# ============================================================


class PartitionSpec(FluxwayBaseSchema):
    """Maps tensor dimensions to device mesh axes by position."""

    model_config = ConfigDict(frozen=True)

    specs: Tuple[Optional[Union[str, Tuple[str, ...]]], ...] = Field(
        default_factory=tuple
    )

    def __init__(self, *specs: Optional[Union[str, Tuple[str, ...]]]):
        if len(specs) == 1 and isinstance(specs[0], (list, tuple)) and not specs[0]:
            super().__init__(specs=())
        else:
            super().__init__(specs=specs)

    @property
    def partitions(self) -> Dict[int, Tuple[str, ...]]:
        result: Dict[int, Tuple[str, ...]] = {}
        for i, s in enumerate(self.specs):
            if s is not None:
                if isinstance(s, str):
                    result[i] = (s,)
                else:
                    result[i] = tuple(str(x) for x in s)
        return result

    def __repr__(self) -> str:
        return f"P({', '.join(repr(s) for s in self.specs)})"

    def __str__(self) -> str:
        return self.__repr__()


P = PartitionSpec


# ============================================================
# ShardingConfig — Declarative Config (Bridge from Dict)
# ============================================================


class ShardingConfig(FluxwayBaseSchema):
    """Bridge model to parse user-provided sharding dictionaries."""

    sharding_type: str = Field(default="static", alias="type")
    axis: List[str] = Field(default_factory=list)
    fn: Optional[str] = None
    routing_dim: Union[int, Tuple[int, ...]] = 0
    source: Optional[str] = None

    def to_spec(self, mesh: DeviceMesh) -> ShardingSpec:
        """Converts declarative config to an executable ShardingSpec."""
        for a in self.axis:
            if not isinstance(a, str):
                raise ValueError(
                    f"Fluxway Sharding Error: Invalid axis '{a}' of type {type(a)}. "
                    f"Sharding axis must be a mesh axis name (str), e.g., 'dp' or 'shard'. "
                    f"Positional integers (0, 1) are not allowed as they are ambiguous in GSPMD."
                )
            if a not in mesh.axis_names:
                raise ValueError(
                    f"Fluxway Sharding Error: Axis '{a}' not found in DeviceMesh axis names: {mesh.axis_names}. "
                    f"Please use valid axis names defined in the @spmd decorator."
                )

        ps = PartitionSpec(*self.axis)

        if self.sharding_type == "replicated" or (
            self.sharding_type == "static" and not self.axis
        ):
            return ShardingSpec(strategy=_CHUNK, mesh=mesh, partition=PartitionSpec())

        # Select strategy based on type/fn
        if self.fn in ("hash_routing", "hash") or self.sharding_type == "dynamic":
            strategy = _INDEXED
        elif self.sharding_type == "mirror":
            if not self.source:
                raise ValueError(
                    "Mirror sharding requires a 'source' identifier (e.g., 'op_name.tensor_name')"
                )
            strategy = MirrorPartition(source=self.source)
        elif self.fn == "cyclic":
            strategy = _CYCLIC
        elif self.fn == "greedy":
            strategy = _GREEDY
        else:
            strategy = _CHUNK

        return ShardingSpec(
            strategy=strategy, mesh=mesh, partition=ps, routing_dim=self.routing_dim
        )


# ============================================================
# PartitionStrategy — Composable Algebraic Mappings
# ============================================================


class PartitionStrategy(FluxwayBaseSchema):
    """Base class for partitioning strategies.

    Encapsulates core algorithm for determining rank assignments (partitioning logic)
    along with derived metadata indicating how the index_desc should be interpreted.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    routing_paradigm: RoutingParadigm = RoutingParadigm.STATIC

    def __call__(
        self,
        indices: Any,
        dim_size: int,
        num_workers: int,
        index_desc: Optional[IndexDescriptor] = None,
    ) -> Any:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def deduce_ranks(
        self,
        indices: Any,
        dim_size: int,
        num_workers: int,
        index_desc: Optional[IndexDescriptor] = None,
    ) -> Any:
        """Determines rank assignments for given indices."""
        return self(indices, dim_size, num_workers, index_desc)

    def get_index_bounds(
        self,
        rank: int,
        world_size: int,
        dim_size: int,
        index_desc: Optional[IndexDescriptor] = None,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Returns (start, end) bounds of the global indices held by this rank.
        Default: covers the entire [0, dim_size).
        """
        return 0, dim_size

    def intersects(
        self,
        src_rank: int,
        src_world_size: int,
        dst_rank: int,
        dst_world_size: int,
        dim_size: int,
        dst_strategy: "PartitionStrategy",
    ) -> bool:
        """Topological test: does this rank have common indices with dst_rank?
        Determines if a 'link' (communication channel) should exist between them.
        """
        # Conservative default for unknown strategies
        return True

    def __repr__(self) -> str:
        return self.name


class ChunkPartition(PartitionStrategy):
    """Chunk partitioning: idx → idx // chunk_size"""

    routing_paradigm: RoutingParadigm = RoutingParadigm.STATIC

    def __call__(self, indices, dim_size, num_workers, index_desc=None):
        backend = get_backend()
        chunk_size = max((dim_size + num_workers - 1) // num_workers, 1)
        ranks = indices // chunk_size
        return backend.where(
            ranks >= num_workers,
            backend.full(backend.shape(ranks), num_workers - 1, dtype=backend.int32()),
            ranks,
        )

    def intersects(
        self, src_rank, src_world_size, dst_rank, dst_world_size, dim_size, dst_strategy
    ) -> bool:
        """Cross-mesh chunk intersection math.
        Focus on physical rank mapping to survive axis name mismatches.
        """
        # Intersection exists iff max(i*N, j*M) < min((i+1)*N, (j+1)*M)
        # where M=src_world, N=dst_world
        return max(src_rank * dst_world_size, dst_rank * src_world_size) < min(
            (src_rank + 1) * dst_world_size, (dst_rank + 1) * src_world_size
        )


class CyclicPartition(PartitionStrategy):
    """Cyclic partitioning: idx → idx % num_workers"""

    routing_paradigm: RoutingParadigm = RoutingParadigm.STATIC

    def __call__(self, indices, dim_size, num_workers, index_desc=None):
        backend = get_backend()
        return backend.mod(indices, num_workers)

    def intersects(
        self, src_rank, src_world_size, dst_rank, dst_world_size, dim_size, dst_strategy
    ) -> bool:
        if not isinstance(dst_strategy, CyclicPartition):
            return True
        import math

        gcd = math.gcd(src_world_size, dst_world_size)
        return (src_rank % gcd) == (dst_rank % gcd)


class IndexedPartition(PartitionStrategy):
    """Dynamic partitioning based on values or lookup tables."""

    routing_paradigm: RoutingParadigm = RoutingParadigm.DYNAMIC

    def __call__(self, indices, dim_size, num_workers, index_desc=None):
        backend = get_backend()

        if isinstance(index_desc, ExplicitIndex):
            values = index_desc.indices
        elif isinstance(index_desc, FunctionalIndex):
            values = index_desc.hash_fn(indices)
        elif isinstance(index_desc, RaggedIndex):
            starts = index_desc.splits[0][:-1]
            values = backend.take(starts, indices)
        else:
            values = indices

        return backend.mod(values, num_workers)


class GreedyPartition(PartitionStrategy):
    """Greedy load balancing partitioning."""

    routing_paradigm: RoutingParadigm = RoutingParadigm.DYNAMIC

    def __call__(self, indices, dim_size, num_workers, index_desc=None):
        backend = get_backend()

        if isinstance(index_desc, RaggedIndex):
            splits = index_desc.splits[0]
            lengths = splits[1:] - splits[:-1]
            lengths_list = backend.to_numpy(lengths).tolist()
        elif isinstance(index_desc, ExplicitIndex):
            lengths = index_desc.indices
            lengths_list = backend.to_numpy(lengths).tolist()
        else:
            lengths_list = [1] * dim_size

        rank_loads = [0] * num_workers
        assigned = []
        for length in lengths_list:
            best = min(range(num_workers), key=lambda i: rank_loads[i])
            assigned.append(best)
            rank_loads[best] += int(length)

        return backend.tensor(assigned, dtype=backend.int32())


class MirrorPartition(PartitionStrategy):
    """Mirror partitioning: routing is identical to a source tensor's receipt."""

    routing_paradigm: RoutingParadigm = RoutingParadigm.DYNAMIC
    source: str

    def __call__(self, indices, dim_size, num_workers, index_desc=None):
        raise RuntimeError(
            f"Mirror Sharding (source={self.source}) must be routed via SymmetryReceipt in the transport layer, "
            "not via PartitionStrategy.__call__."
        )


# Singleton instances for common strategies.
_CHUNK = ChunkPartition()
_CYCLIC = CyclicPartition()
_INDEXED = IndexedPartition()
_GREEDY = GreedyPartition()


# ============================================================
# ShardingSpec — Algebraic Partitioning Description
# ============================================================


def _ravel_multi_index(coords, sizes):
    """Pure-python ravel_multi_index (row-major)."""
    result = 0
    for i, c in enumerate(coords):
        stride = 1
        for j in range(i + 1, len(sizes)):
            stride *= sizes[j]
        result += c * stride
    return result


class ShardingSpec(FluxwayBaseSchema):
    """Algebraic partitioning description linking a stragety, mesh, and specific layout."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    strategy: PartitionStrategy
    mesh: Optional[DeviceMesh] = None
    partition: PartitionSpec
    routing_dim: Union[int, Tuple[int, ...]] = 0
    index_descriptor: Optional[IndexDescriptor] = None

    replica_axes: Optional[Tuple[str, ...]] = Field(default=None)
    replica_size: int = Field(default=1)

    def __init__(
        self,
        strategy: PartitionStrategy,
        partition: PartitionSpec,
        mesh: Optional[DeviceMesh] = None,
        routing_dim: Union[int, Tuple[int, ...]] = 0,
        index_descriptor: Optional[IndexDescriptor] = None,
        **kwargs,
    ):
        super().__init__(
            strategy=strategy,
            partition=partition,
            mesh=mesh,
            routing_dim=routing_dim,
            index_descriptor=index_descriptor,
            **kwargs,
        )

    def with_mesh(self, mesh: DeviceMesh) -> "ShardingSpec":
        """Returns a new ShardingSpec with the provided mesh."""
        return self.model_copy(update={"mesh": mesh})

    @property
    def is_dynamic(self) -> bool:
        """Alias for is_content_addressable for GSPMD-style intuition."""
        return self.strategy.routing_paradigm == RoutingParadigm.DYNAMIC

    @property
    def is_content_addressable(self) -> bool:
        """Technical description: routing is determined by data content/values."""
        return self.is_dynamic

    @property
    def is_static(self) -> bool:
        """Inverse of is_dynamic."""
        return not self.is_dynamic

    @model_validator(mode="after")
    def _compute_replica_info(self) -> "ShardingSpec":
        if self.mesh is not None:
            used = set()
            for axes in self.partition.partitions.values():
                used.update(axes)

            all_axes = set(self.mesh.axis_names)
            repl = sorted(all_axes - used, key=lambda x: self.mesh.axis_names.index(x))
            if repl:
                object.__setattr__(self, "replica_axes", tuple(repl))
                size = 1
                for ax in repl:
                    size *= self.mesh.shape[self.mesh.axis_names.index(ax)]
                object.__setattr__(self, "replica_size", size)
        return self

    @property
    def partitions(self):
        return self.partition.partitions

    def shard_size(self, dim: int) -> int:
        """Returns the effective size of the sub-mesh assigned to a given dimension."""
        if not self.partitions or dim not in self.partitions or self.mesh is None:
            return 1
        result = 1
        for ax in self.partitions[dim]:
            result *= self.mesh.shape[self.mesh.axis_names.index(ax)]
        return result

    def get_shard_rank(self, physical_rank: int, dim: int) -> int:
        """Projects a physical mesh rank down to the logical 1D rank for a given dimension."""
        if not self.partitions or dim not in self.partitions or self.mesh is None:
            return 0
        try:
            coords = self.mesh.get_coordinate(physical_rank)
        except ValueError:
            return 0
        sub_coords = []
        sub_sizes = []
        for ax in self.partitions[dim]:
            axis_idx = self.mesh.axis_names.index(ax)
            sub_coords.append(coords[axis_idx])
            sub_sizes.append(self.mesh.shape[axis_idx])
        return _ravel_multi_index(sub_coords, sub_sizes)

    def get_shard_shape(
        self,
        rank: int,
        data_shape: Tuple[int, ...],
        index_desc: Optional[IndexDescriptor] = None,
    ) -> Tuple[int, ...]:
        """Returns the shape of the local shard for a given rank."""
        backend = get_backend()
        return tuple(
            int(
                backend.shape(
                    self.get_shard_indices(rank, i, data_shape[i], index_desc)
                )[0]
            )
            for i in range(len(data_shape))
        )

    def get_shard_bounds(
        self,
        rank: int,
        dim: int,
        data_shape: Tuple[int, ...],
        index_desc: Optional[IndexDescriptor] = None,
    ) -> Tuple[int, int]:
        """Returns the (start, end) index bounds for a given dimension and rank."""
        world_size = self.shard_size(dim)
        dim_size = data_shape[dim] if data_shape and dim < len(data_shape) else 0

        # Standard full replication or not partitioned on this dim
        if world_size == 1 or not self.partitions or dim not in self.partitions:
            return 0, dim_size

        effective_rank = self.get_shard_rank(rank, dim)
        fn = self.strategy

        if isinstance(fn, ChunkPartition):
            chunk_size = (dim_size + world_size - 1) // world_size
            start = effective_rank * chunk_size
            end = min(start + chunk_size, dim_size)
            return start, end

        # Fallback for non-chunk: use min/max of indices as coarse bounds
        indices = self.get_shard_indices(rank, dim, dim_size, index_desc)
        backend = get_backend()
        if backend.shape(indices)[0] == 0:
            return 0, 0
        return int(backend.item(backend.min(indices))), int(
            backend.item(backend.max(indices))
        ) + 1

    def get_shard_indices(
        self,
        rank: int,
        dim: int,
        dim_size: Optional[int] = None,
        index_desc: Optional[IndexDescriptor] = None,
    ) -> Any:
        """Returns the global indices held by this rank along a specific dimension."""
        backend = get_backend()
        effective_rank = self.get_shard_rank(rank, dim)
        world_size = self.shard_size(dim)

        # Standard full replication or not partitioned on this dim
        if world_size == 1 or not self.partitions or dim not in self.partitions:
            if dim_size is None or dim_size < 0:
                raise ValueError(
                    f"dim_size must be provided and >= 0 for unpartitioned dimension {dim}."
                )
            return backend.arange(0, dim_size, dtype=backend.int32())

        fn = self.strategy
        if isinstance(fn, ChunkPartition):
            if dim_size is None or dim_size < 0:
                raise ValueError(
                    f"dim_size must be provided and >= 0 for chunk partitioned dimension {dim}."
                )
            chunk_size = (dim_size + world_size - 1) // world_size
            start = effective_rank * chunk_size
            end = min(start + chunk_size, dim_size)
            if start >= end:
                return backend.arange(0, 0, dtype=backend.int32())
            return backend.arange(start, end, dtype=backend.int32())

        elif isinstance(fn, CyclicPartition):
            if dim_size is None or dim_size < 0:
                raise ValueError(
                    f"dim_size must be provided and >= 0 for cyclic partitioned dimension {dim}."
                )
            return backend.arange(
                effective_rank, dim_size, step=world_size, dtype=backend.int32()
            )

        # Fallback: materialize and filter
        if dim_size is None or dim_size < 0:
            if self.is_content_addressable:
                return backend.zeros((0,), dtype=backend.int32())
            else:
                raise ValueError(
                    f"dim_size must be provided and >= 0 for static sharding fallback dimension {dim}."
                )

        indices = backend.arange(0, dim_size, dtype=backend.int32())
        if index_desc is None and not self.is_content_addressable:
            index_desc = ContinuousIndex.chunk(0, dim_size)

        target_ranks = fn(indices, dim_size, world_size, index_desc=index_desc)
        mask = target_ranks == effective_rank

        # Static shape required for JIT only if provided.
        # For JAX JIT, nonzero requires a fixed size to avoid dynamic shapes.
        is_jax = backend.name() == "jax"

        if is_jax:
            # In JAX, we try to use size if dim_size is provided to maintain static shapes.
            # However, for exact counts in TransportPlanner, we may need to mask them later.
            # We use fill_value=-1 as a sentinel that TransportPlanner will filter out.
            nz = backend.nonzero(mask, size=dim_size, fill_value=-1)
        else:
            nz = backend.nonzero(mask)
        idx = nz[0] if isinstance(nz, (list, tuple)) else nz
        return backend.reshape(idx, (-1,))

    def infer_global_shape(self, local_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Infer global shape from local shard shape by multiplying dimensions
        partitioned across the mesh. Supports both nested (Ragged) and
        flattened (Dense) views using standard indexing (including -1).
        """
        if self.is_dynamic:
            # Dynamic sharding cannot infer global shape from a single local shard length
            return local_shape

        if self.mesh is None:
            return local_shape

        inferred = list(local_shape)

        # Apply multipliers for each partitioned dimension
        for dim, axes in self.partition.partitions.items():
            # Standard Python indexing handles both positive and negative dims
            try:
                multiplier = 1
                for ax in axes:
                    ax_idx = self.mesh.axis_names.index(ax)
                    multiplier *= self.mesh.shape[ax_idx]
                inferred[dim] *= multiplier
            except IndexError:
                # If sharding dim doesn't exist in local geometry, skip it
                pass

        return tuple(inferred)

    def resolve_producers(
        self, target_spec: "ShardingSpec", target_rank: int, dim: int = 0
    ) -> List[int]:
        """
        Determine which source ranks produce data that overlaps with target_rank's shard.
        Based on the Unified Mathematical Model: S_src(r_s) INTERSECT S_dst(r_d) != EMPTY
        """
        source_world = self.mesh.size if self.mesh else 1

        # Performance optimization: static to static analysis
        if self.is_static and target_spec.is_static:
            s_strategy = self.strategy
            d_strategy = target_spec.strategy

            s_world_size = self.shard_size(dim)
            d_world_size = target_spec.shard_size(dim)

            d_logic = target_spec.get_shard_rank(target_rank, dim)

            producers = []
            for sr in range(source_world):
                s_logic = self.get_shard_rank(sr, dim)
                if s_strategy.intersects(
                    s_logic, s_world_size, d_logic, d_world_size, 0, d_strategy
                ):
                    producers.append(sr)
            return sorted(set(producers))

        # Dynamic/Content-addressable always requires all-to-all fan-in
        return list(range(source_world))

    def resolve_consumers(
        self, target_spec: "ShardingSpec", source_rank: int, dim: int = 0
    ) -> List[int]:
        """
        Determine which target ranks consume data produced by source_rank's shard.
        """
        target_world = target_spec.mesh.size if target_spec.mesh else 1

        if self.is_static and target_spec.is_static:
            s_strategy = self.strategy
            d_strategy = target_spec.strategy

            s_world_size = self.shard_size(dim)
            d_world_size = target_spec.shard_size(dim)

            s_logic = self.get_shard_rank(source_rank, dim)

            consumers = []
            for tr in range(target_world):
                tr_logic = target_spec.get_shard_rank(tr, dim)
                if s_strategy.intersects(
                    s_logic, s_world_size, tr_logic, d_world_size, 0, d_strategy
                ):
                    consumers.append(tr)
            return sorted(set(consumers))

        return list(range(target_world))

    # ---- Factory Methods ----

    @staticmethod
    def chunk(mesh: DeviceMesh, partition: PartitionSpec) -> ShardingSpec:
        return ShardingSpec(_CHUNK, mesh=mesh, partition=partition)

    @staticmethod
    def cyclic(mesh: DeviceMesh, partition: PartitionSpec) -> ShardingSpec:
        return ShardingSpec(_CYCLIC, mesh=mesh, partition=partition)

    @staticmethod
    def index(
        mesh: DeviceMesh, partition: PartitionSpec, routing_dim: int = 0
    ) -> ShardingSpec:
        """Dynamic: routing based on explicit array/tuple indices."""
        return ShardingSpec(
            _INDEXED, mesh=mesh, partition=partition, routing_dim=routing_dim
        )

    @staticmethod
    def hash(
        mesh: DeviceMesh, partition: PartitionSpec, routing_dim: int = 0
    ) -> ShardingSpec:
        """Dynamic: routing based on element values directly."""
        return ShardingSpec(
            _INDEXED, mesh=mesh, partition=partition, routing_dim=routing_dim
        )

    @staticmethod
    def greedy(
        mesh: DeviceMesh, partition: PartitionSpec, routing_dim: int = 0
    ) -> ShardingSpec:
        return ShardingSpec(
            _GREEDY, mesh=mesh, partition=partition, routing_dim=routing_dim
        )

    # ---- Dependency & Derived Sharding ----

    @staticmethod
    def follow(
        driver_spec: "ShardingSpec",
        driver_index_desc: Optional[IndexDescriptor] = None,
        dim: int = 0,
    ) -> "ShardingSpec":
        """Follow another ShardingSpec's partition strategy and mesh."""
        if driver_spec.strategy is None:
            raise ValueError("Cannot follow a ShardingSpec without strategy")

        captured_meta = driver_index_desc
        captured_strategy = driver_spec.strategy

        class _FollowPartition(PartitionStrategy):
            """Wraps a captured strategy with bound index_desc."""

            model_config = ConfigDict(frozen=True)
            routing_paradigm: RoutingParadigm = captured_strategy.routing_paradigm

            def __call__(self, indices, dim_size, num_workers, index_desc=None):
                meta = captured_meta if captured_meta is not None else index_desc
                return captured_strategy(
                    indices, dim_size, num_workers, index_desc=meta
                )

            @property
            def name(self) -> str:
                return f"Follow({captured_strategy.name})"

        specs: List[Optional[Union[str, Tuple[str, ...]]]] = [None] * (dim + 1)
        if driver_spec.partitions and dim in driver_spec.partitions:
            axes = driver_spec.partitions[dim]
            specs[dim] = axes[0] if len(axes) == 1 else axes

        return ShardingSpec(
            _FollowPartition(),
            mesh=driver_spec.mesh,
            partition=PartitionSpec(*specs),
            routing_dim=dim,
        )

    @staticmethod
    def derive_from(
        driver_spec: "ShardingSpec",
        driver_data_shape: Tuple[int, ...],
        driver_index_desc: Any = None,
        follow_dim: int = 0,
    ) -> "ShardingSpec":
        """One-liner for routing_indices distribution."""
        return ShardingSpec.follow(driver_spec, driver_index_desc, dim=follow_dim)

    def __repr__(self) -> str:
        parts = [
            f"fn={self.strategy.name}",
            f"partition={self.partition!r}",
            f"mesh={self.mesh.shape if self.mesh else None}",
        ]
        if self.replica_axes:
            parts.append(f"repl={self.replica_axes}")
        return f"ShardingSpec({', '.join(parts)})"
