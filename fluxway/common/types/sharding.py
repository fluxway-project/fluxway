"""Algebraic Sharding — Layout as Function."""

from __future__ import annotations

import enum
import logging
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
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


class ScalarReduceOp(str, enum.Enum):
    """Reduction operations for scalar replication."""

    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"


# ============================================================
# Routing Sources — Algebraic vs Referential
# ============================================================


class Routing(FluxwayBaseSchema):
    """Base class for routing logic."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    def __hash__(self):
        return hash(tuple(self.__dict__.values()))


class AlgebraicRouting(Routing):
    """Routing derived from algebraic logic (strategy + axes)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    strategy: "PartitionStrategy"
    partition: "PartitionSpec"
    routing_dim: Union[int, Tuple[int, ...]] = 0
    index_descriptor: Optional[IndexDescriptor] = None


class ReferentialRouting(Routing):
    """Routing derived from referencing another TUUID (mirror/follow)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    path: str
    tuuid: Optional[int] = None

    # Map: Target Dimension -> Source Dimension
    dim_map: Dict[int, int] = Field(default_factory=lambda: {0: 0})

    def resolve(self, resolver_fn: Callable[[str], Optional[int]]) -> None:
        """Resolve the logical path to a physical TUUID."""
        if self.tuuid is None:
            object.__setattr__(self, "tuuid", resolver_fn(self.path))

    def __hash__(self):
        # Convert dict to sorted tuple of items for hashing
        items = tuple(sorted(self.dim_map.items())) if self.dim_map else ()
        return hash((self.path, self.tuuid, items))


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

    def to_routing(self) -> ShardingSpec:
        """Converts declarative config to a ShardingSpec (pure logic)."""
        ps = PartitionSpec(*self.axis)

        # 1. Handle Explicitly Replicated or Default Static
        if self.sharding_type == "replicated" or (
            self.sharding_type == "static" and not self.axis
        ):
            return ShardingSpec(
                routing=AlgebraicRouting(strategy=_CHUNK, partition=PartitionSpec())
            )

        # 2. Handle Referential Routing (Follow/Mirror)
        if self.sharding_type == "mirror":
            if not self.source:
                raise ValueError(
                    "[Fluxway Sharding Config Error] 'mirror' (referential) sharding type REQUIRES a 'source' path "
                    "specifying which TUUID to follow (e.g., source='train_step.state')."
                )
            return ShardingSpec.follow(source_path=self.source)

        # 3. Handle Algebraic Routing with Strategy Detection
        if self.sharding_type not in ("static", "dynamic"):
            raise ValueError(
                f"[Fluxway Sharding Config Error] Unsupported sharding type: '{self.sharding_type}'. "
                "Valid types: replicated, static, dynamic, mirror."
            )

        if self.fn in ("hash_routing", "hash"):
            # Explicit hash-based routing ALWAYS uses modulo
            strategy = _INDEXED_MODULO
        elif self.sharding_type == "dynamic":
            # Default dynamic (no fn specified) now uses chunking as requested
            strategy = _INDEXED_CHUNK
        elif self.fn == "cyclic":
            strategy = _CYCLIC
        elif self.fn == "greedy":
            strategy = _GREEDY
        elif self.fn in ("chunk", "block", None) or self.sharding_type == "static":
            # Default to chunk-sharding for static assignments
            strategy = _CHUNK
        else:
            raise ValueError(
                f"[Fluxway Sharding Config Error] Unsupported partitioning function: fn='{self.fn}' for type='{self.sharding_type}'. "
                "Valid fns: hash, cyclic, greedy, chunk."
            )

        return ShardingSpec(
            routing=AlgebraicRouting(
                strategy=strategy, partition=ps, routing_dim=self.routing_dim
            )
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


class IndexedChunkPartition(PartitionStrategy):
    """Dynamic partitioning using continuous chunks based on index values."""

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

        if num_workers <= 1:
            return backend.zeros(backend.shape(values), dtype=backend.int32())

        chunk_size = max((dim_size + num_workers - 1) // num_workers, 1)
        ranks = backend.cast(values // chunk_size, backend.int32())
        return backend.where(
            ranks >= num_workers,
            backend.full(backend.shape(ranks), num_workers - 1, dtype=backend.int32()),
            ranks,
        )


class IndexedModuloPartition(PartitionStrategy):
    """Dynamic partitioning using modulo hashing based on index values."""

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


# Singleton instances for common strategies.
_CHUNK = ChunkPartition()
_CYCLIC = CyclicPartition()
_INDEXED_CHUNK = IndexedChunkPartition()
_INDEXED_MODULO = IndexedModuloPartition()
_GREEDY = GreedyPartition()

# TODO(He Jia) I'm not sure if we need this implicity default alias, maybe it's a bug.
# Compatibility Aliases
IndexedPartition = IndexedChunkPartition


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
    """Logical partitioning description for routing strategy."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    routing: Union[AlgebraicRouting, ReferentialRouting]

    def __init__(self, routing: Union[AlgebraicRouting, ReferentialRouting], **kwargs):
        super().__init__(routing=routing, **kwargs)

    @property
    def strategy(self) -> "PartitionStrategy":
        if isinstance(self.routing, AlgebraicRouting):
            return self.routing.strategy
        raise RuntimeError(
            f"Sharding strategy accessed on unresolved referential routing: {self.routing.path} (TUUID {self.routing.tuuid})"
        )

    @property
    def partition(self) -> "PartitionSpec":
        if isinstance(self.routing, AlgebraicRouting):
            return self.routing.partition
        raise RuntimeError(
            f"Partition spec accessed on unresolved referential routing: {self.routing.path} (TUUID {self.routing.tuuid})"
        )

    @property
    def routing_dim(self) -> Union[int, Tuple[int, ...]]:
        if isinstance(self.routing, AlgebraicRouting):
            return self.routing.routing_dim
        return 0

    @property
    def index_descriptor(self) -> Optional["IndexDescriptor"]:
        if isinstance(self.routing, AlgebraicRouting):
            return self.routing.index_descriptor
        return None

    def bind(self, mesh: DeviceMesh) -> "PhysicalLayout":
        """Bind this logical routing to a physical mesh to create a layout."""
        return PhysicalLayout(spec=self, mesh=mesh)

    @property
    def is_referential(self) -> bool:
        return isinstance(self.routing, ReferentialRouting)

    @property
    def is_dynamic(self) -> bool:
        if self.is_referential:
            return False
        return self.strategy.routing_paradigm == RoutingParadigm.DYNAMIC

    @property
    def is_content_addressable(self) -> bool:
        return self.is_dynamic

    @property
    def is_static(self) -> bool:
        if self.is_referential:
            return False
        return self.strategy.routing_paradigm == RoutingParadigm.STATIC

    @property
    def partitions(self):
        return self.partition.partitions if not self.is_referential else {}

    def __hash__(self):
        return hash(self.routing)

    def __eq__(self, other):
        if not isinstance(other, ShardingSpec):
            return False
        return self.routing == other.routing

    # ---- Factory Methods ----

    @staticmethod
    def chunk(partition: PartitionSpec) -> ShardingSpec:
        return ShardingSpec(
            routing=AlgebraicRouting(strategy=_CHUNK, partition=partition)
        )

    @staticmethod
    def cyclic(partition: PartitionSpec) -> ShardingSpec:
        return ShardingSpec(
            routing=AlgebraicRouting(strategy=_CYCLIC, partition=partition)
        )

    @staticmethod
    def index(partition: PartitionSpec, routing_dim: int = 0) -> ShardingSpec:
        """Dynamic: routing based on explicit array/tuple indices. Default to chunking."""
        return ShardingSpec(
            routing=AlgebraicRouting(
                strategy=_INDEXED_CHUNK, partition=partition, routing_dim=routing_dim
            )
        )

    @staticmethod
    def hash(partition: PartitionSpec, routing_dim: int = 0) -> ShardingSpec:
        """Dynamic: routing based on element values directly. Uses modulo hashing."""
        return ShardingSpec(
            routing=AlgebraicRouting(
                strategy=_INDEXED_MODULO, partition=partition, routing_dim=routing_dim
            )
        )

    @staticmethod
    def greedy(partition: PartitionSpec, routing_dim: int = 0) -> ShardingSpec:
        return ShardingSpec(
            routing=AlgebraicRouting(
                strategy=_GREEDY, partition=partition, routing_dim=routing_dim
            )
        )

    @staticmethod
    def follow(
        source_path: str, dim_map: Optional[Dict[int, int]] = None
    ) -> "ShardingSpec":
        if not isinstance(source_path, str):
            raise TypeError(
                f"ShardingSpec.follow() expects a string path (e.g., 'train_step.embedding.weight'), "
                f"but received {type(source_path).__name__}. "
                f"If you want to copy an existing layout's strategy, use ShardingSpec.chunk(layout.routing.partition) "
                f"and bind it to the same mesh explicitly."
            )
        return ShardingSpec(
            routing=ReferentialRouting(path=source_path, dim_map=dim_map or {0: 0})
        )

    @staticmethod
    def derive_from(
        driver_spec: "PhysicalLayout",
        driver_data_shape: Optional[Tuple[int, ...]] = None,
        driver_index_desc: Any = None,
        follow_dim: int = 0,
    ) -> "ShardingSpec":
        """One-liner for routing_indices distribution. Follows the same strategy but for a different tensor."""
        return ShardingSpec(
            routing=AlgebraicRouting(
                strategy=driver_spec.spec.strategy,
                partition=driver_spec.spec.partition,
                index_descriptor=driver_index_desc,
                routing_dim=follow_dim,
            )
        )

    def __repr__(self) -> str:
        if isinstance(self.routing, AlgebraicRouting):
            return f"ShardingSpec(algebraic, strategy={self.strategy.name}, partition={self.partition!r})"
        return f"ShardingSpec(referential, path={self.routing.path}, tuuid={self.routing.tuuid})"


class PhysicalLayout(FluxwayBaseSchema):
    """Algebraic partitioning bound to a physical Mesh."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    spec: ShardingSpec
    mesh: DeviceMesh

    replica_axes: Optional[Tuple[str, ...]] = Field(default=None)
    replica_size: int = Field(default=1)

    @property
    def partitions(self) -> Dict[int, Tuple[str, ...]]:
        return self.spec.partitions

    @property
    def is_content_addressable(self) -> bool:
        return self.spec.is_content_addressable

    @property
    def is_referential(self) -> bool:
        return self.spec.is_referential

    @property
    def is_static(self) -> bool:
        return self.spec.is_static

    @property
    def routing(self) -> Union[AlgebraicRouting, ReferentialRouting]:
        return self.spec.routing

    @property
    def strategy(self) -> "PartitionStrategy":
        return self.spec.strategy

    @property
    def routing_dim(self) -> Union[int, Tuple[int, ...]]:
        return self.spec.routing_dim

    @property
    def index_descriptor(self) -> Optional["IndexDescriptor"]:
        return self.spec.index_descriptor

    def __init__(self, spec: ShardingSpec, mesh: DeviceMesh, **kwargs):
        super().__init__(spec=spec, mesh=mesh, **kwargs)

    @model_validator(mode="after")
    def _compute_replica_info(self) -> "PhysicalLayout":
        if self.spec.is_referential:
            return self

        used = set()
        for axes in self.spec.partitions.values():
            used.update(ax for ax in axes if ax in self.mesh.axis_names)

        all_axes = set(self.mesh.axis_names)
        repl = sorted(all_axes - used, key=lambda x: self.mesh.axis_names.index(x))
        if repl:
            object.__setattr__(self, "replica_axes", tuple(repl))
            size = 1
            for ax in repl:
                size *= self.mesh.shape[self.mesh.axis_names.index(ax)]
            object.__setattr__(self, "replica_size", size)
        return self

    def shard_size(self, dim: int) -> int:
        """Returns the effective size of the sub-mesh assigned to a given dimension."""
        if not self.spec.partitions or dim not in self.spec.partitions:
            return 1
        result = 1
        for ax in self.spec.partitions[dim]:
            if ax in self.mesh.axis_names:
                result *= self.mesh.shape[self.mesh.axis_names.index(ax)]
        return result

    def get_shard_rank(self, physical_rank: int, dim: int) -> int:
        """Projects a physical mesh rank down to the logical 1D rank for a given dimension."""
        if not self.spec.partitions or dim not in self.spec.partitions:
            return 0
        try:
            coords = self.mesh.get_coordinate(physical_rank)
        except ValueError:
            return 0
        sub_coords = []
        sub_sizes = []
        for ax in self.spec.partitions[dim]:
            if ax in self.mesh.axis_names:
                axis_idx = self.mesh.axis_names.index(ax)
                sub_coords.append(coords[axis_idx])
                sub_sizes.append(self.mesh.shape[axis_idx])
        if not sub_sizes:
            return 0
        return _ravel_multi_index(sub_coords, sub_sizes)

    def get_logical_identity(self, physical_rank: int) -> Tuple[int, ...]:
        """Returns a tuple that uniquely identifies the logical shard (fragment)."""
        if self.spec.is_dynamic:
            return (physical_rank,)

        if self.spec.partitions:
            return tuple(
                self.get_shard_rank(physical_rank, dim)
                for dim in sorted(self.spec.partitions.keys())
            )

        if self.spec.is_static:
            return (0,)

        raise RuntimeError(
            f"Cannot determine logical identity for routing type {type(self.spec.routing).__name__}."
        )

    def get_shard_shape(
        self,
        rank: int,
        data_shape: Tuple[int, ...],
        index_desc: Optional[IndexDescriptor] = None,
    ) -> Tuple[int, ...]:
        """Returns the shape of the local shard for a given rank."""
        backend = get_backend()
        result = []
        for i in range(len(data_shape)):
            raw = self.get_shard_indices(rank, i, data_shape[i], index_desc)
            # JAX nonzero() returns fixed-size arrays padded with -1 for JIT compatibility.
            effective = raw[raw != -1]
            result.append(int(backend.shape(effective)[0]))
        return tuple(result)

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
        if (
            world_size == 1
            or not self.spec.partitions
            or dim not in self.spec.partitions
        ):
            return 0, dim_size

        effective_rank = self.get_shard_rank(rank, dim)
        fn = self.spec.strategy

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
        if (
            world_size == 1
            or not self.spec.partitions
            or dim not in self.spec.partitions
        ):
            if dim_size is None or dim_size < 0:
                raise ValueError(
                    f"dim_size must be provided and >= 0 for unpartitioned dimension {dim}."
                )
            return backend.arange(0, dim_size, dtype=backend.int32())

        fn = self.spec.strategy
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
            if self.spec.is_content_addressable:
                return backend.zeros((0,), dtype=backend.int32())
            else:
                raise ValueError(
                    f"dim_size must be provided and >= 0 for static sharding fallback dimension {dim}."
                )

        indices = backend.arange(0, dim_size, dtype=backend.int32())
        if index_desc is None and not self.spec.is_content_addressable:
            index_desc = ContinuousIndex.chunk(0, dim_size)

        target_ranks = fn(indices, dim_size, world_size, index_desc=index_desc)
        mask = target_ranks == effective_rank

        # Static shape required for JIT only if provided.
        is_jax = backend.name() == "jax"

        if is_jax:
            nz = backend.nonzero(mask, size=dim_size, fill_value=-1)
        else:
            nz = backend.nonzero(mask)
        idx = nz[0] if isinstance(nz, (list, tuple)) else nz
        return backend.reshape(idx, (-1,))

    def infer_global_shape(self, local_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.spec.is_dynamic:
            return local_shape

        inferred = list(local_shape)

        # Apply multipliers for each partitioned dimension
        for dim, axes in self.spec.partitions.items():
            try:
                multiplier = 1
                for ax in axes:
                    ax_idx = self.mesh.axis_names.index(ax)
                    multiplier *= self.mesh.shape[ax_idx]
                inferred[dim] *= multiplier
            except IndexError:
                pass

        return tuple(inferred)

    def resolve_producers(
        self, target_layout: "PhysicalLayout", target_rank: int, dim: int = 0
    ) -> List[int]:
        """
        Determine which source ranks produce data that overlaps with target_rank's shard.
        Based on the Unified Mathematical Model: S_src(r_s) INTERSECT S_dst(r_d) != EMPTY
        """
        source_world = self.mesh.size

        # Performance optimization: static to static analysis
        if self.spec.is_static and target_layout.spec.is_static:
            s_strategy = self.spec.strategy
            d_strategy = target_layout.spec.strategy

            s_world_size = self.shard_size(dim)
            d_world_size = target_layout.shard_size(dim)

            d_logic = target_layout.get_shard_rank(target_rank, dim)

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
        self, target_layout: "PhysicalLayout", source_rank: int, dim: int = 0
    ) -> List[int]:
        """
        Determine which target ranks consume data produced by source_rank's shard.
        """
        target_world = target_layout.mesh.size

        if self.spec.is_static and target_layout.spec.is_static:
            s_strategy = self.spec.strategy
            d_strategy = target_layout.spec.strategy

            s_world_size = self.shard_size(dim)
            d_world_size = target_layout.shard_size(dim)

            s_logic = self.get_shard_rank(source_rank, dim)

            consumers = []
            for tr in range(target_world):
                tr_logic = target_layout.get_shard_rank(tr, dim)
                if s_strategy.intersects(
                    s_logic, s_world_size, tr_logic, d_world_size, 0, d_strategy
                ):
                    consumers.append(tr)
            return sorted(set(consumers))

        return list(range(target_world))

    def __hash__(self):
        return hash((self.spec, self.mesh, self.replica_axes, self.replica_size))

    def __eq__(self, other):
        if not isinstance(other, PhysicalLayout):
            return False
        return (
            self.spec == other.spec
            and self.mesh == other.mesh
            and self.replica_axes == other.replica_axes
            and self.replica_size == other.replica_size
        )

    def __repr__(self) -> str:
        return f"PhysicalLayout(routing={self.routing!r}, mesh={self.mesh.shape}, replica_size={self.replica_size})"

    @staticmethod
    def derive_from(
        source_path: str, source_dim: int = 0, target_dim: int = 0
    ) -> "ShardingSpec":
        """Specialized follow for derived tensors like routing_indices."""
        return ShardingSpec.follow(
            source_path=source_path, dim_map={target_dim: source_dim}
        )
