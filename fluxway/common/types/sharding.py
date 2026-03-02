"""Algebraic Sharding — Layout as Function."""

from __future__ import annotations

import enum
from typing import Any, List, Optional, Tuple, Union, Dict
from pydantic import Field, model_validator

from fluxway.common.backend import get_backend
from fluxway.common.mesh import DeviceMesh
from fluxway.common.index_descriptor import (
    IndexMorphology,
    IndexDescriptor,
    ExplicitIndex,
    RaggedIndex,
    FunctionalIndex,
)
from fluxway.common.types.configs import FluxwayBaseSchema, ConfigDict


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


class CyclicPartition(PartitionStrategy):
    """Cyclic partitioning: idx → idx % num_workers"""

    routing_paradigm: RoutingParadigm = RoutingParadigm.STATIC

    def __call__(self, indices, dim_size, num_workers, index_desc=None):
        backend = get_backend()
        return backend.mod(indices, num_workers)


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


# Singleton instances for common strategies.
_CHUNK = ChunkPartition()
_CYCLIC = CyclicPartition()
_INDEXED = IndexedPartition()
_GREEDY = GreedyPartition()


# ============================================================
# Layout — Data Distribution Description
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


class Layout(FluxwayBaseSchema):
    """Data distribution description describing physical placement on a DeviceMesh."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    data_shape: Tuple[int, ...]
    mesh: DeviceMesh
    index_desc: Optional[IndexDescriptor] = None
    partitions: Optional[Dict[int, Tuple[str, ...]]] = None
    strategy: Optional[PartitionStrategy] = None
    routing_dim: Optional[Union[int, Tuple[int, ...]]] = None
    replica_axes: Optional[Tuple[str, ...]] = None
    replica_size: int = 1

    @model_validator(mode="after")
    def _compute_replica_info(self) -> "Layout":
        if self.replica_axes is None and self.mesh is not None and self.partitions:
            used = set()
            for axes in self.partitions.values():
                used.update(axes)
            all_axes = set(self.mesh.axis_names)
            repl = sorted(all_axes - used, key=lambda x: self.mesh.axis_names.index(x))
            if repl:
                # We use object.__setattr__ because the model is frozen
                object.__setattr__(self, "replica_axes", tuple(repl))
                size = 1
                for ax in repl:
                    size *= self.mesh.shape[self.mesh.axis_names.index(ax)]
                object.__setattr__(self, "replica_size", size)
        return self

    @property
    def index_desc_morphology(self) -> IndexMorphology:
        if self.index_desc is not None:
            return self.index_desc.morphology
        if self.strategy and self.strategy.routing_paradigm == RoutingParadigm.DYNAMIC:
            return IndexMorphology.INDEXED
        return IndexMorphology.CONTINUOUS

    @property
    def is_static(self) -> bool:
        return self.index_desc is None

    @property
    def num_workers(self) -> int:
        return self.mesh.size

    def shard_size(self, dim: int) -> int:
        """Returns the effective size of the sub-mesh assigned to a given dimension."""
        if not self.partitions or dim not in self.partitions:
            return 1
        result = 1
        for ax in self.partitions[dim]:
            result *= self.mesh.shape[self.mesh.axis_names.index(ax)]
        return result

    def get_shard_rank(self, physical_rank: int, dim: int) -> int:
        """Projects a physical mesh rank down to the logical 1D rank for a given dimension."""
        if not self.partitions or dim not in self.partitions:
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

    def get_shard_shape(self, rank: int) -> Tuple[int, ...]:
        """Returns the shape of the local shard for a given rank."""
        backend = get_backend()
        return tuple(
            int(backend.shape(self.get_shard_indices(rank, i))[0])
            for i in range(len(self.data_shape))
        )

    def get_shard_bounds(self, rank: int, dim: int) -> Tuple[int, int]:
        """Returns the (start, end) index bounds for a given dimension and rank."""
        world_size = self.shard_size(dim)
        dim_size = self.data_shape[dim]

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
        indices = self.get_shard_indices(rank, dim)
        backend = get_backend()
        if backend.shape(indices)[0] == 0:
            return 0, 0
        return int(backend.item(backend.min(indices))), int(
            backend.item(backend.max(indices))
        ) + 1

    def get_shard_indices(self, rank: int, dim: int) -> Any:
        """Returns the global indices held by this rank along a specific dimension."""
        backend = get_backend()
        effective_rank = self.get_shard_rank(rank, dim)
        world_size = self.shard_size(dim)
        dim_size = self.data_shape[dim]

        # Standard full replication or not partitioned on this dim
        if world_size == 1 or not self.partitions or dim not in self.partitions:
            return backend.arange(0, dim_size, dtype=backend.int32())

        fn = self.strategy
        if isinstance(fn, ChunkPartition):
            chunk_size = (dim_size + world_size - 1) // world_size
            start = effective_rank * chunk_size
            end = min(start + chunk_size, dim_size)
            if start >= end:
                return backend.arange(0, 0, dtype=backend.int32())
            return backend.arange(start, end, dtype=backend.int32())

        elif isinstance(fn, CyclicPartition):
            return backend.arange(
                effective_rank, dim_size, step=world_size, dtype=backend.int32()
            )

        # Fallback: materialize and filter
        # If dim_size is unknown (-1), we return an empty array to avoid JAX compile errors in nonzero(size=-1)
        if dim_size < 0:
            return backend.zeros((0,), dtype=backend.int32())

        indices = backend.arange(0, dim_size, dtype=backend.int32())
        target_ranks = fn(indices, dim_size, world_size, index_desc=self.index_desc)
        mask = target_ranks == effective_rank
        nz = backend.nonzero(mask, size=dim_size)
        # nz is a tuple (indices_array,) in jax, must unpack the first element for 1D unless use flatnonzero
        idx = nz[0] if isinstance(nz, (list, tuple)) else nz
        return backend.reshape(idx, (-1,))

    @property
    def is_content_addressable(self) -> bool:
        return self.index_desc_morphology in (
            IndexMorphology.RAGGED,
            IndexMorphology.INDEXED,
        )


# ============================================================
# ShardingSpec — Algebraic Partitioning Description
# ============================================================


class ShardingSpec(FluxwayBaseSchema):
    """Algebraic partitioning description linking a stragety, mesh, and specific layout."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    strategy: PartitionStrategy
    mesh: Optional[DeviceMesh] = None
    partition: PartitionSpec
    routing_dim: Union[int, Tuple[int, ...]] = 0

    replica_axes: Optional[Tuple[str, ...]] = Field(default=None)
    replica_size: int = Field(default=1)

    def __init__(
        self,
        strategy: PartitionStrategy,
        partition: PartitionSpec,
        mesh: Optional[DeviceMesh] = None,
        routing_dim: Union[int, Tuple[int, ...]] = 0,
        **kwargs,
    ):
        super().__init__(
            strategy=strategy,
            partition=partition,
            mesh=mesh,
            routing_dim=routing_dim,
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

    def _to_layout(
        self,
        data_shape: Optional[Tuple[int, ...]] = None,
        index_desc: Optional[IndexDescriptor] = None,
    ) -> Layout:
        """Internal: evaluate specification into a concrete Layout."""
        if data_shape is None:
            if not self.is_content_addressable:
                raise ValueError("data_shape is required for static sharding")
            data_shape = (-1,)

        if self.mesh is None:
            raise ValueError("mesh is required to evaluate Layout")

        return Layout(
            data_shape=data_shape,
            mesh=self.mesh,
            index_desc=index_desc,
            partitions=self.partitions,
            strategy=self.strategy,
            routing_dim=self.routing_dim,
            replica_axes=self.replica_axes,
            replica_size=self.replica_size,
        )

    def resolve_producers(
        self, target_spec: "ShardingSpec", target_rank: int, dim: int = 0
    ) -> List[int]:
        """
        Determine which source ranks produce data that overlaps with target_rank's shard.
        """
        source_world = self.mesh.size if self.mesh else 1
        target_world = target_spec.mesh.size if target_spec.mesh else 1

        if (
            getattr(self.strategy, "routing_paradigm", None) != RoutingParadigm.STATIC
            or getattr(target_spec.strategy, "routing_paradigm", None)
            != RoutingParadigm.STATIC
        ):
            return list(range(source_world))

        if not isinstance(self.strategy, ChunkPartition) or not isinstance(
            target_spec.strategy, ChunkPartition
        ):
            return list(range(source_world))

        scale = max(source_world, target_world)
        dummy_dim_size = 1024 * scale

        source_layout = self._to_layout((dummy_dim_size,))
        target_layout = target_spec._to_layout((dummy_dim_size,))

        t_logic = target_layout.get_shard_rank(target_rank, dim)
        t_world_size = target_layout.shard_size(dim)

        def get_chunk_bounds(rank, world_size, dim_size):
            chunk_size = max((dim_size + world_size - 1) // world_size, 1)
            start = rank * chunk_size
            end = min(start + chunk_size, dim_size)
            return start, end

        if t_world_size <= 1:
            t_start, t_end = 0, dummy_dim_size
        else:
            t_start, t_end = get_chunk_bounds(t_logic, t_world_size, dummy_dim_size)

        if t_start >= t_end:
            return []

        producers = []
        s_world_size = source_layout.shard_size(dim)
        for sr in range(source_world):
            if s_world_size <= 1:
                s_start, s_end = 0, dummy_dim_size
            else:
                s_logic = source_layout.get_shard_rank(sr, dim)
                s_start, s_end = get_chunk_bounds(s_logic, s_world_size, dummy_dim_size)

            if s_start < s_end and max(t_start, s_start) < min(t_end, s_end):
                producers.append(sr)

        return sorted(set(producers))

    def resolve_consumers(
        self, target_spec: "ShardingSpec", source_rank: int, dim: int = 0
    ) -> List[int]:
        """
        Determine which target ranks consume data produced by source_rank's shard.
        """
        source_world = self.mesh.size if self.mesh else 1
        target_world = target_spec.mesh.size if target_spec.mesh else 1

        if (
            getattr(self.strategy, "routing_paradigm", None) != RoutingParadigm.STATIC
            or getattr(target_spec.strategy, "routing_paradigm", None)
            != RoutingParadigm.STATIC
        ):
            return list(range(target_world))

        if not isinstance(self.strategy, ChunkPartition) or not isinstance(
            target_spec.strategy, ChunkPartition
        ):
            return list(range(target_world))

        scale = max(source_world, target_world)
        dummy_dim_size = 1024 * scale

        source_layout = self._to_layout((dummy_dim_size,))
        target_layout = target_spec._to_layout((dummy_dim_size,))

        s_logic = source_layout.get_shard_rank(source_rank, dim)
        s_world_size = source_layout.shard_size(dim)

        def get_chunk_bounds(rank, world_size, dim_size):
            chunk_size = max((dim_size + world_size - 1) // world_size, 1)
            start = rank * chunk_size
            end = min(start + chunk_size, dim_size)
            return start, end

        if s_world_size <= 1:
            s_start, s_end = 0, dummy_dim_size
        else:
            s_start, s_end = get_chunk_bounds(s_logic, s_world_size, dummy_dim_size)

        if s_start >= s_end:
            return []

        consumers = []
        t_world_size = target_layout.shard_size(dim)
        for tr in range(target_world):
            if t_world_size <= 1:
                t_start, t_end = 0, dummy_dim_size
            else:
                t_logic = target_layout.get_shard_rank(tr, dim)
                t_start, t_end = get_chunk_bounds(t_logic, t_world_size, dummy_dim_size)

            if t_start < t_end and max(t_start, s_start) < min(t_end, s_end):
                consumers.append(tr)

        return sorted(set(consumers))

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
    def follow(source_layout: Layout, dim: int = 0) -> ShardingSpec:
        """Follow another Layout's partition strategy and mesh."""
        if source_layout.strategy is None:
            raise ValueError("Cannot follow a Layout without strategy")

        captured_meta = source_layout.index_desc
        captured_strategy = source_layout.strategy

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

        # Reconstruct PartitionSpec at the target dim
        specs = [None] * (dim + 1)
        if source_layout.partitions and dim in source_layout.partitions:
            axes = source_layout.partitions[dim]
            specs[dim] = axes[0] if len(axes) == 1 else axes

        return ShardingSpec(
            _FollowPartition(),
            mesh=source_layout.mesh,
            partition=PartitionSpec(*specs),
            routing_dim=dim,
        )

    @staticmethod
    def derive_from(
        driver_spec: ShardingSpec,
        driver_data_shape: Tuple[int, ...],
        driver_index_desc: Any = None,
        follow_dim: int = 0,
    ) -> ShardingSpec:
        """One-liner for spatial_indices distribution."""
        driver_layout = driver_spec._to_layout(
            driver_data_shape, index_desc=driver_index_desc
        )
        return ShardingSpec.follow(driver_layout, dim=follow_dim)

    def __repr__(self) -> str:
        parts = [
            f"fn={self.strategy.name}",
            f"partition={self.partition!r}",
            f"mesh={self.mesh.shape if self.mesh else None}",
        ]
        if self.replica_axes:
            parts.append(f"repl={self.replica_axes}")
        return f"ShardingSpec({', '.join(parts)})"
