"""Algebraic Sharding — Layout as Function."""

from __future__ import annotations

import enum
from typing import Any, List, Optional, Tuple, Union, Dict

from fluxway.common.backend import get_backend
from fluxway.common.mesh import DeviceMesh
from fluxway.common.topology import (
    TopologyMorphology,
    Topology,
    IndexedTopology,
    RaggedTopology,
    CallableTopology,
)


# ============================================================
# Enums — Routing Semantics
# ============================================================


from fluxway.common.types.configs import FluxwayBaseSchema, ConfigDict
from pydantic import Field, model_validator


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

    specs: Tuple[Optional[Union[str, int, Tuple[str, ...]]], ...] = Field(
        default_factory=tuple
    )

    def __init__(self, *specs: Optional[Union[str, int, Tuple[str, ...]]]):
        if len(specs) == 1 and isinstance(specs[0], (list, tuple)) and not specs[0]:
            super().__init__(specs=())
        else:
            super().__init__(specs=specs)

    @property
    def partitions(self) -> Dict[int, Tuple[str, ...]]:
        result = {}
        for i, s in enumerate(self.specs):
            if s is not None:
                if isinstance(s, (str, int)):
                    result[i] = (str(s),)
                else:
                    result[i] = tuple(str(x) for x in s)
        return result

    def to_named_spec(self, mesh: DeviceMesh) -> PartitionSpec:
        """Converts integer axis indices to mesh axis names."""
        new_specs = []
        for s in self.specs:
            if isinstance(s, int):
                if s < len(mesh.axis_names):
                    new_specs.append(mesh.axis_names[s])
                else:
                    new_specs.append(None)
            elif isinstance(s, tuple):
                new_specs.append(
                    tuple(mesh.axis_names[x] if isinstance(x, int) else x for x in s)
                )
            else:
                new_specs.append(s)
        return PartitionSpec(*new_specs)

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
    axis: List[Union[str, int]] = Field(default_factory=list)
    fn: Optional[str] = None
    routing_dim: Union[int, Tuple[int, ...]] = 0

    def to_spec(self, mesh: DeviceMesh) -> ShardingSpec:
        """Converts declarative config to an executable ShardingSpec."""
        ps = PartitionSpec(*self.axis).to_named_spec(mesh)

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
    along with derived metadata indicating how the topology should be interpreted.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    routing_paradigm: RoutingParadigm = RoutingParadigm.STATIC

    def __call__(
        self,
        indices: Any,
        dim_size: int,
        num_workers: int,
        topology: Optional[Topology] = None,
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

    def __call__(self, indices, dim_size, num_workers, topology=None):
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

    def __call__(self, indices, dim_size, num_workers, topology=None):
        backend = get_backend()
        return backend.mod(indices, num_workers)


class IndexedPartition(PartitionStrategy):
    """Dynamic partitioning based on values or lookup tables."""

    routing_paradigm: RoutingParadigm = RoutingParadigm.DYNAMIC

    def __call__(self, indices, dim_size, num_workers, topology=None):
        backend = get_backend()

        if isinstance(topology, IndexedTopology):
            values = topology.indices
        elif isinstance(topology, CallableTopology):
            values = topology.hash_fn(indices)
        elif isinstance(topology, RaggedTopology):
            starts = topology.splits[0][:-1]
            values = backend.take(starts, indices)
        else:
            values = indices

        return backend.mod(values, num_workers)


class GreedyPartition(PartitionStrategy):
    """Greedy load balancing partitioning."""

    routing_paradigm: RoutingParadigm = RoutingParadigm.DYNAMIC

    def __call__(self, indices, dim_size, num_workers, topology=None):
        backend = get_backend()

        if isinstance(topology, RaggedTopology):
            splits = topology.splits[0]
            lengths = splits[1:] - splits[:-1]
            lengths_list = backend.to_numpy(lengths).tolist()
        elif isinstance(topology, IndexedTopology):
            lengths = topology.indices
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
    topology: Optional[Topology] = None
    partitions: Optional[Dict[int, Tuple[str, ...]]] = None
    strategy: Optional[PartitionStrategy] = None
    routing_dim: Optional[Union[int, Tuple[int, ...]]] = None
    replica_axes: Optional[List[str]] = None
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
                object.__setattr__(self, "replica_axes", repl)
                size = 1
                for ax in repl:
                    size *= self.mesh.shape[self.mesh.axis_names.index(ax)]
                object.__setattr__(self, "replica_size", size)
        return self

    @property
    def topology_morphology(self) -> TopologyMorphology:
        if self.topology is not None:
            return self.topology.morphology
        if self.strategy and self.strategy.routing_paradigm == RoutingParadigm.DYNAMIC:
            return TopologyMorphology.INDEXED
        return TopologyMorphology.CONTINUOUS

    @property
    def is_static(self) -> bool:
        return self.topology is None

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

    def logical_to_physical(self, logical_ranks: Any, dim: int) -> Any:
        """Map logical partition ranks to physical device ranks via mesh."""
        backend = get_backend()
        if not self.partitions or dim not in self.partitions:
            return backend.zeros_like(logical_ranks)
        coords = [
            backend.zeros(backend.shape(logical_ranks), dtype=backend.int32())
            for _ in self.mesh.shape
        ]
        # In multi-axis mode per dim, reverse unravel is needed.
        # For our main flows (dynamic routing) it's mostly single axis routing,
        # but we handle the single axis case cleanly here. Complex multi-axis reverse
        # routing may require unravel_index over sub-mesh.
        axes = self.partitions[dim]
        if len(axes) > 1:
            raise NotImplementedError(
                "logical_to_physical not fully supported for multi-axis joint routing yet."
            )
        axis_name = axes[0]
        axis_idx = self.mesh.axis_names.index(axis_name)
        coords[axis_idx] = logical_ranks
        mesh_ids = backend.tensor(self.mesh.device_ids, dtype=backend.int32())
        return mesh_ids[tuple(coords)]

    def get_local_indices(self, rank: int, dim: int) -> Any:
        """Returns the global indices held by this rank along a specific dimension."""
        backend = get_backend()
        effective_rank = self.get_shard_rank(rank, dim)
        ews = self.shard_size(dim)
        dim_size = self.data_shape[dim]

        # Standard full replication or not partitioned on this dim
        if ews == 1 or not self.partitions or dim not in self.partitions:
            return backend.arange(0, dim_size, dtype=backend.int32())

        fn = self.strategy
        if isinstance(fn, ChunkPartition):
            chunk_size = (dim_size + ews - 1) // ews
            start = effective_rank * chunk_size
            end = min(start + chunk_size, dim_size)
            if start >= end:
                return backend.arange(0, 0, dtype=backend.int32())
            return backend.arange(start, end, dtype=backend.int32())

        elif isinstance(fn, CyclicPartition):
            return backend.arange(
                effective_rank, dim_size, step=ews, dtype=backend.int32()
            )

        # Fallback: materialize and filter
        indices = backend.arange(0, dim_size, dtype=backend.int32())
        target_ranks = fn(indices, dim_size, ews, topology=self.topology)
        mask = target_ranks == effective_rank
        nz = backend.nonzero(mask)
        return backend.reshape(nz, (-1,))

    @property
    def is_content_addressable(self) -> bool:
        return self.topology_morphology in (
            TopologyMorphology.RAGGED,
            TopologyMorphology.INDEXED,
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

    replica_axes: Optional[List[str]] = Field(default=None)
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

    @model_validator(mode="after")
    def _compute_replica_info(self) -> "ShardingSpec":
        if self.mesh is not None:
            used = set()
            for axes in self.partition.partitions.values():
                used.update(axes)

            all_axes = set(self.mesh.axis_names)
            repl = sorted(all_axes - used, key=lambda x: self.mesh.axis_names.index(x))
            if repl:
                object.__setattr__(self, "replica_axes", repl)
                size = 1
                for ax in repl:
                    size *= self.mesh.shape[self.mesh.axis_names.index(ax)]
                object.__setattr__(self, "replica_size", size)
        return self

    @property
    def partitions(self):
        return self.partition.partitions

    @property
    def is_content_addressable(self) -> bool:
        return self.strategy.routing_paradigm == RoutingParadigm.DYNAMIC

    def _to_layout(
        self,
        data_shape: Optional[Tuple[int, ...]] = None,
        topology: Optional[Topology] = None,
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
            topology=topology,
            partitions=self.partitions,
            strategy=self.strategy,
            routing_dim=self.routing_dim,
            replica_axes=self.replica_axes,
            replica_size=self.replica_size,
        )

    def get_expected_local_indices(
        self, rank: int, local_dim_size: int, dim: int = 0
    ) -> Optional[Any]:
        """Returns expected global indices for implicit mesh inference (chunk only)."""
        backend = get_backend()
        if not isinstance(self.strategy, ChunkPartition):
            return None

        if self.mesh is None:
            return None

        if not self.partitions or dim not in self.partitions:
            return backend.arange(0, local_dim_size, dtype=backend.int32())

        coords = self.mesh.get_coordinate(rank)
        sub_coords = []
        sub_sizes = []
        for ax in self.partitions[dim]:
            axis_idx = self.mesh.axis_names.index(ax)
            sub_coords.append(coords[axis_idx])
            sub_sizes.append(self.mesh.shape[axis_idx])
        effective_rank = _ravel_multi_index(sub_coords, sub_sizes)
        start = effective_rank * local_dim_size
        return backend.arange(start, start + local_dim_size, dtype=backend.int32())

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

        captured_meta = source_layout.topology
        captured_strategy = source_layout.strategy

        class _FollowPartition(PartitionStrategy):
            """Wraps a captured strategy with bound topology."""

            model_config = ConfigDict(frozen=True)
            routing_paradigm: RoutingParadigm = captured_strategy.routing_paradigm

            def __call__(self, indices, dim_size, num_workers, topology=None):
                meta = captured_meta if captured_meta is not None else topology
                return captured_strategy(indices, dim_size, num_workers, topology=meta)

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
        driver_topology: Any = None,
        follow_dim: int = 0,
    ) -> ShardingSpec:
        """One-liner for routing_indices distribution."""
        driver_layout = driver_spec._to_layout(
            driver_data_shape, topology=driver_topology
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
