"""Algebraic Sharding — Layout as Function."""

from __future__ import annotations

from dataclasses import dataclass
import enum
from typing import Any, List, Optional, Tuple, Union

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


class RoutingParadigm(enum.Enum):
    STATIC = enum.auto()
    DYNAMIC = enum.auto()


# ============================================================
# PartitionSpec — JAX-style Partition Description
# ============================================================


class PartitionSpec:
    """Maps tensor dimensions to device mesh axes by position.

    This specification aligns target tensor dimensions with distributed topology axes.

    Examples:
        P('x')           -> dimension 0 is sharded along device axis 'x'
        P(None, 'x')     -> dimension 1 is sharded along device axis 'x'
        P(('x', 'y'))    -> dimension 0 is jointly sharded along axes ('x', 'y')
    """

    def __init__(self, *specs: Optional[Union[str, Tuple[str, ...]]]):
        self.specs = specs

    @property
    def partitions(self):
        """Returns a mapping from tensor dimension index to mesh axis names."""
        result = {}
        for i, s in enumerate(self.specs):
            if s is not None:
                result[i] = (s,) if isinstance(s, str) else tuple(s)
        if not result:
            pass  # P() implies complete replication, valid.
        return result

    def __repr__(self) -> str:
        return f"P({', '.join(repr(s) for s in self.specs)})"

    def __eq__(self, other):
        return isinstance(other, PartitionSpec) and self.specs == other.specs

    def __hash__(self):
        return hash(self.specs)


P = PartitionSpec


# ============================================================
# PartitionStrategy — Composable Algebraic Mappings
# ============================================================


class PartitionStrategy:
    """Base class for partitioning strategies.

    Encapsulates core algorithm for determining rank assignments (partitioning logic)
    along with derived metadata indicating how the topology should be interpreted.
    """

    routing_paradigm: RoutingParadigm = RoutingParadigm.STATIC

    def __call__(
        self,
        indices: Any,
        dim_size: int,
        world_size: int,
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

    def __call__(self, indices, dim_size, world_size, topology=None):
        backend = get_backend()
        chunk_size = max((dim_size + world_size - 1) // world_size, 1)
        ranks = indices // chunk_size
        return backend.where(
            ranks >= world_size,
            backend.full(backend.shape(ranks), world_size - 1, dtype=backend.int32()),
            ranks,
        )


class CyclicPartition(PartitionStrategy):
    """Cyclic partitioning: idx → idx % world_size"""

    def __call__(self, indices, dim_size, world_size, topology=None):
        backend = get_backend()
        return backend.mod(indices, world_size)


class IndexedPartition(PartitionStrategy):
    """Dynamic partitioning based on values or lookup tables."""

    routing_paradigm = RoutingParadigm.DYNAMIC

    def __call__(self, indices, dim_size, world_size, topology=None):
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

        return backend.mod(values, world_size)


class GreedyPartition(PartitionStrategy):
    """Greedy load balancing partitioning."""

    routing_paradigm = RoutingParadigm.DYNAMIC

    def __call__(self, indices, dim_size, world_size, topology=None):
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

        rank_loads = [0] * world_size
        assigned = []
        for length in lengths_list:
            best = min(range(world_size), key=lambda i: rank_loads[i])
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


@dataclass
class Layout:
    """Data distribution description describing physical placement on a DeviceMesh."""

    data_shape: Tuple[int, ...]
    mesh: DeviceMesh
    topology: Optional[Topology] = None
    partitions: Optional[dict] = None  # Dict[int, Tuple[str, ...]]
    strategy: Optional[PartitionStrategy] = None
    routing_dim: Optional[int] = None
    replica_axes: Optional[Tuple[str, ...]] = None
    replica_size: int = 1

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
    def world_size(self) -> int:
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


class ShardingSpec:
    """Algebraic partitioning description linking a stragety, mesh, and specific layout."""

    def __init__(
        self,
        strategy: PartitionStrategy,
        mesh: DeviceMesh,
        partition: PartitionSpec,
        routing_dim: Optional[int] = None,
    ):
        self.strategy = strategy
        self.mesh = mesh
        self.partition = partition
        self.routing_dim = routing_dim

        # Pre-compute immutable derived properties.
        used = set()
        for axes in self.partition.partitions.values():
            used.update(axes)

        all_axes = set(mesh.axis_names)
        repl = sorted(all_axes - used, key=lambda x: mesh.axis_names.index(x))
        self._replica_axes: Optional[List[str]] = repl if repl else None

        self._replica_size = 1
        if self._replica_axes:
            for ax in self._replica_axes:
                self._replica_size *= mesh.shape[mesh.axis_names.index(ax)]

    @property
    def partitions(self):
        return self.partition.partitions

    @property
    def is_content_addressable(self) -> bool:
        return self.strategy.routing_paradigm == RoutingParadigm.DYNAMIC

    @property
    def replica_axes(self) -> Optional[List[str]]:
        return self._replica_axes

    @property
    def replica_size(self) -> int:
        return self._replica_size

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

        # Infer routing_dim if not provided but topology requires it
        target_routing_dim = self.routing_dim
        if target_routing_dim is None and self.is_content_addressable:
            target_routing_dim = (
                0  # Graceful fallback for legacy codes that didn't provide arg
            )

        return Layout(
            data_shape=data_shape,
            mesh=self.mesh,
            topology=topology,
            partitions=self.partitions,
            strategy=self.strategy,
            routing_dim=target_routing_dim,
            replica_axes=self._replica_axes,
            replica_size=self._replica_size,
        )

    def get_expected_local_indices(
        self, rank: int, local_dim_size: int, dim: int = 0
    ) -> Optional[Any]:
        """Returns expected global indices for implicit mesh inference (chunk only)."""
        backend = get_backend()
        if not isinstance(self.strategy, ChunkPartition):
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

    # ---- Dependency ----

    @staticmethod
    def follow(source_layout: Layout, dim: int = 0) -> ShardingSpec:
        """Follow another Layout's partition strategy and mesh."""
        if source_layout.strategy is None:
            raise ValueError("Cannot follow a Layout without strategy")

        captured_meta = source_layout.topology
        captured_strategy = source_layout.strategy

        class _FollowPartition(PartitionStrategy):
            """Wraps a captured strategy with bound topology."""

            routing_paradigm = captured_strategy.routing_paradigm

            def __call__(self, indices, dim_size, world_size, topology=None):
                meta = captured_meta if captured_meta is not None else topology
                return captured_strategy(indices, dim_size, world_size, topology=meta)

            @property
            def name(self):
                return f"Follow({captured_strategy.name})"

        # Reconstruct PartitionSpec at the target dim (Note: Follow only tracks single dimension)
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
            f"partition={self.partition}",
            f"mesh={self.mesh.shape if self.mesh else None}",
        ]
        if self._replica_axes:
            parts.append(f"repl={self._replica_axes}")
        return f"ShardingSpec({', '.join(parts)})"
