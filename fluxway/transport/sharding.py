"""Algebraic Sharding — Layout as Function."""

from __future__ import annotations

from dataclasses import dataclass
import enum
from typing import Any, List, Optional, Tuple, Union

from fluxway.transport.backend import get_backend
from fluxway.transport.mesh import DeviceMesh
from fluxway.transport.topology import (
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
    def dim(self) -> int:
        for i, s in enumerate(self.specs):
            if s is not None:
                return i
        raise ValueError("PartitionSpec must have at least one non-None axis")

    @property
    def sharding_axes(self) -> Tuple[str, ...]:
        for s in self.specs:
            if s is not None:
                return (s,) if isinstance(s, str) else tuple(s)
        raise ValueError("PartitionSpec must have at least one non-None axis")

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
    """Data distribution description describing physical placement on a DeviceMesh.

    Combines the tensor characteristics (e.g. data shape, routing indices) with
    the distributed architecture details (mesh, strategy) to fully describe
    how data is or should be laid out across the cluster.
    """

    dim: int
    data_shape: Tuple[int, ...]
    mesh: DeviceMesh
    topology: Optional[Topology] = None
    sharding_axes: Optional[Tuple[str, ...]] = None
    strategy: Optional[PartitionStrategy] = None
    replication_axes: Optional[Tuple[str, ...]] = None
    replication_factor: int = 1

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

    @property
    def effective_world_size(self) -> int:
        if not self.sharding_axes:
            return self.mesh.size
        result = 1
        for ax in self.sharding_axes:
            result *= self.mesh.shape[self.mesh.axis_names.index(ax)]
        return result

    def get_effective_rank(self, physical_rank: int) -> int:
        if not self.sharding_axes:
            return physical_rank
        try:
            coords = self.mesh.get_coordinate(physical_rank)
        except ValueError:
            return physical_rank
        sub_coords = []
        sub_sizes = []
        for ax in self.sharding_axes:
            axis_idx = self.mesh.axis_names.index(ax)
            sub_coords.append(coords[axis_idx])
            sub_sizes.append(self.mesh.shape[axis_idx])
        return _ravel_multi_index(sub_coords, sub_sizes)

    def logical_to_physical(self, logical_ranks: Any) -> Any:
        """Map logical partition ranks to physical device ranks via mesh."""
        backend = get_backend()
        if not self.sharding_axes:
            return logical_ranks
        coords = [
            backend.zeros(backend.shape(logical_ranks), dtype=backend.int32())
            for _ in self.mesh.shape
        ]
        axis_name = self.sharding_axes[0]
        axis_idx = self.mesh.axis_names.index(axis_name)
        coords[axis_idx] = logical_ranks
        mesh_ids = backend.tensor(self.mesh.device_ids, dtype=backend.int32())
        return mesh_ids[tuple(coords)]

    def get_local_indices(self, rank: int) -> Any:
        """Returns the global indices held by this rank."""
        backend = get_backend()
        effective_rank = self.get_effective_rank(rank)
        ews = self.effective_world_size
        dim_size = self.data_shape[self.dim]

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
    """Algebraic partitioning description linking a stragety, mesh, and specific layout.

    This immutable value object contains all static information required to generate
    a concrete layout execution plan, reducing overhead during dynamic evaluations.
    """

    def __init__(
        self, strategy: PartitionStrategy, mesh: DeviceMesh, partition: PartitionSpec
    ):
        self.strategy = strategy
        self.mesh = mesh
        self.partition = partition

        # Pre-compute immutable derived properties.
        all_axes = set(mesh.axis_names)
        used = set(self.partition.sharding_axes)
        repl = sorted(all_axes - used, key=lambda x: mesh.axis_names.index(x))
        self._replication_axes: Optional[List[str]] = repl if repl else None

        self._replication_factor = 1
        if self._replication_axes:
            for ax in self._replication_axes:
                self._replication_factor *= mesh.shape[mesh.axis_names.index(ax)]

        # Effective world size (product of sharding axis sizes).
        self._effective_world_size = 1
        for ax in self.partition.sharding_axes:
            self._effective_world_size *= mesh.shape[mesh.axis_names.index(ax)]

    @property
    def dim(self) -> int:
        return self.partition.dim

    @property
    def sharding_axes(self) -> Tuple[str, ...]:
        return self.partition.sharding_axes

    @property
    def is_content_addressable(self) -> bool:
        return self.strategy.routing_paradigm == RoutingParadigm.DYNAMIC

    @property
    def replication_axes(self) -> Optional[List[str]]:
        return self._replication_axes

    @property
    def replication_factor(self) -> int:
        return self._replication_factor

    @property
    def effective_world_size(self) -> int:
        return self._effective_world_size

    def _to_layout(
        self,
        data_shape: Optional[Tuple[int, ...]] = None,
        topology: Optional[Topology] = None,
    ) -> Layout:
        """Internal: evaluate specification into a concrete Layout."""
        if data_shape is None:
            if not self.is_content_addressable:
                raise ValueError("data_shape is required for static sharding")
            data_shape = (-1,) * (self.dim + 1)

        return Layout(
            dim=self.dim,
            data_shape=data_shape,
            mesh=self.mesh,
            topology=topology,
            sharding_axes=self.sharding_axes,
            strategy=self.strategy,
            replication_axes=self._replication_axes,
            replication_factor=self._replication_factor,
        )

    def get_expected_local_indices(
        self, rank: int, local_dim_size: int
    ) -> Optional[Any]:
        """Returns expected global indices for implicit mesh inference (chunk only)."""
        backend = get_backend()
        if not isinstance(self.strategy, ChunkPartition):
            return None

        coords = self.mesh.get_coordinate(rank)
        sub_coords = []
        sub_sizes = []
        for ax in self.sharding_axes:
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
    def index(mesh: DeviceMesh, partition: PartitionSpec) -> ShardingSpec:
        """Dynamic: routing based on explicit array/tuple indices."""
        return ShardingSpec(_INDEXED, mesh=mesh, partition=partition)

    @staticmethod
    def hash(mesh: DeviceMesh, partition: PartitionSpec) -> ShardingSpec:
        """Dynamic: routing based on element values directly."""
        return ShardingSpec(_INDEXED, mesh=mesh, partition=partition)

    @staticmethod
    def greedy(mesh: DeviceMesh, partition: PartitionSpec) -> ShardingSpec:
        return ShardingSpec(_GREEDY, mesh=mesh, partition=partition)

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

        # Reconstruct PartitionSpec at the target dim
        specs = [None] * (dim + 1)
        axes = source_layout.sharding_axes
        if axes:
            specs[dim] = axes[0] if len(axes) == 1 else axes

        return ShardingSpec(
            _FollowPartition(), mesh=source_layout.mesh, partition=PartitionSpec(*specs)
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
            f"mesh={self.mesh.shape}",
        ]
        if self._replication_axes:
            parts.append(f"repl={self._replication_axes}")
        return f"ShardingSpec({', '.join(parts)})"
