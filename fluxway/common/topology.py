from dataclasses import dataclass
import enum
from typing import Any, List, Callable
from fluxway.common.backend import get_backend


class TopologyMorphology(enum.Enum):
    """Describes the shape configuration of a distributed tensor."""

    CONTINUOUS = enum.auto()  # Regular dense multi-dimensional blocks
    RAGGED = enum.auto()  # Segmented blocks
    INDEXED = enum.auto()  # Explicitly indexed routing


class Topology:
    """Base abstraction for tensor distribution metadata and layout mapping."""

    @property
    def morphology(self) -> TopologyMorphology:
        raise NotImplementedError


class IndexPattern(enum.Enum):
    CONTIGUOUS = enum.auto()
    STRIDED = enum.auto()


@dataclass(frozen=True)
class ContinuousTopology(Topology):
    """Compact O(1) representation of regular index patterns (formerly IndexDescriptor).

    Provides a lightweight representation for indexing operations.
    By avoiding the materialization of full index tensors, it improves performance,
    especially for chunk and cyclic sharding patterns.
    """

    pattern: IndexPattern
    start: int = 0
    step: int = 1
    count: int = 0

    @property
    def morphology(self) -> TopologyMorphology:
        return TopologyMorphology.CONTINUOUS

    def materialize(self) -> Any:
        """Expand to a full index tensor (lazy evaluation endpoint)."""
        backend = get_backend()
        if self.pattern == IndexPattern.CONTIGUOUS:
            return backend.arange(
                self.start, self.start + self.count, dtype=backend.int32()
            )
        elif self.pattern == IndexPattern.STRIDED:
            return backend.arange(
                self.start,
                self.start + self.step * self.count,
                step=self.step,
                dtype=backend.int32(),
            )
        raise ValueError(f"Cannot materialize pattern={self.pattern}")

    def slice(self, start: int, end: int) -> "ContinuousTopology":
        """Derive descriptor for a sub-range. O(1) for chunk/cyclic."""
        size = end - start
        if self.pattern == IndexPattern.CONTIGUOUS:
            return ContinuousTopology(
                IndexPattern.CONTIGUOUS, self.start + start, 1, size
            )
        elif self.pattern == IndexPattern.STRIDED:
            return ContinuousTopology(
                IndexPattern.STRIDED, self.start + start * self.step, self.step, size
            )
        raise ValueError(f"Cannot slice pattern={self.pattern}")

    @staticmethod
    def chunk(start: int, count: int) -> "ContinuousTopology":
        return ContinuousTopology(
            IndexPattern.CONTIGUOUS, start=start, step=1, count=count
        )

    @staticmethod
    def cyclic(start: int, step: int, count: int) -> "ContinuousTopology":
        return ContinuousTopology(
            IndexPattern.STRIDED, start=start, step=step, count=count
        )

    @staticmethod
    def from_slice(s: slice, size: int) -> "ContinuousTopology":
        start = s.start if s.start is not None else 0
        step = s.step if s.step is not None else 1
        stop = s.stop if s.stop is not None else size

        if step > 0:
            count = max(0, (stop - start + step - 1) // step)
        else:
            count = max(0, (start - stop - step - 1) // (-step))

        if step == 1:
            return ContinuousTopology(
                IndexPattern.CONTIGUOUS, start=start, step=1, count=count
            )
        else:
            return ContinuousTopology(
                IndexPattern.STRIDED, start=start, step=step, count=count
            )


@dataclass(frozen=True)
class RaggedTopology(Topology):
    """Topology describing segments of variable lengths via splits (formerly nested_splits)."""

    splits: List[Any]

    @property
    def morphology(self) -> TopologyMorphology:
        return TopologyMorphology.RAGGED


@dataclass(frozen=True)
class IndexedTopology(Topology):
    """Topology describing an explicit tensor of routing mapping (formerly routing_indices/lookup table)."""

    indices: Any

    @property
    def morphology(self) -> TopologyMorphology:
        return TopologyMorphology.INDEXED


@dataclass(frozen=True)
class CallableTopology(Topology):
    """Topology describing a dynamic functional mapping for hashing (formerly hash_fn dict)."""

    hash_fn: Callable

    @property
    def morphology(self) -> TopologyMorphology:
        return TopologyMorphology.INDEXED


try:
    import jax

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

if _JAX_AVAILABLE:
    jax.tree_util.register_pytree_node(
        ContinuousTopology,
        lambda x: (
            (),
            {"pattern": x.pattern, "start": x.start, "step": x.step, "count": x.count},
        ),
        lambda static, aux: ContinuousTopology(
            pattern=static["pattern"],
            start=static["start"],
            step=static["step"],
            count=static["count"],
        ),
    )
    jax.tree_util.register_pytree_node(
        RaggedTopology,
        lambda x: ((x.splits,), None),
        lambda _, children: RaggedTopology(splits=children[0]),
    )
    jax.tree_util.register_pytree_node(
        IndexedTopology,
        lambda x: ((x.indices,), None),
        lambda _, children: IndexedTopology(indices=children[0]),
    )
