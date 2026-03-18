from builtins import slice as python_slice
from dataclasses import dataclass
import enum
from typing import Any, List, Callable, Optional


from fluxway.common.backend import get_backend


class IndexMorphology(enum.Enum):
    """Describes the shape configuration of a distributed tensor."""

    CONTINUOUS = enum.auto()  # Regular dense multi-dimensional blocks
    RAGGED = enum.auto()  # Segmented blocks
    INDEXED = enum.auto()  # Explicitly indexed routing


class IndexDescriptor:
    """Base abstraction for tensor distribution metadata and layout mapping."""

    @property
    def morphology(self) -> IndexMorphology:
        raise NotImplementedError

    def materialize(self) -> Any:
        raise NotImplementedError

    def slice(self, start: Any, end: Any) -> "IndexDescriptor":
        raise NotImplementedError

    @staticmethod
    def concatenate(descriptors: List["IndexDescriptor"]) -> "IndexDescriptor":
        """Merge multiple index descriptors into one.

        Attempts O(1) metadata merging for compatible descriptors (e.g., adjacent chunks).
        Falls back to materialization and ExplicitIndex if merging is not possible.
        """
        if not descriptors:
            return ContinuousIndex.chunk(0, 0)

        # Optimization: If only one, return it
        if len(descriptors) == 1:
            return descriptors[0]

        # Optimization: Try to merge ContinuousIndex descriptors
        if all(isinstance(d, ContinuousIndex) for d in descriptors):
            first = descriptors[0]
            merged = first
            all_merged = True

            for next_d in descriptors[1:]:
                merged_candidate = merged._try_merge(next_d)
                if merged_candidate is None:
                    all_merged = False
                    break
                merged = merged_candidate

            if all_merged:
                return merged

        # Fallback: Materialize and concatenate
        backend = get_backend()
        materialized = [d.materialize() for d in descriptors]
        concatenated = backend.cat(materialized, dim=0)
        return ExplicitIndex(concatenated)

    # --- Arithmetic Magic Methods ---
    def __floordiv__(self, other):
        return self.materialize() // other

    def __mod__(self, other):
        return self.materialize() % other

    def __divmod__(self, other):
        return divmod(self.materialize(), other)

    def __truediv__(self, other):
        return self.materialize() / other

    def __mul__(self, other):
        return self.materialize() * other

    def __add__(self, other):
        return self.materialize() + other

    def __sub__(self, other):
        return self.materialize() - other

    def __radd__(self, other):
        return other + self.materialize()

    def __rsub__(self, other):
        return other - self.materialize()

    def __rmul__(self, other):
        return other * self.materialize()

    def __rfloordiv__(self, other):
        return other // self.materialize()

    def __rmod__(self, other):
        return other % self.materialize()

    def __eq__(self, other):
        if isinstance(other, IndexDescriptor):
            return bool(get_backend().equal(self.materialize(), other.materialize()))
        return bool(get_backend().equal(self.materialize(), other))


class IndexPattern(enum.Enum):
    CONTIGUOUS = enum.auto()
    STRIDED = enum.auto()


@dataclass(frozen=True)
class ContinuousIndex(IndexDescriptor):
    """Compact O(1) representation of regular index patterns.

    Provides a lightweight representation for indexing operations.
    By avoiding the materialization of full index tensors, it improves performance,
    especially for chunk and cyclic sharding patterns.
    """

    pattern: IndexPattern
    start: int = 0
    step: int = 1
    count: int = 0

    def to_compact_params(self) -> List[int]:
        """Calculate sharded parameters for compact transport."""
        return [
            int(self.start),
            int(self.step),
            int(self.count),
            int(self.pattern.value),
        ]

    @staticmethod
    def from_compact_params(p: List[int]) -> "ContinuousIndex":
        """Reconstruct from parameters."""
        return ContinuousIndex(
            pattern=IndexPattern(int(p[3])),
            start=int(p[0]),
            step=int(p[1]),
            count=int(p[2]),
        )

    @property
    def morphology(self) -> IndexMorphology:
        return IndexMorphology.CONTINUOUS

    def _try_merge(self, other: "ContinuousIndex") -> Optional["ContinuousIndex"]:
        """Attempt to merge with another descriptor. Returns new descriptor or None."""
        if self.pattern != other.pattern:
            return None

        if self.pattern == IndexPattern.CONTIGUOUS:
            # Check adjacency: next.start == current.range_end
            if other.start == self.start + self.count:
                return ContinuousIndex(
                    IndexPattern.CONTIGUOUS, self.start, 1, self.count + other.count
                )
        elif self.pattern == IndexPattern.STRIDED:
            # Check adjacency and step consistency
            if (
                self.step == other.step
                and other.start == self.start + self.count * self.step
            ):
                return ContinuousIndex(
                    IndexPattern.STRIDED,
                    self.start,
                    self.step,
                    self.count + other.count,
                )

        return None

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
                self.step,
                dtype=backend.int32(),
            )
        raise ValueError(f"Cannot materialize pattern={self.pattern}")

    def slice(self, start: Any, end: Any) -> "ContinuousIndex":
        """Derive descriptor for a sub-range. O(1) for chunk/cyclic."""
        backend = get_backend()
        s = int(backend.item(start)) if backend.is_tensor(start) else int(start)
        e = int(backend.item(end)) if backend.is_tensor(end) else int(end)
        size = e - s
        if self.pattern == IndexPattern.CONTIGUOUS:
            return ContinuousIndex(IndexPattern.CONTIGUOUS, self.start + s, 1, size)
        elif self.pattern == IndexPattern.STRIDED:
            return ContinuousIndex(
                IndexPattern.STRIDED, self.start + s * self.step, self.step, size
            )
        raise ValueError(f"Cannot slice pattern={self.pattern}")

    @staticmethod
    def chunk(start: int, count: int) -> "ContinuousIndex":
        return ContinuousIndex(
            IndexPattern.CONTIGUOUS, start=start, step=1, count=count
        )

    @staticmethod
    def cyclic(start: int, step: int, count: int) -> "ContinuousIndex":
        return ContinuousIndex(
            IndexPattern.STRIDED, start=start, step=step, count=count
        )

    def __eq__(self, other):
        if isinstance(other, ContinuousIndex):
            return (
                self.pattern == other.pattern
                and self.start == other.start
                and self.step == other.step
                and self.count == other.count
            )
        if isinstance(other, IndexDescriptor):
            return bool(get_backend().equal(self.materialize(), other.materialize()))
        return bool(get_backend().equal(self.materialize(), other))

    def __hash__(self):
        return hash((self.pattern, self.start, self.step, self.count))

    @staticmethod
    def from_slice(s: python_slice, size: int) -> "ContinuousIndex":
        start = s.start if s.start is not None else 0
        step = s.step if s.step is not None else 1
        stop = s.stop if s.stop is not None else size

        if step > 0:
            count = max(0, (stop - start + step - 1) // step)
        else:
            count = max(0, (start - stop - step - 1) // (-step))

        if step == 1:
            return ContinuousIndex(
                IndexPattern.CONTIGUOUS, start=start, step=1, count=count
            )
        else:
            return ContinuousIndex(
                IndexPattern.STRIDED, start=start, step=step, count=count
            )


@dataclass(frozen=True)
class RaggedIndex(IndexDescriptor):
    """IndexDescriptor describing segments of variable lengths via splits (formerly nested_splits)."""

    splits: List[Any]

    @property
    def morphology(self) -> IndexMorphology:
        return IndexMorphology.RAGGED

    def materialize(self) -> Any:
        raise NotImplementedError("RaggedIndex cannot be materialized directly.")

    def slice(self, start: Any, end: Any) -> "RaggedIndex":
        raise NotImplementedError(
            "RaggedIndex slice must be handled by TransportTensor."
        )


@dataclass(frozen=True)
class ExplicitIndex(IndexDescriptor):
    """IndexDescriptor describing an explicit tensor of routing mapping (formerly routing_indices/lookup table)."""

    indices: Any

    @property
    def morphology(self) -> IndexMorphology:
        return IndexMorphology.INDEXED

    def materialize(self) -> Any:
        return self.indices

    def slice(self, start: Any, end: Any) -> "ExplicitIndex":
        backend = get_backend()
        is_start_static = isinstance(start, int) or backend.is_scalar(start)
        is_end_static = isinstance(end, int) or backend.is_scalar(end)

        if is_start_static and is_end_static:
            s = int(start)
            e = int(end)
            return ExplicitIndex(self.indices[s:e])

        # Tensor path: Avoid JAX tracer concretization errors
        idx_range = backend.arange(start, end, dtype=backend.int32())
        sliced_indices = backend.take(self.indices, idx_range)
        return ExplicitIndex(sliced_indices)

    def __eq__(self, other):
        bk = get_backend()
        if isinstance(other, ExplicitIndex):
            return bool(bk.equal(self.indices, other.indices))
        if isinstance(other, IndexDescriptor):
            return bool(bk.equal(self.indices, other.materialize()))
        return bool(bk.equal(self.indices, other))

    def __hash__(self):
        return id(self)


@dataclass(frozen=True)
class FunctionalIndex(IndexDescriptor):
    """IndexDescriptor describing a dynamic functional mapping for hashing (formerly hash_fn dict)."""

    hash_fn: Callable

    @property
    def morphology(self) -> IndexMorphology:
        return IndexMorphology.INDEXED

    def materialize(self) -> Any:
        raise NotImplementedError(
            "FunctionalIndex cannot be materialized without input."
        )

    def slice(self, start: Any, end: Any) -> "FunctionalIndex":
        raise NotImplementedError("FunctionalIndex cannot be sliced directly.")


try:
    import jax

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

if _JAX_AVAILABLE:
    jax.tree_util.register_pytree_node(
        ContinuousIndex,
        lambda x: ((x.start, x.step, x.count), {"pattern": x.pattern}),
        lambda aux, children: ContinuousIndex(
            pattern=aux["pattern"],
            start=children[0],
            step=children[1],
            count=children[2],
        ),
    )
    jax.tree_util.register_pytree_node(
        RaggedIndex,
        lambda x: ((x.splits,), None),
        lambda _, children: RaggedIndex(splits=children[0]),
    )
    jax.tree_util.register_pytree_node(
        ExplicitIndex,
        lambda x: ((x.indices,), None),
        lambda _, children: ExplicitIndex(indices=children[0]),
    )
