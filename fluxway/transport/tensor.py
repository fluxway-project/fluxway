from dataclasses import dataclass
from typing import List, Any, Optional
from fluxway.transport.backend import get_backend


# ============================================================
# IndexDescriptor — Compact index pattern representation
# ============================================================


@dataclass(frozen=True)
class IndexDescriptor:
    """Compact O(1) representation of regular index patterns.

    Avoids materializing full index tensors for chunk/cyclic sharding.
    """

    kind: str  # "chunk" | "cyclic"
    start: int = 0
    step: int = 1
    count: int = 0

    def materialize(self) -> Any:
        """Expand to a full index tensor (lazy evaluation endpoint)."""
        backend = get_backend()
        if self.kind == "chunk":
            return backend.arange(
                self.start, self.start + self.count, dtype=backend.int32()
            )
        elif self.kind == "cyclic":
            return backend.arange(
                self.start,
                self.start + self.step * self.count,
                step=self.step,
                dtype=backend.int32(),
            )
        raise ValueError(f"Cannot materialize kind={self.kind}")

    def slice(self, start: int, end: int) -> "IndexDescriptor":
        """Derive descriptor for a sub-range. O(1) for chunk/cyclic."""
        size = end - start
        if self.kind == "chunk":
            return IndexDescriptor("chunk", self.start + start, 1, size)
        elif self.kind == "cyclic":
            return IndexDescriptor(
                "cyclic", self.start + start * self.step, self.step, size
            )
        raise ValueError(f"Cannot slice kind={self.kind}")

    @staticmethod
    def chunk(start: int, count: int) -> "IndexDescriptor":
        return IndexDescriptor("chunk", start=start, step=1, count=count)

    @staticmethod
    def cyclic(start: int, step: int, count: int) -> "IndexDescriptor":
        return IndexDescriptor("cyclic", start=start, step=step, count=count)


# ============================================================
# TransportTensor — Unified tensor wrapper
# ============================================================

try:
    import jax

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


class TransportTensor:
    """Unified tensor wrapper for distributed transport.

    Supports both dense (flat_values multi-dim, empty splits) and
    ragged (flat_values 1D, nested_splits struct) layouts.
    """

    def __init__(
        self,
        flat_values: Any,
        nested_splits: Optional[List[Any]] = None,
        global_indices: Any = None,
        index_descriptor: Optional[IndexDescriptor] = None,
    ):
        self._flat_values = flat_values
        self._nested_splits = nested_splits if nested_splits is not None else []
        self._global_indices = global_indices  # Explicit tensor, or None
        self._index_descriptor = index_descriptor  # Compact O(1) representation

    @property
    def flat_values(self) -> Any:
        return self._flat_values

    @property
    def nested_splits(self) -> List[Any]:
        return self._nested_splits

    @property
    def is_dense(self) -> bool:
        """Returns True if this tensor is dense (no ragged dimensions)."""
        return len(self._nested_splits) == 0

    @property
    def is_ragged(self) -> bool:
        """Returns True if this tensor has ragged dimensions."""
        return len(self._nested_splits) > 0

    @property
    def index_descriptor(self) -> Optional[IndexDescriptor]:
        """Compact index pattern descriptor, if available."""
        return self._index_descriptor

    @property
    def global_indices(self) -> Optional[Any]:
        """Global indices this local data corresponds to.

        Lazy-materializes from descriptor if no explicit tensor is set.
        Returns None if neither is available (implicit/zero-cost).
        """
        if self._global_indices is not None:
            return self._global_indices
        if self._index_descriptor is not None:
            return self._index_descriptor.materialize()
        return None

    @property
    def topology(self) -> Optional[List[Any]]:
        """Topology metadata derived from nested_splits. None for dense."""
        return self._nested_splits if self._nested_splits else None

    @classmethod
    def from_lists(cls, data: List[Any], dtype=None) -> "TransportTensor":
        """
        Creates a TransportTensor from a nested list.
        Automatically optimizes uniform-shaped data as dense tensors.
        """
        backend = get_backend()

        # Helper function to check if data has uniform shape
        def _is_uniform_shape(lst):
            if not isinstance(lst, list) or len(lst) == 0:
                return True, []

            if not isinstance(lst[0], list):
                return all(not isinstance(x, list) for x in lst), [len(lst)]

            first_len = len(lst[0])
            if not all(
                isinstance(item, list) and len(item) == first_len for item in lst
            ):
                return False, []

            is_uniform, child_shape = _is_uniform_shape(lst[0])
            if not is_uniform:
                return False, []

            return True, [len(lst)] + child_shape

        # Check if data has uniform shape
        is_uniform, shape = _is_uniform_shape(data)

        if is_uniform and len(shape) > 0:
            # Dense optimization: create multidimensional tensor directly
            if dtype is None:
                dtype = backend.float32()
            dense_tensor = backend.tensor(data, dtype=dtype)
            return cls(dense_tensor, [])

        # Ragged tensor: use iterative flattening
        current_level_lists = data
        all_splits = []

        while True:
            # Check if next level is lists
            if not current_level_lists or not isinstance(current_level_lists[0], list):
                break

            if len(current_level_lists) > 0 and not isinstance(
                current_level_lists[0], list
            ):
                break

            # Flatten one level
            next_level_lists = []
            current_splits = [0]
            offset = 0

            for item in current_level_lists:
                if not isinstance(item, list):
                    next_level_lists.append(item)
                    offset += 1
                else:
                    offset += len(item)
                    next_level_lists.extend(item)
                current_splits.append(offset)

            splits_tensor = backend.tensor(current_splits, dtype=backend.int32())
            all_splits.append(splits_tensor)

            current_level_lists = next_level_lists

            if len(current_level_lists) > 0 and not isinstance(
                current_level_lists[0], list
            ):
                break

        if dtype is None:
            dtype = backend.float32()

        flat_tensor = backend.tensor(current_level_lists, dtype=dtype)
        return cls(flat_tensor, all_splits)

    def to_lists(self) -> List[Any]:
        """Converts back to a nested list."""
        backend = get_backend()

        if not self._nested_splits:
            return backend.to_numpy(self._flat_values).tolist()

        def build_list(level, start_idx, end_idx):
            if level == len(self._nested_splits):
                chunk = self._flat_values[start_idx:end_idx]
                return backend.to_numpy(chunk).tolist()

            splits = self._nested_splits[level]

            res = []
            for i in range(start_idx, end_idx):
                start = backend.item(splits[i])
                end = backend.item(splits[i + 1])

                child_list = build_list(level + 1, start, end)
                res.append(child_list)
            return res

        top_split = self._nested_splits[0]
        # Ensure we handle case where split might be empty or 1-element (empty tensor)
        if backend.shape(top_split)[0] == 0:
            return []
        num_splits = backend.shape(top_split)[0]

        return build_list(0, 0, num_splits - 1)

    def _propagate_indices_for_dense_slice(self, start: Any, size: Any) -> tuple:
        """Derive new index info for a dense slice. Returns (global_indices, descriptor)."""
        backend = get_backend()

        # Prefer descriptor path (O(1))
        if self._index_descriptor is not None:
            if backend.is_tensor(start):
                s = int(backend.item(start))
            else:
                s = int(start)
            if backend.is_tensor(size):
                sz = int(backend.item(size))
            else:
                sz = int(size)
            new_desc = self._index_descriptor.slice(s, s + sz)
            return None, new_desc

        # Explicit tensor path
        if self._global_indices is not None:
            new_indices = backend.dynamic_slice(self._global_indices, [start], [size])
            return new_indices, None

        return None, None

    def dynamic_slice(self, start: Any, size_tensor: Any) -> "TransportTensor":
        """Slice with tensor arguments using zero-sync backend primitives."""
        backend = get_backend()

        if not self._nested_splits:
            # Dense case
            x = self._flat_values
            full_shape = backend.shape(x)
            ndim = len(full_shape)

            starts = [start] + [0] * (ndim - 1)
            sizes = [size_tensor] + list(full_shape[1:])

            sliced_data = backend.dynamic_slice(x, starts, sizes)

            new_indices, new_desc = self._propagate_indices_for_dense_slice(
                start, size_tensor
            )
            return TransportTensor(
                sliced_data, [], global_indices=new_indices, index_descriptor=new_desc
            )

        else:
            # Ragged case: requires processing nested splits.
            # We can now use slice() directly with tensors, as it avoids synchronization.
            return self.slice(start, start + size_tensor)

    def slice(
        self, start: Any, end: Any, static_shape: Optional[Any] = None
    ) -> "TransportTensor":
        """Slices the tensor along the first dimension.

        Args:
            start: Start index.
            end: End index.
            static_shape: Optional static shape constraint for JIT/vmap.
                          Can be an int (L0 length) or a tuple (L0 length, Values length, ...).
                          If tuple, enables JIT safe slicing for ragged tensors by padding values.
        """
        backend = get_backend()

        # Parse static shape
        l0_static = None
        val_static = None
        if static_shape is not None:
            if isinstance(static_shape, (tuple, list)):
                l0_static = static_shape[0]
                if len(static_shape) > 1:
                    val_static = static_shape[
                        -1
                    ]  # Simple assumption: (L0, Vals) or (L0, L1, Vals) - take last as Vals
            else:
                l0_static = static_shape

        # Input normalization for Tensor operations
        if not backend.is_tensor(start):
            start = backend.tensor(start, dtype=backend.int32())
        if not backend.is_tensor(end):
            end = backend.tensor(end, dtype=backend.int32())

        if not self._nested_splits:
            # Dense Slice
            size = end - start
            return self.dynamic_slice(start, size)

        # Ragged implementation
        # Slicing along Level 0
        l0_splits = self._nested_splits[0]
        l0_len = backend.shape(l0_splits)[0] - 1
        l0_len_t = backend.tensor(l0_len, dtype=backend.int32())
        zero_t = backend.tensor(0, dtype=backend.int32())

        # Clamp start/end without CPU sync
        start = backend.where(start < zero_t, zero_t, start)
        start = backend.where(start > l0_len_t, l0_len_t, start)

        if static_shape is None:
            end = backend.where(end > l0_len_t, l0_len_t, end)
            end = backend.where(end < start, start, end)

            # Dynamic shape path
            l0_indices = backend.arange(start, end + 1, dtype=backend.int32())
            spl_slice = backend.take(l0_splits, l0_indices)
        else:
            # Static shape path (JIT compatible)
            # Mask values beyond end using static_shape
            count = l0_static + 1
            base = backend.arange(0, count, dtype=backend.int32())
            l0_indices = base + start

            # Clamp indices to valid range for 'take' safety
            safe_indices = backend.where(l0_indices > l0_len_t, l0_len_t, l0_indices)
            safe_indices = backend.where(safe_indices < zero_t, zero_t, safe_indices)

            spl_slice = backend.take(l0_splits, safe_indices)

        offset = backend.take(spl_slice, zero_t)
        new_l0_splits = spl_slice - offset

        new_nested_splits = [new_l0_splits]

        # Determine range for next level
        range_start = backend.take(l0_splits, start)
        range_end = backend.take(l0_splits, end)

        curr_start = range_start
        curr_end = range_end

        for i in range(1, len(self._nested_splits)):
            curr_splits = self._nested_splits[i]

            # Current level range: [curr_start : curr_end + 1]
            s = curr_start
            e = curr_end

            # Use gather instead of dynamic_slice to avoid CPU sync
            idx = backend.arange(s, e + 1, dtype=backend.int32())
            spl_slice = backend.take(curr_splits, idx)

            # Normalize
            off = backend.take(spl_slice, zero_t)
            new_spl = spl_slice - off
            new_nested_splits.append(new_spl)

            # Update range for next level
            curr_start = backend.take(curr_splits, s)
            curr_end = backend.take(curr_splits, e)

        # Slice Values
        # Slice Values
        if val_static is not None:
            # Static Value Slice
            count = val_static
            base_idx = backend.arange(0, count, dtype=backend.int32())
            val_idx = base_idx + curr_start

            total_val_len = backend.shape(self._flat_values)[0]
            tvl_t = backend.tensor(total_val_len, dtype=backend.int32())

            # Safe clamping strategy for JIT
            safe_idx = backend.where(val_idx >= curr_end, curr_end - 1, val_idx)
            safe_idx = backend.where(safe_idx >= tvl_t, tvl_t - 1, safe_idx)
            safe_idx = backend.where(safe_idx < zero_t, zero_t, safe_idx)

            val_slice = backend.take(self._flat_values, safe_idx)
        else:
            val_idx = backend.arange(curr_start, curr_end, dtype=backend.int32())
            val_slice = backend.take(self._flat_values, val_idx)

        # Propagate indices for ragged slice
        new_indices = None
        new_desc = None
        if self._index_descriptor is not None:
            new_desc = self._index_descriptor.slice(start, end)
        elif self._global_indices is not None:
            # If static_shape is present, we should avoid dynamic arange.
            # But simple restore first.
            if static_shape is None:
                idx_range = backend.arange(start, end, dtype=backend.int32())
                new_indices = backend.take(self._global_indices, idx_range)

        return TransportTensor(
            val_slice,
            new_nested_splits,
            global_indices=new_indices,
            index_descriptor=new_desc,
        )

    def compact(self) -> "TransportTensor":
        """Trims internal padding to match actual data size.

        This operations performs a data copy (dynamic_slice) to return a clean, minimal tensor.
        Useful before network transmission to avoid sending zero-padding.
        """
        backend = get_backend()

        if not self._nested_splits:
            return self

        last_splits = self._nested_splits[-1]
        # Check last element of splits. If masked/static sliced, the last element might be huge or repeated.
        # But we rely on the implementation of slice that preserves the last valid offset.
        total_elements = backend.take(
            last_splits, backend.tensor(-1, dtype=backend.int32())
        )
        current_val_size = backend.shape(self._flat_values)[0]

        # Optimize: if total_elements is small and buffer huge
        should_compact = backend.item(total_elements) < current_val_size

        if should_compact:
            # Note: dynamic_slice requires static start? [0] is static.
            # This method is Eager-only safe.

            ndim = len(backend.shape(self._flat_values))

            starts = [0] * ndim
            sizes = [total_elements] + list(backend.shape(self._flat_values)[1:])

            new_values = backend.dynamic_slice(self._flat_values, starts, sizes)
            return TransportTensor(
                new_values,
                self._nested_splits,
                global_indices=self._global_indices,
                index_descriptor=self._index_descriptor,
            )
        return self

    def gather(self, indices: Any) -> "TransportTensor":
        """Reorders the tensor along the first dimension according to indices."""
        backend = get_backend()

        # Propagate global_indices through gather
        new_global_indices = None
        src_indices = self.global_indices  # Materializes descriptor if needed
        if src_indices is not None:
            new_global_indices = backend.take(src_indices, indices)
        # Descriptor cannot survive gather (order destroyed), so always explicit tensor

        if not self._nested_splits:
            # Dense case (can be multi-dimensional)
            gathered = backend.take(self._flat_values, indices)
            return TransportTensor(gathered, [], global_indices=new_global_indices)

        current_indices = indices
        new_splits = []

        for level in range(len(self._nested_splits)):
            splits = self._nested_splits[level]
            # Rely on python slicing which usually delegates to __getitem__
            lengths = splits[1:] - splits[:-1]

            # Gather lengths based on current level permutation
            gathered_lengths = backend.take(lengths, current_indices)

            # New splits: [0, cumsum(gathered_lengths)]
            # 1. cumsum
            cs = backend.cumsum(gathered_lengths, dim=0)
            # 2. prepend 0. Need generic zero tensor + concatenation
            zero = backend.zeros((1,), dtype=backend.int32())

            new_level_split = backend.cat([zero, cs], dim=0)
            new_splits.append(new_level_split)

            # Compute next level indices
            # We need to generate the range of indices for the next level.
            # starts = splits[:-1][current_indices]
            starts = backend.take(splits[:-1], current_indices)

            # We need to generate: [starts[0]...starts[0]+len[0], starts[1]...starts[1]+len[1], ...]
            # Vectorized "repeat and increment"

            # counts = gathered_lengths
            total_elements = backend.sum(gathered_lengths)

            # If total is 0, handle empty case
            if backend.item(total_elements) == 0:
                current_indices = backend.zeros((0,), dtype=backend.int32())
                continue

            # bases = repeat(starts, counts)
            bases = backend.repeat_interleave(starts, gathered_lengths, dim=0)

            # offsets = global_seq - adjustments
            c = backend.cumsum(gathered_lengths, dim=0)
            group_starts = c - gathered_lengths

            adjustments = backend.repeat_interleave(
                group_starts, gathered_lengths, dim=0
            )

            global_seq = backend.arange(
                0, backend.item(total_elements), dtype=backend.int32()
            )

            offsets = global_seq - adjustments

            next_indices = bases + offsets
            current_indices = next_indices

        # Finally gather flat values
        new_values = backend.take(self._flat_values, current_indices)

        return TransportTensor(
            new_values, new_splits, global_indices=new_global_indices
        )

    def __reduce__(self):
        """Serialization interface for pickle/Ray."""
        return (
            _reconstruct_transport_tensor,
            (
                self._flat_values,
                self._nested_splits,
                self._global_indices,
                self._index_descriptor,
            ),
        )

    def __repr__(self):
        parts = [f"flat_values={self._flat_values}", f"splits={self._nested_splits}"]
        if self._global_indices is not None:
            parts.append(f"global_indices={self._global_indices}")
        if self._index_descriptor is not None:
            parts.append(f"descriptor={self._index_descriptor}")
        return f"TransportTensor({', '.join(parts)})"


def _reconstruct_transport_tensor(
    flat_values, nested_splits, global_indices, index_descriptor
):
    """Pickle reconstruction helper (top-level function for pickling)."""
    return TransportTensor(
        flat_values,
        nested_splits,
        global_indices=global_indices,
        index_descriptor=index_descriptor,
    )


if _JAX_AVAILABLE:

    def _tt_flatten(v):
        children = (v._flat_values, v._nested_splits, v._global_indices)
        aux_data = (v._index_descriptor,)
        return children, aux_data

    def _tt_unflatten(aux_data, children):
        flat_values, nested_splits, global_indices = children
        return TransportTensor(
            flat_values,
            nested_splits,
            global_indices=global_indices,
            index_descriptor=aux_data[0],
        )

    jax.tree_util.register_pytree_node(TransportTensor, _tt_flatten, _tt_unflatten)
