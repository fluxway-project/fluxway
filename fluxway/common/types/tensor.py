from typing import List, Any, Optional, Tuple
from fluxway.common.backend import get_backend
from fluxway.common.topology import ContinuousTopology, Topology, IndexedTopology

# ============================================================
# TransportTensor — Unified tensor wrapper
# ============================================================


class TransportTensor:
    """Unified tensor wrapper for distributed data transport operations.

    This class abstracts the representation of both dense and ragged tensor layouts.
    Dense tensors are represented via multi-dimensional `flat_values` and empty splits,
    while ragged tensors use 1D `flat_values` accompanied by a `nested_splits` structure.
    """

    def __init__(
        self,
        flat_values: Any,
        nested_splits: Optional[List[Any]] = None,
        indices: Optional[Any] = None,
        global_shape: Optional[Tuple[int, ...]] = None,
        origin_ranks: Optional[Any] = None,
        origin_indices: Optional[Any] = None,
    ):
        self._flat_values = flat_values
        self._nested_splits = nested_splits if nested_splits is not None else []
        self._global_shape = global_shape
        self._origin_ranks = origin_ranks
        self._origin_indices = origin_indices

        self._indices: Optional[Topology]
        if isinstance(indices, ContinuousTopology):
            self._indices = indices
        elif isinstance(indices, slice):
            backend = get_backend()
            if not self._nested_splits:
                if isinstance(self._flat_values, list):
                    count = len(self._flat_values)
                else:
                    count = int(backend.shape(self._flat_values)[0])
            else:
                if isinstance(self._nested_splits[0], list):
                    count = len(self._nested_splits[0]) - 1
                else:
                    count = int(backend.shape(self._nested_splits[0])[0]) - 1

            start = indices.start if indices.start is not None else 0
            stride = indices.step if indices.step is not None else 1
            if indices.stop is not None:
                expected_count = (
                    max(0, (indices.stop - start + stride - 1) // stride)
                    if stride > 0
                    else max(0, (start - indices.stop - stride - 1) // (-stride))
                )
                if expected_count != count:
                    raise ValueError(
                        f"Slice implies count={expected_count}, but tensor has local count={count}"
                    )

            if stride == 1:
                self._indices = ContinuousTopology.chunk(start, count)
            else:
                self._indices = ContinuousTopology.cyclic(start, stride, count)
        elif indices is not None:
            if isinstance(indices, Topology):
                self._indices = indices
            else:
                self._indices = IndexedTopology(indices)
        else:
            self._indices = None

    @classmethod
    def from_tensor(
        cls,
        tensor: Any,
        indices: Optional[Any] = None,
        global_shape: Optional[Tuple[int, ...]] = None,
    ) -> "TransportTensor":
        """Creates a dense TransportTensor directly from a multi-dimensional tensor.

        Args:
            tensor: The dense data tensor.
            indices: Global indices mapping. Can be a materialized tensor, an explicit
                     ContinuousTopology, or a python `slice` object for O(1) regular patterns.
            global_shape: Optional global shape of the full tensor.
        """
        return cls(
            flat_values=tensor,
            nested_splits=[],
            indices=indices,
            global_shape=global_shape,
        )

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
    def indices(self) -> Optional[Any]:
        """Underlying topology indices object."""
        return self._indices

    @property
    def index_descriptor(self) -> Optional[ContinuousTopology]:
        """Compact index pattern descriptor, if available."""
        if isinstance(self._indices, ContinuousTopology):
            return self._indices
        return None

    @property
    def global_indices(self) -> Optional[Any]:
        """Global indices this local data corresponds to.

        Lazy-materializes from descriptor if no explicit tensor is set.
        Returns None if neither is available (implicit/zero-cost).
        """
        from fluxway.common.topology import IndexedTopology, ContinuousTopology

        if self._indices is None:
            return None
        if isinstance(self._indices, IndexedTopology):
            return self._indices.indices
        if isinstance(self._indices, ContinuousTopology):
            return self._indices.materialize()
        return None

    @property
    def global_shape(self) -> Optional[Tuple[int, ...]]:
        """Global shape of the full tensor across all ranks."""
        return self._global_shape

    @property
    def shape(self) -> Tuple[int, ...]:
        """Local shape of the data shard."""
        backend = get_backend()
        return backend.shape(self._flat_values)

    @property
    def dtype(self) -> Any:
        """Data type of the tensor."""
        backend = get_backend()
        return backend.dtype(self._flat_values)

    @property
    def topology(self) -> Optional[List[Any]]:
        """Topology metadata derived from nested_splits. None for dense."""
        return self._nested_splits if self._nested_splits else None

    @property
    def origin_ranks(self) -> Optional[Any]:
        return self._origin_ranks

    @property
    def origin_indices(self) -> Optional[Any]:
        return self._origin_indices

    def unique_with_routing(self) -> Tuple["TransportTensor", "TransportTensor"]:
        """
        Performs unique on flat_values.
        Returns (compacted_tensor, inverse_indices_tensor).
        The compacted_tensor has origin_indices automatically set to allow O(1) restoration.
        """
        backend = get_backend()
        uniques, inverse = backend.unique(self.flat_values, return_inverse=True)

        origin_indices = backend.arange(
            0, backend.shape(uniques)[0], dtype=backend.int32()
        )

        compacted = TransportTensor(
            flat_values=uniques,
            indices=uniques,  # Use values as global routing keys
            origin_indices=origin_indices,
            nested_splits=[],
        )

        # inverse_indices must retain the original nested_splits (ragged structure)
        inverse_tt = TransportTensor(
            flat_values=inverse, nested_splits=self.nested_splits
        )

        return compacted, inverse_tt

    @classmethod
    def from_lists(cls, data: List[Any], dtype=None) -> "TransportTensor":
        """Creates a TransportTensor from a nested list.

        This constructor automatically inspects the input structure. If the data is
        uniformly shaped, it optimizes the representation by creating a dense tensor.
        Otherwise, it constructs a ragged tensor with computed splits.
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

    def _propagate_indices_for_dense_slice(self, start: Any, size: Any) -> Any:
        """Derive new index info for a dense slice. Returns new Topology or None."""
        backend = get_backend()
        from fluxway.common.topology import ContinuousTopology, IndexedTopology

        if self._indices is None:
            return None

        # Prefer descriptor path (O(1))
        if isinstance(self._indices, ContinuousTopology):
            if backend.is_tensor(start):
                s = int(backend.item(start))
            else:
                s = int(start)
            if backend.is_tensor(size):
                sz = int(backend.item(size))
            else:
                sz = int(size)
            return self._indices.slice(s, s + sz)

        # Explicit tensor path
        if isinstance(self._indices, IndexedTopology):
            new_indices = backend.dynamic_slice(self._indices.indices, [start], [size])
            return IndexedTopology(new_indices)

        return None

    def dynamic_slice(
        self, start: Any, size_tensor: Any, unsafe: bool = False
    ) -> "TransportTensor":
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

            new_indices = self._propagate_indices_for_dense_slice(start, size_tensor)
            return TransportTensor(sliced_data, [], indices=new_indices)

        else:
            # Ragged case: requires processing nested splits.
            # We can now use slice() directly with tensors, as it avoids synchronization.
            return self.slice(start, start + size_tensor, unsafe=unsafe)

    def slice(
        self,
        start: Any,
        end: Any,
        static_shape: Optional[Any] = None,
        unsafe: bool = False,
    ) -> "TransportTensor":
        """Slices the tensor along the first dimension.

        Args:
            start: Start index.
            end: End index.
            static_shape: Optional static shape constraint for JIT/vmap.
                          Can be an int (dim0 length) or a tuple (dim0 length, Values length, ...).
                          If tuple, enables JIT safe slicing for ragged tensors by padding values.
            unsafe: If True, skips bounds checking and assumes indices are valid and within bounds.
                    Enable this only when indices are guaranteed correct (e.g. from planner).
        """
        backend = get_backend()

        # Check for integer/scalar optimization path
        is_start_int = isinstance(start, int) or backend.is_scalar(start)
        is_end_int = isinstance(end, int) or backend.is_scalar(end)

        # Parse static shape
        dim0_static = None
        val_static = None
        if static_shape is not None:
            if isinstance(static_shape, (tuple, list)):
                dim0_static = static_shape[0]
                if len(static_shape) > 1:
                    val_static = static_shape[
                        -1
                    ]  # Simple assumption: (dim0, Vals) or (dim0, L1, Vals) - take last as Vals
            else:
                dim0_static = static_shape

        # Input normalization for Tensor operations
        # If unsafe is True and inputs are ints, we keep them as ints for native slicing
        if not unsafe:
            if not backend.is_tensor(start):
                start = backend.tensor(start, dtype=backend.int32())
            if not backend.is_tensor(end):
                end = backend.tensor(end, dtype=backend.int32())

        if not self._nested_splits:
            # Dense Slice
            size = end - start
            return self.dynamic_slice(start, size, unsafe=unsafe)

        # Ragged implementation
        # Slicing along Level 0
        dim0_splits = self._nested_splits[0]

        # Optimization: Use native slicing for integers if unsafe or backend supports it
        # This is CRITICAL for Eager mode performance to avoid JIT/Graph tracing overhead.
        if unsafe and is_start_int and is_end_int:
            # Fast path for known valid integer indices
            s = int(start)
            e = int(end)

            # Slice dim0 splits directly
            # For ragged, we need splits[start : end + 1]
            spl_slice = dim0_splits[s : e + 1]

            # Normalize splits (subtract offset)
            offset = spl_slice[0]
            new_dim0_splits = spl_slice - offset

            new_nested_splits = [new_dim0_splits]

            # Determined range for next level
            # range_start = dim0_splits[s] -> which is offset (already fetched)
            range_start = offset
            range_end = dim0_splits[e]

            curr_start = range_start
            curr_end = range_end

            # Loop for nested levels
            for i in range(1, len(self._nested_splits)):
                curr_splits = self._nested_splits[i]

                # For next levels, use dynamic slice logic but try to keep it cheap
                count = curr_end - curr_start + 1
                new_spl = backend.dynamic_slice(curr_splits, curr_start, count)

                # Normalize
                off = backend.take(new_spl, backend.tensor(0, dtype=backend.int32()))
                new_spl = new_spl - off

                new_nested_splits.append(new_spl)

                # Update next range
                curr_start = backend.take(curr_splits, curr_start)
                curr_end = backend.take(curr_splits, curr_end)

            # Slice Values
            val_count = curr_end - curr_start
            val_slice = backend.dynamic_slice(self._flat_values, curr_start, val_count)

            # Propagate indices
            new_indices = None
            if self._indices is not None:
                from fluxway.common.topology import ContinuousTopology, IndexedTopology

                if isinstance(self._indices, ContinuousTopology):
                    new_indices = self._indices.slice(s, e)
                elif isinstance(self._indices, IndexedTopology):
                    # Fast slice for indices
                    sliced_idx = self._indices.indices[s:e]
                    new_indices = IndexedTopology(sliced_idx)

            return TransportTensor(val_slice, new_nested_splits, indices=new_indices)

        # -------------------------------------------------------------------------
        # Standard / JIT-Safe Path
        # -------------------------------------------------------------------------

        # Make sure inputs are tensors for safe path
        if not backend.is_tensor(start):
            start = backend.tensor(start, dtype=backend.int32())
        if not backend.is_tensor(end):
            end = backend.tensor(end, dtype=backend.int32())

        dim0_len = backend.shape(dim0_splits)[0] - 1
        dim0_len_t = backend.tensor(dim0_len, dtype=backend.int32())
        zero_t = backend.tensor(0, dtype=backend.int32())

        # Clamp start/end without CPU sync (only if not unsafe)
        # If unsafe=True, we rely on caller guarantee and skip clamping to avoid kernels.

        if not unsafe:
            start = backend.where(start < zero_t, zero_t, start)
            start = backend.where(start > dim0_len_t, dim0_len_t, start)

        if static_shape is None:
            if not unsafe:
                end = backend.where(end > dim0_len_t, dim0_len_t, end)
                end = backend.where(end < start, start, end)

            # Dynamic shape path (using arange)
            dim0_indices = backend.arange(start, end + 1, dtype=backend.int32())
            spl_slice = backend.take(dim0_splits, dim0_indices)
        else:
            # Static shape path (JIT compatible)
            # Mask values beyond end using static_shape
            count = dim0_static + 1 if dim0_static is not None else dim0_len + 1
            base = backend.arange(0, count, dtype=backend.int32())
            dim0_indices = base + start

            # Clamp indices to valid range for 'take' safety
            safe_indices = backend.where(
                dim0_indices > dim0_len_t, dim0_len_t, dim0_indices
            )
            safe_indices = backend.where(safe_indices < zero_t, zero_t, safe_indices)

            spl_slice = backend.take(dim0_splits, safe_indices)

        offset = backend.take(spl_slice, zero_t)
        new_dim0_splits = spl_slice - offset

        new_nested_splits = [new_dim0_splits]

        # Determine range for next level
        range_start = backend.take(dim0_splits, start)
        range_end = backend.take(dim0_splits, end)

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
        if self._indices is not None:
            from fluxway.common.topology import ContinuousTopology, IndexedTopology

            if isinstance(self._indices, ContinuousTopology):
                new_indices = self._indices.slice(start, end)
            elif isinstance(self._indices, IndexedTopology):
                if static_shape is None:
                    idx_range = backend.arange(start, end, dtype=backend.int32())
                    idx_val = backend.take(self._indices.indices, idx_range)
                    new_indices = IndexedTopology(idx_val)

        return TransportTensor(val_slice, new_nested_splits, indices=new_indices)

    def as_dlpack_slice(self, cursor: Optional[Any] = None) -> Any:
        """Returns a zero-copy DLPack-compatible slice of the tensor.

        Args:
             cursor: Slice argument (e.g. length of first dimension)
        """
        backend = get_backend()
        return backend.get_dlpack_slice(self._flat_values, cursor)

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
                new_values, self._nested_splits, indices=self._indices
            )
        return self

    def gather(self, indices: Any) -> "TransportTensor":
        """Reorders the tensor along the first dimension according to indices."""
        backend = get_backend()

        idx_flat = (
            indices.flat_values if isinstance(indices, TransportTensor) else indices
        )

        # Propagate global_indices through gather
        new_global_indices = None
        src_indices = self.global_indices  # Materializes descriptor if needed
        if src_indices is not None:
            from fluxway.common.topology import IndexedTopology

            gathered_indices = backend.take(src_indices, idx_flat)
            new_global_indices = IndexedTopology(gathered_indices)
        # Descriptor cannot survive gather (order destroyed), so always explicit tensor

        if not self._nested_splits:
            # Dense case (can be multi-dimensional)
            gathered = backend.take(self._flat_values, idx_flat)
            # If indices were ragged, the result should have the same ragged structure
            new_splits = (
                indices.nested_splits if isinstance(indices, TransportTensor) else []
            )
            return TransportTensor(gathered, new_splits, indices=new_global_indices)

        current_indices = idx_flat
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

        return TransportTensor(new_values, new_splits, indices=new_global_indices)

    def __reduce__(self):
        """Serialization interface for pickle/Ray."""
        return (
            _reconstruct_transport_tensor,
            (
                self._flat_values,
                self._nested_splits,
                self._indices,
                self._origin_ranks,
                self._origin_indices,
            ),
        )

    def __repr__(self):
        parts = [f"flat_values={self._flat_values}", f"splits={self._nested_splits}"]
        if self._indices is not None:
            parts.append(f"indices={self._indices}")
        if self._origin_ranks is not None:
            parts.append(f"origin_ranks={self._origin_ranks}")
        if self._origin_indices is not None:
            parts.append(f"origin_indices={self._origin_indices}")
        return f"TransportTensor({', '.join(parts)})"


def _reconstruct_transport_tensor(
    flat_values, nested_splits, indices, origin_ranks=None, origin_indices=None
):
    """Pickle reconstruction helper (top-level function for pickling)."""
    return TransportTensor(
        flat_values,
        nested_splits,
        indices=indices,
        origin_ranks=origin_ranks,
        origin_indices=origin_indices,
    )


try:
    import jax

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

if _JAX_AVAILABLE:

    def _tt_flatten(v):
        children = (
            v._flat_values,
            v._nested_splits,
            v._indices,
            v._origin_ranks,
            v._origin_indices,
        )
        return children, None

    def _tt_unflatten(aux_data, children):
        flat_values, nested_splits, indices, origin_ranks, origin_indices = children
        return TransportTensor(
            flat_values,
            nested_splits,
            indices=indices,
            origin_ranks=origin_ranks,
            origin_indices=origin_indices,
        )

    jax.tree_util.register_pytree_node(TransportTensor, _tt_flatten, _tt_unflatten)
