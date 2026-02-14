from typing import List, Any
from fluxway.transport.backend import get_backend


class TransportTensor:
    """Unified tensor wrapper for distributed transport.

    Supports both dense (flat_values multi-dim, empty splits) and
    ragged (flat_values 1D, nested_splits struct) layouts.
    """

    def __init__(self, flat_values: Any, nested_splits: List[Any] = []):
        self._flat_values = flat_values
        self._nested_splits = nested_splits

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
            return TransportTensor(sliced_data, [])

        else:
            # Ragged case: requires processing nested splits.
            # We involve CPU synchronization (item()) to handle data-dependent shapes safely.

            curr_start_idx = start
            curr_size = size_tensor

            # Use tensor inputs for uniformity where possible
            if not backend.is_tensor(curr_start_idx):
                curr_start_idx = backend.tensor(curr_start_idx, dtype=backend.int32())
            if not backend.is_tensor(curr_size):
                curr_size = backend.tensor(curr_size, dtype=backend.int32())

            # Optimistically use dynamic_slice for splits if backend supports it efficiently
            for i in range(len(self._nested_splits)):
                curr_splits = self._nested_splits[i]

                # Slice current splits: [curr_start_idx : curr_start_idx + curr_size + 1]
                splits_slice = backend.dynamic_slice(
                    curr_splits, [curr_start_idx], [curr_size + 1]
                )

                # Bridge to legacy slice implementation for robustness
                s = int(backend.item(start)) if backend.is_tensor(start) else start
                e = (
                    int(backend.item(size_tensor))
                    if backend.is_tensor(size_tensor)
                    else size_tensor
                )

                return self.slice(s, s + e)

    def slice(self, start: int, end: int) -> "TransportTensor":
        """Slices the tensor along the first dimension."""
        backend = get_backend()

        if not self._nested_splits:
            # Dense Slice
            size = end - start
            return self.dynamic_slice(start, size)

        # Ragged implementation
        # Slicing along Level 0
        l0_splits = self._nested_splits[0]
        l0_len = backend.shape(l0_splits)[0] - 1

        start = max(0, start)
        end = min(end, l0_len)

        if start >= end:
            empty_splits = [
                backend.tensor([0], dtype=backend.int32()) for _ in self._nested_splits
            ]
            empty_values = self._flat_values[0:0]
            return TransportTensor(empty_values, empty_splits)

        # Slice L0 splits: [start : end + 1]
        spl_slice = backend.dynamic_slice(l0_splits, [start], [end - start + 1])
        offset = backend.take(spl_slice, backend.tensor(0, dtype=backend.int32()))
        new_l0_splits = spl_slice - offset

        new_nested_splits = [new_l0_splits]

        range_start = backend.take(
            l0_splits, backend.tensor(start, dtype=backend.int32())
        )
        range_end = backend.take(l0_splits, backend.tensor(end, dtype=backend.int32()))

        # For subsequent levels
        curr_start = range_start
        curr_end = range_end

        for i in range(1, len(self._nested_splits)):
            curr_splits = self._nested_splits[i]

            # Since ends/starts are tensors, we sync to host to determine exact slice size.
            s = int(backend.item(curr_start))
            e = int(backend.item(curr_end))

            # Slice this level splits
            size = e - s + 1
            spl_slice = backend.dynamic_slice(curr_splits, [s], [size])

            # Normalize
            off = backend.take(spl_slice, backend.tensor(0, dtype=backend.int32()))
            new_spl = spl_slice - off
            new_nested_splits.append(new_spl)

            # Update range
            curr_start = backend.take(
                curr_splits, backend.tensor(s, dtype=backend.int32())
            )
            curr_end = backend.take(
                curr_splits, backend.tensor(e, dtype=backend.int32())
            )

        # Slice Values
        s = int(backend.item(curr_start))
        e = int(backend.item(curr_end))
        val_size = e - s

        val_slice = backend.dynamic_slice(self._flat_values, [s], [val_size])

        return TransportTensor(val_slice, new_nested_splits)

    def gather(self, indices: Any) -> "TransportTensor":
        """Reorders the tensor along the first dimension according to indices."""
        backend = get_backend()

        if not self._nested_splits:
            # Dense case (can be multi-dimensional)
            gathered = backend.take(self._flat_values, indices)
            return TransportTensor(gathered, [])

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

        return TransportTensor(new_values, new_splits)

    def __reduce__(self):
        """Serialization interface for pickle/Ray."""
        return (TransportTensor, (self._flat_values, self._nested_splits))

    def __repr__(self):
        return f"TransportTensor(flat_values={self._flat_values}, splits={self._nested_splits})"
