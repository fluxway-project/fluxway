from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
from numba import njit, prange

from fluxway.common.index_descriptor import (
    ContinuousIndex,
    ExplicitIndex,
    IndexDescriptor,
    IndexPattern,
)
from fluxway.common.coherence import PhysicalMutability
from fluxway.common.numba_runtime import call_numba_cached
from fluxway.common.tensor_backend import force_writable
from fluxway.common.types.sharding import (
    IdentityIndices,
    IndicesSpec,
    LiteralIndices,
    RangeIndices,
)


@dataclass(frozen=True, slots=True)
class TensorPayload:
    values: Any
    physical_mutability: PhysicalMutability

    @classmethod
    def create(cls, values: Any) -> "TensorPayload":
        flags = getattr(values, "flags", None)
        if flags is None or not hasattr(flags, "writeable"):
            mutability = PhysicalMutability.UNKNOWN
        elif flags.writeable:
            mutability = PhysicalMutability.PROVEN_WRITABLE
        else:
            mutability = PhysicalMutability.READONLY
        return cls(values, mutability)


@dataclass(frozen=True, slots=True)
class TensorLayout:
    nested_splits: tuple[Any, ...]
    global_shape: tuple[int, ...] | None
    routing_axes: tuple[int, ...]

    @classmethod
    def create(
        cls,
        nested_splits: list[Any] | tuple[Any, ...] | None,
        global_shape: tuple[int, ...] | None,
        routing_axes: tuple[int, ...],
    ) -> "TensorLayout":
        axes = tuple(axis for axis in routing_axes)
        if len(axes) != len(set(axes)):
            raise ValueError(f"routing_axes must be unique, got {axes}")
        splits = tuple(force_writable(split) for split in (nested_splits or ()))
        shape = tuple(dim for dim in global_shape) if global_shape is not None else None
        return cls(splits, shape, axes)

    @property
    def is_dense(self) -> bool:
        return len(self.nested_splits) == 0

    @property
    def is_ragged(self) -> bool:
        return len(self.nested_splits) > 0

    @property
    def ragged_level_count(self) -> int:
        return len(self.nested_splits)

    @property
    def primary_routing_axis(self) -> int:
        if not self.routing_axes:
            raise ValueError("TensorLayout.routing_axes must be non-empty.")
        return self.routing_axes[0]


@dataclass(slots=True)
class TensorRouting:
    indices: IndexDescriptor | None
    materialized_cache: Any | None = None

    @classmethod
    def create(cls, indices: Any | None) -> "TensorRouting":
        return cls(_as_index_descriptor(indices))

    def materialize(self) -> Any | None:
        if self.indices is None:
            return None
        if self.materialized_cache is not None:
            return self.materialized_cache
        if isinstance(self.indices, ExplicitIndex):
            self.materialized_cache = self.indices.indices
        elif isinstance(self.indices, ContinuousIndex):
            self.materialized_cache = self.indices.materialize()
        else:
            self.materialized_cache = self.indices
        return self.materialized_cache


@dataclass(slots=True)
class TensorLineage:
    origin: TensorRouting
    receipt: Any | None = None


def routing_entity_count(
    payload: TensorPayload, layout: TensorLayout, backend: Any
) -> int:
    if layout.is_ragged:
        first_splits = layout.nested_splits[0]
        first_shape = backend.shape(first_splits)
        if len(first_shape) != 1:
            raise ValueError(
                f"Ragged nested_splits[0] must be 1-D, got shape={first_shape}."
            )
        return int(first_shape[0]) - 1

    payload_shape = backend.shape(payload.values)
    if len(payload_shape) == 0:
        return 1
    axis = layout.primary_routing_axis
    if axis < 0 or axis >= len(payload_shape):
        raise ValueError(
            f"routing axis {axis} is outside payload shape {payload_shape}."
        )
    return int(payload_shape[axis])


def index_descriptor_length(
    indices: IndexDescriptor | None, backend: Any
) -> int | None:
    if indices is None:
        return None
    if isinstance(indices, ContinuousIndex):
        return indices.count
    if isinstance(indices, ExplicitIndex):
        shape = backend.shape(indices.indices)
        if len(shape) != 1:
            raise ValueError(f"ExplicitIndex must be 1-D, got shape={shape}.")
        return int(shape[0])
    raise TypeError(
        f"{type(indices).__name__} does not expose an O(1) finite length for "
        "TransportTensor row metadata."
    )


def normalize_index_descriptor(
    indices: Any | None, count: int
) -> IndexDescriptor | None:
    if indices is None:
        return None
    if isinstance(indices, slice):
        start = 0 if indices.start is None else int(indices.start)
        step = 1 if indices.step is None else int(indices.step)
        if step == 0:
            raise ValueError("slice step must be non-zero.")
        if indices.stop is not None:
            expected = len(range(start, int(indices.stop), step))
            if expected != count:
                raise ValueError(
                    f"slice implies count={expected}, but tensor routing entity count is {count}."
                )
        if step == 1:
            return ContinuousIndex.chunk(start, count)
        return ContinuousIndex.cyclic(start, step, count)
    return _as_index_descriptor(indices)


def validate_dense_global_shape(
    payload: TensorPayload,
    layout: TensorLayout,
    backend: Any,
) -> None:
    if layout.global_shape is None or layout.is_ragged:
        return
    actual_shape = backend.shape(payload.values)
    if len(actual_shape) != len(layout.global_shape):
        raise ValueError(
            f"[Fluxway Validation Error] Dimensionality mismatch: "
            f"flat_values shape={actual_shape} has rank {len(actual_shape)}, "
            f"but global_shape={layout.global_shape} has rank {len(layout.global_shape)}."
        )


def validate_metadata_lengths(
    payload: TensorPayload,
    layout: TensorLayout,
    routing: TensorRouting,
    lineage: TensorLineage,
    backend: Any,
) -> None:
    count = routing_entity_count(payload, layout, backend)
    _validate_one_metadata_length("routing_indices", routing.indices, count, backend)
    _validate_one_metadata_length(
        "origin_indices", lineage.origin.indices, count, backend
    )


def _validate_one_metadata_length(
    name: str,
    indices: IndexDescriptor | None,
    count: int,
    backend: Any,
) -> None:
    length = index_descriptor_length(indices, backend)
    if length is not None and length != count:
        raise ValueError(
            f"[Fluxway TransportTensor Error] {name} length {length} does not match "
            f"tensor routing entity count {count}."
        )


def descriptor_to_indices_spec(
    indices: IndexDescriptor,
    *,
    length: int,
    valid_length: int | None,
    backend: Any,
) -> IndicesSpec:
    descriptor_count = index_descriptor_length(indices, backend)
    expected_valid_length = length if valid_length is None else valid_length
    if descriptor_count != expected_valid_length:
        raise ValueError(
            f"{type(indices).__name__} count {descriptor_count} does not match "
            f"valid routing length {expected_valid_length}."
        )
    if isinstance(indices, ContinuousIndex):
        if indices.pattern == IndexPattern.CONTIGUOUS:
            return RangeIndices.contiguous(
                start=indices.start,
                length=length,
                valid_length=valid_length,
            )
        if indices.pattern == IndexPattern.STRIDED:
            return RangeIndices.strided(
                start=indices.start,
                length=length,
                stride=indices.step,
                valid_length=valid_length,
            )
        raise ValueError(f"Unsupported ContinuousIndex pattern={indices.pattern}.")
    if isinstance(indices, ExplicitIndex):
        values = padded_literal_values(
            indices.indices,
            length=length,
            valid_length=valid_length,
            backend=backend,
        )
        return LiteralIndices(length=length, values=values, valid_length=valid_length)
    raise TypeError(
        f"{type(indices).__name__} cannot be converted to an IndicesSpec without "
        "materializing user-defined logic."
    )


def index_source_to_indices_spec(
    source: Any,
    *,
    length: int,
    valid_length: int | None,
    backend: Any,
) -> IndicesSpec:
    if isinstance(source, IndicesSpec):
        return _resize_indices_spec(source, length=length, valid_length=valid_length)
    if isinstance(source, IndexDescriptor):
        return descriptor_to_indices_spec(
            source,
            length=length,
            valid_length=valid_length,
            backend=backend,
        )
    values = padded_literal_values(
        source,
        length=length,
        valid_length=valid_length,
        backend=backend,
    )
    return LiteralIndices(length=length, values=values, valid_length=valid_length)


def padded_literal_values(
    values: Any,
    *,
    length: int,
    valid_length: int | None,
    backend: Any,
) -> Any:
    tensor = backend.asarray(values, copy=True)
    shape = backend.shape(tensor)
    if len(shape) != 1:
        raise ValueError(f"Literal routing values must be 1-D, got shape={shape}.")
    actual = int(shape[0])
    expected_valid_length = length if valid_length is None else valid_length
    if actual != expected_valid_length:
        raise ValueError(
            f"Literal routing values length {actual} does not match valid routing "
            f"length {expected_valid_length}."
        )
    if actual == length:
        return tensor
    if actual > length:
        raise ValueError(
            f"Literal routing values length {actual} exceeds routing length {length}."
        )
    padding = backend.full((length - actual,), -1, dtype=backend.dtype(tensor))
    return backend.cat([tensor, padding], dim=0)


def _resize_indices_spec(
    spec: IndicesSpec,
    *,
    length: int,
    valid_length: int | None,
) -> IndicesSpec:
    expected_valid_length = length if valid_length is None else valid_length
    if spec.length != expected_valid_length:
        raise ValueError(
            f"{type(spec).__name__} length {spec.length} does not match valid routing "
            f"length {expected_valid_length}."
        )
    if isinstance(spec, IdentityIndices):
        return IdentityIndices(length=length, valid_length=valid_length)
    if isinstance(spec, RangeIndices):
        return RangeIndices.strided(
            start=spec.start,
            length=length,
            stride=spec.stride,
            valid_length=valid_length,
        )
    if isinstance(spec, LiteralIndices):
        raise ValueError(
            "LiteralIndices cannot be resized without an explicit backend-padded values tensor."
        )
    raise TypeError(f"Unsupported IndicesSpec type {type(spec).__name__}.")


def select_source_geometry_index_source(
    *,
    routing_indices: IndexDescriptor | None,
    origin_indices: IndexDescriptor | None,
    source_is_content_addressable: bool,
    target_is_content_addressable: bool,
) -> tuple[Any | None, str]:
    if source_is_content_addressable and not target_is_content_addressable:
        if routing_indices is not None:
            raise ValueError(
                "content-addressable source to static target must use origin_indices "
                "as restore coordinates; routing_indices would be ambiguous."
            )
        if origin_indices is None:
            raise ValueError(
                "content-addressable source to static target requires origin_indices "
                "restore coordinates."
            )
        return origin_indices, "origin_indices"
    if routing_indices is not None:
        return routing_indices, "routing_indices"
    return None, "source_layout"


def select_destination_content_index_source(
    *,
    routing_indices: IndexDescriptor | None,
    flat_values: Any,
) -> Any:
    if routing_indices is not None:
        return routing_indices
    return flat_values


def _as_index_descriptor(indices: Any | None) -> IndexDescriptor | None:
    if indices is None:
        return None
    if isinstance(indices, IndexDescriptor):
        return indices
    return ExplicitIndex(indices)


@dataclass(frozen=True, slots=True)
class NestedSparseLayout:
    flat_values: Any
    nested_splits: tuple[np.ndarray, ...]
    rows: int
    widths: tuple[int, ...]

    @property
    def global_tail_shape(self) -> tuple[int, ...]:
        return self.widths


def row_partition_sparse_feature(
    field_name: str, tensor: Any, dtype: Any = None
) -> NestedSparseLayout:
    from fluxway.common.types.tensor import TransportTensor

    if isinstance(tensor, TransportTensor):
        if not tensor.nested_splits:
            raise ValueError(
                f"Sparse TransportTensor field {field_name!r} must carry nested_splits."
            )
        nested_splits: list[np.ndarray] = []
        widths: list[int] = []
        first_splits = np.asarray(tensor.nested_splits[0], dtype=np.int32)
        rows = first_splits.size - 1
        for splits_obj in tensor.nested_splits:
            splits = np.asarray(splits_obj, dtype=np.int32)
            _validate_splits(field_name, splits)
            lengths = splits[1:] - splits[:-1]
            width = lengths.max() if lengths.size else 0
            nested_splits.append(splits)
            widths.append(width)
        return NestedSparseLayout(
            tensor.flat_values,
            tuple(nested_splits),
            rows,
            tuple(widths),
        )

    ids = np.asarray(tensor, dtype=dtype)
    if ids.ndim != 2:
        raise ValueError(
            f"Fixed-width sparse field {field_name!r} must be a 2D array before "
            f"Reader packaging, got shape={ids.shape}."
        )
    rows, width = int(ids.shape[0]), int(ids.shape[1])
    if rows <= 0 or width <= 0:
        raise ValueError(
            f"Fixed-width sparse field {field_name!r} must have positive rows and width, "
            f"got shape={ids.shape}."
        )
    splits = np.arange(0, ids.size + 1, width, dtype=np.int32)
    return NestedSparseLayout(ids.reshape(-1), (splits,), rows, (width,))


def nested_path_position_maps(
    nested_splits: tuple[np.ndarray, ...] | list[np.ndarray],
    flat_first_indices: np.ndarray,
    widths: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if len(nested_splits) != len(widths):
        raise ValueError(
            f"nested_splits levels {len(nested_splits)} != dense widths {len(widths)}."
        )
    if not widths:
        raise ValueError("Sparse nested path requires at least one width.")

    leaf_indices = np.asarray(flat_first_indices, dtype=np.int64)
    path_offsets = np.empty((leaf_indices.size, len(widths)), dtype=np.int32)
    if leaf_indices.size == 0:
        for level, splits_obj in enumerate(nested_splits):
            splits = np.asarray(splits_obj, dtype=np.int64)
            _validate_splits(f"nested level {level}", splits)
            width = widths[level]
            if width < 0:
                raise ValueError(
                    f"Dense width at level {level} must be non-negative, got {width}."
                )
            lengths = splits[1:] - splits[:-1]
            local_max_width = lengths.max() if lengths.size else 0
            if local_max_width > width:
                raise ValueError(
                    f"Nested level {level} local width {local_max_width} exceeds dense width {width}."
                )
        return np.empty((0,), dtype=np.int32), path_offsets
    child_indices = leaf_indices

    for level in range(len(widths) - 1, -1, -1):
        splits = np.asarray(nested_splits[level], dtype=np.int64)
        _validate_splits(f"nested level {level}", splits)
        lengths = splits[1:] - splits[:-1]
        width = widths[level]
        if width <= 0:
            raise ValueError(
                f"Dense width at level {level} must be positive, got {width}."
            )
        local_max_width = lengths.max() if lengths.size else 0
        if local_max_width > width:
            raise ValueError(
                f"Nested level {level} local width {local_max_width} exceeds dense width {width}."
            )
        parent_indices = np.searchsorted(splits, child_indices, side="right") - 1
        if np.any(parent_indices < 0) or np.any(parent_indices >= lengths.size):
            raise ValueError(
                f"Flat indices are outside nested splits at level {level}: "
                f"indices={child_indices.tolist()}, splits={splits.tolist()}"
            )
        path_offsets[:, level] = (child_indices - splits[parent_indices]).astype(
            np.int32
        )
        child_indices = parent_indices.astype(np.int64)

    return child_indices.astype(np.int32), path_offsets


def dense_path_strides(widths: tuple[int, ...]) -> np.ndarray:
    strides = np.empty(len(widths) + 1, dtype=np.int64)
    stride = 1
    for i in range(len(widths) - 1, -1, -1):
        strides[i + 1] = stride
        stride *= widths[i]
    strides[0] = stride
    return strides


def _numba_kernel(**options: Any):
    def decorate(function):
        dispatcher = njit(**options)(function)

        @wraps(function)
        def call(*args):
            return call_numba_cached(dispatcher, *args)

        return call

    return decorate


@_numba_kernel(parallel=True, nogil=True, cache=True)
def nested_global_positions_kernel(
    local_rows: np.ndarray,
    path_offsets: np.ndarray,
    origin_indices: np.ndarray,
    strides: np.ndarray,
    out: np.ndarray,
) -> None:
    for i in prange(local_rows.shape[0]):  # type: ignore
        slot = origin_indices[local_rows[i]] * strides[0]
        for level in range(path_offsets.shape[1]):
            slot += path_offsets[i, level] * strides[level + 1]
        out[i] = slot


@_numba_kernel(parallel=True, nogil=True, cache=True)
def scatter_inv_local_kernel(
    global_positions: np.ndarray,
    inv_1d: np.ndarray,
    global_start: int,
    out: np.ndarray,
) -> None:
    for i in prange(inv_1d.shape[0]):  # type: ignore
        out[i] = global_positions[inv_1d[i]] - global_start


_MAX_NESTED_LEVELS = 16


@njit(nogil=True)
def _walk_forward_tree_pad(
    b: int,
    nested_splits: tuple[np.ndarray, ...],
    strides: np.ndarray,
    widths: np.ndarray,
    flat_to_unique_local: np.ndarray,
    pad_index: int,
    out_flat: np.ndarray,
) -> None:
    levels = len(nested_splits)
    stride0 = strides[0]
    base0 = b * stride0

    for slot in range(stride0):
        out_flat[base0 + slot] = pad_index

    if levels == 1:
        start = nested_splits[0][b]
        end = nested_splits[0][b + 1]
        length = end - start
        width = widths[0]
        limit = length if length < width else width
        stride1 = strides[1]
        for i in range(limit):
            out_flat[base0 + i * stride1] = flat_to_unique_local[start + i]
        return

    node_idx = np.zeros(_MAX_NESTED_LEVELS, dtype=np.int64)
    node_end = np.zeros(_MAX_NESTED_LEVELS, dtype=np.int64)
    local_idx = np.zeros(_MAX_NESTED_LEVELS, dtype=np.int64)
    offset_base = np.zeros(_MAX_NESTED_LEVELS, dtype=np.int64)

    offset_base[0] = base0
    node_idx[0] = b
    node_end[0] = b + 1

    node_idx[1] = nested_splits[0][b]
    node_end[1] = nested_splits[0][b + 1]
    local_idx[1] = 0
    offset_base[1] = base0

    curr_level = 1
    while curr_level > 0:
        if (
            node_idx[curr_level] < node_end[curr_level]
            and local_idx[curr_level] < widths[curr_level - 1]
        ):
            curr_node = node_idx[curr_level]
            curr_local = local_idx[curr_level]
            curr_offset = offset_base[curr_level] + curr_local * strides[curr_level]

            if curr_level == levels - 1:
                leaf_start = nested_splits[curr_level][curr_node]
                leaf_end = nested_splits[curr_level][curr_node + 1]
                leaf_len = leaf_end - leaf_start
                leaf_stride = strides[curr_level + 1]
                w_leaf = widths[curr_level]
                limit = leaf_len if leaf_len < w_leaf else w_leaf
                for i in range(limit):
                    out_flat[curr_offset + i * leaf_stride] = flat_to_unique_local[
                        leaf_start + i
                    ]
                node_idx[curr_level] += 1
                local_idx[curr_level] += 1
            else:
                next_level = curr_level + 1
                node_idx[next_level] = nested_splits[curr_level][curr_node]
                node_end[next_level] = nested_splits[curr_level][curr_node + 1]
                local_idx[next_level] = 0
                offset_base[next_level] = curr_offset
                curr_level = next_level
        else:
            curr_level -= 1
            if curr_level > 0:
                node_idx[curr_level] += 1
                local_idx[curr_level] += 1


@njit(nogil=True)
def _walk_forward_tree_slot_holes(
    b: int,
    nested_splits: tuple[np.ndarray, ...],
    strides: np.ndarray,
    widths: np.ndarray,
    flat_to_unique_local: np.ndarray,
    out_flat: np.ndarray,
) -> None:
    levels = len(nested_splits)
    stride0 = strides[0]
    base0 = b * stride0

    for slot in range(stride0):
        out_flat[base0 + slot] = base0 + slot

    if levels == 1:
        start = nested_splits[0][b]
        end = nested_splits[0][b + 1]
        length = end - start
        width = widths[0]
        limit = length if length < width else width
        stride1 = strides[1]
        for i in range(limit):
            out_flat[base0 + i * stride1] = flat_to_unique_local[start + i]
        return

    node_idx = np.zeros(_MAX_NESTED_LEVELS, dtype=np.int64)
    node_end = np.zeros(_MAX_NESTED_LEVELS, dtype=np.int64)
    local_idx = np.zeros(_MAX_NESTED_LEVELS, dtype=np.int64)
    offset_base = np.zeros(_MAX_NESTED_LEVELS, dtype=np.int64)

    offset_base[0] = base0
    node_idx[0] = b
    node_end[0] = b + 1

    node_idx[1] = nested_splits[0][b]
    node_end[1] = nested_splits[0][b + 1]
    local_idx[1] = 0
    offset_base[1] = base0

    curr_level = 1
    while curr_level > 0:
        if (
            node_idx[curr_level] < node_end[curr_level]
            and local_idx[curr_level] < widths[curr_level - 1]
        ):
            curr_node = node_idx[curr_level]
            curr_local = local_idx[curr_level]
            curr_offset = offset_base[curr_level] + curr_local * strides[curr_level]

            if curr_level == levels - 1:
                leaf_start = nested_splits[curr_level][curr_node]
                leaf_end = nested_splits[curr_level][curr_node + 1]
                leaf_len = leaf_end - leaf_start
                leaf_stride = strides[curr_level + 1]
                w_leaf = widths[curr_level]
                limit = leaf_len if leaf_len < w_leaf else w_leaf
                for i in range(limit):
                    out_flat[curr_offset + i * leaf_stride] = flat_to_unique_local[
                        leaf_start + i
                    ]
                node_idx[curr_level] += 1
                local_idx[curr_level] += 1
            else:
                next_level = curr_level + 1
                node_idx[next_level] = nested_splits[curr_level][curr_node]
                node_end[next_level] = nested_splits[curr_level][curr_node + 1]
                local_idx[next_level] = 0
                offset_base[next_level] = curr_offset
                curr_level = next_level
        else:
            curr_level -= 1
            if curr_level > 0:
                node_idx[curr_level] += 1
                local_idx[curr_level] += 1


@_numba_kernel(parallel=True, nogil=True, cache=True)
def _fill_nested_dense_inv_kernel(
    flat_to_unique_local: np.ndarray,
    nested_splits: tuple[np.ndarray, ...],
    strides: np.ndarray,
    widths: np.ndarray,
    out: np.ndarray,
) -> None:
    batch_size = nested_splits[0].shape[0] - 1
    out_flat = out.ravel()
    for b in prange(batch_size):  # type: ignore
        _walk_forward_tree_slot_holes(
            b,
            nested_splits,
            strides,
            widths,
            flat_to_unique_local,
            out_flat,
        )


def fill_nested_dense_inv_kernel(
    flat_to_unique_local: np.ndarray,
    nested_splits: tuple[np.ndarray, ...],
    strides: np.ndarray,
    out: np.ndarray,
) -> None:
    widths = np.asarray(out.shape[1:], dtype=np.int64)
    _fill_nested_dense_inv_kernel(
        flat_to_unique_local,
        nested_splits,
        strides,
        widths,
        out,
    )


@_numba_kernel(parallel=True, nogil=True, cache=True)
def _fill_nested_dense_inv_with_pad_kernel(
    flat_to_unique_local: np.ndarray,
    nested_splits: tuple[np.ndarray, ...],
    strides: np.ndarray,
    widths: np.ndarray,
    pad_index: int,
    out: np.ndarray,
) -> None:
    batch_size = nested_splits[0].shape[0] - 1
    out_flat = out.ravel()
    for b in prange(batch_size):  # type: ignore
        _walk_forward_tree_pad(
            b,
            nested_splits,
            strides,
            widths,
            flat_to_unique_local,
            pad_index,
            out_flat,
        )


def fill_nested_dense_inv_with_pad_kernel(
    flat_to_unique_local: np.ndarray,
    nested_splits: tuple[np.ndarray, ...],
    strides: np.ndarray,
    pad_index: int,
    out: np.ndarray,
) -> None:
    widths = np.asarray(out.shape[1:], dtype=np.int64)
    _fill_nested_dense_inv_with_pad_kernel(
        flat_to_unique_local,
        nested_splits,
        strides,
        widths,
        pad_index,
        out,
    )


@_numba_kernel(parallel=True, nogil=True, cache=True)
def fill_single_level_dense_inv_with_pad_kernel(
    flat_to_unique_local: np.ndarray,
    splits: np.ndarray,
    pad_index: int,
    out: np.ndarray,
) -> None:
    rows = splits.shape[0] - 1
    width = out.shape[1]
    for row in prange(rows):  # type: ignore
        start = splits[row]
        end = splits[row + 1]
        length = end - start
        for slot in range(width):
            if slot < length:
                out[row, slot] = flat_to_unique_local[start + slot]
            else:
                out[row, slot] = pad_index


def fill_nested_dense_inv_with_pad(
    flat_to_unique_local: np.ndarray,
    nested_splits: tuple[np.ndarray, ...],
    strides: np.ndarray,
    pad_index: int,
    out: np.ndarray,
) -> None:
    if not nested_splits:
        raise ValueError("nested_splits must contain at least one level.")
    if len(nested_splits) > _MAX_NESTED_LEVELS:
        raise ValueError(
            f"nested_splits depth {len(nested_splits)} exceeds maximum "
            f"supported depth {_MAX_NESTED_LEVELS}."
        )
    if len(nested_splits) == 1 and out.ndim == 2:
        splits = nested_splits[0]
        if splits.ndim != 1:
            raise ValueError(f"single-level splits must be 1-D, got {splits.shape}.")
        if splits.size != out.shape[0] + 1:
            raise ValueError(
                f"single-level splits length {splits.size} does not match "
                f"out rows {out.shape[0]}."
            )
        lengths = splits[1:] - splits[:-1]
        if lengths.size and int(lengths.max()) > int(out.shape[1]):
            raise ValueError(
                f"single-level sparse row width {int(lengths.max())} exceeds "
                f"dense width {int(out.shape[1])}."
            )
        fill_single_level_dense_inv_with_pad_kernel(
            flat_to_unique_local,
            splits,
            pad_index,
            out,
        )
        return
    fill_nested_dense_inv_with_pad_kernel(
        flat_to_unique_local,
        nested_splits,
        strides,
        pad_index,
        out,
    )


def _validate_splits(field_name: str, splits: np.ndarray) -> None:
    if splits.ndim != 1 or splits.size < 1:
        raise ValueError(
            f"Sparse field {field_name!r} has invalid splits shape={splits.shape}."
        )
    if int(splits[0]) != 0:
        raise ValueError(
            f"Sparse field {field_name!r} splits must start at 0: {splits.tolist()}."
        )
    lengths = splits[1:] - splits[:-1]
    if np.any(lengths < 0):
        raise ValueError(
            f"Sparse field {field_name!r} splits must be monotonic: {splits.tolist()}."
        )
