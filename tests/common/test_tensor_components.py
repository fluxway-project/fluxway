import numpy as np

from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.tensor_components import (
    descriptor_to_indices_spec,
    TensorLayout,
    TensorLineage,
    TensorPayload,
    TensorRouting,
    dense_path_strides,
    fill_nested_dense_inv_kernel,
    fill_nested_dense_inv_with_pad,
    nested_global_positions_kernel,
    nested_path_position_maps,
    row_partition_sparse_feature,
)
from fluxway.common.tensor_backend import get_backend, set_backend
from fluxway.common.types.sharding import RangeIndices
from fluxway.common.types.tensor import TransportTensor


def test_continuous_index_converts_to_range_indices_without_materialize():
    from fluxway.common.index_descriptor import ContinuousIndex

    set_backend("hybrid")
    backend = get_backend()
    desc = ContinuousIndex.chunk(7, 4)
    original = ContinuousIndex.materialize

    def forbidden(self):
        raise AssertionError("materialize should not be called")

    ContinuousIndex.materialize = forbidden
    try:
        spec = descriptor_to_indices_spec(
            desc,
            length=4,
            valid_length=None,
            backend=backend,
        )
    finally:
        ContinuousIndex.materialize = original

    assert isinstance(spec, RangeIndices)
    assert (spec.start, spec.length, spec.stride) == (7, 4, 1)


def test_transport_tensor_exposes_layered_components():
    tensor = TransportTensor(
        np.array([11, 12, 13], dtype=np.int64),
        nested_splits=[np.array([0, 2, 3], dtype=np.int32)],
        origin_indices=ExplicitIndex(np.array([5, 6], dtype=np.int64)),
        global_shape=(10, 2),
    )

    assert isinstance(tensor.payload, TensorPayload)
    assert isinstance(tensor.layout, TensorLayout)
    assert isinstance(tensor.routing, TensorRouting)
    assert isinstance(tensor.lineage, TensorLineage)
    assert tensor.flat_values.tolist() == [11, 12, 13]
    assert tensor.nested_splits[0].tolist() == [0, 2, 3]
    assert tensor.global_shape == (10, 2)
    assert tensor.origin_indices_materialized is not None
    assert tensor.origin_indices_materialized.tolist() == [5, 6]


def test_nested_sparse_layout_maps_to_dense_path_slots():
    tensor = TransportTensor.from_lists(
        [[[11, 12], [13]], [[21, 22]]],
        dtype=np.uint64,
    )
    sparse_layout = row_partition_sparse_feature("nested_sparse_dim_8", tensor)
    rows, offsets = nested_path_position_maps(
        sparse_layout.nested_splits,
        np.arange(5, dtype=np.int32),
        sparse_layout.widths,
    )
    strides = dense_path_strides(sparse_layout.widths)
    global_positions = np.empty(5, dtype=np.int64)

    nested_global_positions_kernel(
        rows,
        offsets,
        np.array([0, 1], dtype=np.int64),
        strides,
        global_positions,
    )

    assert sparse_layout.widths == (2, 2)
    assert global_positions.tolist() == [0, 1, 2, 4, 5]


def test_empty_nested_sparse_layout_preserves_zero_width_business_tensor():
    rows = 320
    tensor = TransportTensor(
        np.empty((0,), dtype=np.uint64),
        nested_splits=[np.zeros((rows + 1,), dtype=np.int32)],
        global_shape=(rows, 0),
    )

    sparse_layout = row_partition_sparse_feature("as_it_is_misc_dim_152", tensor)

    assert sparse_layout.rows == rows
    assert sparse_layout.widths == (0,)
    assert sparse_layout.flat_values.shape == (0,)
    assert sparse_layout.nested_splits[0].tolist() == [0] * (rows + 1)


def test_empty_nested_path_position_maps_do_not_require_positive_width():
    local_rows, offsets = nested_path_position_maps(
        (np.zeros((321,), dtype=np.int32),),
        np.empty((0,), dtype=np.int32),
        (0,),
    )

    assert local_rows.shape == (0,)
    assert offsets.shape == (0, 1)


def test_empty_multilevel_nested_sparse_layout_preserves_zero_widths():
    rows = 4
    tensor = TransportTensor(
        np.empty((0,), dtype=np.uint64),
        nested_splits=[
            np.zeros((rows + 1,), dtype=np.int32),
            np.zeros((1,), dtype=np.int32),
        ],
        global_shape=(rows, 0, 0),
    )

    sparse_layout = row_partition_sparse_feature("nested_empty_sparse", tensor)
    local_rows, offsets = nested_path_position_maps(
        sparse_layout.nested_splits,
        np.empty((0,), dtype=np.int32),
        sparse_layout.widths,
    )

    assert sparse_layout.rows == rows
    assert sparse_layout.widths == (0, 0)
    assert local_rows.shape == (0,)
    assert offsets.shape == (0, 2)


def test_nested_sparse_padding_inv_uses_dense_slot_holes():
    tensor = TransportTensor.from_lists(
        [[[11, 12], [13]], [[21, 22]]],
        dtype=np.uint64,
    )
    sparse_layout = row_partition_sparse_feature("nested_sparse_dim_8", tensor)
    out = np.empty((2, 2, 2), dtype=np.int64)

    fill_nested_dense_inv_kernel(
        np.array([10, 11, 12, 14, 15], dtype=np.int64),
        sparse_layout.nested_splits,
        dense_path_strides(sparse_layout.widths),
        out,
    )

    assert out.tolist() == [
        [[10, 11], [12, 3]],
        [[14, 15], [6, 7]],
    ]


def test_fill_nested_dense_inv_with_pad_multilevel():
    tensor = TransportTensor.from_lists(
        [[[11, 12], [13]], [[21, 22]]],
        dtype=np.uint64,
    )
    sparse_layout = row_partition_sparse_feature("nested_sparse_dim_8", tensor)
    out = np.empty((2, 2, 2), dtype=np.int64)

    fill_nested_dense_inv_with_pad(
        np.array([10, 11, 12, 14, 15], dtype=np.int64),
        sparse_layout.nested_splits,
        dense_path_strides(sparse_layout.widths),
        -1,
        out,
    )

    assert out.tolist() == [
        [[10, 11], [12, -1]],
        [[14, 15], [-1, -1]],
    ]


def test_fill_nested_dense_inv_with_pad_empty_splits_raises():
    import pytest

    out = np.empty((2, 2), dtype=np.int64)
    with pytest.raises(
        ValueError, match="nested_splits must contain at least one level"
    ):
        fill_nested_dense_inv_with_pad(
            np.array([1, 2], dtype=np.int64),
            (),
            np.array([2, 1], dtype=np.int64),
            -1,
            out,
        )
