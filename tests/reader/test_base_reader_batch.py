from __future__ import annotations

import numpy as np
import pytest

from fluxway.common.decorators import operator_context
from fluxway.common.types.configs import OperatorContext
from fluxway.reader.base_reader import BaseReader

pa = pytest.importorskip("pyarrow")


def _make_operator_context(
    rank: int = 0, world_size: int = 2, step: int = 0
) -> OperatorContext:
    return OperatorContext(
        physical_rank=rank,
        physical_world_size=world_size,
        logical_rank=rank,
        logical_world_size=world_size,
        step=step,
        global_source_offset=rank * 10,
    )


def test_concatenate_batches_arrow_table():
    t1 = pa.Table.from_arrays([pa.array([1, 2])], names=["col"])
    t2 = pa.Table.from_arrays([pa.array([3, 4, 5])], names=["col"])
    merged = BaseReader.concatenate_batches([t1, t2])
    assert isinstance(merged, pa.Table)
    assert merged.num_rows == 5
    assert merged["col"].to_pylist() == [1, 2, 3, 4, 5]


def test_concatenate_batches_record_batch():
    b1 = pa.RecordBatch.from_arrays([pa.array([10, 20])], names=["x"])
    b2 = pa.RecordBatch.from_arrays([pa.array([30, 40])], names=["x"])
    merged = BaseReader.concatenate_batches([b1, b2])
    assert isinstance(merged, pa.Table)
    assert merged.num_rows == 4
    assert merged["x"].to_pylist() == [10, 20, 30, 40]


def test_concatenate_batches_mapping():
    m1 = {"arr": np.array([1, 2]), "lst": ["a", "b"]}
    m2 = {"arr": np.array([3, 4]), "lst": ["c"]}
    merged = BaseReader.concatenate_batches([m1, m2])
    np.testing.assert_array_equal(merged["arr"], np.array([1, 2, 3, 4]))
    assert merged["lst"] == ["a", "b", "c"]


def test_concatenate_batches_errors():
    with pytest.raises(ValueError, match="requires at least one batch"):
        BaseReader.concatenate_batches([])

    t = pa.Table.from_arrays([pa.array([1])], names=["a"])
    with pytest.raises(TypeError, match="must be pyarrow.Table"):
        BaseReader.concatenate_batches([t, {"a": [2]}])

    with pytest.raises(TypeError, match="Unsupported batch type"):
        BaseReader.concatenate_batches([123, 456])


def test_create_tensor_arrow_arrays():
    reader = BaseReader()
    ctx = _make_operator_context(rank=0, world_size=2)
    with operator_context(static_args={}, rank=0, world_size=2, operator_ctx=ctx):
        # Dense Arrow array
        dense_arr = pa.array([10, 20, 30], type=pa.int64())
        tt_dense = reader.create_tensor(dense_arr, dtype=np.int32)
        assert tt_dense.shape == (3,)
        assert tt_dense.flat_values.dtype == np.int32
        assert tt_dense.global_shape == (6,)

        # Ragged list Arrow array
        list_arr = pa.array([[1, 2], [3], [4, 5, 6]], type=pa.list_(pa.int32()))
        tt_ragged = reader.create_tensor(list_arr)
        assert tt_ragged.logical_shape == (3, None)
        assert tt_ragged.global_shape == (6, None)
        assert len(tt_ragged.nested_splits) == 1
        np.testing.assert_array_equal(
            tt_ragged.nested_splits[0], np.array([0, 2, 3, 6], dtype=np.int32)
        )


def test_create_tensors_from_batch_arrow_table():
    reader = BaseReader()
    ctx = _make_operator_context(rank=0, world_size=2)
    with operator_context(static_args={}, rank=0, world_size=2, operator_ctx=ctx):
        table = pa.Table.from_arrays(
            [
                pa.array([101, 102], type=pa.int64()),
                pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
            ],
            names=["user_id", "clicks"],
        )
        tensors = reader.create_tensors_from_batch(
            table, column_dtypes={"user_id": np.int32}
        )
        assert "user_id" in tensors
        assert "clicks" in tensors
        assert tensors["user_id"].shape == (2,)
        assert tensors["user_id"].flat_values.dtype == np.int32
        assert tensors["clicks"].logical_shape == (2, None)
        assert tensors["clicks"].global_shape == (4, None)


def test_create_tensors_from_batch_mapping():
    reader = BaseReader()
    ctx = _make_operator_context(rank=1, world_size=2)
    with operator_context(static_args={}, rank=1, world_size=2, operator_ctx=ctx):
        batch = {
            "dense": np.array([1.0, 2.0], dtype=np.float32),
            "seq": [[10, 20], [30]],
        }
        tensors = reader.create_tensors_from_batch(batch)
        assert tensors["dense"].shape == (2,)
        assert tensors["dense"].global_shape == (4,)
        assert tensors["seq"].logical_shape == (2, None)
