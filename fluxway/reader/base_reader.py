"""Base class for all Reader SPMDs."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from fluxway.common.decorators import operator, spmd
from fluxway.common.index_descriptor import ContinuousIndex, IndexDescriptor
from fluxway.common.tensor_backend import get_backend
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.ingestion import IngestionLayoutActivation


@spmd()
class BaseReader:
    """Base implementation for data-producing SPMD readers."""

    rank: int
    world_size: int
    static_args: dict[str, Any]

    def __init__(
        self,
        name: str = "",
        mesh: Any = None,
        runtime_env: Any = None,
        resource: Any = None,
        **kwargs: Any,
    ):
        pass

    async def activate_ingestion_layout(
        self, activation: IngestionLayoutActivation
    ) -> None:
        if not isinstance(activation, IngestionLayoutActivation):
            raise TypeError(
                "reader ingestion activation requires IngestionLayoutActivation"
            )

    @operator("read")
    def read(self) -> Any:
        """Produce a batch as a data-type object or tensor pytree."""
        raise NotImplementedError("Subclasses must implement read()")

    async def run_in_lane(
        self,
        fn: Callable[..., Any],
        *args: Any,
        lane: str = "default",
        **kwargs: Any,
    ) -> Any:
        """Execute a blocking or CPU-bound callable in a worker-managed thread lane."""
        if self.ctx is None:
            raise RuntimeError(
                "run_in_lane requires an active OperatorContext on this Reader"
            )
        return await self.ctx.run_in_lane(fn, *args, lane=lane, **kwargs)

    @classmethod
    def concatenate_batches(cls, batches: Sequence[Any]) -> Any:
        """Concatenate a sequence of Arrow tables, record batches, or dictionaries along the batch axis."""
        if not batches:
            raise ValueError("concatenate_batches requires at least one batch")
        first = batches[0]
        if len(batches) == 1:
            return first
        pa = _pyarrow()

        if pa is not None and isinstance(first, pa.Table):
            if any(not isinstance(b, pa.Table) for b in batches[1:]):
                raise TypeError("All batches must be pyarrow.Table instances")
            return pa.concat_tables(batches, promote_options="permissive")
        if pa is not None and isinstance(first, pa.RecordBatch):
            if any(not isinstance(b, pa.RecordBatch) for b in batches[1:]):
                raise TypeError("All batches must be pyarrow.RecordBatch instances")
            return pa.Table.from_batches(batches)
        if isinstance(first, Mapping):
            result: dict[str, Any] = {}
            for key in first:
                vals = [b[key] for b in batches]
                if all(isinstance(v, np.ndarray) for v in vals):
                    result[key] = np.concatenate(vals, axis=0)
                elif all(isinstance(v, list) for v in vals):
                    result[key] = [item for sub in vals for item in sub]
                elif pa is not None and all(
                    isinstance(v, (pa.Array, pa.ChunkedArray)) for v in vals
                ):
                    result[key] = pa.chunked_array(vals)
                else:
                    raise TypeError(
                        f"Cannot concatenate batch field {key!r} with heterogeneous or unsupported types"
                    )
            return result
        raise TypeError(
            f"Unsupported batch type for concatenation: {type(first).__name__}"
        )

    def create_tensors_from_batch(
        self,
        batch: Any,
        *,
        column_dtypes: Mapping[str, Any] | None = None,
        uniform_dense_partition: bool = False,
    ) -> dict[str, TransportTensor]:
        """Materialize an Arrow Table, RecordBatch, or column mapping into TransportTensors."""
        pa = _pyarrow()

        if pa is not None and isinstance(batch, (pa.Table, pa.RecordBatch)):
            column_items = [(name, batch[name]) for name in batch.column_names]
            row_count = batch.num_rows
        elif isinstance(batch, Mapping):
            column_items = list(batch.items())
            if not column_items:
                raise ValueError("batch mapping must not be empty")
            first_val = column_items[0][1]
            row_count = _outer_length(first_val)
        else:
            raise TypeError(
                f"Unsupported batch type for tensor materialization: {type(batch).__name__}"
            )

        origin_indices, global_rows = self._source_batch_coordinates(row_count)
        tensors: dict[str, TransportTensor] = {}
        for name, col in column_items:
            expected_dtype = None if column_dtypes is None else column_dtypes.get(name)
            if pa is not None and isinstance(col, (pa.Array, pa.ChunkedArray)):
                col_type = col.type
                if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
                    chunk = (
                        col.combine_chunks()
                        if isinstance(col, pa.ChunkedArray)
                        else col
                    )
                    offsets = np.asarray(chunk.offsets.to_numpy(), dtype=np.int32)
                    values = np.asarray(chunk.values.to_numpy())
                    if expected_dtype is not None:
                        values = values.astype(expected_dtype, copy=False)
                    tensors[name] = TransportTensor(
                        flat_values=values,
                        nested_splits=[offsets],
                        origin_indices=origin_indices,
                        global_shape=(global_rows, None),
                    )
                else:
                    arr = np.asarray(col.to_numpy(zero_copy_only=False))
                    if expected_dtype is not None:
                        arr = arr.astype(expected_dtype, copy=False)
                    tensors[name] = TransportTensor.from_tensor(
                        arr,
                        global_shape=(global_rows, *arr.shape[1:]),
                        origin_indices=origin_indices,
                        is_uniform_dense_partition=uniform_dense_partition,
                    )
            elif isinstance(col, np.ndarray):
                arr = (
                    col
                    if expected_dtype is None
                    else col.astype(expected_dtype, copy=False)
                )
                tensors[name] = TransportTensor.from_tensor(
                    arr,
                    global_shape=(global_rows, *arr.shape[1:]),
                    origin_indices=origin_indices,
                    is_uniform_dense_partition=uniform_dense_partition,
                )
            elif isinstance(col, list):
                tensors[name] = TransportTensor.from_lists(
                    col,
                    dtype=expected_dtype,
                    logical_world_size=self.ctx.logical_world_size,
                    origin_indices=origin_indices,
                    is_uniform_dense_partition=uniform_dense_partition,
                )
            elif isinstance(col, TransportTensor):
                tensors[name] = col
            else:
                backend = get_backend()
                tensor = backend.tensor(col, dtype=expected_dtype)
                tensors[name] = TransportTensor.from_tensor(
                    tensor,
                    global_shape=(global_rows, *tensor.shape[1:]),
                    origin_indices=origin_indices,
                    is_uniform_dense_partition=uniform_dense_partition,
                )
        return tensors

    def create_tensor(
        self, data: Any, dtype: Any = None, uniform_dense_partition: bool = False
    ) -> "TransportTensor":
        """Create a dense or structurally inferred tensor for the active source batch."""
        context = self.ctx
        origin_indices, global_rows = self._source_batch_coordinates(
            _outer_length(data)
        )
        pa = _pyarrow()

        if pa is not None and isinstance(data, (pa.Array, pa.ChunkedArray)):
            col_type = data.type
            if pa.types.is_list(col_type) or pa.types.is_large_list(col_type):
                chunk = (
                    data.combine_chunks() if isinstance(data, pa.ChunkedArray) else data
                )
                offsets = np.asarray(chunk.offsets.to_numpy(), dtype=np.int32)
                values = np.asarray(chunk.values.to_numpy())
                if dtype is not None:
                    values = values.astype(dtype, copy=False)
                return TransportTensor(
                    flat_values=values,
                    nested_splits=[offsets],
                    origin_indices=origin_indices,
                    global_shape=(global_rows, None),
                )
            arr = np.asarray(data.to_numpy(zero_copy_only=False))
            if dtype is not None:
                arr = arr.astype(dtype, copy=False)
            return TransportTensor.from_tensor(
                arr,
                global_shape=(global_rows, *arr.shape[1:]),
                origin_indices=origin_indices,
                is_uniform_dense_partition=uniform_dense_partition,
            )
        if uniform_dense_partition:
            backend = get_backend()
            tensor = backend.tensor(data, dtype=dtype)
            return TransportTensor.from_tensor(
                tensor,
                global_shape=(global_rows, *tensor.shape[1:]),
                origin_indices=origin_indices,
            )
        if isinstance(data, list):
            result = TransportTensor.from_lists(
                data,
                dtype=dtype,
                logical_world_size=context.logical_world_size,
                origin_indices=origin_indices,
                is_uniform_dense_partition=uniform_dense_partition,
            )
        else:
            backend = get_backend()
            tensor = backend.tensor(data, dtype=dtype)
            result = TransportTensor.from_tensor(
                tensor,
                logical_world_size=context.logical_world_size,
                origin_indices=origin_indices,
                is_uniform_dense_partition=uniform_dense_partition,
            )
        return result

    def create_row_sparse_tensor(
        self, rows: list[list[Any]], *, width: int, dtype: Any = None
    ) -> "TransportTensor":
        """Create a row-partitioned sparse occurrence stream."""
        origin_indices, global_rows = self._source_batch_coordinates(len(rows))
        return TransportTensor.from_row_partitioned(
            rows,
            width=width,
            dtype=dtype,
            global_shape=(global_rows, width),
            origin_indices=origin_indices,
        )

    def _source_batch_coordinates(self, row_count: int) -> tuple[IndexDescriptor, int]:
        context = self.ctx
        if context is None:
            raise RuntimeError(
                "reader tensor creation requires an active OperatorContext"
            )
        items = context.source_work_items
        if items is None:
            if context.global_source_offset is None:
                raise ValueError(
                    "reader requires source work or explicit global provenance"
                )
            return ContinuousIndex.chunk(
                context.global_source_offset, row_count
            ), row_count * context.logical_world_size
        if not items or sum(item.cardinality for item in items) != row_count:
            raise ValueError("reader output rows do not match acquired source work")
        batches = {item.assignment.global_batch_size for item in items}
        if len(batches) != 1 or any(item.key.step != context.step for item in items):
            raise ValueError("reader work items do not belong to one logical batch")
        return IndexDescriptor.concatenate(
            [
                ContinuousIndex.chunk(item.global_start, item.cardinality)
                for item in items
            ]
        ), batches.pop()


def _pyarrow() -> Any:
    try:
        import pyarrow as pa

        return pa
    except (ImportError, TypeError):
        return None


def _outer_length(data: Any) -> int:
    if isinstance(data, (list, tuple)):
        return len(data)
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    pa = _pyarrow()

    if pa is not None and isinstance(data, (pa.Array, pa.ChunkedArray)):
        return len(data)
    raise TypeError(
        f"reader tensor data must have an outer batch dimension, got {type(data).__name__}"
    )
