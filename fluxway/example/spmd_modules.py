"""
SPMD modules for the example.

All SPMDs:
- Use CPU only (no GPU)
- Simulate data transmission only (no actual computation)
- Have total_rank <= 6
- Cover different stages of a machine learning pipeline:
  - Preprocessing: sparse -> sparse (eager mode)
  - Embedding: sparse -> sparse
  - Model: sparse -> dense

Note: DataReaderSPMD has been replaced by the new add_reader() API
which creates a dedicated Reader node that reads data from files.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np  # used indirectly via tensor operations

from fluxway.common.decorators import spmd, operator
from fluxway.common.types import ShardingSpec
from .configs import TensorOnly, DataState1, DataState2, DataState3, Gradient2


logger = logging.getLogger(__name__)

# =============================================================================
# Dynamic Routing Functions
# =============================================================================


def _aggregate_rows_by_rank(
    rank_to_rows: Dict[int, List[int]],
) -> List[Tuple[Tuple, int]]:
    """
    Aggregate per-rank row indices into a single data_mask entry per rank.

    If all rows for a rank form a contiguous range, use ``(slice(start, end),)``.
    Otherwise use ``(np.array(sorted_row_indices),)`` for numpy fancy indexing.

    Args:
        rank_to_rows: Mapping from target_rank to list of local row indices

    Returns:
        List of (data_mask, target_rank) with exactly one entry per rank
    """
    results: List[Tuple[Tuple, int]] = []
    for rank in sorted(rank_to_rows.keys()):
        rows = sorted(rank_to_rows[rank])
        if not rows:
            continue
        # Check if rows are contiguous
        if rows[-1] - rows[0] + 1 == len(rows):
            results.append(((slice(rows[0], rows[-1] + 1),), rank))
        else:
            results.append(((np.array(rows, dtype=np.intp),), rank))
    return results


def hash_routing_fn(
    tensor: Any, tensor_meta: Any, post_spec: ShardingSpec
) -> List[Tuple[Tuple, int]]:
    """
    Index-modulo routing function for dynamic sharding.

    Routes each row based on its global_index % num_ranks, then aggregates
    all rows destined for the same rank into a single entry.

    Args:
        tensor: The tensor data to route
        tensor_meta: TensorMeta with global_indices info
        post_spec: Destination sharding spec (provides mesh info)

    Returns:
        List of (data_mask, target_rank) tuples, one entry per rank.
        data_mask is ``(slice(start, end),)`` for contiguous rows or
        ``(np.ndarray,)`` for non-contiguous rows.
    """
    from collections import defaultdict

    num_ranks = post_spec.layout_mesh.num_devices
    global_indices = tensor_meta.global_indices
    rank_to_rows: Dict[int, List[int]] = defaultdict(list)

    if global_indices is not None and len(global_indices) > 0:
        for i in range(len(global_indices)):
            target_rank = int(global_indices[i]) % num_ranks
            rank_to_rows[target_rank].append(i)
    else:
        batch_size = (
            tensor.shape[0]
            if hasattr(tensor, "shape")
            else (len(tensor) if hasattr(tensor, "__len__") else 1)
        )
        for i in range(batch_size):
            target_rank = i % num_ranks
            rank_to_rows[target_rank].append(i)

    results = _aggregate_rows_by_rank(rank_to_rows)

    logger.info(
        f"[hash_routing_fn] num_ranks={num_ranks}, "
        f"entries={[(type(r[0][0]).__name__, r[1]) for r in results]}"
    )

    return results


def modulo_routing_fn(
    tensor: Any, tensor_meta: Any, post_spec: ShardingSpec
) -> List[Tuple[Tuple[slice, ...], int]]:
    """
    Block-partition routing function for dynamic sharding.

    Routes data by partitioning the dp axis (axis 0) of global_shape
    according to post_spec, then using global_indices to determine
    which local rows should be sent to each downstream rank.

    Algorithm:
        1. Use global_shape and post_spec to compute axis-0 block partition
           boundaries for each downstream rank.
        2. Use global_indices to find local rows whose global index falls
           within that rank's partition, and emit per-row slice tuples.

    Args:
        tensor: The tensor data to route
        tensor_meta: TensorMeta with global_indices and global_shape info
        post_spec: Destination sharding spec (provides mesh info for partitioning)

    Returns:
        List of (slice_tuple, target_rank) tuples where slice_tuple is a
        tuple of slices selecting a row from the local tensor.
    """
    import math

    global_shape = tensor_meta.global_shape
    global_indices = tensor_meta.global_indices

    if global_shape is None or global_indices is None:
        raise ValueError(
            "modulo_routing_fn requires tensor_meta.global_shape and "
            "tensor_meta.global_indices to be set"
        )

    num_ranks = post_spec.layout_mesh.num_devices
    results: List[Tuple[Tuple[slice, ...], int]] = []

    if not post_spec.is_content_addressable:
        # Use post_spec.lower() to get each rank's axis-0 range
        for rank in range(num_ranks):
            layout = post_spec.lower(global_shape, rank)
            axis0_slice = layout.dense_slices[0]
            start, end = axis0_slice.start, axis0_slice.end

            # Find local rows whose global index falls in [start, end)
            mask = (global_indices >= start) & (global_indices < end)
            matching_rows = np.where(mask)[0]
            for row_idx in matching_rows:
                results.append(((slice(int(row_idx), int(row_idx) + 1),), rank))
    else:
        # Fallback for DynamicShardingSpec: simple block partition on axis 0
        global_dim0 = global_shape[0]
        chunk_size = math.ceil(global_dim0 / num_ranks)

        for rank in range(num_ranks):
            start = rank * chunk_size
            end = min(start + chunk_size, global_dim0)

            mask = (global_indices >= start) & (global_indices < end)
            matching_rows = np.where(mask)[0]
            for row_idx in matching_rows:
                results.append(((slice(int(row_idx), int(row_idx) + 1),), rank))

    logger.info(
        f"[modulo_routing_fn] num_rows={len(results)}, num_ranks={num_ranks}, results={results}"
    )

    return results


# =============================================================================
# Note: Data reading is now handled by the Reader node using add_reader() API
# The DataReaderSPMD class has been removed as it's replaced by the new
# Reader functionality that actively reads data from files with backpressure
# support.
# =============================================================================


# =============================================================================
# Preprocessing SPMD (Eager version)
# =============================================================================


@spmd(
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
    mesh_axis_names=("data",),
    mesh_axis_shapes=(4,),
    max_async_steps=2,
    params_update_steps=1,
    enable_output_step_remap=True,
)
class PreprocessingSPMD:
    """
    Preprocessing SPMD (Eager mode).

    Performs data preprocessing: receives sparse data from reader, outputs sparse data.
    Uses eager_trigger=True to process data immediately.
    Uses modulo_routing_fn for input routing and hash_routing_fn for output routing.

    Note: For eager_trigger, only tensor_argument is used (no state_argument).
    The operator receives only the tensor from DataState1, not the full state.
    """

    @operator(
        name="forward",
        input_sharding={
            # Sparse input: uses fn for dynamic routing
            "input": {"type": "sparse", "axis": ["data"]}
        },
        output_sharding={
            # Sparse output: uses fn for dynamic routing
            "tensor": {"type": "sparse", "axis": ["data"], "fn": hash_routing_fn}
        },
        tensor_argument=["input"],
        eager_trigger=True,
    )
    def forward(self, data: TensorOnly) -> DataState1:
        """Preprocess sparse input from reader and output sparse data (eager mode)."""
        start_time = time.time()
        input_shape_info = (
            f"len={len(data.input) if hasattr(data.input, '__len__') else 'N/A'}"
        )
        logger.info(
            f"[PreprocessingSPMD.forward] input: type={type(data.input).__name__}, "
            f"{input_shape_info}"
        )

        # Extract 'tensor' field from each dict in the input list
        # data.input is list[dict], each dict has 'tensor' field (type: list)
        tensor_list = []
        for record in data.input:
            if isinstance(record, dict) and "tensor" in record:
                tensor_list.append(record["tensor"])

        # Merge all extracted tensors and convert to np.ndarray
        # Concatenate all tensors along the first axis (batch dimension)
        if tensor_list:
            # Convert each tensor to np.ndarray first, then concatenate
            output_tensor = np.array(tensor_list)
        else:
            # Empty input, create empty array
            output_tensor = np.array([])

        # Simulate processing time
        time.sleep(0.5)

        end_time = time.time()
        logger.info(
            f"[PreprocessingSPMD.forward] input_records={len(data.input)}, "
            f"input_shape={input_shape_info}, output_shape={output_tensor.shape}, "
            f"duration={end_time - start_time:.6f}s"
        )

        return DataState1(tensor=output_tensor)


# =============================================================================
# Model SPMD
# =============================================================================


@spmd(
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
    mesh_axis_names=("data",),
    mesh_axis_shapes=(2,),
    max_async_steps=2,
    params_update_steps=1,
)
class ModelSPMD:
    """
    Model SPMD.

    Performs model computation: receives sparse data (from embedding), outputs dense data.
    Uses modulo_routing_fn for input routing.
    """

    @operator(
        name="forward",
        input_sharding={
            # Sparse input: uses fn for dynamic routing
            "tensor": {"type": "dense", "axis": ["data"]}
        },
        output_sharding={
            # Dense output: regular block sharding
            "tensor": {"type": "dense", "axis": ["data"]}
        },
        state_argument=["processed"],
        tensor_argument=["tensor"],
    )
    def forward(self, state: DataState2) -> DataState3:
        """Model forward pass: process sparse input and output dense data."""
        start_time = time.time()
        input_shape = state.tensor.shape if hasattr(state.tensor, "shape") else "N/A"
        logger.info(
            f"[ModelSPMD.forward] input: input_shape={input_shape}, "
            f"processed={state.processed}"
        )

        # Aggregate sparse inputs to dense output (model computation)
        output_tensor = state.tensor.copy()

        # Simulate processing time
        time.sleep(0.5)

        end_time = time.time()
        logger.info(
            f"[ModelSPMD.forward] completed, "
            f"input_shape={input_shape}, output_shape={output_tensor.shape}, "
            f"duration={end_time - start_time:.6f}s"
        )

        return DataState3(tensor=output_tensor, stage=1)

    @operator(
        name="backward",
        input_sharding={
            # Dense input: regular block sharding
            # Note: This must match forward output sharding for self-loop edge
            "tensor": {"type": "dense", "axis": ["data"]}
        },
        output_sharding={
            # Sparse output: uses fn for dynamic routing
            # Note: axis must match this SPMD's mesh, which is ("data",)
            "tensor": {"type": "dense", "axis": ["data"]}
        },
        state_argument=["stage"],
        tensor_argument=["tensor"],
    )
    def backward(self, state: DataState3) -> Gradient2:
        """Model backward pass: compute model gradients from forward output."""
        start_time = time.time()
        input_shape = state.tensor.shape if hasattr(state.tensor, "shape") else "N/A"
        logger.info(
            f"[ModelSPMD.backward] input: input_shape={input_shape}, "
            f"stage={state.stage}"
        )

        # Convert forward output to gradient (simulate gradient computation)
        output_grad = state.tensor.copy()

        # Simulate processing time
        time.sleep(0.5)

        end_time = time.time()
        logger.info(
            f"[ModelSPMD.backward] completed, "
            f"input_shape={input_shape}, output_shape={output_grad.shape}, "
            f"duration={end_time - start_time:.6f}s"
        )

        return Gradient2(tensor=output_grad)


# =============================================================================
# Embedding SPMD
# =============================================================================


@spmd(
    resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
    mesh_axis_names=("embed",),
    mesh_axis_shapes=(4,),
    max_async_steps=2,
    params_update_steps=1,
)
class EmbeddingSPMD:
    """
    Embedding SPMD.

    Performs embedding operations: receives sparse data, outputs sparse data using dynamic routing.
    Both input and output use dynamic routing.
    Uses hash_routing_fn for input routing and modulo_routing_fn for output routing.
    """

    @operator(
        name="forward",
        input_sharding={
            # Sparse input
            "tensor": {"type": "sparse", "axis": ["embed"], "fn": hash_routing_fn}
        },
        output_sharding={
            # Sparse output
            "tensor": {"type": "sparse", "axis": ["embed"], "fn": modulo_routing_fn}
        },
        tensor_argument=["tensor"],
        eager_trigger=True,
    )
    def forward(self, state: DataState1) -> DataState2:
        """Perform embedding: process sparse input and output sparse data."""
        start_time = time.time()
        input_shape = state.tensor.shape if hasattr(state.tensor, "shape") else "N/A"
        logger.info(f"[EmbeddingSPMD.forward] input: input_shape={input_shape}")

        # Transform sparse data (embedding operation)
        output_tensor = state.tensor.copy()

        # Simulate processing time
        time.sleep(0.5)

        end_time = time.time()
        logger.info(
            f"[EmbeddingSPMD.forward] completed, "
            f"input_shape={input_shape}, output_shape={output_tensor.shape}, "
            f"duration={end_time - start_time:.6f}s"
        )

        return DataState2(tensor=output_tensor, processed=True)

    @operator(
        name="backward",
        input_sharding={"tensor": {"type": "sparse", "axis": ["embed"]}},
        output_sharding={"tensor": {"type": "sparse", "axis": ["embed"]}},
        tensor_argument=["tensor"],
        eager_trigger=True,
    )
    def backward(self, grad: Gradient2) -> Gradient2:
        """Backward pass: compute embedding gradients."""
        start_time = time.time()
        input_shape = grad.tensor.shape if hasattr(grad.tensor, "shape") else "N/A"
        logger.info(f"[EmbeddingSPMD.backward] input: input_shape={input_shape}")

        output_grad = grad.tensor.copy() * 0.5  # Gradient scaling

        # Simulate processing time
        time.sleep(0.5)

        end_time = time.time()
        logger.info(
            f"[EmbeddingSPMD.backward] completed, "
            f"input_shape={input_shape}, output_shape={output_grad.shape}, "
            f"duration={end_time - start_time:.6f}s"
        )

        return Gradient2(tensor=output_grad)
