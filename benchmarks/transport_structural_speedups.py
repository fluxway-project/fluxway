# pyright: reportMissingImports=false
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

from fluxway.common.coherence import CoherenceAction
from fluxway.transport.coherence import pack_entry_state_flags
from fluxway.transport.manifest import (
    ENTRY_AUX,
    ENTRY_COMPACT_BEGIN,
    ENTRY_EXPECTED_COUNT,
    ENTRY_GLOBAL_SHAPE_BEGIN,
    ENTRY_GLOBAL_SHAPE_COUNT,
    ENTRY_PAYLOAD_BEGIN,
    ENTRY_PAYLOAD_COUNT,
    ENTRY_ROUTE_CODE,
    ENTRY_STATE_FLAGS,
    ENTRY_WIDTH,
    FRAME_ABI_TAG_FIELD,
    FRAME_ENTRY_COUNT,
    FRAME_FLAGS,
    FRAME_PAYLOAD_COUNT,
    FRAME_SENDER_RANK,
    FRAME_SENDER_SPMD_CODE,
    FRAME_SENDER_WORLD_SIZE,
    FRAME_STEP,
    FRAME_WIDTH,
    PAYLOAD_DTYPE,
    PAYLOAD_FLAGS,
    PAYLOAD_NDIM,
    PAYLOAD_ROLE,
    PAYLOAD_SEGMENT_ARG_INDEX,
    PAYLOAD_SHAPE_BEGIN,
    PAYLOAD_SHAPE_COUNT,
    PAYLOAD_VALID_COUNT,
    PAYLOAD_WIDTH,
    TRANSPORT_ABI_TAG,
    PayloadRole,
    TensorEntryKind,
    TransportFrame,
    TransportFrameBuilder,
    dtype_code,
)


@dataclass(frozen=True)
class BenchResult:
    name: str
    current_ms: float
    optimized_ms: float
    speedup: float
    current_bytes: int
    optimized_bytes: int
    bytes_saved: int
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "current_ms": self.current_ms,
            "optimized_ms": self.optimized_ms,
            "speedup": self.speedup,
            "current_bytes": self.current_bytes,
            "optimized_bytes": self.optimized_bytes,
            "bytes_saved": self.bytes_saved,
            "detail": self.detail,
        }


def median_ms(
    fn: Callable[[], Any],
    *,
    warmup: int,
    runs: int,
) -> tuple[float, Any]:
    result = None
    for _ in range(warmup):
        result = fn()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)
    if result is None:
        raise RuntimeError("benchmark function returned None.")
    return statistics.median(timings), result


def frame_metadata_nbytes(frame: TransportFrame) -> int:
    return (
        frame.frame_i32.nbytes
        + frame.entry_i32.nbytes
        + frame.payload_i32.nbytes
        + frame.shape_i32.nbytes
        + frame.global_shape_i64.nbytes
        + frame.compact_i64.nbytes
    )


def build_current_frame(edge_count: int, rows: int, cols: int) -> TransportFrame:
    segment = np.zeros((rows, cols), dtype=np.float32)
    builder = TransportFrameBuilder(
        step=7,
        sender_spmd_code=3,
        sender_rank=1,
        sender_world_size=4,
    )
    for edge_idx in range(edge_count):
        edge = builder.add_tensor_transfer_entry(
            transfer_code=1000 + edge_idx,
            layout_state=0,
            tensor_kind=TensorEntryKind.TENSOR,
            expected_count=4,
            coherence_epoch=0,
            global_shape=(rows, cols),
        )
        builder.add_payload(edge, PayloadRole.VALUES, segment, rows)
    return builder.build()


def build_preallocated_frame(edge_count: int, rows: int, cols: int) -> TransportFrame:
    segment = np.zeros((rows, cols), dtype=np.float32)
    idx = np.arange(edge_count, dtype=np.int32)
    frame_i32 = np.empty(FRAME_WIDTH, dtype=np.int32)
    frame_i32[FRAME_ABI_TAG_FIELD] = TRANSPORT_ABI_TAG
    frame_i32[FRAME_STEP] = 7
    frame_i32[FRAME_SENDER_SPMD_CODE] = 3
    frame_i32[FRAME_SENDER_RANK] = 1
    frame_i32[FRAME_SENDER_WORLD_SIZE] = 4
    frame_i32[FRAME_ENTRY_COUNT] = edge_count
    frame_i32[FRAME_PAYLOAD_COUNT] = edge_count
    frame_i32[FRAME_FLAGS] = 0

    entry_i32 = np.zeros((edge_count, ENTRY_WIDTH), dtype=np.int32)
    entry_i32[:, ENTRY_ROUTE_CODE] = 1000 + idx
    entry_i32[:, ENTRY_STATE_FLAGS] = pack_entry_state_flags(
        0, int(TensorEntryKind.TENSOR), CoherenceAction.BUS_RD
    )
    entry_i32[:, ENTRY_PAYLOAD_BEGIN] = idx
    entry_i32[:, ENTRY_PAYLOAD_COUNT] = 1
    entry_i32[:, ENTRY_EXPECTED_COUNT] = 4
    entry_i32[:, ENTRY_COMPACT_BEGIN] = 0
    entry_i32[:, ENTRY_GLOBAL_SHAPE_BEGIN] = idx * 2
    entry_i32[:, ENTRY_GLOBAL_SHAPE_COUNT] = 2
    entry_i32[:, ENTRY_AUX] = 0

    payload_i32 = np.zeros((edge_count, PAYLOAD_WIDTH), dtype=np.int32)
    payload_i32[:, PAYLOAD_ROLE] = int(PayloadRole.VALUES)
    payload_i32[:, PAYLOAD_DTYPE] = dtype_code(segment.dtype)
    payload_i32[:, PAYLOAD_NDIM] = 2
    payload_i32[:, PAYLOAD_SHAPE_BEGIN] = idx * 2
    payload_i32[:, PAYLOAD_SHAPE_COUNT] = 2
    payload_i32[:, PAYLOAD_VALID_COUNT] = rows
    payload_i32[:, PAYLOAD_SEGMENT_ARG_INDEX] = idx
    payload_i32[:, PAYLOAD_FLAGS] = 0

    shape_i32 = np.empty(edge_count * 2, dtype=np.int32)
    shape_i32[0::2] = rows
    shape_i32[1::2] = cols
    global_shape_i64 = np.empty(edge_count * 2, dtype=np.int64)
    global_shape_i64[0::2] = rows
    global_shape_i64[1::2] = cols

    frame = TransportFrame(
        frame_i32=frame_i32,
        entry_i32=entry_i32,
        payload_i32=payload_i32,
        shape_i32=shape_i32,
        global_shape_i64=global_shape_i64,
        compact_i64=np.empty(0, dtype=np.int64),
        payload_segments=tuple([segment] * edge_count),
    )
    frame.validate()
    return frame


def bench_frame_builder(args: argparse.Namespace) -> BenchResult:
    current_ms, current_frame = median_ms(
        lambda: build_current_frame(args.frame_edges, 8, 16),
        warmup=2,
        runs=args.runs,
    )
    optimized_ms, optimized_frame = median_ms(
        lambda: build_preallocated_frame(args.frame_edges, 8, 16),
        warmup=2,
        runs=args.runs,
    )
    if frame_metadata_nbytes(current_frame) != frame_metadata_nbytes(optimized_frame):
        raise AssertionError("frame metadata byte size mismatch.")
    if len(current_frame.payload_segments) != len(optimized_frame.payload_segments):
        raise AssertionError("payload segment count mismatch.")
    return make_result(
        "frame_builder_preallocated_flat_arrays",
        current_ms,
        optimized_ms,
        frame_metadata_nbytes(current_frame),
        frame_metadata_nbytes(optimized_frame),
        {"edges": args.frame_edges},
    )


def assemble_current_python(shards: list[np.ndarray]) -> np.ndarray:
    rows = sum(shard.shape[0] for shard in shards)
    cols = shards[0].shape[1]
    out = np.empty((rows, cols), dtype=shards[0].dtype)
    cursor = 0
    for shard in shards:
        for row_idx in range(shard.shape[0]):
            out[cursor] = shard[row_idx]
            cursor += 1
    return out


def assemble_vectorized(shards: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(shards, axis=0)


def bench_assemble(args: argparse.Namespace) -> BenchResult:
    shards = [
        np.full((args.assemble_rows, args.assemble_cols), rank, dtype=np.float32)
        for rank in range(args.assemble_shards)
    ]
    current_ms, current = median_ms(
        lambda: assemble_current_python(shards),
        warmup=1,
        runs=args.runs,
    )
    optimized_ms, optimized = median_ms(
        lambda: assemble_vectorized(shards),
        warmup=2,
        runs=args.runs,
    )
    if not np.array_equal(current, optimized):
        raise AssertionError("assembled tensors differ.")
    return make_result(
        "assemble_python_rows_to_numpy_concatenate",
        current_ms,
        optimized_ms,
        current.nbytes,
        optimized.nbytes,
        {
            "shards": args.assemble_shards,
            "rows_per_shard": args.assemble_rows,
            "cols": args.assemble_cols,
        },
    )


def route_current_python(
    values: np.ndarray,
    dest_ids: np.ndarray,
    world_size: int,
) -> list[np.ndarray]:
    buckets: list[list[np.ndarray]] = [[] for _ in range(world_size)]
    for row_idx, dest_id in enumerate(dest_ids):
        buckets[int(dest_id)].append(values[row_idx])
    return [
        np.asarray(bucket, dtype=values.dtype).reshape(-1, values.shape[1])
        for bucket in buckets
    ]


@njit(nogil=True)
def route_numba_nogil(
    values: np.ndarray,
    dest_ids: np.ndarray,
    world_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(world_size, dtype=np.int64)
    for row_idx in range(dest_ids.size):
        counts[dest_ids[row_idx]] += 1

    offsets = np.empty(world_size + 1, dtype=np.int64)
    offsets[0] = 0
    for rank in range(world_size):
        offsets[rank + 1] = offsets[rank] + counts[rank]

    cursors = offsets[:-1].copy()
    out = np.empty_like(values)
    for row_idx in range(values.shape[0]):
        dest_id = dest_ids[row_idx]
        out_idx = cursors[dest_id]
        for col_idx in range(values.shape[1]):
            out[out_idx, col_idx] = values[row_idx, col_idx]
        cursors[dest_id] = out_idx + 1
    return out, offsets


def routed_checksums_from_buckets(buckets: list[np.ndarray]) -> np.ndarray:
    return np.asarray([bucket.sum(dtype=np.float64) for bucket in buckets])


def routed_checksums_from_flat(out: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    checksums = []
    for rank in range(offsets.size - 1):
        start = offsets[rank]
        end = offsets[rank + 1]
        checksums.append(out[start:end].sum(dtype=np.float64))
    return np.asarray(checksums)


def bench_dynamic_route(args: argparse.Namespace) -> BenchResult:
    rng = np.random.default_rng(20260702)
    values = rng.random((args.route_rows, args.route_cols), dtype=np.float32)
    dest_ids = rng.integers(0, args.world_size, size=args.route_rows, dtype=np.int64)
    route_numba_nogil(values[:8], dest_ids[:8], args.world_size)

    current_ms, current = median_ms(
        lambda: route_current_python(values, dest_ids, args.world_size),
        warmup=1,
        runs=max(3, args.runs // 2),
    )
    optimized_ms, optimized = median_ms(
        lambda: route_numba_nogil(values, dest_ids, args.world_size),
        warmup=2,
        runs=args.runs,
    )
    out, offsets = optimized
    current_checksums = routed_checksums_from_buckets(current)
    optimized_checksums = routed_checksums_from_flat(out, offsets)
    if not np.allclose(current_checksums, optimized_checksums, rtol=1e-5, atol=1e-3):
        raise AssertionError("routed checksums differ.")
    current_bytes = sum(bucket.nbytes for bucket in current)
    optimized_bytes = out.nbytes + offsets.nbytes
    return make_result(
        "dynamic_route_python_buckets_to_numba_nogil",
        current_ms,
        optimized_ms,
        current_bytes,
        optimized_bytes,
        {
            "rows": args.route_rows,
            "cols": args.route_cols,
            "world_size": args.world_size,
        },
    )


def regular_index_current(
    data: np.ndarray, start: int, count: int
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(start, start + count, dtype=np.int64)
    return indices, data[indices]


def regular_index_view(data: np.ndarray, start: int, count: int) -> np.ndarray:
    return data[start : start + count]


def bench_regular_index(args: argparse.Namespace) -> BenchResult:
    data = np.arange(args.index_rows * args.index_cols, dtype=np.float32).reshape(
        args.index_rows,
        args.index_cols,
    )
    start = args.index_rows // 5
    count = args.index_rows // 2
    current_ms, current = median_ms(
        lambda: regular_index_current(data, start, count),
        warmup=2,
        runs=args.runs,
    )
    optimized_ms, optimized = median_ms(
        lambda: regular_index_view(data, start, count),
        warmup=2,
        runs=args.runs,
    )
    indices, gathered = current
    if not np.array_equal(gathered, optimized):
        raise AssertionError("regular index view differs from gathered copy.")
    if not np.shares_memory(data, optimized):
        raise AssertionError("optimized regular index path must share memory.")
    current_bytes = indices.nbytes + gathered.nbytes
    return make_result(
        "regular_index_materialized_to_compact_view",
        current_ms,
        optimized_ms,
        current_bytes,
        0,
        {
            "rows": args.index_rows,
            "cols": args.index_cols,
            "selected_rows": count,
        },
    )


def copy_adopt_current(source: np.ndarray) -> np.ndarray:
    return np.array(source, copy=True)


def zero_copy_adopt(source: np.ndarray) -> np.ndarray:
    return np.asarray(source)


def bench_zero_copy_adopt(args: argparse.Namespace) -> BenchResult:
    source = np.arange(args.copy_elements, dtype=np.float32)
    current_ms, current = median_ms(
        lambda: copy_adopt_current(source),
        warmup=2,
        runs=args.runs,
    )
    optimized_ms, optimized = median_ms(
        lambda: zero_copy_adopt(source),
        warmup=2,
        runs=args.runs,
    )
    if not np.array_equal(current, optimized):
        raise AssertionError("copy and zero-copy values differ.")
    if np.shares_memory(source, current):
        raise AssertionError("current copy path unexpectedly shares memory.")
    if not np.shares_memory(source, optimized):
        raise AssertionError("optimized adopt path must share memory.")
    return make_result(
        "backend_boundary_copy_to_zero_copy_adopt",
        current_ms,
        optimized_ms,
        source.nbytes,
        0,
        {"elements": args.copy_elements},
    )


@njit(nogil=True)
def heavy_numba_kernel(values: np.ndarray, rounds: int) -> float:
    total = 0.0
    for round_idx in range(rounds):
        for idx in range(values.size):
            total += values[idx] * 1.0000001 + round_idx
    return total


def bench_gil_free_threadpool(args: argparse.Namespace) -> BenchResult:
    values = np.linspace(0.0, 1.0, args.gilfree_elements, dtype=np.float64)
    heavy_numba_kernel(values[:16], 1)
    start = time.perf_counter()
    left = heavy_numba_kernel(values, args.gilfree_rounds)
    right = heavy_numba_kernel(values, args.gilfree_rounds)
    sequential_ms = (time.perf_counter() - start) * 1000.0
    with ThreadPoolExecutor(max_workers=2) as executor:
        start = time.perf_counter()
        futures = [
            executor.submit(heavy_numba_kernel, values, args.gilfree_rounds),
            executor.submit(heavy_numba_kernel, values, args.gilfree_rounds),
        ]
        parallel_values = [future.result() for future in futures]
        parallel_ms = (time.perf_counter() - start) * 1000.0
    if not np.allclose([left, right], parallel_values):
        raise AssertionError("GIL-free threadpool results differ.")
    return make_result(
        "threadpool_python_cpu_to_numba_nogil_parallel_overlap",
        sequential_ms,
        parallel_ms,
        values.nbytes * 2,
        values.nbytes * 2,
        {
            "elements": args.gilfree_elements,
            "rounds": args.gilfree_rounds,
        },
    )


def make_result(
    name: str,
    current_ms: float,
    optimized_ms: float,
    current_bytes: int,
    optimized_bytes: int,
    detail: dict[str, Any],
) -> BenchResult:
    if optimized_ms <= 0:
        raise ValueError(f"optimized_ms must be positive, got {optimized_ms}.")
    return BenchResult(
        name=name,
        current_ms=current_ms,
        optimized_ms=optimized_ms,
        speedup=current_ms / optimized_ms,
        current_bytes=current_bytes,
        optimized_bytes=optimized_bytes,
        bytes_saved=max(0, current_bytes - optimized_bytes),
        detail=detail,
    )


def geometric_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must not be empty.")
    product = 1.0
    for value in values:
        if value <= 0:
            raise ValueError(f"geometric mean values must be positive, got {value}.")
        product *= value
    return product ** (1.0 / len(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--frame-edges", type=int, default=2048)
    parser.add_argument("--assemble-shards", type=int, default=8)
    parser.add_argument("--assemble-rows", type=int, default=2048)
    parser.add_argument("--assemble-cols", type=int, default=32)
    parser.add_argument("--route-rows", type=int, default=65536)
    parser.add_argument("--route-cols", type=int, default=16)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--index-rows", type=int, default=524288)
    parser.add_argument("--index-cols", type=int, default=16)
    parser.add_argument("--copy-elements", type=int, default=8_388_608)
    parser.add_argument("--gilfree-elements", type=int, default=5_000_000)
    parser.add_argument("--gilfree-rounds", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = [
        "runs",
        "frame_edges",
        "assemble_shards",
        "assemble_rows",
        "assemble_cols",
        "route_rows",
        "route_cols",
        "world_size",
        "index_rows",
        "index_cols",
        "copy_elements",
        "gilfree_elements",
        "gilfree_rounds",
    ]
    for field in positive_fields:
        value = getattr(args, field)
        if value <= 0:
            raise ValueError(f"{field} must be positive, got {value}.")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    results = [
        bench_frame_builder(args),
        bench_assemble(args),
        bench_dynamic_route(args),
        bench_regular_index(args),
        bench_zero_copy_adopt(args),
        bench_gil_free_threadpool(args),
    ]
    speedups = [result.speedup for result in results]
    return {
        "passed": True,
        "geomean_speedup": geometric_mean(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "results": [result.to_dict() for result in results],
    }


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    if args.json:
        print(json.dumps(report, sort_keys=True))
        return
    print(
        "PASS transport structural speedups "
        f"geomean={report['geomean_speedup']:.3f}x "
        f"min={report['min_speedup']:.3f}x "
        f"max={report['max_speedup']:.3f}x"
    )
    for result in report["results"]:
        print(
            f"{result['name']}: {result['current_ms']:.3f}ms -> "
            f"{result['optimized_ms']:.3f}ms = {result['speedup']:.3f}x"
        )


if __name__ == "__main__":
    main()
