"""Tests for HybridBackend covering CPU (Numba JIT), GPU (CuPy), and CPU-GPU interop.

These tests simulate real communication layer workloads:
- Sparse feature routing (argsort + bincount + cumsum)
- Embedding gather (take with high-dim data)
- Scatter position computation (floor_divide + mod + unravel_index)
- Dynamic batch processing (varying sizes without recompilation)
"""

import numpy as np
import pytest

from fluxway.common.tensor_backend import HybridBackend
from fluxway.common.tensor_components import (
    dense_path_strides,
    nested_global_positions_kernel,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def cpu_backend():
    """Backend with CPU as default device and Numba JIT enabled."""
    b = HybridBackend(default_device="cpu", enable_numba_jit=True)
    assert b._use_jit is True, "Numba JIT must be available, no fallback allowed"
    assert b._default_device == "cpu"
    return b


@pytest.fixture
def gpu_backend():
    """Backend with GPU as default device — must NOT fallback to CPU."""
    try:
        b = HybridBackend(default_device="gpu")
    except Exception:
        pytest.skip("CuPy/CUDA not available")
        return  # unreachable, but informs pyrefly that b is always bound below
    assert b._cp is not None, "GPU mode must be active, no fallback to CPU"
    return b


def test_hybrid_backend_rejects_unknown_default_device():
    with pytest.raises(ValueError, match="default_device"):
        HybridBackend(default_device="automatic")


# ============================================================
# CPU Mode — Numba JIT Verification (NO FALLBACK)
# ============================================================


class TestCPUNumbaJIT:
    """Verify CPU path uses Numba JIT for hot operations, not pure NumPy."""

    def test_cpu_jit_and_layout_lowering_survive_numba_cuda_initialization(
        self, cpu_backend
    ):
        from numba import cuda

        cuda.is_available()

        counts = cpu_backend.bincount(
            np.array([2, 0, 2, 1], dtype=np.int64), minlength=4
        )
        offsets = cpu_backend.cumsum(np.array([2, 1, 0, 3], dtype=np.int64))
        taken = cpu_backend.take(
            np.array([10, 20, 30, 40], dtype=np.float32),
            np.array([3, 1, 0], dtype=np.int32),
        )
        positions = np.empty(2, dtype=np.int64)
        nested_global_positions_kernel(
            np.array([0, 1], dtype=np.int32),
            np.array([[0, 0], [1, 0]], dtype=np.int32),
            np.array([0, 1], dtype=np.int64),
            dense_path_strides((2, 2)),
            positions,
        )

        np.testing.assert_array_equal(counts, np.array([1, 1, 2, 0]))
        np.testing.assert_array_equal(offsets, np.array([2, 3, 3, 6]))
        np.testing.assert_array_equal(taken, np.array([40, 20, 10], dtype=np.float32))
        np.testing.assert_array_equal(positions, np.array([0, 6], dtype=np.int64))

    def test_bincount_sparse_routing(self, cpu_backend):
        """Real scenario: count how many features route to each worker."""
        rng = np.random.default_rng(42)
        world_size = 8
        n_features = 100_000
        destinations = rng.integers(0, world_size, n_features, dtype=np.int32)
        x = cpu_backend.tensor(destinations)
        counts = cpu_backend.bincount(x, minlength=world_size)
        expected = np.bincount(destinations, minlength=world_size)
        np.testing.assert_array_equal(cpu_backend.to_numpy(counts), expected)
        assert isinstance(counts, np.ndarray)

    def test_cumsum_offset_computation(self, cpu_backend):
        """Real scenario: compute send offsets from per-worker counts."""
        counts = np.array([1200, 800, 1500, 600, 2000, 400, 1800, 700], dtype=np.int64)
        x = cpu_backend.tensor(counts)
        offsets = cpu_backend.cumsum(x, dim=0)
        expected = np.cumsum(counts)
        np.testing.assert_array_equal(cpu_backend.to_numpy(offsets), expected)
        assert isinstance(offsets, np.ndarray)

    def test_take_embedding_gather_1d(self, cpu_backend):
        """Real scenario: 1-D take should use Numba JIT path."""
        rng = np.random.default_rng(42)
        vocab_size = 50_000
        batch_size = 8192
        embedding_ids = rng.integers(0, vocab_size, batch_size, dtype=np.int32)
        lookup_table = rng.standard_normal(vocab_size).astype(np.float32)
        table = cpu_backend.tensor(lookup_table)
        indices = cpu_backend.tensor(embedding_ids)
        result = cpu_backend.take(table, indices)
        expected = lookup_table[embedding_ids]
        np.testing.assert_allclose(cpu_backend.to_numpy(result), expected, rtol=1e-6)
        assert isinstance(result, np.ndarray)

    def test_argsort_rank_routing(self, cpu_backend):
        """Real scenario: sort features by destination worker."""
        rng = np.random.default_rng(42)
        n = 10_000
        destinations = rng.integers(0, 8, n, dtype=np.int32)
        x = cpu_backend.tensor(destinations)
        indices = cpu_backend.argsort(x, dim=0, stable=True)
        sorted_dests = cpu_backend.to_numpy(x)[cpu_backend.to_numpy(indices)]
        assert np.all(sorted_dests[:-1] <= sorted_dests[1:])

    def test_dynamic_shapes_no_recompile(self, cpu_backend):
        """Real scenario: varying batch sizes — Numba handles without recompilation."""
        rng = np.random.default_rng(42)
        sizes = [512, 1024, 2048, 4096, 8192, 16384, 3000, 7777]
        for size in sizes:
            destinations = rng.integers(0, 8, size, dtype=np.int32)
            x = cpu_backend.tensor(destinations)
            counts = cpu_backend.bincount(x, minlength=8)
            expected = np.bincount(destinations, minlength=8)
            np.testing.assert_array_equal(cpu_backend.to_numpy(counts), expected)
            offsets = cpu_backend.cumsum(counts, dim=0)
            np.testing.assert_array_equal(
                cpu_backend.to_numpy(offsets), np.cumsum(expected)
            )

    def test_bincount_jit_not_numpy_fallback(self, cpu_backend):
        """Verify _use_jit is active and bincount returns np.ndarray."""
        assert cpu_backend._use_jit is True, "JIT must be active"
        x = cpu_backend.tensor(np.array([0, 1, 2, 1, 0, 3], dtype=np.int32))
        result = cpu_backend.bincount(x, minlength=4)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [2, 2, 1, 1])

    def test_mod_jit_matches_numpy_for_all_sparse_integer_encodings(self, cpu_backend):
        """Modulo keeps NumPy semantics, including high-bit uint64 values."""
        size = 1 << 17  # Crosses the measured Numba parallelization threshold.
        cases = {
            np.dtype(np.int32): np.array(
                [-(2**31), -3, -1, 0, 1, 2**31 - 1], dtype=np.int32
            ),
            np.dtype(np.uint32): np.array([0, 1, 3, 2**32 - 1], dtype=np.uint32),
            np.dtype(np.int64): np.array(
                [-(2**63), -3, -1, 0, 1, 2**63 - 1], dtype=np.int64
            ),
            np.dtype(np.uint64): np.array([0, 1, 3, 2**63, 2**64 - 1], dtype=np.uint64),
        }
        for dtype, edge_values in cases.items():
            values = np.resize(edge_values, size).astype(dtype, copy=False)
            result = cpu_backend.mod(values, 1009)
            expected = np.mod(values, 1009)
            np.testing.assert_array_equal(result, expected)
            assert result.dtype == values.dtype

    def test_cumsum_jit_1d_path(self, cpu_backend):
        """Verify 1D cumsum uses Numba JIT path."""
        assert cpu_backend._use_jit is True
        x = cpu_backend.tensor(np.array([1, 2, 3, 4, 5], dtype=np.int64))
        result = cpu_backend.cumsum(x, dim=0)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 3, 6, 10, 15])

    def test_take_1d_jit_path(self, cpu_backend):
        """Verify 1D take uses Numba JIT path."""
        assert cpu_backend._use_jit is True
        data = cpu_backend.tensor(
            np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        )
        indices = cpu_backend.tensor(np.array([4, 2, 0, 3, 1], dtype=np.int32))
        result = cpu_backend.take(data, indices)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [50.0, 30.0, 10.0, 40.0, 20.0])

    def test_full_routing_pipeline_cpu(self, cpu_backend):
        """Full routing pipeline: argsort → bincount → cumsum → take."""
        rng = np.random.default_rng(99)
        n = 20_000
        world_size = 8
        destinations = rng.integers(0, world_size, n, dtype=np.int32)
        x = cpu_backend.tensor(destinations)
        sort_idx = cpu_backend.argsort(x, stable=True)
        assert isinstance(sort_idx, np.ndarray)
        counts = cpu_backend.bincount(x, minlength=world_size)
        assert isinstance(counts, np.ndarray)
        assert counts.sum() == n
        offsets = cpu_backend.cumsum(counts, dim=0)
        assert isinstance(offsets, np.ndarray)
        assert int(offsets[-1]) == n
        feature_values = rng.standard_normal(n).astype(np.float32)
        fv = cpu_backend.tensor(feature_values)
        gathered = cpu_backend.take(fv, sort_idx)
        assert isinstance(gathered, np.ndarray)
        np.testing.assert_allclose(gathered, feature_values[sort_idx], rtol=1e-6)
        np.testing.assert_array_equal(
            counts, np.bincount(destinations, minlength=world_size)
        )
        np.testing.assert_array_equal(
            offsets, np.cumsum(np.bincount(destinations, minlength=world_size))
        )


# ============================================================
# GPU Mode — CuPy Native Verification (NO FALLBACK)
# ============================================================


class TestGPUCuPyNative:
    """Verify GPU path uses CuPy native ops, not CPU fallback."""

    def test_bincount_gpu_native(self, gpu_backend):
        """Sparse routing on GPU — must use cupy.bincount."""
        import cupy as cp

        rng = np.random.default_rng(42)
        world_size = 8
        n_features = 100_000
        destinations = rng.integers(0, world_size, n_features, dtype=np.int32)
        x = gpu_backend.tensor(destinations)
        assert isinstance(x, cp.ndarray), "Input must be CuPy array, not NumPy"
        counts = gpu_backend.bincount(x, minlength=world_size)
        assert isinstance(counts, cp.ndarray), (
            "Output must be CuPy array (GPU-resident)"
        )
        expected = np.bincount(destinations, minlength=world_size)
        np.testing.assert_array_equal(gpu_backend.to_numpy(counts), expected)

    def test_cumsum_gpu_native(self, gpu_backend):
        """Offset computation on GPU."""
        import cupy as cp

        counts = np.array([1200, 800, 1500, 600, 2000, 400, 1800, 700], dtype=np.int64)
        x = gpu_backend.tensor(counts)
        assert isinstance(x, cp.ndarray)
        offsets = gpu_backend.cumsum(x, dim=0)
        assert isinstance(offsets, cp.ndarray), "Must stay on GPU"
        np.testing.assert_array_equal(gpu_backend.to_numpy(offsets), np.cumsum(counts))

    def test_take_high_dim_embedding(self, gpu_backend):
        """Gather embeddings (N=8192, D=64) on GPU."""
        import cupy as cp

        rng = np.random.default_rng(42)
        num_embeddings = 10_000
        embed_dim = 64
        batch_size = 8192
        table_np = rng.standard_normal((num_embeddings, embed_dim)).astype(np.float32)
        indices_np = rng.integers(0, num_embeddings, batch_size, dtype=np.int32)
        table = gpu_backend.tensor(table_np)
        indices = gpu_backend.tensor(indices_np)
        assert isinstance(table, cp.ndarray)
        result = gpu_backend.take(table, indices)
        assert isinstance(result, cp.ndarray), "Must stay on GPU, no host roundtrip"
        assert result.shape == (batch_size, embed_dim)
        expected = table_np[indices_np]
        np.testing.assert_allclose(gpu_backend.to_numpy(result), expected, rtol=1e-5)

    def test_argsort_gpu(self, gpu_backend):
        """Sort on GPU for routing."""
        import cupy as cp

        rng = np.random.default_rng(42)
        n = 50_000
        destinations = rng.integers(0, 32, n, dtype=np.int32)
        x = gpu_backend.tensor(destinations)
        indices = gpu_backend.argsort(x, dim=0, stable=True)
        assert isinstance(indices, cp.ndarray), "Must stay on GPU"
        sorted_vals = gpu_backend.to_numpy(x)[gpu_backend.to_numpy(indices)]
        assert np.all(sorted_vals[:-1] <= sorted_vals[1:])

    def test_scatter_position_pipeline(self, gpu_backend):
        """Full scatter position computation: argsort → bincount → cumsum → floor_divide → mod."""
        import cupy as cp

        rng = np.random.default_rng(42)
        n = 32_768
        world_size = 8
        num_cols = 4
        destinations = rng.integers(0, world_size, n, dtype=np.int32)
        x = gpu_backend.tensor(destinations, device="gpu")
        sort_idx = gpu_backend.argsort(x, stable=True)
        assert isinstance(sort_idx, cp.ndarray)
        counts = gpu_backend.bincount(x, minlength=world_size)
        assert isinstance(counts, cp.ndarray)
        offsets = gpu_backend.cumsum(counts, dim=0)
        assert isinstance(offsets, cp.ndarray)
        positions = gpu_backend.arange(0, n, dtype=gpu_backend.int32(), device="gpu")
        rows = gpu_backend.floor_divide(positions, num_cols)
        cols = gpu_backend.mod(positions, num_cols)
        assert isinstance(rows, cp.ndarray)
        assert isinstance(cols, cp.ndarray)
        expected_counts = np.bincount(destinations, minlength=world_size)
        np.testing.assert_array_equal(gpu_backend.to_numpy(counts), expected_counts)

    def test_dynamic_shapes_gpu_no_recompile(self, gpu_backend):
        """Verify CuPy handles dynamic shapes."""
        import cupy as cp

        rng = np.random.default_rng(42)
        sizes = [512, 1024, 2048, 4096, 8192, 16384, 3000, 7777]
        for size in sizes:
            destinations = rng.integers(0, 8, size, dtype=np.int32)
            x = gpu_backend.tensor(destinations)
            assert isinstance(x, cp.ndarray)
            counts = gpu_backend.bincount(x, minlength=8)
            assert isinstance(counts, cp.ndarray)
            offsets = gpu_backend.cumsum(counts, dim=0)
            assert isinstance(offsets, cp.ndarray)
            expected = np.cumsum(np.bincount(destinations, minlength=8))
            np.testing.assert_array_equal(gpu_backend.to_numpy(offsets), expected)

    def test_gpu_tensor_is_cupy(self, gpu_backend):
        """Sanity: factory methods return CuPy arrays when device='gpu'."""
        import cupy as cp

        assert isinstance(
            gpu_backend.zeros((4,), dtype=gpu_backend.float32(), device="gpu"),
            cp.ndarray,
        )
        assert isinstance(
            gpu_backend.ones((4,), dtype=gpu_backend.float32(), device="gpu"),
            cp.ndarray,
        )
        assert isinstance(
            gpu_backend.arange(0, 10, dtype=gpu_backend.int32(), device="gpu"),
            cp.ndarray,
        )
        assert isinstance(
            gpu_backend.full((4,), 7, dtype=gpu_backend.int32(), device="gpu"),
            cp.ndarray,
        )

    def test_full_routing_pipeline_gpu(self, gpu_backend):
        """Full GPU routing pipeline: argsort → bincount → cumsum."""
        import cupy as cp

        rng = np.random.default_rng(77)
        n = 65_536
        world_size = 16
        destinations_np = rng.integers(0, world_size, n, dtype=np.int32)
        x = gpu_backend.tensor(destinations_np)
        sort_idx = gpu_backend.argsort(x, stable=True)
        counts = gpu_backend.bincount(x, minlength=world_size)
        offsets = gpu_backend.cumsum(counts, dim=0)
        assert isinstance(sort_idx, cp.ndarray)
        assert isinstance(counts, cp.ndarray)
        assert isinstance(offsets, cp.ndarray)
        expected_counts = np.bincount(destinations_np, minlength=world_size)
        expected_offsets = np.cumsum(expected_counts)
        np.testing.assert_array_equal(gpu_backend.to_numpy(counts), expected_counts)
        np.testing.assert_array_equal(gpu_backend.to_numpy(offsets), expected_offsets)


# ============================================================
# CPU ↔ GPU Interop — Zero-Copy Transfer
# ============================================================


class TestCPUGPUInterop:
    """Verify data transfer between CPU and GPU backends."""

    def test_cpu_to_gpu_transfer(self, cpu_backend, gpu_backend):
        """CPU prepares routing table, transfers to GPU for execution."""
        import cupy as cp

        rng = np.random.default_rng(42)
        n = 10_000
        routing_table = rng.integers(0, 8, n, dtype=np.int32)
        cpu_tensor = cpu_backend.tensor(routing_table)
        gpu_tensor = gpu_backend.tensor(cpu_backend.to_numpy(cpu_tensor))
        assert isinstance(gpu_tensor, cp.ndarray)
        counts = gpu_backend.bincount(gpu_tensor, minlength=8)
        assert isinstance(counts, cp.ndarray)
        expected = np.bincount(routing_table, minlength=8)
        np.testing.assert_array_equal(gpu_backend.to_numpy(counts), expected)

    def test_gpu_to_cpu_transfer(self, cpu_backend, gpu_backend):
        """GPU computes results, transfers back to CPU for aggregation."""
        import cupy as cp

        rng = np.random.default_rng(42)
        n = 8192
        data = rng.standard_normal(n).astype(np.float32)
        gpu_data = gpu_backend.tensor(data)
        gpu_result = gpu_backend.cumsum(gpu_data, dim=0)
        assert isinstance(gpu_result, cp.ndarray)
        cpu_result = cpu_backend.tensor(gpu_backend.to_numpy(gpu_result))
        assert isinstance(cpu_result, np.ndarray)
        total = cpu_backend.item(cpu_backend.sum(cpu_result))
        assert isinstance(total, float)

    def test_dlpack_gpu_zero_copy(self, gpu_backend):
        """Verify DLPack zero-copy on GPU — no cudaMemcpy."""
        import cupy as cp

        data = gpu_backend.arange(0, 1000, dtype=gpu_backend.float32())
        assert isinstance(data, cp.ndarray)
        dlpack_capsule = gpu_backend.to_dlpack(data)
        reimported = gpu_backend.from_dlpack(
            type(
                "_DLPackWrapper",
                (),
                {
                    "__dlpack__": lambda self, **kwargs: dlpack_capsule,
                    "__dlpack_device__": lambda self: data.__dlpack_device__(),
                },
            )()
        )
        assert isinstance(reimported, cp.ndarray)
        np.testing.assert_array_equal(
            gpu_backend.to_numpy(reimported), gpu_backend.to_numpy(data)
        )

    def test_mixed_pipeline_cpu_gpu(self, cpu_backend, gpu_backend):
        """CPU preprocessing → GPU heavy compute → CPU postprocessing."""
        import cupy as cp

        rng = np.random.default_rng(42)
        n = 32_768
        world_size = 8
        embed_dim = 64
        raw_ids = rng.integers(0, 1_000_000, n, dtype=np.int64)
        destinations = (raw_ids % world_size).astype(np.int32)
        cpu_dests = cpu_backend.tensor(destinations)
        sort_idx = cpu_backend.argsort(cpu_dests, stable=True)
        sorted_dests = cpu_backend.to_numpy(cpu_dests)[cpu_backend.to_numpy(sort_idx)]
        assert np.all(sorted_dests[:-1] <= sorted_dests[1:])
        embeddings_np = rng.standard_normal((n, embed_dim)).astype(np.float32)
        gpu_embeddings = gpu_backend.tensor(embeddings_np)
        gpu_sort_idx = gpu_backend.tensor(cpu_backend.to_numpy(sort_idx))
        sorted_embeddings = gpu_backend.take(gpu_embeddings, gpu_sort_idx)
        assert isinstance(sorted_embeddings, cp.ndarray)
        assert sorted_embeddings.shape == (n, embed_dim)
        gpu_dests = gpu_backend.tensor(destinations)
        counts = gpu_backend.bincount(gpu_dests, minlength=world_size)
        offsets = gpu_backend.cumsum(counts, dim=0)
        assert isinstance(offsets, cp.ndarray)
        final_offsets = cpu_backend.tensor(gpu_backend.to_numpy(offsets))
        assert isinstance(final_offsets, np.ndarray)
        expected_counts = np.bincount(destinations, minlength=world_size)
        expected_offsets = np.cumsum(expected_counts)
        np.testing.assert_array_equal(
            cpu_backend.to_numpy(final_offsets), expected_offsets
        )

    def test_interop_results_numerically_identical(self, cpu_backend, gpu_backend):
        """CPU and GPU bincount+cumsum should produce identical results."""
        rng = np.random.default_rng(123)
        n = 50_000
        world_size = 8
        destinations = rng.integers(0, world_size, n, dtype=np.int32)
        cpu_x = cpu_backend.tensor(destinations)
        cpu_counts = cpu_backend.bincount(cpu_x, minlength=world_size)
        cpu_offsets = cpu_backend.cumsum(cpu_counts, dim=0)
        gpu_x = gpu_backend.tensor(destinations)
        gpu_counts = gpu_backend.bincount(gpu_x, minlength=world_size)
        gpu_offsets = gpu_backend.cumsum(gpu_counts, dim=0)
        np.testing.assert_array_equal(
            cpu_backend.to_numpy(cpu_offsets), gpu_backend.to_numpy(gpu_offsets)
        )


# ============================================================
# Edge Cases — Business-Critical
# ============================================================


class TestEdgeCases:
    """Edge cases that occur in production communication patterns."""

    def test_empty_batch(self, cpu_backend, gpu_backend):
        """A worker may receive zero features in a batch.

        Note: CuPy's bincount raises ValueError on empty arrays (no identity for CUB_MAX
        reduction). For empty batches the caller must guard before calling bincount.
        We test CPU path directly; GPU path tests the guard pattern.
        """
        # CPU: Numba JIT path handles empty arrays correctly
        empty_cpu = cpu_backend.tensor(np.array([], dtype=np.int32))
        counts_cpu = cpu_backend.bincount(empty_cpu, minlength=8)
        np.testing.assert_array_equal(
            cpu_backend.to_numpy(counts_cpu), np.zeros(8, dtype=np.int64)
        )

        # GPU: production guard pattern — skip bincount for empty inputs
        empty_gpu = gpu_backend.tensor(np.array([], dtype=np.int32))
        if gpu_backend.size(empty_gpu) == 0:
            counts_gpu = gpu_backend.zeros(
                (8,), dtype=gpu_backend.tensor(np.array([0], dtype=np.int64)).dtype
            )
        else:
            counts_gpu = gpu_backend.bincount(empty_gpu, minlength=8)
        np.testing.assert_array_equal(
            gpu_backend.to_numpy(counts_gpu), np.zeros(8, dtype=np.int64)
        )

    def test_single_destination(self, cpu_backend, gpu_backend):
        """All features go to the same worker (skewed distribution)."""
        for b in [cpu_backend, gpu_backend]:
            n = 10_000
            x = b.zeros((n,), dtype=b.int32())
            counts = b.bincount(b.cast(x, b.int32()), minlength=8)
            result = b.to_numpy(counts)
            assert result[0] == n
            assert np.sum(result[1:]) == 0

    def test_large_world_size(self, gpu_backend):
        """Production may have 128+ workers."""
        import cupy as cp

        rng = np.random.default_rng(42)
        world_size = 128
        n = 100_000
        destinations = rng.integers(0, world_size, n, dtype=np.int32)
        x = gpu_backend.tensor(destinations)
        counts = gpu_backend.bincount(x, minlength=world_size)
        assert isinstance(counts, cp.ndarray)
        assert counts.shape[0] >= world_size
        assert int(gpu_backend.sum(counts).item()) == n

    def test_weighted_bincount(self, cpu_backend):
        """Weighted bincount for computing per-worker data volumes."""
        rng = np.random.default_rng(42)
        n = 10_000
        world_size = 8
        destinations = rng.integers(0, world_size, n, dtype=np.int32)
        weights = rng.uniform(1.0, 100.0, n).astype(np.float64)
        x = cpu_backend.tensor(destinations)
        w = cpu_backend.tensor(weights)
        result = cpu_backend.bincount(x, weights=w, minlength=world_size)
        expected = np.bincount(destinations, weights=weights, minlength=world_size)
        np.testing.assert_allclose(cpu_backend.to_numpy(result), expected, rtol=1e-10)

    def test_single_element(self, cpu_backend, gpu_backend):
        """Single element batch — boundary condition."""
        for b in [cpu_backend, gpu_backend]:
            x = b.tensor(np.array([3], dtype=np.int32))
            counts = b.bincount(x, minlength=8)
            result = b.to_numpy(counts)
            assert result[3] == 1
            assert np.sum(result) == 1

    def test_cumsum_single_element(self, cpu_backend, gpu_backend):
        """Cumsum of single element."""
        for b in [cpu_backend, gpu_backend]:
            x = b.tensor(np.array([42], dtype=np.int64))
            result = b.cumsum(x, dim=0)
            np.testing.assert_array_equal(b.to_numpy(result), [42])

    def test_take_empty_indices(self, cpu_backend, gpu_backend):
        """Take with empty indices should return empty array."""
        for b in [cpu_backend, gpu_backend]:
            data = b.tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32))
            indices = b.tensor(np.array([], dtype=np.int32))
            result = b.take(data, indices)
            assert b.to_numpy(result).shape == (0,)

    def test_argsort_already_sorted(self, cpu_backend, gpu_backend):
        """Argsort of already-sorted array should return identity permutation."""
        for b in [cpu_backend, gpu_backend]:
            x = b.tensor(np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32))
            indices = b.argsort(x, dim=0, stable=True)
            np.testing.assert_array_equal(b.to_numpy(indices), np.arange(8))

    def test_large_world_size_cpu(self, cpu_backend):
        """CPU: 128 workers bincount."""
        rng = np.random.default_rng(42)
        world_size = 128
        n = 50_000
        destinations = rng.integers(0, world_size, n, dtype=np.int32)
        x = cpu_backend.tensor(destinations)
        counts = cpu_backend.bincount(x, minlength=world_size)
        assert isinstance(counts, np.ndarray)
        assert counts.shape[0] >= world_size
        assert int(counts.sum()) == n

    def test_minlength_larger_than_max_value(self, cpu_backend, gpu_backend):
        """minlength pads zeros when larger than actual range."""
        for b in [cpu_backend, gpu_backend]:
            x = b.tensor(np.array([0, 1, 2], dtype=np.int32))
            counts = b.bincount(x, minlength=16)
            result = b.to_numpy(counts)
            assert result.shape[0] == 16
            assert result[0] == 1
            assert result[1] == 1
            assert result[2] == 1
            assert np.sum(result[3:]) == 0


class TestHalfPrecisionSupport:
    """Half-precision (bfloat16, float16) tests across HybridBackend."""

    def test_bfloat16_creation_and_conversion(self, cpu_backend, gpu_backend):
        for b in [cpu_backend, gpu_backend]:
            arr = b.tensor([1.5, 2.5], dtype="bfloat16")
            assert str(arr.dtype) == "bfloat16"
            np_arr = b.to_numpy(arr)
            assert str(np_arr.dtype) == "bfloat16"
            assert np_arr.shape == (2,)

    def test_float16_creation_and_conversion(self, cpu_backend, gpu_backend):
        for b in [cpu_backend, gpu_backend]:
            arr = b.tensor([1.5, 2.5], dtype="float16")
            assert str(arr.dtype) == "float16"
            np_arr = b.to_numpy(arr)
            assert str(np_arr.dtype) == "float16"
            assert np_arr.shape == (2,)

    def test_dlpack_bfloat16_cpu_round_trip(self, cpu_backend):
        arr = np.array([1.5, 2.5, 3.5], dtype="bfloat16")
        cap = cpu_backend.to_dlpack(arr)
        adopted = cpu_backend.from_dlpack(cap)
        assert str(adopted.dtype) == "bfloat16"
        assert adopted.shape == (3,)
        np.testing.assert_array_equal(
            adopted.astype(np.float32), arr.astype(np.float32)
        )
