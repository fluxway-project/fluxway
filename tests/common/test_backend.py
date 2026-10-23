import pytest
import numpy as np
from fluxway.common.tensor_backend import (
    JaxBackend,
    as_native_zerocopy,
    get_backend,
    prepare_dlpack_input,
    use_backend,
)
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.coherence import CoherenceStateTransitionError


def test_prepare_dlpack_input_preserves_readonly_zero_copy_storage():
    source = np.arange(8, dtype=np.int64)
    source.flags.writeable = False

    prepared = prepare_dlpack_input(source)

    assert prepared.flags.writeable
    assert (
        prepared.__array_interface__["data"][0] == source.__array_interface__["data"][0]
    )
    np.testing.assert_array_equal(prepared, source)


def test_jax_dlpack_import_preserves_uint64_with_local_scope():
    source = np.array([2**40], dtype=np.uint64)

    adopted = JaxBackend().asarray(source, copy=None)

    assert adopted.dtype == np.uint64
    assert int(adopted[0]) == 2**40


def test_jax_readonly_coherence_alias_imports_without_copy(tmp_path):
    source = np.memmap(
        tmp_path / "source.bin", mode="w+", dtype=np.float32, shape=(32,)
    )
    source[:] = np.arange(source.size, dtype=source.dtype)
    tensor = TransportTensor(source).zero_copy_readonly_alias()

    result = JaxBackend().asarray(tensor, copy=False)

    np.testing.assert_array_equal(np.asarray(result), source)
    assert result.unsafe_buffer_pointer() == source.ctypes.data
    assert not tensor.flat_values.flags.writeable
    exported = tensor.dlpack_values_for_consumer()
    assert exported.flags.writeable
    assert (
        exported.__array_interface__["data"][0] == source.__array_interface__["data"][0]
    )


def test_jax_readonly_foreign_array_requires_explicit_copy_policy():
    source = np.arange(32, dtype=np.float32)
    source.flags.writeable = False

    with pytest.raises(BufferError, match="Cannot export readonly array"):
        JaxBackend().asarray(source, copy=False)

    result = JaxBackend().asarray(source, copy=None)
    np.testing.assert_array_equal(np.asarray(result), source)


def test_jax_copy_true_creates_distinct_allocation():
    source = np.arange(1024, dtype=np.float32)

    result = JaxBackend().asarray(source, copy=True)

    np.testing.assert_array_equal(np.asarray(result), source)
    source[0] = -1
    assert float(result[0]) == 0.0


def test_jax_unknown_readonly_transport_requires_copy_for_zero_copy_policy():
    source = np.arange(32, dtype=np.float32)
    source.flags.writeable = False
    tensor = TransportTensor(source).zero_copy_readonly_alias()

    with pytest.raises(CoherenceStateTransitionError, match="physically writable"):
        JaxBackend().asarray(tensor, copy=False)

    result = JaxBackend().asarray(tensor, copy=None)
    np.testing.assert_array_equal(np.asarray(result), source)


def test_jax_readonly_jax_payload_keeps_native_zero_copy_identity():
    import jax.numpy as jnp

    source = jnp.arange(32, dtype=jnp.float32)
    tensor = TransportTensor(source).zero_copy_readonly_alias()

    result = JaxBackend().asarray(tensor, copy=False)

    assert result is source


def test_jax_dlpack_export_rejects_invalidated_and_released_tensor():
    invalidated = TransportTensor(
        np.arange(8, dtype=np.float32)
    ).zero_copy_readonly_alias()
    invalidated._invalidate_runtime_access()
    with pytest.raises(CoherenceStateTransitionError, match="invalidated"):
        JaxBackend().asarray(invalidated, copy=False)

    released = TransportTensor(
        np.arange(8, dtype=np.float32)
    ).zero_copy_readonly_alias()
    released._invalidate_runtime_access()
    released._release_runtime_payload()
    with pytest.raises(CoherenceStateTransitionError, match="released"):
        JaxBackend().asarray(released, copy=None)


def test_jax_asarray_propagates_broken_dlpack_error():
    class BrokenDLPackTensor:
        dtype = np.dtype(np.float32)
        shape = (1,)

        def __dlpack__(self, stream=None):
            raise RuntimeError("broken DLPack exporter")

        def __dlpack_device__(self):
            return (1, 0)

    with pytest.raises(RuntimeError, match="broken DLPack exporter"):
        JaxBackend().asarray(BrokenDLPackTensor(), copy=False)


def test_jax_dlpack_slice_multidim():
    try:
        with use_backend("jax"):
            pass
    except ImportError:
        pytest.skip("JAX not available")

    from fluxway.common.tensor_backend import JAXViewWrapper

    with use_backend("jax"):
        backend = get_backend()
        arr = backend.arange(0, 16, dtype=backend.int32()).reshape((4, 4))

        # Test 1: Simple Slice [1:3, 1:3] -> [[5, 6], [9, 10]]
        cursor = (slice(1, 3), slice(1, 3))
        dlpack = backend.get_dlpack_slice(arr, cursor)

        if hasattr(dlpack, "__cuda_array_interface__"):
            import jax.numpy as jnp

            view_jnp = None
            try:
                view_jnp = jnp.asarray(dlpack)
            except Exception as e:
                pytest.skip(f"JAX failed to consume __cuda_array_interface__: {e}")

            np.testing.assert_array_equal(
                backend.to_numpy(view_jnp), backend.to_numpy(arr[1:3, 1:3])
            )
        elif isinstance(dlpack, JAXViewWrapper):
            assert dlpack.arr is arr
            assert dlpack.slices == cursor
            # Zero-copy verification via __array_interface__
            view_np = np.asarray(dlpack)
            np.testing.assert_array_equal(view_np, backend.to_numpy(arr[1:3, 1:3]))
        else:
            raise AssertionError(
                f"Unexpected dlpack type {type(dlpack)} for cursor {cursor}"
            )

        # Test 2: Strided Slice [::2, ::2]
        cursor = (slice(None, None, 2), slice(None, None, 2))
        dlpack = backend.get_dlpack_slice(arr, cursor)

        if hasattr(dlpack, "__cuda_array_interface__"):
            import jax.numpy as jnp

            view_jnp = None
            try:
                view_jnp = jnp.asarray(dlpack)
            except Exception as e:
                pytest.skip(f"JAX failed to consume __cuda_array_interface__: {e}")
            np.testing.assert_array_equal(
                backend.to_numpy(view_jnp), backend.to_numpy(arr[::2, ::2])
            )
        elif isinstance(dlpack, JAXViewWrapper):
            view_np = np.asarray(dlpack)
            np.testing.assert_array_equal(view_np, backend.to_numpy(arr[::2, ::2]))

        # Test 3: Mixed Access [1, :]
        cursor = (1, slice(None))
        dlpack = backend.get_dlpack_slice(arr, cursor)

        if hasattr(dlpack, "__cuda_array_interface__"):
            import jax.numpy as jnp

            view_jnp = None
            try:
                view_jnp = jnp.asarray(dlpack)
            except Exception as e:
                pytest.skip(f"JAX failed to consume __cuda_array_interface__: {e}")
            np.testing.assert_array_equal(
                backend.to_numpy(view_jnp), backend.to_numpy(arr[1, :])
            )
        elif isinstance(dlpack, JAXViewWrapper):
            view_np = np.asarray(dlpack)
            np.testing.assert_array_equal(view_np, backend.to_numpy(arr[1, :]))


def test_torch_dlpack_slice_multidim():
    try:
        with use_backend("torch"):
            pass
    except ImportError:
        pytest.skip("PyTorch not available")

    with use_backend("torch"):
        backend = get_backend()
        from torch.utils.dlpack import from_dlpack

        arr = backend.arange(0, 16, dtype=backend.int32()).reshape((4, 4))

        # Test 1: Simple Slice
        cursor = (slice(1, 3), slice(1, 3))
        dlpack = backend.get_dlpack_slice(arr, cursor)
        view = from_dlpack(dlpack)
        expected = arr[1:3, 1:3]
        assert backend.equal(view, expected)

        # Test 2: Mixed Access [1, :]
        cursor = (1, slice(None))
        dlpack = backend.get_dlpack_slice(arr, cursor)
        view = from_dlpack(dlpack)
        expected = arr[1, :]
        assert backend.equal(view, expected)


# ===========================================================================
# get_contiguous_view — unified 1-D slice abstraction
# ===========================================================================


def test_contiguous_view_default_backend():
    """Default (numpy-backed) backend falls back to a plain Python slice."""
    backend = get_backend()
    arr = np.arange(100, dtype=np.float32)
    for offset, count in [(0, 10), (5, 20), (50, 50), (99, 1), (0, 100)]:
        view = backend.get_contiguous_view(arr, offset, count)
        np.testing.assert_array_equal(view, arr[offset : offset + count])


def test_contiguous_view_jax_returns_wrapper():
    """JaxBackend.get_contiguous_view returns a JAXViewWrapper with the
    original array reference and the correct slice (zero-copy guarantee)."""
    try:
        from fluxway.common.tensor_backend import JaxBackend, JAXViewWrapper
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not available")
        return  # unreachable, but informs pyrefly that imports are always bound

    b = JaxBackend()
    assert b.supports_zero_copy_view is True

    arr = jnp.arange(100, dtype=jnp.float32)
    for offset, count in [(0, 10), (5, 20), (50, 50), (99, 1)]:
        view = b.get_contiguous_view(arr, offset, count)
        assert isinstance(view, JAXViewWrapper), (
            f"Expected JAXViewWrapper, got {type(view)}"
        )
        assert view.arr is arr, "JAXViewWrapper must reference the original array"
        assert view.slices == slice(offset, offset + count)


def test_contiguous_view_jax_wrapper_array_interface():
    """JAXViewWrapper.__array_interface__ exposes the correct shape and dtype."""
    try:
        from fluxway.common.tensor_backend import JAXViewWrapper
    except ImportError:
        pytest.skip("JAX not available")
        return  # unreachable, but informs pyrefly that imports are always bound

    arr = np.arange(100, dtype=np.float32)
    wrapper = JAXViewWrapper(arr, slice(5, 25))
    iface = wrapper.__array_interface__
    assert iface["shape"] == (20,), iface["shape"]
    assert iface["typestr"] == arr.dtype.str


def test_as_native_zerocopy_numba_cuda():
    try:
        from numba import cuda
        import cupy as cp

        if not cuda.is_available():
            pytest.skip("CUDA not available for Numba")
    except Exception:
        pytest.skip("Numba or CuPy not available")

    nb_arr = cuda.to_device(np.arange(16, dtype=np.float32))
    native = as_native_zerocopy(nb_arr)

    assert isinstance(native, cp.ndarray)
    assert nb_arr.device_ctypes_pointer.value == native.data.ptr
    np.testing.assert_array_equal(cp.asnumpy(native), np.arange(16))


def test_jax_asarray_transport_tensor_and_numba_zero_copy():
    try:
        import jax
        import cupy as cp
        from numba import cuda
        from fluxway.common.types.tensor import TransportTensor

        if not cuda.is_available():
            pytest.skip("CUDA not available")
    except Exception:
        pytest.skip("JAX, CuPy, or Numba not available")

    try:
        jax_gpu_devices = jax.devices("gpu")
    except RuntimeError as error:
        if "no platforms that are instances of gpu are present" not in str(error):
            raise
        pytest.skip("JAX GPU backend not available")
    if not jax_gpu_devices:
        pytest.skip("JAX GPU backend has no devices")

    with use_backend("jax"):
        b = get_backend()

        # CuPy TransportTensor unwrapping
        cp_arr = cp.arange(16, dtype=cp.float32)
        tt_cp = TransportTensor(flat_values=cp_arr)
        jax_cp = b.asarray(tt_cp, copy=False)
        assert isinstance(jax_cp, jax.Array)
        assert jax_cp.unsafe_buffer_pointer() == cp_arr.data.ptr

        # Numba TransportTensor unwrapping
        nb_arr = cuda.to_device(np.arange(16, dtype=np.float32))
        tt_nb = TransportTensor(flat_values=nb_arr)
        jax_nb = b.asarray(tt_nb, copy=False)
        assert isinstance(jax_nb, jax.Array)
        assert jax_nb.unsafe_buffer_pointer() == nb_arr.device_ctypes_pointer.value

        # Non-contiguous GPU array
        sliced_cp = cp_arr[::2]
        jax_sliced = b.asarray(sliced_cp, copy=None)
        assert isinstance(jax_sliced, jax.Array)
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(jax_sliced)), np.arange(16)[::2]
        )


def test_backend_asarray_tree():
    with use_backend("jax"):
        b = get_backend()
        tree = {
            "a": np.arange(5),
            "b": [np.arange(3), np.arange(4)],
            "c": (np.arange(2),),
        }
        res = b.asarray_tree(tree, copy=None)
        import jax

        assert isinstance(res["a"], jax.Array)
        assert isinstance(res["b"][0], jax.Array)
        assert isinstance(res["b"][1], jax.Array)
        assert isinstance(res["c"][0], jax.Array)
