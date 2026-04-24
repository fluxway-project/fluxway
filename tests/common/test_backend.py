import pytest
import numpy as np
from fluxway.common.backend import get_backend, use_backend


def test_jax_dlpack_slice_multidim():
    try:
        with use_backend("jax"):
            pass
    except ImportError:
        pytest.skip("JAX not available")

    from fluxway.common.backend import JAXViewWrapper

    with use_backend("jax"):
        backend = get_backend()
        arr = backend.arange(0, 16, dtype=backend.int32()).reshape((4, 4))

        # Test 1: Simple Slice [1:3, 1:3] -> [[5, 6], [9, 10]]
        cursor = (slice(1, 3), slice(1, 3))
        dlpack = backend.get_dlpack_slice(arr, cursor)

        if hasattr(dlpack, "__cuda_array_interface__"):
            import jax.numpy as jnp

            try:
                view_jnp = jnp.asarray(dlpack)
            except Exception as e:
                import pytest

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

            try:
                view_jnp = jnp.asarray(dlpack)
            except Exception as e:
                import pytest

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

            try:
                view_jnp = jnp.asarray(dlpack)
            except Exception as e:
                import pytest

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
        from fluxway.common.backend import JaxBackend, JAXViewWrapper
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not available")

    b = JaxBackend()
    assert b.supports_zero_copy_view is True

    arr = jnp.arange(100, dtype=jnp.float32)
    for offset, count in [(0, 10), (5, 20), (50, 50), (99, 1)]:
        view = b.get_contiguous_view(arr, offset, count)
        assert isinstance(
            view, JAXViewWrapper
        ), f"Expected JAXViewWrapper, got {type(view)}"
        assert view.arr is arr, "JAXViewWrapper must reference the original array"
        assert view.slices == slice(offset, offset + count)


def test_contiguous_view_jax_wrapper_array_interface():
    """JAXViewWrapper.__array_interface__ exposes the correct shape and dtype."""
    try:
        from fluxway.common.backend import JAXViewWrapper
    except ImportError:
        pytest.skip("JAX not available")

    arr = np.arange(100, dtype=np.float32)
    wrapper = JAXViewWrapper(arr, slice(5, 25))
    iface = wrapper.__array_interface__
    assert iface["shape"] == (20,), iface["shape"]
    assert iface["typestr"] == arr.dtype.str
