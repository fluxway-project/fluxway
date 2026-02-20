import pytest
import numpy as np
from fluxway.transport.backend import get_backend, use_backend


def test_jax_dlpack_slice_multidim():
    try:
        with use_backend("jax"):
            pass
    except ImportError:
        pytest.skip("JAX not available")

    with use_backend("jax"):
        backend = get_backend()
        # Shape (4, 4), values 0..15
        arr = backend.arange(0, 16, dtype=backend.int32()).reshape((4, 4))

        # Test 1: Simple Slice [1:3, 1:3] -> [[5, 6], [9, 10]]
        cursor = (slice(1, 3), slice(1, 3))
        dlpack = backend.get_dlpack_slice(arr, cursor)

        import jax

        if hasattr(dlpack, "__dlpack__"):
            # It's an object with __dlpack__ method
            view = jax.dlpack.from_dlpack(dlpack)
        elif hasattr(dlpack, "__cuda_array_interface__") or hasattr(
            dlpack, "__array_interface__"
        ):
            # It's a wrapper. Use torch to bridge (zero-copy)
            import torch

            # wrapper -> torch tensor (zero copy) -> dlpack -> jax
            # But we can just compare against torch tensor data to verify correctness
            t = torch.as_tensor(
                dlpack, device="cpu"
            )  # Assuming CPU for test simplicity or check device
            # Convert expected to numpy for comparison
            expected_np = backend.to_numpy(arr[1:3, 1:3])
            t_np = t.numpy()
            np.testing.assert_array_equal(t_np, expected_np)
        else:
            # Assume capsule
            view = jax.dlpack.from_dlpack(dlpack)
            expected = arr[1:3, 1:3]
            assert backend.equal(view, expected)

        # Test 2: Strided Slice [::2, ::2]
        cursor = (slice(None, None, 2), slice(None, None, 2))
        dlpack = backend.get_dlpack_slice(arr, cursor)

        if hasattr(dlpack, "__array_interface__"):
            t = torch.as_tensor(dlpack, device="cpu")
            expected_np = backend.to_numpy(arr[::2, ::2])
            np.testing.assert_array_equal(t.numpy(), expected_np)

        # Test 3: Mixed Access [1, :]
        cursor = (1, slice(None))
        dlpack = backend.get_dlpack_slice(arr, cursor)

        if hasattr(dlpack, "__array_interface__"):
            t = torch.as_tensor(dlpack, device="cpu")
            expected_np = backend.to_numpy(arr[1, :])
            np.testing.assert_array_equal(t.numpy(), expected_np)


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
