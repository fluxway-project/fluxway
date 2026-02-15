from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import contextlib


class TensorBackend(ABC):
    """
    Abstract interface for tensor operations.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def tensor(self, data: Any, dtype: Any = None) -> Any:
        pass

    @abstractmethod
    def as_tensor(self, data: Any, dtype: Any = None) -> Any:
        pass

    @abstractmethod
    def to_numpy(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        pass

    @abstractmethod
    def ones(self, shape: Any, dtype: Any = None) -> Any:
        pass

    @abstractmethod
    def full(self, shape: Any, fill_value: Any, dtype: Any = None) -> Any:
        pass

    @abstractmethod
    def arange(self, start: int, end: int, step: int = 1, dtype: Any = None) -> Any:
        pass

    @abstractmethod
    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        pass

    @abstractmethod
    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        pass

    @abstractmethod
    def equal(self, x: Any, y: Any) -> bool:
        pass

    @abstractmethod
    def float32(self) -> Any:
        pass

    @abstractmethod
    def int32(self) -> Any:
        pass

    @abstractmethod
    def is_tensor(self, obj: Any) -> bool:
        pass

    @abstractmethod
    def shape(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def item(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        pass

    @abstractmethod
    def unravel_index(self, indices: Any, shape: Tuple[int, ...]) -> Any:
        """Converts flat indices into tuple of coordinate arrays."""
        pass

    @abstractmethod
    def floor_divide(self, x: Any, y: Any) -> Any:
        pass

    # --- Algebraic / Vectorized Ops ---
    @abstractmethod
    def mod(self, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def argsort(
        self, x: Any, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> Any:
        pass

    @abstractmethod
    def sort(
        self, x: Any, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> Tuple[Any, Any]:
        """Returns (sorted_values, indices)"""
        pass

    @abstractmethod
    def gather(self, x: Any, dim: int, index: Any) -> Any:
        pass

    @abstractmethod
    def take(self, x: Any, indices: Any) -> Any:
        """Flattened take/gather"""
        pass

    @abstractmethod
    def select(self, x: Any, dim: int, index: int) -> Any:
        """Select a single slice along a dimension (x.select(dim, index) or x[:, index])"""
        pass

    @abstractmethod
    def diff(self, x: Any, dim: int = -1) -> Any:
        pass

    @abstractmethod
    def unique_consecutive(
        self, x: Any, return_inverse: bool = False, return_counts: bool = False
    ) -> Any:
        """
        Returns unique elements that are consecutive.
        Similar to torch.unique_consecutive.
        """
        pass

    @abstractmethod
    def sum(self, x: Any, dim: Any = None) -> Any:
        pass

    @abstractmethod
    def max(self, x: Any) -> Any:
        pass

    @abstractmethod
    def min(self, x: Any) -> Any:
        pass

    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        pass

    @abstractmethod
    def repeat_interleave(self, x: Any, repeats: Any, dim: int = 0) -> Any:
        pass

    @abstractmethod
    def cumsum(self, x: Any, dim: int = 0) -> Any:
        pass

    @abstractmethod
    def nonzero(self, x: Any) -> Any:
        pass

    @abstractmethod
    def dynamic_slice(self, x: Any, start: Any, size: Any) -> Any:
        """Slice x starting at `start` with shape `size`. `start` can be a tensor."""
        pass

    @abstractmethod
    def bincount(self, x: Any, minlength: int = 0) -> Any:
        """Count occurrences of each value in array of non-negative ints."""
        pass

    @abstractmethod
    def vmap(self, fun: Any, in_axes: Any = 0, out_axes: Any = 0) -> Any:
        """Vectorizing map."""
        pass

    @abstractmethod
    def tree_map(self, fun: Any, tree: Any) -> Any:
        """Maps fun over pytree leaves."""
        pass

    @abstractmethod
    def tile(self, x: Any, dims: Tuple[int, ...]) -> Any:
        """Constructs a new tensor by repeating input."""
        pass


class TorchBackend(TensorBackend):
    def __init__(self):
        import torch

        self._torch = torch

    def name(self) -> str:
        return "torch"

    def tensor(self, data: Any, dtype: Any = None) -> Any:
        return self._torch.tensor(data, dtype=dtype)

    def as_tensor(self, data: Any, dtype: Any = None) -> Any:
        return self._torch.as_tensor(data, dtype=dtype)

    def to_numpy(self, tensor: Any) -> Any:
        if isinstance(tensor, list):
            import numpy as np

            return np.array(tensor)
        return tensor.detach().cpu().numpy()

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        return self._torch.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None) -> Any:
        return self._torch.ones(shape, dtype=dtype)

    def full(self, shape: Any, fill_value: Any, dtype: Any = None) -> Any:
        return self._torch.full(shape, fill_value, dtype=dtype)

    def arange(self, start: int, end: int, step: int = 1, dtype: Any = None) -> Any:
        return self._torch.arange(start, end, step, dtype=dtype)

    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        return self._torch.cat(tensors, dim=dim)

    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        return self._torch.stack(tensors, dim=dim)

    def equal(self, x: Any, y: Any) -> bool:
        return self._torch.equal(x, y)

    def float32(self) -> Any:
        return self._torch.float32

    def int32(self) -> Any:
        return self._torch.int32

    def is_tensor(self, obj: Any) -> bool:
        return self._torch.is_tensor(obj)

    def shape(self, tensor: Any) -> Any:
        return tensor.shape

    def item(self, tensor: Any) -> Any:
        return tensor.item()

    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        return self._torch.reshape(x, shape)

    def unravel_index(self, indices: Any, shape: Tuple[int, ...]) -> Any:
        if hasattr(self._torch, "unravel_index"):
            return self._torch.unravel_index(indices, shape)

        # Fallback implementation
        res = []
        curr = indices
        for dim in reversed(shape):
            res.append(curr % dim)
            curr = curr // dim
        return tuple(reversed(res))

    def floor_divide(self, x: Any, y: Any) -> Any:
        return self._torch.floor_divide(x, y)

    # --- Algebraic Ops ---
    def mod(self, x: Any, y: Any) -> Any:
        return x % y

    def argsort(
        self, x: Any, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> Any:
        return self._torch.argsort(x, dim=dim, descending=descending, stable=stable)

    def sort(
        self, x: Any, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> Tuple[Any, Any]:
        return self._torch.sort(x, dim=dim, descending=descending, stable=stable)

    def gather(self, x: Any, dim: int, index: Any) -> Any:
        return self._torch.gather(x, dim, index)

    def take(self, x: Any, indices: Any) -> Any:
        return x[indices]  # Torch supports indexing by tensor

    def select(self, x: Any, dim: int, index: int) -> Any:
        return x.select(dim, index)

    def diff(self, x: Any, dim: int = -1) -> Any:
        return self._torch.diff(x, dim=dim)

    def unique_consecutive(
        self, x: Any, return_inverse: bool = False, return_counts: bool = False
    ) -> Any:
        return self._torch.unique_consecutive(
            x, return_inverse=return_inverse, return_counts=return_counts
        )

    def sum(self, x: Any, dim: Any = None) -> Any:
        return self._torch.sum(x, dim=dim)

    def max(self, x: Any) -> Any:
        return self._torch.max(x)

    def min(self, x: Any) -> Any:
        return self._torch.min(x)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return self._torch.where(condition, x, y)

    def repeat_interleave(self, x: Any, repeats: Any, dim: int = None) -> Any:
        return self._torch.repeat_interleave(x, repeats, dim=dim)

    def cumsum(self, x: Any, dim: int = 0) -> Any:
        return self._torch.cumsum(x, dim=dim)

    def nonzero(self, x: Any) -> Any:
        return self._torch.nonzero(x, as_tuple=False)

    def dynamic_slice(self, x: Any, start: Any, size: Any) -> Any:
        """Simulates dynamic slicing using index_select."""
        # Note: robust for dynamic start without CPU sync if we index manually
        for i, (s, l) in enumerate(zip(start, size)):
            idx = self._torch.arange(l, device=x.device, dtype=self._torch.int64) + s
            x = self._torch.index_select(x, i, idx)

        return x

    def bincount(self, x: Any, minlength: int = 0) -> Any:
        return self._torch.bincount(x, minlength=minlength)

    def vmap(self, fun: Any, in_axes: Any = 0, out_axes: Any = 0) -> Any:
        if hasattr(self._torch, "vmap"):
            return self._torch.vmap(fun, in_dims=in_axes, out_dims=out_axes)
        raise NotImplementedError("torch.vmap requires newer PyTorch version")

    def tree_map(self, fun: Any, tree: Any) -> Any:
        try:
            import torch.utils._pytree as pytree

            return pytree.tree_map(fun, tree)
        except ImportError:
            # Fallback for simple structures
            if isinstance(tree, (list, tuple)):
                return type(tree)(fun(x) for x in tree)
            return fun(tree)

    def tile(self, x: Any, dims: Tuple[int, ...]) -> Any:
        return self._torch.tile(x, dims)


class JaxBackend(TensorBackend):
    def __init__(self):
        import jax.numpy as jnp
        import jax
        import numpy as np

        self._jnp = jnp
        self._jax = jax
        self._np = np

    def name(self) -> str:
        return "jax"

    def tensor(self, data: Any, dtype: Any = None) -> Any:
        return self._jnp.array(data, dtype=dtype)

    def as_tensor(self, data: Any, dtype: Any = None) -> Any:
        return self._jnp.asarray(data, dtype=dtype)

    def to_numpy(self, tensor: Any) -> Any:
        return self._np.array(tensor)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        return self._jnp.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None) -> Any:
        return self._jnp.ones(shape, dtype=dtype)

    def full(self, shape: Any, fill_value: Any, dtype: Any = None) -> Any:
        return self._jnp.full(shape, fill_value, dtype=dtype)

    def arange(self, start: int, end: int, step: int = 1, dtype: Any = None) -> Any:
        return self._jnp.arange(start, end, step, dtype=dtype)

    def cat(self, tensors: List[Any], dim: int = 0) -> Any:
        return self._jnp.concatenate(tensors, axis=dim)

    def stack(self, tensors: List[Any], dim: int = 0) -> Any:
        return self._jnp.stack(tensors, axis=dim)

    def equal(self, x: Any, y: Any) -> bool:
        return self._jnp.array_equal(x, y)

    def float32(self) -> Any:
        return self._jnp.float32

    def int32(self) -> Any:
        return self._jnp.int32

    def is_tensor(self, obj: Any) -> bool:
        return isinstance(obj, (self._jnp.ndarray, self._np.ndarray))

    def shape(self, tensor: Any) -> Any:
        return tensor.shape

    def item(self, tensor: Any) -> Any:
        return tensor.item()

    def reshape(self, x: Any, shape: Tuple[int, ...]) -> Any:
        return self._jnp.reshape(x, shape)

    def unravel_index(self, indices: Any, shape: Tuple[int, ...]) -> Any:
        return self._jnp.unravel_index(indices, shape)

    def floor_divide(self, x: Any, y: Any) -> Any:
        return self._jnp.floor_divide(x, y)

    # --- Algebraic Ops ---
    def mod(self, x: Any, y: Any) -> Any:
        return self._jnp.mod(x, y)

    def argsort(
        self, x: Any, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> Any:
        # JAX sort/argsort are stable by default.
        if descending:
            # JAX argsort doesn't have descending flag, need to sort -x
            # Assumes numerical
            return self._jnp.argsort(-x, axis=dim)
        return self._jnp.argsort(x, axis=dim)

    def sort(
        self, x: Any, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> Tuple[Any, Any]:
        """Returns (sorted_values, indices)."""
        idx = self.argsort(x, dim, descending, stable=stable)
        val = self._jnp.take_along_axis(x, idx, axis=dim)
        return val, idx

    def gather(self, x: Any, dim: int, index: Any) -> Any:
        return self._jnp.take_along_axis(x, index, axis=dim)

    def take(self, x: Any, indices: Any) -> Any:
        return x[indices]

    def select(self, x: Any, dim: int, index: int) -> Any:
        return self._jnp.take(x, index, axis=dim)

    def diff(self, x: Any, dim: int = -1) -> Any:
        return self._jnp.diff(x, axis=dim)

    def unique_consecutive(
        self, x: Any, return_inverse: bool = False, return_counts: bool = False
    ) -> Any:
        # JAX implementation of unique_consecutive using diff
        mask = self._jnp.concatenate([self._jnp.array([True]), x[1:] != x[:-1]])
        uniques = x[mask]

        if return_inverse:
            # Inverse requires cumsum of mask (indices change when mask is true)
            inverse = self._jnp.cumsum(mask) - 1
            if return_counts:
                indices = self._jnp.flatnonzero(mask)
                appended = self._jnp.append(indices, self._jnp.array([len(x)]))
                counts = self._jnp.diff(appended)
                return uniques, inverse, counts
            return uniques, inverse

        if return_counts:
            indices = self._jnp.flatnonzero(mask)
            appended = self._jnp.append(indices, self._jnp.array([len(x)]))
            counts = self._jnp.diff(appended)
            return uniques, counts

        return uniques

    def sum(self, x: Any, dim: Any = None) -> Any:
        return self._jnp.sum(x, axis=dim)

    def max(self, x: Any) -> Any:
        return self._jnp.max(x)

    def min(self, x: Any) -> Any:
        return self._jnp.min(x)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return self._jnp.where(condition, x, y)

    def repeat_interleave(self, x: Any, repeats: Any, dim: int = 0) -> Any:
        # JAX doesn't have repeat_interleave exactly matching torch's signature for tensors
        # jnp.repeat repeats elements.
        return self._jnp.repeat(x, repeats, axis=dim, total_repeat_length=None)

    def cumsum(self, x: Any, dim: int = 0) -> Any:
        return self._jnp.cumsum(x, axis=dim)

    def nonzero(self, x: Any) -> Any:
        return self._jnp.argwhere(x)

    def dynamic_slice(self, x: Any, start: Any, size: Any) -> Any:
        return self._jax.lax.dynamic_slice(x, start, size)

    def bincount(self, x: Any, minlength: int = 0) -> Any:
        return self._jnp.bincount(x, length=minlength)

    def vmap(self, fun: Any, in_axes: Any = 0, out_axes: Any = 0) -> Any:
        return self._jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)

    def tree_map(self, fun: Any, tree: Any) -> Any:
        return self._jax.tree.map(fun, tree)

    def tile(self, x: Any, dims: Tuple[int, ...]) -> Any:
        return self._jnp.tile(x, dims)


_BACKEND: Optional[TensorBackend] = None


def get_backend() -> TensorBackend:
    global _BACKEND
    if _BACKEND is None:
        try:
            _BACKEND = JaxBackend()
        except ImportError:
            try:
                _BACKEND = TorchBackend()
            except ImportError:
                raise ImportError("Neither JAX nor PyTorch found.") from None
    return _BACKEND


def set_backend(name: str):
    global _BACKEND
    if name.lower() == "jax":
        _BACKEND = JaxBackend()
    elif name.lower() in ["torch", "pytorch"]:
        _BACKEND = TorchBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")


@contextlib.contextmanager
def use_backend(name: str):
    global _BACKEND
    old_backend = _BACKEND
    try:
        set_backend(name)
        yield
    finally:
        _BACKEND = old_backend
