from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import contextlib


class DLPackObject:
    """Wrapper to convert a PyCapsule into a DLPack-compatible object.

    This facilitates zero-copy tensor sharing between different frameworks.
    """

    def __init__(self, capsule, device=None):
        self._capsule = capsule
        self._device = device

    def __dlpack__(self, stream=None):
        return self._capsule

    def __dlpack_device__(self):
        if self._device is None:
            return (1, 0)
        return self._device


class TensorBackend(ABC):
    """Abstract interface for tensor operations across different frameworks.

    This class defines a unified API for common tensor operations like creation,
    manipulation, and mathematical functions, enabling backend-agnostic code.
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
    def dtype(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def shape(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def size(self, tensor: Any) -> int:
        """Returns the total number of elements in the tensor."""
        pass

    @abstractmethod
    def print(self, fmt: str, **kwargs) -> None:
        """Backend-safe print (e.g., jax.debug.print)."""
        pass

    @abstractmethod
    def cast(self, tensor: Any, dtype: Any) -> Any:
        """Casts the tensor to the specified dtype."""
        pass

    @abstractmethod
    def item(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def reshape(self, tensor: Any, shape: Tuple[int, ...]) -> Any:
        pass

    @abstractmethod
    def unravel_index(self, indices: Any, shape: Tuple[int, ...]) -> Any:
        """Converts flat indices into tuple of coordinate arrays."""
        pass

    @abstractmethod
    def is_scalar(self, obj: Any) -> bool:
        """Check if obj is a scalar tensor (ndim == 0)."""
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
    def unique(
        self,
        x: Any,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
    ) -> Any:
        """Returns the unique elements of the input tensor."""
        pass

    @abstractmethod
    def sum(self, x: Any, dim: Any = None) -> Any:
        pass

    @abstractmethod
    def mean(self, x: Any) -> Any:
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
    def repeat_interleave(
        self,
        x: Any,
        repeats: Any,
        dim: Optional[int] = None,
        total_repeat_length: Optional[int] = None,
    ) -> Any:
        pass

    @abstractmethod
    def cumsum(self, x: Any, dim: int = 0) -> Any:
        pass

    @abstractmethod
    def nonzero(self, x: Any, *, size: Any = None, fill_value: Any = None) -> Any:
        pass

    @abstractmethod
    def dynamic_slice(self, x: Any, start: Any, size: Any) -> Any:
        """Slice x starting at `start` with shape `size`. `start` can be a tensor."""
        pass

    @abstractmethod
    def dynamic_slice_in_dim(self, x: Any, start: Any, size: int, axis: int = 0) -> Any:
        """Slice x along a specific axis starting at `start` with length `size`."""
        pass

    @abstractmethod
    def dynamic_slice_update(self, x: Any, update: Any, start: Any) -> Any:
        """Update x starting at `start` with `update`. Returns updated tensor."""
        pass

    @abstractmethod
    def bincount(
        self, x: Any, weights: Optional[Any] = None, minlength: int = 0
    ) -> Any:
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
    def tree_flatten(self, tree: Any) -> Tuple[list, Any]:
        """Flatten a pytree into (leaves_list, treedef)."""
        pass

    @abstractmethod
    def tile(self, x: Any, dims: Tuple[int, ...]) -> Any:
        """Constructs a new tensor by repeating input."""
        pass

    @abstractmethod
    def scan(self, f: Any, init: Any, xs: Any, length: Optional[int] = None) -> Any:
        """
        Scans a function over leading array axes while carrying along state.

        Args:
            f: Function of the form (carry, x) -> (carry, y)
            init: Initial carry value
            xs: Input arrays to scan over
            length: Optional length of scan (if xs is None)

        Returns:
            (final_carry, stacked_outputs)
        """
        pass

    @abstractmethod
    def get_dlpack_slice(self, tensor: Any, cursor: Optional[Any] = None) -> Any:
        """
        Returns a DLPack-compatible object viewing a slice of the tensor without copying.
        Typically used for ragged/sparse transport optimization.
        """
        pass

    @abstractmethod
    def index_update(self, x: Any, indices: Any, values: Any) -> Any:
        """
        Perform indexed update: x[indices] = values.
        Returns the updated tensor (may be in-place for some backends).
        """
        pass

    @property
    def supports_zero_copy_view(self) -> bool:
        return False

    def get_contiguous_view(self, tensor: Any, offset: int, count: int) -> Any:
        return tensor[offset : offset + count]


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

    def is_scalar(self, obj: Any) -> bool:
        return self._torch.is_tensor(obj) and obj.ndim == 0

    def dtype(self, tensor: Any) -> Any:
        return tensor.dtype

    def shape(self, tensor: Any) -> Any:
        return tensor.shape

    def size(self, tensor: Any) -> int:
        return tensor.numel()

    def cast(self, tensor: Any, dtype: Any) -> Any:
        return tensor.to(dtype)

    def item(self, tensor: Any) -> Any:
        return tensor.item()

    def reshape(self, tensor: Any, shape: Tuple[int, ...]) -> Any:
        return self._torch.reshape(tensor, shape)

    def unravel_index(self, indices: Any, shape: Tuple[int, ...]) -> Any:
        try:
            return self._torch.unravel_index(indices, shape)
        except AttributeError:
            # Fallback implementation for older PyTorch versions (< 1.10)
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

    def print(self, fmt: str, **kwargs) -> None:
        print(fmt.format(**kwargs))

    def unique_consecutive(
        self, x: Any, return_inverse: bool = False, return_counts: bool = False
    ) -> Any:
        return self._torch.unique_consecutive(
            x, return_inverse=return_inverse, return_counts=return_counts
        )

    def sum(self, x: Any, dim: Any = None) -> Any:
        if dim is not None:
            return x.sum(dim=dim)
        return x.sum()

    def mean(self, x: Any) -> Any:
        return x.mean()

    def max(self, x: Any) -> Any:
        return self._torch.max(x)

    def min(self, x: Any) -> Any:
        return self._torch.min(x)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return self._torch.where(condition, x, y)

    def repeat_interleave(
        self,
        x: Any,
        repeats: Any,
        dim: Optional[int] = None,
        total_repeat_length: Optional[int] = None,
    ) -> Any:
        return self._torch.repeat_interleave(
            x, repeats, dim=dim, output_size=total_repeat_length
        )

    def cumsum(self, x: Any, dim: int = 0) -> Any:
        return self._torch.cumsum(x, dim=dim)

    def nonzero(self, x: Any, *, size: Any = None, fill_value: Any = None) -> Any:
        return self._torch.nonzero(x, as_tuple=False)

    def dynamic_slice(self, x: Any, start: Any, size: Any) -> Any:
        """Simulates dynamic slicing using index_select."""
        # Note: robust for dynamic start without CPU sync if we index manually
        for i, (s, l) in enumerate(zip(start, size, strict=True)):  # type: ignore
            idx = self._torch.arange(l, device=x.device, dtype=self._torch.int64) + s
            x = self._torch.index_select(x, i, idx)

        return x

    def dynamic_slice_in_dim(self, x: Any, start: Any, size: int, axis: int = 0) -> Any:
        idx = self._torch.arange(size, device=x.device, dtype=self._torch.int64) + start
        return self._torch.index_select(x, axis, idx)

    def dynamic_slice_update(self, x: Any, update: Any, start: Any) -> Any:
        """Simulates dynamic update (in-place for Torch).

        Args:
            x: Target tensor to be updated.
            update: Tensor containing the update values.
            start: Tuple of starting indices.

        Returns:
            The updated tensor.
        """
        if isinstance(start, int):
            x[start : start + update.shape[0]] = update
        elif isinstance(start, (list, tuple)):
            slices = []
            for i, s in enumerate(start):
                slice_len = update.shape[i]
                slices.append(slice(s, s + slice_len))
            x[tuple(slices)] = update
        elif self.is_scalar(start):
            s = int(start)
            x[s : s + update.shape[0]] = update
        else:
            raise NotImplementedError(f"Unsupported start type: {type(start)}")

        return x

    def bincount(
        self, x: Any, weights: Optional[Any] = None, minlength: int = 0
    ) -> Any:
        return self._torch.bincount(x, weights=weights, minlength=minlength)

    def unique(
        self,
        x: Any,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
    ) -> Any:
        # Note: return_index is not directly supported in torch.unique for 1D.
        # But we can emulate it if needed. For now, we assume return_inverse.
        return self._torch.unique(
            x, return_inverse=return_inverse, return_counts=return_counts
        )

    def vmap(self, fun: Any, in_axes: Any = 0, out_axes: Any = 0) -> Any:
        return self._torch.vmap(fun, in_dims=in_axes, out_dims=out_axes)

    def tree_map(self, fun: Any, tree: Any) -> Any:
        try:
            import torch.utils._pytree as pytree

            return pytree.tree_map(fun, tree)
        except ImportError:
            # Fallback for simple structures
            if isinstance(tree, (list, tuple)):
                return type(tree)(fun(x) for x in tree)
            return fun(tree)

    def tree_flatten(self, tree: Any) -> Tuple[list, Any]:
        try:
            import torch.utils._pytree as pytree

            return pytree.tree_flatten(tree)
        except ImportError:
            if isinstance(tree, (list, tuple)):
                return list(tree), type(tree)
            elif isinstance(tree, dict):
                return list(tree.values()), tree.keys()
            return [tree], None

    def tile(self, x: Any, dims: Tuple[int, ...]) -> Any:
        return self._torch.tile(x, dims)

    def scan(self, f: Any, init: Any, xs: Any, length: Optional[int] = None) -> Any:
        """PyTorch implementation of scan using a simple loop."""
        carry = init
        ys = []

        if xs is None:
            if length is None:
                raise ValueError("length must be provided if xs is None")
            xs_iter = range(length)
        else:
            if self.is_tensor(xs):
                xs_iter = xs
            else:
                raise NotImplementedError(
                    "Tuple xs scan not fully supported in simple torch loop yet"
                )

        for x in xs_iter:
            carry, y = f(carry, x)
            ys.append(y)

        if not ys:
            return carry, []

        return carry, self.stack(ys)

    def get_dlpack_slice(self, tensor: Any, cursor: Optional[Any] = None) -> Any:
        from torch.utils.dlpack import to_dlpack

        sliced_tensor = tensor
        if cursor is not None:
            if isinstance(cursor, (int, slice, tuple, list)):
                sliced_tensor = tensor[cursor]
            else:
                # Fallback for other types?
                sliced_tensor = tensor[cursor]

        # Helper to extract device tuple for DLPack
        # DLDeviceType: kDLCPU=1, kDLCUDA=2, kDLCUDAHost=3, ...
        # Torch devices map: cpu -> (1, 0), cuda:N -> (2, N)
        device_tuple = (1, 0)
        if sliced_tensor.is_cuda:
            device_tuple = (
                2,
                (
                    sliced_tensor.device.index
                    if sliced_tensor.device.index is not None
                    else 0
                ),
            )

        return DLPackObject(to_dlpack(sliced_tensor), device=device_tuple)

    def index_update(self, x: Any, indices: Any, values: Any) -> Any:
        x[indices] = values
        return x

    @property
    def supports_zero_copy_view(self) -> bool:
        return True

    def get_contiguous_view(self, tensor: Any, offset: int, count: int) -> Any:
        return tensor[offset : offset + count]


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
        return isinstance(obj, (self._jnp.ndarray, self._np.ndarray, JAXViewWrapper))

    def is_scalar(self, obj: Any) -> bool:
        return self.is_tensor(obj) and obj.ndim == 0

    def dtype(self, tensor: Any) -> Any:
        return tensor.dtype

    def shape(self, tensor: Any) -> Any:
        return tensor.shape

    def reshape(self, tensor: Any, shape: Any) -> Any:
        return self._jnp.reshape(tensor, shape)

    def view(self, tensor: Any, shape: Any) -> Any:
        return self._jnp.reshape(tensor, shape)

    def size(self, tensor: Any) -> int:
        return tensor.size

    def cast(self, tensor: Any, dtype: Any) -> Any:
        return tensor.astype(dtype)

    def item(self, tensor: Any) -> Any:
        return tensor.item()

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

    def print(self, fmt: str, **kwargs) -> None:
        self._jax.debug.print(fmt, **kwargs)

    def sort(
        self, x: Any, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> Tuple[Any, Any]:
        """Returns (sorted_values, indices)."""
        idx = self.argsort(x, dim, descending, stable=stable)
        val = self._jnp.take_along_axis(x, idx, axis=dim)
        return val, idx

    def gather(self, x: Any, dim: int, index: Any) -> Any:
        # Avoid JAX gather error for zero-sized dimensions in JIT
        if x.shape[dim] == 0:
            return self._jnp.zeros(index.shape, dtype=x.dtype)
        return self._jnp.take_along_axis(x, index, axis=dim)

    def take(self, x: Any, indices: Any) -> Any:
        # Avoid JAX gather error for zero-sized dimensions in JIT
        if x.shape[0] == 0:
            indices_shape = getattr(indices, "shape", ())
            return self._jnp.zeros(indices_shape + x.shape[1:], dtype=x.dtype)
        return x[indices]

    def select(self, x: Any, dim: int, index: int) -> Any:
        # JAX take/select will fail if dimension is 0
        if x.shape[dim] == 0:
            new_shape = x.shape[:dim] + x.shape[dim + 1 :]
            return self._jnp.zeros(new_shape, dtype=x.dtype)
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

    def unique(
        self,
        x: Any,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
    ) -> Any:
        # jnp.unique returns (uniques, [indices], [inverse], [counts])
        return self._jnp.unique(
            x,
            return_index=return_index,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )

    def sum(self, x: Any, dim: Any = None) -> Any:
        return self._jnp.sum(x, axis=dim)

    def mean(self, x: Any) -> Any:
        return self._jnp.mean(x)

    def max(self, x: Any) -> Any:
        return self._jnp.max(x)

    def min(self, x: Any) -> Any:
        return self._jnp.min(x)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return self._jnp.where(condition, x, y)

    def repeat_interleave(
        self,
        x: Any,
        repeats: Any,
        dim: Optional[int] = None,
        total_repeat_length: Optional[int] = None,
    ) -> Any:
        # JAX jnp.repeat repeats elements.
        return self._jnp.repeat(
            x, repeats, axis=dim, total_repeat_length=total_repeat_length
        )

    def cumsum(self, x: Any, dim: int = 0) -> Any:
        return self._jnp.cumsum(x, axis=dim)

    def nonzero(self, x: Any, *, size: Any = None, fill_value: Any = None) -> Any:
        return self._jnp.nonzero(x, size=size, fill_value=fill_value)

    def dynamic_slice(self, x: Any, start: Any, size: Any) -> Any:
        if isinstance(x, JAXViewWrapper):
            x = x.arr[x.slices]
        # Ensure start/size are tuples for JAX compatibility
        ndim = x.ndim
        if isinstance(start, (int, float)) or self.is_scalar(start):
            start = (start,) + (0,) * (ndim - 1)

        if (isinstance(size, (int, float)) or self.is_scalar(size)) and ndim == 1:
            size = (size,)

        return self._jax.lax.dynamic_slice(x, start, size)

    def dynamic_slice_in_dim(self, x: Any, start: Any, size: int, axis: int = 0) -> Any:
        ndim = x.ndim
        start_indices = [0] * ndim
        start_indices[axis] = start

        slice_sizes = list(x.shape)
        slice_sizes[axis] = size

        return self._jax.lax.dynamic_slice(x, tuple(start_indices), tuple(slice_sizes))

    def dynamic_slice_update(self, x: Any, update: Any, start: Any) -> Any:
        # Ensure start is a tuple matching input dimensions
        if isinstance(start, int):
            start = (start,) + (0,) * (x.ndim - 1)
        elif self.is_scalar(start):
            start = (start,) + (0,) * (x.ndim - 1)
        elif isinstance(start, (list, tuple)) and len(start) < x.ndim:
            start = tuple(start) + (0,) * (x.ndim - len(start))

        return self._jax.lax.dynamic_update_slice(x, update, start)

    def bincount(
        self, x: Any, weights: Optional[Any] = None, minlength: int = 0
    ) -> Any:
        return self._jnp.bincount(x, weights=weights, length=minlength)

    def vmap(self, fun: Any, in_axes: Any = 0, out_axes: Any = 0) -> Any:
        return self._jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)

    def tree_map(self, fun: Any, tree: Any) -> Any:
        return self._jax.tree.map(fun, tree)

    def tree_flatten(self, tree: Any) -> Tuple[list, Any]:
        return self._jax.tree.flatten(tree)

    def tile(self, x: Any, dims: Tuple[int, ...]) -> Any:
        return self._jnp.tile(x, dims)

    def scan(self, f: Any, init: Any, xs: Any, length: Optional[int] = None) -> Any:
        return self._jax.lax.scan(f, init, xs, length=length)

    def get_dlpack_slice(self, tensor: Any, cursor: Optional[Any] = None) -> Any:
        # returns a zero-copy wrapper for JAX
        return JAXViewWrapper(tensor, cursor) if cursor is not None else tensor

    def index_update(self, x: Any, indices: Any, values: Any) -> Any:
        return x.at[indices].set(values)

    @property
    def supports_zero_copy_view(self) -> bool:
        return True

    def get_contiguous_view(self, tensor: Any, offset: int, count: int) -> Any:
        return JAXViewWrapper(tensor, slice(offset, offset + count))


class JAXViewWrapper:
    def __init__(self, arr, slices):
        self.arr = arr
        self.slices = slices  # int, slice, or tuple
        desc = {
            "shape": self.arr.shape,
            "strides": None,
            "data": (0, False),
            "typestr": self.arr.dtype.str,
        }
        self._apply_slicing(desc)
        self._shape = desc["shape"]
        self._dtype = self.arr.dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key):
        return self.arr[self.slices][key]

    def __jax_array__(self):
        return self.arr[self.slices]

    @property
    def __cuda_array_interface__(self):
        if not hasattr(self.arr, "__cuda_array_interface__"):
            raise AttributeError
        desc = self.arr.__cuda_array_interface__.copy()
        self._apply_slicing(desc)
        return desc

    @property
    def __array_interface__(self):
        if hasattr(self.arr, "__array_interface__"):
            desc = self.arr.__array_interface__.copy()
        elif hasattr(self.arr, "unsafe_buffer_pointer"):
            desc = {
                "shape": self.arr.shape,
                "typestr": self.arr.dtype.str,
                "data": (self.arr.unsafe_buffer_pointer(), False),
                "version": 3,
            }
        else:
            raise AttributeError
        self._apply_slicing(desc)
        return desc

    @property
    def _sliced_arr(self):
        if not hasattr(self, "_cached_sliced_arr"):
            self._cached_sliced_arr = self.arr[self.slices]
        return self._cached_sliced_arr

    def __array__(self, *args, **kwargs):
        import numpy as np

        return np.asarray(self._sliced_arr, *args, **kwargs)

    def __dlpack__(self, stream=None):
        return self._sliced_arr.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self._sliced_arr.__dlpack_device__()

    def _apply_slicing(self, desc):
        shape = desc["shape"]
        strides = desc.get("strides")

        if strides is None:
            dtype_str = desc["typestr"]
            itemsize = int(dtype_str[2:])

            strides = []
            st = itemsize
            for d in reversed(shape):
                strides.append(st)
                st *= d
            strides = tuple(reversed(strides))

        if isinstance(self.slices, int):
            key = (slice(0, self.slices, 1),)
        elif isinstance(self.slices, slice):
            key = (self.slices,)
        elif isinstance(self.slices, (tuple, list)):
            key = tuple(self.slices)
        else:
            raise ValueError(f"Unsupported slice type: {type(self.slices)}")

        new_shape = []
        new_strides = []
        ptr_offset = 0
        dim_idx = 0

        for k in key:
            if isinstance(k, int):
                d_len = shape[dim_idx]
                d_stride = strides[dim_idx]

                idx = k + d_len if k < 0 else k
                if idx < 0 or idx >= d_len:
                    raise IndexError(
                        f"Index {k} out of bounds for dim {dim_idx} size {d_len}"
                    )

                ptr_offset += idx * d_stride
                dim_idx += 1

            elif isinstance(k, slice):
                d_len = shape[dim_idx]
                d_stride = strides[dim_idx]

                start, stop, step = k.indices(d_len)

                if step > 0:
                    new_len = (stop - start + step - 1) // step
                    new_len = max(0, new_len)
                else:
                    new_len = len(range(start, stop, step))

                new_shape.append(new_len)
                new_strides.append(d_stride * step)

                ptr_offset += start * d_stride
                dim_idx += 1
            else:
                raise NotImplementedError(
                    f"Slice type {type(k)} not supported in JAX zero-copy view"
                )

        # Append remaining dims
        while dim_idx < len(shape):
            new_shape.append(shape[dim_idx])
            new_strides.append(strides[dim_idx])
            dim_idx += 1

        # Update desc
        desc["shape"] = tuple(new_shape)
        desc["strides"] = tuple(new_strides)

        ptr, readonly = desc["data"]
        desc["data"] = (ptr + ptr_offset, False)  # Force writeable as per previous hack


try:
    import jax

    jax.tree_util.register_pytree_node(
        JAXViewWrapper,
        lambda w: ((w.arr,), w.slices),
        lambda slices, arr_tuple: JAXViewWrapper(arr_tuple[0], slices),
    )
except ImportError:
    pass

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
