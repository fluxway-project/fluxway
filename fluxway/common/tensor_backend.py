import contextlib
import ctypes
import operator
import threading
from abc import ABC, abstractmethod
from functools import wraps
from math import prod
from typing import Any

import ml_dtypes
import numpy as np

from fluxway.common.allocation import AllocationDescriptor, AllocationIdentity
from fluxway.common.coherence import (
    CoherenceStateTransitionError,
    PhysicalMutability,
    UnsupportedAllocationHandoffError,
)
from fluxway.common.numba_runtime import call_numba_cached


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("type_code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))

_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
]


_BFLOAT16_DTYPE = np.dtype(ml_dtypes.bfloat16)
_ACTIVE_DL_PACK_CONTEXTS: list[Any] = []


class DLPackCompatibleNDArray(np.ndarray):
    """NumPy ndarray subclass providing standard __dlpack__ for bfloat16 arrays."""

    def __dlpack__(self, stream=None):
        if self.dtype == _BFLOAT16_DTYPE:
            ndim = self.ndim
            shape_arr = (ctypes.c_int64 * ndim)(*self.shape)
            itemsize = self.itemsize
            if self.strides:
                strides_arr = (ctypes.c_int64 * ndim)(
                    *[s // itemsize for s in self.strides]
                )
            else:
                strides_arr = None

            managed = _DLManagedTensor()
            managed.dl_tensor.data = ctypes.c_void_p(
                int(self.__array_interface__["data"][0])
            )
            managed.dl_tensor.device = _DLDevice(1, 0)
            managed.dl_tensor.ndim = ndim
            managed.dl_tensor.dtype = _DLDataType(4, 16, 1)
            managed.dl_tensor.shape = shape_arr
            managed.dl_tensor.strides = strides_arr
            managed.dl_tensor.byte_offset = 0
            managed.manager_ctx = None
            managed.deleter = _DLManagedTensorDeleter(0)

            _ACTIVE_DL_PACK_CONTEXTS.append((managed, shape_arr, strides_arr, self))
            if len(_ACTIVE_DL_PACK_CONTEXTS) > 1024:
                del _ACTIVE_DL_PACK_CONTEXTS[:512]

            return ctypes.pythonapi.PyCapsule_New(
                ctypes.pointer(managed),
                b"dltensor",
                None,
            )
        return super().__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return (1, 0)


def _adopt_dlpack_cpu(capsule_or_obj: Any) -> np.ndarray:
    """Zero-copy adopt a CPU array or DLPack capsule into a NumPy ndarray."""
    if hasattr(capsule_or_obj, "__dlpack__") or not hasattr(
        capsule_or_obj, "__array__"
    ):
        capsule = (
            capsule_or_obj.__dlpack__()
            if hasattr(capsule_or_obj, "__dlpack__")
            else capsule_or_obj
        )
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"dltensor")
        if not ptr:
            raise ValueError("Invalid dltensor capsule pointer.")

        managed = _DLManagedTensor.from_address(ptr)
        dlt = managed.dl_tensor
        if dlt.device.device_type != 1:
            raise ValueError(
                f"_adopt_dlpack_cpu requires CPU tensor (device_type=1), got {dlt.device.device_type}."
            )

        if dlt.dtype.type_code == 4 and dlt.dtype.bits == 16:
            shape = tuple(dlt.shape[i] for i in range(dlt.ndim))
            if dlt.strides:
                itemsize = (dlt.dtype.bits + 7) // 8
                strides = tuple(dlt.strides[i] * itemsize for i in range(dlt.ndim))
            else:
                strides = None
            data_ptr = dlt.data + dlt.byte_offset

            class _DLPackHolder:
                def __init__(self):
                    iface = {
                        "data": (data_ptr, False),
                        "shape": shape,
                        "typestr": "|V2",
                        "version": 3,
                    }
                    if strides is not None:
                        iface["strides"] = strides
                    self.__array_interface__ = iface
                    self._keepalive = capsule

            raw_arr = np.asarray(_DLPackHolder())
            return raw_arr.view(_BFLOAT16_DTYPE).view(DLPackCompatibleNDArray)

        obj = DLPackObject(
            capsule, device=(dlt.device.device_type, dlt.device.device_id)
        )
        return np.from_dlpack(obj)

    arr = capsule_or_obj.__array__()
    if arr.dtype == _BFLOAT16_DTYPE:
        return arr.view(DLPackCompatibleNDArray)
    return arr


def force_writable(arr: Any) -> Any:
    """Make a readonly array writable **without copying**.

    DLPack / Ray shared-memory imports produce arrays whose data buffer is
    genuinely mutable (the C++ allocator always uses writable memory), but
    NumPy / CuPy conservatively marks them readonly because they cannot
    verify ownership.

    This function patches ``__array_interface__`` (CPU) or
    ``__cuda_array_interface__`` (GPU) to clear the readonly flag and
    reconstructs a new view that shares the **same underlying memory**:

    * zero-copy  - no data duplication
    * writable   - downstream code can modify in-place
    * safe       - the source array is kept alive via ``_source`` to
      prevent premature DLPack deallocation

    Only call this when you *know* the backing memory is writable (which is
    always the case for DLPack tensors produced by sparse_embedding or
    Fluxway transport).

    CUDA inputs require CuPy; backend and conversion errors propagate.
    """
    # Fast path: already writable or not an ndarray type we handle
    if not hasattr(arr, "flags") or not hasattr(arr.flags, "writeable"):
        return arr

    if isinstance(arr, np.ndarray):
        if arr.flags.writeable:
            if arr.dtype == _BFLOAT16_DTYPE and not isinstance(
                arr, DLPackCompatibleNDArray
            ):
                return arr.view(DLPackCompatibleNDArray)
            return arr
        try:
            arr.flags.writeable = True
            if arr.dtype == _BFLOAT16_DTYPE and not isinstance(
                arr, DLPackCompatibleNDArray
            ):
                return arr.view(DLPackCompatibleNDArray)
            return arr
        except ValueError:
            pass

        # CPU path: patch __array_interface__
        class _WritableCPUView:
            __slots__ = ("__array_interface__", "_source")

            def __init__(self, source: np.ndarray):
                iface = source.__array_interface__
                ptr, _ = iface["data"]
                new_iface = dict(iface)
                new_iface["data"] = (ptr, False)
                self.__array_interface__ = new_iface
                self._source = source  # prevent GC → prevents DLPack free

        res = np.asarray(_WritableCPUView(arr))
        if res.dtype == _BFLOAT16_DTYPE and not isinstance(
            res, DLPackCompatibleNDArray
        ):
            return res.view(DLPackCompatibleNDArray)
        return res

    if not hasattr(arr, "__cuda_array_interface__"):
        return arr

    import cupy as cp

    if isinstance(arr, cp.ndarray):
        if arr.flags.writeable:
            return arr
        try:
            arr.flags.writeable = True
            return arr
        except ValueError:
            pass

    class _WritableGPUView:
        __slots__ = ("__cuda_array_interface__", "_source")

        def __init__(self, source: Any):
            iface = source.__cuda_array_interface__
            ptr, _ = iface["data"]
            new_iface = dict(iface)
            new_iface["data"] = (ptr, False)
            self.__cuda_array_interface__ = new_iface
            self._source = source

    return cp.asarray(_WritableGPUView(arr))


def prepare_dlpack_input(value: Any) -> Any:
    writable = force_writable(value)
    if (
        isinstance(writable, np.ndarray)
        and writable.dtype == _BFLOAT16_DTYPE
        and not isinstance(writable, DLPackCompatibleNDArray)
    ):
        return writable.view(DLPackCompatibleNDArray)
    return writable


def ensure_cpu_serializable(arr: Any) -> Any:
    """Move any device-resident tensor to host memory for safe pickle/Ray
    serialization across processes that may not have a DL backend runtime.

    Fully duck-typed via Python array protocols -- no backend imports,
    no registry, no isinstance chains:
      - GPU-resident (``__cuda_array_interface__``: cupy / jax-cuda /
        torch-cuda) -> ``cupy.asnumpy``.
      - CPU-resident DL tensors (``__array__``: jax / torch CPU lazy
        arrays) -> ``np.asarray`` (triggers device_get / detach via the
        protocol).
      - DLPack-only foreign tensors -> ``np.from_dlpack``.
      - numpy views -> owned copy.
      - everything else (Python scalars, dicts, None, ...) -> identity.

    **Call-site guidance**: use this only on Ray (pickle-based) transport
    boundaries.  Other transports (NCCL, UCX) handle CUDA buffers natively
    and should NOT call this.
    """
    if isinstance(arr, np.ndarray):
        return arr.copy() if not arr.flags.owndata else arr

    if hasattr(arr, "__cuda_array_interface__"):
        try:
            import cupy as cp

            return cp.asnumpy(cp.asarray(arr))
        except ImportError:
            pass

    if hasattr(arr, "__array__"):
        return np.asarray(arr)

    if hasattr(arr, "__dlpack__"):
        return _adopt_dlpack_cpu(arr)

    return arr


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


def _unwrap_ragged(x: Any) -> Any:
    """Unwrap one level of ragged/sparse tensor to its underlying dense data.

    Known first-party types with ``flat_values``:
      - ``TransportTensor``   (fluxway.common.types.tensor)
      - ``TransportShardView`` (fluxway.spmd.worker.mailbox.mailbox)

    ``TransportShardView`` cannot be imported here at module level because
    ``mailbox.py`` itself imports ``tensor_backend``, creating a circular
    dependency.  It is therefore handled by the duck-typed ``getattr`` fallback
    below.  External sparse types (TF RaggedTensor / SparseTensor) with a
    non-callable ``.values`` attribute are also caught by that fallback.
    """
    from fluxway.common.types.tensor import TransportTensor

    if isinstance(x, TransportTensor):
        return x.flat_values
    # TransportShardView and external sparse types (e.g. TF RaggedTensor).
    flat = getattr(x, "flat_values", None)
    if flat is not None and not callable(flat):
        return flat
    values = getattr(x, "values", None)
    if (
        values is not None
        and not callable(values)
        and not isinstance(x, (dict, list, tuple))
    ):
        return values
    return x


def as_native_zerocopy(x: Any) -> Any:
    """Zero-copy normalize any tensor (jax/torch/np/cupy/...) to native numpy or cupy.

    HybridBackend supports *both* numpy (CPU) and cupy (GPU) and chooses
    between them per-array via :py:meth:`HybridBackend._lib_for`.  When
    upstream layers (routing/communication) feed a foreign array (JAX,
    PyTorch, ...) into HybridBackend, every method silently re-dispatches
    through ``__array_function__`` into ``jnp`` or hits ``BufferError``
    in numba kernels.  This helper is the *single chokepoint* that
    converts foreign arrays into the native domain.

    Routing rule -- the target domain follows the **source device**, so
    the converted array stays on the same device and the rest of the
    pipeline retentions benefitting from ``HybridBackend._lib_for``:

    +----------------------+--------------------+--------------------+
    | Source               | CPU device         | CUDA/ROCM device   |
    +======================+====================+====================+
    | numpy.ndarray        | as-is              | n/a                |
    | cupy.ndarray         | n/a                | as-is              |
    | jax.Array            | np.from_dlpack     | cupy.from_dlpack   |
    | torch.Tensor         | np.from_dlpack     | cupy.from_dlpack   |
    | other __dlpack__     | np.from_dlpack     | cupy.from_dlpack   |
    | unsupported          | TypeError           | n/a                |
    +----------------------+--------------------+--------------------+

    Every supported source takes the zero-copy path. Unsupported protocol
    combinations fail at this boundary instead of being materialized on CPU.
    """
    # numpy fast-path
    if isinstance(x, np.ndarray):
        if x.dtype == _BFLOAT16_DTYPE and not isinstance(x, DLPackCompatibleNDArray):
            return x.view(DLPackCompatibleNDArray)
        return x

    cp_mod = None
    if hasattr(x, "__dlpack__") or hasattr(x, "__cuda_array_interface__"):
        try:
            import cupy as _cp  # type: ignore
        except ImportError:
            _cp = None
        cp_mod = _cp
        if cp_mod is not None and isinstance(x, cp_mod.ndarray):
            return x

    if hasattr(x, "__dlpack__"):
        if not hasattr(x, "__dlpack_device__"):
            raise TypeError(
                f"DLPack source {type(x).__name__} must expose __dlpack_device__."
            )
        source_device, _ = x.__dlpack_device__()
        if source_device == 1:
            return _adopt_dlpack_cpu(x)
        if source_device in (2, 10):
            if cp_mod is None:
                raise RuntimeError(
                    "GPU DLPack adoption requires an available CuPy backend."
                )
            return cp_mod.from_dlpack(x)
        raise ValueError(f"unsupported DLPack source device {source_device!r}")

    if hasattr(x, "__cuda_array_interface__"):
        if cp_mod is None:
            raise RuntimeError(
                "CUDA array-interface adoption requires an available CuPy backend."
            )
        return cp_mod.asarray(x)

    raise TypeError(
        f"as_native_zerocopy requires a NumPy, CuPy, DLPack, or CUDA array-interface "
        f"source, got {type(x).__name__}."
    )


class TensorBackend(ABC):
    @property
    def default_device(self) -> str:
        return "cpu"

    """Abstract interface for tensor operations across different frameworks.

    This class defines a unified API for common tensor operations like creation,
    manipulation, and mathematical functions, enabling backend-agnostic code.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def empty_cache(self) -> None:
        """Release unused memory held by this backend's caching allocator."""
        pass

    @abstractmethod
    def from_dlpack(self, dlpack_capsule: Any) -> Any:
        """Create a tensor from a DLPack capsule."""
        pass

    @abstractmethod
    def to_dlpack(self, x: Any) -> Any:
        """Create a DLPack capsule from a tensor."""
        pass

    def allocation_descriptor(self, value: Any) -> AllocationDescriptor:
        raise UnsupportedAllocationHandoffError(
            f"backend {self.name()!r} cannot prove allocation identity for "
            f"{type(value).__name__}."
        )

    def readonly_view(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            view = value.view()
            view.flags.writeable = False
            return view
        raise CoherenceStateTransitionError(
            f"backend {self.name()!r} cannot enforce readonly access for "
            f"{type(value).__name__}."
        )

    # ------------------------------------------------------------------
    # Capability flags (used to avoid string-based backend dispatch)
    # ------------------------------------------------------------------
    @property
    def is_jax(self) -> bool:
        """True if this backend uses JAX tracing semantics (JIT, static shapes)."""
        return False

    @property
    def is_torch(self) -> bool:
        """True if this backend uses PyTorch (eager / torch.compile)."""
        return False

    def requires_static_shapes(self) -> bool:
        """Whether downstream pipelines must produce static-shape buffers.

        Backends that wrap a tracing/compilation engine (jax.jit,
        torch.compile) need padded capacity buffers so traces are
        reused across varying batch sizes.  Pure eager backends (numpy,
        cupy, torch eager) can run on dynamic shapes directly and skip
        padding entirely on both send and receive sides.

        Default is False; backends override to opt-in.
        """
        return False

    @abstractmethod
    def asarray(
        self, x: Any, *, copy: bool | None = None, device: str | None = None
    ) -> Any:
        """Convert a foreign tensor with an explicit copy policy."""
        ...

    def asarray_tree(
        self, tree: Any, *, copy: bool | None = None, device: str | None = None
    ) -> Any:
        """Recursively convert all leaf tensors with one copy policy."""
        if isinstance(tree, dict):
            return {
                k: self.asarray_tree(v, copy=copy, device=device)
                for k, v in tree.items()
            }
        if isinstance(tree, list):
            return [self.asarray_tree(v, copy=copy, device=device) for v in tree]
        if isinstance(tree, tuple) and hasattr(tree, "_fields"):
            return type(tree)(
                *(self.asarray_tree(v, copy=copy, device=device) for v in tree)
            )
        if isinstance(tree, tuple):
            return tuple(self.asarray_tree(v, copy=copy, device=device) for v in tree)
        return self.asarray(tree, copy=copy, device=device)

    def asarray_with_lease(
        self, value: Any, *, copy: bool | None = None, device: str | None = None
    ) -> tuple[Any, Any | None]:
        from fluxway.common.types.tensor import TransportTensor

        if not isinstance(value, TransportTensor):
            return self.asarray(value, copy=copy, device=device), None
        lease = value.acquire_consumer_lease()
        try:
            return self.asarray(value, copy=copy, device=device), lease
        except BaseException:
            lease.cancel()
            raise

    @abstractmethod
    def tensor(self, data: Any, dtype: Any = None, device: str | None = None) -> Any:
        pass

    @abstractmethod
    def as_tensor(self, data: Any, dtype: Any = None, device: str | None = None) -> Any:
        pass

    @abstractmethod
    def to_numpy(self, tensor: Any) -> Any:
        pass

    @abstractmethod
    def zeros(self, shape: Any, dtype: Any = None, device: str | None = None) -> Any:
        pass

    @abstractmethod
    def ones(self, shape: Any, dtype: Any = None, device: str | None = None) -> Any:
        pass

    @abstractmethod
    def full(
        self, shape: Any, fill_value: Any, dtype: Any = None, device: str | None = None
    ) -> Any:
        pass

    @abstractmethod
    def arange(
        self,
        start: int,
        end: int,
        step: int = 1,
        dtype: Any = None,
        device: str | None = None,
    ) -> Any:
        pass

    @abstractmethod
    def cat(self, tensors: list[Any], dim: int = 0) -> Any:
        pass

    @abstractmethod
    def stack(self, tensors: list[Any], dim: int = 0) -> Any:
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
    def reshape(self, tensor: Any, shape: tuple[int, ...]) -> Any:
        pass

    @abstractmethod
    def unravel_index(self, indices: Any, shape: tuple[int, ...]) -> Any:
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
    ) -> tuple[Any, Any]:
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
        dim: int | None = None,
        total_repeat_length: int | None = None,
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
    def bincount(self, x: Any, weights: Any | None = None, minlength: int = 0) -> Any:
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
    def tree_flatten(self, tree: Any) -> tuple[list[Any], Any]:
        """Flatten a pytree into (leaves_list, treedef)."""
        pass

    @abstractmethod
    def tile(self, x: Any, dims: tuple[int, ...]) -> Any:
        """Constructs a new tensor by repeating input."""
        pass

    @abstractmethod
    def scan(self, f: Any, init: Any, xs: Any, length: int | None = None) -> Any:
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
    def get_dlpack_slice(self, tensor: Any, cursor: Any | None = None) -> Any:
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
    def __init__(self, compile: bool = False):
        import torch  # type: ignore

        self._torch = torch
        # When ``compile=True`` the user has opted into ``torch.compile``
        # (or any tracing/JIT path).  Communication will then bucket
        # send/receive shapes to capacity so the compiled graph is
        # reused across batches.  Default is False (pure eager).
        self._compile_enabled = compile

    def name(self) -> str:
        return "torch"

    def empty_cache(self) -> None:
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def allocation_descriptor(self, value: Any) -> AllocationDescriptor:
        if not isinstance(value, self._torch.Tensor):
            return super().allocation_descriptor(value)
        if not value.is_contiguous():
            raise UnsupportedAllocationHandoffError(
                "torch allocation handoff requires a contiguous tensor."
            )
        storage = value.untyped_storage()
        root_address = int(storage.data_ptr())
        root_nbytes = int(storage.nbytes())
        view_address = int(value.data_ptr())
        view_nbytes = int(value.nelement() * value.element_size())
        return AllocationDescriptor(
            AllocationIdentity(
                "torch",
                str(value.device),
                root_address,
                root_nbytes,
                view_address,
                view_nbytes,
            ),
            storage,
        )

    @property
    def is_torch(self) -> bool:
        return True

    def requires_static_shapes(self) -> bool:
        return self._compile_enabled

    def asarray(
        self, x: Any, *, copy: bool | None = None, device: str | None = None
    ) -> Any:
        x = _unwrap_ragged(x)

        if isinstance(x, self._torch.Tensor):
            return x.clone() if copy else x
        if hasattr(x, "__dlpack__"):
            result = self._torch.from_dlpack(x)
            return result.clone() if copy else result
        if isinstance(x, np.ndarray):
            return self._torch.tensor(x) if copy else self._torch.from_numpy(x)
        if copy is False:
            raise TypeError(
                f"torch zero-copy conversion does not support {type(x).__name__}."
            )
        return self._torch.tensor(x, device=device)

    def tensor(self, data: Any, dtype: Any = None, device: str | None = None) -> Any:
        return self._torch.tensor(data, dtype=dtype)

    def as_tensor(self, data: Any, dtype: Any = None, device: str | None = None) -> Any:
        return self._torch.as_tensor(data, dtype=dtype)

    def to_numpy(self, tensor: Any) -> Any:
        if isinstance(tensor, list):
            return np.array(tensor)
        return tensor.detach().cpu().numpy()

    def zeros(self, shape: Any, dtype: Any = None, device: str | None = None) -> Any:
        return self._torch.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None, device: str | None = None) -> Any:
        return self._torch.ones(shape, dtype=dtype)

    def full(
        self, shape: Any, fill_value: Any, dtype: Any = None, device: str | None = None
    ) -> Any:
        return self._torch.full(shape, fill_value, dtype=dtype)

    def arange(
        self,
        start: int,
        end: int,
        step: int = 1,
        dtype: Any = None,
        device: str | None = None,
    ) -> Any:
        return self._torch.arange(start, end, step, dtype=dtype)

    def cat(self, tensors: list[Any], dim: int = 0) -> Any:
        return self._torch.cat(tensors, dim=dim)

    def stack(self, tensors: list[Any], dim: int = 0) -> Any:
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

    def reshape(self, tensor: Any, shape: tuple[int, ...]) -> Any:
        return self._torch.reshape(tensor, shape)

    def unravel_index(self, indices: Any, shape: tuple[int, ...]) -> Any:
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
    ) -> tuple[Any, Any]:
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
        dim: int | None = None,
        total_repeat_length: int | None = None,
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
        elif isinstance(start, list | tuple):
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

    def bincount(self, x: Any, weights: Any | None = None, minlength: int = 0) -> Any:
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
            import torch.utils._pytree as pytree  # type: ignore

            return pytree.tree_map(fun, tree)
        except ImportError:
            # Fallback for simple structures
            if isinstance(tree, list | tuple):
                return type(tree)(fun(x) for x in tree)
            return fun(tree)

    def tree_flatten(self, tree: Any) -> tuple[list[Any], Any]:
        try:
            import torch.utils._pytree as pytree  # type: ignore

            return pytree.tree_flatten(tree)
        except ImportError:
            if isinstance(tree, list | tuple):
                return list(tree), type(tree)
            elif isinstance(tree, dict):
                return list(tree.values()), tree.keys()
            return [tree], None

    def tile(self, x: Any, dims: tuple[int, ...]) -> Any:
        return self._torch.tile(x, dims)

    def scan(self, f: Any, init: Any, xs: Any, length: int | None = None) -> Any:
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

    def get_dlpack_slice(self, tensor: Any, cursor: Any | None = None) -> Any:
        from torch.utils.dlpack import to_dlpack  # type: ignore

        sliced_tensor = tensor
        if cursor is not None:
            if isinstance(cursor, int | slice | tuple | list):
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

    def from_dlpack(self, dlpack_capsule: Any) -> Any:
        from torch.utils.dlpack import from_dlpack  # type: ignore

        capsule_obj = (
            dlpack_capsule
            if hasattr(dlpack_capsule, "__dlpack__")
            else DLPackObject(dlpack_capsule)
        )
        return from_dlpack(capsule_obj)

    def to_dlpack(self, x: Any) -> Any:
        from torch.utils.dlpack import to_dlpack  # type: ignore

        return to_dlpack(x)

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
        # pyrefly: ignore [missing-import]
        import jax
        import jax.numpy as jnp
        from jax.extend import backend

        self._jnp = jnp
        self._jax = jax
        self._jax_backend = backend
        self._enable_x64 = jax.enable_x64

    def _x64_scope(self):
        return self._enable_x64()

    def _resolve_device(self, device: str | None) -> Any:
        if device is None:
            return None
        devices = self._jax.devices(device)
        if len(devices) != 1:
            raise ValueError(
                f"JAX device selector must resolve to exactly one device: {device!r}."
            )
        return devices[0]

    def name(self) -> str:
        return "jax"

    def empty_cache(self) -> None:
        self._jax.clear_caches()
        self._jax_backend.clear_backends()

    def allocation_descriptor(self, value: Any) -> AllocationDescriptor:
        if not isinstance(value, self._jax.Array):
            return super().allocation_descriptor(value)
        pointer = value.unsafe_buffer_pointer()
        nbytes = value.nbytes
        return AllocationDescriptor(
            AllocationIdentity(
                "jax",
                str(value.device),
                pointer,
                nbytes,
                pointer,
                nbytes,
            ),
            value,
        )

    def readonly_view(self, value: Any) -> Any:
        if not isinstance(value, self._jax.Array):
            return super().readonly_view(value)
        return value

    @property
    def is_jax(self) -> bool:
        return True

    def requires_static_shapes(self) -> bool:
        # JaxBackend always traces through ``jax.jit``; capacity
        # bucketing is required to retention recompilation rate bounded.
        return True

    def asarray(
        self, x: Any, *, copy: bool | None = None, device: str | None = None
    ) -> Any:
        from fluxway.common.types.tensor import TransportTensor

        resolved_device = self._resolve_device(device)

        coherent_readonly = isinstance(x, TransportTensor) and x.is_readonly
        if coherent_readonly and isinstance(x._payload.values, self._jax.Array):
            if copy is True:
                return self._jnp.array(x._payload.values, copy=True)
            return x._payload.values
        if coherent_readonly:
            if x.physical_mutability is PhysicalMutability.PROVEN_WRITABLE:
                x = x.dlpack_values_for_consumer()
            elif copy is False:
                x.dlpack_values_for_consumer()
            else:
                x = x.flat_values
                coherent_readonly = False
        else:
            x = _unwrap_ragged(x)

        if isinstance(x, self._jax.Array):
            return self._jnp.array(x, copy=True) if copy else x

        if (
            hasattr(x, "flags")
            and hasattr(x.flags, "c_contiguous")
            and not x.flags.c_contiguous
        ):
            if copy is False:
                raise ValueError(
                    "jax zero-copy conversion requires a contiguous source."
                )
            import cupy as cp

            x = (
                cp.ascontiguousarray(x)
                if hasattr(x, "__cuda_array_interface__")
                else np.ascontiguousarray(x)
            )

        with self._x64_scope():
            if copy is True and isinstance(x, np.ndarray):
                return self._jnp.array(np.array(x, copy=True), device=resolved_device)
            if (
                hasattr(x, "flags")
                and hasattr(x.flags, "writeable")
                and not x.flags.writeable
                and not coherent_readonly
            ):
                if copy is False:
                    raise BufferError(
                        "Cannot export readonly array with copy=False without a "
                        "proven writable allocation."
                    )
                return self._jnp.asarray(x, device=resolved_device)
            if hasattr(x, "__dlpack__"):
                return self._jax.dlpack.from_dlpack(
                    x, device=resolved_device, copy=copy
                )
            if hasattr(x, "__cuda_array_interface__"):
                import cupy as cp

                cp_view = cp.asarray(x)
                if not cp_view.flags.c_contiguous:
                    if copy is False:
                        raise ValueError(
                            "jax zero-copy conversion requires a contiguous CUDA source."
                        )
                    cp_view = cp.ascontiguousarray(cp_view)
                return self._jax.dlpack.from_dlpack(
                    cp_view, device=resolved_device, copy=copy
                )
            if copy is False:
                raise TypeError(
                    f"jax zero-copy conversion does not support {type(x).__name__}."
                )
            return self._jnp.asarray(x, device=resolved_device)

    def tensor(self, data: Any, dtype: Any = None, device: str | None = None) -> Any:
        with self._x64_scope():
            return self._jnp.array(data, dtype=dtype)

    def as_tensor(self, data: Any, dtype: Any = None, device: str | None = None) -> Any:
        with self._x64_scope():
            return self._jnp.asarray(data, dtype=dtype)

    def to_numpy(self, tensor: Any) -> Any:
        return np.array(tensor)

    def zeros(self, shape: Any, dtype: Any = None, device: str | None = None) -> Any:
        with self._x64_scope():
            return self._jnp.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None, device: str | None = None) -> Any:
        with self._x64_scope():
            return self._jnp.ones(shape, dtype=dtype)

    def full(
        self, shape: Any, fill_value: Any, dtype: Any = None, device: str | None = None
    ) -> Any:
        with self._x64_scope():
            return self._jnp.full(shape, fill_value, dtype=dtype)

    def arange(
        self,
        start: int,
        end: int,
        step: int = 1,
        dtype: Any = None,
        device: str | None = None,
    ) -> Any:
        with self._x64_scope():
            return self._jnp.arange(start, end, step, dtype=dtype)

    def cat(self, tensors: list[Any], dim: int = 0) -> Any:
        return self._jnp.concatenate(tensors, axis=dim)

    def stack(self, tensors: list[Any], dim: int = 0) -> Any:
        return self._jnp.stack(tensors, axis=dim)

    def equal(self, x: Any, y: Any) -> bool:
        return self._jnp.array_equal(x, y)  # type: ignore

    def float32(self) -> Any:
        return self._jnp.float32

    def int32(self) -> Any:
        return self._jnp.int32

    def is_tensor(self, obj: Any) -> bool:
        return isinstance(obj, self._jnp.ndarray | np.ndarray | JAXViewWrapper)

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
        with self._x64_scope():
            return tensor.astype(dtype)

    def item(self, tensor: Any) -> Any:
        return tensor.item()

    def unravel_index(self, indices: Any, shape: tuple[int, ...]) -> Any:
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
    ) -> tuple[Any, Any]:
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
        dim: int | None = None,
        total_repeat_length: int | None = None,
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
        if isinstance(start, int | float) or self.is_scalar(start):
            start = (start,) + (0,) * (ndim - 1)

        if (isinstance(size, int | float) or self.is_scalar(size)) and ndim == 1:
            size = (size,)

        return self._jax.lax.dynamic_slice(x, start, size)  # type: ignore

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
        elif isinstance(start, list | tuple) and len(start) < x.ndim:
            start = tuple(start) + (0,) * (x.ndim - len(start))

        return self._jax.lax.dynamic_update_slice(x, update, start)

    def bincount(self, x: Any, weights: Any | None = None, minlength: int = 0) -> Any:
        return self._jnp.bincount(x, weights=weights, length=minlength)

    def vmap(self, fun: Any, in_axes: Any = 0, out_axes: Any = 0) -> Any:
        return self._jax.vmap(fun, in_axes=in_axes, out_axes=out_axes)

    def tree_map(self, fun: Any, tree: Any) -> Any:
        return self._jax.tree.map(fun, tree)

    def tree_flatten(self, tree: Any) -> tuple[list[Any], Any]:
        return self._jax.tree.flatten(tree)

    def tile(self, x: Any, dims: tuple[int, ...]) -> Any:
        return self._jnp.tile(x, dims)

    def scan(self, f: Any, init: Any, xs: Any, length: int | None = None) -> Any:
        return self._jax.lax.scan(f, init, xs, length=length)

    def get_dlpack_slice(self, tensor: Any, cursor: Any | None = None) -> Any:
        # returns a zero-copy wrapper for JAX
        return JAXViewWrapper(tensor, cursor) if cursor is not None else tensor

    def from_dlpack(self, dlpack_capsule: Any) -> Any:
        capsule_obj = (
            dlpack_capsule
            if hasattr(dlpack_capsule, "__dlpack__")
            else DLPackObject(dlpack_capsule)
        )
        with self._x64_scope():
            return self._jax.dlpack.from_dlpack(capsule_obj)

    def to_dlpack(self, x: Any) -> Any:
        return x.__dlpack__()

    def index_update(self, x: Any, indices: Any, values: Any) -> Any:
        return x.at[indices].set(values)

    @property
    def supports_zero_copy_view(self) -> bool:
        return True

    def get_contiguous_view(self, tensor: Any, offset: int, count: int) -> Any:
        return JAXViewWrapper(tensor, slice(offset, offset + count))


def _reconstruct_jax_array(np_array):
    import jax

    return jax.device_put(np_array)


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
    def size(self):
        return prod(self._shape)

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
        if hasattr(self.arr, "devices") and any(
            d.platform != "cpu" for d in self.arr.devices()
        ):
            raise AttributeError
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
        return np.asarray(self._sliced_arr, *args, **kwargs)

    def __dlpack__(self, stream=None):
        return self._sliced_arr.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self._sliced_arr.__dlpack_device__()

    def __reduce__(self):
        sliced_np = np.asarray(self._sliced_arr)
        return (_reconstruct_jax_array, (sliced_np,))

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
        elif isinstance(self.slices, tuple | list):
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


# --- Numba JIT helpers (CPU acceleration) ---
_NUMBA_AVAILABLE = False
# Keep parallel-launch overhead out of small routing batches.  This is an
# execution threshold only; modulo semantics remain owned by TensorBackend.
_NUMBA_MODULO_MIN_SIZE = 1 << 17

try:
    import numba as _numba

    _NUMBA_AVAILABLE = True

    def _cached_numba_kernel(**options):
        def decorate(function):
            dispatcher = _numba.jit(**options)(function)

            @wraps(function)
            def call(*args):
                return call_numba_cached(dispatcher, *args)

            return call

        return decorate

    @_cached_numba_kernel(nopython=True, cache=True)
    def _numba_bincount_loop(x, out):
        """Increment out[x[i]] for each element. Caller must zero *out* first."""
        for i in range(x.shape[0]):
            idx = x[i]
            if 0 <= idx < out.shape[0]:
                out[idx] += 1

    @_cached_numba_kernel(nopython=True, cache=True)
    def _numba_bincount_weighted_loop(x, weights, out):
        """Like _numba_bincount_loop but adds weights[i] instead of 1."""
        for i in range(x.shape[0]):
            idx = x[i]
            if 0 <= idx < out.shape[0]:
                out[idx] += weights[i]

    @_cached_numba_kernel(nopython=True, cache=True)
    def _numba_cumsum_loop(x, out):
        """1-D prefix sum. Caller allocates *out*."""
        if x.shape[0] == 0:
            return
        out[0] = x[0]
        for i in range(1, x.shape[0]):
            out[i] = out[i - 1] + x[i]

    @_cached_numba_kernel(nopython=True, cache=True)
    def _numba_take_loop(data, indices, out):
        """1-D gather: out[i] = data[indices[i]]."""
        for i in range(indices.shape[0]):
            out[i] = data[indices[i]]

    @_cached_numba_kernel(nopython=True, parallel=True, nogil=True, cache=True)
    def _numba_modulo_loop(values, partition_count, out):
        """Compute integer modulo into *out* with a dtype-matched divisor.

        The caller must pass ``partition_count`` as a NumPy scalar matching
        ``values.dtype``.  In particular, passing a Python ``int`` for a
        ``uint64`` array makes Numba lower the divisor as signed and produces
        incorrect results for keys above ``2**63 - 1``.
        """
        values_flat = values.reshape(-1)
        out_flat = out.reshape(-1)
        for index in _numba.prange(values_flat.size):
            out_flat[index] = values_flat[index] % partition_count

except ImportError:
    pass


class HybridBackend(TensorBackend):
    """Hybrid backend: NumPy + optional Numba JIT (CPU) | CuPy (GPU).

    Device selection follows the "compute follows data" principle:
    - Factory methods (zeros/ones/arange/tensor/as_tensor) accept an explicit
      ``device`` parameter ('cpu' or 'gpu') that determines where the tensor lives.
    - Compute methods (argsort/bincount/cumsum/…) dispatch on the runtime type of
      their input arrays (np.ndarray → CPU path, cp.ndarray → GPU path).  No
      global device state is consulted, so CPU and GPU tensors can coexist in the
      same process without interference.
    """

    def __init__(self, default_device: str = "cpu", enable_numba_jit: bool = True):
        if default_device not in ("cpu", "cuda", "gpu"):
            raise ValueError(
                "HybridBackend.default_device must be 'cpu' or 'cuda', "
                f"got {default_device!r}."
            )
        if default_device == "gpu":
            default_device = "cuda"
        self._cp = None
        self._torch = None
        self._use_jit = enable_numba_jit and _NUMBA_AVAILABLE
        self._default_device = default_device
        self._worker_runtime_initialization_lock = threading.Lock()
        self._worker_runtime_initialized = False

        try:
            import torch  # pyrefly: ignore [missing-import]

            self._torch = torch
        except ImportError:
            pass

        try:
            import cupy as cp

            cp.cuda.runtime.getDeviceCount()
            self._cp = cp
        except Exception as exc:
            if default_device in ("gpu", "cuda"):
                raise RuntimeError(
                    f"HybridBackend(default_device={default_device!r}) requested "
                    f"but CuPy/CUDA is not available: {exc}"
                ) from exc

    @property
    def default_device(self) -> str:
        return self._default_device

    def initialize_worker_runtime(self) -> None:
        if self._worker_runtime_initialized:
            return
        with self._worker_runtime_initialization_lock:
            if self._worker_runtime_initialized:
                return
            if self._default_device == "cuda":
                self._initialize_cuda_worker_runtime()
            elif self._use_jit:
                self._initialize_numba_worker_runtime()
            self._worker_runtime_initialized = True

    def _initialize_cuda_worker_runtime(self) -> None:
        if self._cp is None:
            raise RuntimeError("CUDA worker runtime initialization requires CuPy.")
        values = self._cp.arange(2, dtype=self._cp.int32)
        is_strictly_increasing = self._cp.min(values[1:] > values[:-1])
        if not bool(is_strictly_increasing):
            raise RuntimeError(
                "CUDA worker runtime initialization produced an invalid ordering."
            )
        self._cp.cuda.Stream.null.synchronize()

    @staticmethod
    def _initialize_numba_worker_runtime() -> None:
        indices = np.asarray([0, 2, 1], dtype=np.intp)
        counts = np.zeros(3, dtype=np.int64)
        _numba_bincount_loop(indices, counts)

        weights = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        weighted_counts = np.zeros(3, dtype=np.float32)
        _numba_bincount_weighted_loop(indices, weights, weighted_counts)

        cumulative_input = np.asarray([1, 2, 3], dtype=np.int64)
        cumulative_output = np.empty_like(cumulative_input)
        _numba_cumsum_loop(cumulative_input, cumulative_output)

        source = np.asarray([4, 5, 6], dtype=np.int64)
        taken = np.empty(2, dtype=np.int64)
        _numba_take_loop(source, np.asarray([2, 0], dtype=np.int64), taken)

    def _lib(self, device: str):
        """Return the array library for *device* ('cpu' or 'gpu')."""
        if device in ("gpu", "cuda"):
            if self._cp is None:
                raise RuntimeError("GPU requested but CuPy / CUDA is not available.")
            return self._cp
        return np

    def _lib_for(self, x: Any):
        """Return the library matching the runtime type of *x*."""
        if self._cp is not None and isinstance(x, self._cp.ndarray):
            return self._cp
        return np

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def name(self) -> str:
        return "hybrid_gpu_avail" if self._cp is not None else "hybrid_cpu_only"

    def empty_cache(self) -> None:
        if self._cp is not None:
            self._cp.get_default_memory_pool().free_all_blocks()
            self._cp.get_default_pinned_memory_pool().free_all_blocks()

    def allocation_descriptor(self, value: Any) -> AllocationDescriptor:
        if hasattr(value, "unsafe_buffer_pointer") and hasattr(value, "nbytes"):
            pointer = int(value.unsafe_buffer_pointer())
            nbytes = int(value.nbytes)
            return AllocationDescriptor(
                AllocationIdentity(
                    "jax",
                    str(value.device),
                    pointer,
                    nbytes,
                    pointer,
                    nbytes,
                ),
                value,
            )
        if self._torch is not None and isinstance(value, self._torch.Tensor):
            if not value.is_contiguous():
                raise UnsupportedAllocationHandoffError(
                    "torch allocation handoff requires a contiguous tensor."
                )
            storage = value.untyped_storage()
            return AllocationDescriptor(
                AllocationIdentity(
                    "torch",
                    str(value.device),
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                    int(value.data_ptr()),
                    int(value.nelement() * value.element_size()),
                ),
                storage,
            )
        if isinstance(value, np.ndarray):
            if not (value.flags.c_contiguous or value.flags.f_contiguous):
                raise UnsupportedAllocationHandoffError(
                    "numpy allocation handoff requires a contiguous array."
                )
            owner = value
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            root_address = int(owner.__array_interface__["data"][0])
            view_address = int(value.__array_interface__["data"][0])
            return AllocationDescriptor(
                AllocationIdentity(
                    "numpy",
                    "cpu",
                    root_address,
                    owner.nbytes,
                    view_address,
                    value.nbytes,
                ),
                owner,
            )
        if self._cp is not None and isinstance(value, self._cp.ndarray):
            if not (value.flags.c_contiguous or value.flags.f_contiguous):
                raise UnsupportedAllocationHandoffError(
                    "cupy allocation handoff requires a contiguous array."
                )
            memory = value.data.mem
            return AllocationDescriptor(
                AllocationIdentity(
                    "cupy",
                    f"cuda:{int(value.device.id)}",
                    int(memory.ptr),
                    int(memory.size),
                    int(value.data.ptr),
                    int(value.nbytes),
                ),
                memory,
            )
        return super().allocation_descriptor(value)

    def readonly_view(self, value: Any) -> Any:
        if hasattr(value, "unsafe_buffer_pointer") and hasattr(value, "nbytes"):
            return value
        if isinstance(value, np.ndarray):
            readonly = value.view()
            readonly.flags.writeable = False
            return readonly
        if self._cp is not None and isinstance(value, self._cp.ndarray):
            return value
        return super().readonly_view(value)

    def asarray(
        self, x: Any, *, copy: bool | None = None, device: str | None = None
    ) -> Any:
        """Zero-copy adopt foreign tensor into this HybridBackend's default device domain.

        Why follow ``self._default_device`` instead of the source device:
        HybridBackend dispatches per-call via ``_lib_for(x)``, but most
        downstream operators (``where``, ``zeros``, ``cat`` ...) take
        *multiple* arguments and only inspect the first.  If ``adopt``
        followed the source device, a CUDA-resident jax.Array would be
        normalized to cupy while ``backend.zeros()`` (driven by
        ``_default_device``) retentions producing numpy, causing inevitable
        cross-device argument mixes (cupy condition + numpy branches).
        Forcing the *target* domain to ``_default_device`` retentions the
        eager pipeline strictly single-domain.

        Same-device foreign arrays use NumPy or CuPy DLPack adoption. Explicit
        copy=False rejects device-domain changes. Missing CUDA support and
        exporter errors propagate to the caller.
        """
        if copy is not True and device is None:
            raise ValueError(
                "HybridBackend.asarray requires an explicit device unless copy=True."
            )
        target_device = device or self._default_device
        if target_device not in ("cpu", "gpu", "cuda"):
            raise ValueError(
                f"HybridBackend.asarray device must be 'cpu' or 'cuda', got {device!r}."
            )
        target_is_gpu = target_device in ("gpu", "cuda")
        cp_mod = self._cp
        if target_is_gpu and cp_mod is None:
            raise RuntimeError(
                "CUDA array adoption requires CuPy and an available CUDA runtime."
            )

        x = _unwrap_ragged(x)

        # Native fast paths.
        if isinstance(x, np.ndarray):
            if target_is_gpu:
                if copy is False:
                    raise ValueError(
                        "HybridBackend.asarray(copy=False) cannot convert CPU data to CUDA."
                    )
                assert cp_mod is not None
                return cp_mod.array(x)
            return x.copy() if copy else x
        if cp_mod is not None and isinstance(x, cp_mod.ndarray):
            if target_is_gpu:
                return x.copy() if copy else x
            # cupy -> numpy via DLPack zero-copy when possible.
            if copy is False:
                raise ValueError(
                    "HybridBackend.asarray(copy=False) cannot convert CUDA data to CPU."
                )
            return cp_mod.asnumpy(x)

        # Identify torch tensors so we can honor .is_cuda even on builds
        # whose __dlpack_device__ is not perfectly aligned with reality.
        is_torch_tensor = False
        torch_on_cuda = False
        try:
            import torch  # type: ignore

            if isinstance(x, torch.Tensor):
                is_torch_tensor = True
                torch_on_cuda = bool(x.is_cuda)
        except ImportError:
            pass

        # Foreign DLPack-capable arrays (jax.Array, torch.Tensor, ...).
        if hasattr(x, "__dlpack__"):
            source_device, _ = x.__dlpack_device__()
            if source_device not in (1, 2, 10):
                raise ValueError(f"unsupported DLPack source device {source_device}")
            source_is_gpu = source_device in (2, 10)
            if copy is False and source_is_gpu != target_is_gpu:
                raise ValueError(
                    "HybridBackend.asarray(copy=False) cannot change device domains."
                )
            if not target_is_gpu and source_is_gpu:
                return np.array(x)
            if target_is_gpu:
                assert cp_mod is not None
                result = cp_mod.from_dlpack(x)
                return result.copy() if copy else result
            result = _adopt_dlpack_cpu(x)
            return result.copy() if copy else result

        if copy is False:
            raise TypeError(
                f"hybrid zero-copy conversion does not support {type(x).__name__}."
            )
        if is_torch_tensor:
            if target_is_gpu and not torch_on_cuda:
                assert cp_mod is not None
                return cp_mod.asarray(x.detach().cpu().numpy())
            if target_is_gpu and torch_on_cuda:
                assert cp_mod is not None
                return cp_mod.asarray(x.detach())
            # target=cpu
            return np.array(x.detach().cpu())
        if target_is_gpu:
            assert cp_mod is not None
            return cp_mod.array(x)
        return np.array(x)

    # ------------------------------------------------------------------
    # Creation / conversion
    # ------------------------------------------------------------------
    def tensor(self, data: Any, dtype: Any = None, device: str | None = None) -> Any:
        lib = self._lib(device or self._default_device)
        if lib is np:
            return force_writable(np.asarray(data, dtype=dtype))
        return lib.array(data, dtype=dtype)

    def as_tensor(self, data: Any, dtype: Any = None, device: str | None = None) -> Any:
        # If the input is already a native array (numpy / cupy), preserve
        # its device domain.  HybridBackend dispatches per-array, so a
        # cupy input must stay cupy even when ``self._default_device``
        # is 'cpu' -- otherwise we'd hit cupy's implicit device->host
        # conversion ban.  Only honor an *explicit* device override.
        if device is None:
            if isinstance(data, np.ndarray):
                return force_writable(
                    data if dtype is None else np.asarray(data, dtype=dtype)
                )
            if self._cp is not None and isinstance(data, self._cp.ndarray):
                return data if dtype is None else self._cp.asarray(data, dtype=dtype)
        lib = self._lib(device or self._default_device)
        if lib is np:
            if self._cp is not None and isinstance(data, self._cp.ndarray):
                data = self._cp.asnumpy(data)
            elif self._cp is not None and hasattr(data, "__cuda_array_interface__"):
                data = self._cp.asarray(data).get()
            return force_writable(np.asarray(data, dtype=dtype))
        return lib.asarray(data, dtype=dtype)

    def to_numpy(self, tensor: Any) -> Any:
        if self._cp is not None and isinstance(tensor, self._cp.ndarray):
            return self._cp.asnumpy(tensor)
        return np.asarray(tensor)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    def zeros(self, shape: Any, dtype: Any = None, device: str | None = None) -> Any:
        return self._lib(device or self._default_device).zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None, device: str | None = None) -> Any:
        return self._lib(device or self._default_device).ones(shape, dtype=dtype)

    def full(
        self, shape: Any, fill_value: Any, dtype: Any = None, device: str | None = None
    ) -> Any:
        return self._lib(device or self._default_device).full(
            shape, fill_value, dtype=dtype
        )

    def arange(
        self,
        start: int,
        end: int,
        step: int = 1,
        dtype: Any = None,
        device: str | None = None,
    ) -> Any:
        return self._lib(device or self._default_device).arange(
            start, end, step, dtype=dtype
        )

    # ------------------------------------------------------------------
    # Shape / dtype / metadata
    # ------------------------------------------------------------------
    def is_tensor(self, obj: Any) -> bool:
        if self._cp is not None and isinstance(obj, self._cp.ndarray):
            return True
        return isinstance(obj, np.ndarray)

    def is_scalar(self, obj: Any) -> bool:
        if self._cp is not None and isinstance(obj, self._cp.ndarray):
            return obj.ndim == 0
        return isinstance(obj, np.ndarray) and obj.ndim == 0

    def dtype(self, tensor: Any) -> Any:
        return tensor.dtype

    def shape(self, tensor: Any) -> Any:
        return tensor.shape

    def size(self, tensor: Any) -> int:
        return tensor.size

    def cast(self, tensor: Any, dtype: Any) -> Any:
        return tensor.astype(dtype)

    def item(self, tensor: Any) -> Any:
        return tensor.item()

    def reshape(self, tensor: Any, shape: tuple[int, ...]) -> Any:
        return self._lib_for(tensor).reshape(tensor, shape)

    def unravel_index(self, indices: Any, shape: tuple[int, ...]) -> Any:
        return self._lib_for(indices).unravel_index(indices, shape)

    def float32(self) -> Any:
        return np.float32

    def int32(self) -> Any:
        return np.int32

    def equal(self, x: Any, y: Any) -> bool:
        lib = self._lib_for(x)
        return bool(lib.array_equal(x, y))

    # ------------------------------------------------------------------
    # Concat / stack
    # ------------------------------------------------------------------
    def cat(self, tensors: list[Any], dim: int = 0) -> Any:
        lib = self._lib_for(tensors[0]) if tensors else np
        return lib.concatenate(tensors, axis=dim)

    def stack(self, tensors: list[Any], dim: int = 0) -> Any:
        lib = self._lib_for(tensors[0]) if tensors else np
        return lib.stack(tensors, axis=dim)

    # ------------------------------------------------------------------
    # Arithmetic / element-wise
    # ------------------------------------------------------------------
    def floor_divide(self, x: Any, y: Any) -> Any:
        return self._lib_for(x).floor_divide(x, y)

    def mod(self, x: Any, y: Any) -> Any:
        lib = self._lib_for(x)
        if (
            lib is np
            and self._use_jit
            and isinstance(x, np.ndarray)
            and x.flags.c_contiguous
            and x.size >= _NUMBA_MODULO_MIN_SIZE
            and x.dtype
            in {
                np.dtype(np.int32),
                np.dtype(np.uint32),
                np.dtype(np.int64),
                np.dtype(np.uint64),
            }
        ):
            try:
                partition_count = operator.index(y)
            except TypeError:
                partition_count = None
            if partition_count is not None and partition_count > 0:
                dtype_limits = np.iinfo(x.dtype)
                if partition_count <= int(dtype_limits.max):
                    divisor = np.asarray(partition_count, dtype=x.dtype)[()]
                    out = np.empty_like(x)
                    _numba_modulo_loop(x, divisor, out)
                    return out
        return lib.mod(x, y)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        if self._cp is not None and (
            isinstance(condition, self._cp.ndarray)
            or isinstance(x, self._cp.ndarray)
            or isinstance(y, self._cp.ndarray)
        ):
            condition_arg = (
                condition
                if isinstance(condition, self._cp.ndarray)
                else self._cp.asarray(condition)
            )
            x_arg = x if not isinstance(x, np.ndarray) else self._cp.asarray(x)
            y_arg = y if not isinstance(y, np.ndarray) else self._cp.asarray(y)
            return self._cp.where(condition_arg, x_arg, y_arg)
        return np.where(condition, x, y)

    # ------------------------------------------------------------------
    # Reduction
    # ------------------------------------------------------------------
    def sum(self, x: Any, dim: Any = None) -> Any:
        return self._lib_for(x).sum(x, axis=dim)

    def mean(self, x: Any) -> Any:
        return self._lib_for(x).mean(x)

    def max(self, x: Any) -> Any:
        return self._lib_for(x).max(x)

    def min(self, x: Any) -> Any:
        return self._lib_for(x).min(x)

    def cumsum(self, x: Any, dim: int = 0) -> Any:
        if self._cp is not None and isinstance(x, self._cp.ndarray):
            return self._cp.cumsum(x, axis=dim)
        # CPU: optional Numba JIT for 1-D
        if self._use_jit and x.ndim == 1:
            out = np.empty_like(x)
            _numba_cumsum_loop(x, out)
            return out
        return np.cumsum(x, axis=dim)

    # ------------------------------------------------------------------
    # Sorting / indexing
    # ------------------------------------------------------------------
    def argsort(
        self, x: Any, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> Any:
        kind = "stable" if stable else None
        lib = self._lib_for(x)
        return lib.argsort(-x if descending else x, axis=dim, kind=kind)

    def sort(
        self, x: Any, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> tuple[Any, Any]:
        idx = self.argsort(x, dim, descending, stable=stable)
        lib = self._lib_for(x)
        return lib.take_along_axis(x, idx, axis=dim), idx

    def gather(self, x: Any, dim: int, index: Any) -> Any:
        lib = self._lib_for(x)
        if x.shape[dim] == 0:
            return lib.zeros(index.shape, dtype=x.dtype)
        return lib.take_along_axis(x, index, axis=dim)

    def take(self, x: Any, indices: Any) -> Any:
        if self._cp is not None and isinstance(x, self._cp.ndarray):
            if x.shape[0] == 0:
                indices_shape = getattr(indices, "shape", ())
                return self._cp.zeros(indices_shape + x.shape[1:], dtype=x.dtype)
            return x[indices]
        # CPU path
        if x.shape[0] == 0:
            indices_shape = getattr(indices, "shape", ())
            return np.zeros(indices_shape + x.shape[1:], dtype=x.dtype)
        if self._use_jit and x.ndim == 1 and indices.ndim == 1:
            flat_x = x.ravel()
            flat_idx = indices.ravel()
            out = np.empty(flat_idx.shape, dtype=flat_x.dtype)
            _numba_take_loop(flat_x, flat_idx, out)
            return out.reshape(indices.shape)
        return np.take(x, indices, axis=0)

    def select(self, x: Any, dim: int, index: int) -> Any:
        lib = self._lib_for(x)
        if x.shape[dim] == 0:
            new_shape = x.shape[:dim] + x.shape[dim + 1 :]
            return lib.zeros(new_shape, dtype=x.dtype)
        return lib.take(x, index, axis=dim)

    def diff(self, x: Any, dim: int = -1) -> Any:
        return self._lib_for(x).diff(x, axis=dim)

    def bincount(self, x: Any, weights: Any | None = None, minlength: int = 0) -> Any:
        if self._cp is not None and isinstance(x, self._cp.ndarray):
            return self._cp.bincount(x, weights=weights, minlength=minlength)
        # CPU: optional Numba JIT
        if self._use_jit:
            host_x = np.asarray(x, dtype=np.intp)
            n_out = max(int(host_x.max()) + 1 if host_x.size > 0 else 0, minlength)
            if weights is not None:
                w_arr = np.asarray(weights)
                out = np.zeros(n_out, dtype=w_arr.dtype)
                _numba_bincount_weighted_loop(host_x, w_arr, out)
            else:
                out = np.zeros(n_out, dtype=np.int64)
                _numba_bincount_loop(host_x, out)
            return out
        return np.bincount(x, weights=weights, minlength=minlength)

    def repeat_interleave(
        self,
        x: Any,
        repeats: Any,
        dim: int | None = None,
        total_repeat_length: int | None = None,
    ) -> Any:
        lib = self._lib_for(x)
        if (
            self._cp is not None
            and lib is self._cp
            and isinstance(repeats, self._cp.ndarray)
            and repeats.ndim == 1
        ):
            if x.ndim != 1 or dim not in (None, 0):
                raise ValueError(
                    "CuPy vector repeat requires a 1D input and dim=None or dim=0."
                )
            if total_repeat_length is None:
                total_repeat_length = int(self._cp.sum(repeats).item())
            if x.shape[0] == 0:
                return self._cp.zeros((total_repeat_length,), dtype=x.dtype)
            boundaries = self._cp.cumsum(repeats)
            positions = self._cp.arange(total_repeat_length, dtype=self._cp.int32)
            source_indices = self._cp.searchsorted(boundaries, positions, side="right")
            valid = source_indices < x.shape[0]
            safe_indices = self._cp.where(valid, source_indices, 0)
            return self._cp.where(valid, x[safe_indices], 0)

        result = lib.repeat(x, repeats, axis=dim)
        if total_repeat_length is not None:
            repeat_dim = dim if dim is not None else 0
            actual_len = result.shape[repeat_dim]
            if actual_len < total_repeat_length:
                pad_shape = list(result.shape)
                pad_shape[repeat_dim] = total_repeat_length - actual_len
                padding = lib.zeros(pad_shape, dtype=result.dtype)
                result = lib.concatenate([result, padding], axis=repeat_dim)
            elif actual_len > total_repeat_length:
                slices = [slice(None)] * result.ndim
                slices[repeat_dim] = slice(0, total_repeat_length)
                result = result[tuple(slices)]
        return result

    def nonzero(self, x: Any, *, size: Any = None, fill_value: Any = None) -> Any:
        lib = self._lib_for(x)
        result = lib.nonzero(x)
        if size is not None:
            padded = []
            for arr in result:
                if arr.size < size:
                    pad_len = size - arr.size
                    pad_val = fill_value if fill_value is not None else 0
                    pad = lib.full(pad_len, pad_val, dtype=arr.dtype)
                    padded.append(lib.concatenate([arr, pad]))
                else:
                    padded.append(arr[:size])
            result = tuple(padded)
        return result

    # ------------------------------------------------------------------
    # Dynamic slicing
    # ------------------------------------------------------------------
    def dynamic_slice(self, x: Any, start: Any, size: Any) -> Any:
        slices = []
        for s, l in zip(start, size, strict=True):
            if self.is_scalar(s):
                s = int(s)
            if self.is_scalar(l):
                l = int(l)
            slices.append(slice(int(s), int(s) + int(l)))
        return x[tuple(slices)]

    def dynamic_slice_in_dim(self, x: Any, start: Any, size: int, axis: int = 0) -> Any:
        if self.is_scalar(start):
            start = int(start)
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(int(start), int(start) + size)
        return x[tuple(slices)]

    def dynamic_slice_update(self, x: Any, update: Any, start: Any) -> Any:
        if isinstance(start, int):
            x[start : start + update.shape[0]] = update
        elif isinstance(start, list | tuple):
            sl = [slice(s, s + update.shape[i]) for i, s in enumerate(start)]
            x[tuple(sl)] = update
        elif self.is_scalar(start):
            s = int(start)
            x[s : s + update.shape[0]] = update
        else:
            raise NotImplementedError(f"Unsupported start type: {type(start)}")
        return x

    # ------------------------------------------------------------------
    # Unique
    # ------------------------------------------------------------------
    def unique_consecutive(
        self, x: Any, return_inverse: bool = False, return_counts: bool = False
    ) -> Any:
        lib = self._lib_for(x)
        mask = lib.concatenate([lib.array([True]), x[1:] != x[:-1]])
        uniques = x[mask]
        if return_inverse:
            inverse = lib.cumsum(mask) - 1
            if return_counts:
                indices = lib.flatnonzero(mask)
                appended = lib.concatenate([indices, lib.array([len(x)])])
                counts = lib.diff(appended)
                return uniques, inverse, counts
            return uniques, inverse
        if return_counts:
            indices = lib.flatnonzero(mask)
            appended = lib.concatenate([indices, lib.array([len(x)])])
            counts = lib.diff(appended)
            return uniques, counts
        return uniques

    def unique(
        self,
        x: Any,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
    ) -> Any:
        lib = self._lib_for(x)
        return lib.unique(
            x,
            return_index=return_index,
            return_inverse=return_inverse,
            return_counts=return_counts,
        )

    # ------------------------------------------------------------------
    # Tile / index update
    # ------------------------------------------------------------------
    def tile(self, x: Any, dims: tuple[int, ...]) -> Any:
        return self._lib_for(x).tile(x, dims)

    def index_update(self, x: Any, indices: Any, values: Any) -> Any:
        x[indices] = values
        return x

    # ------------------------------------------------------------------
    # Print
    # ------------------------------------------------------------------
    def print(self, fmt: str, **kwargs) -> None:
        print(fmt.format(**kwargs))

    # ------------------------------------------------------------------
    # Functional: vmap / tree / scan
    # ------------------------------------------------------------------
    def vmap(self, fun: Any, in_axes: Any = 0, out_axes: Any = 0) -> Any:
        backend = self

        def vmap_fn(*args):
            if isinstance(in_axes, int):
                batch_size = args[0].shape[in_axes]
            elif isinstance(in_axes, tuple):
                for a, ax in zip(args, in_axes, strict=True):
                    if ax is not None:
                        batch_size = a.shape[ax]
                        break
                else:
                    raise ValueError("No vmapped axis found in in_axes")
            else:
                batch_size = args[0].shape[0]

            results = []
            for i in range(batch_size):
                vmapped_args = []
                for j, arg in enumerate(args):
                    if not backend.is_tensor(arg):
                        vmapped_args.append(arg)
                        continue
                    if isinstance(in_axes, int):
                        ax = in_axes
                    elif isinstance(in_axes, tuple):
                        ax = in_axes[j] if j < len(in_axes) else 0
                    else:
                        ax = 0
                    if ax is None:
                        vmapped_args.append(arg)
                    else:
                        idx = [slice(None)] * arg.ndim
                        idx[ax] = i
                        vmapped_args.append(arg[tuple(idx)])
                results.append(fun(*vmapped_args))

            out_axis = out_axes if isinstance(out_axes, int) else 0
            return backend.stack(results, dim=out_axis)

        return vmap_fn

    def tree_map(self, fun: Any, tree: Any) -> Any:
        # Pure-Python pytree map; HybridBackend does not depend on JAX.
        if isinstance(tree, dict):
            return {k: self.tree_map(fun, v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [self.tree_map(fun, v) for v in tree]
        if isinstance(tree, tuple):
            return type(tree)(self.tree_map(fun, v) for v in tree)
        return fun(tree)

    def tree_flatten(self, tree: Any) -> tuple[list[Any], Any]:
        # Pure-Python flatten producing a flat leaves list. The second
        # element is a structural placeholder (top-level container type).
        leaves: list[Any] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list | tuple):
                for v in node:
                    _walk(v)
            else:
                leaves.append(node)

        _walk(tree)
        return leaves, type(tree)

    def scan(self, f: Any, init: Any, xs: Any, length: int | None = None) -> Any:
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
                raise NotImplementedError("Tuple xs scan not fully supported yet")
        for x_item in xs_iter:
            carry, y = f(carry, x_item)
            ys.append(y)
        if not ys:
            return carry, []
        return carry, self.stack(ys)

    # ------------------------------------------------------------------
    # DLPack interop -- zero-copy
    # ------------------------------------------------------------------
    @property
    def supports_zero_copy_view(self) -> bool:
        return True  # Both NumPy and CuPy support zero-copy DLPack

    def get_contiguous_view(self, tensor: Any, offset: int, count: int) -> Any:
        return tensor[offset : offset + count]

    def get_dlpack_slice(self, tensor: Any, cursor: Any | None = None) -> Any:
        sliced = tensor[cursor] if cursor is not None else tensor
        dev_type, dev_id = sliced.__dlpack_device__()
        return DLPackObject(sliced.__dlpack__(), device=(dev_type, dev_id))

    def from_dlpack(self, dlpack_capsule: Any) -> Any:
        # Select library based on the DLPack device field, not global state.
        if hasattr(dlpack_capsule, "__dlpack_device__"):
            device_type, _ = dlpack_capsule.__dlpack_device__()
            # DLDeviceType: kDLCPU=1, kDLCUDA=2
            lib = self._cp if (self._cp is not None and device_type == 2) else np
        else:
            lib = np
        if lib is np:
            return force_writable(_adopt_dlpack_cpu(dlpack_capsule))
        capsule_obj = (
            dlpack_capsule
            if hasattr(dlpack_capsule, "__dlpack__")
            else DLPackObject(dlpack_capsule)
        )
        return force_writable(lib.from_dlpack(capsule_obj))

    def to_dlpack(self, x: Any) -> Any:
        return prepare_dlpack_input(x).__dlpack__()


_BACKEND: TensorBackend | None = None


def get_backend() -> TensorBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = HybridBackend()
    return _BACKEND


def set_backend(name: str):
    global _BACKEND
    if name.lower() == "jax":
        _BACKEND = JaxBackend()
    elif name.lower() in ["torch", "pytorch"]:
        _BACKEND = TorchBackend()
    elif name.lower() == "cupy":
        _BACKEND = HybridBackend(default_device="gpu")
    elif name.lower() in ("numba", "numba_cpu"):
        _BACKEND = HybridBackend(default_device="cpu", enable_numba_jit=True)
    elif name.lower() == "hybrid":
        _BACKEND = HybridBackend()
    else:
        raise ValueError(f"Unknown backend: {name}")


def set_hybrid_device(device: str) -> None:
    global _BACKEND
    backend = HybridBackend(
        default_device=device,
        enable_numba_jit=device == "cpu",
    )
    backend.initialize_worker_runtime()
    _BACKEND = backend


@contextlib.contextmanager
def use_backend(name: str):
    global _BACKEND
    old_backend = _BACKEND
    try:
        set_backend(name)
        yield
    finally:
        _BACKEND = old_backend
