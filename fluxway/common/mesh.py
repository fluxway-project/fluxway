"""
Device Mesh Abstraction.
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DeviceMesh:
    """
    DeviceMesh represents the topology of devices for distributed computation.

    It maps logical coordinates (axis names/indices) to physical device ranks.

    Attributes:
        shape (Tuple[int, ...]): The shape of the mesh (number of devices per axis).
        axis_names (Tuple[str, ...]): Names for each axis of the mesh.
        device_ids (np.ndarray): Array of physical device IDs (ranks) with shape `shape`.
    """

    def __init__(
        self,
        device_shape: Optional[Tuple[int, ...]] = None,
        axis_names: Tuple[str, ...] = None,
        device_ids: Optional[Any] = None,
    ):
        """
        Initialize a DeviceMesh.

        Args:
            device_shape: Optional tuple defining the dimensions of the mesh.
            axis_names: Tuple of names for each axis.
            device_ids: Optional physical device IDs.
                        If provided as a multi-dimensional array, device_shape will be inferred.
                        If provided as a 1D array and device_shape is given, it will be reshaped.
        """
        if axis_names is None:
            raise ValueError("axis_names must be provided.")

        # Determine device_ids and shape
        if device_ids is not None:
            self.device_ids = np.array(device_ids)
            if device_shape is None:
                # Infer shape from device_ids
                if self.device_ids.ndim == 1:
                    # Default to 1D if it's a flat list and no shape provided
                    # But we usually want it to match axis_names
                    if len(axis_names) == 1:
                        self.shape = self.device_ids.shape
                    else:
                        raise ValueError(
                            f"Cannot infer multi-dimensional shape from 1D device_ids without device_shape. axis_names: {axis_names}"
                        )
                else:
                    self.shape = self.device_ids.shape
            else:
                # Shape provided, validate and reshape if needed
                self.shape = tuple(device_shape)
                if self.device_ids.size != np.prod(self.shape):
                    raise ValueError(
                        f"device_ids size {self.device_ids.size} does not match device_shape {self.shape}"
                    )
                self.device_ids = self.device_ids.reshape(self.shape)
        elif device_shape is not None:
            # Only shape provided, generate default IDs
            self.shape = tuple(device_shape)
            self.device_ids = np.arange(np.prod(self.shape)).reshape(self.shape)
        else:
            raise ValueError("Either device_shape or device_ids must be provided.")

        if len(self.shape) != len(axis_names):
            raise ValueError(
                f"Shape {self.shape} and axis_names {axis_names} must have same length."
            )

        self.axis_names = tuple(axis_names)
        self.axis_map = dict(zip(self.axis_names, range(len(self.axis_names))))
        self.size_map = dict(zip(self.axis_names, self.shape))

        # Inverse map: rank -> coordinates (cached)
        self._inverse_map = {}
        it = np.nditer(self.device_ids, flags=["multi_index", "refs_ok"])
        for x in it:
            rank = int(x)
            self._inverse_map[rank] = it.multi_index

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return self.device_ids.size

    def get_axis_size(self, axis_name: str) -> int:
        """Get the size (number of devices) of the specified axis."""
        if axis_name not in self.size_map:
            raise ValueError(f"Axis {axis_name} not found in mesh {self.axis_names}")
        return self.size_map[axis_name]

    def get_coordinate(self, rank_id: int) -> Tuple[int, ...]:
        """Get the multi-dimensional logical coordinates of a physical rank in the mesh."""
        if rank_id not in self._inverse_map:
            raise ValueError(f"Rank {rank_id} not found in mesh")
        return self._inverse_map[rank_id]

    def get_rank(self, coord: Tuple[int, ...]) -> int:
        """Get the physical rank corresponding to logical coordinates."""
        if len(coord) != self.ndim:
            raise ValueError(
                f"Coordinate {coord} dim does not match mesh ndim {self.ndim}"
            )
        return self.device_ids[coord]

    def coordinate_to_device(self, coords: Tuple[Any, ...]) -> Any:
        """Vectorized get_rank. coords elements can be tensors or numpy arrays."""
        # Ensure we use JAX-compatible indexing if device_ids is a Tracer
        # backend.take or simple indexing should work.
        return self.device_ids[coords]

    def unravel_index(self, ranks: Any) -> Tuple[Any, ...]:
        """Convert physical ranks to multi-dimensional logical coordinates."""
        from fluxway.common.backend import get_backend

        try:
            backend = get_backend()
            return backend.unravel_index(ranks, self.shape)
        except Exception:
            # Fallback for simple numpy usage if backend fails or not set
            return np.unravel_index(ranks, self.shape)

    def flatten(self) -> List[int]:
        """Return a flattened list of physical device IDs."""
        from fluxway.common.backend import get_backend

        try:
            backend = get_backend()
            # If device_ids is tracer, to_numpy will fail, but flatten() usage implies host-side often.
            # If used in JIT, we return list? No, JIT can't return list of ints derived from tracers.
            # Assume flatten is for eager use.
            return backend.to_numpy(self.device_ids).flatten().tolist()
        except Exception:
            return self.device_ids.flatten().tolist()

    def __repr__(self) -> str:
        return f"DeviceMesh(shape={self.shape}, axis_names={self.axis_names})"

    def to_backend_mesh(self, backend_engine: Any = None):
        """Convert to the mesh representation of the corresponding backend framework (e.g. JAX Mesh or PyTorch DeviceMesh)."""
        raise NotImplementedError("Backend conversion not implemented yet.")


# JAX Pytree Registration
try:
    import jax

    def _dm_flatten(v):
        children = (v.device_ids,)
        aux_data = (v.shape, v.axis_names)
        return children, aux_data

    def _dm_unflatten(aux_data, children):
        # Reconstruct without invoking __init__ (avoids numpy conversion of Tracers)
        obj = object.__new__(DeviceMesh)
        obj.shape, obj.axis_names = aux_data
        obj.device_ids = children[0]

        # Restore derived maps (static)
        obj.axis_map = dict(zip(obj.axis_names, range(len(obj.axis_names))))
        obj.size_map = dict(zip(obj.axis_names, obj.shape))

        # _inverse_map relies on iterating device_ids.
        # If device_ids is a Tracer, we CANNOT iterate it.
        # So _inverse_map will be empty or invalid inside JIT.
        # BUT _inverse_map is only used by `get_coordinate` (scalar) and `get_rank` (via logic).
        # Vectorized methods (unravel_index, coordinate_to_device) use tensor ops (backend)
        # and do NOT use _inverse_map.
        # So it is safe to skip _inverse_map construction inside JIT.
        obj._inverse_map = {}

        return obj

    jax.tree_util.register_pytree_node(DeviceMesh, _dm_flatten, _dm_unflatten)
except ImportError:
    pass
