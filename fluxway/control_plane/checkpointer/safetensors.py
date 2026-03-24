"""
SafeTensorCheckpointer: safetensors-backed implementation of FluxwayCheckpointer.

Designed for PyTorch backends; also supports JAX and NumPy arrays via implicit
numpy conversion.

State structure from workers: {spmd_id: {rank: {field: value}}}

Values can be:
  - JAX arrays / nnx.State pytrees  → flattened via jax.tree_util, stored in .safetensors
  - PyTorch tensors                  → stored in .safetensors
  - NumPy arrays                     → stored in .safetensors
  - Python scalars / dicts / lists   → stored in meta.json as "non-tensor" metadata

Each checkpoint step is written to ``<ckpt_dir>/<step>/``:
  model.safetensors   — flattened array leaves
  meta.json           — key structure + non-tensor fields
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fluxway.control_plane.checkpointer.base import FluxwayCheckpointer

# Separator that cannot appear in spmd_id, rank, or field names
_SEP = "\x00"


def _try_jax_flatten(value: Any) -> Optional[Dict[str, np.ndarray]]:
    """
    If ``value`` is a JAX pytree (including nnx.State), flatten it to
    {leaf_path_str: ndarray}.  Returns None if JAX is not available or the
    value is not a JAX pytree.
    """
    try:
        import jax

        leaves_with_paths = jax.tree_util.tree_leaves_with_path(value)
        if not leaves_with_paths:
            return None
        result = {}
        for path, leaf in leaves_with_paths:
            path_str = "/".join(
                str(p.key) if hasattr(p, "key") else str(p) for p in path
            )
            result[path_str] = np.asarray(leaf)
        return result
    except Exception:
        return None


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a single array to NumPy. Raises TypeError for unsupported types."""
    if isinstance(value, np.ndarray):
        return value
    # JAX
    if hasattr(value, "__jax_array__") or type(value).__module__.startswith("jax"):
        return np.asarray(value)
    # PyTorch
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    raise TypeError(
        f"SafeTensorCheckpointer: unsupported array type {type(value)!r}. "
        f"Supported: JAX arrays, NumPy arrays, PyTorch tensors."
    )


def _is_array_like(value: Any) -> bool:
    """True if value is a raw array (not a pytree container)."""
    if isinstance(value, (np.ndarray, list, int, float, bool, str, dict, type(None))):
        return False
    if hasattr(value, "detach") or hasattr(value, "__jax_array__"):
        return True
    module = type(value).__module__
    return module.startswith("jax") or module.startswith("numpy")


def _flatten_state(
    state: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Recursively flatten nested state to:
      - flat_tensors: {composite_key: ndarray}  for safetensors
      - meta_fields:  {composite_key: value}    for JSON (non-tensor leaves)

    composite_key format: "{spmd_id}\\x00{rank}\\x00{field}[\\x00{pytree_path}]"
    """
    flat_tensors: Dict[str, np.ndarray] = {}
    meta_fields: Dict[str, Any] = {}

    for spmd_id, rank_states in state.items():
        for rank, fields in rank_states.items():
            for field_key, value in fields.items():
                prefix = f"{spmd_id}{_SEP}{rank}{_SEP}{field_key}"

                if isinstance(value, np.ndarray):
                    # Plain NumPy — store directly
                    flat_tensors[prefix] = value
                elif hasattr(value, "detach"):
                    # PyTorch tensor
                    flat_tensors[prefix] = value.detach().cpu().numpy()
                elif hasattr(value, "__jax_array__") or (
                    not isinstance(
                        value, (int, float, bool, str, dict, list, type(None))
                    )
                    and type(value).__module__.startswith("jax")
                ):
                    # Raw JAX DeviceArray — convert directly
                    flat_tensors[prefix] = np.asarray(value)
                else:
                    # Try JAX pytree flatten (handles nnx.State and other JAX containers).
                    # Never call on plain Python types — they are valid JAX pytree leaves
                    # but produce spurious empty-path entries.
                    is_python_primitive = isinstance(
                        value,
                        (int, float, bool, str, bytes, dict, list, tuple, type(None)),
                    )
                    if not is_python_primitive:
                        jax_leaves = _try_jax_flatten(value)
                        if jax_leaves is not None and len(jax_leaves) > 0:
                            for path_str, arr in jax_leaves.items():
                                flat_tensors[f"{prefix}{_SEP}{path_str}"] = arr
                            continue
                    # Non-tensor: Python scalar, dict, list — goes to JSON
                    meta_fields[prefix] = value

    return flat_tensors, meta_fields


def _unflatten_state(
    flat_tensors: Dict[str, Any], meta_fields: Dict[str, Any], key_map: Dict[str, Dict]
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Reconstruct the nested state dict from flat safetensors + JSON metadata.

    ``key_map`` (from meta.json) records for each composite_key whether its
    value was a plain tensor, a JAX pytree leaf, or a non-tensor field.
    """
    state: Dict[str, Any] = {}

    def _set_nested(spmd_id, rank, field_key, value):
        state.setdefault(spmd_id, {}).setdefault(rank, {})[field_key] = value

    # Group pytree leaves back together
    pytree_groups: Dict[str, Dict[str, np.ndarray]] = {}

    for composite_key, arr in flat_tensors.items():
        parts = composite_key.split(_SEP)
        if len(parts) == 3:
            spmd_id, rank_str, field_key = parts
            _set_nested(spmd_id, int(rank_str), field_key, arr)
        elif len(parts) == 4:
            spmd_id, rank_str, field_key, path_str = parts
            prefix = f"{spmd_id}{_SEP}{rank_str}{_SEP}{field_key}"
            pytree_groups.setdefault(prefix, {})[path_str] = arr
        else:
            raise ValueError(
                f"SafeTensorCheckpointer: unexpected key format {composite_key!r}"
            )

    # Restore pytree groups — return as plain dict of numpy arrays;
    # DenseSPMD.restore_state calls nnx.update which accepts numpy arrays.
    for prefix, leaves in pytree_groups.items():
        parts = prefix.split(_SEP)
        spmd_id, rank_str, field_key = parts
        # Build a flat key->array dict that nnx.update accepts
        _set_nested(spmd_id, int(rank_str), field_key, leaves)

    # Non-tensor fields
    for composite_key, value in meta_fields.items():
        parts = composite_key.split(_SEP)
        if len(parts) != 3:
            raise ValueError(
                f"SafeTensorCheckpointer: unexpected meta key format {composite_key!r}"
            )
        spmd_id, rank_str, field_key = parts
        _set_nested(spmd_id, int(rank_str), field_key, value)

    return state


class SafeTensorCheckpointer(FluxwayCheckpointer):
    """
    Saves and loads Fluxway snapshots using the ``safetensors`` format.

    Compatible with JAX (including nnx.State pytrees), NumPy, and PyTorch arrays.
    Non-tensor data (e.g. reader dataset position) is stored in a JSON sidecar.

    Each checkpoint step is written to ``<ckpt_dir>/<step>/``:
      model.safetensors — flattened array leaves
      meta.json         — key structure + non-tensor metadata

    Args:
        controller:     Ray remote handle returned by ``create_controller``.
        ckpt_dir:       Root directory for checkpoints.
        interval_steps: Steps between successive automatic checkpoints.
        max_to_keep:    Maximum number of on-disk checkpoints to retain.
    """

    def __init__(
        self, controller, ckpt_dir: str, *, interval_steps: int, max_to_keep: int = 2
    ) -> None:
        super().__init__(controller, ckpt_dir, interval_steps=interval_steps)

        import safetensors  # noqa: F401 — validate dependency at construction

        self._max_to_keep = max_to_keep

    def list_steps(self) -> List[int]:
        root = Path(self._ckpt_dir)
        if not root.exists():
            return []
        return sorted(
            int(d.name) for d in root.iterdir() if d.is_dir() and d.name.isdigit()
        )

    def _persist(self, step: int, state: Dict[str, Any]) -> None:
        from safetensors.numpy import save_file

        step_dir = Path(self._ckpt_dir) / str(step)
        step_dir.mkdir(parents=True, exist_ok=True)

        flat_tensors, meta_fields = _flatten_state(state)
        save_file(flat_tensors, str(step_dir / "model.safetensors"))

        with open(step_dir / "meta.json", "w") as f:
            json.dump(meta_fields, f)

        self._evict_old_checkpoints()

    def _load(self, step: int) -> Dict[str, Any]:
        from safetensors.numpy import load_file

        step_dir = Path(self._ckpt_dir) / str(step)
        tensor_path = step_dir / "model.safetensors"
        meta_path = step_dir / "meta.json"

        if not tensor_path.exists():
            raise FileNotFoundError(
                f"SafeTensorCheckpointer: checkpoint file not found: {tensor_path}"
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"SafeTensorCheckpointer: metadata file not found: {meta_path}"
            )

        flat_tensors = load_file(str(tensor_path))
        with open(meta_path) as f:
            meta_fields = json.load(f)

        return _unflatten_state(flat_tensors, meta_fields, key_map={})

    def _evict_old_checkpoints(self) -> None:
        """Remove oldest checkpoint directories exceeding max_to_keep."""
        steps = self.list_steps()
        for old_step in steps[: -self._max_to_keep]:
            shutil.rmtree(Path(self._ckpt_dir) / str(old_step))
