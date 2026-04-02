"""
Per-Worker Orbax serialization utilities.

OrbaxBackend persists the user's pytree treedef alongside arrays via
CompositeCheckpointHandler(state=StandardSave, pytree_meta=JsonSave).
On restore, the treedef reconstructs the EXACT same pytree structure
(including nnx.State / nnx.Variable wrappers), so users get back
exactly what they returned from save_state() — zero framework leakage.
"""

import asyncio
import base64
import pickle
from pathlib import Path
from typing import Any, Dict

import jax.tree_util as jtu
from etils import epath

from fluxway.control_plane.checkpointer.backend import WorkerCheckpointBackend

_EMPTY = object()


def _split_hybrid_state(state: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    import jax

    def _split(obj: Any) -> tuple[Any, Any]:
        try:
            from flax import nnx

            if isinstance(obj, nnx.State):
                obj = dict(obj)
            if isinstance(obj, nnx.Variable):
                obj = obj.get_raw_value()
        except ImportError:
            pass

        if isinstance(obj, dict):
            global_part, local_part = {}, {}
            for k, v in obj.items():
                gv, lv = _split(v)
                if gv is not _EMPTY:
                    global_part[k] = gv
                if lv is not _EMPTY:
                    local_part[k] = lv
            return (global_part or _EMPTY), (local_part or _EMPTY)

        if isinstance(obj, (jax.Array, jax.ShapeDtypeStruct)) and not (
            hasattr(obj, "sharding")
            and isinstance(obj.sharding, jax.sharding.SingleDeviceSharding)
        ):
            return obj, _EMPTY

        return _EMPTY, obj

    global_state, local_state = _split(state)
    return (global_state if global_state is not _EMPTY else {}), (
        local_state if local_state is not _EMPTY else {}
    )


def _deep_merge(d1: dict, d2: dict) -> dict:
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            _deep_merge(d1[k], v)
        else:
            d1[k] = v
    return d1


def _serialize_treedef(state: Any) -> str:
    _, treedef = jtu.tree_flatten(state)
    return base64.b64encode(pickle.dumps(treedef)).decode("ascii")


def _deserialize_treedef(encoded: str) -> jtu.PyTreeDef:
    return pickle.loads(base64.b64decode(encoded))


class OrbaxBackend(WorkerCheckpointBackend, backend_name="orbax"):
    def __init__(self, force_write: bool = False):
        self._handler = None
        self._force_write = force_write

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handler = None

    def _get_handler(self):
        if self._handler is None:
            import orbax.checkpoint as ocp

            self._handler = ocp.StandardCheckpointHandler()
        return self._handler

    async def save(
        self, state: Dict[str, Any], ckpt_dir: str, spmd_id: str, rank: int, step: int
    ) -> Dict[str, Any]:
        import orbax.checkpoint as ocp

        pytree_meta_encoded = _serialize_treedef(state)

        global_state, local_state = _split_hybrid_state(state)
        metadata = {"pytree_meta": pytree_meta_encoded}

        handler = self._get_handler()

        if local_state:
            import jax
            import numpy as np

            local_state_np = jax.tree_util.tree_map(
                lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, local_state
            )

            local_final_path = (
                Path(ckpt_dir) / str(step) / spmd_id / f"worker_{rank}_local_orbax"
            )
            local_tmp_path = local_final_path.with_suffix(".tmp")

            await asyncio.to_thread(
                local_tmp_path.parent.mkdir, parents=True, exist_ok=True
            )

            def _write_local():
                local_handler = ocp.PyTreeCheckpointHandler()
                ckptr = ocp.Checkpointer(local_handler)
                ckptr.save(local_tmp_path, args=ocp.args.PyTreeSave(local_state_np))
                local_tmp_path.rename(local_final_path)

            await asyncio.to_thread(_write_local)
            metadata["orbax_local_dir"] = str(local_final_path)

        if global_state:
            final_path = Path(ckpt_dir) / str(step) / spmd_id / "orbax_state"
            tmp_path = final_path.parent / "orbax_state.tmp"

            await asyncio.to_thread(tmp_path.mkdir, parents=True, exist_ok=True)

            commit_futures = (
                await handler.async_save(
                    epath.Path(tmp_path), args=ocp.args.StandardSave(global_state)
                )
                or []
            )

            await asyncio.gather(*[asyncio.to_thread(f.result) for f in commit_futures])

            from jax.experimental import multihost_utils

            await asyncio.to_thread(
                multihost_utils.sync_global_devices, f"orbax_save_{spmd_id}_{step}"
            )

            if rank == 0:
                await asyncio.to_thread(handler.finalize, epath.Path(tmp_path))

            metadata["orbax_tmp_path"] = str(tmp_path)
            metadata["orbax_final_path"] = str(final_path)
            metadata["ckpt_dir"] = str(final_path)

        return metadata

    async def load(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._get_handler()

        global_state = {}
        if "orbax_final_path" in metadata:
            ckpt_path = epath.Path(metadata["orbax_final_path"])
            global_state = await asyncio.to_thread(handler.restore, ckpt_path)

        local_state = {}
        if "orbax_local_dir" in metadata:
            import orbax.checkpoint as ocp

            def _read_local():
                local_handler = ocp.PyTreeCheckpointHandler()
                ckptr = ocp.Checkpointer(local_handler)
                return ckptr.restore(epath.Path(metadata["orbax_local_dir"]))

            local_state = await asyncio.to_thread(_read_local)

        flat_restored = _deep_merge(local_state, global_state)

        if "pytree_meta" in metadata:
            saved_treedef = _deserialize_treedef(metadata["pytree_meta"])
            restored_leaves = jtu.tree_leaves(flat_restored)
            return jtu.tree_unflatten(saved_treedef, restored_leaves)

        return flat_restored

    def finalize(self) -> None:
        pass

    def close(self) -> None:
        if self._handler:
            self._handler.close()
            self._handler = None
