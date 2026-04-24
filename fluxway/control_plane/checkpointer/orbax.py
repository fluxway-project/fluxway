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
import numpy as np
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

        # The canonical SPMD-level checkpoint directory for this step.  All
        # artifact sub-paths are derived from it.  Setting ckpt_dir here
        # unconditionally ensures load_state_from_disk can locate the checkpoint
        # even when the checkpoint contains only local_state (no global arrays).
        spmd_step_dir = Path(ckpt_dir) / str(step) / spmd_id
        metadata: Dict[str, Any] = {
            "pytree_meta": pytree_meta_encoded,
            "ckpt_dir": str(spmd_step_dir),
        }

        handler = self._get_handler()

        if local_state:
            import jax

            local_state_np = jax.tree_util.tree_map(
                lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, local_state
            )

            local_final_path = spmd_step_dir / f"worker_{rank}_local_state"

            def _write_local():
                # Bypass Orbax Checkpointer entirely for per-rank local state.
                # Orbax's Checkpointer always coordinates across all JAX
                # distributed processes via barriers, and for non-sharded numpy
                # arrays only one process actually writes data regardless of
                # primary_host / active_processes settings.  Direct numpy
                # serialization lets each rank write independently without any
                # cross-process coordination.
                import shutil

                leaves, treedef = jtu.tree_flatten(local_state_np)
                tmp_path = local_final_path.with_suffix(".tmp")
                # Clean up any leftover tmp dir from a previous interrupted write.
                if tmp_path.exists():
                    shutil.rmtree(tmp_path)
                tmp_path.mkdir(parents=True, exist_ok=True)
                for i, leaf in enumerate(leaves):
                    leaf_path = tmp_path / f"leaf_{i}.npy"
                    if isinstance(leaf, np.ndarray):
                        np.save(str(leaf_path), leaf)
                    else:
                        # Non-array leaves (scalars, strings, etc.)
                        with open(str(leaf_path.with_suffix(".pkl")), "wb") as f:
                            pickle.dump(leaf, f)
                # Save the pytree structure so we can reconstruct on load.
                with open(str(tmp_path / "_treedef.pkl"), "wb") as f:
                    pickle.dump(treedef, f)
                # Atomic rename: tmp → final.
                if local_final_path.exists():
                    if not self._force_write:
                        raise FileExistsError(
                            f"Local checkpoint already exists at {local_final_path}. "
                            "Set force_write=True in CheckpointConfig to overwrite."
                        )
                    shutil.rmtree(local_final_path)
                # Use shutil.move instead of Path.rename to handle the case
                # where tmp and final paths reside on different filesystems.
                shutil.move(str(tmp_path), str(local_final_path))

            await asyncio.to_thread(_write_local)
            metadata["local_state_dir"] = str(local_final_path)

        if global_state:
            final_path = spmd_step_dir / "orbax_state"
            await asyncio.to_thread(
                final_path.parent.mkdir, parents=True, exist_ok=True
            )

            def _write_global():
                # The global phase uses its own step-scoped prefix so it does
                # not collide with either another checkpoint step or the local
                # save phase above.
                mp_options = ocp.options.MultiprocessingOptions(
                    barrier_sync_key_prefix=f"{spmd_id}_{step}_global"
                )
                ckptr = ocp.Checkpointer(handler, multiprocessing_options=mp_options)
                ckptr.save(
                    epath.Path(final_path),
                    force=self._force_write,
                    args=ocp.args.StandardSave(global_state),
                )

            await asyncio.to_thread(_write_global)
            metadata["orbax_final_path"] = str(final_path)

        return metadata

    async def load(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        handler = self._get_handler()

        global_state = {}
        if "orbax_final_path" in metadata:
            import orbax.checkpoint as ocp

            ckpt_path = epath.Path(metadata["orbax_final_path"])

            def _read_global():
                ckptr = ocp.Checkpointer(handler)
                return ckptr.restore(ckpt_path)

            global_state = await asyncio.to_thread(_read_global)

        local_state = {}
        if "local_state_dir" in metadata:

            def _read_local():
                local_dir = Path(metadata["local_state_dir"])
                with open(str(local_dir / "_treedef.pkl"), "rb") as f:
                    treedef = pickle.load(f)
                leaves = []
                for i in range(treedef.num_leaves):
                    npy_path = local_dir / f"leaf_{i}.npy"
                    pkl_path = local_dir / f"leaf_{i}.pkl"
                    if npy_path.exists():
                        leaves.append(np.load(str(npy_path)))
                    elif pkl_path.exists():
                        with open(str(pkl_path), "rb") as f:
                            leaves.append(pickle.load(f))
                    else:
                        raise FileNotFoundError(f"Missing leaf {i} in {local_dir}")
                return treedef.unflatten(leaves)

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
