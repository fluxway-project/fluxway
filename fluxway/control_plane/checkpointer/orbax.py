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
from typing import Any

import jax.tree_util as jtu
import numpy as np
from etils import epath

from fluxway.control_plane.checkpointer.backend import WorkerCheckpointBackend


def _escape(s: str) -> str:
    """
    Escape special characters in path strings to ensure safe directory names.

    This is critical to prevent Key Collision. If a user defines a dict with a key
    that contains a slash (e.g., {"a/b": 1}), leaving it unescaped would cause it to
    collide with a nested dict (e.g., {"a": {"b": 1}}). By URL-encoding "/" into "%2F"
    and "%" into "%25", we mathematically guarantee that every unique tree topology
    maps to a unique, collision-free flat string path.
    """
    return s.replace("%", "%25").replace("/", "%2F")


def _get_path_str(path: tuple) -> str:
    """
    Convert a JAX KeyPath into a clean, collision-free string for flat dict keys.

    Why not use JAX's built-in `jax.tree_util.keystr(path, simple=True, separator='/')`?
    Because the official JAX function is unsafe for filesystem paths: it does NOT escape
    dictionary keys. It will incorrectly format both {"a/b": 1} and {"a": {"b": 1}} into
    the exact same string "a/b", causing catastrophic data overwriting during checkpointing.

    This function implements a safe serializer that enumerates standard JAX Keys (which is
    the standard JAX paradigm) and enforces our `_escape` protection on all node names,
    ensuring 100% data safety while maintaining human-readable folder structures in Orbax.
    """
    parts = []
    for p in path:
        if isinstance(p, jtu.DictKey):
            parts.append(_escape(str(p.key)))
        elif isinstance(p, jtu.SequenceKey):
            parts.append(str(p.idx))
        elif hasattr(p, "name"):
            parts.append(_escape(p.name))
        else:
            parts.append(_escape(str(p)))
    return "/".join(parts)


def _is_global_leaf(leaf: Any) -> bool:
    """Determine if a pytree leaf should be saved via Orbax StandardSave (global)."""
    import jax

    if not isinstance(leaf, jax.Array | jax.ShapeDtypeStruct):
        return False
    if hasattr(leaf, "sharding") and isinstance(
        leaf.sharding, jax.sharding.SingleDeviceSharding
    ):
        return False
    return True


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
        self, state: dict[str, Any], ckpt_dir: str, spmd_id: str, rank: int, step: int
    ) -> dict[str, Any]:
        import orbax.checkpoint as ocp
        import jax

        leaves_with_path, treedef = jtu.tree_flatten_with_path(state)

        global_dict = {}
        local_dict = {}
        for path, leaf in leaves_with_path:
            path_str = _get_path_str(path)
            if _is_global_leaf(leaf):
                global_dict[path_str] = leaf
            else:
                local_dict[path_str] = leaf

        pytree_meta_encoded = base64.b64encode(pickle.dumps(treedef)).decode("ascii")

        # The canonical SPMD-level checkpoint directory for this step.  All
        # artifact sub-paths are derived from it.  Setting ckpt_dir here
        # unconditionally ensures load_state_from_disk can locate the checkpoint
        # even when the checkpoint contains only local_state (no global arrays).
        spmd_step_dir = Path(ckpt_dir) / str(step) / spmd_id
        metadata: dict[str, Any] = {
            "pytree_meta": pytree_meta_encoded,
            "ckpt_dir": str(spmd_step_dir),
        }

        handler = self._get_handler()

        if local_dict:
            local_state_np = jax.tree_util.tree_map(
                lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, local_dict
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

                tmp_path = local_final_path.with_suffix(".tmp")
                # Clean up any leftover tmp dir from a previous interrupted write.
                if tmp_path.exists():
                    shutil.rmtree(tmp_path)
                tmp_path.mkdir(parents=True, exist_ok=True)

                leaves, local_treedef = jtu.tree_flatten(local_state_np)
                for i, leaf in enumerate(leaves):
                    leaf_path = tmp_path / f"leaf_{i}.npy"
                    if isinstance(leaf, np.ndarray):
                        np.save(str(leaf_path), leaf)
                    else:
                        # Non-array leaves (scalars, strings, etc.)
                        with open(str(leaf_path.with_suffix(".pkl")), "wb") as f:
                            pickle.dump(leaf, f)

                # Save the pytree structure so we can reconstruct on load.
                with open(str(tmp_path / "_local_treedef.pkl"), "wb") as f:
                    pickle.dump(local_treedef, f)

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

        if global_dict:
            final_path = spmd_step_dir / "orbax_state"
            await asyncio.to_thread(
                final_path.parent.mkdir, parents=True, exist_ok=True
            )

            # Save the abstract structure of global_dict so that restore
            # can use StandardRestore(target) for deterministic key types
            # and tree structure, instead of relying on Orbax to infer
            # them from directory names.
            abstract_global = jax.tree.map(
                lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), global_dict
            )
            metadata["global_abstract"] = base64.b64encode(
                pickle.dumps(abstract_global)
            ).decode("ascii")

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
                    args=ocp.args.StandardSave(global_dict),
                )

            await asyncio.to_thread(_write_global)
            metadata["orbax_final_path"] = str(final_path)

        return metadata

    async def load(self, metadata: dict[str, Any]) -> dict[str, Any]:
        handler = self._get_handler()

        global_dict = {}
        if "orbax_final_path" in metadata:
            import orbax.checkpoint as ocp

            ckpt_path = epath.Path(metadata["orbax_final_path"])

            abstract_target = None
            if "global_abstract" in metadata:
                try:
                    abstract_target = pickle.loads(
                        base64.b64decode(metadata["global_abstract"])
                    )
                except Exception:
                    abstract_target = None

            def _read_global():
                ckptr = ocp.Checkpointer(handler)
                if abstract_target is not None:
                    return ckptr.restore(
                        ckpt_path, args=ocp.args.StandardRestore(abstract_target)
                    )
                return ckptr.restore(ckpt_path)

            global_dict = await asyncio.to_thread(_read_global)

        local_dict = {}
        if "local_state_dir" in metadata:

            def _read_local():
                local_dir = Path(metadata["local_state_dir"])
                with open(str(local_dir / "_local_treedef.pkl"), "rb") as f:
                    local_treedef = pickle.load(f)
                leaves = []
                for i in range(local_treedef.num_leaves):
                    npy_path = local_dir / f"leaf_{i}.npy"
                    pkl_path = local_dir / f"leaf_{i}.pkl"
                    if npy_path.exists():
                        leaves.append(np.load(str(npy_path)))
                    elif pkl_path.exists():
                        with open(str(pkl_path), "rb") as f:
                            leaves.append(pickle.load(f))
                    else:
                        raise FileNotFoundError(f"Missing leaf {i} in {local_dir}")
                return local_treedef.unflatten(leaves)

            local_dict = await asyncio.to_thread(_read_local)

        # Merge the two flat dictionaries.
        # Since keys are unique paths, there are no conflicts or mixed keys.
        all_dict = {**global_dict, **local_dict}

        if "pytree_meta" in metadata:
            saved_treedef = pickle.loads(base64.b64decode(metadata["pytree_meta"]))

            # Reconstruct the original path sequence using a dummy unflatten
            dummy_leaves = list(range(saved_treedef.num_leaves))
            dummy_tree = jtu.tree_unflatten(saved_treedef, dummy_leaves)
            paths_and_indices, _ = jtu.tree_flatten_with_path(dummy_tree)

            restored_leaves = []
            for path, _ in paths_and_indices:
                path_str = _get_path_str(path)
                restored_leaves.append(all_dict[path_str])

            return jtu.tree_unflatten(saved_treedef, restored_leaves)

        return all_dict

    def finalize(self) -> None:
        pass

    def close(self) -> None:
        if self._handler:
            self._handler.close()
            self._handler = None
