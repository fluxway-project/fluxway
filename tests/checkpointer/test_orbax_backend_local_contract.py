import importlib.util
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest


class _FakeTreeDef:
    """Minimal pytree treedef mock supporting pickle + unflatten."""

    def __init__(self, keys):
        self._keys = keys

    @property
    def num_leaves(self):
        return len(self._keys)

    def unflatten(self, leaves):
        return dict(zip(self._keys, leaves))


def _install_fake_orbax_runtime(monkeypatch):
    jax_module = types.ModuleType("jax")

    class _Array:
        def __init__(self, sharding=None):
            self.sharding = sharding

    class _ShapeDtypeStruct:
        def __init__(self, sharding=None):
            self.sharding = sharding

    class _SingleDeviceSharding:
        pass

    class _NamedSharding:
        pass

    tree_util_module = types.ModuleType("jax.tree_util")

    def _tree_flatten(state):
        if isinstance(state, dict):
            keys = sorted(state.keys())
            return [state[k] for k in keys], _FakeTreeDef(keys)
        return [state], _FakeTreeDef([])

    def _tree_map(fn, value):
        if isinstance(value, dict):
            return {k: _tree_map(fn, v) for k, v in value.items()}
        return fn(value)

    tree_util_module.tree_flatten = _tree_flatten
    tree_util_module.tree_map = _tree_map
    tree_util_module.PyTreeDef = tuple
    tree_util_module.tree_leaves = lambda x: _tree_flatten(x)[0]

    jax_module.Array = _Array
    jax_module.ShapeDtypeStruct = _ShapeDtypeStruct
    jax_module.tree_util = tree_util_module
    jax_module.sharding = types.SimpleNamespace(
        SingleDeviceSharding=_SingleDeviceSharding,
        NamedSharding=_NamedSharding,
    )
    jax_module.process_index = lambda: 0

    sync_global_devices_calls = []

    def _unexpected_sync_global_devices(*args, **kwargs):
        sync_global_devices_calls.append((args, kwargs))
        raise AssertionError(
            "global Orbax save should use Checkpointer synchronization, not an extra "
            "manual multihost_utils.sync_global_devices call"
        )

    multihost_utils_module = types.ModuleType("jax.experimental.multihost_utils")
    multihost_utils_module.sync_global_devices = _unexpected_sync_global_devices
    experimental_module = types.ModuleType("jax.experimental")
    experimental_module.multihost_utils = multihost_utils_module
    jax_module.experimental = experimental_module

    etils_module = types.ModuleType("etils")
    epath_module = types.ModuleType("etils.epath")
    epath_module.Path = Path
    etils_module.epath = epath_module

    save_calls = []

    class _FakeStandardCheckpointHandler:
        async def async_save(self, *args, **kwargs):
            raise AssertionError(
                "global Orbax save should not call StandardCheckpointHandler.async_save "
                "directly"
            )

    class _FakePyTreeCheckpointHandler:
        pass

    checkpointer_save_calls = []

    class _FakeCheckpointer:
        def __init__(self, handler, *, multiprocessing_options=None):
            self.handler = handler
            self.multiprocessing_options = multiprocessing_options

        def save(self, directory, args=None, force=False):
            checkpointer_save_calls.append(
                {
                    "directory": Path(directory),
                    "args": args,
                    "force": force,
                    "mp_options": self.multiprocessing_options,
                }
            )
            save_calls.append(Path(directory))
            Path(directory).mkdir(parents=True, exist_ok=force)

        def restore(self, directory):
            return {"restored_from": str(directory)}

    ocp_module = types.ModuleType("orbax.checkpoint")
    ocp_module.StandardCheckpointHandler = _FakeStandardCheckpointHandler
    ocp_module.PyTreeCheckpointHandler = _FakePyTreeCheckpointHandler
    ocp_module.Checkpointer = _FakeCheckpointer
    ocp_module.args = types.SimpleNamespace(
        PyTreeSave=lambda item: item,
        StandardSave=lambda item: item,
    )
    ocp_module.options = types.SimpleNamespace(
        MultiprocessingOptions=lambda **kw: types.SimpleNamespace(**kw)
    )

    monkeypatch.setitem(sys.modules, "jax", jax_module)
    monkeypatch.setitem(sys.modules, "jax.tree_util", tree_util_module)
    monkeypatch.setitem(sys.modules, "jax.experimental", experimental_module)
    monkeypatch.setitem(
        sys.modules, "jax.experimental.multihost_utils", multihost_utils_module
    )
    monkeypatch.setitem(sys.modules, "etils", etils_module)
    monkeypatch.setitem(sys.modules, "etils.epath", epath_module)
    monkeypatch.setitem(sys.modules, "orbax", types.ModuleType("orbax"))
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", ocp_module)
    monkeypatch.setitem(sys.modules, "fluxway", types.ModuleType("fluxway"))
    monkeypatch.setitem(
        sys.modules, "fluxway.control_plane", types.ModuleType("fluxway.control_plane")
    )
    monkeypatch.setitem(
        sys.modules,
        "fluxway.control_plane.checkpointer",
        types.ModuleType("fluxway.control_plane.checkpointer"),
    )

    backend_path = (
        Path(__file__).resolve().parents[2]
        / "fluxway"
        / "control_plane"
        / "checkpointer"
        / "backend.py"
    )
    backend_spec = importlib.util.spec_from_file_location(
        "fluxway.control_plane.checkpointer.backend", backend_path
    )
    backend_module = importlib.util.module_from_spec(backend_spec)
    assert backend_spec.loader is not None
    backend_spec.loader.exec_module(backend_module)
    monkeypatch.setitem(
        sys.modules, "fluxway.control_plane.checkpointer.backend", backend_module
    )

    orbax_path = (
        Path(__file__).resolve().parents[2]
        / "fluxway"
        / "control_plane"
        / "checkpointer"
        / "orbax.py"
    )
    orbax_spec = importlib.util.spec_from_file_location(
        "fluxway.control_plane.checkpointer.orbax", orbax_path
    )
    orbax_module = importlib.util.module_from_spec(orbax_spec)
    assert orbax_spec.loader is not None
    orbax_spec.loader.exec_module(orbax_module)

    return orbax_module, save_calls, checkpointer_save_calls, sync_global_devices_calls


@pytest.mark.asyncio
async def test_local_save_bypasses_orbax_checkpointer(monkeypatch, tmp_path):
    """Local state must be saved via direct numpy serialization, NOT via
    Orbax Checkpointer, because Orbax coordinates all JAX distributed
    processes and only one process writes non-sharded numpy data."""
    orbax_module, save_calls, checkpointer_save_calls, _ = _install_fake_orbax_runtime(
        monkeypatch
    )

    backend = orbax_module.OrbaxBackend()
    metadata = await backend.save(
        {"operator_states": {"DenseLayers": {"step": np.array(2)}}},
        str(tmp_path),
        "DenseLayers",
        rank=4,
        step=2,
    )

    expected_dir = tmp_path / "2" / "DenseLayers" / "worker_4_local_orbax"
    # Must NOT use Orbax Checkpointer for local state.
    assert checkpointer_save_calls == [], (
        "Local state should bypass Orbax Checkpointer entirely"
    )
    # But the metadata and final directory must be correct.
    assert metadata["orbax_local_dir"] == str(expected_dir)
    assert expected_dir.exists()
    # Verify numpy files were written.
    assert (expected_dir / "_treedef.pkl").exists()
    assert any(expected_dir.glob("leaf_*.npy")) or any(expected_dir.glob("leaf_*.pkl"))


@pytest.mark.asyncio
async def test_local_save_multiple_ranks_independent(monkeypatch, tmp_path):
    """Each rank must write its own local state independently.
    No Orbax barriers, no cross-rank coordination."""
    orbax_module, _, checkpointer_save_calls, _ = _install_fake_orbax_runtime(
        monkeypatch
    )
    jax_module = sys.modules["jax"]

    backend = orbax_module.OrbaxBackend(force_write=True)
    state = {
        "optimizer": np.array([1.0, 2.0, 3.0], dtype=np.float32)
    }

    await backend.save(state, str(tmp_path), "DenseLayers", rank=0, step=9)
    await backend.save(state, str(tmp_path), "DenseLayers", rank=5, step=9)

    # Local saves must NOT touch Orbax Checkpointer.
    assert checkpointer_save_calls == [], (
        "Local saves must bypass Orbax Checkpointer"
    )
    # Both ranks must have their own directories with actual data.
    dir_rank0 = tmp_path / "9" / "DenseLayers" / "worker_0_local_orbax"
    dir_rank5 = tmp_path / "9" / "DenseLayers" / "worker_5_local_orbax"
    assert dir_rank0.exists()
    assert dir_rank5.exists()
    assert (dir_rank0 / "_treedef.pkl").exists()
    assert (dir_rank5 / "_treedef.pkl").exists()


@pytest.mark.asyncio
async def test_global_save_uses_checkpointer_atomic_save(monkeypatch, tmp_path):
    orbax_module, _, checkpointer_save_calls, sync_global_devices_calls = (
        _install_fake_orbax_runtime(monkeypatch)
    )
    jax_module = sys.modules["jax"]

    backend = orbax_module.OrbaxBackend(force_write=True)
    metadata = await backend.save(
        {
            "operator_states": {
                "DenseLayers": {
                    "weights": jax_module.Array(jax_module.sharding.NamedSharding())
                }
            }
        },
        str(tmp_path),
        "DenseLayers",
        rank=2,
        step=7,
    )

    expected_dir = tmp_path / "7" / "DenseLayers" / "orbax_state"
    assert len(checkpointer_save_calls) == 1
    assert checkpointer_save_calls[0]["directory"] == expected_dir
    assert checkpointer_save_calls[0]["force"] is True
    assert "operator_states" in checkpointer_save_calls[0]["args"]
    assert metadata["orbax_final_path"] == str(expected_dir)
    assert "orbax_tmp_path" not in metadata
    assert expected_dir.exists()
    assert sync_global_devices_calls == []
    # The Checkpointer must receive a step-scoped barrier_sync_key_prefix so
    # that repeated saves do not reuse the same JAX distributed barrier key.
    mp_opts = checkpointer_save_calls[0]["mp_options"]
    assert mp_opts is not None, "multiprocessing_options must be passed to Checkpointer"
    assert "DenseLayers_7" in mp_opts.barrier_sync_key_prefix, (
        f"barrier prefix should embed spmd_id and step; got {mp_opts.barrier_sync_key_prefix!r}"
    )


@pytest.mark.asyncio
async def test_global_save_uses_distinct_barrier_prefix_per_step(monkeypatch, tmp_path):
    """Regression test for the distributed barrier-key reuse bug.

    Orbax's Checkpointer.save() calls sync_global_processes() with fixed keys
    like 'Checkpointer:save_start'. JAX's distributed barrier requires that each
    key is used exactly once — reusing the same key on a second checkpoint call
    causes a 300-second deadlock in multi-host environments.

    The fix is to pass a step-scoped ``barrier_sync_key_prefix`` to
    MultiprocessingOptions so that each checkpoint's barrier keys are unique,
    e.g. 'DenseLayers_50_Checkpointer:save_start' vs
         'DenseLayers_100_Checkpointer:save_start'.
    """
    orbax_module, _, checkpointer_save_calls, _ = _install_fake_orbax_runtime(monkeypatch)
    jax_module = sys.modules["jax"]

    state = {
        "operator_states": {
            "DenseLayers": {
                "weights": jax_module.Array(jax_module.sharding.NamedSharding())
            }
        }
    }
    backend = orbax_module.OrbaxBackend(force_write=True)

    # First checkpoint
    await backend.save(state, str(tmp_path), "DenseLayers", rank=0, step=50)
    # Second checkpoint (different step)
    await backend.save(state, str(tmp_path), "DenseLayers", rank=0, step=100)

    assert len(checkpointer_save_calls) == 2
    prefix_50 = checkpointer_save_calls[0]["mp_options"].barrier_sync_key_prefix
    prefix_100 = checkpointer_save_calls[1]["mp_options"].barrier_sync_key_prefix

    assert prefix_50 != prefix_100, (
        "Each checkpoint step must use a distinct barrier_sync_key_prefix. "
        f"Both steps produced the same prefix: {prefix_50!r}"
    )
    assert "50" in prefix_50
    assert "100" in prefix_100


@pytest.mark.asyncio
async def test_ckpt_dir_always_points_to_spmd_level_dir(monkeypatch, tmp_path):
    """ckpt_dir must be ckpt_dir/step/spmd_id, not the orbax_state subdirectory.

    Operators receive the SPMD-level directory during the save phase; the load
    path must expose the same directory so operators can locate their own
    artifacts consistently.
    """
    orbax_module, _, _, _ = _install_fake_orbax_runtime(monkeypatch)
    jax_module = sys.modules["jax"]

    state = {
        "weights": jax_module.Array(jax_module.sharding.NamedSharding())
    }
    backend = orbax_module.OrbaxBackend(force_write=True)
    metadata = await backend.save(
        state, str(tmp_path), "DenseLayers", rank=0, step=5
    )

    expected_spmd_dir = tmp_path / "5" / "DenseLayers"
    assert metadata["ckpt_dir"] == str(expected_spmd_dir), (
        f"ckpt_dir should be the SPMD-level directory {expected_spmd_dir}, "
        f"not the orbax_state subdirectory; got {metadata['ckpt_dir']!r}"
    )
    # orbax_state lives one level below
    assert metadata["orbax_final_path"] == str(expected_spmd_dir / "orbax_state")


@pytest.mark.asyncio
async def test_ckpt_dir_set_for_local_only_checkpoint(monkeypatch, tmp_path):
    """ckpt_dir must be present even when the checkpoint contains only local_state.

    If ckpt_dir is absent, load_state_from_disk silently skips restoration,
    leading to a silent state reset that is extremely hard to diagnose.
    """
    orbax_module, _, checkpointer_save_calls, _ = _install_fake_orbax_runtime(
        monkeypatch
    )
    jax_module = sys.modules["jax"]

    # SingleDeviceSharding → classified as local_state by _split_hybrid_state
    state = {
        "optimizer": np.array([1.0], dtype=np.float32)
    }
    backend = orbax_module.OrbaxBackend(force_write=True)
    metadata = await backend.save(
        state, str(tmp_path), "DenseLayers", rank=1, step=3
    )

    expected_spmd_dir = tmp_path / "3" / "DenseLayers"
    assert "ckpt_dir" in metadata, (
        "ckpt_dir must be set for local-only checkpoints so that "
        "load_state_from_disk does not silently skip restoration"
    )
    assert metadata["ckpt_dir"] == str(expected_spmd_dir)
    assert "orbax_final_path" not in metadata, (
        "A local-only checkpoint should not produce orbax_final_path"
    )
    assert "orbax_local_dir" in metadata
