import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _install_fake_orbax_runtime(monkeypatch):
    jax_module = types.ModuleType("jax")

    class _Array:
        pass

    class _ShapeDtypeStruct:
        pass

    class _SingleDeviceSharding:
        pass

    tree_util_module = types.ModuleType("jax.tree_util")

    def _tree_flatten(state):
        return [], ("fake_treedef", tuple(state.keys()))

    def _tree_map(fn, value):
        if isinstance(value, dict):
            return {k: _tree_map(fn, v) for k, v in value.items()}
        return fn(value)

    tree_util_module.tree_flatten = _tree_flatten
    tree_util_module.tree_map = _tree_map
    tree_util_module.PyTreeDef = tuple
    tree_util_module.register_pytree_node = lambda *args, **kwargs: None

    jax_module.Array = _Array
    jax_module.ShapeDtypeStruct = _ShapeDtypeStruct
    jax_module.tree_util = tree_util_module
    jax_module.sharding = types.SimpleNamespace(
        SingleDeviceSharding=_SingleDeviceSharding
    )

    etils_module = types.ModuleType("etils")
    epath_module = types.ModuleType("etils.epath")
    epath_module.Path = Path
    etils_module.epath = epath_module

    save_calls = []

    class _FakeStandardCheckpointHandler:
        pass

    class _FakePyTreeCheckpointHandler:
        pass

    class _FakeCheckpointer:
        def __init__(self, handler):
            self.handler = handler

        def save(self, directory, args=None, force=False):
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

    monkeypatch.setitem(sys.modules, "jax", jax_module)
    monkeypatch.setitem(sys.modules, "jax.tree_util", tree_util_module)
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

    return orbax_module, save_calls


@pytest.mark.asyncio
async def test_local_save_uses_final_path_without_manual_rename(monkeypatch, tmp_path):
    orbax_module, save_calls = _install_fake_orbax_runtime(monkeypatch)
    rename_calls = []

    def _unexpected_rename(self, target):
        rename_calls.append((Path(self), Path(target)))
        raise AssertionError("local Orbax save should not require a manual rename")

    monkeypatch.setattr(Path, "rename", _unexpected_rename)

    backend = orbax_module.OrbaxBackend()
    metadata = await backend.save(
        {"operator_states": {"DenseLayers": {"step": 2}}},
        str(tmp_path),
        "DenseLayers",
        rank=4,
        step=2,
    )

    expected_dir = tmp_path / "2" / "DenseLayers" / "worker_4_local_orbax"
    assert save_calls == [expected_dir]
    assert metadata["orbax_local_dir"] == str(expected_dir)
    assert expected_dir.exists()
    assert rename_calls == []
