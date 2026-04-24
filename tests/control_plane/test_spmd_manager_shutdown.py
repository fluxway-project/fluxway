from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fluxway.control_plane.spmd_manager import manager as manager_module
from fluxway.control_plane.spmd_manager.manager import SPMDHandle, SPMDManager


class _AsyncRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._result()

    async def _result(self):
        return None


@pytest.mark.asyncio
async def test_stop_spmd_releases_driver_without_force_kill(monkeypatch):
    manager = SPMDManager()
    handle = SPMDHandle("dense")
    handle.set_initialized(True)

    stop_method = _AsyncRemoteMethod()
    driver_handle = SimpleNamespace(stop=stop_method)
    handle.set_driver_handle(driver_handle)
    manager.register_spmd("dense", handle)

    kill_mock = MagicMock()
    monkeypatch.setattr(manager_module, "RAY_AVAILABLE", True)
    monkeypatch.setattr(manager_module, "ray", SimpleNamespace(kill=kill_mock))

    await manager.stop_spmd("dense", save_checkpoint=False)

    assert stop_method.calls == [((False,), {})]
    kill_mock.assert_not_called()
    assert handle.driver_handle is None
    assert handle.is_initialized is False


@pytest.mark.asyncio
async def test_stop_spmd_can_preserve_driver_reference_during_terminal_cleanup(
    monkeypatch,
):
    manager = SPMDManager()
    handle = SPMDHandle("dense")
    handle.set_initialized(True)

    stop_method = _AsyncRemoteMethod()
    driver_handle = SimpleNamespace(stop=stop_method)
    handle.set_driver_handle(driver_handle)
    manager.register_spmd("dense", handle)

    kill_mock = MagicMock()
    monkeypatch.setattr(manager_module, "RAY_AVAILABLE", True)
    monkeypatch.setattr(manager_module, "ray", SimpleNamespace(kill=kill_mock))

    await manager.stop_spmd("dense", save_checkpoint=False, release_driver_handle=False)

    assert stop_method.calls == [((False,), {})]
    kill_mock.assert_not_called()
    assert handle.driver_handle is driver_handle
    assert handle.is_initialized is False
