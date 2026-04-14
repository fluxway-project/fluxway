import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.common.types import SPMDConfig
from fluxway.spmd.driver.driver import SPMDDriver


@pytest.mark.asyncio
async def test_driver_stop_waits_for_worker_shutdown():
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="dense",
        config=SPMDConfig(mesh=SimpleNamespace(size=2)),
        placement_group=None,
        worker_bundle_indices=[1, 2],
        controller_handle=None,
        graph_topology={},
        checkpoint_backend=None,
    )

    worker0 = MagicMock()
    worker0.stop.remote = AsyncMock(return_value=None)
    worker0.set_driver_handle.remote = AsyncMock(return_value=None)
    worker1 = MagicMock()
    worker1.stop.remote = AsyncMock(return_value=None)
    worker1.set_driver_handle.remote = AsyncMock(return_value=None)

    driver._worker_handles = {0: worker0, 1: worker1}
    driver._stop_heartbeat = MagicMock()

    await driver.stop()

    driver._stop_heartbeat.assert_called_once_with()
    worker0.stop.remote.assert_awaited_once_with()
    worker1.stop.remote.assert_awaited_once_with()
    worker0.set_driver_handle.remote.assert_awaited_once_with(None)
    worker1.set_driver_handle.remote.assert_awaited_once_with(None)
    assert driver._worker_handles == {}


def test_stop_heartbeat_joins_background_thread():
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="dense",
        config=SPMDConfig(mesh=SimpleNamespace(size=1)),
        placement_group=None,
        worker_bundle_indices=[1],
        controller_handle=None,
        graph_topology={},
        checkpoint_backend=None,
    )

    join_mock = MagicMock()
    fake_thread = MagicMock(spec=threading.Thread)
    fake_thread.is_alive.return_value = True
    fake_thread.join = join_mock

    driver._heartbeat_running = True
    driver._heartbeat_thread = fake_thread

    driver._stop_heartbeat()

    assert driver._heartbeat_running is False
    join_mock.assert_called_once()
    assert driver._heartbeat_thread is None
