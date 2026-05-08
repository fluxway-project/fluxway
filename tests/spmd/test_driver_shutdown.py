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
    driver._stop_heartbeat = AsyncMock()

    await driver.stop()

    driver._stop_heartbeat.assert_called_once_with()
    worker0.stop.remote.assert_awaited_once_with()
    worker1.stop.remote.assert_awaited_once_with()
    worker0.set_driver_handle.remote.assert_awaited_once_with(None)
    worker1.set_driver_handle.remote.assert_awaited_once_with(None)
    assert driver._worker_handles == {}


@pytest.mark.asyncio
async def test_stop_heartbeat_joins_background_thread():
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

    await driver._stop_heartbeat()

    assert driver._heartbeat_running is False
    join_mock.assert_called_once()
    assert driver._heartbeat_thread is None


def test_driver_ignores_worker_callbacks_while_stopping():
    driver = SPMDDriver.__ray_actor_class__(
        spmd_id="dense",
        config=SPMDConfig(mesh=SimpleNamespace(size=1)),
        placement_group=None,
        worker_bundle_indices=[1],
        controller_handle=MagicMock(),
        graph_topology={},
        checkpoint_backend=None,
    )

    heartbeat = MagicMock()
    heartbeat.error_message = None
    heartbeat.telemetry = None
    heartbeat.mailbox_status = None

    driver._stopping = True
    driver.spmd_synchronizer.report_worker_op_completed = MagicMock()

    driver.receive_worker_heartbeat(0, heartbeat)
    driver.report_worker_op_completed(0, "step", 1)
    driver.routing_control_argument_to_controller("target", "op", ["x"], 1, [1])

    assert driver._worker_heartbeats == {}
    assert driver._worker_heartbeat_times == {}
    driver.spmd_synchronizer.report_worker_op_completed.assert_not_called()
    driver.controller_handle.receive_control_argument.remote.assert_not_called()
