from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.common.types.drain import DrainStrategy
from fluxway.control_plane.controller import ControllerActor


class MockDrainStrategy(DrainStrategy):
    def get_initial_drain_target(self, baseline: int = 0):
        return baseline + 2

    def get_next_drain_target(self, current_completed_step: int):
        return current_completed_step + 2


@pytest.fixture
def mock_spmd_manager():
    mgr = MagicMock()
    mgr.get_all_spmd_ids.return_value = ["op1", "op2"]

    mock_driver1 = AsyncMock()
    mock_driver2 = AsyncMock()

    def get_driver(spmd_id):
        if spmd_id == "op1":
            return mock_driver1
        return mock_driver2

    mgr.get_driver_handle = get_driver
    mgr.mock_driver1 = mock_driver1
    mgr.mock_driver2 = mock_driver2
    return mgr


@pytest.fixture
def controller(mock_spmd_manager):
    # Setup partial graph stub
    graph_stub = MagicMock()
    graph_stub.nodes.keys.return_value = ["op1", "op2"]

    ctrl = ControllerActor.__ray_actor_class__(graph=graph_stub)
    ctrl._spmd_manager = mock_spmd_manager
    ctrl._graph = graph_stub

    # inject mock strategy and checkpointer
    ctrl._drain_strategy = MockDrainStrategy()
    ctrl._checkpointer = AsyncMock()

    return ctrl


@pytest.mark.asyncio
async def test_controller_orchestrates_drain(controller, mock_spmd_manager):
    controller._last_requested_drain_target = 0
    controller._drained_drivers = set()

    # Initial drain issued
    await controller._issue_initial_drain()
    mock_spmd_manager.mock_driver1.request_drain.remote.assert_called_with(2)
    mock_spmd_manager.mock_driver2.request_drain.remote.assert_called_with(2)
    assert controller._last_requested_drain_target == 2

    # Driver 1 reports drained
    await controller.report_driver_drained("op1")
    controller._checkpointer.save.assert_not_called()  # op2 not yet drained

    # Driver 2 reports drained => Triggers ALL_DRIVERS_DRAINED => Checkpoint
    await controller.report_driver_drained("op2")

    # We call this manually simulating the FSM dispatching it
    await controller._on_all_drivers_drained()
    await controller._checkpoint_task

    controller._checkpointer.save.assert_called_with(2)
    mock_spmd_manager.mock_driver1.resume.remote.assert_called()
    mock_spmd_manager.mock_driver2.resume.remote.assert_called()

    # Verifies next target issued
    mock_spmd_manager.mock_driver1.request_drain.remote.assert_called_with(4)
