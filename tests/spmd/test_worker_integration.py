import sys
import os
import numpy as np
import pytest
from unittest.mock import AsyncMock

sys.path.append(os.getcwd())

from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.common.types.configs import SPMDConfig, OperatorMeta
from fluxway.common.mesh import DeviceMesh


def _make_test_config(mesh: DeviceMesh, spmd_id: str = "") -> SPMDConfig:
    """Create a minimal SPMDConfig with a no-op 'test_op' operator registered."""
    meta = OperatorMeta(name="test_op", input_arguments=None, output_arguments=None)

    def dummy_fn():
        pass

    return SPMDConfig(
        mesh=mesh,
        spmd_id=spmd_id,
        operators={"test_op": dummy_fn},
        operator_metas={"test_op": meta},
    )


@pytest.fixture
def mock_mesh():
    return DeviceMesh(axis_names=("data",), device_shape=(1,))


@pytest.fixture
def worker_config(mock_mesh):
    return _make_test_config(mock_mesh, spmd_id="test_spmd")


@pytest.mark.asyncio
async def test_worker_receive_shards_dispatches_to_mailbox(worker_config):
    """Verify that receive_shards correctly dispatches to the mailbox via deposit_shards."""
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(worker_config, rank=0, spmd_id="test_spmd")

    from unittest.mock import MagicMock, AsyncMock

    worker._mailbox.deposit_shards = AsyncMock(return_value=0)
    worker._op_manager.get_operators_for_tuuid = MagicMock(return_value=["test_op"])
    worker._op_manager.get_sender_world_size = MagicMock(return_value=1)
    worker._ensure_step_declared = MagicMock()

    test_data = np.array([1.0, 2.0])
    await worker.receive_shards(
        tuuids=[42],
        step=0,
        sender_id="src_spmd",
        sender_rank=0,
        payloads=[test_data],
        counts=[2],
    )

    worker._mailbox.deposit_shards.assert_called_once()
    call_args = worker._mailbox.deposit_shards.call_args
    assert call_args[0][0] == "test_op"
    assert call_args[0][1] == [42]
    assert call_args[0][2] == 0
    assert call_args[0][3] == "src_spmd"
    assert call_args[0][4] == 0  # sender_rank


@pytest.mark.asyncio
async def test_worker_receive_control_dispatches_to_mailbox(worker_config):
    """Verify that receive_control correctly calls deposit_states on the mailbox."""
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(worker_config, rank=0, spmd_id="test_spmd")

    worker._mailbox.deposit_states = AsyncMock()
    worker._ensure_step_declared = lambda *a, **kw: None

    await worker.receive_control(op_name="test_op", step=3, names=["lr"], values=[0.01])

    worker._mailbox.deposit_states.assert_called_once()
    _, kwargs = worker._mailbox.deposit_states.call_args
    assert kwargs.get("names") == ["lr"] or worker._mailbox.deposit_states.call_args[0][
        2:4
    ] == (["lr"], [0.01])
