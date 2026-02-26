import sys
import os
import numpy as np
import pytest
from unittest.mock import AsyncMock

sys.path.append(os.getcwd())

from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.common.types.configs import SPMDConfig, OperatorMeta
from fluxway.common.types.tensor import TransportTensor
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
    return DeviceMesh((1,), axis_names=("data",))


@pytest.fixture
def worker_config(mock_mesh):
    return _make_test_config(mock_mesh, spmd_id="test_spmd")


@pytest.mark.asyncio
async def test_worker_self_receive_tensor(worker_config):
    """Verify that _self_receive_tensor correctly calls receive_shards with unpacked payloads."""
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(worker_config, rank=0)

    # Mock receive_shards
    worker.receive_shards = AsyncMock()

    test_tensor = TransportTensor.from_tensor(np.array([1.0, 2.0]))

    await worker._self_receive_tensor(
        target_op="op2", name="grad", step=5, tensor=test_tensor
    )

    # Verify call signature matches the NEW receive_shards signature
    worker.receive_shards.assert_called_once()
    args, kwargs = worker.receive_shards.call_args
    assert args[0] == "op2"
    assert args[1] == "grad"
    assert args[2] == 5
    assert args[3] == 0  # self.rank
    assert isinstance(args[4], list)  # payloads
    assert isinstance(args[5], list)  # counts
    np.testing.assert_array_equal(args[4][0], test_tensor.flat_values)
    assert args[5][0] == 2  # count


@pytest.mark.asyncio
async def test_worker_self_receive_state(worker_config):
    """Verify that _self_receive_state correctly calls receive_states with wrapped arguments."""
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(worker_config, rank=0)

    # Mock receive_states
    worker.receive_states = AsyncMock()

    await worker._self_receive_state(target_op="update", name="lr", step=10, data=0.01)

    # Verify call signature matches the NEW receive_states signature
    worker.receive_states.assert_called_once()
    _, kwargs = worker.receive_states.call_args
    assert kwargs["op_name"] == "update"
    assert kwargs["step"] == 10
    assert kwargs["names"] == ["lr"]
    assert kwargs["values"] == [0.01]
