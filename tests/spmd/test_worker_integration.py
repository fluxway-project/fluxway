import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, AsyncMock

sys.path.append(os.getcwd())

from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.common.types.configs import SPMDConfig, ReaderConfig, NodeType
from fluxway.common.types.tensor import TransportTensor
from fluxway.spmd.worker.mailbox import OpType
from fluxway.common.mesh import DeviceMesh


@pytest.fixture
def mock_mesh():
    return DeviceMesh((1,), axis_names=("data",))


@pytest.fixture
def worker_config(mock_mesh):
    def dummy_fn(state, batch):
        return state

    return SPMDConfig(mesh=mock_mesh, user_forward_fn=dummy_fn, spmd_id="test_spmd")


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
        target_op=OpType.BACKWARD, name="grad", step=5, tensor=test_tensor
    )

    # Verify call signature matches the NEW receive_shards signature
    worker.receive_shards.assert_called_once()
    args, kwargs = worker.receive_shards.call_args
    assert args[0] == OpType.BACKWARD
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

    await worker._self_receive_state(
        target_op=OpType.UPDATE, name="lr", step=10, data=0.01
    )

    # Verify call signature matches the NEW receive_states signature
    worker.receive_states.assert_called_once()
    _, kwargs = worker.receive_states.call_args
    assert kwargs["op_type"] == OpType.UPDATE
    assert kwargs["step"] == 10
    assert kwargs["names"] == ["lr"]
    assert kwargs["values"] == [0.01]


@pytest.mark.asyncio
async def test_worker_reader_send_batch(mock_mesh):
    """Verify that _send_reader_batch correctly calls receive_shards on a mocked handler."""
    reader_config = ReaderConfig(
        src_dir="/tmp", file_pattern="*", data_type="dummy", tensor_name="input_data"
    )

    config = SPMDConfig(
        mesh=mock_mesh, node_type=NodeType.READER, reader_config=reader_config
    )

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    # Reader worker requires node_type and reader_config in __init__
    worker = cls(config, rank=0, node_type=NodeType.READER, reader_config=reader_config)

    # Mock successor handlers
    mock_handler = MagicMock()
    worker._successor_handlers = {"downstream_spmd": {0: mock_handler}}

    batch_data = np.array([1, 2, 3])

    await worker._send_reader_batch(batch_data, step=0)

    # Verify RPC call to receive_shards
    mock_handler.receive_shards.remote.assert_called_once()
    args, _ = mock_handler.receive_shards.remote.call_args

    assert args[0] == OpType.FORWARD  # Default target_op for reader
    assert args[1] == "input_data"
    assert args[2] == 0  # step
    assert args[3] == 0  # worker.rank
    assert isinstance(args[4], list)  # payloads
    assert isinstance(args[5], list)  # counts
    np.testing.assert_array_equal(args[4][0], batch_data)
    assert args[5][0] == 3
