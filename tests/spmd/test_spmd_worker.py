import sys
import os


# Add project root to path
sys.path.append(os.getcwd())

from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.spmd.worker.mailbox import WorkerMailbox, OpType
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.common.types.data_types import SPMDConfig
from fluxway.common.mesh import DeviceMesh
import pytest


# Mock user functions
def forward_fn(state, batch):
    return state


def backward_fn(state, grad):
    return state


def update_fn(state, grad):
    return state


def test_layout_mesh():
    """Verify DeviceMesh works with user-defined axis names."""
    print("Testing DeviceMesh with user-defined axes...")

    mesh = DeviceMesh((4, 2), axis_names=("data", "model"))

    # Check properties
    assert mesh.size == 8
    assert mesh.ndim == 2
    assert mesh.axis_names == ("data", "model")
    assert mesh.shape == (4, 2)

    # Check axis lookups
    assert mesh.get_axis_size("data") == 4
    assert mesh.get_axis_size("model") == 2
    assert mesh.axis_names.index("data") == 0
    assert mesh.axis_names.index("model") == 1

    # Check coordinate lookups
    coord = mesh.get_coordinate(5)  # rank 5 in 4x2 mesh
    assert coord == (2, 1)  # row 2, col 1

    # Check reverse lookup
    assert mesh.get_rank((2, 1)) == 5

    print("DeviceMesh with user-defined axes successful.")


@pytest.mark.asyncio
async def test_mailbox_named_storage():
    """Verify WorkerMailbox works with named tensors/states."""
    print("Testing WorkerMailbox named storage...")

    mailbox = WorkerMailbox()

    # Test state storage with names
    await mailbox.put_state(OpType.FORWARD, "id", step=1, data={"value": 42})
    await mailbox.put_state(OpType.FORWARD, "config", step=1, data={"lr": 0.01})

    assert await mailbox.has_state(OpType.FORWARD, "id", step=1)
    assert await mailbox.has_state(OpType.FORWARD, "config", step=1)
    assert not await mailbox.has_state(OpType.FORWARD, "missing", step=1)

    # Test get_all_states
    all_states = await mailbox.get_all_states(OpType.FORWARD, step=1)
    assert "id" in all_states
    assert "config" in all_states
    assert all_states["id"]["value"] == 42

    # Test tensor storage with names
    await mailbox.put_tensor(OpType.FORWARD, "tensor_a", step=1, rank=0, data=[1, 2, 3])
    await mailbox.put_tensor(OpType.FORWARD, "tensor_a", step=1, rank=1, data=[4, 5, 6])
    await mailbox.put_tensor(OpType.FORWARD, "tensor_b", step=1, rank=0, data=[7, 8, 9])

    assert await mailbox.get_tensor_shard_count(OpType.FORWARD, "tensor_a", step=1) == 2
    assert await mailbox.get_tensor_shard_count(OpType.FORWARD, "tensor_b", step=1) == 1

    # Test has_all_tensors
    assert await mailbox.has_all_tensors(
        OpType.FORWARD, step=1, expected_counts={"tensor_a": 2}
    )
    assert not await mailbox.has_all_tensors(
        OpType.FORWARD, step=1, expected_counts={"tensor_a": 3}
    )

    # Test clear_step
    await mailbox.clear_step(OpType.FORWARD, step=1)
    assert not await mailbox.has_state(OpType.FORWARD, "id", step=1)
    assert await mailbox.get_tensor_shard_count(OpType.FORWARD, "tensor_a", step=1) == 0

    print("WorkerMailbox named storage successful.")


def test_spmd_config_with_mesh():
    """Verify SPMDConfig stores DeviceMesh directly."""
    print("Testing SPMDConfig with DeviceMesh...")

    mesh = DeviceMesh((4, 2), axis_names=("data", "model"))
    config = SPMDConfig(mesh=mesh, user_forward_fn=forward_fn)

    assert config.has_mesh
    assert config.mesh is not None
    assert config.mesh.axis_names == ("data", "model")
    assert config.mesh.shape == (4, 2)
    assert config.get_mesh_size() == 8

    # Test without mesh
    config_no_mesh = SPMDConfig(user_forward_fn=forward_fn)
    assert not config_no_mesh.has_mesh
    assert config_no_mesh.get_mesh_size() == 1

    print("SPMDConfig with DeviceMesh successful.")


def test_operator_manager_gets_mesh_from_config():
    """Verify OperatorManager gets mesh from config directly."""
    print("Testing OperatorManager mesh from config...")

    mesh = DeviceMesh((2,), axis_names=("data",))
    config = SPMDConfig(
        mesh=mesh,
        user_forward_fn=forward_fn,
        user_backward_fn=backward_fn,
        user_update_fn=update_fn,
    )

    manager = OperatorManager(config, rank=0)

    # Verify mesh was set from config
    assert manager._mesh is mesh

    print("OperatorManager mesh from config successful.")


@pytest.mark.asyncio
async def test_spmd_worker_gets_mesh_from_config():
    """Verify SPMDWorker gets mesh from config directly."""
    print("Testing SPMDWorker mesh from config...")

    mesh = DeviceMesh((2,), axis_names=("data",))
    config = SPMDConfig(
        mesh=mesh,
        user_forward_fn=forward_fn,
        user_backward_fn=backward_fn,
        user_update_fn=update_fn,
    )

    worker_cls = getattr(SPMDWorker, "__ray_metadata__", None)
    if worker_cls:
        worker_cls = worker_cls.modified_class
    else:
        worker_cls = SPMDWorker
    worker = worker_cls(config, rank=0)

    # Verify worker has mesh from config
    assert worker._mesh is mesh
    assert worker._mesh.axis_names == ("data",)
    assert worker._mesh.shape == (2,)

    # Verify op_manager also has the mesh
    assert worker._op_manager._mesh is mesh

    print("SPMDWorker mesh from config successful.")


@pytest.mark.asyncio
async def test_receive_data_with_name():
    """Verify receive_state_data and receive_tensor_data work with name parameter."""
    print("Testing receive data with name parameter...")

    mesh = DeviceMesh((2,), axis_names=("data",))
    config = SPMDConfig(mesh=mesh, user_forward_fn=forward_fn)

    worker_cls = getattr(SPMDWorker, "__ray_metadata__", None)
    if worker_cls:
        worker_cls = worker_cls.modified_class
    else:
        worker_cls = SPMDWorker
    worker = worker_cls(config, rank=0)

    # Receive state data with name
    await worker.receive_state_data(OpType.FORWARD, "id", step=1, data={"value": 42})

    # Check data is in mailbox
    state = await worker._mailbox.get_state(OpType.FORWARD, "id", step=1)
    assert state is not None
    assert state["value"] == 42

    # Receive tensor data with name
    await worker.receive_tensor_data(
        OpType.FORWARD, "tensor", step=1, sender_rank=0, data=[1, 2, 3]
    )

    # Check tensor is in mailbox
    shards = await worker._mailbox.get_tensor_shards(OpType.FORWARD, "tensor", step=1)
    assert 0 in shards
    assert shards[0] == ([1, 2, 3], None)

    print("receive data with name parameter successful.")


if __name__ == "__main__":
    try:
        test_layout_mesh()
        test_mailbox_named_storage()
        test_spmd_config_with_mesh()
        test_operator_manager_gets_mesh_from_config()
        test_spmd_worker_gets_mesh_from_config()
        test_receive_data_with_name()
        print("\n" + "=" * 50)
        print("All verification tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
