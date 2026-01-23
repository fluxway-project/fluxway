
import sys
import os
import threading
import time
from typing import Dict, Any, Optional
from unittest.mock import MagicMock

# Mock ray before importing fluxway to avoid starting Ray instance
mock_ray = MagicMock()
# Mock @ray.remote to just return the class unmodified
def mock_remote(*args, **kwargs):
    def decorator(cls):
        return cls
    return decorator
mock_ray.remote = mock_remote
sys.modules["ray"] = mock_ray

# Add project root to path
sys.path.append(os.getcwd())

from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.spmd.worker.mailbox import WorkerMailbox, OpType
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.common.types.data_types import SPMDConfig
from fluxway.common.types.sharding import LayoutMesh, create_mesh

# Mock user functions
def forward_fn(state, batch):
    return state

def backward_fn(state, grad):
    return state

def update_fn(state, grad):
    return state


def test_layout_mesh():
    """Verify LayoutMesh works with user-defined axis names."""
    print("Testing LayoutMesh with user-defined axes...")
    
    mesh = LayoutMesh(
        axis_names=("data", "model"),
        axis_shapes=(4, 2)
    )
    
    # Check properties
    assert mesh.num_devices == 8
    assert mesh.ndim == 2
    assert mesh.axis_names == ("data", "model")
    assert mesh.axis_shapes == (4, 2)
    
    # Check axis lookups
    assert mesh.get_axis_size("data") == 4
    assert mesh.get_axis_size("model") == 2
    assert mesh.get_axis_index("data") == 0
    assert mesh.get_axis_index("model") == 1
    
    # Check coordinate lookups
    coord = mesh.get_coordinate(5)  # rank 5 in 4x2 mesh
    assert coord == (2, 1)  # row 2, col 1
    
    # Check reverse lookup
    assert mesh.get_rank((2, 1)) == 5
    
    print("LayoutMesh with user-defined axes successful.")


def test_create_mesh_helper():
    """Verify create_mesh convenience function."""
    print("Testing create_mesh helper...")
    
    mesh = create_mesh(
        axis_names=("dp", "tp", "pp"),
        axis_shapes=(2, 2, 2)
    )
    
    assert mesh.num_devices == 8
    assert mesh.ndim == 3
    
    print("create_mesh helper successful.")


def test_mailbox_named_storage():
    """Verify WorkerMailbox works with named tensors/states."""
    print("Testing WorkerMailbox named storage...")
    
    mailbox = WorkerMailbox()
    
    # Test state storage with names
    mailbox.put_state(OpType.FORWARD, "id", step=1, data={"value": 42})
    mailbox.put_state(OpType.FORWARD, "config", step=1, data={"lr": 0.01})
    
    assert mailbox.has_state(OpType.FORWARD, "id", step=1)
    assert mailbox.has_state(OpType.FORWARD, "config", step=1)
    assert not mailbox.has_state(OpType.FORWARD, "missing", step=1)
    
    # Test get_all_states
    all_states = mailbox.get_all_states(OpType.FORWARD, step=1)
    assert "id" in all_states
    assert "config" in all_states
    assert all_states["id"]["value"] == 42
    
    # Test tensor storage with names
    mailbox.put_tensor(OpType.FORWARD, "tensor_a", step=1, rank=0, data=[1, 2, 3])
    mailbox.put_tensor(OpType.FORWARD, "tensor_a", step=1, rank=1, data=[4, 5, 6])
    mailbox.put_tensor(OpType.FORWARD, "tensor_b", step=1, rank=0, data=[7, 8, 9])
    
    assert mailbox.get_tensor_shard_count(OpType.FORWARD, "tensor_a", step=1) == 2
    assert mailbox.get_tensor_shard_count(OpType.FORWARD, "tensor_b", step=1) == 1
    
    # Test has_all_tensors
    assert mailbox.has_all_tensors(OpType.FORWARD, step=1, expected_counts={"tensor_a": 2})
    assert not mailbox.has_all_tensors(OpType.FORWARD, step=1, expected_counts={"tensor_a": 3})
    
    # Test clear_step
    mailbox.clear_step(OpType.FORWARD, step=1)
    assert not mailbox.has_state(OpType.FORWARD, "id", step=1)
    assert mailbox.get_tensor_shard_count(OpType.FORWARD, "tensor_a", step=1) == 0
    
    print("WorkerMailbox named storage successful.")


def test_spmd_config_with_mesh():
    """Verify SPMDConfig stores LayoutMesh directly."""
    print("Testing SPMDConfig with LayoutMesh...")
    
    mesh = LayoutMesh(axis_names=("data", "model"), axis_shapes=(4, 2))
    config = SPMDConfig(mesh=mesh, user_forward_fn=forward_fn)
    
    assert config.has_mesh
    assert config.mesh is not None
    assert config.mesh.axis_names == ("data", "model")
    assert config.mesh.axis_shapes == (4, 2)
    assert config.get_mesh_size() == 8
    
    # Test without mesh
    config_no_mesh = SPMDConfig(user_forward_fn=forward_fn)
    assert not config_no_mesh.has_mesh
    assert config_no_mesh.get_mesh_size() == 1
    
    print("SPMDConfig with LayoutMesh successful.")


def test_operator_manager_gets_mesh_from_config():
    """Verify OperatorManager gets mesh from config directly."""
    print("Testing OperatorManager mesh from config...")
    
    mesh = LayoutMesh(axis_names=("data",), axis_shapes=(2,))
    config = SPMDConfig(
        mesh=mesh,
        user_forward_fn=forward_fn,
        user_backward_fn=backward_fn,
        user_update_fn=update_fn,
    )
    
    manager = OperatorManager(config)
    manager.set_self_rank(0)
    
    # Verify mesh was set from config
    assert manager._mesh is mesh
    
    # Register predecessor
    pred_sharding = {
        "tensor": {"data": 2},
        "id": {},
    }
    manager.register_predecessor(
        spmd_id="pred_spmd",
        mesh=mesh,
        sharding=pred_sharding,
        handlers={0: "mock_actor_0", 1: "mock_actor_1"},
        op_type=OpType.FORWARD,
    )
    
    # Check expected tensor names
    expected_names = manager.get_expected_tensor_names(OpType.FORWARD)
    assert "tensor" in expected_names
    assert "id" in expected_names
    
    print("OperatorManager mesh from config successful.")


def test_spmd_worker_gets_mesh_from_config():
    """Verify SPMDWorker gets mesh from config directly."""
    print("Testing SPMDWorker mesh from config...")
    
    mesh = LayoutMesh(axis_names=("data",), axis_shapes=(2,))
    config = SPMDConfig(
        mesh=mesh,
        user_forward_fn=forward_fn,
        user_backward_fn=backward_fn,
        user_update_fn=update_fn,
    )
    
    worker = SPMDWorker(config, rank=0)
    
    # Verify worker has mesh from config
    assert worker._mesh is mesh
    assert worker._mesh.axis_names == ("data",)
    assert worker._mesh.axis_shapes == (2,)
    
    # Verify op_manager also has the mesh
    assert worker._op_manager._mesh is mesh
    
    print("SPMDWorker mesh from config successful.")


def test_receive_data_with_name():
    """Verify receive_state_data and receive_tensor_data work with name parameter."""
    print("Testing receive data with name parameter...")
    
    mesh = LayoutMesh(axis_names=("data",), axis_shapes=(2,))
    config = SPMDConfig(mesh=mesh, user_forward_fn=forward_fn)
    
    worker = SPMDWorker(config, rank=0)
    
    # Receive state data with name
    worker.receive_state_data(OpType.FORWARD, "id", step=1, data={"value": 42})
    
    # Check data is in mailbox
    state = worker._mailbox.get_state(OpType.FORWARD, "id", step=1)
    assert state is not None
    assert state["value"] == 42
    
    # Receive tensor data with name
    worker.receive_tensor_data(OpType.FORWARD, "tensor", step=1, sender_rank=0, data=[1, 2, 3])
    
    # Check tensor is in mailbox
    shards = worker._mailbox.get_tensor_shards(OpType.FORWARD, "tensor", step=1)
    assert 0 in shards
    assert shards[0] == [1, 2, 3]
    
    print("receive data with name parameter successful.")


if __name__ == "__main__":
    try:
        test_layout_mesh()
        test_create_mesh_helper()
        test_mailbox_named_storage()
        test_spmd_config_with_mesh()
        test_operator_manager_gets_mesh_from_config()
        test_spmd_worker_gets_mesh_from_config()
        test_receive_data_with_name()
        print("\n" + "="*50)
        print("All verification tests passed!")
        print("="*50)
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
