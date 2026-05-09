import shutil
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from fluxway.control_plane.checkpointer.orbax import OrbaxBackend


@pytest.mark.asyncio
async def test_orbax_resharding():
    # Only run if we have at least 4 devices (we can mock this via XLA_FLAGS in the test environment if needed,
    # but typically pytest handles this or we can just use however many devices are available.
    # To be safe, we'll try to get 4 devices, or skip if not possible.
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip(
            f"Need at least 4 devices to test resharding from 2 to 4, got {len(devices)}"
        )

    test_dir = "/tmp/reshard_test_pytest"
    shutil.rmtree(test_dir, ignore_errors=True)
    backend = OrbaxBackend()

    # 1. dp2 environment (simulated with 2 devices)
    mesh2 = jax.sharding.Mesh(devices[:2], ("x",))
    sharding2 = jax.sharding.NamedSharding(mesh2, jax.sharding.PartitionSpec("x"))

    x = jnp.arange(16)
    x_sharded = jax.device_put(x, sharding2)

    state = {"operator_states": {"my_op": x_sharded}}

    # Save with dp2
    metadata = await backend.save(state, test_dir, "spmd1", 0, 1)

    # 2. dp4 environment
    mesh4 = jax.sharding.Mesh(devices[:4], ("x",))
    sharding4 = jax.sharding.NamedSharding(mesh4, jax.sharding.PartitionSpec("x"))

    # Target state for dp4
    target_x = jnp.zeros(16)
    target_x_sharded = jax.device_put(target_x, sharding4)
    target_state = {"operator_states": {"my_op": target_x_sharded}}

    # Load using the backend with target_state support
    loaded_state = await backend.load(metadata, target_state=target_state)
    loaded_x = loaded_state["operator_states"]["my_op"]

    assert loaded_x.sharding == target_x_sharded.sharding, (
        "Loaded sharding does not match target sharding."
    )

    # Verify values are correctly restored
    assert jnp.array_equal(loaded_x, x), "Loaded values do not match original values."

    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_orbax_mixed_resharding():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip(
            f"Need at least 4 devices to test 3D mixed resharding, got {len(devices)}"
        )

    test_dir = "/tmp/reshard_test_mixed_pytest"
    shutil.rmtree(test_dir, ignore_errors=True)
    backend = OrbaxBackend()

    # 1. Source environment: dp=2, fsdp=1, tp=2
    # We reshape the 4 devices into (2, 1, 2)
    mesh_src = jax.sharding.Mesh(
        np.array(devices[:4]).reshape((2, 1, 2)), ("dp", "fsdp", "tp")
    )
    sharding_src = jax.sharding.NamedSharding(
        mesh_src, jax.sharding.PartitionSpec("dp", "fsdp", "tp")
    )

    # Create a 3D array of shape (16, 16, 16)
    x = jnp.arange(16 * 16 * 16).reshape((16, 16, 16))
    x_sharded = jax.device_put(x, sharding_src)

    state = {"operator_states": {"mixed_op": x_sharded}}

    # Save
    metadata = await backend.save(state, test_dir, "spmd1", 0, 1)

    # 2. Target environment: dp=1, fsdp=2, tp=2
    # We reshape the 4 devices into (1, 2, 2)
    mesh_tgt = jax.sharding.Mesh(
        np.array(devices[:4]).reshape((1, 2, 2)), ("dp", "fsdp", "tp")
    )
    sharding_tgt = jax.sharding.NamedSharding(
        mesh_tgt, jax.sharding.PartitionSpec("dp", "fsdp", "tp")
    )

    target_x = jnp.zeros((16, 16, 16))
    target_x_sharded = jax.device_put(target_x, sharding_tgt)
    target_state = {"operator_states": {"mixed_op": target_x_sharded}}

    # Load with target_state
    loaded_state = await backend.load(metadata, target_state=target_state)
    loaded_x = loaded_state["operator_states"]["mixed_op"]

    assert loaded_x.sharding == target_x_sharded.sharding, (
        f"Loaded sharding {loaded_x.sharding} does not match target {target_x_sharded.sharding}."
    )

    assert jnp.array_equal(loaded_x, x), "Loaded values do not match original values."

    shutil.rmtree(test_dir, ignore_errors=True)


class MockUserOperator:
    """Mock user operator demonstrating how save_state/load_state expose target_state."""

    def __init__(self, mesh, partition_spec):
        self.mesh = mesh
        self.sharding = jax.sharding.NamedSharding(mesh, partition_spec)
        # Initialize with zeros, but crucially, with the CORRECT target sharding!
        self.weights = jax.device_put(jnp.zeros((16, 16)), self.sharding)

    def save_state(self) -> dict:
        # The user's save_state automatically acts as the target_state provider during load!
        return {"weights": self.weights}

    def restore_state(self, state: dict) -> None:
        self.weights = state["weights"]


@pytest.mark.asyncio
async def test_user_operator_api():
    """Test how a user operator exposes target_sharding implicitly via save_state."""
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("Need at least 4 devices")

    test_dir = "/tmp/reshard_test_user_op"
    shutil.rmtree(test_dir, ignore_errors=True)
    backend = OrbaxBackend()

    # 1. dp=2 environment (training Phase)
    mesh2 = jax.sharding.Mesh(np.array(devices[:2]), ("dp",))
    op_dp2 = MockUserOperator(mesh2, jax.sharding.PartitionSpec("dp"))
    # Simulate some training progress
    op_dp2.weights = jax.device_put(jnp.ones((16, 16)), op_dp2.sharding)

    # Checkpointer calls operator's save_state
    saved_state = {"operator_states": {"my_op": op_dp2.save_state()}}
    metadata = await backend.save(saved_state, test_dir, "spmd1", 0, 1)

    # 2. dp=4 environment (restore Phase)
    mesh4 = jax.sharding.Mesh(np.array(devices[:4]), ("dp",))
    # User operator is instantiated in the NEW environment, naturally getting the new sharding
    op_dp4 = MockUserOperator(mesh4, jax.sharding.PartitionSpec("dp"))

    # The Fluxway Worker dynamically queries the operator for its target state:
    target_state = {"operator_states": {"my_op": op_dp4.save_state()}}

    # Backend performs the resharded load
    loaded_state = await backend.load(metadata, target_state=target_state)

    # The Fluxway Worker pushes the loaded state back into the operator
    op_dp4.restore_state(loaded_state["operator_states"]["my_op"])

    # Verify that the operator now holds the correct values WITH the new sharding!
    assert op_dp4.weights.sharding == op_dp4.sharding
    assert jnp.array_equal(op_dp4.weights, jnp.ones((16, 16)))

    shutil.rmtree(test_dir, ignore_errors=True)
