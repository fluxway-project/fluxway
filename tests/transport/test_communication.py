import pytest
import jax.numpy as jnp
import numpy as np

from fluxway.transport.tensor import TransportTensor
from fluxway.transport.sharding import ShardingSpec, P
from fluxway.transport.mesh import DeviceMesh
from fluxway.transport.communication import DataDistributor, DataReceiverMixin
from fluxway.transport.backend import set_backend


# Force JAX backend
@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


@pytest.fixture(autouse=False)
def use_torch_backend():
    set_backend("torch")
    yield


class MockReceiver(DataReceiverMixin):
    def __init__(self):
        super().__init__()


def test_distributor_receiver_integration():
    local_rank = 0

    src_mesh = DeviceMesh((1,), ("x",), jnp.array([0]))
    dst_mesh = DeviceMesh((2,), ("x",), jnp.array([0, 1]))

    source_spec = ShardingSpec.chunk(mesh=src_mesh, partition=P("x"))
    target_spec = ShardingSpec.chunk(mesh=dst_mesh, partition=P("x"))

    # Mock Data: 2 rows with unequal lengths (Ragged)
    # Row 0: 60 elements
    # Row 1: 40 elements
    # Global Shape (logical) is 2 rows? No, ShardingSpec.chunk(0) partitions dimension 0.
    # If dim 0 is rows, we have 2 rows.
    # Rank 0 gets Row 0. Rank 1 gets Row 1.

    # splits = [0, 60, 100] -> Row 0 len 60, Row 1 len 40.
    # Total elements = 100.
    splits = [jnp.array([0, 60, 100], dtype=jnp.int32)]
    values = jnp.array(np.random.randn(100))

    data = TransportTensor(values, splits)

    # Mock receivers
    receiver0 = MockReceiver()
    receiver1 = MockReceiver()
    receivers = [receiver0, receiver1]

    def mock_send(target_rank, payload):
        rec = receivers[target_rank]
        rec.receive_shard(local_rank, payload)
        return True

    distributor = DataDistributor(local_rank)

    # Execute Distribute
    results = distributor.distribute(
        local_data=data, source=source_spec, target=target_spec, send_fn=mock_send
    )

    assert len(results) == 2
    assert all(results)

    # Verify Receiver 0
    shards0 = receiver0.get_received_shards()
    assert len(shards0) == 1
    payload0 = shards0[0]
    assert payload0.global_offset == 0
    assert payload0.data.shape[0] == 60
    assert jnp.array_equal(payload0.data, values[0:60])
    assert len(payload0.nested_splits) > 0
    assert payload0.nested_splits[0][1] == 60

    # Verify Receiver 1
    shards1 = receiver1.get_received_shards()
    assert len(shards1) == 1
    payload1 = shards1[0]
    assert payload1.global_offset == 1
    assert payload1.data.shape[0] == 40
    assert jnp.array_equal(payload1.data, values[60:100])
    assert payload1.nested_splits[0][1] == 40
