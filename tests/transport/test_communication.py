import pytest
import jax.numpy as jnp
import numpy as np

from fluxway.transport.tensor import TransportTensor
from fluxway.common.types.sharding import ShardingSpec, P
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.communication import DataDistributor, TransportReceiver
from fluxway.common.backend import set_backend


# Force JAX backend
@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


@pytest.fixture(autouse=False)
def use_torch_backend():
    set_backend("torch")
    yield


def test_distributor_receiver_integration():
    local_rank = 0

    src_mesh = DeviceMesh((1,), ("x",), jnp.array([0]))
    dst_mesh = DeviceMesh((2,), ("x",), jnp.array([0, 1]))

    source_spec = ShardingSpec.chunk(mesh=src_mesh, partition=P("x"))
    target_spec = ShardingSpec.chunk(mesh=dst_mesh, partition=P("x"))

    # Use Dense Data for distribution test
    # 2 rows, 50 elements each
    values = jnp.array(np.random.randn(2, 50))
    data = TransportTensor(values)

    # Mock receivers
    receiver0 = TransportReceiver(wait_for_n_slices=0)
    receiver1 = TransportReceiver(wait_for_n_slices=0)
    receivers = [receiver0, receiver1]

    def mock_send(target_rank, tensors, counts):
        rec = receivers[target_rank]
        rec.receive_slices(local_rank, tensors, counts)
        return True

    distributor = DataDistributor(local_rank)

    results, _ = distributor.distribute(
        local_data=data, source=source_spec, target=target_spec, send_fn=mock_send
    )

    assert len(results) == 2
    assert all(results)

    # Verify Receiver 0
    shards0 = receiver0.get_buffered_slices()
    assert len(shards0) == 1
    tensors0, counts0 = shards0[0]
    payload_data0 = tensors0[0]

    assert counts0[0] == 1
    assert payload_data0.shape[0] >= 1
    assert jnp.array_equal(payload_data0[0:1], values[0:1])

    # Verify Receiver 1
    shards1 = receiver1.get_buffered_slices()
    assert len(shards1) == 1
    tensors1, counts1 = shards1[0]
    payload_data1 = tensors1[0]

    assert counts1[0] == 1
    assert payload_data1.shape[0] >= 1
    assert jnp.array_equal(payload_data1[0:1], values[1:2])
