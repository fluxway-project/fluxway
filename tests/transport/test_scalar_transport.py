import pytest
import jax.numpy as jnp

from fluxway.common.types.tensor import TransportTensor
from fluxway.common.types.sharding import ShardingConfig
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.communication import DataDistributor, TransportReceiver
from fluxway.common.backend import set_backend


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


def test_scalar_replication_self_delivery():
    """
    For a scalar replicated across a 2-device mesh, each sender rank delivers
    only to its own corresponding target rank — not a broadcast.
    This reflects the current architecture where each rank holds its own replica.
    """
    mesh = DeviceMesh(axis_names=("dp",), device_shape=(2,))
    source_spec = ShardingConfig(type="static", axis=[]).to_spec().bind(mesh)
    target_spec = ShardingConfig(type="static", axis=[]).to_spec().bind(mesh)

    scalar_val = jnp.array(42.0)
    tt = TransportTensor.from_tensor(scalar_val, global_shape=())

    # Rank 0: sends only to itself (rank 0)
    receivers_0 = [TransportReceiver(), TransportReceiver()]

    def mock_send_0(
        target_rank,
        tuuids,
        tensors,
        counts,
        global_shapes=None,
        meta_state_list=None,
        compact_params=None,
    ):
        receivers_0[target_rank].receive_slices(0, tensors, counts)
        return True

    distributor_0 = DataDistributor(0)
    distributor_0.distribute(
        tt, source_spec, target_spec, send_fn=mock_send_0, tuuid=20
    )
    assert len(receivers_0[0].get_buffered_slices()) == 1
    assert (
        len(receivers_0[1].get_buffered_slices()) == 0
    )  # rank 0 does NOT broadcast to rank 1

    # Rank 1: sends only to itself (rank 1)
    receivers_1 = [TransportReceiver(), TransportReceiver()]

    def mock_send_1(
        target_rank,
        tuuids,
        tensors,
        counts,
        global_shapes=None,
        meta_state_list=None,
        compact_params=None,
    ):
        receivers_1[target_rank].receive_slices(1, tensors, counts)
        return True

    distributor_1 = DataDistributor(1)
    distributor_1.distribute(
        tt, source_spec, target_spec, send_fn=mock_send_1, tuuid=20
    )
    assert len(receivers_1[0].get_buffered_slices()) == 0
    assert (
        len(receivers_1[1].get_buffered_slices()) == 1
    )  # rank 1 sends to its own slot
