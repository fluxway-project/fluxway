import pytest
import jax.numpy as jnp
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.types.sharding import ShardingConfig
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.communication import DataDistributor, TransportReceiver
from fluxway.common.backend import set_backend
from fluxway.common.decorators import operator


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


def test_scalar_validation():
    mesh = DeviceMesh(axis_names=("dp",), device_shape=(2,))

    # Valid configs
    ShardingConfig(type="static", axis=[]).to_spec(mesh)
    ShardingConfig(type="static", axis=[], reduce="sum").to_spec(mesh)

    from fluxway.common.decorators import data_type

    @data_type()
    class MyOut:
        y: jnp.ndarray

    # Invalid: reduction on sharded axis
    with pytest.raises(
        ValueError, match="Scalar reduction .* cannot be used with sharding axis"
    ):

        @operator(
            output_sharding={"y": {"type": "static", "axis": ["dp"], "reduce": "sum"}}
        )
        def bad_op(x: jnp.ndarray) -> MyOut:
            return MyOut(y=x)


def test_scalar_replication_rank_0_only():
    """Rank 0 sends the scalar, others don't if reduction is None."""
    mesh = DeviceMesh(axis_names=("dp",), device_shape=(2,))
    source_spec = ShardingConfig(type="static", axis=[]).to_spec(mesh)
    target_spec = ShardingConfig(type="static", axis=[]).to_spec(mesh)  # No reduction

    scalar_val = jnp.array(42.0)
    tt = TransportTensor.from_tensor(scalar_val, global_shape=())

    # Rank 0 case: should send to all targets
    receivers_0 = [TransportReceiver(), TransportReceiver()]

    def mock_send_0(target_rank, tuuids, tensors, counts, *args, **kwargs):
        receivers_0[target_rank].receive_slices(0, tensors, counts)
        return True

    distributor_0 = DataDistributor(0)
    distributor_0.distribute(
        tt, source_spec, target_spec, send_fn=mock_send_0, tuuid=20
    )
    assert len(receivers_0[0].get_buffered_slices()) == 1
    assert len(receivers_0[1].get_buffered_slices()) == 1

    # Rank 1 case: should skip sending
    receivers_1 = [TransportReceiver(), TransportReceiver()]

    def mock_send_1(target_rank, tuuids, tensors, counts, *args, **kwargs):
        receivers_1[target_rank].receive_slices(1, tensors, counts)
        return True

    distributor_1 = DataDistributor(1)
    distributor_1.distribute(
        tt, source_spec, target_spec, send_fn=mock_send_1, tuuid=20
    )
    assert len(receivers_1[0].get_buffered_slices()) == 0
    assert len(receivers_1[1].get_buffered_slices()) == 0


if __name__ == "__main__":
    pytest.main([__file__])
