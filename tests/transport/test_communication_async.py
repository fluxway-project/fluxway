import asyncio
from unittest.mock import MagicMock
import jax.numpy as jnp
import numpy as np
import pytest

from fluxway.transport.communication import SyncDataDistributor, AsyncDataDistributor
from fluxway.common.types.sharding import ShardingSpec, P
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.backend import set_backend


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


@pytest.fixture
def setup():
    local_rank = 0
    data = jnp.arange(10, dtype=jnp.float32)
    transport_tensor = TransportTensor(data, global_shape=(20,))
    mesh = DeviceMesh(axis_names=("x",), device_shape=(2,), device_ids=np.array([0, 1]))
    src_spec = ShardingSpec.chunk(P("x")).bind(mesh)
    dst_spec = ShardingSpec.chunk(P("x")).bind(mesh)
    return local_rank, transport_tensor, src_spec, dst_spec


def test_sync_distributor(setup):
    local_rank, transport_tensor, src_spec, dst_spec = setup
    dist = SyncDataDistributor(local_rank)

    # send_fn new signature: (target_id, tuuids, tensors, counts, global_shapes, meta_state_list, compact_params)
    send_fn = MagicMock(return_value="ok")

    dist.distribute(
        local_data=transport_tensor, source=src_spec, target=dst_spec, send_fn=send_fn
    )

    assert send_fn.called


def test_async_distributor(setup):
    local_rank, transport_tensor, src_spec, dst_spec = setup
    dist = AsyncDataDistributor(local_rank)

    async def async_send_fn(
        target_id,
        tuuids,
        tensors,
        counts,
        global_shapes,
        meta_state_list,
        compact_params,
    ):
        await asyncio.sleep(0.01)
        return f"ok_{target_id}"

    async def run_test():
        results, _ = await dist.distribute(
            local_data=transport_tensor,
            source=src_spec,
            target=dst_spec,
            send_fn=async_send_fn,
        )
        return results

    results = asyncio.run(run_test())
    assert len(results) > 0
    for r in results:
        assert r.startswith("ok_")


def test_async_distributor_complex_batch(setup):
    local_rank, _, src_spec, dst_spec = setup
    dist = AsyncDataDistributor(local_rank)

    dense_data = jnp.arange(10, dtype=jnp.float32)

    ragged_data = jnp.arange(6, dtype=jnp.float32)
    # lengths [2, 1, 3] -> splits [0, 2, 3, 6]
    ragged_tt = TransportTensor(
        ragged_data, nested_splits=[jnp.array([0, 2, 3, 6])], global_shape=(12,)
    )

    routing_indices = jnp.arange(10, dtype=jnp.int32)
    spatial_tt = TransportTensor(
        dense_data, routing_indices=routing_indices, global_shape=(20,)
    )

    origin_indices = jnp.array([100, 101, 102], dtype=jnp.int32)
    origin_data = jnp.arange(3, dtype=jnp.float32)
    origin_tt = TransportTensor(
        origin_data, origin_indices=origin_indices, global_shape=(6,)
    )

    batch = [spatial_tt, ragged_tt, origin_tt]

    async def async_send_fn(
        target_id,
        tuuids,
        tensors,
        counts,
        global_shapes,
        meta_state_list,
        compact_params,
    ):
        assert len(meta_state_list) == 3
        # spatial_tt has routing_indices (drives routing but no meta bit set)
        assert meta_state_list[0] == 0
        # ragged_tt: bit0=ragged
        assert bool(meta_state_list[1] & 1)
        # origin_tt: bit1=origin
        assert bool(meta_state_list[2] & 2)
        return f"ok_{target_id}"

    async def run_test():
        results, _ = await dist.distribute(
            local_data=batch, source=src_spec, target=dst_spec, send_fn=async_send_fn
        )
        return results

    asyncio.run(run_test())


def test_async_distributor_all_with_indices(setup):
    """Test a batch where all tensors have spatial indices."""
    local_rank, _, src_spec, dst_spec = setup
    dist = AsyncDataDistributor(local_rank)

    routing_indices = jnp.arange(10, dtype=jnp.int32)
    tt1 = TransportTensor(
        jnp.arange(10), routing_indices=routing_indices, global_shape=(20,)
    )
    tt2 = TransportTensor(
        jnp.arange(10) + 10, routing_indices=routing_indices, global_shape=(20,)
    )

    batch = [tt1, tt2]

    async def async_send_fn(
        target_id,
        tuuids,
        tensors,
        counts,
        global_shapes,
        meta_state_list,
        compact_params,
    ):
        # routing_indices drives routing but sets no meta bit
        assert meta_state_list == [0, 0]
        # Shared routing_indices deduplicated: only [tt1_data, tt2_data] sent
        assert len(tensors) == 2
        return f"ok_{target_id}"

    async def run_test():
        await dist.distribute(batch, src_spec, dst_spec, async_send_fn)

    asyncio.run(run_test())


def test_async_distributor_multiple_mixed(setup):
    """Test a mixed batch: [dense, ragged+indices, dense+indices]."""
    local_rank, _, src_spec, dst_spec = setup
    dist = AsyncDataDistributor(local_rank)

    idx = jnp.arange(5, dtype=jnp.int32)
    t1 = TransportTensor(jnp.zeros(5), global_shape=(10,))  # dense
    t2 = TransportTensor(
        jnp.ones(5),
        nested_splits=[jnp.array([0, 2, 5])],
        routing_indices=idx,
        global_shape=(10,),
    )  # ragged+idx
    t3 = TransportTensor(
        jnp.ones(5) * 2, routing_indices=idx, global_shape=(10,)
    )  # dense+idx

    batch = [t1, t2, t3]

    async def async_send_fn(
        target_id,
        tuuids,
        tensors,
        counts,
        global_shapes,
        meta_state_list,
        compact_params,
    ):
        # t1=dense(0), t2=ragged(1), t3=dense+routing_indices(routing sets no bit=0)
        assert meta_state_list == [0, 1, 0]
        # flat tensors: [t1_data, t2_data, t2_len, t3_data]
        assert len(tensors) == 4
        assert len(counts) == 4
        return f"ok_{target_id}"

    async def run_test():
        await dist.distribute(batch, src_spec, dst_spec, async_send_fn)

    asyncio.run(run_test())
