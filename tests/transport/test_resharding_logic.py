import pytest
import jax.numpy as jnp
from fluxway.common.backend import set_backend, get_backend
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.planner import TransportPlanner
from fluxway.common.types.sharding import ShardingSpec, P


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


def test_resharding_2_to_6_static_chunk():
    """Reproduce the Mesh 2 -> Mesh 6 data loss/mis-segmentation."""
    # Source Mesh: 2 nodes
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    # Target Mesh: 6 nodes
    dst_mesh = DeviceMesh(device_shape=(6,), axis_names=("x",))

    src_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst_spec = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    global_shape = (24,)

    # Analyze topology from Dense Rank 0's perspective
    # Dense Rank 0 should connect to whichever Processor Ranks have indices [0, 1, 2, 3]
    # Processor Rank 0 has [0..11], Processor Rank 1 has [12..23].
    # So Dense Rank 0 should ONLY connect to Processor Rank 0.

    producers_for_dense_0 = src_spec.resolve_producers(dst_spec, target_rank=0)
    assert 0 in producers_for_dense_0
    assert (
        1 not in producers_for_dense_0
    )  # Index 12 is way beyond Dense 0's range [0..3]

    # Analyze topology from Dense Rank 3's perspective
    # Dense Rank 3 (Mesh 6) should have indices [12, 13, 14, 15]
    # These indices belong to Processor Rank 1 [12..23].
    # So Dense Rank 3 should ONLY connect to Processor Rank 1.

    producers_for_dense_3 = src_spec.resolve_producers(dst_spec, target_rank=3)
    assert 0 not in producers_for_dense_3
    assert 1 in producers_for_dense_3

    # From Processor Rank 0's perspective
    # It should send 4 items to Rank 0, 4 to Rank 1, 4 to Rank 2, and 0 to Ranks 3,4,5.

    plan_0 = TransportPlanner.plan(
        src_local_rank=0,
        src_spec=src_spec,
        dst_spec=dst_spec,
        src_data_shape=global_shape,
    )
    counts_0 = plan_0.send_counts.tolist()
    assert counts_0 == [4, 4, 4, 0, 0, 0]

    # From Processor Rank 1's perspective
    # It should send 0 to Ranks 0,1,2, and 4 to Ranks 3,4,5.

    plan_1 = TransportPlanner.plan(
        src_local_rank=1,
        src_spec=src_spec,
        dst_spec=dst_spec,
        src_data_shape=global_shape,
    )
    counts_1 = plan_1.send_counts.tolist()
    assert counts_1 == [0, 0, 0, 4, 4, 4]


def test_identity_collapse_hash():
    """Reproduce the bug where unique_ids are overwritten by rank indices."""
    mesh = DeviceMesh(device_shape=(6,), axis_names=("x",))
    # Unique IDs [0, 6, 12, 18] -> all hash to 0 mod 6
    data = jnp.array([0, 6, 12, 18], dtype=jnp.int32)

    src_mesh = DeviceMesh(device_shape=(1,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)

    # Dynamic routing based on the values themselves
    dst_spec = ShardingSpec.hash(P("x")).bind(mesh)

    plan = TransportPlanner.compute_plan(
        src_local_rank=0,
        src_data_shape=(4,),
        src_spec=src_spec,
        dst_spec=dst_spec,
        src_routing_based_values=data,  # Data determines route
    )

    counts = plan.send_counts.tolist()
    # All should go to Rank 0
    assert counts[0] == 4
    for i in range(1, 6):
        assert counts[i] == 0

    # Simulate packing logic to ensure data is not corrupted by metadata
    backend = get_backend()
    reordered_data = backend.take(data, plan.sort_indices)
    assert backend.to_numpy(reordered_data).tolist() == [0, 6, 12, 18]

    # If the bug were present, reordered_data would be the ranks: [0, 0, 0, 0]
    assert not all(x == 0 for x in backend.to_numpy(reordered_data).tolist())
