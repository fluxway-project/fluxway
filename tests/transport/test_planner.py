"""Tests for TransportPlanner: Diff(L_src, L_dst) across all sharding combinations."""

import pytest
import jax.numpy as jnp

from fluxway.transport.backend import set_backend
from fluxway.transport.planner import TransportPlanner
from fluxway.transport.sharding import ShardingSpec
from fluxway.transport.tensor import TransportTensor


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


@pytest.fixture(autouse=False)
def use_torch_backend():
    set_backend("torch")
    yield


# ---- Scenario 1: static -> static (chunk -> chunk, different world_size) ----


def test_static_to_static():
    """Source: chunk(2), Rank 0 has [0,50). Target: chunk(4). Expect 2 plans."""
    src_spec = ShardingSpec.chunk(dim=0)
    dst_spec = ShardingSpec.chunk(dim=0)

    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(100,),
        local_rank=0,
        src_world_size=2,
        dst_world_size=4,
    )

    assert plan.partition_dim == 0
    assert plan.target_world_size == 4

    counts = plan.send_counts.tolist()
    assert counts[0] == 25
    assert counts[1] == 25
    assert counts[2] == 0
    assert counts[3] == 0


def test_static_to_static_rank1():
    """Same scenario, but from Rank 1's perspective."""
    src_spec = ShardingSpec.chunk(dim=0)
    dst_spec = ShardingSpec.chunk(dim=0)

    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(100,),
        local_rank=1,
        src_world_size=2,
        dst_world_size=4,
    )

    counts = plan.send_counts.tolist()
    assert counts[0] == 0
    assert counts[1] == 0
    assert counts[2] == 25
    assert counts[3] == 25


# ---- Scenario 2: static -> dynamic (chunk -> hash) ----


def test_static_to_dynamic_hash():
    """Source: chunk(1). Target: hash(2) via topology."""
    splits = [jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)]

    src_spec = ShardingSpec.chunk(dim=0)
    dst_spec = ShardingSpec.hash(dim=0)

    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(4,),
        local_rank=0,
        src_world_size=1,
        dst_world_size=2,
        topology_meta=splits,
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 2]


# ---- Scenario 3: dynamic -> static (hash -> chunk) ----


def test_dynamic_to_static():
    """Source: hash(2), local_rank=0 owns some rows. Target: chunk(2)."""
    dep = jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    src_spec = ShardingSpec.hash(dim=0)
    dst_spec = ShardingSpec.chunk(dim=0)

    # Compute source layout with dependency to know who owns what
    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(6,),
        local_rank=0,
        src_world_size=2,
        dst_world_size=2,
        dependency=dep,  # src uses this dependency
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 1]


# ---- Scenario 4: dynamic -> dynamic (greedy -> hash) ----


def test_dynamic_to_dynamic():
    """Source: greedy(2). Target: hash(2)."""
    splits = [jnp.array([0, 10, 30, 40], dtype=jnp.int32)]

    src_spec = ShardingSpec.greedy(dim=0)
    dst_spec = ShardingSpec.hash(dim=0)

    # Greedy: [10, 20, 10] -> rank 0 gets rows [0, 2], rank 1 gets row [1]
    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(3,),
        local_rank=0,
        src_world_size=2,
        dst_world_size=2,
        topology_meta=splits,
    )

    counts = plan.send_counts.tolist()
    assert counts[0] == 2
    assert counts[1] == 0


# ---- Scenario 5: dependency (key -> value dual-tensor) ----


def test_dependency_dual_tensor():
    """Key-value dispatch: key determines device, value follows."""
    # key: [4], value: [4, 8]
    # key hash determines: [0, 1, 0, 1]
    key_data = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

    val_spec = ShardingSpec.derive_from(
        driver_spec=ShardingSpec.hash(dim=0),
        driver_data_shape=(4,),
        world_size=2,
        driver_topology=key_data,
        follow_dim=0,
    )

    # Source: all on one rank (chunk ws=1)
    src_spec = ShardingSpec.chunk(dim=0)

    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=val_spec,
        data_shape=(4,),
        local_rank=0,
        src_world_size=1,
        dst_world_size=2,
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 2]


# ---- Scenario 6: replicate (fan-out) ----


def test_replicate_fanout():
    """chunk(2) -> implicit replicate on 2nd mesh axis."""
    from fluxway.transport.sharding import DeviceMesh

    # Mesh: (2, 2)
    mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))

    # Source: chunk(2) on world_size 2 (logical X). Implicitly mapped?
    # Let's say src has no mesh, just rank 0 and 1.
    src_spec = ShardingSpec.chunk(dim=0)

    # Dest: Shard on "x", Replicate on "y".
    dst_spec = ShardingSpec.chunk(dim=0)
    dst_spec.mesh = mesh
    dst_spec.sharding_axis = "x"

    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(10,),
        local_rank=0,
        src_world_size=2,
        dst_world_size=4,  # 2x2 mesh
    )

    assert plan.replication_factor == 2

    counts = plan.send_counts.tolist()
    assert counts[0] == 5
    assert counts[1] == 0
    assert counts[2] == 0
    assert counts[3] == 0


# ---- Scenario 7: ragged data (gather/slice correctness) ----


def test_ragged_data_reorder():
    """Verify that gathering with plan.sort_indices correctly reorders ragged data."""
    # 4 ragged rows: [[a, b], [c], [d, e, f], [g, h]]
    splits = [jnp.array([0, 2, 3, 6, 8], dtype=jnp.int32)]
    values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    local_data = TransportTensor(values, splits)

    src_spec = ShardingSpec.chunk(dim=0)
    dst_spec = ShardingSpec.cyclic(dim=0)

    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(4,),
        local_rank=0,
        src_world_size=1,
        dst_world_size=2,
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 2]

    # Reorder and verify structure
    reordered = local_data.gather(plan.sort_indices)
    reordered_lists = reordered.to_lists()

    assert reordered_lists[0] == [1.0, 2.0]
    assert reordered_lists[1] == [4.0, 5.0, 6.0]
    assert reordered_lists[2] == [3.0]
    assert reordered_lists[3] == [7.0, 8.0]


# ---- iter_device_slices ----


def test_iter_device_slices():
    """Verify iter_device_slices yields correct (device, offset, count) tuples."""
    src_spec = ShardingSpec.chunk(dim=0)
    dst_spec = ShardingSpec.cyclic(dim=0)

    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(6,),
        local_rank=0,
        src_world_size=1,
        dst_world_size=3,
    )

    slices = list(plan.iter_device_slices())
    # cyclic(3): [0,3]->dev0, [1,4]->dev1, [2,5]->dev2

    assert len(slices) == 3
    total = sum(count for _, _, count in slices)
    assert total == 6
