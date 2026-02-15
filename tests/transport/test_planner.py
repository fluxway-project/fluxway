"""Tests for TransportPlanner: Diff(L_src, L_dst) across all sharding combinations."""

import pytest
import jax.numpy as jnp

from fluxway.transport.backend import set_backend
from fluxway.transport.mesh import DeviceMesh
from fluxway.transport.planner import TransportPlanner
from fluxway.transport.sharding import ShardingSpec, P
from fluxway.transport.tensor import TransportTensor


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


# ---- Scenario 1: static -> static (chunk -> chunk, different world_size) ----


def test_static_to_static():
    """Source: chunk(2), Rank 0 has [0,50). Target: chunk(4). Expect 2 plans."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(mesh=src_mesh, partition=P("x"))
    dst_spec = ShardingSpec.chunk(mesh=dst_mesh, partition=P("x"))

    plan = TransportPlanner.plan(
        src_spec=src_spec, dst_spec=dst_spec, data_shape=(100,), local_rank=0
    )

    assert plan.dim == 0
    assert plan.target_world_size == 4

    counts = plan.send_counts.tolist()
    assert counts[0] == 25
    assert counts[1] == 25
    assert counts[2] == 0
    assert counts[3] == 0


def test_static_to_static_rank1():
    """Same scenario, but from Rank 1's perspective."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(mesh=src_mesh, partition=P("x"))
    dst_spec = ShardingSpec.chunk(mesh=dst_mesh, partition=P("x"))

    plan = TransportPlanner.plan(
        src_spec=src_spec, dst_spec=dst_spec, data_shape=(100,), local_rank=1
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

    src_mesh = DeviceMesh(device_shape=(1,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(mesh=src_mesh, partition=P("x"))
    dst_spec = ShardingSpec.hash(mesh=dst_mesh, partition=P("x"))

    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(4,),
        local_rank=0,
        topology=splits,
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 2]


# ---- Scenario 3: dynamic -> static (hash -> chunk) ----


def test_dynamic_to_static():
    """Source: hash(2), local_rank=0 owns some rows. Target: chunk(2)."""
    dep = jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    src_spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    dst_spec = ShardingSpec.chunk(mesh=mesh, partition=P("x"))

    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(6,),
        local_rank=0,
        routing_indices=dep,
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 1]


# ---- Scenario 4: dynamic -> dynamic (greedy -> hash) ----


def test_dynamic_to_dynamic():
    """Source: greedy(2). Target: hash(2)."""
    splits = [jnp.array([0, 10, 30, 40], dtype=jnp.int32)]

    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    src_spec = ShardingSpec.greedy(mesh=mesh, partition=P("x"))
    dst_spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))

    plan = TransportPlanner.plan(
        src_spec=src_spec,
        dst_spec=dst_spec,
        data_shape=(3,),
        local_rank=0,
        topology=splits,
    )

    counts = plan.send_counts.tolist()
    assert counts[0] == 2
    assert counts[1] == 0


# ---- Scenario 5: routing_indices (key -> value dual-tensor) ----


def test_routing_indices_dual_tensor():
    """Key-value dispatch: key determines device, value follows."""
    key_data = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))

    val_spec = ShardingSpec.derive_from(
        driver_spec=ShardingSpec.hash(mesh=mesh, partition=P("x")),
        driver_data_shape=(4,),
        driver_topology=key_data,
        follow_dim=0,
    )

    src_mesh = DeviceMesh(device_shape=(1,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(mesh=src_mesh, partition=P("x"))

    plan = TransportPlanner.plan(
        src_spec=src_spec, dst_spec=val_spec, data_shape=(4,), local_rank=0
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 2]


# ---- Scenario 6: replicate (fan-out) ----


def test_replicate_fanout():
    """chunk(2) -> implicit replicate on 2nd mesh axis."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))

    src_spec = ShardingSpec.chunk(mesh=src_mesh, partition=P("x"))
    dst_spec = ShardingSpec.chunk(mesh=dst_mesh, partition=P("x"))

    plan = TransportPlanner.plan(
        src_spec=src_spec, dst_spec=dst_spec, data_shape=(10,), local_rank=0
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
    splits = [jnp.array([0, 2, 3, 6, 8], dtype=jnp.int32)]
    values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    local_data = TransportTensor(values, splits)

    src_mesh = DeviceMesh(device_shape=(1,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(mesh=src_mesh, partition=P("x"))
    dst_spec = ShardingSpec.cyclic(mesh=dst_mesh, partition=P("x"))

    plan = TransportPlanner.plan(
        src_spec=src_spec, dst_spec=dst_spec, data_shape=(4,), local_rank=0
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 2]

    reordered = local_data.gather(plan.sort_indices)
    reordered_lists = reordered.to_lists()

    assert reordered_lists[0] == [1.0, 2.0]
    assert reordered_lists[1] == [4.0, 5.0, 6.0]
    assert reordered_lists[2] == [3.0]
    assert reordered_lists[3] == [7.0, 8.0]


# ---- iter_device_slices ----


def test_iter_device_slices():
    """Verify iter_device_slices yields correct (device, offset, count) tuples."""
    src_mesh = DeviceMesh(device_shape=(1,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(mesh=src_mesh, partition=P("x"))
    dst_spec = ShardingSpec.cyclic(mesh=dst_mesh, partition=P("x"))

    plan = TransportPlanner.plan(
        src_spec=src_spec, dst_spec=dst_spec, data_shape=(6,), local_rank=0
    )

    slices = list(plan.iter_device_slices())

    assert len(slices) == 3
    total = sum(count for _, _, count in slices)
    assert total == 6
