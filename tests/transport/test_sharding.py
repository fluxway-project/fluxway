"""Tests for algebraic sharding: Layout + ShardingSpec + partition functions."""

import pytest
import jax.numpy as jnp

from fluxway.transport.backend import set_backend
from fluxway.transport.mesh import DeviceMesh
from fluxway.transport.sharding import ShardingSpec, P


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


# ---- chunk ----


def test_chunk_layout_dense():
    """chunk on dense [100] -> 2 devices."""
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(100,))

    assert layout.dim == 0
    assert layout.world_size == 2
    assert layout.is_static is True
    assert layout.topology_kind == "dense"

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.arange(0, 50, dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.arange(50, 100, dtype=jnp.int32))


def test_chunk_layout_multidim():
    """chunk on dim=1 of [4, 10] -> 2 devices."""
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P(None, "x"))
    layout = spec._to_layout(data_shape=(4, 10))

    assert layout.dim == 1

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.arange(0, 5, dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.arange(5, 10, dtype=jnp.int32))


def test_chunk_layout_ragged():
    """chunk on ragged tensor (topology = splits)."""
    splits = [jnp.array([0, 3, 5], dtype=jnp.int32)]
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(2,), topology=splits)

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.array([0], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1], dtype=jnp.int32))


# ---- cyclic ----


def test_cyclic_layout():
    """cyclic on [10] -> 2 devices."""
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.cyclic(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(10,))

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.arange(0, 10, 2, dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.arange(1, 10, 2, dtype=jnp.int32))


# ---- hash ----


def test_hash_layout_with_topology():
    """hash using ragged splits as hash seed."""
    splits = [jnp.array([0, 10, 20], dtype=jnp.int32)]
    mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(2,), topology=splits)

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)
    indices2 = layout.get_local_indices(2)

    assert jnp.array_equal(indices0, jnp.array([0], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1], dtype=jnp.int32))
    assert len(indices2) == 0
    assert layout.is_static is False


def test_hash_layout_with_routing_indices():
    """hash using external routing_indices indices."""
    dep = jnp.array([5, 3, 7, 2], dtype=jnp.int32)
    mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(4,), routing_indices=dep)

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)
    indices2 = layout.get_local_indices(2)

    assert jnp.array_equal(indices0, jnp.array([1], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([2], dtype=jnp.int32))
    assert jnp.array_equal(indices2, jnp.array([0, 3], dtype=jnp.int32))


def test_hash_layout_with_custom_function():
    """hash using custom hash function via topology."""

    def custom_hash(indices):
        return indices * 2

    mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(4,), topology={"hash_fn": custom_hash})

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)
    indices2 = layout.get_local_indices(2)

    assert jnp.array_equal(indices0, jnp.array([0, 3], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([2], dtype=jnp.int32))
    assert jnp.array_equal(indices2, jnp.array([1], dtype=jnp.int32))


def test_hash_layout_default():
    """hash with no topology or routing_indices - hash indices directly."""
    mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(6,))

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)
    indices2 = layout.get_local_indices(2)

    assert jnp.array_equal(indices0, jnp.array([0, 3], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1, 4], dtype=jnp.int32))
    assert jnp.array_equal(indices2, jnp.array([2, 5], dtype=jnp.int32))


# ---- greedy ----


def test_greedy_layout():
    """greedy load-balanced on ragged tensor."""
    splits = [jnp.array([0, 10, 30, 40], dtype=jnp.int32)]
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.greedy(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(3,), topology=splits)

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.array([0, 2], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1], dtype=jnp.int32))


# ---- follow (routing_indices) ----


def test_follow_layout():
    """follow another layout's device_mesh."""
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    src_spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    src_layout = src_spec._to_layout(
        data_shape=(4,), routing_indices=jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    )

    follow_spec = ShardingSpec.follow(src_layout, dim=0)
    follow_layout = follow_spec._to_layout(data_shape=(4,))

    indices0 = follow_layout.get_local_indices(0)
    indices1 = follow_layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.array([0, 2], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1, 3], dtype=jnp.int32))


# ---- derive_from (convenience) ----


def test_derive_from():
    """derive_from: one-line key->value routing_indices."""
    key_data = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    val_spec = ShardingSpec.derive_from(
        driver_spec=ShardingSpec.hash(mesh=mesh, partition=P("x")),
        driver_data_shape=(4,),
        driver_topology=key_data,
        follow_dim=0,
    )

    layout = val_spec._to_layout(data_shape=(4, 16))

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.array([0, 2], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1, 3], dtype=jnp.int32))


# ---- replicated ----


def test_implicit_replication():
    """Sharding on one axis implies replication on others."""
    mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(10,))

    chunk0 = jnp.arange(0, 5, dtype=jnp.int32)
    chunk1 = jnp.arange(5, 10, dtype=jnp.int32)

    assert jnp.array_equal(layout.get_local_indices(0), chunk0)
    assert jnp.array_equal(layout.get_local_indices(1), chunk0)
    assert jnp.array_equal(layout.get_local_indices(2), chunk1)
    assert jnp.array_equal(layout.get_local_indices(3), chunk1)

    assert layout.replication_axes == ["y"]
    assert layout.replication_factor == 2


# ---- multi-axis sharding ----


def test_multi_axis_chunk_2x3():
    """Chunk on mesh (x=2, y=3) with P(('x','y')) → 6-way split."""
    mesh = DeviceMesh(device_shape=(2, 3), axis_names=("x", "y"))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P(("x", "y")))
    layout = spec._to_layout(data_shape=(12,))

    for r in range(6):
        indices = layout.get_local_indices(r)
        expected = jnp.arange(r * 2, r * 2 + 2, dtype=jnp.int32)
        assert jnp.array_equal(indices, expected), f"Rank {r}: {indices} != {expected}"


def test_multi_axis_replication():
    """Mesh (x=2, y=2, z=3), shard on ('x','y') → z is replicated (factor 3)."""
    mesh = DeviceMesh(device_shape=(2, 2, 3), axis_names=("x", "y", "z"))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P(("x", "y")))
    layout = spec._to_layout(data_shape=(8,))

    assert layout.replication_axes == ["z"]
    assert layout.replication_factor == 3

    chunk0 = jnp.arange(0, 2, dtype=jnp.int32)
    for r in [0, 1, 2]:
        assert jnp.array_equal(layout.get_local_indices(r), chunk0), f"Rank {r}"

    chunk1 = jnp.arange(2, 4, dtype=jnp.int32)
    for r in [3, 4, 5]:
        assert jnp.array_equal(layout.get_local_indices(r), chunk1), f"Rank {r}"


def test_multi_axis_cyclic():
    """Cyclic on mesh (x=2, y=2) with P(('x','y')) → 4-way cyclic."""
    mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    spec = ShardingSpec.cyclic(mesh=mesh, partition=P(("x", "y")))
    layout = spec._to_layout(data_shape=(8,))

    assert jnp.array_equal(
        layout.get_local_indices(0), jnp.array([0, 4], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        layout.get_local_indices(1), jnp.array([1, 5], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        layout.get_local_indices(2), jnp.array([2, 6], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        layout.get_local_indices(3), jnp.array([3, 7], dtype=jnp.int32)
    )


# ---- repr ----


def test_sharding_spec_repr():
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P("x"))
    r = repr(spec)
    assert "ChunkPartition" in r
    assert "P(" in r
