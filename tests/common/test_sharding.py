"""Tests for algebraic sharding: Layout + ShardingSpec + partition functions."""

import pytest
import jax.numpy as jnp

from fluxway.common.backend import set_backend
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.sharding import ShardingSpec, P
from fluxway.common.topology import (
    TopologyMorphology,
    RaggedTopology,
    IndexedTopology,
    CallableTopology,
)


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

    assert layout.partitions[0] == ("x",)
    assert layout.world_size == 2
    assert layout.is_static is True
    assert layout.topology_morphology == TopologyMorphology.CONTINUOUS

    indices0 = layout.get_local_indices(0, 0)
    indices1 = layout.get_local_indices(1, 0)

    assert jnp.array_equal(indices0, jnp.arange(0, 50, dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.arange(50, 100, dtype=jnp.int32))


def test_chunk_layout_multidim():
    """chunk on dim=1 of [4, 10] -> 2 devices."""
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P(None, "x"))
    layout = spec._to_layout(data_shape=(4, 10))

    assert layout.partitions[1] == ("x",)

    indices0 = layout.get_local_indices(0, 1)
    indices1 = layout.get_local_indices(1, 1)

    assert jnp.array_equal(indices0, jnp.arange(0, 5, dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.arange(5, 10, dtype=jnp.int32))


def test_chunk_layout_ragged():
    """chunk on ragged tensor (topology = splits)."""
    splits = [jnp.array([0, 3, 5], dtype=jnp.int32)]
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(2,), topology=RaggedTopology(splits=splits))

    indices0 = layout.get_local_indices(0, 0)
    indices1 = layout.get_local_indices(1, 0)

    assert jnp.array_equal(indices0, jnp.array([0], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1], dtype=jnp.int32))


# ---- cyclic ----


def test_cyclic_layout():
    """cyclic on [10] -> 2 devices."""
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.cyclic(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(10,))

    indices0 = layout.get_local_indices(0, 0)
    indices1 = layout.get_local_indices(1, 0)

    assert jnp.array_equal(indices0, jnp.arange(0, 10, 2, dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.arange(1, 10, 2, dtype=jnp.int32))


# ---- hash ----


def test_hash_layout_with_topology():
    """hash using ragged splits as hash seed."""
    splits = [jnp.array([0, 10, 20], dtype=jnp.int32)]
    mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(2,), topology=RaggedTopology(splits=splits))

    indices0 = layout.get_local_indices(0, 0)
    indices1 = layout.get_local_indices(1, 0)
    indices2 = layout.get_local_indices(2, 0)

    assert jnp.array_equal(indices0, jnp.array([0], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1], dtype=jnp.int32))
    assert tuple(indices2.tolist()) == ()
    assert layout.is_static is False
    assert layout.topology_morphology == TopologyMorphology.RAGGED


def test_hash_layout_with_routing_indices():
    """hash using external routing_indices indices."""
    dep = jnp.array([5, 3, 7, 2], dtype=jnp.int32)
    mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(4,), topology=IndexedTopology(indices=dep))

    indices0 = layout.get_local_indices(0, 0)
    indices1 = layout.get_local_indices(1, 0)
    indices2 = layout.get_local_indices(2, 0)

    assert jnp.array_equal(indices0, jnp.array([1], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([2], dtype=jnp.int32))
    assert jnp.array_equal(indices2, jnp.array([0, 3], dtype=jnp.int32))


def test_hash_layout_with_custom_function():
    """hash using custom hash function via topology."""

    def custom_hash(indices):
        return indices * 2

    mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(
        data_shape=(4,), topology=CallableTopology(hash_fn=custom_hash)
    )

    indices0 = layout.get_local_indices(0, 0)
    indices1 = layout.get_local_indices(1, 0)
    indices2 = layout.get_local_indices(2, 0)

    assert jnp.array_equal(indices0, jnp.array([0, 3], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([2], dtype=jnp.int32))
    assert jnp.array_equal(indices2, jnp.array([1], dtype=jnp.int32))


def test_hash_layout_default():
    """hash with no topology or routing_indices - hash indices directly."""
    mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(6,))

    indices0 = layout.get_local_indices(0, 0)
    indices1 = layout.get_local_indices(1, 0)
    indices2 = layout.get_local_indices(2, 0)

    assert jnp.array_equal(indices0, jnp.array([0, 3], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1, 4], dtype=jnp.int32))
    assert jnp.array_equal(indices2, jnp.array([2, 5], dtype=jnp.int32))


# ---- greedy ----


def test_greedy_layout():
    """greedy load-balanced on ragged tensor."""
    splits = [jnp.array([0, 10, 30, 40], dtype=jnp.int32)]
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.greedy(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(3,), topology=RaggedTopology(splits=splits))

    indices0 = layout.get_local_indices(0, 0)
    indices1 = layout.get_local_indices(1, 0)

    assert jnp.array_equal(indices0, jnp.array([0, 2], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1], dtype=jnp.int32))


# ---- follow (routing_indices) ----


def test_follow_layout():
    """follow another layout's device_mesh."""
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    src_spec = ShardingSpec.hash(mesh=mesh, partition=P("x"))
    src_layout = src_spec._to_layout(
        data_shape=(4,),
        topology=IndexedTopology(indices=jnp.array([0, 1, 0, 1], dtype=jnp.int32)),
    )

    follow_spec = ShardingSpec.follow(src_layout, dim=0)
    follow_layout = follow_spec._to_layout(data_shape=(4,))

    indices0 = follow_layout.get_local_indices(0, 0)
    indices1 = follow_layout.get_local_indices(1, 0)

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
        driver_topology=IndexedTopology(indices=key_data),
        follow_dim=0,
    )

    layout = val_spec._to_layout(data_shape=(4, 16))

    indices0 = layout.get_local_indices(0, 0)
    indices1 = layout.get_local_indices(1, 0)

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

    assert jnp.array_equal(layout.get_local_indices(0, 0), chunk0)
    assert jnp.array_equal(layout.get_local_indices(1, 0), chunk0)
    assert jnp.array_equal(layout.get_local_indices(2, 0), chunk1)
    assert jnp.array_equal(layout.get_local_indices(3, 0), chunk1)

    assert layout.replica_axes == ["y"]
    assert layout.replica_size == 2


# ---- multi-axis sharding ----


def test_multi_axis_chunk_2x3():
    """Chunk on mesh (x=2, y=3) with P(('x','y')) → 6-way split."""
    mesh = DeviceMesh(device_shape=(2, 3), axis_names=("x", "y"))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P(("x", "y")))
    layout = spec._to_layout(data_shape=(12,))

    for r in range(6):
        indices = layout.get_local_indices(r, 0)
        expected = jnp.arange(r * 2, r * 2 + 2, dtype=jnp.int32)
        assert jnp.array_equal(indices, expected), f"Rank {r}: {indices} != {expected}"


def test_multi_axis_replication():
    """Mesh (x=2, y=2, z=3), shard on ('x','y') → z is replicated (factor 3)."""
    mesh = DeviceMesh(device_shape=(2, 2, 3), axis_names=("x", "y", "z"))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P(("x", "y")))
    layout = spec._to_layout(data_shape=(8,))

    assert layout.replica_axes == ["z"]
    assert layout.replica_size == 3

    chunk0 = jnp.arange(0, 2, dtype=jnp.int32)
    for r in [0, 1, 2]:
        assert jnp.array_equal(layout.get_local_indices(r, 0), chunk0), f"Rank {r}"

    chunk1 = jnp.arange(2, 4, dtype=jnp.int32)
    for r in [3, 4, 5]:
        assert jnp.array_equal(layout.get_local_indices(r, 0), chunk1), f"Rank {r}"


def test_multi_axis_cyclic():
    """Cyclic on mesh (x=2, y=2) with P(('x','y')) → 4-way cyclic."""
    mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    spec = ShardingSpec.cyclic(mesh=mesh, partition=P(("x", "y")))
    layout = spec._to_layout(data_shape=(8,))

    assert jnp.array_equal(
        layout.get_local_indices(0, 0), jnp.array([0, 4], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        layout.get_local_indices(1, 0), jnp.array([1, 5], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        layout.get_local_indices(2, 0), jnp.array([2, 6], dtype=jnp.int32)
    )
    assert jnp.array_equal(
        layout.get_local_indices(3, 0), jnp.array([3, 7], dtype=jnp.int32)
    )


# ---- repr ----


def test_sharding_spec_repr():
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P("x"))
    r = repr(spec)
    assert "ChunkPartition" in r
    assert "P(" in r


# ---- new corner cases ----


def test_non_divisible_shard_sizes():
    """Verify non-divisible sizes: 10 items chunked over 3 devices."""
    mesh = DeviceMesh(device_shape=(3,), axis_names=("x",))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(10,))

    # chunk size = max((10 + 3 - 1) // 3, 1) = 4
    # r0: 0:4, r1: 4:8, r2: 8:10
    assert jnp.array_equal(
        layout.get_local_indices(0, 0), jnp.arange(0, 4, dtype=jnp.int32)
    )
    assert jnp.array_equal(
        layout.get_local_indices(1, 0), jnp.arange(4, 8, dtype=jnp.int32)
    )
    assert jnp.array_equal(
        layout.get_local_indices(2, 0), jnp.arange(8, 10, dtype=jnp.int32)
    )


def test_zero_sized_outputs():
    """Verify chunk handles dim_sizes or chunk_sizes that result in 0 workload."""
    mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P("x"))
    # Size 2 over 4 devices -> chunk_size = 1. Remaining 2 devices get empty.
    layout = spec._to_layout(data_shape=(2,))

    assert jnp.array_equal(
        layout.get_local_indices(0, 0), jnp.arange(0, 1, dtype=jnp.int32)
    )
    assert jnp.array_equal(
        layout.get_local_indices(1, 0), jnp.arange(1, 2, dtype=jnp.int32)
    )
    assert len(layout.get_local_indices(2, 0)) == 0
    assert len(layout.get_local_indices(3, 0)) == 0


def test_mixed_static_dynamic_strategies():
    """Combining dynamic hash(routing_dim=0) with implicitly static layouts."""
    mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    # Dynamic routing on dim 0 along x, static replication on y.
    spec = ShardingSpec.hash(mesh=mesh, partition=P("x"), routing_dim=0)
    layout = spec._to_layout(data_shape=(6, 128))

    # rank = 0: coords(0, 0), dim 0 routing = 0 (hash)
    indices0_0 = layout.get_local_indices(0, 0)  # dim 0
    indices0_1 = layout.get_local_indices(0, 1)  # dim 1

    assert jnp.array_equal(indices0_0, jnp.array([0, 2, 4], dtype=jnp.int32))
    assert jnp.array_equal(indices0_1, jnp.arange(0, 128, dtype=jnp.int32))


def test_scalar_tensor_empty_partitions():
    """Complete replication for empty partitioning `P()`."""
    mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P())
    layout = spec._to_layout(data_shape=(10, 10))

    assert layout.replica_size == 4
    assert layout.replica_axes == ["x", "y"]

    assert jnp.array_equal(
        layout.get_local_indices(0, 0), jnp.arange(0, 10, dtype=jnp.int32)
    )
    assert jnp.array_equal(
        layout.get_local_indices(3, 1), jnp.arange(0, 10, dtype=jnp.int32)
    )


def test_redundant_replenishment():
    """P('x') on a 3D mesh ('x', 'y', 'z') -> replica_size = y * z."""
    mesh = DeviceMesh(device_shape=(2, 3, 4), axis_names=("x", "y", "z"))
    spec = ShardingSpec.chunk(mesh=mesh, partition=P("x"))
    layout = spec._to_layout(data_shape=(8,))

    assert layout.replica_size == 12
    assert layout.replica_axes == ["y", "z"]
