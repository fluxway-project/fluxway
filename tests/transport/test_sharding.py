"""Tests for algebraic sharding: Layout + ShardingSpec + partition functions."""

import pytest
import jax.numpy as jnp

from fluxway.transport.backend import set_backend
from fluxway.transport.sharding import ShardingSpec


# Force JAX backend
@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


@pytest.fixture(autouse=False)
def use_torch_backend():
    set_backend("torch")
    yield


# ---- chunk ----


def test_chunk_layout_dense():
    """chunk on dense [100] -> 2 devices."""
    spec = ShardingSpec.chunk(dim=0)
    layout = spec.compute_layout(data_shape=(100,), world_size=2)

    assert layout.partition_dim == 0
    assert layout.world_size == 2
    assert layout.is_static is True
    assert layout.topology_kind == "dense"

    # Validating using get_local_indices
    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    # 0..49 -> Rank 0
    assert jnp.array_equal(indices0, jnp.arange(0, 50, dtype=jnp.int32))
    # 50..99 -> Rank 1
    assert jnp.array_equal(indices1, jnp.arange(50, 100, dtype=jnp.int32))


def test_chunk_layout_multidim():
    """chunk on dim=1 of [4, 10] -> 2 devices."""
    spec = ShardingSpec.chunk(dim=1)
    layout = spec.compute_layout(data_shape=(4, 10), world_size=2)

    assert layout.partition_dim == 1
    # Rank 0 should have 0..4 in dim 0 ? No, dim=1.
    # Shape (4, 10). Partition dim 1.
    # Rank 0 gets indices [0..4] in dim 1.
    # Current get_local_indices returns indices along partition_dim.
    # It returns [0, 1, 2, 3, 4] for rank 0.

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.arange(0, 5, dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.arange(5, 10, dtype=jnp.int32))


def test_chunk_layout_ragged():
    """chunk on ragged tensor (topology_meta = splits)."""
    splits = [jnp.array([0, 3, 5], dtype=jnp.int32)]
    spec = ShardingSpec.chunk(dim=0)
    layout = spec.compute_layout(data_shape=(2,), world_size=2, topology_meta=splits)

    # Ragged splits: [0, 3, 5] -> lengths [3, 2]
    # Rank 0 gets row 0 (length 3). Rank 1 gets row 1 (length 2).
    # Since we partition on dim 0, Rank 0 gets index 0, Rank 1 gets index 1.

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.array([0], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1], dtype=jnp.int32))


# ---- cyclic ----


def test_cyclic_layout():
    """cyclic on [10] -> 2 devices."""
    spec = ShardingSpec.cyclic(dim=0)
    layout = spec.compute_layout(data_shape=(10,), world_size=2)

    # [0, 2, 4, 6, 8] -> Rank 0
    # [1, 3, 5, 7, 9] -> Rank 1

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    expected0 = jnp.arange(0, 10, 2, dtype=jnp.int32)
    expected1 = jnp.arange(1, 10, 2, dtype=jnp.int32)

    assert jnp.array_equal(indices0, expected0)
    assert jnp.array_equal(indices1, expected1)


# ---- hash ----


def test_hash_layout_with_topology():
    """hash using ragged splits as hash seed."""
    splits = [jnp.array([0, 10, 20], dtype=jnp.int32)]
    spec = ShardingSpec.hash(dim=0)
    layout = spec.compute_layout(data_shape=(2,), world_size=3, topology_meta=splits)

    # splits: [0, 10, 20]. Starts: 0, 10.
    # 0 % 3 = 0 -> Rank 0
    # 10 % 3 = 1 -> Rank 1

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)
    indices2 = layout.get_local_indices(2)

    assert jnp.array_equal(indices0, jnp.array([0], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1], dtype=jnp.int32))
    assert len(indices2) == 0
    assert layout.is_static is False


def test_hash_layout_with_dependency():
    """hash using external dependency indices."""
    dep = jnp.array([5, 3, 7, 2], dtype=jnp.int32)
    spec = ShardingSpec.hash(dim=0)
    layout = spec.compute_layout(data_shape=(4,), world_size=3, dependency=dep)

    # dep: [5, 3, 7, 2]
    # 5%3=2 -> Rank 2 (index 0)
    # 3%3=0 -> Rank 0 (index 1)
    # 7%3=1 -> Rank 1 (index 2)
    # 2%3=2 -> Rank 2 (index 3)

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)
    indices2 = layout.get_local_indices(2)

    assert jnp.array_equal(indices0, jnp.array([1], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([2], dtype=jnp.int32))
    assert jnp.array_equal(indices2, jnp.array([0, 3], dtype=jnp.int32))


def test_hash_layout_with_custom_function():
    """hash using custom hash function via topology_meta."""

    # Custom hash: double the index before hashing
    def custom_hash(indices):
        return indices * 2

    spec = ShardingSpec.hash(dim=0)
    layout = spec.compute_layout(
        data_shape=(4,), world_size=3, topology_meta={"hash_fn": custom_hash}
    )

    # indices: [0, 1, 2, 3] -> *2 -> [0, 2, 4, 6] -> %3
    # 0%3=0 -> Rank 0 (idx 0)
    # 2%3=2 -> Rank 2 (idx 1)
    # 4%3=1 -> Rank 1 (idx 2)
    # 6%3=0 -> Rank 0 (idx 3)

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)
    indices2 = layout.get_local_indices(2)

    assert jnp.array_equal(indices0, jnp.array([0, 3], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([2], dtype=jnp.int32))
    assert jnp.array_equal(indices2, jnp.array([1], dtype=jnp.int32))


def test_hash_layout_default():
    """hash with no topology_meta or dependency - hash indices directly."""
    spec = ShardingSpec.hash(dim=0)
    layout = spec.compute_layout(data_shape=(6,), world_size=3)

    # indices: [0, 1, 2, 3, 4, 5] -> %3
    # 0,3 -> Rank 0
    # 1,4 -> Rank 1
    # 2,5 -> Rank 2

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
    spec = ShardingSpec.greedy(dim=0)
    layout = spec.compute_layout(data_shape=(3,), world_size=2, topology_meta=splits)

    # Lengths: [10, 20, 10]. Greedy:
    # row 0 (10) -> rank 0 (load 10, 0)
    # row 1 (20) -> rank 1 (load 10, 20)
    # row 2 (10) -> rank 0 (load 20, 20)
    # Greedy:
    # Row 0 (len 10) -> Rank 0 (load 10) -> idx 0
    # Row 1 (len 20) -> Rank 1 (load 20) -> idx 1
    # Row 2 (len 10) -> Rank 0 (load 20) -> idx 2

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.array([0, 2], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1], dtype=jnp.int32))


# ---- follow (dependency) ----


def test_follow_layout():
    """follow another layout's device_mesh."""
    # Source layout: hash computed
    src_spec = ShardingSpec.hash(dim=0)
    src_layout = src_spec.compute_layout(
        data_shape=(4,),
        world_size=2,
        dependency=jnp.array([0, 1, 0, 1], dtype=jnp.int32),
    )

    follow_spec = ShardingSpec.follow(src_layout, dim=0)
    follow_layout = follow_spec.compute_layout(data_shape=(4,), world_size=2)

    # Src: [0, 1, 0, 1] dependency
    # Rank 0: idx 0, 2
    # Rank 1: idx 1, 3

    indices0 = follow_layout.get_local_indices(0)
    indices1 = follow_layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.array([0, 2], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1, 3], dtype=jnp.int32))


# ---- derive_from (convenience) ----


def test_derive_from():
    """derive_from: one-line key->value dependency."""
    # key: [4], hash -> [0, 1, 0, 1]
    key_data = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    val_spec = ShardingSpec.derive_from(
        driver_spec=ShardingSpec.hash(dim=0),
        driver_data_shape=(4,),
        world_size=2,
        driver_topology=key_data,
        follow_dim=0,
    )

    layout = val_spec.compute_layout(data_shape=(4, 16), world_size=2)
    # key: [0, 1, 2, 3] -hash-> [0, 1, 0, 1]
    # Rank 0: 0, 2
    # Rank 1: 1, 3

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)

    assert jnp.array_equal(indices0, jnp.array([0, 2], dtype=jnp.int32))
    assert jnp.array_equal(indices1, jnp.array([1, 3], dtype=jnp.int32))


# ---- replicated ----


def test_implicit_replication():
    """Sharding on one axis implies replication on others."""
    from fluxway.transport.sharding import DeviceMesh

    # Mesh: (x=2, y=2)
    mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))

    # Shard on "x" -> "y" is replicated
    spec = ShardingSpec.chunk(dim=0)
    spec.mesh = mesh
    spec.sharding_axis = "x"

    layout = spec.compute_layout(data_shape=(10,), world_size=4)

    # device_mesh should reflect sharding on X
    # X=0 (ranks 0,1) get chunk 0?
    # Wait, layout.device_mesh is PRIMARY devices (y=0).
    # ranks: 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1)
    # Shard on X. chunk(2).
    # chunk 0 -> X=0. Primary is (0,0) -> rank 0.
    # chunk 1 -> X=1. Primary is (1,0) -> rank 2.

    # DM shape (2,) for 2 chunks.
    # index 0 -> 0. index 1 -> 2.
    # 10 elements. chunk size 5. [0..4] -> 0. [5..9] -> 2.

    # Check replication via get_local_indices for ALL physical ranks
    # Rank 0 (0,0) -> Chunk 0 [0..4]
    # Rank 1 (0,1) -> Chunk 0 [0..4] (Replicated)
    # Rank 2 (1,0) -> Chunk 1 [5..9]
    # Rank 3 (1,1) -> Chunk 1 [5..9] (Replicated)

    indices0 = layout.get_local_indices(0)
    indices1 = layout.get_local_indices(1)
    indices2 = layout.get_local_indices(2)
    indices3 = layout.get_local_indices(3)

    chunk0 = jnp.arange(0, 5, dtype=jnp.int32)
    chunk1 = jnp.arange(5, 10, dtype=jnp.int32)

    assert jnp.array_equal(indices0, chunk0)
    assert jnp.array_equal(indices1, chunk0)
    assert jnp.array_equal(indices2, chunk1)
    assert jnp.array_equal(indices3, chunk1)

    # Check replication metadata
    # replication_axis should be ["y"]
    assert layout.replication_axes == ["y"]
    assert layout.replication_factor == 2


# ---- multi-axis sharding ----


def test_multi_axis_chunk_2x3():
    """Chunk on mesh (x=2, y=3) with sharding_axis=("x","y") → 6-way split."""
    from fluxway.transport.sharding import DeviceMesh

    mesh = DeviceMesh(device_shape=(2, 3), axis_names=("x", "y"))

    spec = ShardingSpec.chunk(dim=0, mesh=mesh, axis=("x", "y"))
    layout = spec.compute_layout(data_shape=(12,), world_size=6)

    # effective_world_size = 2*3 = 6, chunk_size = 2
    # rank → coord → effective_rank → indices
    # rank 0 (0,0) → eff 0 → [0,1]
    # rank 1 (0,1) → eff 1 → [2,3]
    # rank 2 (0,2) → eff 2 → [4,5]
    # rank 3 (1,0) → eff 3 → [6,7]
    # rank 4 (1,1) → eff 4 → [8,9]
    # rank 5 (1,2) → eff 5 → [10,11]

    for r in range(6):
        indices = layout.get_local_indices(r)
        expected = jnp.arange(r * 2, r * 2 + 2, dtype=jnp.int32)
        assert jnp.array_equal(indices, expected), f"Rank {r}: {indices} != {expected}"


def test_multi_axis_replication():
    """Mesh (x=2, y=2, z=3), shard on ("x","y") → z is replicated (factor 3)."""
    from fluxway.transport.sharding import DeviceMesh

    mesh = DeviceMesh(device_shape=(2, 2, 3), axis_names=("x", "y", "z"))

    spec = ShardingSpec.chunk(dim=0, mesh=mesh, axis=("x", "y"))
    layout = spec.compute_layout(data_shape=(8,), world_size=12)

    # effective_world_size = 2*2 = 4, chunk_size = 2
    # Replication axis = ["z"], factor = 3

    assert layout.replication_axes == ["z"]
    assert layout.replication_factor == 3

    # Ranks sharing same (x,y) but different z should get identical indices
    # rank 0 (0,0,0), rank 1 (0,0,1), rank 2 (0,0,2) → all eff_rank 0 → [0,1]
    chunk0 = jnp.arange(0, 2, dtype=jnp.int32)
    for r in [0, 1, 2]:
        assert jnp.array_equal(layout.get_local_indices(r), chunk0), f"Rank {r}"

    # rank 3 (0,1,0), rank 4 (0,1,1), rank 5 (0,1,2) → eff_rank 1 → [2,3]
    chunk1 = jnp.arange(2, 4, dtype=jnp.int32)
    for r in [3, 4, 5]:
        assert jnp.array_equal(layout.get_local_indices(r), chunk1), f"Rank {r}"


def test_multi_axis_cyclic():
    """Cyclic on mesh (x=2, y=2) with sharding_axis=("x","y") → 4-way cyclic."""
    from fluxway.transport.sharding import DeviceMesh

    mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))

    spec = ShardingSpec.cyclic(dim=0, mesh=mesh, axis=("x", "y"))
    layout = spec.compute_layout(data_shape=(8,), world_size=4)

    # effective_world_size = 4, cyclic
    # rank 0 (0,0) eff 0 → [0,4]
    # rank 1 (0,1) eff 1 → [1,5]
    # rank 2 (1,0) eff 2 → [2,6]
    # rank 3 (1,1) eff 3 → [3,7]

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
    spec = ShardingSpec.chunk(dim=0)
    r = repr(spec)
    assert "chunk_partition" in r
    assert "dim=0" in r
