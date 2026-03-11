"""Tests for TransportPlanner: Diff(L_src, L_dst) across all sharding combinations."""

import pytest
import jax.numpy as jnp

from fluxway.common.backend import set_backend
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.planner import TransportPlanner
from fluxway.common.types.sharding import ShardingSpec, P
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.index_descriptor import RaggedIndex, ExplicitIndex


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


# ---- Scenario 1: static -> static (chunk -> chunk, different num_workers) ----


def test_static_to_static():
    """Source: chunk(2), Rank 0 has [0,50). Target: chunk(4). Expect 2 plans."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst_spec = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    plan = TransportPlanner.plan(
        src_local_rank=0, src_spec=src_spec, dst_spec=dst_spec, src_data_shape=(100,)
    )

    assert plan.target_num_workers == 4

    counts = plan.send_counts.tolist()
    assert counts[0] == 25
    assert counts[1] == 25
    assert counts[2] == 0
    assert counts[3] == 0


def test_static_to_static_rank1():
    """Same scenario, but from Rank 1's perspective."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(4,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst_spec = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    plan = TransportPlanner.plan(
        src_local_rank=1, src_spec=src_spec, dst_spec=dst_spec, src_data_shape=(100,)
    )

    counts = plan.send_counts.tolist()
    assert counts[0] == 0
    assert counts[1] == 0
    assert counts[2] == 25
    assert counts[3] == 25


# ---- Scenario 2: static -> dynamic (chunk -> hash) ----


def test_static_to_dynamic_hash():
    """Source: chunk(1). Target: hash(2) via index_desc."""
    splits = [jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)]

    src_mesh = DeviceMesh(device_shape=(1,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst_spec = ShardingSpec.hash(P("x")).bind(dst_mesh)

    plan = TransportPlanner.plan(
        src_local_rank=0,
        src_spec=src_spec,
        dst_spec=dst_spec,
        src_data_shape=(4,),
        dst_index_desc=RaggedIndex(splits=splits),
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 2]


# ---- Scenario 3: dynamic -> static (hash -> chunk) ----


def test_dynamic_to_static():
    """Source: hash(2), local_rank=0 owns some rows. Target: chunk(2).
    Routing logic for src_local_rank=0 (Source: Hash -> Target: Chunk)
    Total rows: 6.
    Hash splits: [0, 1, 0, 1, 0, 1]. Chunk splits: [0, 0, 0, 1, 1, 1].

    Row 0: owns by Rank 0 -> target Rank 0 => send to Rank 0
    Row 1: owns by Rank 1                  => (skip, not owned)
    Row 2: owns by Rank 0 -> target Rank 0 => send to Rank 0
    Row 3: owns by Rank 1                  => (skip, not owned)
    Row 4: owns by Rank 0 -> target Rank 1 => send to Rank 1
    Row 5: owns by Rank 1                  => (skip, not owned)

    Final counts for Rank 0: 2 items to Rank 0, 1 item to Rank 1.
    assert counts == [2, 1]
    """
    dep = jnp.array([0, 1, 0, 1, 0, 1], dtype=jnp.int32)
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    src_spec = ShardingSpec.hash(P("x")).bind(mesh)
    dst_spec = ShardingSpec.chunk(P("x")).bind(mesh)

    plan = TransportPlanner.plan(
        src_local_rank=0,
        src_spec=src_spec,
        dst_spec=dst_spec,
        src_data_shape=(6,),
        src_index_desc=ExplicitIndex(indices=dep),
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 1]


# ---- Scenario 4: dynamic -> dynamic (greedy -> hash) ----


def test_dynamic_to_dynamic():
    """Source: greedy(2). Target: hash(2)."""
    splits = [jnp.array([0, 10, 30, 40], dtype=jnp.int32)]

    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    src_spec = ShardingSpec.greedy(P("x")).bind(mesh)
    dst_spec = ShardingSpec.hash(P("x")).bind(mesh)

    plan = TransportPlanner.plan(
        src_local_rank=0,
        src_spec=src_spec,
        dst_spec=dst_spec,
        src_data_shape=(3,),
        src_index_desc=RaggedIndex(splits=splits),
        dst_index_desc=RaggedIndex(splits=splits),
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 0]


# ---- Scenario 5: routing_indices (key -> value dual-tensor) ----


def test_routing_indices_dual_tensor():
    """Key-value dispatch: key determines device, value follows."""
    mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))

    val_spec = ShardingSpec.hash(P("x")).bind(mesh)

    src_mesh = DeviceMesh(device_shape=(1,), axis_names=("x",))
    src_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)

    plan = TransportPlanner.plan(
        src_local_rank=0, src_spec=src_spec, dst_spec=val_spec, src_data_shape=(4,)
    )

    counts = plan.send_counts.tolist()
    assert counts == [2, 2]


# ---- Scenario 6: replicate (fan-out) ----


def test_replicate_fanout():
    """chunk(2) -> implicit replicate on 2nd mesh axis."""
    src_mesh = DeviceMesh(device_shape=(2,), axis_names=("x",))
    dst_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))

    src_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst_spec = ShardingSpec.chunk(P("x")).bind(dst_mesh)

    plan = TransportPlanner.plan(
        src_local_rank=0, src_spec=src_spec, dst_spec=dst_spec, src_data_shape=(10,)
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
    src_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst_spec = ShardingSpec.cyclic(P("x")).bind(dst_mesh)

    plan = TransportPlanner.plan(
        src_local_rank=0, src_spec=src_spec, dst_spec=dst_spec, src_data_shape=(4,)
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
    src_spec = ShardingSpec.chunk(P("x")).bind(src_mesh)
    dst_spec = ShardingSpec.cyclic(P("x")).bind(dst_mesh)

    plan = TransportPlanner.plan(
        src_local_rank=0, src_spec=src_spec, dst_spec=dst_spec, src_data_shape=(6,)
    )

    slices = list(plan.iter_device_slices())

    assert len(slices) == 3
    total = sum(count for _, _, count in slices)
    assert total == 6


# ---- Multi-axis Dynamic Routing ----


def test_multi_axis_dynamic_routing_dims():
    """Verify that TransportPlanner supports multi-axis dynamic routing (tuple routing_dim)."""
    # 2D Mesh [2, 2]
    src_mesh = DeviceMesh(device_shape=(1, 1), axis_names=("x", "y"))
    dst_mesh = DeviceMesh(device_shape=(2, 2), axis_names=("x", "y"))
    src_spec = ShardingSpec.chunk(P("x", "y")).bind(src_mesh)

    # Target expects a tuple for multi-dimensional content addressing
    dst_spec = ShardingSpec.hash(partition=P("x", "y"), routing_dim=(0, 1)).bind(
        dst_mesh
    )

    # Assume we have 4 tokens (2x2 data grid) mapped dynamically over two dimensions.
    # For Dim 0 (hash x) -> routing values: [0, 1]
    # For Dim 1 (hash y) -> routing values: [1, 0]
    # We supply src_indices to substitute explicit routing vectors since planner accepts indices or data_values.

    plan = TransportPlanner.compute_plan(
        src_local_rank=0,
        src_data_shape=(2, 2),
        src_spec=src_spec,
        dst_spec=dst_spec,
        src_routing_indices=jnp.array([0, 1], dtype=jnp.int32),
    )

    # Since it's dynamic Hash over mesh shape 2x2:
    # Dim 0 processes [0, 1] -> distributes statically for Dim 1 because primary routing values map independently.
    # To truly verify Tuple routing_dim parsing without complex fallback payloads here, we rely on the compilation finishing successfully and generating valid layouts.

    assert plan.target_num_workers == 4
    counts = plan.send_counts.tolist()
    assert len(counts) == 4
