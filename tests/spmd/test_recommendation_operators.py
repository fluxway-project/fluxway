"""
Tests for the recommendation-style SPMD operator pattern:
  - dedup + inverse index (InputSPMD pattern)
  - sharded embedding lookup + update (SparseEmbeddingSPMD pattern)
  - dense forward/backward with scatter-add gradient (DenseSPMD pattern)

All SPMD classes and data_types are defined inline; this file has no dependency
on fluxway.example.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from fluxway.common.backend import set_backend
from fluxway.common.decorators import data_type, operator, operator_context, spmd
from fluxway.common.types.tensor import TransportTensor


# ---------------------------------------------------------------------------
# Inline data types (mirrors the structure of example/recommendation/data_types.py
# without importing from it)
# ---------------------------------------------------------------------------


@data_type()
@dataclass
class Item:
    item_id: TransportTensor
    item_stats: TransportTensor


@data_type()
@dataclass
class Batch:
    user_id: TransportTensor
    item: Item
    label: TransportTensor


@data_type()
@dataclass
class Deduped:
    unique_users: TransportTensor
    user_inv: TransportTensor
    unique_items: TransportTensor
    item_inv: TransportTensor
    stats: TransportTensor
    label: TransportTensor


@data_type()
@dataclass
class EmbOutput:
    embeddings: TransportTensor


# ---------------------------------------------------------------------------
# Inline SPMD operators
# ---------------------------------------------------------------------------


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class InputOp:
    @operator(
        name="process",
        input_sharding={"data": {"type": "static", "axis": ["dp"]}},
        output_sharding={
            "unique_users": {"type": "dynamic", "axis": ["dp"], "fn": "hash_routing"},
            "unique_items": {"type": "dynamic", "axis": ["dp"], "fn": "hash_routing"},
            "user_inv": {"type": "static", "axis": ["dp"]},
            "item_inv": {"type": "static", "axis": ["dp"]},
            "stats": {"type": "static", "axis": ["dp"]},
            "label": {"type": "static", "axis": ["dp"]},
        },
    )
    def process(self, data: Batch) -> Deduped:
        u_users, inv_users = data.user_id.unique_with_routing()
        u_items, inv_items = data.item.item_id.unique_with_routing()
        return Deduped(
            unique_users=u_users,
            user_inv=inv_users,
            unique_items=u_items,
            item_inv=inv_items,
            stats=data.item.item_stats,
            label=data.label,
        )


def hash_routing(tensor, mesh):
    val = (
        tensor.routing_indices
        if tensor.routing_indices is not None
        else tensor.flat_values
    )
    return jnp.mod(val, mesh.size)


@spmd(mesh_axis_names=("shard",), mesh_axis_shapes=(4,))
class EmbeddingOp:
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 4):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.params = jnp.ones((vocab_size, embed_dim)) * 0.01
        self._cache: dict = {}

    @operator(
        name="lookup",
        data_argument=["unique_ids"],
        input_sharding={
            "unique_ids": {"type": "dynamic", "axis": ["shard"], "fn": "hash_routing"}
        },
        output_sharding={"embeddings": {"type": "dynamic", "axis": ["shard"]}},
    )
    def lookup(self, unique_ids: TransportTensor) -> EmbOutput:
        step = self.ctx.step
        ids = unique_ids.flat_values
        self._cache[step] = ids
        local_ids = jnp.mod(ids, self.vocab_size)
        emb = self.params[local_ids] * local_ids[:, None]
        g_shape = None
        if unique_ids.global_shape:
            g_shape = (unique_ids.global_shape[0], self.embed_dim)
        return EmbOutput(
            embeddings=TransportTensor(
                emb, origin_indices=unique_ids.origin_indices, global_shape=g_shape
            )
        )

    @operator(
        name="update",
        data_argument=["grads"],
        input_sharding={"grads": {"type": "mirror", "reference": "lookup.embeddings"}},
    )
    def update(self, grads: TransportTensor) -> None:
        step = self.ctx.step
        ids = self._cache.pop(step)
        local_ids = jnp.mod(ids, self.vocab_size)
        self.params = self.params.at[local_ids].add(-0.1 * grads.flat_values)


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class DenseOp:
    def __init__(self, embed_dim: int = 4, stats_dim: int = 2):
        self.embed_dim = embed_dim
        self.stats_dim = stats_dim
        # Simple linear weight: w shape (embed_dim*2 + stats_dim,)
        self.w = jnp.ones((embed_dim * 2 + stats_dim,)) * 0.1

    @operator(
        name="step",
        input_sharding={
            "emb1": {"type": "static", "axis": ["dp"]},
            "inv1": {"type": "static", "axis": ["dp"]},
            "emb2": {"type": "static", "axis": ["dp"]},
            "inv2": {"type": "static", "axis": ["dp"]},
            "stats": {"type": "static", "axis": ["dp"]},
        },
        output_sharding={
            "grad1": {"type": "mirror", "reference": "step.emb1"},
            "grad2": {"type": "mirror", "reference": "step.emb2"},
        },
    )
    def step(
        self,
        emb1: TransportTensor,
        inv1: TransportTensor,
        emb2: TransportTensor,
        inv2: TransportTensor,
        stats: TransportTensor,
    ) -> dict:
        e1 = emb1.flat_values.reshape(-1, self.embed_dim)
        e2 = emb2.flat_values.reshape(-1, self.embed_dim)
        i1 = inv1.flat_values.reshape(-1)
        i2 = inv2.flat_values.reshape(-1)
        s = stats.flat_values.reshape(-1, self.stats_dim)

        f1 = e1[i1]
        f2 = e2[i2]
        x = jnp.concatenate([f1, f2, s], axis=1)  # (batch, embed_dim*2+stats_dim)

        # MSE gradient of dot(w, x.mean(0)) w.r.t. e1_unique, e2_unique via chain rule
        # grad_x shape: (batch, D), uniform for this simple linear
        grad_x = jnp.ones_like(x) * self.w  # element-wise, pretend loss=sum(x*w)

        grad_f1 = grad_x[:, : self.embed_dim]
        grad_f2 = grad_x[:, self.embed_dim : self.embed_dim * 2]

        # scatter-add back to unique embeddings
        g1 = jnp.zeros_like(e1).at[i1].add(grad_f1)
        g2 = jnp.zeros_like(e2).at[i2].add(grad_f2)

        return {"grad1": g1, "grad2": g2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Ctx:
    def __init__(self, step=0):
        self.step = step


def _ctx(step=0):
    return _Ctx(step=step)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_jax_backend():
    set_backend("jax")


def test_input_op_dedup_and_inv():
    """InputOp.process deduplicates IDs and produces correct inverse indices."""
    batch = Batch(
        user_id=TransportTensor(
            jnp.array([100, 101], dtype=jnp.int32), global_shape=(2,)
        ),
        item=Item(
            item_id=TransportTensor(
                jnp.array([500, 501], dtype=jnp.int32), global_shape=(2,)
            ),
            item_stats=TransportTensor(
                jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32),
                global_shape=(2,),
            ),
        ),
        label=TransportTensor(jnp.array([1, 0], dtype=jnp.int32), global_shape=(2,)),
    )

    with operator_context(static_args={}, rank=0, world_size=1, operator_ctx=_ctx(0)):
        out = InputOp.process(batch)

    assert jnp.array_equal(out.unique_users.flat_values, jnp.array([100, 101]))
    assert jnp.array_equal(out.user_inv.flat_values, jnp.array([0, 1]))
    assert jnp.array_equal(out.unique_items.flat_values, jnp.array([500, 501]))
    assert jnp.array_equal(out.item_inv.flat_values, jnp.array([0, 1]))


def test_input_op_dedup_repeated_ids():
    """Repeated IDs must be deduplicated; inverse must reconstruct originals."""
    batch = Batch(
        user_id=TransportTensor(
            jnp.array([7, 7, 9], dtype=jnp.int32), global_shape=(3,)
        ),
        item=Item(
            item_id=TransportTensor(
                jnp.array([3, 5, 3], dtype=jnp.int32), global_shape=(3,)
            ),
            item_stats=TransportTensor(
                jnp.array([[0.0, 0.0]] * 3, dtype=jnp.float32), global_shape=(3,)
            ),
        ),
        label=TransportTensor(jnp.array([0, 0, 1], dtype=jnp.int32), global_shape=(3,)),
    )

    with operator_context(static_args={}, rank=0, world_size=1, operator_ctx=_ctx(0)):
        out = InputOp.process(batch)

    # unique_users must contain exactly {7, 9}
    assert set(out.unique_users.flat_values.tolist()) == {7, 9}
    # Reconstructed via inv must equal original
    reconstructed_users = out.unique_users.flat_values[out.user_inv.flat_values]
    assert jnp.array_equal(reconstructed_users, jnp.array([7, 7, 9]))

    assert set(out.unique_items.flat_values.tolist()) == {3, 5}
    reconstructed_items = out.unique_items.flat_values[out.item_inv.flat_values]
    assert jnp.array_equal(reconstructed_items, jnp.array([3, 5, 3]))


class _UserTable(EmbeddingOp):
    pass


class _ItemTable(EmbeddingOp):
    pass


def test_embedding_lookup_values():
    """EmbeddingOp.lookup produces expected embedding values based on params and IDs."""
    ids = jnp.array([100, 101], dtype=jnp.int32)
    tt = TransportTensor(ids, global_shape=(2,))

    with operator_context(static_args={}, rank=0, world_size=1, operator_ctx=_ctx(0)):
        out = _UserTable.lookup(tt)

    local_ids = jnp.mod(ids, 1000)
    # params=0.01, emb = params[local_id] * local_id
    expected = (jnp.ones((2, 4)) * 0.01) * local_ids[:, None]
    assert jnp.allclose(out.embeddings.flat_values, expected)


def test_embedding_update_modifies_params():
    """EmbeddingOp.update applies gradient descent; verified by observing changed lookup output."""
    ids = jnp.array([200, 201], dtype=jnp.int32)
    tt = TransportTensor(ids, global_shape=(2,))

    # Step 0 forward: record baseline embeddings
    with operator_context(
        static_args={}, rank=0, world_size=1, operator_ctx=_ctx(step=0)
    ):
        out_before = _ItemTable.lookup(tt)

    emb_before = out_before.embeddings.flat_values

    # Step 0 backward: apply uniform gradient of 0.5
    grads = TransportTensor(jnp.ones((2, 4), dtype=jnp.float32) * 0.5)
    with operator_context(
        static_args={}, rank=0, world_size=1, operator_ctx=_ctx(step=0)
    ):
        _ItemTable.update(grads)

    # Step 1 forward: lookup the same IDs again — embeddings must differ by -lr*grad effect
    tt2 = TransportTensor(ids, global_shape=(2,))
    with operator_context(
        static_args={}, rank=0, world_size=1, operator_ctx=_ctx(step=1)
    ):
        out_after = _ItemTable.lookup(tt2)
        # Clear step-1 cache to avoid polluting update tests
        _ItemTable.update(TransportTensor(jnp.zeros((2, 4), dtype=jnp.float32)))

    emb_after = out_after.embeddings.flat_values

    # params decreased by lr * grad = 0.1 * 0.5 = 0.05 per entry
    # embedding = params[local_id] * local_id, so delta = -0.05 * local_id
    local_ids = jnp.mod(ids, 1000)
    expected_delta = -0.05 * local_ids[:, None] * jnp.ones((1, 4))
    assert jnp.allclose(emb_after - emb_before, expected_delta, atol=1e-5), (
        f"Expected param update delta {expected_delta}, got {emb_after - emb_before}"
    )


def test_dense_op_step_grad_shape():
    """DenseOp.step returns grad1/grad2 with shapes matching the unique embedding inputs."""
    embed_dim = 4
    stats_dim = 2
    n_unique_users = 2
    n_unique_items = 3
    batch = 4

    emb1 = TransportTensor(
        jnp.ones((n_unique_users, embed_dim)), global_shape=(n_unique_users,)
    )
    emb2 = TransportTensor(
        jnp.ones((n_unique_items, embed_dim)), global_shape=(n_unique_items,)
    )
    inv1 = TransportTensor(
        jnp.array([0, 1, 0, 1], dtype=jnp.int32), global_shape=(batch,)
    )
    inv2 = TransportTensor(
        jnp.array([0, 1, 2, 0], dtype=jnp.int32), global_shape=(batch,)
    )
    stats = TransportTensor(jnp.zeros((batch, stats_dim)), global_shape=(batch,))

    with operator_context(static_args={}, rank=0, world_size=1, operator_ctx=_ctx(0)):
        result = DenseOp.step(emb1=emb1, inv1=inv1, emb2=emb2, inv2=inv2, stats=stats)

    assert result["grad1"].shape == (n_unique_users, embed_dim)
    assert result["grad2"].shape == (n_unique_items, embed_dim)


def test_dense_op_step_scatter_add_correctness():
    """DenseOp.step correctly scatter-adds per-sample gradients back to unique indices."""
    embed_dim = 4
    stats_dim = 2

    # user 0 appears in sample 0 and 2 → grad accumulates twice
    # user 1 appears in sample 1 and 3 → grad accumulates twice
    emb1 = TransportTensor(jnp.ones((2, embed_dim)), global_shape=(2,))
    emb2 = TransportTensor(jnp.ones((2, embed_dim)), global_shape=(2,))
    inv1 = TransportTensor(jnp.array([0, 1, 0, 1], dtype=jnp.int32), global_shape=(4,))
    inv2 = TransportTensor(jnp.array([0, 0, 1, 1], dtype=jnp.int32), global_shape=(4,))
    stats = TransportTensor(jnp.zeros((4, stats_dim)), global_shape=(4,))

    with operator_context(static_args={}, rank=0, world_size=1, operator_ctx=_ctx(0)):
        result = DenseOp.step(emb1=emb1, inv1=inv1, emb2=emb2, inv2=inv2, stats=stats)

    # w=0.1 for all dims; each unique user accumulates from 2 samples → grad = 2 * 0.1
    expected_g1 = jnp.ones((2, embed_dim)) * 0.2
    assert jnp.allclose(result["grad1"], expected_g1), (
        f"Expected {expected_g1}, got {result['grad1']}"
    )
    expected_g2 = jnp.ones((2, embed_dim)) * 0.2
    assert jnp.allclose(result["grad2"], expected_g2)
