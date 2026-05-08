import os
import jax
import jax.dlpack
import jax.numpy as jnp

import sparse_embedding as se

from fluxway.common.decorators import spmd, operator
from fluxway.common.utils.logging import get_logger
from fluxway.common.types.tensor import TransportTensor
from fluxway.example.recommendation.data_types import LookupOutput

logger = get_logger(__name__)


@spmd(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("shard",),
    mesh_axis_shapes=(4,),
)
class SparseEmbeddingSPMD:
    """Sharded embedding table backed by sparse_embedding.EmbeddingStore.

    Each shard owns the embeddings for keys hash-routed to it.  Forward lookup
    uses EmbeddingStore.lookup_or_insert (lazy initialisation on first access),
    and the backward pass accumulates gradients with Adam then calls
    optimizer_step once per training step.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 3,
        num_threads: int | None = None,
    ):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_threads = num_threads or os.cpu_count() or 4

        self.store = se.EmbeddingStore(
            embedding_config=se.EmbeddingConfig(
                slot_dims={0: embed_dim},
                num_threads=self.num_threads,
                dtype="float32",
            ),
            initializer_config=se.Uniform(min_val=-0.01, max_val=0.01),
            optimizer_config=se.Adam(lr=0.01),
        )
        self._forward_keys_cache: dict[int, jnp.ndarray] = {}

        logger.debug(
            "SparseEmbeddingSPMD_init rank=%s world_size=%s vocab_size=%s embed_dim=%s",
            getattr(self, "rank", None),
            getattr(self, "world_size", None),
            vocab_size,
            embed_dim,
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_state(self, ckpt_dir: str) -> dict:
        uri = f"{ckpt_dir}/sparse_shard_{self.rank}"
        self.store.save_checkpoint(
            uri=uri,
            mode="full",
            min_score=0,
            include_optimizer_state=True,
            use_compression=False,
        )
        return {"uri": uri}

    def restore_state(self, state: dict, ckpt_dir: str) -> None:
        uri = state.get("uri", f"{ckpt_dir}/sparse_shard_{self.rank}")
        self.store.load_checkpoint(
            uri=uri,
            version=0,
            clear=True,
            load_optimizer_state=True,
        )

    # ------------------------------------------------------------------
    # Forward: lookup
    # ------------------------------------------------------------------

    @operator(
        name="lookup",
        data_argument=["unique_ids"],
        input_sharding={
            "unique_ids": {"type": "dynamic", "axis": ["shard"], "fn": "hash_routing"}
        },
        output_sharding={"sparse_features": {"type": "dynamic", "axis": ["shard"]}},
    )
    def lookup(self, unique_ids: TransportTensor) -> LookupOutput:
        step = self.ctx.step
        ids_jax = unique_ids.flat_values.ravel().astype(jnp.int64)

        # Lazy-insert new keys, return embeddings for all keys in this shard.
        vals_dlm, _, _ = self.store.lookup_or_insert(ids_jax)
        emb_jax = jax.dlpack.from_dlpack(vals_dlm).reshape(ids_jax.size, self.embed_dim)

        self._forward_keys_cache[step] = ids_jax

        g_shape = None
        if unique_ids.global_shape and len(unique_ids.global_shape) >= 1:
            g_shape = (unique_ids.global_shape[0], self.embed_dim)

        output = TransportTensor(
            emb_jax,
            origin_indices=unique_ids.origin_indices,
            global_shape=g_shape,
            receipt=unique_ids.receipt,
        )
        print(
            f"[SparseEmbeddingSPMD.lookup] rank={self.rank} step={step} "
            f"keys={ids_jax} emb_norm={float(jnp.linalg.norm(emb_jax)):.4f}"
        )
        return LookupOutput(sparse_features=output)

    # ------------------------------------------------------------------
    # Backward: accumulate gradients and take an optimizer step
    # ------------------------------------------------------------------

    @operator(
        name="update",
        data_argument=["grads"],
        input_sharding={
            "grads": {"type": "mirror", "reference": "lookup.sparse_features"}
        },
    )
    def update(self, grads: TransportTensor) -> None:
        step = self.ctx.step
        keys_jax = self._forward_keys_cache.pop(step, None)

        if keys_jax is None:
            raise ValueError(
                f"SparseEmbeddingSPMD rank={self.rank}: "
                f"missing forward cache for step {step}"
            )

        grads_jax = grads.flat_values.reshape(keys_jax.size, self.embed_dim)

        self.store.accumulate_gradients(keys_jax, grads_jax)
        self.store.optimizer_step()

        print(
            f"[SparseEmbeddingSPMD.update] rank={self.rank} step={step} "
            f"grad_norm={float(jnp.linalg.norm(grads_jax)):.4f}"
        )


def hash_routing(tensor, mesh):
    if isinstance(tensor, TransportTensor) and tensor.routing_indices is not None:
        val = tensor.routing_indices
    else:
        val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    return jnp.mod(val, mesh.size)
