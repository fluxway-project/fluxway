import pathlib

import numpy as np
import jax.numpy as jnp

from fluxway.common.decorators import spmd, operator
from fluxway.common.utils.logging import get_logger
from fluxway.common.types.tensor import TransportTensor
from .data_types import LookupOutput

logger = get_logger(__name__)


@spmd(resource={"num_cpus": 1, "memory_gb": 1}, mesh_axis_names=("shard",), mesh_axis_shapes=(4,))
class SparseEmbeddingSPMD:
    """Generic sharded embedding table with forward lookup and gradient-based update."""

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 3):
        logger.debug(
            "SparseEmbeddingSPMD_init_begin rank=%s world_size=%s vocab_size=%s embed_dim=%s",
            getattr(self, "rank", None),
            getattr(self, "world_size", None),
            vocab_size,
            embed_dim,
        )
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.params = jnp.ones((vocab_size, embed_dim)) * 0.01
        self._forward_ids_cache = {}
        logger.debug(
            "SparseEmbeddingSPMD_init_complete rank=%s param_shape=%s",
            getattr(self, "rank", None),
            tuple(self.params.shape),
        )

    def save_state(self, ckpt_dir: str) -> dict:
        # Each rank owns disjoint rows (hash-routed).  Save this rank's full
        # params array so restore can merge all-rank contributions.
        path = pathlib.Path(ckpt_dir) / f"sparse_rank_{self.rank}.npy"
        np.save(str(path), np.asarray(self.params))
        return {}

    def restore_state(self, state: dict, ckpt_dir: str) -> None:
        # Rows start at 0.01 and are updated by exactly one rank.
        # sum(all_shards) - (N-1)*0.01 recovers the trained value per row.
        shards = sorted(pathlib.Path(ckpt_dir).glob("sparse_rank_*.npy"))
        if not shards:
            raise FileNotFoundError(f"No sparse shards found in {ckpt_dir}")
        merged = sum(jnp.asarray(np.load(str(p))) for p in shards)
        self.params = merged - (len(shards) - 1) * 0.01

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
        ids = unique_ids.flat_values
        self._forward_ids_cache[step] = ids

        local_ids = jnp.mod(ids, self.vocab_size)
        embeddings = self.params[local_ids] * local_ids[:, None]

        g_shape = None
        if unique_ids.global_shape and len(unique_ids.global_shape) >= 1:
            g_shape = (unique_ids.global_shape[0], self.embed_dim)

        output = TransportTensor(
            embeddings, origin_indices=unique_ids.origin_indices, global_shape=g_shape
        )
        print(
            f"[SparseEmbeddingSPMD.lookup] rank={self.rank} step={step} "
            f"ids={ids} embeddings={output.flat_values}"
        )
        return LookupOutput(sparse_features=output)

    @operator(
        name="update",
        data_argument=["grads"],
        input_sharding={
            "grads": {"type": "mirror", "reference": "lookup.sparse_features"}
        },
    )
    def update(self, grads: TransportTensor) -> None:
        step = self.ctx.step
        ids = self._forward_ids_cache.pop(step, None)

        if ids is None:
            raise ValueError(
                f"Worker {self.rank}: missing forward cache for step {step}"
            )

        local_ids = jnp.mod(ids, self.vocab_size)
        self.params = self.params.at[local_ids].add(-0.1 * grads.flat_values)

        print(
            f"[SparseEmbeddingSPMD.update] rank={self.rank} step={step} "
            f"ids={ids} grads={grads.flat_values}"
        )


def hash_routing(tensor, mesh):
    if isinstance(tensor, TransportTensor) and tensor.routing_indices is not None:
        val = tensor.routing_indices
    else:
        val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    return jnp.mod(val, mesh.size)
