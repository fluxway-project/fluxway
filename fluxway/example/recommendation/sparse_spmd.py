import jax.numpy as jnp
from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from .data_types import LookupOutput


@spmd(mesh_axis_names=("shard",), mesh_axis_shapes=(4,))
class SparseEmbeddingSPMD:
    """
    Generic lookup table.
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 3):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.params = jnp.ones((vocab_size, embed_dim)) * 0.01
        self._forward_ids_cache = {}

    @operator(
        name="lookup",
        data_argument=["unique_ids"],
        input_sharding={
            "unique_ids": {"type": "dynamic", "axis": ["shard"], "fn": "hash_routing"}
        },
        output_sharding={"sparse_features": {"type": "dynamic", "axis": ["shard"]}},
    )
    def lookup(self, unique_ids: TransportTensor) -> LookupOutput:
        rank = self.rank
        world_size = self.world_size
        step = self.ctx.step

        ids = unique_ids.flat_values
        self._forward_ids_cache[step] = ids

        local_ids = jnp.mod(ids, self.vocab_size)
        embeddings = self.params[local_ids] * rank * world_size

        g_shape = None
        if unique_ids.global_shape and len(unique_ids.global_shape) >= 1:
            g_shape = (unique_ids.global_shape[0], self.embed_dim)

        output = TransportTensor(
            embeddings,
            origin_indices=unique_ids.origin_indices,
            # Use unique IDs as routing indices for better introspection/fallback
            routing_indices=ids,
            global_shape=g_shape,
        )
        print(
            f"[SparseSPMD] lookup rank={rank}, step={step}, unique ids={ids}, output embeddings={output.flat_values}"
        )
        return LookupOutput(sparse_features=output)

    @operator(
        name="update",
        data_argument=["grads"],
        input_sharding={
            "grads": {"type": "mirror", "source": "lookup.sparse_features"}
        },
    )
    def update(self, grads: TransportTensor) -> None:
        step = self.ctx.step
        ids = self._forward_ids_cache.pop(step, None)

        if ids is None:
            raise ValueError(
                f"Worker {self.rank} Update Error: Missing forward cache for step {step}"
            )

        g_flat = grads.flat_values
        local_ids = jnp.mod(ids, self.vocab_size)

        self.params = self.params.at[local_ids].add(-0.1 * g_flat)

        print(
            f"[SparseSPMD] Worker {self.rank} Update Task: Step {step}, updated {len(ids)} IDs. Grad Sum: {jnp.sum(g_flat)}"
        )


def hash_routing(tensor, mesh):
    if isinstance(tensor, TransportTensor) and tensor.routing_indices is not None:
        val = tensor.routing_indices
    else:
        val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    return jnp.mod(val, mesh.size)
