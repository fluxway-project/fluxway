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
        # Simple parameter initialization
        self.params = jnp.ones((vocab_size, embed_dim)) * 0.01

    @operator(
        name="lookup",
        # We can dynamically name the tensor argument via metadata
        data_argument=["unique_ids"],
        input_sharding={
            "unique_ids": {"type": "dynamic", "axis": ["shard"], "fn": "hash_routing"}
        },
        output_sharding={"sparse_features": {"type": "dynamic", "axis": ["shard"]}},
    )
    def lookup(self, unique_ids: TransportTensor) -> LookupOutput:
        """
        Receives 'unique_ids' directly.
        """
        rank = self.rank  # type: ignore rank and world_size has been injected by the decorator
        world_size = self.world_size  # type: ignore

        ids = unique_ids.flat_values
        # Use modulo to stay within vocab_size for demo
        local_ids = jnp.mod(ids, self.vocab_size)
        embeddings = self.params[local_ids] * rank * world_size

        # Determine global shape for downstream routing.
        # Embeddings are (num_uniques, embed_dim), so global shape corresponds to (global_batch, embed_dim).
        g_shape = None
        if unique_ids.global_shape and len(unique_ids.global_shape) >= 1:
            g_shape = (unique_ids.global_shape[0], self.embed_dim)

        output = TransportTensor(
            embeddings,
            origin_indices=unique_ids.origin_indices,
            routing_indices=unique_ids.origin_indices,
            global_shape=g_shape,
        )
        print(
            f"[SparseSPMD] lookup rank={rank}, unique ids={unique_ids.flat_values}, global_shape={g_shape}, output embeddings={output.flat_values}"
        )
        return LookupOutput(sparse_features=output)

    @operator(
        name="update",
        data_argument=["grads"],
        input_sharding={
            "grads": {"type": "dynamic", "axis": ["shard"], "fn": "hash_routing"}
        },
    )
    def update(self, grads: TransportTensor) -> None:
        ids = grads.routing_indices
        if ids is None:
            print(
                f"[SparseSPMD] Worker {self.rank} Update Error: Gradients missing routing_indices"
            )
            return

        g_flat = grads.flat_values

        # Correctly handle unique gradients if they weren't uniquified upstream
        # Although ideally they should be uniquified at Dense worker.
        # For robustness, we can do it here too if multiple dense workers send grads for same ID.

        # Modulo for demo params
        local_ids = jnp.mod(ids, self.vocab_size)

        # Scatter-add gradients to parameters (simple SGD update)
        self.params = self.params.at[local_ids].add(-0.1 * g_flat)

        print(
            f"[SparseSPMD] Worker {self.rank} Update Task: Updated {len(ids)} IDs. Grad Sum: {jnp.sum(g_flat)}"
        )


def hash_routing(tensor, mesh):
    if isinstance(tensor, TransportTensor) and tensor.routing_indices is not None:
        val = tensor.routing_indices
    else:
        val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    return jnp.mod(val, mesh.size)
