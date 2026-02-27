import jax.numpy as jnp
from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from .data_types import LookupOutput


def route_back(tensor, mesh):
    """Routing unique embeddings back to origin Dense workers."""
    return tensor.origin_ranks


@spmd(mesh_axis_names=("shard",), mesh_axis_shapes=(4,))
class SparseEmbeddingSPMD:
    """
    Generic lookup table.
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        # Simple parameter initialization
        self.params = jnp.ones((vocab_size, embed_dim)) * 0.01

    @operator(
        name="lookup",
        # We can dynamically name the tensor argument via metadata
        data_argument=["unique_ids"],
        input_sharding={
            "unique_ids": {"type": "dynamic", "axis": [0], "fn": "hash_routing"}
        },
        output_sharding={
            "sparse_features": {"type": "dynamic", "axis": [0], "fn": "route_back"}
        },
    )
    def lookup(self, unique_ids: TransportTensor) -> LookupOutput:
        """
        Receives 'unique_ids' directly.
        """
        rank = self.rank  # type: ignore rank and world_size has been injected by the decorator

        ids = unique_ids.flat_values
        # Use modulo to stay within vocab_size for demo
        local_ids = jnp.mod(ids, self.vocab_size)
        embeddings = self.params[local_ids]

        print(
            f"[SparseSPMD] Worker {rank}: Looked up {len(ids)} IDs. First few: {ids[:3] if len(ids) > 0 else []}"
        )
        print(
            f"[SparseSPMD] Worker {rank}: Looked up embeddings sum {jnp.sum(embeddings)}"
        )

        output = TransportTensor(
            embeddings,
            indices=unique_ids.global_indices,  # Propagate global IDs for backward routing
            origin_ranks=unique_ids.origin_ranks,
            origin_indices=unique_ids.origin_indices,
        )
        return LookupOutput(sparse_features=output)

    @operator(
        name="update",
        data_argument=["grads"],
        input_sharding={
            "grads": {"type": "dynamic", "axis": [0], "fn": "hash_routing"}
        },
    )
    def update(self, grads: TransportTensor) -> None:
        ids = grads.global_indices
        if ids is None:
            print(
                f"[SparseSPMD] Worker {self.rank} Update Error: Gradients missing global_indices"
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
    if isinstance(tensor, TransportTensor) and tensor.global_indices is not None:
        val = tensor.global_indices
    else:
        val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    return jnp.mod(val, mesh.size)
