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
    def lookup(self, data: dict) -> LookupOutput:
        """
        Receives a dict containing 'unique_ids'.
        """
        rank = self.rank
        uids = data["unique_ids"]

        ids = uids.flat_values
        embeddings = jnp.ones((ids.shape[0], self.embed_dim)) * (rank + 1)

        print(f"[SparseSPMD] Worker {rank}: Looked up IDs {ids}")
        print(
            f"[SparseSPMD] Worker {rank}: Returning embeddings sum {jnp.sum(embeddings)}"
        )

        output = TransportTensor(
            embeddings,
            origin_ranks=uids.origin_ranks,
            origin_indices=uids.origin_indices,
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
        print(
            f"[SparseSPMD] Update Task: Received Gradients for IDs. Grad Sum: {jnp.sum(grads.flat_values)}"
        )


def hash_routing(tensor, mesh):
    val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    return jnp.mod(val, mesh.size)
