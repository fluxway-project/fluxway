import jax.numpy as jnp
from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from .data_types import LookupOutput


def route_to_dense(tensor, mesh):
    """Calculates destination rank based on origin_indices."""
    if tensor.origin_indices is None:
        return 0

    # Assume uniform chunk sharding on the target (Dense)
    # We use global_shape to determine the logical batch size
    global_rows = tensor.global_shape[0] if tensor.global_shape else 100
    rows_per_rank = (global_rows + mesh.size - 1) // mesh.size

    target_ranks = tensor.origin_indices // rows_per_rank
    return jnp.clip(target_ranks, 0, mesh.size - 1)


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
            "unique_ids": {"type": "dynamic", "axis": ["shard"], "fn": "hash_routing"}
        },
        output_sharding={
            "sparse_features": {
                "type": "dynamic",
                "axis": ["shard"],
                "fn": "route_to_dense",
            }
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
            spatial_indices=unique_ids.spatial_indices,  # Propagate global IDs for backward routing
            origin_indices=unique_ids.origin_indices,
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
        ids = grads.spatial_indices
        if ids is None:
            print(
                f"[SparseSPMD] Worker {self.rank} Update Error: Gradients missing spatial_indices"
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
    if isinstance(tensor, TransportTensor) and tensor.spatial_indices is not None:
        val = tensor.spatial_indices
    else:
        val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    return jnp.mod(val, mesh.size)
