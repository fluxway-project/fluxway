import jax
import jax.numpy as jnp
from fluxway.common.decorators import spmd, operator
from .data_types import EmbeddingInput, EmbeddingOutput, GradientOutput

@spmd(
    resource={"nums_cpu": 1},
    mesh_axis_names=("shard",),
    mesh_axis_shapes=(2,),  # Example: 2 workers for embedding shard
)
class SparseEmbeddingSPMD:
    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Local shard of embedding table
        # In a real scenario, this would be initialized based on rank/mesh
        self.weights = jnp.zeros((vocab_size // 2, embed_dim)) 

    @operator(
        name="lookup",
        tensor_argument=["feature_ids"],
        input_sharding={
            "feature_ids": {"type": "dynamic", "axis": [0], "fn": "hash_routing"}
        },
        output_sharding={
            "embeddings": {"type": "static", "axis": ["shard", None, None]}
        }
    )
    def lookup(self, data: EmbeddingInput) -> EmbeddingOutput:
        """
        Perform embedding lookup using hash-based routing.
        """
        # Feature IDs are already routed to this worker by Fluxway
        print(f"[SparseSPMD] lookup: received feature_ids {data.feature_ids.shape}")
        
        # Simulate lookup
        batch_size = data.feature_ids.shape[0]
        num_features = data.feature_ids.shape[1]
        emb_values = jnp.zeros((batch_size, num_features, 64))
        
        return EmbeddingOutput(embeddings=emb_values, feature_ids=data.feature_ids)

    @operator(
        name="update",
        tensor_argument=["grads", "feature_ids"],
        input_sharding={
            "grads": {"type": "static", "axis": ["shard", None, None]},
            "feature_ids": {"type": "static", "axis": ["shard", None]}
        }
    )
    def update(self, data: GradientOutput) -> None:
        """Update local embedding weights with gradients."""
        print(f"[SparseSPMD] update: received grads {data.grads.shape} for ids {data.feature_ids.shape}")
        # Simple SGD update for demo
        # ids = data.feature_ids % (self.vocab_size // 2)
        # self.weights = self.weights.at[ids].add(-0.01 * data.grads)
        pass

def hash_routing(tensor, mesh):
    """
    Routing function for sparse embedding IDs.
    Determines which worker rank handles which ID.
    """
    # hash(id) % mesh_size
    return jax.hash(tensor) % mesh.size
