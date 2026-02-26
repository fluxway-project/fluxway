import jax.numpy as jnp
from fluxway.common.decorators import spmd, operator
from .data_types import DenseInput, EmbeddingOutput, GradientOutput


def hash_routing(indices, num_workers, **kwargs):
    """Hash-based routing for All-to-All communication."""
    return jnp.mod(indices, num_workers)


@spmd(
    resource={"nums_cpu": 1},
    mesh_axis_names=("dp", "mp"),
    mesh_axis_shapes=(2, 2),  # 2 data parallel, 2 model parallel
)
class DenseSPMD:
    def __init__(self, hidden_dim: int = 256):
        self.hidden_dim = hidden_dim
        # Model weights would be initialized here and sharded via jax.jit/sharding
        # For demo, we just simulate the computation logic
        pass

    @operator(
        name="forward",
        tensor_argument=["emb1", "emb2"],
        input_sharding={
            "emb1": {"type": "static", "axis": ["dp", None, None]},
            "emb2": {"type": "static", "axis": ["dp", None, None]},
        },
        output_sharding={"prediction": {"type": "static", "axis": ["dp", None]}},
    )
    def forward(self, emb1: EmbeddingOutput, emb2: EmbeddingOutput) -> DenseInput:
        """
        Concatenate embeddings and perform dense layers.
        Incoming embeddings are already partitioned by DP axis.
        """
        print(
            f"[DenseSPMD] forward: received emb1 {emb1.embeddings.shape}, emb2 {emb2.embeddings.shape}"
        )

        # Concatenate embeddings: (batch, num_features, dim) -> (batch, flattened)
        merged = jnp.concatenate([emb1.embeddings, emb2.embeddings], axis=1)
        flattened = merged.reshape(merged.shape[0], -1)

        # We store feature_ids so we can return them in backward
        return DenseInput(
            vector=flattened,
            feature_ids1=emb1.feature_ids,
            feature_ids2=emb2.feature_ids,
        )

    @operator(
        name="backward",
        tensor_argument=["prediction_grad"],
        input_sharding={"prediction_grad": {"type": "static", "axis": ["dp", None]}},
        output_sharding={
            "grad1": {"type": "dynamic", "axis": [0], "fn": "hash_routing"},
            "grad2": {"type": "dynamic", "axis": [0], "fn": "hash_routing"},
        },
    )
    def backward(self, data: DenseInput) -> GradientOutput:
        """Compute gradients and route back to sparse SPMDs."""
        # Simulated backprop
        batch_size = data.vector.shape[0]
        print(f"[DenseSPMD] backward: computing grads for batch {batch_size}")

        # grad1 corresponds to emb1/ids1, grad2 corresponds to emb2/ids2
        # In a real model, we'd slice 'data.vector' gradients
        dummy_grad1 = jnp.zeros((batch_size, 2, 64))  # Assuming 2 features for Sparse1
        dummy_grad2 = jnp.zeros((batch_size, 2, 64))  # Assuming 2 features for Sparse2

        return GradientOutput(
            grads=dummy_grad1,  # Primary grad field
            feature_ids=data.feature_ids1,
            grad1=dummy_grad1,
            grad2=dummy_grad2,
        )
