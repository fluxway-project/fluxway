from typing import Optional
from dataclasses import dataclass
import jax.numpy as jnp
from fluxway.common.decorators import data_type


@data_type()
@dataclass
class EmbeddingInput:
    """Input for embedding lookup."""

    feature_ids: jnp.ndarray  # (batch_size, num_features)


@data_type()
@dataclass
class EmbeddingOutput:
    """Output of embedding lookup."""

    embeddings: jnp.ndarray  # (batch_size, num_features, embed_dim)
    feature_ids: Optional[jnp.ndarray] = None  # Added for routing
    # Fields to match @operator(tensor_argument) in DenseSPMD.forward
    emb1: Optional[jnp.ndarray] = None
    emb2: Optional[jnp.ndarray] = None


@data_type()
@dataclass
class DenseInput:
    """Input for dense computation."""

    vector: jnp.ndarray  # (batch_size, flattened_dim)
    feature_ids1: Optional[jnp.ndarray] = None  # Added for routing
    feature_ids2: Optional[jnp.ndarray] = None  # Added for routing
    # Fields for DenseSPMD.backward
    prediction_grad: Optional[jnp.ndarray] = None
    # Field for DenseSPMD.forward output_sharding
    prediction: Optional[jnp.ndarray] = None


@data_type()
@dataclass
class GradientOutput:
    """Gradients for embedding updates."""

    grads: jnp.ndarray  # Same shape as embeddings
    feature_ids: jnp.ndarray  # To know which weights to update
    # Field to match @operator(tensor_argument) in DenseSPMD.backward
    grad1: Optional[jnp.ndarray] = None
    grad2: Optional[jnp.ndarray] = None
