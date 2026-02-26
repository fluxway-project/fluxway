from dataclasses import dataclass
import jax.numpy as jnp
from fluxway.common.decorators import data_type
from fluxway.common.types.tensor import TransportTensor


@data_type()
@dataclass
class RaggedInput:
    """Raw ragged input with multiple features."""

    feature1: TransportTensor
    feature2: TransportTensor


@data_type()
@dataclass
class InputNodeOutput:
    """Consolidated output from the Input Processor with flattened fields."""

    unique_ids1: TransportTensor
    inv1: TransportTensor
    unique_ids2: TransportTensor
    inv2: TransportTensor


@data_type()
@dataclass
class LookupOutput:
    """Output from an embedding table."""

    sparse_features: TransportTensor


@data_type()
@dataclass
class DenseOutput:
    """The result of the dense forward pass, serving as context for backward."""

    prediction: jnp.ndarray
    # Metadata for backward pass (to be used by Self-Loop)
    inv1: TransportTensor
    inv2: TransportTensor


@data_type()
@dataclass
class GradientOutput:
    """Gradients produced by the dense layer."""

    # Matches update operators' expected inputs
    grads: TransportTensor
