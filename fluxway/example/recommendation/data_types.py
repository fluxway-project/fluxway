from dataclasses import dataclass
import jax.numpy as jnp
from fluxway.common.decorators import data_type
from fluxway.common.types.tensor import TransportTensor


@data_type()
@dataclass
class ActiveFeatures:
    """User and context features (mostly static per sample)."""

    user_id: TransportTensor
    user_age_level: TransportTensor
    request_hour: TransportTensor
    hist_click_item_seq: TransportTensor  # Ragged
    hist_seq_length: TransportTensor


@data_type()
@dataclass
class PassiveFeatures:
    """Candidate item features."""

    target_item_id: TransportTensor
    target_item_category: TransportTensor
    target_item_stats: TransportTensor


@data_type()
@dataclass
class LabelFeatures:
    """Ground truth labels."""

    is_click: TransportTensor


@data_type()
@dataclass
class UserBatch:
    """A complete batch following col-store layout."""

    active: ActiveFeatures
    passive: PassiveFeatures
    labels: LabelFeatures


@data_type()
@dataclass
class DPFeatures:
    """Sample-aligned features that stay on the DP rank."""

    labels: LabelFeatures
    dense_stats: TransportTensor


@data_type()
@dataclass
class InputNodeOutput:
    """Consolidated output from the Input Processor."""

    unique_user_ids: TransportTensor
    user_inv: TransportTensor
    unique_item_ids: TransportTensor
    item_inv: TransportTensor
    # DP-aligned features
    dp_data: DPFeatures


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
    # Metadata for backward
    user_inv: TransportTensor
    item_inv: TransportTensor
    user_emb_unique: TransportTensor
    item_emb_unique: TransportTensor
    dp_data: DPFeatures


@data_type()
@dataclass
class GradientOutput:
    """Gradients produced by the dense layer."""

    grads: TransportTensor
