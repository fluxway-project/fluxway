"""
Data type definitions for the example.

All data types include both state and tensor arguments to demonstrate
dual-stream convergence transmission.
"""

from dataclasses import dataclass
import numpy as np

from fluxway import data_type


@data_type()
@dataclass
class TensorOnly:
    """Data type with only tensor for eager_trigger operators."""
    input: list # tensor_argument only


@data_type()
@dataclass
class DataState1:
    """Intermediate data state."""
    tensor: np.ndarray  # tensor_argument


@data_type()
@dataclass
class DataState2:
    """Intermediate data state."""
    tensor: np.ndarray  # tensor_argument
    processed: bool  # state_argument


@data_type()
@dataclass
class DataState3:
    """Another intermediate data state."""
    tensor: np.ndarray  # tensor_argument
    stage: int  # state_argument


@data_type()
@dataclass
class Gradient2:
    """Gradient for backward pass."""
    tensor: np.ndarray  # tensor_argument


