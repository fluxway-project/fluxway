"""
Data type definitions for the example.

All data types include both state and tensor arguments to demonstrate
dual-stream convergence transmission.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from fluxway import data_type


@data_type()
@dataclass
class DataState1:
    """Output from data reader SPMD."""
    batch_id: int  # state_argument
    tensor: np.ndarray  # tensor_argument
    metadata: str  # state_argument


@data_type()
@dataclass
class DataState2:
    """Intermediate data state."""
    batch_id: int  # state_argument
    tensor: np.ndarray  # tensor_argument
    processed: bool  # state_argument


@data_type()
@dataclass
class DataState3:
    """Another intermediate data state."""
    batch_id: int  # state_argument
    tensor: np.ndarray  # tensor_argument
    stage: int  # state_argument


@data_type()
@dataclass
class DataState4:
    """Final data state."""
    batch_id: int  # state_argument
    tensor: np.ndarray  # tensor_argument
    final: bool  # state_argument


@data_type()
@dataclass
class Gradient1:
    """Gradient for backward pass."""
    batch_id: int  # state_argument
    tensor: np.ndarray  # tensor_argument
    grad_step: int  # state_argument


@data_type()
@dataclass
class Gradient2:
    """Gradient for backward pass."""
    batch_id: int  # state_argument
    tensor: np.ndarray  # tensor_argument
    grad_step: int  # state_argument


@data_type()
@dataclass
class Gradient3:
    """Gradient for backward pass."""
    batch_id: int  # state_argument
    tensor: np.ndarray  # tensor_argument
    grad_step: int  # state_argument


@data_type()
@dataclass
class Gradient4:
    """Gradient for backward pass."""
    batch_id: int  # state_argument
    tensor: np.ndarray  # tensor_argument
    grad_step: int  # state_argument
