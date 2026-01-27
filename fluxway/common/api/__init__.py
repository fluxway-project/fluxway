"""
Fluxway API module.

Provides decorators for defining Fluxway data types and SPMD modules.
"""

from .data import data_type
from .spmd import spmd

__all__ = [
    "data_type",
    "spmd",
]
