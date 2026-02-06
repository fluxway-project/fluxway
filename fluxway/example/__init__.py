"""
Fluxway Example: Demonstration of the framework.

This example demonstrates:
- Sparse and dense sharding combinations
- Dynamic routing with user-defined fn callbacks
- Eager mode execution
- Bidirectional edges
- Forward/backward edge differences
- State and tensor data transmission
- CPU-only resource allocation
- Data simulation without actual computation

SPMD Modules:
- PreprocessingSPMD: preprocessing stage (eager mode, sparse -> dense)
- EmbeddingSPMD: embedding stage (sparse -> sparse)
- ModelSPMD: model computation stage (sparse -> dense)
"""

from .spmd_modules import (
    PreprocessingSPMD,
    EmbeddingSPMD,
    ModelSPMD,
    # Routing functions
    hash_routing_fn,
    modulo_routing_fn,
)

__all__ = [
    "PreprocessingSPMD",
    "EmbeddingSPMD",
    "ModelSPMD",
    "hash_routing_fn",
    "modulo_routing_fn",
]
