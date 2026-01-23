"""
Fluxway Example: Comprehensive demonstration of the framework.

This example demonstrates:
- Dense and sparse sharding combinations (all 4 combinations)
- Dynamic routing with user-defined fn callbacks
- Bidirectional edges
- Forward/backward edge differences
- State and tensor data transmission
- CPU-only resource allocation
- Data simulation without actual computation

Dynamic Sharding Examples:
- DenseSparseSPMD: dense input -> sparse output (uses dst's fn)
- SparseDenseSPMD: sparse input -> dense output (uses src's fn)
- SparseSparseSPMD: sparse input -> sparse output (uses src's fn)
"""

from .spmd_modules import (
    DenseDenseSPMD,
    DenseDenseSPMD1,
    DenseSparseSPMD,
    SparseDenseSPMD,
    SparseSparseSPMD,
    # Routing functions
    hash_routing_fn,
    modulo_routing_fn,
    expert_routing_fn,
)

__all__ = [
    "DenseDenseSPMD",
    "DenseDenseSPMD1",
    "DenseSparseSPMD",
    "SparseDenseSPMD",
    "SparseSparseSPMD",
    "hash_routing_fn",
    "modulo_routing_fn",
    "expert_routing_fn",
]
