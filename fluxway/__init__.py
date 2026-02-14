"""
Fluxway: A distributed deep learning framework for large-scale training.

Features:
- Dual-Plane Propagation Protocol
- Dual-Stream Convergence
- Asynchronous Execution
- Graph-based Orchestration

Usage:
    import fluxway

    @fluxway.data_type(state_argument=["id"], tensor_argument=["tensor"])
    class MyData:
        id: int
        tensor: torch.Tensor

    @fluxway.spmd(
        resource={"nums_cpu": 4, "nums_gpu": 2},
        input_sharding={PartitionType.DP: 4},
        max_async_steps=10
    )
    class MySPMD:
        def forward(self, data):
            return process(data)
"""

__version__ = "0.1.0"

# Import decorators for top-level access
from fluxway.common.api import data_type, spmd

# Import common types
from fluxway.common.types import (
    PartitionType,
    SPMDConfig,
    Graph,
    Node,
    Edge,
    ResourceSpec,
)

# Import control plane components
from fluxway.control_plane import GraphParser, Builder

__all__ = [
    # Version
    "__version__",
    # Decorators
    "data_type",
    "spmd",
    # Types
    "PartitionType",
    "SPMDConfig",
    "Graph",
    "Node",
    "Edge",
    "ResourceSpec",
    # Control Plane
    "GraphParser",
    "Builder",
]
