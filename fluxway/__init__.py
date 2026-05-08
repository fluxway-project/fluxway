"""
Fluxway: A distributed deep learning framework for large-scale training.

Features:
- Dual-Plane Propagation Protocol
- Dual-Stream Convergence
- Asynchronous Execution
- Graph-based Orchestration

Usage:
    import fluxway

    @fluxway.data_type(control_argument=["id"], data_argument=["tensor"])
    class MyData:
        id: int
        tensor: torch.Tensor

    # Use tuple-based mesh with user-defined axis names
    @fluxway.spmd(
        resource={"nums_cpu": 4, "nums_gpu": 2},
        mesh_axis_names=("data", "model"),
        mesh_axis_shapes=(4, 2),
        max_async_steps=10
    )
    class MySPMD:
        def forward(self, data):
            return process(data)
"""

__version__ = "0.1.0"

# Force logging to be visible across driver + Ray worker processes.
# Users of this repo want INFO logs by default.
from fluxway.common.utils.logging import ensure_logging_configured

# Import decorators for top-level access
from fluxway.common.decorators import data_type, spmd, operator
from fluxway.common.bind import bind_spmd

# Import common types
from fluxway.common.types import (
    SPMDConfig,
    Graph,
    Node,
    Edge,
    ResourceSpec,
    # Sharding types
    ShardingSpec,
    PhysicalLayout,
    PartitionSpec,
    ChunkPartition,
    CyclicPartition,
    IndexedPartition,
    GreedyPartition,
)

# Import control plane components
from fluxway.control_plane import GraphParser, Builder

# Import runner
from fluxway.runner import launch

__all__ = [
    # Version
    "__version__",
    # Runner
    "launch",
    # Decorators
    "data_type",
    "spmd",
    "operator",
    "bind_spmd",
    # Types
    "PartitionSpec",
    "SPMDConfig",
    "Graph",
    "Node",
    "Edge",
    "ResourceSpec",
    # Sharding
    "ShardingSpec",
    "PhysicalLayout",
    "PartitionSpec",
    "ChunkPartition",
    "CyclicPartition",
    "IndexedPartition",
    "GreedyPartition",
    # Control Plane
    "GraphParser",
    "Builder",
]

ensure_logging_configured()
