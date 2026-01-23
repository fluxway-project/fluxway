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

# Import decorators for top-level access
from fluxway.common.decorators import data_type, spmd, operator

# Import common types
from fluxway.common.types import (
    SPMDConfig,
    Graph,
    Node,
    Edge,
    ResourceSpec,
    # Sharding types
    LayoutMesh,
    LayoutOp,
    LayoutType,
    ShardingSpec,
    StaticShardingSpec,
    DynamicShardingSpec,
    DynamicRoutingFn,
    create_mesh,
)

# Import control plane components
from fluxway.control_plane import GraphParser, Builder

__all__ = [
    # Version
    "__version__",
    # Decorators
    "data_type",
    "spmd",
    "operator",
    # Types
    "PartitionType",
    "SPMDConfig",
    "Graph",
    "Node",
    "Edge",
    "ResourceSpec",
    # Sharding
    'LayoutMesh',
    'LayoutOp',
    'LayoutType',
    'ShardingSpec',
    'StaticShardingSpec',
    'DynamicShardingSpec',
    'DynamicRoutingFn',
    'create_mesh',
    # Control Plane
    "GraphParser",
    "Builder",
]
