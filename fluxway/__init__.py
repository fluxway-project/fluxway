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
    from fluxway.common.decorators import spmd
    from fluxway import ResourceProfile

    @spmd(
        resource_profile=ResourceProfile(base={"CPU": 4, "GPU": 2}),
        mesh_axis_names=("data", "model"),
        mesh_axis_shapes=(4, 2),
        max_async_steps=10
    )
    class MySPMD:
        def forward(self, data):
            return process(data)
"""

import ml_dtypes  # noqa: F401

# Force logging to be visible across driver + Ray worker processes.
# Users of this repo want INFO logs by default.
from fluxway.common.utils.logging import ensure_logging_configured

# SPMD & Common Types (imported first to initialize type definitions and prevent circular imports)
from fluxway.common.types import (
    ExecutionFailure,
    Graph,
    Node,
    SPMDConfig,
    # Sharding & Layout Primitives
    ChunkPartition,
    CyclicPartition,
    FunctionPartition,
    GreedyPartition,
    PartitionSpec,
    PhysicalLayout,
    ShardingSpec,
)

# Decorators & Operator Declarations
from fluxway.common.decorators import (
    data_type,
    operator,
    requires_worker_process,
)
from fluxway.common.bind import bind_spmd

# Resource Registry & Execution Binding
from fluxway.common.resource_registry import (
    ResourceHandle,
    ResourceRegistry,
    bind_resource_registry,
    get_active_resource_registry,
    reset_active_resource_registry,
    set_active_resource_registry,
)
from fluxway.common.logical_partition_binding import LogicalPartitionBinding

# Control Plane, Graph & Orchestration
from fluxway.control_plane import (
    Builder,
    ExecutionFailedError,
    GraphExecution,
    start,
)
from fluxway.runner import run_in_sandbox
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.coherence import TensorAccessIntent
from fluxway import profiler
from fluxway.visualization import view_graph, visualize_graph

# Placement & Scheduling Engine
from fluxway.placement import (
    AFFINITY_FIRST,
    PlacementAttestationError,
    PlacementContractError,
    PlacementError,
    PlacementGroupTimeoutError,
    PlacementInfeasibleError,
    PlacementMaterializationError,
    PlacementRecipe,
    PlacementRuntimeConfig,
    PlacementSolveTimeoutError,
    ResourceProfile,
    ResourceScale,
    SPMDRef,
    affinity_constraints,
    affinity_coverage,
    all_of,
    all_workers,
    allocated_quanta,
    any_of,
    cardinality,
    co_location_mass,
    difference,
    disjoint,
    equal,
    incident_spmds,
    instance,
    intersection,
    isolated_spmds,
    less_equal,
    lexicographic,
    max_min,
    maximize,
    maximize_majorization,
    minimize,
    negate,
    occupancy,
    rank,
    ranks,
    resource_capacity,
    resource_fragmentation,
    resource_usage,
    subseteq,
    support,
    target_occupancy,
    union,
    weighted_dominant_share,
    workers,
)

__all__ = [
    # Decorators & Operator Declarations
    "data_type",
    "operator",
    "requires_worker_process",
    "bind_spmd",
    "OperatorSpec",
    "TensorAccessIntent",
    # Graph & Orchestration
    "Builder",
    "GraphExecution",
    "start",
    "run_in_sandbox",
    "view_graph",
    "visualize_graph",
    # SPMD & Common Types
    "Graph",
    "Node",
    "SPMDConfig",
    "ExecutionFailure",
    "ExecutionFailedError",
    # Sharding & Layout Primitives
    "ChunkPartition",
    "CyclicPartition",
    "FunctionPartition",
    "GreedyPartition",
    "PartitionSpec",
    "PhysicalLayout",
    "ShardingSpec",
    # Resource Registry & Execution Binding
    "ResourceHandle",
    "ResourceRegistry",
    "LogicalPartitionBinding",
    "bind_resource_registry",
    "get_active_resource_registry",
    "reset_active_resource_registry",
    "set_active_resource_registry",
    # Profiler
    "profiler",
    # Placement & Scheduling Engine
    "AFFINITY_FIRST",
    "PlacementRecipe",
    "PlacementRuntimeConfig",
    "ResourceProfile",
    "ResourceScale",
    "SPMDRef",
    # Placement Exceptions
    "PlacementAttestationError",
    "PlacementContractError",
    "PlacementError",
    "PlacementGroupTimeoutError",
    "PlacementInfeasibleError",
    "PlacementMaterializationError",
    "PlacementSolveTimeoutError",
    # Placement Constraint & Objective DSL
    "affinity_constraints",
    "affinity_coverage",
    "all_of",
    "all_workers",
    "allocated_quanta",
    "any_of",
    "cardinality",
    "co_location_mass",
    "difference",
    "disjoint",
    "equal",
    "incident_spmds",
    "instance",
    "intersection",
    "isolated_spmds",
    "less_equal",
    "lexicographic",
    "max_min",
    "maximize",
    "maximize_majorization",
    "minimize",
    "negate",
    "occupancy",
    "rank",
    "ranks",
    "resource_capacity",
    "resource_fragmentation",
    "resource_usage",
    "subseteq",
    "support",
    "target_occupancy",
    "union",
    "weighted_dominant_share",
    "workers",
]

ensure_logging_configured()
