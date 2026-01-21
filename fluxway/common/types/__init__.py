"""Common type definitions for Fluxway framework."""

from .data_types import (
    # Partition and Sharding Types
    PartitionType,
    # SPMD Configuration
    SPMDConfig,
    # Graph Builder Types
    Edge,
    Node,
    Graph,
    GraphValidationResult,
    # Resource and Scheduling Types (Section 2.1.2)
    ResourceSpec,
    PlacementPlan,
    PlacementStrategy,
    ResourcePriority,
    # State Management Types (Section 2.1.4)
    GlobalState,
    SPMDState,
    WorkerTelemetry,
    # Handle Types (Section 3.2.1, 6.1)
    LogicalHandle,
    ActualHandle,
    NodeView,
    # Dual-Stream Convergence Types (Section 3.2.2)
    ControlConfig,
    # ULP Protocol Types (Section 4.2, 4.3)
    PartitionOp,
    LayoutSpec,
    DependencySpec,
    RPCPayload,
)
