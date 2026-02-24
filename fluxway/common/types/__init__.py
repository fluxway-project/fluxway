"""Common type definitions for Fluxway framework."""

from .data_types import (
    # SPMD Configuration
    SPMDConfig,
    OpType,
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
    DriverResourceSpec,
    SPMDPlacementSpec,
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
    # Tensor Metadata (for routing and backward tracking)
    TensorMeta,
    # Operator metadata
    EdgeShardingInfo,
    OperatorMeta,
    ArgumentType,
    # Driver and Worker Status Types
    DriverStatus,
    WorkerStatus,
    ControllerState,
    ControllerEvent,
    DriverState,
    DriverEvent,
    WorkerHeartbeat,
    DriverHeartbeat,
    HeartbeatConfig,
    AsyncStepConfig,
    SPMDOpStep,
    OpLockState,
    # Reader types (new)
    NodeType,
    ReaderConfig,
    ReaderState,
    MailboxStatus,
)

# Sharding types: algebraic layer
from .sharding import (
    # Algebraic sharding (data-plane)
    PartitionSpec,
    P,
    PartitionStrategy,
    ChunkPartition,
    CyclicPartition,
    IndexedPartition,
    GreedyPartition,
    Layout,
    ShardingSpec,

)

__all__ = [
    # SPMD Configuration
    "SPMDConfig",
    "OpType",
    # Graph Builder Types
    "Edge",
    "Node",
    "Graph",
    "GraphValidationResult",
    # Resource and Scheduling Types
    "ResourceSpec",
    "PlacementPlan",
    "PlacementStrategy",
    "ResourcePriority",
    "DriverResourceSpec",
    "SPMDPlacementSpec",
    # State Management Types
    "GlobalState",
    "SPMDState",
    "WorkerTelemetry",
    # Handle Types
    "LogicalHandle",
    "ActualHandle",
    "NodeView",
    # Dual-Stream Convergence Types
    "ControlConfig",
    # ULP Protocol Types
    "PartitionOp",
    "LayoutSpec",
    "DependencySpec",
    "RPCPayload",
    # Tensor Metadata
    "TensorMeta",
    # Operator metadata
    "EdgeShardingInfo",
    "OperatorMeta",
    "ArgumentType",
    # Driver and Worker Status Types
    "DriverStatus",
    "WorkerStatus",
    "ControllerState",
    "ControllerEvent",
    "DriverState",
    "DriverEvent",
    "WorkerHeartbeat",
    "DriverHeartbeat",
    "HeartbeatConfig",
    "AsyncStepConfig",
    "SPMDOpStep",
    "OpLockState",
    # Reader types
    "NodeType",
    "ReaderConfig",
    "ReaderState",
    "MailboxStatus",
    # Sharding types: algebraic (data-plane)
    "PartitionSpec",
    "P",
    "PartitionStrategy",
    "ChunkPartition",
    "CyclicPartition",
    "IndexedPartition",
    "GreedyPartition",
    "Layout",
    "ShardingSpec",

]
