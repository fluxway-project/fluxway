"""Common type definitions for Fluxway framework."""

from .configs import (
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
    DriverResourceSpec,
    # State Management Types (Section 2.1.4)
    GlobalState,
    SPMDState,
    WorkerTelemetry,
    # Handle Types (Section 3.2.1, 6.1)
    LogicalHandle,
    ActualHandle,
    NodeView,
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

# Unified tensor wrapper
from .tensor import TransportTensor

__all__ = [
    # SPMD Configuration
    "SPMDConfig",
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
    # State Management Types
    "GlobalState",
    "SPMDState",
    "WorkerTelemetry",
    # Handle Types
    "LogicalHandle",
    "ActualHandle",
    "NodeView",
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
    "TransportTensor",
]
