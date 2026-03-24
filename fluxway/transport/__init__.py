"""Transport layer: Data transmission between SPMDs."""

# Ensure INFO logs are visible even when importing this subpackage directly.
from fluxway.common.utils.logging import ensure_logging_configured


from fluxway.common.types import TransportTensor
from fluxway.common.types.sharding import (
    ShardingSpec,
    PhysicalLayout,
    PartitionStrategy,
    ChunkPartition,
    CyclicPartition,
    IndexedPartition,
    GreedyPartition,
)
from .planner import TransportPlanner, TransportPlan
from .communication import DataDistributor, TransportReceiver

__all__ = [
    "TransportTensor",
    "ShardingSpec",
    "PhysicalLayout",
    "PartitionStrategy",
    "ChunkPartition",
    "CyclicPartition",
    "IndexedPartition",
    "GreedyPartition",
    "TransportPlanner",
    "TransportPlan",
    "DataDistributor",
    "TransportReceiver",
]

ensure_logging_configured()
