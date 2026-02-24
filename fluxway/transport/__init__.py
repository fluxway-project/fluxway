"""Transport layer: Data transmission between SPMDs."""

# Ensure INFO logs are visible even when importing this subpackage directly.
from ..common.utils.logging import ensure_logging_configured


from .tensor import TransportTensor
from fluxway.common.types.sharding import (
    Layout,
    ShardingSpec,
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
    "Layout",
    "ShardingSpec",
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
