"""
Standardized Reader implementation leveraging Google Grain.
"""

import logging
from typing import Any, Dict, Optional

from fluxway.common.decorators import spmd, operator
from .base_reader import BaseReader

logger = logging.getLogger(__name__)

try:
    import grain.python as grain
except ImportError:
    grain = None


@spmd()
class GrainReader(BaseReader):
    """
    Standard Reader implementation using Google Grain.

    Supports Elastic Sharding via Logical Shard Pools.
    """

    def __init__(self):
        super().__init__()
        self._dataloader = None
        self._iterator = None
        self._output_type = None

    def _create_data_loader(
        self,
        static_args: Dict[str, Any],
        rank: int,
        world_size: int,
        shard_index: Optional[int] = None,
        shard_count: Optional[int] = None,
    ):
        """
        Constructs the Google Grain DataLoader pipeline.

        Args:
            shard_index: If provided, overrides rank for data sharding.
            shard_count: If provided, overrides world_size for data sharding.
        """
        if grain is None:
            raise ImportError("google-grain not found. Install it to use GrainReader.")

        data_source = static_args.get("data_source")
        if data_source is None:
            raise ValueError("GrainReader requires 'data_source' in static_args.")

        # 1. Base Dataset
        dataset = grain.MapDataset.source(data_source)

        transformations = static_args.get("transformations", [])
        for tx in transformations:
            dataset = dataset.map(tx)

        # 2. logical sharding for elasticity
        # If the user specifies num_logical_shards, we decouple rank from sharding.
        num_logical_shards = static_args.get("num_logical_shards")

        if shard_index is not None and shard_count is not None:
            s_idx, s_cnt = shard_index, shard_count
        elif num_logical_shards:
            s_cnt = num_logical_shards
            # Map physical rank to one or more logical shards
            # For simplicity, we assume 1 worker -> 1 logical shard or range
            s_idx = rank % num_logical_shards
        else:
            s_idx, s_cnt = rank, world_size

        sampler = static_args.get("sampler")
        if sampler is None:
            num_records = len(data_source)
            sampler = grain.IndexSampler(
                num_records=num_records,
                shard_options=grain.ShardOptions(
                    shard_index=s_idx,
                    shard_count=s_cnt,
                    drop_remainder=static_args.get("drop_remainder", False),
                ),
                shuffle=static_args.get("shuffle", True),
                seed=static_args.get("seed", 42),
                num_epochs=static_args.get("num_epochs"),
            )

        self._output_type = static_args.get("output_data_type")

        # DataLoader creation
        self._dataloader = grain.DataLoader(
            data_source=dataset,
            sampler=sampler,
            worker_count=static_args.get("num_threads", 0),
        )
        self._iterator = iter(self._dataloader)

        logger.info(
            f"[spmd_id={getattr(self, 'spmd_id', 'unknown')}] Rank {rank} mapped to Logical Shard {s_idx}/{s_cnt}. "
            f"Serving {len(sampler)} records."
        )

    @operator("read", is_source=True)
    def read(self) -> Any:
        return self.next_item()

    def next_item(self) -> Any:
        rank = getattr(self, "rank", 0)
        world_size = getattr(self, "world_size", 1)
        static_args = getattr(self, "static_args", {})

        if self._iterator is None:
            self._create_data_loader(static_args, rank, world_size)

        try:
            batch = next(self._iterator)
            if self._output_type and not isinstance(batch, self._output_type):
                if isinstance(batch, dict):
                    return self._output_type(**batch)
                return self._output_type(batch)
            return batch
        except StopIteration:
            return None
        except Exception as e:
            logger.error(f"GrainReader pipeline failure: {e}")
            raise

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the iterator for checkpointing."""
        if hasattr(self._iterator, "get_state"):
            return self._iterator.get_state()
        return {}

    def set_state(self, state: Dict[str, Any]):
        """Restores the iterator state from a checkpoint."""
        if hasattr(self._iterator, "set_state"):
            self._iterator.set_state(state)
