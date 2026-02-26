"""
Standardized Reader implementation leveraging Google Grain.
"""

import logging
from typing import Any, Dict

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

    This class acts as an orchestration engine for Grain's data pipelines.
    It automatically handles distributed sharding based on the SPMD worker's
    rank and world size.
    """

    def __init__(self):
        super().__init__()
        self._dataloader = None
        self._iterator = None
        self._output_type = None

    def _create_data_loader(
        self, static_args: Dict[str, Any], rank: int, world_size: int
    ):
        """
        Constructs the Google Grain DataLoader pipeline.
        """
        print(
            f"[GrainReader DEBUG] Starting _create_data_loader for rank {rank}/{world_size}"
        )
        if grain is None:
            raise ImportError("google-grain not found. Install it to use GrainReader.")

        # 1. Get DataSource
        data_source = static_args.get("data_source")
        if data_source is None:
            raise ValueError("GrainReader requires 'data_source' in static_args.")
        print(f"[GrainReader DEBUG] Data source length: {len(data_source)}")

        # 2. Create MapDataset
        print("[GrainReader DEBUG] Creating MapDataset source")
        dataset = grain.MapDataset.source(data_source)

        # 3. Apply Transformations
        transformations = static_args.get("transformations", [])
        print(f"[GrainReader DEBUG] Applying {len(transformations)} transformations")
        for tx in transformations:
            dataset = dataset.map(tx)

        # 4. Get Sampler (Default to IndexSampler with SPMD sharding)
        print("[GrainReader DEBUG] Creating IndexSampler")
        sampler = static_args.get("sampler")
        if sampler is None:
            num_records = len(data_source)
            sampler = grain.IndexSampler(
                num_records=num_records,
                shard_options=grain.ShardOptions(
                    shard_index=rank,
                    shard_count=world_size,
                    drop_remainder=static_args.get("drop_remainder", False),
                ),
                shuffle=static_args.get("shuffle", True),
                seed=static_args.get("seed", 42),
                num_epochs=static_args.get("num_epochs"),
            )
        print(
            f"[GrainReader DEBUG] Sampler created, expected records for this rank: {len(sampler)}"
        )

        # 5. Target output data type
        self._output_type = static_args.get("output_data_type")

        # 6. Build DataLoader
        print("[GrainReader DEBUG] Building DataLoader")
        self._dataloader = grain.DataLoader(
            data_source=dataset,
            sampler=sampler,
            worker_count=static_args.get("num_threads", 0),
        )
        print("[GrainReader DEBUG] DataLoader built, creating iterator")
        self._iterator = iter(self._dataloader)
        print("[GrainReader DEBUG] Iterator created")

        logger.info(
            f"[spmd_id={getattr(self, 'spmd_id', 'unknown')}] Grain pipeline initialized. "
            f"Rank {rank}/{world_size} serving {len(sampler)} records."
        )

    @operator("read")
    def read(self) -> Any:
        """
        Produces the next batch from the Grain iterator.
        """
        return self.next_item()

    def next_item(self) -> Any:
        """
        Internal implementation to fetch the next item from Grain iterator.
        This can be safely called by subclasses.
        """
        rank = getattr(self, "rank", 0)
        world_size = getattr(self, "world_size", 1)
        static_args = getattr(self, "static_args", {})

        # Check if rank or world_size changed (e.g. in multi-rank tests in same thread)
        if self._iterator is not None:
            if (
                getattr(self, "_initialized_rank", None) != rank
                or getattr(self, "_initialized_world_size", None) != world_size
            ):
                logger.info(
                    f"Re-initializing GrainReader for new rank/world_size: {rank}/{world_size}"
                )
                self._iterator = None

        if self._iterator is None:
            self._create_data_loader(static_args, rank, world_size)
            self._initialized_rank = rank
            self._initialized_world_size = world_size

        try:
            batch = next(self._iterator)

            # Cast to specialized data_type if requested
            if self._output_type and not isinstance(batch, self._output_type):
                if isinstance(batch, dict):
                    return self._output_type(**batch)
                return self._output_type(batch)

            return batch

        except StopIteration:
            logger.debug("Reader reached end of data source.")
            return None
        except Exception as e:
            logger.error(f"GrainReader pipeline failure: {e}")
            raise
