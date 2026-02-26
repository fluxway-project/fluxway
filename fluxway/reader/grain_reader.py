"""
Reader implementation using Google Grain.
"""

import logging
from typing import Any, Dict, List
import numpy as np
import json
import os

from ..common.decorators import spmd, operator
from .base_reader import BaseReader, ReaderBatch

logger = logging.getLogger(__name__)

# Attempt to import grain (will be used inside classes)
try:
    import grain.python as grain
except ImportError:
    grain = None


class JsonDataSource:
    """Simple JSON data source for Google Grain."""

    def __init__(self, files: List[str]):
        self._data = []
        for f in files:
            if not os.path.isfile(f):
                continue
            try:
                with open(f, "r") as json_f:
                    for line in json_f:
                        if not line.strip():
                            continue
                        self._data.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Failed to read JSON file {f}: {e}")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


@spmd()
class GrainReader(BaseReader):
    """
    Reader implementation leveraging Google Grain for efficient data loading and sharding.
    """

    def __init__(self):
        super().__init__()
        self._dataloader = None
        self._iterator = None

    def _initialize_grain(
        self, static_args: Dict[str, Any], rank: int, world_size: int
    ):
        """Initialize Google Grain dataloader."""
        if grain is None:
            raise ImportError("google-grain not installed")

        src_dir = static_args.get("src_dir")
        file_pattern = static_args.get("file_pattern", "*")
        batch_size = static_args.get("batch_size", 1)

        logger.info(
            f"[spmd_id={getattr(self, 'spmd_id', 'unknown')}] Initializing GrainReader rank={rank}/{world_size} "
            f"at {src_dir} with pattern {file_pattern}, batch_size={batch_size}"
        )

        try:
            if not os.path.isdir(src_dir):
                raise ValueError(f"Source directory {src_dir} does not exist")

            files = [
                os.path.join(src_dir, f)
                for f in os.listdir(src_dir)
                if f.endswith(".json")
            ]
            if not files:
                raise ValueError(f"No JSON files found in {src_dir}")

            # 1. Create Data Source
            ds = JsonDataSource(files)
            if len(ds) == 0:
                raise ValueError(f"No data found in JSON files in {src_dir}")

            # 2. Create MapDataset
            dataset = grain.MapDataset.source(ds)

            # 3. Sharding via IndexSampler
            index_sampler = grain.IndexSampler(
                num_records=len(dataset),
                shard_options=grain.ShardOptions(
                    shard_index=rank, shard_count=world_size, drop_remainder=False
                ),
                shuffle=True,
                seed=42,
            )

            # 4. Create DataLoader
            self._dataloader = grain.DataLoader(
                data_source=dataset, sampler=index_sampler, worker_count=0
            )
            self._iterator = iter(self._dataloader)
            logger.info(f"GrainReader initialized successfully with {len(ds)} records")

        except Exception as e:
            logger.error(f"Failed to initialize GrainReader: {e}")
            raise

    @operator("read")
    def read(self) -> ReaderBatch:
        """
        Produce a batch of data using Google Grain.
        """
        if self._iterator is None:
            # Context should have been injected by the wrapper
            static_args = getattr(self, "static_args", {})
            rank = getattr(self, "rank", 0)
            world_size = getattr(self, "world_size", 1)
            self._initialize_grain(static_args, rank, world_size)

        try:
            # For simplicity, we read one record at a time
            record = next(self._iterator)
            print(
                f"[spmd_id={getattr(self, 'spmd_id', 'unknown')} rank={getattr(self, 'rank', 0)}] Read record: {record}"
            )

            # Flatten record to ReaderBatch fields
            # Ensure output arrays are at least 1D for TransportPlanner
            if isinstance(record, dict):
                batch = ReaderBatch(
                    ids=np.atleast_1d(np.array(record.get("id", -1))),
                    values=np.atleast_1d(np.array(record.get("value", 0.0))),
                )
            else:
                batch = ReaderBatch(
                    ids=np.atleast_1d(np.array([-1])),
                    values=np.atleast_1d(np.array(record)),
                )

            print(f"  - Produced batch object: {batch}")
            # Ensure it's a dict for easy extraction if _extract_tensors expects it
            # Actually with my recent change to worker.py, it should handle BaseModel or as_dict
            return batch

        except StopIteration:
            logger.info("GrainReader reached end of dataset")
            return None
        except Exception as e:
            logger.error(f"Error reading from Grain: {e}")
            return None
