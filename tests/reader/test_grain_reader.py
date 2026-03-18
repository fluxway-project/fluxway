"""
Unit tests for GrainReader.

GrainReader.read() is a @operator-decorated staticmethod backed by a
per-class thread-local instance (see decorators._instance_storage).
Unit tests that need to exercise different rank configurations must
instantiate GrainReader directly and call next_item() / _create_data_loader()
rather than going through the staticmethod wrapper.
"""

import numpy as np
import pytest

try:
    import grain.python as grain
except ImportError:
    grain = None

from dataclasses import dataclass

from fluxway.common.decorators import data_type
from fluxway.reader.grain_reader import GrainReader


@data_type()
@dataclass
class SimpleBatch:
    features: np.ndarray


class MockDataSource:
    def __init__(self, size: int):
        self._data = [{"features": np.array([i])} for i in range(size)]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int):
        return self._data[index]


def _make_reader(data_source, rank: int, world_size: int, **kwargs) -> GrainReader:
    """
    Build a GrainReader with its iterator initialised for the given rank.

    Uses the same path as the production worker: injects static_args / rank /
    world_size as instance attributes, then calls _create_data_loader directly.
    """
    reader = GrainReader()
    reader.static_args = {
        "data_source": data_source,
        "shuffle": False,
        "num_epochs": 1,
        "output_data_type": SimpleBatch,
        **kwargs,
    }
    reader.rank = rank
    reader.world_size = world_size
    reader._create_data_loader(reader.static_args, rank, world_size)
    return reader


def _drain(reader: GrainReader):
    results = []
    while True:
        batch = reader.next_item()
        if batch is None:
            break
        results.append(batch)
    return results


@pytest.mark.skipif(grain is None, reason="google-grain not installed")
class TestGrainReaderSharding:
    """Verify that GrainReader partitions data correctly across ranks."""

    def test_two_shards_cover_all_records(self):
        ds = MockDataSource(10)
        rank0 = _drain(_make_reader(ds, rank=0, world_size=2))
        rank1 = _drain(_make_reader(ds, rank=1, world_size=2))

        assert len(rank0) == 5
        assert len(rank1) == 5
        assert all(isinstance(b, SimpleBatch) for b in rank0 + rank1)

        ids0 = {int(b.features[0]) for b in rank0}
        ids1 = {int(b.features[0]) for b in rank1}
        assert ids0.isdisjoint(ids1), "Shards must not overlap"
        assert ids0 | ids1 == set(range(10)), "Shards must cover all records"

    def test_single_shard_reads_all_records(self):
        ds = MockDataSource(4)
        batches = _drain(_make_reader(ds, rank=0, world_size=1))

        assert len(batches) == 4
        assert all(isinstance(b, SimpleBatch) for b in batches)


@pytest.mark.skipif(grain is None, reason="google-grain not installed")
class TestGrainReaderOutputType:
    """Verify that output_data_type wraps raw dicts into the target dataclass."""

    def test_output_type_conversion(self):
        ds = MockDataSource(4)
        reader = _make_reader(ds, rank=0, world_size=1)
        batch = reader.next_item()

        assert batch is not None
        assert isinstance(batch, SimpleBatch)
        np.testing.assert_array_equal(batch.features, np.array([0]))
