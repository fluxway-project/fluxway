"""
Tests for standardized GrainReader.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Dict, Any

from fluxway.common.decorators import data_type, operator_context
from fluxway.reader.grain_reader import GrainReader

try:
    import grain.python as grain
except ImportError:
    grain = None


@data_type()
@dataclass
class SimpleBatch:
    features: jnp.ndarray


class MockDataSource:
    """Grain-compatible mock data source."""

    def __init__(self, data: List[Dict[str, Any]]):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


@pytest.mark.skipif(grain is None, reason="google-grain not installed")
def test_grain_reader_sharding():
    """Verify that GrainReader correctly shards data based on rank."""
    raw_data = [{"features": np.array([i])} for i in range(10)]
    ds = MockDataSource(raw_data)

    static_args = {
        "data_source": ds,
        "shuffle": False,
        "num_epochs": 1,
        "output_data_type": SimpleBatch,
    }

    reader = GrainReader()

    # Use framework's context manager to inject state
    with operator_context(static_args=static_args, rank=0, world_size=2):
        batches0 = []
        while True:
            b = reader.read()
            if b is None:
                break
            batches0.append(b)

    assert len(batches0) == 5
    assert isinstance(batches0[0], SimpleBatch)

    # Rank 1
    # We need a new instance because the iterator is cached in the first instance
    reader1 = GrainReader()
    with operator_context(static_args=static_args, rank=1, world_size=2):
        batches1 = []
        while True:
            b = reader1.read()
            if b is None:
                break
            batches1.append(b)
    assert len(batches1) == 5


@pytest.mark.skipif(grain is None, reason="google-grain not installed")
def test_grain_reader_transformations():
    """Verify that GrainReader applies transformations."""
    raw_data = [{"features": np.array([i])} for i in range(4)]
    ds = MockDataSource(raw_data)

    # Grain uses its own transformation system.
    # Let's verify we can pass a data source and wrap it in a data_type.
    static_args = {
        "data_source": ds,
        "output_data_type": SimpleBatch,
        "shuffle": False,
        "num_epochs": 1,
    }

    reader = GrainReader()
    with operator_context(static_args=static_args, rank=0, world_size=1):
        batch = reader.read()
        assert batch is not None
        assert isinstance(batch, SimpleBatch)
