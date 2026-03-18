"""
Helper SPMD nodes for reader end-to-end tests.

Defined in a separate module (not the test script itself) so Ray workers can
correctly import and deserialize them across the cluster.
"""

from dataclasses import dataclass

import numpy as np

from fluxway.common.decorators import data_type, operator, spmd
from fluxway.common.types import Data
from fluxway.reader.grain_reader import GrainReader


@data_type()
@dataclass
class SimpleData:
    id: Data[int]
    value: Data[float]


class MockDataSource:
    """Grain-compatible in-memory data source for tests."""

    def __init__(self, size: int):
        self._data = [
            {"id": np.array([i]), "value": np.array([i * 1.5])} for i in range(size)
        ]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int):
        return self._data[index]


@spmd(mesh_axis_names=("data",), mesh_axis_shapes=(1,))
class SimpleProcessor:
    @operator(
        "process",
        input_sharding={"batch": {"type": "static", "axis": ["data"]}},
        output_sharding={"type": "static", "axis": ["data"]},
    )
    def process(self, batch: SimpleData) -> SimpleData:
        return SimpleData(id=batch.id, value=batch.value)


@spmd(mesh_axis_names=("data",), mesh_axis_shapes=(1,))
class TypedGrainReader(GrainReader):
    """
    A typed GrainReader that converts raw dict output from Grain into SimpleData.

    GrainReader.read() returns Any; this subclass fixes the return type so the
    graph compiler knows the output sharding specification.
    """

    @operator(
        "read", is_source=True, output_sharding={"type": "static", "axis": ["data"]}
    )
    def read(self) -> SimpleData:
        raw = super().read()
        if raw is None:
            return None
        if isinstance(raw, dict):
            return SimpleData(id=raw["id"], value=raw["value"])
        return raw
