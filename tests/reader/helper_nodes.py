"""
Helper nodes for reader end-to-end tests.

Defining these classes in a separate module (instead of the test script itself)
ensures that Ray workers can correctly import and deserialize them across the cluster.
"""

from dataclasses import dataclass
from fluxway.common.decorators import spmd, operator, data_type
from fluxway.common.types import Data
from fluxway.reader.base_reader import ReaderBatch


@data_type()
@dataclass
class SimpleData:
    id: Data[int]
    value: Data[float]


@spmd()
class SimpleProcessor:
    @operator("process")
    def process(self, batch: ReaderBatch) -> ReaderBatch:
        # Just pass through.
        # Now batch.ids and batch.values are automatically detected as Data plane tensors.
        return batch
