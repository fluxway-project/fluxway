"""
Base class for all Reader SPMDs.
"""

from typing import Any

from dataclasses import dataclass
from ..common.decorators import spmd, operator, data_type
from ..common.types import Data


@data_type()
@dataclass
class ReaderBatch:
    """Generic batch returned by a reader with flattened fields."""

    ids: Data[Any]
    values: Data[Any]


@spmd()
class BaseReader:
    """
    Abstract base class for data readers.

    A Reader is just a specialized SPMD that has no input operators.
    It produces data and sends it downstream.
    """

    def __init__(self):
        # Framework will inject static_args into the instance if needed,
        # but usually we read from self.static_args which is part of SPMD config.
        pass

    @operator("read")
    def read(self) -> ReaderBatch:
        """
        Default read operator. Subclasses should override this or define their own.

        Returns:
            A dictionary of tensors to be sent downstream.
        """
        raise NotImplementedError("Subclasses must implement read()")
