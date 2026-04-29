from abc import ABC, abstractmethod

from fluxway.common.types.configs import OperatorMeta, OperatorContext

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class ExecutionStatus(Enum):
    """Sentinels for operator execution status."""

    SUCCESS = auto()
    SKIPPED = auto()
    STALE_AFTER_RESTORE = auto()
    EMPTY_PROPAGATION = auto()


@dataclass
class ExecutionResult:
    """Standardized result for operator execution."""

    status: ExecutionStatus
    data: Any = None


class ExecutionDispatcher(ABC):
    """
    Abstract strategy for operator execution dispatching.
    Decides whether an operator should run at a given step,
    and computes the `OperatorContext` injected into the operator.
    """

    @abstractmethod
    def should_execute(
        self, physical_rank: int, physical_world_size: int, step: int
    ) -> bool:
        pass

    @abstractmethod
    def get_logical_context(
        self, physical_rank: int, physical_world_size: int, step: int
    ) -> OperatorContext:
        pass


class StandardDispatcher(ExecutionDispatcher):
    """Standard execution: runs on every step, logical == physical."""

    def should_execute(
        self, physical_rank: int, physical_world_size: int, step: int
    ) -> bool:
        return True

    def get_logical_context(
        self, physical_rank: int, physical_world_size: int, step: int
    ) -> OperatorContext:
        return OperatorContext(
            physical_rank=physical_rank,
            physical_world_size=physical_world_size,
            logical_rank=physical_rank,
            logical_world_size=physical_world_size,
            step=step,
            is_interleaved=False,
        )


class InterleavedDispatcher(ExecutionDispatcher):
    """
    Interleaved execution (Virtual SPMD):
    Only runs if the physical rank belongs to the active unit for the current step.
    Each unit acts as an isolated world of size `interleave_unit_size`.
    """

    def __init__(self, interleave_unit_size: int):
        self.interleave_unit_size = interleave_unit_size

    def should_execute(
        self, physical_rank: int, physical_world_size: int, step: int
    ) -> bool:
        unit_size = self.interleave_unit_size
        stride = physical_world_size // unit_size
        active_unit_idx = step % stride
        my_unit_idx = physical_rank // unit_size
        return my_unit_idx == active_unit_idx

    def get_logical_context(
        self, physical_rank: int, physical_world_size: int, step: int
    ) -> OperatorContext:
        logical_rank = physical_rank % self.interleave_unit_size
        return OperatorContext(
            physical_rank=physical_rank,
            physical_world_size=physical_world_size,
            logical_rank=logical_rank,
            logical_world_size=self.interleave_unit_size,
            step=step,
            is_interleaved=True,
        )


def create_dispatcher(meta: OperatorMeta) -> ExecutionDispatcher:
    """Factory to create dispatcher strategy based on operator metadata."""
    if getattr(meta, "is_interleaved", False):
        return InterleavedDispatcher(
            interleave_unit_size=getattr(meta, "interleave_unit_size", 1)
        )
    return StandardDispatcher()
