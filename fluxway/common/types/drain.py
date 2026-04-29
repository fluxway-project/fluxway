"""
Drain mechanism strategy and types.

Drain is an execution barrier that limits how many steps a worker can process
ahead of the control plane. It natively separates the execution control from
the persistence mechanism (like Checkpointing).
"""

from abc import ABC, abstractmethod
from enum import Enum


class DrainMode(str, Enum):
    GRAPH = "graph"
    EAGER = "eager"


class DrainStrategy(ABC):
    """
    Abstract strategy for calculating when to drain the pipeline.
    """

    @abstractmethod
    def get_initial_drain_target(self, baseline: int = 0) -> int | None:
        """Get the step target for the very first drain ceiling."""
        pass

    @abstractmethod
    def get_next_drain_target(self, current_completed_step: int) -> int | None:
        """Given the current completed step, calculate the next drain target."""
        pass


class GraphModeDrainStrategy(DrainStrategy):
    """
    Graph mode: drain periodically based on a fixed interval of steps.
    """

    def __init__(self, interval_steps: int):
        if interval_steps < 1:
            raise ValueError(f"interval_steps must be >= 1, got {interval_steps}")
        self.interval_steps = interval_steps

    def get_initial_drain_target(self, baseline: int = 0) -> int | None:
        return baseline + self.interval_steps

    def get_next_drain_target(self, current_completed_step: int) -> int | None:
        return current_completed_step + self.interval_steps


class EagerModeDrainStrategy(DrainStrategy):
    """
    (TODO) Eager mode: drain immediately at a specified step or operate step-by-step.
    In the future, this will receive dynamic step targets from the eagerness shell.
    """

    def __init__(self):
        pass

    def get_initial_drain_target(self, baseline: int = 0) -> int | None:
        raise NotImplementedError("Eager mode drain is not yet implemented")

    def get_next_drain_target(self, current_completed_step: int) -> int | None:
        raise NotImplementedError("Eager mode drain is not yet implemented")
