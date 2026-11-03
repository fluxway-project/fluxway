from __future__ import annotations

import threading
import numpy as np

from fluxway.profiler.record import TraceMetadata


class StepFilter:
    __slots__ = (
        "_skip_initial_steps",
        "_step_interval",
        "_target_steps",
        "_max_steps",
        "_collected_steps",
        "_lock",
        "_has_filter",
    )

    def __init__(
        self,
        *,
        skip_initial_steps: int = 0,
        step_interval: int = 1,
        target_steps: frozenset[int] | None = None,
        max_steps: int | None = None,
    ) -> None:
        if (
            isinstance(skip_initial_steps, bool)
            or not isinstance(skip_initial_steps, int)
            or skip_initial_steps < 0
        ):
            raise ValueError("skip_initial_steps must be non-negative.")
        if (
            isinstance(step_interval, bool)
            or not isinstance(step_interval, int)
            or step_interval <= 0
        ):
            raise ValueError("step_interval must be positive.")
        if target_steps is not None:
            if not isinstance(target_steps, frozenset):
                raise TypeError("target_steps must be a frozenset of integers or None.")
            if any(
                isinstance(s, bool) or not isinstance(s, int) or s < 0
                for s in target_steps
            ):
                raise ValueError("target_steps elements must be non-negative integers.")
        if max_steps is not None:
            if (
                isinstance(max_steps, bool)
                or not isinstance(max_steps, int)
                or max_steps <= 0
            ):
                raise ValueError("max_steps must be positive.")
        self._skip_initial_steps = skip_initial_steps
        self._step_interval = step_interval
        self._target_steps = target_steps
        self._max_steps = max_steps
        self._collected_steps: set[int] = set()
        self._lock = threading.Lock()
        self._has_filter = (
            skip_initial_steps > 0
            or step_interval > 1
            or target_steps is not None
            or max_steps is not None
        )

    @property
    def has_filter(self) -> bool:
        return self._has_filter

    def is_step_profiled(self, step: int) -> bool:
        if not self._has_filter:
            return True
        if step < self._skip_initial_steps:
            return False
        if self._target_steps is not None and step not in self._target_steps:
            return False
        if self._step_interval > 1:
            if (step - self._skip_initial_steps) % self._step_interval != 0:
                return False
        if self._max_steps is not None:
            if step in self._collected_steps:
                return True
            with self._lock:
                if step in self._collected_steps:
                    return True
                if len(self._collected_steps) >= self._max_steps:
                    return False
                self._collected_steps.add(step)
                return True
        return True


STEADY_STATE_EVENT_NAMES = frozenset(
    {
        "controller.initialize",
        "controller.completion",
        "controller.reclaim",
        "driver.initialize",
        "driver.start_workers",
        "spmd_manager.placement_group_create",
        "spmd_manager.driver_start",
        "spmd_manager.driver_stop",
    }
)


def filter_steady_state_records(
    records: np.ndarray,
    metadata: TraceMetadata,
) -> np.ndarray:
    if records.ndim != 2:
        raise ValueError(f"records must be a 2D array, got ndim={records.ndim}.")
    if len(records) == 0:
        return records
    retention = np.ones((len(records),), dtype=bool)
    for index, row in enumerate(records):
        track_name = metadata.name_for(int(row[3]))
        event_name = metadata.name_for(int(row[4]))
        if "/lifecycle" in track_name or event_name in STEADY_STATE_EVENT_NAMES:
            retention[index] = False
    return records[retention]
