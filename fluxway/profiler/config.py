from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from fluxway.profiler.filtering import StepFilter


class ProfilerDiagnostic(StrEnum):
    HEARTBEAT = "heartbeat"


class ProfilerLevel(StrEnum):
    SUMMARY = "summary"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


@dataclass(frozen=True, slots=True)
class ProfilerConfig:
    trace_dir: str
    run_id: str
    buffer_bytes_per_process: int = 64 * 1024 * 1024
    clock_sync_rounds: int = 8
    max_clock_uncertainty_ns: int = 1_000_000
    fail_on_buffer_exhaustion: bool = True
    output_file: str | None = None
    diagnostics: frozenset[ProfilerDiagnostic] = frozenset()
    skip_initial_steps: int = 0
    step_interval: int = 1
    target_steps: Iterable[int] | None = None
    max_steps: int | None = None
    steady_state_only: bool = False
    level: ProfilerLevel = ProfilerLevel.STANDARD

    def __post_init__(self) -> None:
        if not self.trace_dir:
            raise ValueError("ProfilerConfig.trace_dir must be non-empty.")
        if not self.run_id:
            raise ValueError("ProfilerConfig.run_id must be non-empty.")
        if not isinstance(self.level, ProfilerLevel):
            raise TypeError("ProfilerConfig.level must be a ProfilerLevel")
        if self.buffer_bytes_per_process <= 0:
            raise ValueError(
                "ProfilerConfig.buffer_bytes_per_process must be positive."
            )
        if self.clock_sync_rounds <= 0:
            raise ValueError("ProfilerConfig.clock_sync_rounds must be positive.")
        if self.max_clock_uncertainty_ns < 0:
            raise ValueError(
                "ProfilerConfig.max_clock_uncertainty_ns must be non-negative."
            )
        if self.target_steps is not None:
            if not isinstance(self.target_steps, Iterable) or isinstance(
                self.target_steps, (str, bytes)
            ):
                raise TypeError(
                    "ProfilerConfig.target_steps must be an iterable of integers."
                )
            target_steps_set = frozenset(self.target_steps)
            object.__setattr__(self, "target_steps", target_steps_set)
        self.create_step_filter()
        if not isinstance(self.diagnostics, frozenset):
            raise TypeError("ProfilerConfig.diagnostics must be a frozenset")
        if any(not isinstance(item, ProfilerDiagnostic) for item in self.diagnostics):
            raise TypeError(
                "ProfilerConfig.diagnostics must contain ProfilerDiagnostic values"
            )

    @property
    def run_dir(self) -> Path:
        return Path(self.trace_dir) / self.run_id

    @property
    def output_path(self) -> Path:
        if self.output_file is not None:
            return Path(self.output_file)
        return self.run_dir / "trace.pftrace"

    def create_step_filter(self) -> StepFilter:
        from fluxway.profiler.filtering import StepFilter

        return StepFilter(
            skip_initial_steps=self.skip_initial_steps,
            step_interval=self.step_interval,
            target_steps=cast(frozenset[int] | None, self.target_steps),
            max_steps=self.max_steps,
        )
