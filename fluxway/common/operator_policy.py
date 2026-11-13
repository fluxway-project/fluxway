from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AfterValidator, ConfigDict, Field, PlainSerializer

from fluxway.common.frozen_mapping import freeze_mapping
from fluxway.common.schema import FluxwayBaseSchema


type IntMap = Annotated[
    Mapping[str, int],
    AfterValidator(freeze_mapping),
    PlainSerializer(lambda value: dict(value), return_type=dict),
]


class ArgumentType(StrEnum):
    CONTROL = "control"
    DATA = "data"
    STATIC = "static"


class BarrierType(StrEnum):
    STANDARD = "standard"
    PERIODIC = "periodic"


class BarrierConfig(FluxwayBaseSchema):
    type: BarrierType = BarrierType.STANDARD


class StandardBarrierConfig(BarrierConfig):
    type: BarrierType = BarrierType.STANDARD


class PeriodicBarrierConfig(BarrierConfig):
    type: BarrierType = BarrierType.PERIODIC
    n: int = 1


class StatePolicy(FluxwayBaseSchema):
    model_config = ConfigDict(frozen=True, extra="forbid")

    access: str = "exclusive"
    resource: str | None = None

    @classmethod
    def exclusive(
        cls,
        resource: str | None = None,
    ) -> StatePolicy:
        return cls(
            access="exclusive",
            resource=resource,
        )

    @classmethod
    def thread_safe(
        cls,
    ) -> StatePolicy:
        return cls(
            access="thread_safe",
            resource=None,
        )

    def model_post_init(self, __context: Any) -> None:
        if self.access not in ("exclusive", "thread_safe"):
            raise ValueError(
                f"StatePolicy.access must be 'exclusive' or 'thread_safe', got {self.access!r}."
            )
        if self.resource is not None and not isinstance(self.resource, str):
            raise ValueError(
                f"StatePolicy.resource must be a string or None, got {type(self.resource).__name__}."
            )
        if self.access == "thread_safe" and self.resource is not None:
            raise ValueError(
                "StatePolicy.thread_safe() cannot be combined with a serialized resource."
            )


class SchedulePolicy(FluxwayBaseSchema):
    model_config = ConfigDict(populate_by_name=True, frozen=True, extra="forbid")

    kind: str = "fifo"
    base_priority: int = 0
    boost_priority: int = 100
    trigger: str = "staleness_limit"
    phase: str | None = None
    phase_ordering: tuple[str, ...] = Field(default=(), alias="phase_order")
    phase_dependencies: tuple[tuple[str, str, int], ...] = ()
    lane: str | None = None
    lane_ordering: tuple[str, ...] = Field(default=(), alias="lane_order")
    lane_priorities: IntMap = Field(default_factory=dict)
    phase_priorities: IntMap = Field(default_factory=dict)
    micro_step_key: str = "step"

    @classmethod
    def fifo(
        cls,
        base_priority: int = 0,
        *,
        lane: str | None = None,
        lane_order: tuple[str, ...] = (),
        lane_priorities: dict[str, int] | None = None,
    ) -> SchedulePolicy:
        return cls(
            kind="fifo",
            base_priority=base_priority,
            lane=lane,
            lane_order=lane_order,
            lane_priorities={} if lane_priorities is None else lane_priorities,
        )

    @classmethod
    def deadline_boost(
        cls,
        base_priority: int = 0,
        boost_priority: int = 100,
        trigger: str = "staleness_limit",
        *,
        lane: str | None = None,
        lane_order: tuple[str, ...] = (),
        lane_priorities: dict[str, int] | None = None,
    ) -> SchedulePolicy:
        return cls(
            kind="deadline_boost",
            base_priority=base_priority,
            boost_priority=boost_priority,
            trigger=trigger,
            lane=lane,
            lane_order=lane_order,
            lane_priorities={} if lane_priorities is None else lane_priorities,
        )

    @classmethod
    def phase_order(
        cls,
        *,
        phase: str,
        phase_order: tuple[str, ...],
        micro_step_key: str = "step",
        base_priority: int = 0,
        lane: str | None = None,
        lane_order: tuple[str, ...] = (),
        lane_priorities: dict[str, int] | None = None,
    ) -> SchedulePolicy:
        return cls(
            kind="phase_order",
            phase=phase,
            phase_order=phase_order,
            micro_step_key=micro_step_key,
            base_priority=base_priority,
            lane=lane,
            lane_order=lane_order,
            lane_priorities={} if lane_priorities is None else lane_priorities,
        )

    @classmethod
    def phase_graph(
        cls,
        *,
        phase: str,
        phase_order: tuple[str, ...],
        dependencies: tuple[tuple[str, str, int], ...],
        lane: str | None = None,
        lane_order: tuple[str, ...] = (),
        lane_priorities: dict[str, int] | None = None,
        phase_priorities: dict[str, int] | None = None,
        micro_step_key: str = "step",
        base_priority: int = 0,
    ) -> SchedulePolicy:
        return cls(
            kind="phase_graph",
            phase=phase,
            phase_order=phase_order,
            phase_dependencies=dependencies,
            lane=lane,
            lane_order=lane_order,
            lane_priorities={} if lane_priorities is None else lane_priorities,
            phase_priorities={} if phase_priorities is None else phase_priorities,
            micro_step_key=micro_step_key,
            base_priority=base_priority,
        )

    def model_post_init(self, __context: Any) -> None:
        if self.kind not in ("fifo", "deadline_boost", "phase_order", "phase_graph"):
            raise ValueError(f"Unsupported SchedulePolicy kind {self.kind!r}.")
        if self.trigger != "staleness_limit":
            raise ValueError(
                f"SchedulePolicy.trigger must be 'staleness_limit', got {self.trigger!r}."
            )
        if self.micro_step_key != "step":
            raise ValueError(
                f"SchedulePolicy.micro_step_key must be 'step', got {self.micro_step_key!r}."
            )
        if self.kind in ("phase_order", "phase_graph"):
            if not self.phase:
                raise ValueError(f"SchedulePolicy.{self.kind} requires a phase.")
            if not self.phase_ordering:
                raise ValueError(f"SchedulePolicy.{self.kind} requires phase_order.")
            if self.phase not in self.phase_ordering:
                raise ValueError(
                    f"phase {self.phase!r} must be present in phase_order={self.phase_ordering!r}."
                )
        if self.kind == "phase_graph":
            self._validate_phase_graph()
        self._validate_lane_schedule()

    def _validate_phase_graph(self) -> None:
        phase_order = tuple(self.phase_ordering)
        if not self.phase_dependencies:
            raise ValueError("SchedulePolicy.phase_graph requires dependencies.")
        seen: set[tuple[str, str, int]] = set()
        same_step_edges: dict[str, set[str]] = {phase: set() for phase in phase_order}
        for dependency in self.phase_dependencies:
            if len(dependency) != 3:
                raise ValueError(
                    "SchedulePolicy.phase_graph dependencies must be "
                    "(before_phase, after_phase, step_delta) triples."
                )
            before, after, step_delta = dependency
            if not before or not after:
                raise ValueError(
                    "SchedulePolicy.phase_graph dependency phases must be non-empty."
                )
            if before not in phase_order:
                raise ValueError(
                    f"phase_graph dependency before phase {before!r} is not in "
                    f"phase_order={phase_order!r}."
                )
            if after not in phase_order:
                raise ValueError(
                    f"phase_graph dependency after phase {after!r} is not in "
                    f"phase_order={phase_order!r}."
                )
            if not isinstance(step_delta, int):
                raise ValueError(
                    "SchedulePolicy.phase_graph dependency step_delta must be an int."
                )
            if step_delta < 0:
                raise ValueError(
                    "SchedulePolicy.phase_graph dependency step_delta must be non-negative."
                )
            if before == after and step_delta == 0:
                raise ValueError(
                    "SchedulePolicy.phase_graph same-step self-dependencies are invalid."
                )
            key = (before, after, step_delta)
            if key in seen:
                raise ValueError(
                    f"SchedulePolicy.phase_graph duplicate dependency {key!r}."
                )
            seen.add(key)
            if step_delta == 0:
                same_step_edges[before].add(after)
        self._validate_phase_graph_same_step_dag(same_step_edges)

    @staticmethod
    def _validate_phase_graph_same_step_dag(edges: dict[str, set[str]]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(phase: str) -> None:
            if phase in visited:
                return
            if phase in visiting:
                raise ValueError(
                    "SchedulePolicy.phase_graph same-step dependencies must be acyclic."
                )
            visiting.add(phase)
            for successor in edges[phase]:
                visit(successor)
            visiting.remove(phase)
            visited.add(phase)

        for phase in edges:
            visit(phase)

    def _validate_lane_schedule(self) -> None:
        if self.lane is None:
            if self.lane_ordering:
                raise ValueError("SchedulePolicy.lane_order requires lane.")
            if self.lane_priorities:
                raise ValueError("SchedulePolicy.lane_priorities requires lane.")
            return
        if not self.lane:
            raise ValueError("SchedulePolicy.lane must be non-empty when provided.")
        if self.lane_ordering and self.lane not in self.lane_ordering:
            raise ValueError(
                f"lane {self.lane!r} must be present in lane_order={self.lane_ordering!r}."
            )
        unknown_lanes = set(self.lane_priorities).difference(self.lane_ordering)
        if unknown_lanes and self.lane_ordering:
            raise ValueError(
                f"lane_priorities references lanes outside lane_order: {sorted(unknown_lanes)}."
            )
        if self.phase_priorities:
            phase_order = set(self.phase_ordering)
            unknown_phases = set(self.phase_priorities).difference(phase_order)
            if unknown_phases:
                raise ValueError(
                    f"phase_priorities references phases outside phase_order: {sorted(unknown_phases)}."
                )


class AsyncPolicy(FluxwayBaseSchema):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: str = "sync"
    depth: int = 1
    commit: str = "ordered"
    state: StatePolicy = Field(default_factory=StatePolicy.exclusive)
    schedule: SchedulePolicy = Field(default_factory=SchedulePolicy.fifo)
    max_staleness: int | None = None

    @classmethod
    def sync(
        cls,
        schedule: SchedulePolicy | None = None,
        state: StatePolicy | None = None,
    ) -> AsyncPolicy:
        return cls(
            mode="sync",
            schedule=SchedulePolicy.fifo() if schedule is None else schedule,
            state=StatePolicy.exclusive() if state is None else state,
        )

    @classmethod
    def pipelined(
        cls,
        depth: int,
        commit: str = "ordered",
        schedule: SchedulePolicy | None = None,
        state: StatePolicy | None = None,
    ) -> AsyncPolicy:
        return cls(
            mode="pipelined",
            depth=depth,
            commit=commit,
            schedule=SchedulePolicy.fifo() if schedule is None else schedule,
            state=StatePolicy.exclusive() if state is None else state,
        )

    @classmethod
    def bounded_stale(
        cls,
        depth: int,
        max_staleness: int,
        commit: str = "relaxed",
        schedule: SchedulePolicy | None = None,
        state: StatePolicy | None = None,
    ) -> AsyncPolicy:
        return cls(
            mode="bounded_stale",
            depth=depth,
            commit=commit,
            max_staleness=max_staleness,
            schedule=SchedulePolicy.deadline_boost() if schedule is None else schedule,
            state=StatePolicy.thread_safe() if state is None else state,
        )

    def model_post_init(self, __context: Any) -> None:
        if self.mode not in ("sync", "pipelined", "bounded_stale"):
            raise ValueError(f"Unsupported AsyncPolicy mode {self.mode!r}.")
        if self.depth <= 0:
            raise ValueError(f"AsyncPolicy.depth must be positive, got {self.depth}.")
        if self.commit not in ("ordered", "relaxed"):
            raise ValueError(
                f"AsyncPolicy.commit must be 'ordered' or 'relaxed', got {self.commit!r}."
            )
        if self.mode == "sync" and self.depth != 1:
            raise ValueError("AsyncPolicy.sync() must use depth=1.")
        if self.mode == "bounded_stale":
            if self.max_staleness is None:
                raise ValueError("AsyncPolicy.bounded_stale requires max_staleness.")
            if self.max_staleness < 0:
                raise ValueError(
                    f"AsyncPolicy.max_staleness must be non-negative, got {self.max_staleness}."
                )
            if self.depth > self.max_staleness + 1:
                raise ValueError(
                    "AsyncPolicy.bounded_stale requires depth <= max_staleness + 1, "
                    f"got depth={self.depth} max_staleness={self.max_staleness}."
                )
        elif self.max_staleness is not None:
            raise ValueError(
                f"AsyncPolicy mode {self.mode!r} cannot define max_staleness."
            )

    def compile(self) -> CompiledAsyncPolicy:
        max_async_lag = (
            self.depth - 1 if self.mode == "pipelined" else self.max_staleness
        )
        return CompiledAsyncPolicy(
            mode=self.mode,
            pipeline_depth=self.depth,
            commit_policy=self.commit,
            max_async_lag=max_async_lag,
            max_inflight_steps=0 if max_async_lag is None else max_async_lag,
            state_access=self.state.access,
            state_resource=self.state.resource,
            schedule=self.schedule,
            priority=self.schedule.base_priority,
            max_staleness=self.max_staleness,
        )


@dataclass(frozen=True, slots=True)
class CompiledAsyncPolicy:
    mode: str
    pipeline_depth: int
    commit_policy: str
    max_async_lag: int | None
    max_inflight_steps: int
    state_access: str
    state_resource: str | None
    schedule: SchedulePolicy
    priority: int
    max_staleness: int | None = None

    @property
    def is_pipelined(self) -> bool:
        return self.pipeline_depth > 1

    @property
    def has_bounded_staleness(self) -> bool:
        return self.max_staleness is not None
