"""Source-neutral adapter contract for algebraic ingestion planning."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from fluxway.common.ingestion.source_scan import (
    OpaqueSourceCursor,
    SourceBindingIdentity,
    SourceScanBudget,
    SourceScanResult,
)
from fluxway.control_plane.ingestion.compiler import AssignmentPolicy


@runtime_checkable
class IngestionSourceAdapter(Protocol):
    """Bounded source scan result owner shared by all controlled source kinds."""

    def scan_next(
        self, cursor: OpaqueSourceCursor, budget: SourceScanBudget
    ) -> SourceScanResult | Awaitable[SourceScanResult]: ...

    def close(self) -> None | Awaitable[None]: ...

    def advance_cursor(
        self, cursor: OpaqueSourceCursor, records: int
    ) -> OpaqueSourceCursor: ...


@dataclass(frozen=True, slots=True)
class SourceScanBinding:
    """One source session owner for topology, scan adapter, and initial position."""

    adapter: IngestionSourceAdapter
    identity: SourceBindingIdentity
    initial_cursor_factory: Callable[
        [], OpaqueSourceCursor | None | Awaitable[OpaqueSourceCursor | None]
    ]
    topology: object
    tail_layout_mode: object
    assignment_policy: AssignmentPolicy = field(
        default_factory=AssignmentPolicy.default
    )

    def __post_init__(self) -> None:
        if not isinstance(self.adapter, IngestionSourceAdapter):
            raise TypeError("source binding requires an ingestion source adapter")
        if not isinstance(self.identity, SourceBindingIdentity):
            raise TypeError("source binding requires SourceBindingIdentity")
        if not callable(self.initial_cursor_factory):
            raise TypeError("source binding requires an initial cursor factory")
        if not isinstance(self.assignment_policy, AssignmentPolicy):
            object.__setattr__(
                self,
                "assignment_policy",
                AssignmentPolicy.from_value(self.assignment_policy),
            )


__all__ = [
    "IngestionSourceAdapter",
    "SourceScanBinding",
]
