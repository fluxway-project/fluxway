"""Optional low-overhead profiler runtime for Fluxway."""

from fluxway.profiler.config import ProfilerConfig, ProfilerDiagnostic
from fluxway.profiler.errors import (
    CausalClockViolation,
    ClockSyncUnstable,
    GraphTraceSymbolConflict,
    MissingAxonNativeHlc,
    MissingGraphTraceSymbol,
    MissingTraceMetadata,
    ProfilerBufferExhausted,
)
from fluxway.profiler.record import TraceCarrier, TraceEventKind, TraceMetadata
from fluxway.profiler.runtime import ProfilerRuntimeFactory
from fluxway.profiler.user_api import (
    async_scope,
    enabled,
    event,
    function_scope,
    launch_scope,
    scope,
    start_async_operation,
)

__all__ = [
    "CausalClockViolation",
    "ClockSyncUnstable",
    "GraphTraceSymbolConflict",
    "MissingAxonNativeHlc",
    "MissingGraphTraceSymbol",
    "MissingTraceMetadata",
    "ProfilerBufferExhausted",
    "ProfilerConfig",
    "ProfilerDiagnostic",
    "ProfilerRuntimeFactory",
    "TraceCarrier",
    "TraceEventKind",
    "TraceMetadata",
    "async_scope",
    "enabled",
    "event",
    "function_scope",
    "launch_scope",
    "scope",
    "start_async_operation",
]
