from __future__ import annotations

from typing import Any, cast

from fluxway.profiler.config import ProfilerConfig, ProfilerDiagnostic
from fluxway.profiler.clock import ClockCalibration
from fluxway.profiler.diagnostics import HeartbeatProfilerObserver
from fluxway.profiler.record import ProfilerSnapshot, TraceArena
from fluxway.profiler.recorder import TraceRecorder
from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.profiler.wrappers import (
    TracingDataPlaneBackendExecutor,
    TracingDataPlaneCallable,
    TracingLocalStepPlanExecutor,
    TracingUserCallableExecutor,
    TracingWorkerMailbox,
)


class ProfilerRuntimeFactory:
    def __init__(
        self,
        recorder: TraceRecorder | None = None,
        diagnostics: frozenset[ProfilerDiagnostic] = frozenset(),
    ) -> None:
        self._recorder = recorder
        self._diagnostics = diagnostics

    @classmethod
    def from_config(
        cls,
        config: ProfilerConfig | None,
        *,
        process_name: str,
    ) -> ProfilerRuntimeFactory:
        if config is None:
            return cls()
        arena = TraceArena(config.buffer_bytes_per_process)
        recorder = TraceRecorder(
            arena,
            run_id=config.run_id,
            process_name=process_name,
            clock_calibration=ClockCalibration.measure(
                rounds=config.clock_sync_rounds,
                max_uncertainty_ns=config.max_clock_uncertainty_ns,
            ),
            step_filter=config.create_step_filter(),
        )
        return cls(
            recorder,
            diagnostics=cast(frozenset[ProfilerDiagnostic], config.diagnostics),
        )

    @property
    def enabled(self) -> bool:
        return self._recorder is not None

    @property
    def recorder(self) -> TraceRecorder:
        if self._recorder is None:
            raise RuntimeError("Profiler runtime is disabled.")
        return self._recorder

    def data_plane_backend_executor(self, executor: Any) -> Any:
        if self._recorder is None:
            return executor
        return TracingDataPlaneBackendExecutor(executor, self._recorder)

    def local_step_plan_executor(self, executor: Any) -> Any:
        if self._recorder is None:
            return executor
        return TracingLocalStepPlanExecutor(executor, self._recorder)

    def user_callable_executor(self, executor: Any) -> Any:
        if self._recorder is None:
            return executor
        return TracingUserCallableExecutor(executor, self._recorder)

    def create_worker_mailbox(self, **kwargs: Any) -> WorkerMailbox:
        if self._recorder is None:
            return WorkerMailbox(**kwargs)
        return TracingWorkerMailbox(self._recorder, **kwargs)

    def heartbeat_observer(self) -> HeartbeatProfilerObserver | None:
        if self._recorder is None:
            return None
        if ProfilerDiagnostic.HEARTBEAT not in self._diagnostics:
            return None
        return HeartbeatProfilerObserver(self._recorder)

    def heartbeat_diagnostics_enabled(self) -> bool:
        return (
            self._recorder is not None
            and ProfilerDiagnostic.HEARTBEAT in self._diagnostics
        )

    def wrap_data_plane_handlers(
        self, spmd_id: str, handlers: dict[int, Any]
    ) -> dict[int, Any]:
        if self._recorder is None:
            return handlers
        return {
            rank: TracingDataPlaneCallable(
                handler,
                self._recorder,
                remote_spmd_id=spmd_id,
                remote_rank=rank,
            )
            for rank, handler in handlers.items()
        }

    def snapshot(self) -> ProfilerSnapshot | None:
        if self._recorder is None:
            return None
        return self._recorder.snapshot()
