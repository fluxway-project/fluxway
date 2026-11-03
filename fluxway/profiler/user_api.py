from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import contextvars
import json
import time
from typing import Any

from fluxway.profiler.recorder import TraceRecorder


_ACTIVE_USER_PROFILER: contextvars.ContextVar[UserProfilerHandle | None] = (
    contextvars.ContextVar("fluxway_active_user_profiler", default=None)
)
_RESERVED_ATTR_KEYS = frozenset({"op_id", "step", "rank", "execution_mode", "attrs_id"})


class _NoopScope:
    __slots__ = ()

    def __enter__(self) -> _NoopScope:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        return False

    async def __aenter__(self) -> _NoopScope:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        return False

    def launched(self) -> None:
        pass

    def complete(self) -> None:
        pass


_NOOP_SCOPE = _NoopScope()


class UserScopeContext:
    __slots__ = (
        "_recorder",
        "_op_name",
        "_step",
        "_rank",
        "_execution_mode",
        "_name",
        "_attrs_json",
        "_start_ns",
        "_launch_start_ns",
        "_launch_dur_ns",
        "_launched",
        "_completed",
    )

    def __init__(
        self,
        *,
        recorder: TraceRecorder,
        op_name: str,
        step: int,
        rank: int,
        execution_mode: str,
        name: str,
        attrs_json: str,
    ) -> None:
        self._recorder = recorder
        self._op_name = op_name
        self._step = step
        self._rank = rank
        self._execution_mode = execution_mode
        self._name = name
        self._attrs_json = attrs_json
        self._start_ns = 0
        self._launch_start_ns = 0
        self._launch_dur_ns = 0
        self._launched = False
        self._completed = False

    def __enter__(self) -> UserScopeContext:
        self._start_ns = time.perf_counter_ns()
        self._launch_start_ns = self._start_ns
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        if not self._completed and self._start_ns > 0:
            dur_ns = time.perf_counter_ns() - self._start_ns
            self._record_slice(self._name, self._start_ns, dur_ns)
            self._completed = True
        return False

    async def __aenter__(self) -> UserScopeContext:
        self._start_ns = time.perf_counter_ns()
        self._launch_start_ns = self._start_ns
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        if not self._completed and self._start_ns > 0:
            dur_ns = time.perf_counter_ns() - self._start_ns
            self._record_slice(self._name, self._start_ns, dur_ns)
            self._completed = True
        return False

    def launched(self) -> None:
        if not self._launched and self._launch_start_ns > 0:
            self._launch_dur_ns = time.perf_counter_ns() - self._launch_start_ns
            self._record_slice(
                f"{self._name}.launch", self._launch_start_ns, self._launch_dur_ns
            )
            self._launched = True

    def complete(self) -> None:
        if not self._completed and self._start_ns > 0:
            dur_ns = time.perf_counter_ns() - self._start_ns
            self._record_slice(self._name, self._start_ns, dur_ns)
            self._completed = True

    def _record_slice(self, event_name: str, start_ns: int, dur_ns: int) -> None:
        spec = self._recorder._user_custom_spec(
            name=event_name, execution_mode=self._execution_mode
        )
        self._recorder._slice(
            spec,
            start_ns=start_ns,
            dur_ns=dur_ns,
            payload=(
                self._recorder.intern(self._op_name),
                self._step,
                self._rank,
                self._recorder.intern(self._execution_mode),
                self._recorder.intern(self._attrs_json),
            ),
        )


class UserProfilerHandle:
    __slots__ = ("_recorder", "_op_name", "_step", "_rank", "_execution_mode")

    def __init__(
        self,
        *,
        recorder: TraceRecorder,
        op_name: str,
        step: int,
        rank: int,
        execution_mode: str,
    ) -> None:
        self._recorder = recorder
        self._op_name = op_name
        self._step = step
        self._rank = rank
        self._execution_mode = execution_mode

    @property
    def enabled(self) -> bool:
        return self._recorder.is_step_profiled(self._step)

    def scope(self, name: str, attrs: dict[str, Any]) -> UserScopeContext:
        event_name = _event_name(name)
        attrs_json = _attrs_json(attrs)
        return UserScopeContext(
            recorder=self._recorder,
            op_name=self._op_name,
            step=self._step,
            rank=self._rank,
            execution_mode=self._execution_mode,
            name=event_name,
            attrs_json=attrs_json,
        )

    def event(self, name: str, attrs: dict[str, Any]) -> None:
        self._recorder.user_custom_event(
            name=_event_name(name),
            op_name=self._op_name,
            step=self._step,
            rank=self._rank,
            execution_mode=self._execution_mode,
            attrs_json=_attrs_json(attrs),
        )

    def function_scope(self) -> Any:
        return self._recorder.user_callable_function_run(
            op_name=self._op_name,
            step=self._step,
            rank=self._rank,
            execution_mode=self._execution_mode,
        )

    def launch_scope(self) -> Any:
        return self._recorder.user_callable_launch(
            op_name=self._op_name,
            step=self._step,
            rank=self._rank,
            execution_mode=self._execution_mode,
        )

    def start_async_operation(
        self, name: str, attrs: dict[str, Any]
    ) -> UserScopeContext:
        ctx = self.scope(name, attrs)
        ctx.__enter__()
        return ctx


def enabled() -> bool:
    handle = _ACTIVE_USER_PROFILER.get()
    if handle is None:
        return False
    return handle.enabled


def scope(name: str, **attrs: Any) -> Any:
    handle = _ACTIVE_USER_PROFILER.get()
    if handle is None or not handle.enabled:
        return _NOOP_SCOPE
    return handle.scope(name, attrs)


def async_scope(name: str, **attrs: Any) -> Any:
    return scope(name, **attrs)


def start_async_operation(name: str, **attrs: Any) -> Any:
    handle = _ACTIVE_USER_PROFILER.get()
    if handle is None or not handle.enabled:
        return _NOOP_SCOPE
    return handle.start_async_operation(name, attrs)


def event(name: str, **attrs: Any) -> None:
    handle = _ACTIVE_USER_PROFILER.get()
    if handle is None or not handle.enabled:
        return
    handle.event(name, attrs)


def function_scope() -> Any:
    handle = _ACTIVE_USER_PROFILER.get()
    if handle is None or not handle.enabled:
        return _NOOP_SCOPE
    return handle.function_scope()


def launch_scope() -> Any:
    handle = _ACTIVE_USER_PROFILER.get()
    if handle is None or not handle.enabled:
        return _NOOP_SCOPE
    return handle.launch_scope()


@contextmanager
def activate_user_profiler(
    *,
    recorder: TraceRecorder,
    op_name: str,
    step: int,
    rank: int,
    execution_mode: str,
) -> Generator[None, None, None]:
    handle = UserProfilerHandle(
        recorder=recorder,
        op_name=op_name,
        step=step,
        rank=rank,
        execution_mode=execution_mode,
    )
    token = _ACTIVE_USER_PROFILER.set(handle)
    try:
        yield
    finally:
        _ACTIVE_USER_PROFILER.reset(token)


def _event_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("profiler scope/event name must be a string.")
    if not name:
        raise ValueError("profiler scope/event name must be non-empty.")
    if name.startswith("user."):
        raise ValueError("profiler scope/event name must not include the user. prefix.")
    if any(ord(char) < 32 for char in name):
        raise ValueError("profiler scope/event name must not contain control chars.")
    return f"user.{name}"


def _attrs_json(attrs: dict[str, Any]) -> str:
    for key, value in attrs.items():
        _validate_attr_key(key)
        _validate_attr_value(key, value)
    return json.dumps(attrs, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _validate_attr_key(key: str) -> None:
    if not isinstance(key, str):
        raise TypeError("profiler attr keys must be strings.")
    if not key:
        raise ValueError("profiler attr keys must be non-empty.")
    if key in _RESERVED_ATTR_KEYS:
        raise ValueError(f"profiler attr key {key!r} is reserved.")
    if any(ord(char) < 32 for char in key):
        raise ValueError("profiler attr keys must not contain control chars.")


def _validate_attr_value(key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, str | int | float):
        return
    raise TypeError(
        f"profiler attr {key!r} must be str, int, float, bool, or None; "
        f"got {type(value).__name__}."
    )
