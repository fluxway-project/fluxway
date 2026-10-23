"""Supervise accepted operations so they remain owned when callers stop waiting."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, TypeVar

_Result = TypeVar("_Result")
_active_supervisors: ContextVar[tuple[AsyncOperationSupervisor, ...]] = ContextVar(
    "active_operation_supervisors", default=()
)


class AsyncOperationSupervisor:
    """Supervises in-flight asynchronous operations with custody and drain guarantees.

    Rationale:
    Standard asyncio.shield leaves abandoned tasks unmonitored on caller timeout,
    leaking unobserved exceptions and risking shutdown corruption. This supervisor
    holds custody of accepted tasks, captures late failures, and guarantees safe
    drainage before resource disposal.
    """

    def __init__(self) -> None:
        self._accepting = True
        self._tasks: set[asyncio.Task[Any]] = set()
        self._abandoned: set[asyncio.Task[Any]] = set()
        self._unobserved: dict[asyncio.Task[Any], BaseException] = {}

    @property
    def accepting(self) -> bool:
        return self._accepting

    def ensure_not_active(self) -> None:
        """Prevent self-await deadlocks during drain.

        If a running task or its child task calls drain(), drain() would await
        the calling task itself, causing a permanent circular wait.
        """
        if self in _active_supervisors.get():
            raise RuntimeError("a supervised operation cannot close its supervisor")

    def raise_unobserved(self) -> None:
        """Raise accumulated failures from abandoned tasks."""
        if self._unobserved:
            raise BaseExceptionGroup(
                "unobserved supervised operation failure",
                list(self._unobserved.values()),
            )

    async def run(self, operation: Callable[[], Awaitable[_Result]]) -> _Result:
        """Admit and supervise an operation to completion.

        Uses asyncio.wait to decouple caller cancellation from the underlying task:
        - If the caller cancels, the underlying task continues in background custody.
        - If the underlying task raises CancelledError, it returns via task.result()
          directly to the caller without false-positive unobserved error recording.
        """
        if not self._accepting:
            raise RuntimeError("operation supervisor is closing")
        task = asyncio.create_task(self._invoke(operation))
        self._tasks.add(task)
        task.add_done_callback(self._settled)
        try:
            # Wait without shielding: caller cancellation cancels the wait, not the task.
            await asyncio.wait((task,))
        except asyncio.CancelledError:
            if task.done():
                self._record_failure(task)
            else:
                self._abandoned.add(task)
            raise
        return task.result()

    async def _invoke(self, operation: Callable[[], Awaitable[_Result]]) -> _Result:
        # Track active supervisor in ContextVar to detect reentrant drain deadlocks.
        token = _active_supervisors.set((*_active_supervisors.get(), self))
        try:
            return await operation()
        finally:
            _active_supervisors.reset(token)

    def stop_accepting(self) -> None:
        """Stop admitting new tasks, ensuring caller is not a managed task."""
        self.ensure_not_active()
        self._accepting = False

    async def drain(self) -> tuple[BaseException, ...]:
        """Stop accepting new tasks and await all in-flight work to complete.

        Uses return_exceptions=True so individual errors do not abort sibling teardown.
        """
        self.stop_accepting()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        return tuple(self._unobserved.values())

    def _settled(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task in self._abandoned:
            self._abandoned.remove(task)
            self._record_failure(task)
        elif not task.cancelled():
            task.exception()

    def _record_failure(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            self._unobserved[task] = asyncio.CancelledError()
        elif (error := task.exception()) is not None:
            self._unobserved[task] = error
