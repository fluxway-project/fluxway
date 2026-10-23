"""Cancellation-safe, idempotent closure shared by asynchronous resource owners."""

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

_active_closes: ContextVar[tuple[object, ...]] = ContextVar("active_closes", default=())


class AsyncCloseOnce:
    def __init__(self, operation: Callable[[], Awaitable[None]]) -> None:
        self._operation = operation
        self._task: asyncio.Task[None] | None = None

    @property
    def started(self) -> bool:
        return self._task is not None

    async def close(self) -> None:
        if self in _active_closes.get():
            raise RuntimeError("a close operation cannot await itself")
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            self._task.add_done_callback(self._observe_result)
        await asyncio.shield(self._task)

    async def _run(self) -> None:
        token = _active_closes.set((*_active_closes.get(), self))
        try:
            await self._operation()
        finally:
            _active_closes.reset(token)

    @staticmethod
    def _observe_result(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()
