from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from concurrent.futures import Future
from threading import Thread
from typing import Any

from fluxway.common.async_close import AsyncCloseOnce
from fluxway.common.async_operations import AsyncOperationSupervisor


class IOLane:
    def __init__(self, name: str, *, capacity: int) -> None:
        if not name or capacity < 1:
            raise ValueError("I/O lane requires a name and positive capacity")
        self._name = name
        self._capacity = asyncio.Semaphore(capacity)
        self._owner: asyncio.AbstractEventLoop | None = None
        self._ready: asyncio.Future[asyncio.AbstractEventLoop] | None = None
        self._exited: asyncio.Future[None] | None = None
        self._thread: Thread | None = None
        self._operations = AsyncOperationSupervisor()
        self._close_once = AsyncCloseOnce(self._close)

    def _bind_owner(self) -> asyncio.AbstractEventLoop:
        owner = asyncio.get_running_loop()
        if self._owner is None:
            self._owner = owner
        elif self._owner is not owner:
            raise RuntimeError("I/O lane cannot be shared across owner event loops")
        return owner

    def _start(self) -> asyncio.Future[asyncio.AbstractEventLoop]:
        owner = self._bind_owner()
        if self._ready is None:
            self._ready = owner.create_future()
            self._exited = owner.create_future()
            self._thread = Thread(target=self._run, name=self._name)
            self._thread.start()
        return self._ready

    def _run(self) -> None:
        assert self._owner is not None
        assert self._ready is not None
        assert self._exited is not None
        try:
            with asyncio.Runner() as runner:
                loop = runner.get_loop()
                self._owner.call_soon_threadsafe(self._ready.set_result, loop)
                loop.run_forever()
        except BaseException as error:
            self._owner.call_soon_threadsafe(self._thread_failed, error)
        else:
            self._owner.call_soon_threadsafe(self._exited.set_result, None)

    def _thread_failed(self, error: BaseException) -> None:
        assert self._ready is not None
        assert self._exited is not None
        if not self._ready.done():
            self._ready.set_exception(error)
        self._exited.set_exception(error)

    async def call(self, operation: Callable[..., Any], /, *args, **kwargs) -> Any:
        self._bind_owner()
        if not self._operations.accepting:
            raise RuntimeError("I/O lane is closing")
        self._operations.raise_unobserved()
        await self._capacity.acquire()
        if not self._operations.accepting:
            self._capacity.release()
            raise RuntimeError("I/O lane is closing")
        try:
            self._operations.raise_unobserved()
        except BaseException:
            self._capacity.release()
            raise

        async def dispatch():
            try:
                return await self._dispatch(operation, args, kwargs)
            finally:
                self._capacity.release()

        return await self._operations.run(dispatch)

    async def _dispatch(self, operation, args, kwargs):
        lane = await asyncio.shield(self._start())

        async def invoke():
            result = operation(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        submitted: Future[Any] = asyncio.run_coroutine_threadsafe(invoke(), lane)
        return await asyncio.wrap_future(submitted)

    async def close(self) -> None:
        self._bind_owner()
        self._operations.stop_accepting()
        await self._close_once.close()

    async def _close(self) -> None:
        failures = await self._operations.drain()
        if self._ready is not None:
            try:
                lane = await asyncio.shield(self._ready)
                lane.call_soon_threadsafe(lane.stop)
            finally:
                assert self._exited is not None
                try:
                    await asyncio.shield(self._exited)
                finally:
                    await self._join_thread()
        if failures:
            raise BaseExceptionGroup("unobserved I/O lane failure", list(failures))

    async def _join_thread(self) -> None:
        owner = self._bind_owner()
        joined = owner.create_future()
        assert self._thread is not None

        def join():
            self._thread.join()
            owner.call_soon_threadsafe(joined.set_result, None)

        Thread(target=join, name=f"{self._name}-join").start()
        await joined
