"""Worker-local lifecycle runtimes."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
import contextvars
from dataclasses import dataclass, field
import heapq
import inspect
import queue
import threading
from typing import Any, cast
import weakref

from fluxway.common import asyncio_runtime
from fluxway.profiler.recorder import WORKER_OUTBOUND_TASK


WORKER_PROCESS_DEFAULT_EXECUTION_LANE = "__default__"
ScheduleKey = tuple[int, int, int, str]


class SourceAcknowledgementRuntime:
    def __init__(self, *, capacity: int, send: Callable, spawn: Callable) -> None:
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("source acknowledgement capacity must be positive")
        self._capacity = capacity
        self._send = send
        self._spawn = spawn
        self._queued: deque[tuple] = deque()
        self._pending = 0
        self._changed = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._error: BaseException | None = None

    async def submit(self, acknowledgements: tuple) -> None:
        if not acknowledgements:
            raise ValueError("source acknowledgement batch must be nonempty")
        while self._pending >= self._capacity and self._error is None:
            await self._changed.wait()
        if self._error is not None:
            raise self._error
        self._queued.append(acknowledgements)
        self._pending += 1
        if self._task is None or self._task.done():
            self._task = self._spawn(self._drain())

    async def _drain(self) -> None:
        try:
            while self._queued:
                count = len(self._queued)
                batch = tuple(
                    value for _ in range(count) for value in self._queued.popleft()
                )
                await self._send(batch)
                self._pending -= count
                self._notify_changed()
        except BaseException as error:
            self._error = error
            self._notify_changed()
            raise

    def _notify_changed(self) -> None:
        changed = self._changed
        self._changed = asyncio.Event()
        changed.set()

    async def flush(self) -> None:
        if self._task is not None:
            await asyncio.shield(self._task)
        if self._error is not None:
            raise self._error


@dataclass(order=True, slots=True)
class _PriorityThreadWork:
    schedule_key: ScheduleKey
    sequence: int
    fn: Callable[[], Any] | None = field(compare=False)
    future: Future[Any] | None = field(compare=False)


class _PriorityThreadExecutor:
    def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
        if max_workers <= 0:
            raise ValueError(
                f"priority thread executor max_workers must be positive, got {max_workers}."
            )
        self._max_workers = max_workers
        self._queue: queue.PriorityQueue[_PriorityThreadWork] = queue.PriorityQueue()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._lock = threading.Lock()
        self._closed = False
        self._sequence = 0
        self._workers: list[Future[Any]] = []
        with _priority_thread_executors_lock:
            _priority_thread_executors.add(self)

    def submit(self, schedule_key: ScheduleKey, fn: Callable[[], Any]) -> Future[Any]:
        if len(schedule_key) != 4:
            raise ValueError(
                f"schedule key must have four fields, got {schedule_key!r}."
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("Priority thread executor is closed.")
            if not self._workers:
                self._workers = [
                    self._executor.submit(self._run) for _ in range(self._max_workers)
                ]
            future: Future[Any] = Future()
            sequence = self._sequence
            self._sequence += 1
            self._queue.put(
                _PriorityThreadWork(
                    schedule_key=schedule_key,
                    sequence=sequence,
                    fn=fn,
                    future=future,
                )
            )
            return future

    def _run(self) -> None:
        while True:
            work = self._queue.get()
            try:
                if work.fn is None:
                    return
                future = work.future
                if future is None:
                    raise RuntimeError(
                        "Priority thread work is missing its result future."
                    )
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    future.set_result(work.fn())
                except BaseException as exc:
                    future.set_exception(exc)
            finally:
                self._queue.task_done()

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if cancel_futures:
                while True:
                    try:
                        work = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        if work.future is not None:
                            work.future.cancel()
                    finally:
                        self._queue.task_done()
            stop_key: ScheduleKey = (
                2**63 - 1,
                2**63 - 1,
                2**63 - 1,
                "\uffff",
            )
            for worker_index in range(len(self._workers)):
                self._queue.put(
                    _PriorityThreadWork(
                        schedule_key=stop_key,
                        sequence=self._sequence + worker_index,
                        fn=None,
                        future=None,
                    )
                )
        self._executor.shutdown(wait=wait, cancel_futures=False)


_priority_thread_executors: weakref.WeakSet[_PriorityThreadExecutor] = weakref.WeakSet()
_priority_thread_executors_lock = threading.Lock()


def _wake_priority_thread_executors_on_interpreter_exit() -> None:
    with _priority_thread_executors_lock:
        executors = tuple(_priority_thread_executors)
    for executor in executors:
        executor.shutdown(wait=False, cancel_futures=True)


def _register_threading_atexit(callback: Callable[[], Any]) -> None:
    register_fn = getattr(threading, "_register_atexit", None)
    if not callable(register_fn):
        raise RuntimeError(
            "threading._register_atexit is required for clean worker shutdown."
        )
    cast(Callable[[Callable[[], Any]], None], register_fn)(callback)


_register_threading_atexit(_wake_priority_thread_executors_on_interpreter_exit)


class WorkerProcessCallRuntime:
    def __init__(
        self,
        *,
        max_workers: int,
        profiler_runtime: Any | None = None,
        thread_name_prefix: str = "fluxway-worker-call",
    ) -> None:
        if max_workers <= 0:
            raise ValueError(
                f"worker process call max_workers must be positive, got {max_workers}."
            )
        self._max_workers = max_workers
        self._profiler_runtime = profiler_runtime
        self._thread_name_prefix = thread_name_prefix
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._futures_by_lane: dict[str, set[Future[Any]]] = {}
        self._futures_lock = threading.Lock()
        self._closed = False
        self._executor_for_lane(WORKER_PROCESS_DEFAULT_EXECUTION_LANE)

    async def submit(
        self,
        fn: Callable[[], Any],
        *,
        callable_id: str,
        resource_id: int,
        execution_lane: str,
    ) -> Any:
        if self._closed:
            raise RuntimeError("WorkerProcessCallRuntime is closed.")
        if not execution_lane:
            raise ValueError("worker process execution_lane must be non-empty.")
        executor = self._executor_for_lane(execution_lane)
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        profiler_runtime = self._profiler_runtime
        start_future: asyncio.Future[None] | None = None
        if profiler_runtime is not None and profiler_runtime.enabled:
            start_future = loop.create_future()

        def run_in_worker_thread() -> Any:
            if start_future is not None:
                loop.call_soon_threadsafe(self._mark_thread_started, start_future)
            recorder = (
                profiler_runtime.recorder
                if profiler_runtime is not None and profiler_runtime.enabled
                else None
            )

            def invoke() -> Any:
                result = fn()
                if inspect.iscoroutine(result):
                    return asyncio_runtime.run(result)
                return result

            if recorder is None:
                return context.run(invoke)
            with recorder.worker_process_thread_run(
                callable_id=callable_id,
                resource_id=resource_id,
            ):
                return context.run(invoke)

        recorder = (
            profiler_runtime.recorder
            if profiler_runtime is not None and profiler_runtime.enabled
            else None
        )
        if recorder is None:
            future = executor.submit(run_in_worker_thread)
        else:
            with recorder.worker_process_submit(
                callable_id=callable_id,
                resource_id=resource_id,
                pending_call_count=self.pending_count(execution_lane) + 1,
            ):
                future = executor.submit(run_in_worker_thread)
        self._add_future(execution_lane, future)

        def on_done(done_future: Future[Any]) -> None:
            self._discard_future(execution_lane, done_future)
            if start_future is not None and not start_future.done():
                loop.call_soon_threadsafe(self._mark_thread_started, start_future)

        future.add_done_callback(on_done)

        try:
            if recorder is not None and start_future is not None:
                async with recorder.worker_process_queue_wait(
                    callable_id=callable_id,
                    resource_id=resource_id,
                    pending_call_count=self.pending_count(execution_lane),
                ):
                    await start_future
                async with recorder.worker_process_future_wait(
                    callable_id=callable_id,
                    resource_id=resource_id,
                    pending_call_count=self.pending_count(execution_lane),
                ):
                    return await asyncio.wrap_future(future)
            return await asyncio.wrap_future(future)
        finally:
            if future.done():
                self._discard_future(execution_lane, future)

    @staticmethod
    def _mark_thread_started(start_future: asyncio.Future[None]) -> None:
        if not start_future.done():
            start_future.set_result(None)

    def _executor_for_lane(self, lane: str) -> ThreadPoolExecutor:
        if not lane:
            raise ValueError("worker process lane must be non-empty.")
        executor = self._executors.get(lane)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix=f"{self._thread_name_prefix}-{lane}",
            )
            self._executors[lane] = executor
            self._futures_by_lane[lane] = set()
        return executor

    def _add_future(self, lane: str, future: Future[Any]) -> None:
        with self._futures_lock:
            self._futures_by_lane[lane].add(future)

    def _discard_future(self, lane: str, future: Future[Any]) -> None:
        with self._futures_lock:
            self._futures_by_lane[lane].discard(future)

    def _pending_futures(self, lane: str | None = None) -> list[Future[Any]]:
        with self._futures_lock:
            if lane is not None:
                return [
                    future
                    for future in self._futures_by_lane[lane]
                    if not future.done()
                ]
            return [
                future
                for futures in self._futures_by_lane.values()
                for future in futures
                if not future.done()
            ]

    def pending_count(self, lane: str | None = None) -> int:
        return len(self._pending_futures(lane))

    def execution_lane_names(self) -> tuple[str, ...]:
        return tuple(self._executors.keys())

    async def drain(self) -> None:
        while True:
            futures = self._pending_futures()
            if not futures:
                return
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
            )

    async def cancel_pending(self) -> None:
        futures = self._pending_futures()
        for future in futures:
            future.cancel()
        if futures:
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=True,
            )
        with self._futures_lock:
            for futures_by_lane in self._futures_by_lane.values():
                futures_by_lane.clear()

    def close(self) -> None:
        self._closed = True
        for executor in self._executors.values():
            executor.shutdown(wait=False, cancel_futures=True)


class LocalInputPrepareRuntime:
    def __init__(
        self,
        *,
        background_tasks: set[asyncio.Task],
        on_task_done: Any,
        profiler_runtime: Any | None = None,
        on_prepare_ready: Callable[[str, int], None] | None = None,
        max_workers: int = 1,
        schedule_key: Callable[[str, int], ScheduleKey],
    ) -> None:
        if max_workers <= 0:
            raise ValueError(
                f"input prepare max_workers must be positive, got {max_workers}."
            )
        self._background_tasks = background_tasks
        self._on_task_done = on_task_done
        self._profiler_runtime = profiler_runtime
        self._on_prepare_ready = on_prepare_ready
        self._schedule_key = schedule_key
        self._tasks: dict[tuple[str, int], asyncio.Task[Any]] = {}
        self._executor = _PriorityThreadExecutor(
            max_workers=max_workers,
            thread_name_prefix="fluxway-input-prepare",
        )
        self._thread_futures: set[Future[Any]] = set()
        self._thread_futures_lock = threading.Lock()

    def start(
        self,
        op_name: str,
        step: int,
        awaitable_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> asyncio.Task[Any]:
        key = (op_name, step)
        task = self._tasks.get(key)
        if task is not None:
            return task
        awaitable = awaitable_factory()
        if self._profiler_runtime is not None and self._profiler_runtime.enabled:
            recorder = self._profiler_runtime.recorder

            async def traced_prepare() -> Any:
                async with recorder.prepare_run(op_name=op_name, step=step):
                    return await awaitable

            coro = traced_prepare()
            with recorder.prepare_register(
                op_name=op_name,
                step=step,
                pending_prepare_count=len(self._tasks) + 1,
            ):
                task = asyncio.create_task(coro)
        else:
            task = asyncio.create_task(awaitable)
        self._tasks[key] = task
        self._background_tasks.add(task)

        def _done(done_task: asyncio.Task) -> None:
            self._on_task_done(done_task)
            if (
                not done_task.cancelled()
                and done_task.exception() is None
                and self._on_prepare_ready is not None
            ):
                self._on_prepare_ready(op_name, step)

        task.add_done_callback(_done)
        return task

    async def run_assemble(
        self,
        op_name: str,
        step: int,
        fn: Callable[[], Any],
    ) -> Any:
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        profiler_runtime = self._profiler_runtime
        schedule_key = self._schedule_key(op_name, step)
        start_future: asyncio.Future[None] | None = None
        if profiler_runtime is not None and profiler_runtime.enabled:
            start_future = loop.create_future()

        def run_in_prepare_thread() -> Any:
            if start_future is not None:
                loop.call_soon_threadsafe(self._mark_assemble_started, start_future)
            recorder = (
                profiler_runtime.recorder
                if profiler_runtime is not None and profiler_runtime.enabled
                else None
            )
            if recorder is None:
                return context.run(fn)
            with recorder.prepare_assemble_thread(op_name=op_name, step=step):
                return context.run(fn)

        future = self._executor.submit(schedule_key, run_in_prepare_thread)
        self._add_thread_future(future)
        future.add_done_callback(self._discard_thread_future)
        try:
            if start_future is not None:
                assert profiler_runtime is not None
                async with profiler_runtime.recorder.prepare_assemble_queue_wait(
                    op_name=op_name,
                    step=step,
                    schedule_key=schedule_key,
                ):
                    await start_future
            return await asyncio.wrap_future(future, loop=loop)
        finally:
            if future.done():
                self._discard_thread_future(future)

    @staticmethod
    def _mark_assemble_started(start_future: asyncio.Future[None]) -> None:
        if not start_future.done():
            start_future.set_result(None)

    def _add_thread_future(self, future: Future[Any]) -> None:
        with self._thread_futures_lock:
            self._thread_futures.add(future)

    def _discard_thread_future(self, future: Future[Any]) -> None:
        with self._thread_futures_lock:
            self._thread_futures.discard(future)

    def _pending_thread_futures(self) -> list[Future[Any]]:
        with self._thread_futures_lock:
            return [future for future in self._thread_futures if not future.done()]

    async def consume(
        self,
        op_name: str,
        step: int,
        awaitable_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> Any:
        key = (op_name, step)
        task = self.start(op_name, step, awaitable_factory)
        try:
            if self._profiler_runtime is not None and self._profiler_runtime.enabled:
                async with self._profiler_runtime.recorder.prepare_consume(
                    op_name=op_name,
                    step=step,
                ):
                    return await task
            return await task
        finally:
            if self._tasks.get(key) is task:
                self._tasks.pop(key, None)

    async def await_ready(
        self,
        op_name: str,
        step: int,
        awaitable_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> Any:
        task = self.start(op_name, step, awaitable_factory)
        return await task

    async def drain(self) -> None:
        while True:
            tasks = [task for task in self._tasks.values() if not task.done()]
            thread_futures = self._pending_thread_futures()
            if not tasks and not thread_futures:
                return
            if self._profiler_runtime is not None and self._profiler_runtime.enabled:
                async with self._profiler_runtime.recorder.prepare_fence(
                    pending_prepare_count=len(tasks) + len(thread_futures),
                ):
                    await self._await_prepare_work(tasks, thread_futures)
            else:
                await self._await_prepare_work(tasks, thread_futures)

    async def _await_prepare_work(
        self,
        tasks: list[asyncio.Task[Any]],
        thread_futures: list[Future[Any]],
    ) -> None:
        awaitables: list[Awaitable[Any]] = list(tasks)
        awaitables.extend(asyncio.wrap_future(future) for future in thread_futures)
        if awaitables:
            await asyncio.gather(*awaitables)

    def is_pending(self, op_name: str, step: int) -> bool:
        task = self._tasks.get((op_name, step))
        return task is not None and not task.done()

    def is_ready(self, op_name: str, step: int) -> bool:
        task = self._tasks.get((op_name, step))
        return (
            task is not None
            and task.done()
            and not task.cancelled()
            and task.exception() is None
        )

    def pending_count(self) -> int:
        return sum(1 for task in self._tasks.values() if not task.done())

    async def cancel_pending(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        thread_futures = self._pending_thread_futures()
        for task in tasks:
            task.cancel()
        for future in thread_futures:
            future.cancel()
        if tasks or thread_futures:
            await asyncio.gather(
                *tasks,
                *(asyncio.wrap_future(future) for future in thread_futures),
                return_exceptions=True,
            )
        self._tasks.clear()
        with self._thread_futures_lock:
            self._thread_futures.clear()

    async def cancel_matching(
        self,
        predicate: Callable[[str, int], bool],
    ) -> None:
        selected = [
            (key, task)
            for key, task in self._tasks.items()
            if not task.done() and predicate(key[0], key[1])
        ]
        for _key, task in selected:
            task.cancel()
        if selected:
            await asyncio.gather(
                *(task for _key, task in selected),
                return_exceptions=True,
            )
        for key, task in selected:
            if self._tasks.get(key) is task:
                self._tasks.pop(key, None)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class OutboundDeliveryRuntime:
    def __init__(
        self,
        *,
        max_in_flight: int,
        profiler_runtime: Any,
        background_tasks: set[asyncio.Task],
        on_task_done: Any,
    ) -> None:
        if max_in_flight <= 0:
            raise ValueError(
                f"data_plane_max_in_flight must be positive, got {max_in_flight}."
            )
        self._capacity = _PriorityCapacityGate(max_in_flight)
        self._profiler_runtime = profiler_runtime
        self._background_tasks = background_tasks
        self._on_task_done = on_task_done
        self._tasks: set[asyncio.Task] = set()
        self._task_keys: dict[asyncio.Task, ScheduleKey] = {}

    async def submit(
        self,
        awaitable: Coroutine[Any, Any, Any],
        *,
        schedule_key: ScheduleKey,
    ) -> asyncio.Task:
        original_coro = awaitable
        if self._profiler_runtime.enabled:
            recorder = self._profiler_runtime.recorder
            task_name = getattr(
                getattr(original_coro, "cr_code", None),
                "co_name",
                type(original_coro).__name__,
            )

            async def traced_coro() -> Any:
                async with recorder.worker_scheduler_span(
                    spec=WORKER_OUTBOUND_TASK,
                    payload=(recorder.intern(task_name),),
                ):
                    return await self._run_delivery(original_coro, schedule_key)

            coro = traced_coro()
        else:
            coro = self._run_delivery(original_coro, schedule_key)
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.delivery_register_token(
                pending_delivery_count=len(self._tasks) + 1,
            ):
                task = asyncio.create_task(coro)
                self._background_tasks.add(task)
                self._tasks.add(task)
                self._task_keys[task] = schedule_key
        else:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            self._tasks.add(task)
            self._task_keys[task] = schedule_key

        def _done(done_task: asyncio.Task) -> None:
            self._tasks.discard(done_task)
            self._task_keys.pop(done_task, None)
            if self._profiler_runtime.enabled and not done_task.cancelled():
                exc = done_task.exception()
                if exc is not None:
                    with self._profiler_runtime.recorder.delivery_poison(
                        error=type(exc).__name__,
                    ):
                        pass
            self._on_task_done(done_task)

        task.add_done_callback(_done)
        return task

    async def _run_delivery(
        self,
        awaitable: Coroutine[Any, Any, Any],
        schedule_key: ScheduleKey,
    ) -> Any:
        acquired = False
        started = False
        try:
            if self._profiler_runtime.enabled:
                async with self._profiler_runtime.recorder.delivery_semaphore_wait(
                    pending_delivery_count=self.pending_count(),
                ):
                    await self._capacity.acquire(schedule_key)
            else:
                await self._capacity.acquire(schedule_key)
            acquired = True
            started = True
            if self._profiler_runtime.enabled:
                async with self._profiler_runtime.recorder.delivery_run(
                    pending_delivery_count=self.pending_count(),
                ):
                    return await awaitable
            return await awaitable
        finally:
            if acquired:
                await self._capacity.release()
            if not started:
                close = getattr(awaitable, "close", None)
                if callable(close):
                    close()

    def pending_count(self) -> int:
        return sum(1 for task in self._tasks if not task.done())

    async def drain(self, fence_id: str) -> None:
        recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        while True:
            tasks = [task for task in self._tasks if not task.done()]
            if not tasks:
                return
            if recorder is None:
                await asyncio.gather(*tasks)
            else:
                async with recorder.delivery_fence(
                    fence_id=fence_id,
                    pending_delivery_count=len(tasks),
                ):
                    await asyncio.gather(*tasks)

    async def cancel_pending(self) -> None:
        tasks = [task for task in self._tasks if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._task_keys.clear()

    async def cancel_matching(self, predicate: Callable[[ScheduleKey], bool]) -> None:
        tasks = [
            task
            for task in self._tasks
            if not task.done() and predicate(self._task_keys[task])
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            self._tasks.discard(task)
            self._task_keys.pop(task, None)


class _PriorityCapacityGate:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"priority capacity must be positive, got {capacity}.")
        self._capacity = capacity
        self._active = 0
        self._sequence = 0
        self._waiters: list[tuple[ScheduleKey, int, asyncio.Future[None]]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, schedule_key: ScheduleKey) -> None:
        if len(schedule_key) != 4:
            raise ValueError(
                f"schedule key must have four fields, got {schedule_key!r}."
            )
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] | None = None
        async with self._lock:
            self._discard_cancelled_waiters()
            if self._active < self._capacity and not self._waiters:
                self._active += 1
                return
            waiter = loop.create_future()
            sequence = self._sequence
            self._sequence += 1
            heapq.heappush(self._waiters, (schedule_key, sequence, waiter))
        try:
            await waiter
        except BaseException:
            granted = waiter.done() and not waiter.cancelled()
            if granted:
                await asyncio.shield(self.release())
            else:
                waiter.cancel()
            raise

    async def release(self) -> None:
        async with self._lock:
            if self._active <= 0:
                raise RuntimeError(
                    "priority capacity released without an active holder."
                )
            self._discard_cancelled_waiters()
            while self._waiters:
                _schedule_key, _sequence, waiter = heapq.heappop(self._waiters)
                if waiter.done():
                    continue
                waiter.set_result(None)
                return
            self._active -= 1

    def _discard_cancelled_waiters(self) -> None:
        while self._waiters and self._waiters[0][2].done():
            heapq.heappop(self._waiters)
