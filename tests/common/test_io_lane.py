import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, get_ident

import pytest

from fluxway.common.io_lane import IOLane


@pytest.mark.asyncio
async def test_independent_loop_and_executor_capacity():
    loop = asyncio.get_running_loop()
    release = Event()
    entered = loop.create_future()
    pool = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(pool)

    def blocker():
        loop.call_soon_threadsafe(entered.set_result, None)
        release.wait()

    blocked = asyncio.create_task(asyncio.to_thread(blocker))
    lane = IOLane("test-storage", capacity=1)
    try:
        await entered

        async def identity():
            return get_ident(), asyncio.get_running_loop()

        thread, remote_loop = await asyncio.wait_for(lane.call(identity), 2)
        assert thread != get_ident()
        assert remote_loop is not loop
    finally:
        release.set()
        await blocked
        await lane.close()
    assert lane._exited.done()
    assert not lane._thread.is_alive()


@pytest.mark.asyncio
async def test_cancelled_waiter_retains_capacity_and_close_owns_cleanup():
    loop = asyncio.get_running_loop()
    entered = loop.create_future()
    release = Event()
    second_entered = Event()
    lane = IOLane("test-storage-cancel", capacity=1)

    def blocking_write():
        loop.call_soon_threadsafe(entered.set_result, None)
        release.wait()

    first = asyncio.create_task(lane.call(blocking_write))
    try:
        await entered
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(lane.call(second_entered.set))
        await asyncio.sleep(0)
        assert not second_entered.is_set()
        closing = asyncio.create_task(lane.close())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        waiting = asyncio.create_task(lane.close())
        await asyncio.sleep(0)
        assert not waiting.done()
        release.set()
        with pytest.raises(RuntimeError, match="closing"):
            await second
        await waiting
    finally:
        release.set()
        await lane.close()
    assert lane._exited.done()
    assert not lane._thread.is_alive()
    assert not second_entered.is_set()


@pytest.mark.asyncio
async def test_cancelled_waiter_failure_is_reported_at_close():
    loop = asyncio.get_running_loop()
    entered = loop.create_future()
    release = Event()
    lane = IOLane("test-storage-failure", capacity=1)

    def write():
        loop.call_soon_threadsafe(entered.set_result, None)
        release.wait()
        raise ValueError("late write failure")

    caller = asyncio.create_task(lane.call(write))
    await entered
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    release.set()
    with pytest.raises(ExceptionGroup, match="unobserved") as caught:
        await lane.close()
    assert str(caught.value.exceptions[0]) == "late write failure"
    assert lane._exited.done()
    assert not lane._thread.is_alive()


@pytest.mark.asyncio
async def test_observed_failure_can_be_compensated_on_same_lane():
    lane = IOLane("test-storage-compensate", capacity=1)

    def fail():
        raise ValueError("write failure")

    try:
        with pytest.raises(ValueError, match="write failure"):
            await lane.call(fail)
        assert await lane.call(lambda: "rollback") == "rollback"
    finally:
        await lane.close()


@pytest.mark.asyncio
async def test_observed_operation_cancellation_allows_later_calls_and_close():
    lane = IOLane("test-observed-cancellation", capacity=1)

    async def operation():
        raise asyncio.CancelledError("operation cancelled itself")

    try:
        with pytest.raises(asyncio.CancelledError):
            await lane.call(operation)
        assert await lane.call(lambda: "next operation") == "next operation"
    finally:
        await lane.close()


@pytest.mark.asyncio
async def test_blocked_io_plane_does_not_occupy_another_plane():
    owner = asyncio.get_running_loop()
    entered = owner.create_future()
    release = Event()
    storage = IOLane("blocked-storage", capacity=1)
    transport = IOLane("independent-transport", capacity=1)

    def blocking_storage():
        owner.call_soon_threadsafe(entered.set_result, None)
        release.wait()

    operation = asyncio.create_task(storage.call(blocking_storage))
    try:
        await entered
        assert (
            await asyncio.wait_for(transport.call(lambda: "progress"), 2) == "progress"
        )
        assert not operation.done()
    finally:
        release.set()
        await operation
        await asyncio.gather(storage.close(), transport.close())


@pytest.mark.asyncio
async def test_waiting_call_is_not_poisoned_by_failure_with_live_waiter():
    owner = asyncio.get_running_loop()
    entered = owner.create_future()
    release = Event()
    lane = IOLane("test-observed-failure-order", capacity=1)

    def write():
        owner.call_soon_threadsafe(entered.set_result, None)
        release.wait()
        raise ValueError("handled write failure")

    async def observed_write():
        with pytest.raises(ValueError, match="handled write failure"):
            await lane.call(write)

    first = asyncio.create_task(observed_write())
    await entered
    next_call = asyncio.create_task(lane.call(lambda: "next operation"))
    await asyncio.sleep(0)
    release.set()
    try:
        results = await asyncio.gather(first, next_call, return_exceptions=True)
        assert results == [None, "next operation"]
    finally:
        await lane.close()


@pytest.mark.asyncio
async def test_lane_overlaps_awaitable_io_with_bounded_admission():
    owner = asyncio.get_running_loop()
    entered = [owner.create_future() for _ in range(3)]
    releases = [Future() for _ in range(3)]
    lane = IOLane("test-concurrent-io", capacity=2)

    async def operation(index):
        owner.call_soon_threadsafe(entered[index].set_result, None)
        await asyncio.wrap_future(releases[index])
        return index

    tasks = [asyncio.create_task(lane.call(operation, i)) for i in range(3)]
    try:
        await asyncio.wait_for(asyncio.gather(entered[0], entered[1]), 2)
        assert not entered[2].done()
        releases[0].set_result(None)
        await asyncio.wait_for(asyncio.shield(entered[2]), 2)
    finally:
        for release in releases:
            if not release.done():
                release.set_result(None)
        results = await asyncio.gather(*tasks)
        await lane.close()
    assert results == [0, 1, 2]


@pytest.mark.asyncio
async def test_lane_rejects_a_different_caller_loop():
    lane = IOLane("test-owner-contract", capacity=1)
    foreign = IOLane("foreign-caller", capacity=1)
    try:
        assert await lane.call(lambda: True)
        with pytest.raises(RuntimeError, match="owner event loops"):
            await foreign.call(lane.call, lambda: None)
    finally:
        await asyncio.gather(lane.close(), foreign.close())


@pytest.mark.asyncio
async def test_delayed_lane_call_keeps_event_loop_progress_and_backpressure():
    loop = asyncio.get_running_loop()
    entered = loop.create_future()
    release = Event()
    lane = IOLane("test-heartbeat-backpressure", capacity=1)
    heartbeats = 0
    heartbeat_running = True

    async def heartbeat() -> None:
        nonlocal heartbeats
        while heartbeat_running:
            heartbeats += 1
            await asyncio.sleep(0)

    def delayed_write() -> None:
        loop.call_soon_threadsafe(entered.set_result, None)
        release.wait()

    heartbeat_task = asyncio.create_task(heartbeat())
    first = asyncio.create_task(lane.call(delayed_write))
    second = asyncio.create_task(lane.call(lambda: None))
    try:
        await asyncio.wait_for(asyncio.shield(entered), 2)
        await asyncio.sleep(0.01)
        assert heartbeats > 0
        assert not second.done()
    finally:
        heartbeat_running = False
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)
        release.set()
        await first
        await second
        await lane.close()
