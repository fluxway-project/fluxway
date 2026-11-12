import asyncio
import inspect
import typing
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from fluxway import OperatorSpec
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_policy import ArgumentType
from fluxway.profiler.record import TraceArena
from fluxway.profiler.recorder import TraceRecorder
from fluxway.spmd.worker.worker import SPMDWorker
import fluxway.spmd.worker.runtime as worker_runtime
from tests.compiled_graph_fixtures import single_spmd_config


def _make_config(mesh, *op_names):
    operators = {}
    for name in op_names:
        operator = OperatorSpec(
            name=name,
            input_arguments={ArgumentType.DATA.value: ["in"]},
            output_arguments={ArgumentType.DATA.value: ["out"]},
            input_sharding={"in": {"type": "replicated"}},
            output_sharding={"out": {"type": "replicated"}},
        )
        operators[name] = operator

    return single_spmd_config(
        name="worker-backpressure",
        spmd_id="test",
        mesh=mesh,
        operators=operators,
    )


def _make_worker(max_in_flight=1):
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    config = _make_config(mesh, "compute")
    config.data_plane_max_in_flight = max_in_flight
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    return (cls.modified_class if cls else SPMDWorker)(
        config, rank=0, spmd_id=config.spmd_id
    )


@pytest.mark.asyncio
async def test_spmd_worker_delivery_admission_is_bounded_before_task_creation():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    config = _make_config(mesh, "compute")
    config.data_plane_max_in_flight = 1

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker

    worker = cls(config, rank=0)
    release_first = asyncio.Event()
    started: list[str] = []

    async def delivery(name: str) -> str:
        started.append(name)
        if name == "first":
            await release_first.wait()
        return name

    first = await worker._delivery_runtime.submit(
        delivery("first"), schedule_key=(0, 0, 0, "op")
    )
    second_submit = asyncio.create_task(
        worker._delivery_runtime.submit(
            delivery("second"), schedule_key=(0, 0, 1, "op")
        )
    )
    await asyncio.sleep(0.01)

    assert started == ["first"]
    assert not second_submit.done()
    release_first.set()
    assert await first == "first"
    second = await second_submit
    assert not second.done()
    assert await second == "second"
    assert started == ["first", "second"]


@pytest.mark.asyncio
async def test_spmd_worker_delivery_runtime_profiler_spans_lifecycle():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    config = _make_config(mesh, "compute")
    config.data_plane_max_in_flight = 1

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker

    worker = cls(config, rank=0)
    recorder = TraceRecorder(
        TraceArena(1024 * 1024),
        run_id="delivery-test",
        process_name="worker",
    )
    worker._profiler_runtime = typing.cast(
        typing.Any, SimpleNamespace(enabled=True, recorder=recorder)
    )
    worker._delivery_runtime._profiler_runtime = worker._profiler_runtime

    release_first = asyncio.Event()
    started: list[str] = []

    async def delivery(name: str) -> str:
        started.append(name)
        if name == "first":
            await release_first.wait()
        return name

    first = await worker._delivery_runtime.submit(
        delivery("first"), schedule_key=(0, 0, 0, "op")
    )
    second_submit = asyncio.create_task(
        worker._delivery_runtime.submit(
            delivery("second"), schedule_key=(0, 0, 1, "op")
        )
    )
    await asyncio.sleep(0.01)
    assert started == ["first"]
    assert not second_submit.done()

    release_first.set()
    second = await second_submit
    await worker._delivery_runtime.drain("unit_test")
    assert await first == "first"
    assert await second == "second"

    snapshot = recorder.arena.snapshot()
    names = [recorder.metadata.name_for(int(row[4])) for row in snapshot]

    assert names.count("delivery.register_token") == 2
    assert names.count("delivery.semaphore_wait") == 2
    assert names.count("delivery.run") == 2
    assert "data_plane.delivery_fence" not in names


@pytest.mark.asyncio
async def test_delivery_waiter_cancellation_does_not_consume_capacity():
    worker = _make_worker()
    release_first = asyncio.Event()
    started = []

    async def delivery(name):
        started.append(name)
        if name == "first":
            await release_first.wait()
        return name

    first = await worker._delivery_runtime.submit(
        delivery("first"), schedule_key=(0, 0, 0, "op")
    )
    waiting_submit = asyncio.create_task(
        worker._delivery_runtime.submit(
            delivery("cancelled"), schedule_key=(0, 0, 1, "op")
        )
    )
    await asyncio.sleep(0)
    assert len(worker._delivery_runtime._capacity._waiters) == 1

    waiting_submit.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting_submit

    release_first.set()
    assert await first == "first"
    recovered = await worker._delivery_runtime.submit(
        delivery("recovered"), schedule_key=(0, 0, 2, "op")
    )
    assert await recovered == "recovered"
    assert started == ["first", "recovered"]


@pytest.mark.asyncio
async def test_delivery_registration_cancellation_releases_acquired_capacity(
    monkeypatch,
):
    worker = _make_worker()
    recorder = TraceRecorder(
        TraceArena(1024 * 1024),
        run_id="delivery-cancel-test",
        process_name="worker",
    )
    worker._profiler_runtime = typing.cast(
        typing.Any, SimpleNamespace(enabled=True, recorder=recorder)
    )
    worker._delivery_runtime._profiler_runtime = worker._profiler_runtime
    registration_entered = asyncio.Event()
    hold_registration = asyncio.Event()
    started = []
    original_register_token = recorder.delivery_register_token

    @asynccontextmanager
    async def blocked_registration(*, pending_delivery_count):
        registration_entered.set()
        await hold_registration.wait()
        yield

    monkeypatch.setattr(recorder, "delivery_register_token", blocked_registration)

    async def delivery(name):
        started.append(name)
        return name

    cancelled_delivery = delivery("cancelled")
    submit = asyncio.create_task(
        worker._delivery_runtime.submit(
            cancelled_delivery, schedule_key=(0, 0, 0, "op")
        )
    )
    await registration_entered.wait()
    submit.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submit

    assert inspect.getcoroutinestate(cancelled_delivery) == inspect.CORO_CLOSED
    monkeypatch.setattr(recorder, "delivery_register_token", original_register_token)
    recovered = await worker._delivery_runtime.submit(
        delivery("recovered"), schedule_key=(0, 0, 1, "op")
    )
    assert await recovered == "recovered"
    assert started == ["recovered"]


@pytest.mark.asyncio
async def test_delivery_task_creation_failure_releases_acquired_capacity(
    monkeypatch,
):
    worker = _make_worker()
    started = []

    async def delivery(name):
        started.append(name)
        return name

    failed_delivery = delivery("failed")
    with monkeypatch.context() as patch:
        patch.setattr(
            worker_runtime.asyncio,
            "create_task",
            lambda _coro: (_ for _ in ()).throw(
                RuntimeError("task registration failed")
            ),
        )
        with pytest.raises(RuntimeError, match="task registration failed"):
            await worker._delivery_runtime.submit(
                failed_delivery, schedule_key=(0, 0, 0, "op")
            )

    assert inspect.getcoroutinestate(failed_delivery) == inspect.CORO_CLOSED
    recovered = await worker._delivery_runtime.submit(
        delivery("recovered"), schedule_key=(0, 0, 1, "op")
    )
    assert await recovered == "recovered"
    assert started == ["recovered"]


@pytest.mark.asyncio
async def test_delivery_cancel_before_first_run_releases_capacity():
    worker = _make_worker()
    started = []

    async def delivery(name):
        started.append(name)
        return name

    cancelled = await worker._delivery_runtime.submit(
        delivery("cancelled"), schedule_key=(0, 0, 0, "op")
    )
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    recovered = await worker._delivery_runtime.submit(
        delivery("recovered"), schedule_key=(0, 0, 1, "op")
    )
    assert await recovered == "recovered"
    assert started == ["recovered"]


@pytest.mark.asyncio
async def test_delivery_waiters_preserve_schedule_priority():
    worker = _make_worker()
    release_first = asyncio.Event()
    started = []

    async def delivery(name):
        started.append(name)
        if name == "first":
            await release_first.wait()
        return name

    first = await worker._delivery_runtime.submit(
        delivery("first"), schedule_key=(0, 0, 0, "op")
    )
    low_submit = asyncio.create_task(
        worker._delivery_runtime.submit(delivery("low"), schedule_key=(0, 0, 2, "op"))
    )
    high_submit = asyncio.create_task(
        worker._delivery_runtime.submit(delivery("high"), schedule_key=(0, 0, 1, "op"))
    )
    while len(worker._delivery_runtime._capacity._waiters) < 2:
        await asyncio.sleep(0)
    assert not low_submit.done()
    assert not high_submit.done()
    release_first.set()
    await first
    high = await high_submit
    low = await low_submit
    await high
    await low
    assert started == ["first", "high", "low"]


@pytest.mark.asyncio
async def test_delivery_profiler_failure_before_first_run_releases_capacity(
    monkeypatch,
):
    worker = _make_worker()
    recorder = TraceRecorder(
        TraceArena(1024 * 1024),
        run_id="delivery-test",
        process_name="worker",
    )
    worker._profiler_runtime = typing.cast(
        typing.Any, SimpleNamespace(enabled=True, recorder=recorder)
    )
    worker._delivery_runtime._profiler_runtime = worker._profiler_runtime
    started = []

    async def delivery(name):
        started.append(name)
        return name

    orig_intern = worker._profiler_runtime.recorder.intern

    def failing_intern(s: str) -> int:
        if s == "delivery":
            raise RuntimeError("profiler intern failure")
        return orig_intern(s)

    with monkeypatch.context() as patch:
        patch.setattr(
            worker._profiler_runtime.recorder,
            "intern",
            failing_intern,
        )
        task = await worker._delivery_runtime.submit(
            delivery("failed"), schedule_key=(0, 0, 0, "op")
        )
        with pytest.raises(RuntimeError, match="profiler intern failure"):
            await task

    while worker._delivery_runtime._capacity_release_tasks:
        await asyncio.sleep(0)

    assert worker._delivery_runtime._capacity._active == 0
    recovered = await worker._delivery_runtime.submit(
        delivery("recovered"), schedule_key=(0, 0, 1, "op")
    )
    assert await recovered == "recovered"
    assert started == ["recovered"]


@pytest.mark.asyncio
async def test_delivery_admission_context_failure_releases_capacity(monkeypatch):
    worker = _make_worker()
    recorder = TraceRecorder(
        TraceArena(1024 * 1024),
        run_id="delivery-admission-failure-test",
        process_name="worker",
    )
    worker._profiler_runtime = typing.cast(
        typing.Any, SimpleNamespace(enabled=True, recorder=recorder)
    )
    worker._delivery_runtime._profiler_runtime = worker._profiler_runtime
    original_wait = recorder.delivery_semaphore_wait

    @asynccontextmanager
    async def failing_wait(*, pending_delivery_count):
        async with original_wait(pending_delivery_count=pending_delivery_count):
            yield
        raise RuntimeError("admission span exit failure")

    monkeypatch.setattr(recorder, "delivery_semaphore_wait", failing_wait)

    async def delivery():
        return "failed"

    failed_delivery = delivery()
    with pytest.raises(RuntimeError, match="admission span exit failure"):
        await worker._delivery_runtime.submit(
            failed_delivery,
            schedule_key=(0, 0, 0, "op"),
        )

    assert inspect.getcoroutinestate(failed_delivery) == inspect.CORO_CLOSED
    assert worker._delivery_runtime._capacity._active == 0

    monkeypatch.setattr(recorder, "delivery_semaphore_wait", original_wait)
    recovered = await worker._delivery_runtime.submit(
        delivery(),
        schedule_key=(0, 0, 1, "op"),
    )
    assert await recovered == "failed"


@pytest.mark.asyncio
async def test_delivery_cancel_matching_waits_for_capacity_release():
    worker = _make_worker()
    started = []

    async def delivery(name):
        started.append(name)
        return name

    _task = await worker._delivery_runtime.submit(
        delivery("target"), schedule_key=(0, 0, 0, "target_op")
    )
    assert worker._delivery_runtime._capacity._active == 1

    await worker._delivery_runtime.cancel_matching(
        lambda k: k == (0, 0, 0, "target_op")
    )

    assert len(worker._delivery_runtime._capacity_release_tasks) == 0
    assert worker._delivery_runtime._capacity._active == 0
    recovered = await worker._delivery_runtime.submit(
        delivery("recovered"), schedule_key=(0, 0, 1, "op")
    )
    assert await recovered == "recovered"
    assert started == ["recovered"]
