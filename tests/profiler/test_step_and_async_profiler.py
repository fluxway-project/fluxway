import asyncio
import time
import pytest

import fluxway as fw
from fluxway.profiler.config import ProfilerConfig
from fluxway.profiler.record import TraceArena
from fluxway.profiler.recorder import TraceRecorder
from fluxway.profiler.runtime import ProfilerRuntimeFactory
from fluxway.profiler.user_api import activate_user_profiler
from tests.profiler.test_operator_function_span import _bare_manager


def test_step_interval():
    config = ProfilerConfig(trace_dir="/tmp", run_id="int_test", step_interval=2)
    factory = ProfilerRuntimeFactory.from_config(config, process_name="rank:0")
    rec = factory.recorder
    for s in range(6):
        with rec.user_callable_function_run(
            op_name="op", step=s, rank=0, execution_mode="worker_process"
        ):
            pass
    snap = rec.snapshot()
    steps = [int(r[9]) for r in snap["records"]]
    assert steps == [0, 2, 4]


def test_target_steps():
    config = ProfilerConfig(trace_dir="/tmp", run_id="tgt_test", target_steps=[1, 3, 7])
    factory = ProfilerRuntimeFactory.from_config(config, process_name="rank:0")
    rec = factory.recorder
    for s in range(10):
        with rec.user_callable_function_run(
            op_name="op", step=s, rank=0, execution_mode="worker_process"
        ):
            pass
    snap = rec.snapshot()
    steps = [int(r[9]) for r in snap["records"]]
    assert steps == [1, 3, 7]


def test_max_steps():
    config = ProfilerConfig(trace_dir="/tmp", run_id="max_test", max_steps=2)
    factory = ProfilerRuntimeFactory.from_config(config, process_name="rank:0")
    rec = factory.recorder
    for s in range(5):
        with rec.user_callable_function_run(
            op_name="op1", step=s, rank=0, execution_mode="worker_process"
        ):
            pass
        with rec.user_callable_function_run(
            op_name="op2", step=s, rank=0, execution_mode="worker_process"
        ):
            pass
    snap = rec.snapshot()
    steps = [int(r[9]) for r in snap["records"]]
    assert steps == [0, 0, 1, 1]


def test_combined_step_filters():
    config = ProfilerConfig(
        trace_dir="/tmp",
        run_id="comb_test",
        skip_initial_steps=2,
        step_interval=2,
        target_steps=[2, 4, 6, 8],
        max_steps=2,
    )
    factory = ProfilerRuntimeFactory.from_config(config, process_name="rank:0")
    rec = factory.recorder
    for s in range(12):
        with rec.user_callable_function_run(
            op_name="op", step=s, rank=0, execution_mode="worker_process"
        ):
            pass
    snap = rec.snapshot()
    steps = [int(r[9]) for r in snap["records"]]
    assert steps == [2, 4]


def test_step_control_validation():
    with pytest.raises(
        ValueError, match="ProfilerConfig.step_interval must be positive"
    ):
        ProfilerConfig(trace_dir="/tmp", run_id="bad", step_interval=0)
    with pytest.raises(
        ValueError, match="ProfilerConfig.step_interval must be positive"
    ):
        ProfilerConfig(trace_dir="/tmp", run_id="bad", step_interval=True)
    with pytest.raises(
        ValueError, match="ProfilerConfig.max_steps must be a positive integer"
    ):
        ProfilerConfig(trace_dir="/tmp", run_id="bad", max_steps=-1)
    with pytest.raises(
        ValueError, match="ProfilerConfig.max_steps must be a positive integer"
    ):
        ProfilerConfig(trace_dir="/tmp", run_id="bad", max_steps=True)
    with pytest.raises(
        ValueError, match="ProfilerConfig.target_steps elements must be non-negative"
    ):
        ProfilerConfig(trace_dir="/tmp", run_id="bad", target_steps=[-1])
    with pytest.raises(
        TypeError, match="ProfilerConfig.target_steps must be an iterable"
    ):
        ProfilerConfig(trace_dir="/tmp", run_id="bad", target_steps=123)  # type: ignore[arg-type]


def test_user_profiler_enabled_reflects_step_filter():
    config = ProfilerConfig(trace_dir="/tmp", run_id="filter_en", target_steps=[2])
    factory = ProfilerRuntimeFactory.from_config(config, process_name="rank:0")
    rec = factory.recorder
    with activate_user_profiler(
        recorder=rec, op_name="op", step=1, rank=0, execution_mode="worker_process"
    ):
        assert fw.profiler.enabled() is False
        with fw.profiler.scope("test"):
            pass
    with activate_user_profiler(
        recorder=rec, op_name="op", step=2, rank=0, execution_mode="worker_process"
    ):
        assert fw.profiler.enabled() is True
        with fw.profiler.scope("test"):
            pass
    snap = rec.snapshot()
    records = snap["records"]
    assert len(records) == 1


@pytest.mark.asyncio
async def test_async_with_user_scope():
    rec = TraceRecorder(
        TraceArena(1024 * 1024), run_id="async_test", process_name="rank:0"
    )
    with activate_user_profiler(
        recorder=rec,
        op_name="async_op",
        step=0,
        rank=0,
        execution_mode="coroutine_async",
    ):
        async with fw.profiler.scope("async_scope_event", key="val"):
            await asyncio.sleep(0.01)
    snap = rec.snapshot()
    names = [snap["metadata"].name_for(int(r[4])) for r in snap["records"]]
    assert names == ["user.async_scope_event"]
    dur_ms = snap["records"][0][2] / 1e6
    assert dur_ms >= 10.0


@pytest.mark.asyncio
async def test_async_scope_launch_and_completion():
    rec = TraceRecorder(
        TraceArena(1024 * 1024), run_id="launch_comp", process_name="rank:0"
    )
    with activate_user_profiler(
        recorder=rec,
        op_name="async_op",
        step=0,
        rank=0,
        execution_mode="coroutine_async",
    ):
        async with fw.profiler.async_scope("kernel_op") as op:
            time.sleep(0.002)
            op.launched()
            await asyncio.sleep(0.02)
    snap = rec.snapshot()
    records = snap["records"]
    names = [snap["metadata"].name_for(int(r[4])) for r in records]
    assert names == ["user.kernel_op.launch", "user.kernel_op"]
    launch_dur_ms = records[0][2] / 1e6
    full_dur_ms = records[1][2] / 1e6
    assert 1.5 <= launch_dur_ms < 10.0
    assert full_dur_ms >= 20.0


@pytest.mark.asyncio
async def test_callback_async_operation():
    rec = TraceRecorder(
        TraceArena(1024 * 1024), run_id="callback_test", process_name="rank:0"
    )
    with activate_user_profiler(
        recorder=rec,
        op_name="async_op",
        step=0,
        rank=0,
        execution_mode="coroutine_async",
    ):
        op = fw.profiler.start_async_operation("callback_kernel")
        time.sleep(0.002)
        op.launched()

        fut = asyncio.get_running_loop().create_future()
        fut.add_done_callback(lambda _: op.complete())
        asyncio.get_running_loop().call_later(0.015, fut.set_result, 42)
        await fut

    snap = rec.snapshot()
    records = snap["records"]
    names = [snap["metadata"].name_for(int(r[4])) for r in records]
    assert names == ["user.callback_kernel.launch", "user.callback_kernel"]
    launch_dur_ms = records[0][2] / 1e6
    full_dur_ms = records[1][2] / 1e6
    assert 1.5 <= launch_dur_ms < 10.0
    assert full_dur_ms >= 14.0


@pytest.mark.asyncio
async def test_coroutine_operator_records_launch_and_function_run():
    rec = TraceRecorder(
        TraceArena(1024 * 1024), run_id="coro_op", process_name="rank:0"
    )

    async def sample_coroutine(x):
        time.sleep(0.003)
        await asyncio.sleep(0.02)
        return x * 2

    manager = _bare_manager("coro_calc", sample_coroutine, coroutine_function=True)
    with activate_user_profiler(
        recorder=rec,
        op_name="coro_calc",
        step=0,
        rank=0,
        execution_mode="coroutine_async",
    ):
        res = await manager.execute_async("coro_calc", 0, 21)
    assert res == 42

    snap = rec.snapshot()
    records = snap["records"]
    names = [snap["metadata"].name_for(int(r[4])) for r in records]
    assert names == ["user_callable.launch", "user_callable.function_run"]
    launch_dur_ms = records[0][2] / 1e6
    func_dur_ms = records[1][2] / 1e6
    assert launch_dur_ms < 5.0
    assert func_dur_ms >= 20.0


@pytest.mark.asyncio
async def test_noop_async_scope_when_disabled():
    assert fw.profiler.enabled() is False
    async with fw.profiler.scope("noop_async") as op:
        op.launched()
        await asyncio.sleep(0.001)
        op.complete()
    async with fw.profiler.async_scope("noop_async2") as op:
        op.launched()
        await asyncio.sleep(0.001)
    op3 = fw.profiler.start_async_operation("noop_async3")
    op3.launched()
    op3.complete()
