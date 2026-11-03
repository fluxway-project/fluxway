from __future__ import annotations

import asyncio
import threading
from typing import Any
import pytest

from fluxway.common.asyncio_runtime import invoke_callable_async
from fluxway.common.decorators import operator_context
from fluxway.common.types.configs import OperatorContext
from fluxway.reader.base_reader import BaseReader
from fluxway.spmd.worker.runtime import WorkerProcessCallRuntime


@pytest.mark.asyncio
async def test_run_in_lane_managed_thread():
    runtime = WorkerProcessCallRuntime(thread_name_prefix="test_lane", max_workers=2)

    async def lane_runner(
        fn: Any, *args: Any, lane: str = "default", **kwargs: Any
    ) -> Any:
        target = (lambda: fn(*args, **kwargs)) if (args or kwargs) else fn
        return await runtime.submit(
            target,
            callable_id="test_lane_op",
            resource_id=0,
            execution_lane=lane,
        )

    ctx = OperatorContext(
        physical_rank=0,
        physical_world_size=1,
        logical_rank=0,
        logical_world_size=1,
        step=0,
    )

    with operator_context(
        static_args={},
        rank=0,
        world_size=1,
        operator_ctx=ctx,
        lane_runner=lane_runner,
    ):

        def compute_worker(x: int, y: int) -> tuple[int, str]:
            return x * y, threading.current_thread().name

        # Execute in dedicated 'compute' lane
        val, thread_name = await ctx.run_in_lane(compute_worker, 6, 7, lane="compute")
        assert val == 42
        assert "compute" in thread_name

        # Also via BaseReader helper
        reader = BaseReader()
        val2, thread_name2 = await reader.run_in_lane(compute_worker, 3, 5, lane="io")
        assert val2 == 15
        assert "io" in thread_name2

    runtime.close()


@pytest.mark.asyncio
async def test_run_in_lane_error_when_no_runner():
    ctx = OperatorContext(
        physical_rank=0,
        physical_world_size=1,
        logical_rank=0,
        logical_world_size=1,
        step=0,
    )

    with operator_context(
        static_args={},
        rank=0,
        world_size=1,
        operator_ctx=ctx,
    ):
        with pytest.raises(RuntimeError, match="no active lane runner configured"):
            await ctx.run_in_lane(lambda: 123, lane="compute")

    reader = BaseReader()
    with pytest.raises(RuntimeError, match="requires an active OperatorContext"):
        await reader.run_in_lane(lambda: 123)


@pytest.mark.asyncio
async def test_run_in_lane_exception_propagation():
    runtime = WorkerProcessCallRuntime(thread_name_prefix="test_err", max_workers=2)

    async def lane_runner(
        fn: Any, *args: Any, lane: str = "default", **kwargs: Any
    ) -> Any:
        target = (lambda: fn(*args, **kwargs)) if (args or kwargs) else fn
        return await runtime.submit(
            target,
            callable_id="test_err_op",
            resource_id=0,
            execution_lane=lane,
        )

    ctx = OperatorContext(
        physical_rank=0,
        physical_world_size=1,
        logical_rank=0,
        logical_world_size=1,
        step=0,
    )

    with operator_context(
        static_args={},
        rank=0,
        world_size=1,
        operator_ctx=ctx,
        lane_runner=lane_runner,
    ):

        def failing_task():
            raise ZeroDivisionError("simulated compute failure")

        with pytest.raises(ZeroDivisionError, match="simulated compute failure"):
            await ctx.run_in_lane(failing_task, lane="compute")

    runtime.close()


@pytest.mark.asyncio
async def test_invoke_callable_async_unified():
    runtime = WorkerProcessCallRuntime(thread_name_prefix="test_invoke", max_workers=2)

    async def lane_runner(
        fn: Any, *args: Any, lane: str = "default", **kwargs: Any
    ) -> Any:
        target = (lambda: fn(*args, **kwargs)) if (args or kwargs) else fn
        return await runtime.submit(
            target,
            callable_id="test_invoke_op",
            resource_id=0,
            execution_lane=lane,
        )

    # 1. Coroutine function invocation
    async def async_method(val: int) -> int:
        await asyncio.sleep(0.01)
        return val * 10

    res_async = await invoke_callable_async(async_method, 5)
    assert res_async == 50

    # 2. Sync function with explicit lane_runner
    def sync_method(val: int) -> tuple[int, str]:
        return val + 10, threading.current_thread().name

    res_sync, thread_name = await invoke_callable_async(
        sync_method, 5, lane="my_lane", lane_runner=lane_runner
    )
    assert res_sync == 15
    assert "my_lane" in thread_name

    # 3. Sync function with ambient operator context lane runner
    ctx = OperatorContext(
        physical_rank=0,
        physical_world_size=1,
        logical_rank=0,
        logical_world_size=1,
        step=0,
    )
    with operator_context(
        static_args={},
        rank=0,
        world_size=1,
        operator_ctx=ctx,
        lane_runner=lane_runner,
    ):
        res_ambient, ambient_thread = await invoke_callable_async(
            sync_method, 20, lane="ambient"
        )
        assert res_ambient == 30
        assert "ambient" in ambient_thread

    # 4. Error without runner
    with pytest.raises(RuntimeError, match="no lane runner is available"):
        await invoke_callable_async(sync_method, 1)

    runtime.close()
