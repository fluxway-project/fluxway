from __future__ import annotations

import threading
from typing import Any
import pytest

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
