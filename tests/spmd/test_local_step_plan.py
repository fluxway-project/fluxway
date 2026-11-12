import asyncio

import pytest

from fluxway.spmd.worker.local_plan import (
    LocalStepNodeKind,
    LocalStepPlan,
    LocalStepPlanExecutor,
    MappingCallableRegistry,
)
from fluxway.spmd.worker.runtime import WorkerProcessCallRuntime
from fluxway.spmd.worker.user_callable_executor import UserCallableExecutor


def _make_executor() -> LocalStepPlanExecutor:
    return LocalStepPlanExecutor(
        UserCallableExecutor(
            worker_process_runtime=WorkerProcessCallRuntime(max_workers=1),
        )
    )


def test_plan_rejects_invalid_metadata_at_construction():
    with pytest.raises(ValueError, match="dependency_offsets length"):
        LocalStepPlan(
            node_kinds=(int(LocalStepNodeKind.CALL),),
            callable_ids=("op",),
            resource_ids=(-1,),
            dependency_offsets=(0,),
            dependencies=(),
            result_node=0,
        )


def test_plan_freezes_sequence_inputs_before_caching_derived_state():
    node_kinds = [int(LocalStepNodeKind.CALL)]
    callable_ids = ["op"]
    plan = LocalStepPlan(
        node_kinds=node_kinds,
        callable_ids=callable_ids,
        resource_ids=[-1],
        dependency_offsets=[0, 0],
        dependencies=[],
        result_node=0,
    )

    node_kinds.append(int(LocalStepNodeKind.SEND))
    callable_ids.clear()

    assert plan.node_kinds == (int(LocalStepNodeKind.CALL),)
    assert plan.callable_ids == ("op",)
    assert plan._serial_chain


@pytest.mark.asyncio
async def test_single_call_plan_returns_result():
    executor = _make_executor()
    plan = LocalStepPlan.single_call("op")
    registry = MappingCallableRegistry({"op": lambda: 7})

    assert await executor.execute(plan, registry) == 7


@pytest.mark.asyncio
async def test_execution_uses_construction_time_plan_validation(monkeypatch):
    validate = LocalStepPlan.validate
    validation_count = 0

    def count_validation(plan):
        nonlocal validation_count
        validation_count += 1
        validate(plan)

    monkeypatch.setattr(LocalStepPlan, "validate", count_validation)
    plan = LocalStepPlan.single_call("op")
    assert validation_count == 1

    executor = _make_executor()
    result = await executor.execute(plan, MappingCallableRegistry({"op": lambda: 7}))

    assert result == 7
    assert validation_count == 1


@pytest.mark.asyncio
async def test_same_resource_nodes_are_serialized():
    executor = _make_executor()
    events: list[str] = []

    async def first():
        events.append("first:start")
        await asyncio.sleep(0.01)
        events.append("first:end")
        return 1

    async def second():
        events.append("second:start")
        await asyncio.sleep(0.01)
        events.append("second:end")
        return 2

    plan = LocalStepPlan(
        node_kinds=(int(LocalStepNodeKind.CALL), int(LocalStepNodeKind.CALL)),
        callable_ids=("first", "second"),
        resource_ids=(3, 3),
        dependency_offsets=(0, 0, 0),
        dependencies=(),
        result_node=1,
    )
    registry = MappingCallableRegistry({"first": first, "second": second})

    assert await executor.execute(plan, registry) == 2
    assert events in (
        ["first:start", "first:end", "second:start", "second:end"],
        ["second:start", "second:end", "first:start", "first:end"],
    )


@pytest.mark.asyncio
async def test_different_resources_can_overlap():
    executor = _make_executor()
    started = asyncio.Event()
    release = asyncio.Event()
    overlap_seen = False

    async def first():
        nonlocal overlap_seen
        started.set()
        await release.wait()
        return 1

    async def second():
        nonlocal overlap_seen
        await started.wait()
        overlap_seen = True
        release.set()
        return 2

    plan = LocalStepPlan(
        node_kinds=(int(LocalStepNodeKind.CALL), int(LocalStepNodeKind.CALL)),
        callable_ids=("first", "second"),
        resource_ids=(1, 2),
        dependency_offsets=(0, 0, 0),
        dependencies=(),
        result_node=1,
    )
    registry = MappingCallableRegistry({"first": first, "second": second})

    assert await executor.execute(plan, registry) == 2
    assert overlap_seen


@pytest.mark.asyncio
async def test_missing_runtime_callable_raises():
    executor = _make_executor()
    plan = LocalStepPlan(
        node_kinds=(int(LocalStepNodeKind.SEND),),
        callable_ids=("",),
        resource_ids=(-1,),
        dependency_offsets=(0, 0),
        dependencies=(),
        result_node=0,
    )

    with pytest.raises(KeyError):
        await executor.execute(plan, MappingCallableRegistry({}))


@pytest.mark.asyncio
async def test_wait_call_send_barrier_dependency_chain():
    executor = _make_executor()
    events: list[str] = []

    async def wait_inputs():
        events.append("wait")
        return {"x": 3}

    async def call_user(inputs):
        events.append("call")
        return inputs["x"] * 2

    async def send_output(value):
        events.append("send")

        async def sent():
            events.append("send:future")
            return value + 1

        return asyncio.create_task(sent())

    async def commit(send_future):
        events.append("commit")
        return await send_future

    plan = LocalStepPlan(
        node_kinds=(
            int(LocalStepNodeKind.WAIT),
            int(LocalStepNodeKind.CALL),
            int(LocalStepNodeKind.SEND),
            int(LocalStepNodeKind.BARRIER),
        ),
        callable_ids=("wait", "call", "send", "commit"),
        resource_ids=(-1, -1, -1, -1),
        dependency_offsets=(0, 0, 1, 2, 3),
        dependencies=(0, 1, 2),
        result_node=3,
    )

    result = await executor.execute(
        plan,
        MappingCallableRegistry(
            {
                "wait": wait_inputs,
                "call": call_user,
                "send": send_output,
                "commit": commit,
            }
        ),
    )

    assert result == 7
    assert events[:4] == ["wait", "call", "send", "commit"]
    assert events[-1] == "send:future"


@pytest.mark.asyncio
async def test_same_resource_is_serialized_across_plans():
    executor = _make_executor()
    running = 0
    max_running = 0
    release = asyncio.Event()

    async def shared_resource_call():
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        await release.wait()
        running -= 1
        return "done"

    plan = LocalStepPlan.single_call("op", resource_id=42)
    registry = MappingCallableRegistry({"op": shared_resource_call})

    first = asyncio.create_task(executor.execute(plan, registry))
    await asyncio.sleep(0)
    second = asyncio.create_task(executor.execute(plan, registry))
    await asyncio.sleep(0.01)
    assert max_running == 1
    release.set()

    assert await first == "done"
    assert await second == "done"
    assert max_running == 1


@pytest.mark.asyncio
async def test_barrier_observes_send_future_exception():
    executor = _make_executor()
    observed = False

    async def send_output():
        async def fail_send():
            raise RuntimeError("transport failed")

        return asyncio.create_task(fail_send())

    async def commit(send_future):
        nonlocal observed
        observed = True
        await send_future

    plan = LocalStepPlan(
        node_kinds=(int(LocalStepNodeKind.SEND), int(LocalStepNodeKind.BARRIER)),
        callable_ids=("send", "commit"),
        resource_ids=(-1, -1),
        dependency_offsets=(0, 0, 1),
        dependencies=(0,),
        result_node=1,
    )

    with pytest.raises(RuntimeError, match="transport failed"):
        await executor.execute(
            plan,
            MappingCallableRegistry({"send": send_output, "commit": commit}),
        )

    assert observed


@pytest.mark.asyncio
async def test_drain_waits_for_active_plan():
    executor = _make_executor()
    release = asyncio.Event()
    finished = False

    async def slow_call():
        nonlocal finished
        await release.wait()
        finished = True
        return 1

    task = asyncio.create_task(
        executor.execute(
            LocalStepPlan.single_call("slow"),
            MappingCallableRegistry({"slow": slow_call}),
        )
    )
    await asyncio.sleep(0)
    drain_task = asyncio.create_task(executor.drain())
    await asyncio.sleep(0.01)
    assert not drain_task.done()
    release.set()
    await task
    await drain_task
    assert finished


@pytest.mark.asyncio
async def test_transitive_dependency_chain_uses_serial_execution():
    executor = _make_executor()
    events: list[str] = []

    async def wait_inputs():
        events.append("wait")
        return 1

    async def call_user(value):
        events.append("call")
        return value + 1

    async def send(wait_value, call_value):
        events.append("send")
        return wait_value + call_value

    async def commit(value):
        events.append("commit")
        return value

    plan = LocalStepPlan(
        node_kinds=(
            int(LocalStepNodeKind.WAIT),
            int(LocalStepNodeKind.CALL),
            int(LocalStepNodeKind.SEND),
            int(LocalStepNodeKind.BARRIER),
        ),
        callable_ids=("wait", "call", "send", "commit"),
        resource_ids=(-1, -1, -1, -1),
        dependency_offsets=(0, 0, 1, 3, 4),
        dependencies=(0, 0, 1, 2),
        result_node=3,
    )

    result = await executor.execute(
        plan,
        MappingCallableRegistry(
            {
                "wait": wait_inputs,
                "call": call_user,
                "send": send,
                "commit": commit,
            }
        ),
    )

    assert result == 3
    assert events == ["wait", "call", "send", "commit"]
