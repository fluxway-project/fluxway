import asyncio
import threading
import time
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from fluxway.common.operator_policy import AsyncPolicy, SchedulePolicy
from fluxway.common.schedule_semantics import compile_phase_schedule
from fluxway.common.types import DrainBoundary
from fluxway import OperatorSpec
from fluxway.profiler import ProfilerConfig, ProfilerRuntimeFactory
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionResult,
    ExecutionStatus,
)
from fluxway.spmd.worker.admission import StepAdmissionState
from fluxway.spmd.worker.source_maintenance import SourcePipelineMaintenance
from fluxway.spmd.worker.worker import (
    LocalInputPrepareRuntime,
    SPMDWorker,
    _PreparedOperatorInput,
)


def _raw_worker_class():
    # pyrefly: ignore [missing-attribute]
    return SPMDWorker.__ray_actor_class__


def _attach_schedule_key_methods(stub, raw_cls):
    stub._uses_lane_aware_phase_priority = raw_cls._uses_lane_aware_phase_priority
    stub._lane_priority = raw_cls._lane_priority
    stub._phase_priority = raw_cls._phase_priority


def _attach_execution_queue_methods(stub, raw_cls):
    stub._execution_queues = {"__default__": stub._execution_queue}
    stub._initialize_execution_lanes = types.MethodType(
        raw_cls._initialize_execution_lanes,
        stub,
    )
    stub._execution_lane_for_op = types.MethodType(
        raw_cls._execution_lane_for_op,
        stub,
    )
    stub._execution_queue_for_lane = types.MethodType(
        raw_cls._execution_queue_for_lane,
        stub,
    )
    stub._execution_queue_for_op = types.MethodType(
        raw_cls._execution_queue_for_op,
        stub,
    )
    stub._put_execution_item = types.MethodType(raw_cls._put_execution_item, stub)
    _attach_schedule_key_methods(stub, raw_cls)


def _attach_admission_state(stub) -> None:
    stub._admission_state = StepAdmissionState(
        completed=stub._completed_step_count,
        scheduled=stub._scheduled_step_count,
        queue_membership=stub._in_op_step_queue,
        pending_blocked=stub._pending_blocked,
        pending_future=stub._pending_future_steps,
        pending_async_lag=stub._pending_async_lag_steps,
        pending_phase_order=stub._pending_phase_order_steps,
        ready_pipeline=stub._ready_pending_pipeline_steps,
        mailbox_ready=stub._mailbox_ready_steps,
        running=stub._running_plan_steps,
        finished=stub._finished_plan_steps,
        is_periodic=lambda op_name: stub.op_manager.is_periodic_trigger(op_name),
        is_runnable=lambda _op_name: True,
        hold_prepare_token=lambda _op_name, _step: False,
        async_lag_blocked=stub._async_lag_limit_reached,
        phase_order_blocked=stub._phase_order_step_blocked,
        can_start=stub._can_start_local_plan,
    )


def _make_pipeline_stub(meta: OperatorSpec):
    return _make_pipeline_stub_with_profiler(meta, profiler_enabled=False)


def _make_pipeline_stub_with_profiler(
    meta: OperatorSpec,
    *,
    profiler_enabled: bool,
):
    raw_cls = _raw_worker_class()
    op_manager = MagicMock()
    op_manager.get_operator_spec.return_value = meta
    op_manager._get_operator_spec.return_value = meta
    op_manager.is_periodic_trigger.return_value = False
    op_manager.is_source_operator.return_value = False
    profiler_runtime = (
        ProfilerRuntimeFactory.from_config(
            ProfilerConfig(trace_dir="tmp", run_id="worker_local_pipeline"),
            process_name="worker/test/rank:0",
        )
        if profiler_enabled
        else types.SimpleNamespace(enabled=False)
    )

    stub = types.SimpleNamespace(
        _op_manager=op_manager,
        op_manager=op_manager,
        _running_plan_steps={"op": set()},
        _finished_plan_steps={"op": {}},
        _ready_pending_pipeline_steps={"op": []},
        _pending_blocked={"op": []},
        _pending_future_steps={"op": []},
        _pending_async_lag_steps={"op": []},
        _pending_phase_order_steps={"op": []},
        _completed_step_count={"op": 0},
        _scheduled_step_count={"op": 0},
        _mailbox_ready_steps={"op": set()},
        _barriers_declared={},
        _in_op_step_queue=set(),
        _execution_queue=asyncio.PriorityQueue(),
        _running=True,
        _drain_step_count=None,
        _drain_cond=asyncio.Condition(),
        _mailbox=MagicMock(),
        _failure=None,
        _driver_handle=None,
        _profiler_runtime=profiler_runtime,
        _clear_direct_local_step=MagicMock(),
        config=types.SimpleNamespace(max_steps=None, operator_specs={"op": meta}),
        _operator_specs={"op": meta},
        spmd_id="test_spmd",
        rank=0,
        _local_spmd_code=7,
        _coherence_input_grants={},
        _coherence_input_intents={},
        _coherence_input_bindings={},
        _local_input_origins={},
        _coherence_revocations=set(),
        _coherence_input_tensors={},
        _coherence_alias_dependencies={},
        _background_tasks=set(),
        _plan_tasks=set(),
        _send_tasks={},
        _drain_boundary=DrainBoundary(),
        _source_operator_names=(),
        _source_control_operator_names=(),
        _source_maintenance_tasks={},
    )
    stub._mailbox.clear_step = AsyncMock()
    stub._operator_pipeline_depth = types.MethodType(
        raw_cls._operator_pipeline_depth, stub
    )
    stub._can_start_local_plan = types.MethodType(raw_cls._can_start_local_plan, stub)
    stub._record_local_plan_scheduled = types.MethodType(
        raw_cls._record_local_plan_scheduled, stub
    )
    stub._commit_finished_steps = types.MethodType(raw_cls._commit_finished_steps, stub)
    stub._commit_local_plan_result = types.MethodType(
        raw_cls._commit_local_plan_result, stub
    )
    stub._local_plan_send_output = types.MethodType(
        raw_cls._local_plan_send_output, stub
    )
    stub._local_plan_wait_inputs = types.MethodType(
        raw_cls._local_plan_wait_inputs, stub
    )
    stub._send_local_plan_result = types.MethodType(
        raw_cls._send_local_plan_result, stub
    )
    stub._local_plan_commit = types.MethodType(raw_cls._local_plan_commit, stub)
    stub._local_plan_await_send = types.MethodType(raw_cls._local_plan_await_send, stub)
    stub._spawn_send_task = types.MethodType(raw_cls._spawn_send_task, stub)
    stub._register_local_send = types.MethodType(raw_cls._register_local_send, stub)
    stub._step_transfer_cache_from_prepared_input = types.MethodType(
        raw_cls._step_transfer_cache_from_prepared_input, stub
    )
    stub._normalize_local_plan_result = types.MethodType(
        raw_cls._normalize_local_plan_result, stub
    )
    stub._result_requires_send = types.MethodType(raw_cls._result_requires_send, stub)
    stub._observe_local_send_registration = types.MethodType(
        raw_cls._observe_local_send_registration, stub
    )
    stub._require_local_spmd_code = types.MethodType(
        raw_cls._require_local_spmd_code, stub
    )
    stub._return_input_coherence = types.MethodType(
        raw_cls._return_input_coherence, stub
    )
    stub._register_coherence_alias_dependencies = types.MethodType(
        raw_cls._register_coherence_alias_dependencies,
        stub,
    )
    stub._commit_allocation_handoffs = types.MethodType(
        raw_cls._commit_allocation_handoffs,
        stub,
    )
    stub._await_send_registration = types.MethodType(
        raw_cls._await_send_registration,
        stub,
    )
    stub._on_background_task_done = types.MethodType(
        raw_cls._on_background_task_done, stub
    )
    stub._record_failure = types.MethodType(raw_cls._record_failure, stub)
    stub._spawn_background_task = types.MethodType(raw_cls._spawn_background_task, stub)
    stub._create_supervised_task = types.MethodType(
        raw_cls._create_supervised_task, stub
    )
    stub._on_prepare_ready = types.MethodType(raw_cls._on_prepare_ready, stub)
    stub._input_prepare_runtime = LocalInputPrepareRuntime(
        background_tasks=stub._background_tasks,
        on_task_done=stub._on_background_task_done,
        profiler_runtime=profiler_runtime,
        on_prepare_ready=stub._on_prepare_ready,
        schedule_key=lambda op_name, step: (0, 0, step, op_name),
    )
    stub._prepare_inputs_if_policy_allows = types.MethodType(
        raw_cls._prepare_inputs_if_policy_allows,
        stub,
    )
    stub._drain_blocks_step = types.MethodType(raw_cls._drain_blocks_step, stub)
    stub._start_prepare_and_enqueue_token = types.MethodType(
        raw_cls._start_prepare_and_enqueue_token,
        stub,
    )
    stub._enqueue_prepare_ready_refresh = types.MethodType(
        raw_cls._enqueue_prepare_ready_refresh,
        stub,
    )
    stub._enqueue = types.MethodType(raw_cls._enqueue, stub)
    _attach_execution_queue_methods(stub, raw_cls)
    stub._effective_schedule_key = types.MethodType(
        raw_cls._effective_schedule_key,
        stub,
    )
    stub._prepare_schedule_state = types.MethodType(
        raw_cls._prepare_schedule_state,
        stub,
    )
    stub._should_prepare_inputs_on_ready = types.MethodType(
        raw_cls._should_prepare_inputs_on_ready,
        stub,
    )
    stub._deadline_boost_active = types.MethodType(
        raw_cls._deadline_boost_active,
        stub,
    )
    stub._should_hold_prepare_pending_token = types.MethodType(
        raw_cls._should_hold_prepare_pending_token,
        stub,
    )
    stub._bounded_async_ops = types.MethodType(raw_cls._bounded_async_ops, stub)
    stub._async_lag_limit_reached = types.MethodType(
        raw_cls._async_lag_limit_reached,
        stub,
    )
    stub._phase_schedule = compile_phase_schedule(
        {
            name: spec.runtime_policy.schedule
            for name, spec in stub._operator_specs.items()
        }
    )
    stub._phase_order_step_blocked = types.MethodType(
        raw_cls._phase_order_step_blocked,
        stub,
    )
    stub.set_drain_boundary = types.MethodType(
        raw_cls.set_drain_boundary,
        stub,
    )
    stub._set_drain_boundary = types.MethodType(
        raw_cls._set_drain_boundary,
        stub,
    )
    stub._source_is_exhausted = types.MethodType(raw_cls._source_is_exhausted, stub)
    stub._source_blocks_step = types.MethodType(raw_cls._source_blocks_step, stub)
    stub._advance_step = types.MethodType(raw_cls._advance_step, stub)
    stub._trigger_source_operators = types.MethodType(
        raw_cls._trigger_source_operators,
        stub,
    )
    _attach_admission_state(stub)
    stub._start_local_step_plan = types.MethodType(
        raw_cls._start_local_step_plan,
        stub,
    )
    stub._source_maintenance = SourcePipelineMaintenance(
        tasks=stub._source_maintenance_tasks,
        spawn=stub._spawn_background_task,
        is_source=lambda op_name: stub.op_manager.is_source_operator(op_name),
        running=lambda: stub._running,
        exhausted=stub._source_is_exhausted,
        depth=stub._operator_pipeline_depth,
        completed=lambda op_name: stub._completed_step_count.get(op_name, 0),
        max_steps=lambda: stub.config.max_steps,
        drain_blocks=stub._drain_blocks_step,
        known_steps=stub._admission_state.known_steps,
        enqueue=stub._enqueue,
    )
    stub._report_op_completed = MagicMock()
    return stub


def _prepared(cache=None):
    return _PreparedOperatorInput(value={}, step_transfer_cache=dict(cache or {}))


@pytest.mark.asyncio
async def test_input_prepare_threads_use_stable_schedule_priority():
    started = []
    release = threading.Event()
    priorities = {
        ("scatter", 0): (30, 0, 0, "scatter"),
        ("prepare", 2): (20, 0, 2, "prepare"),
        ("prepare", 1): (10, 0, 1, "prepare"),
    }
    runtime = LocalInputPrepareRuntime(
        background_tasks=set(),
        on_task_done=lambda _task: None,
        max_workers=1,
        schedule_key=lambda op_name, step: priorities[(op_name, step)],
    )

    def run(op_name, step):
        started.append((op_name, step))
        if (op_name, step) == ("scatter", 0):
            release.wait(timeout=5)
        return op_name, step

    async def submit(op_name, step):
        return await runtime.run_assemble(
            op_name,
            step,
            lambda: run(op_name, step),
        )

    try:
        running = asyncio.create_task(submit("scatter", 0))
        while not started:
            await asyncio.sleep(0)
        future_prepare = asyncio.create_task(submit("prepare", 2))
        next_prepare = asyncio.create_task(submit("prepare", 1))
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.gather(running, future_prepare, next_prepare)
        assert started == [("scatter", 0), ("prepare", 1), ("prepare", 2)]
    finally:
        release.set()
        await runtime.cancel_pending()
        runtime.close()


@pytest.mark.asyncio
async def test_input_prepare_threads_retention_fifo_for_equal_schedule_keys():
    started = []
    release = threading.Event()
    runtime = LocalInputPrepareRuntime(
        background_tasks=set(),
        on_task_done=lambda _task: None,
        max_workers=1,
        schedule_key=lambda op_name, step: (0, 0, step, op_name),
    )

    def run(name):
        started.append(name)
        if name == "running":
            release.wait(timeout=5)
        return name

    try:
        running = asyncio.create_task(
            runtime.run_assemble("prepare", 0, lambda: run("running"))
        )
        while not started:
            await asyncio.sleep(0)
        first = asyncio.create_task(
            runtime.run_assemble("prepare", 1, lambda: run("first"))
        )
        second = asyncio.create_task(
            runtime.run_assemble("prepare", 1, lambda: run("second"))
        )
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.gather(running, first, second)
        assert started == ["running", "first", "second"]
    finally:
        release.set()
        await runtime.cancel_pending()
        runtime.close()


def test_local_pipeline_depth_one_preserves_single_running_slot():
    stub = _make_pipeline_stub(OperatorSpec(name="op"))

    assert stub._can_start_local_plan("op")
    stub._running_plan_steps["op"].add(0)
    assert not stub._can_start_local_plan("op")


def test_local_pipeline_depth_allows_multiple_running_steps():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=8, commit="relaxed"),
        )
    )

    for step in range(8):
        assert stub._can_start_local_plan("op")
        stub._record_local_plan_scheduled("op", step)
        stub._running_plan_steps["op"].add(step)

    assert stub._scheduled_step_count["op"] == 8
    assert not stub._can_start_local_plan("op")


def test_deadline_boost_schedule_prioritizes_stale_boundary_update():
    raw_cls = _raw_worker_class()
    lookup = OperatorSpec(
        name="lookup",
        async_policy=AsyncPolicy.bounded_stale(
            depth=4,
            max_staleness=3,
            schedule=SchedulePolicy.deadline_boost(boost_priority=100),
        ),
    )
    update = OperatorSpec(
        name="update",
        async_policy=AsyncPolicy.bounded_stale(
            depth=4,
            max_staleness=3,
            schedule=SchedulePolicy.deadline_boost(boost_priority=200),
        ),
    )
    op_manager = MagicMock()
    op_manager._get_operator_spec.side_effect = lambda name: {
        "lookup": lookup,
        "update": update,
    }[name]
    stub = types.SimpleNamespace(
        op_manager=op_manager,
        config=types.SimpleNamespace(
            operator_specs={"lookup": lookup, "update": update}
        ),
        _operator_specs={"lookup": lookup, "update": update},
        _completed_step_count={"lookup": 0, "update": 0},
    )
    stub._effective_schedule_key = types.MethodType(
        raw_cls._effective_schedule_key, stub
    )
    stub._prepare_schedule_state = types.MethodType(
        raw_cls._prepare_schedule_state, stub
    )
    stub._deadline_boost_active = types.MethodType(raw_cls._deadline_boost_active, stub)
    _attach_schedule_key_methods(stub, raw_cls)

    assert stub._effective_schedule_key("update", 3) < stub._effective_schedule_key(
        "lookup",
        3,
    )
    assert stub._effective_schedule_key("update", 2) == (0, 0, 2, "update")


def test_bounded_stale_lookup_stops_at_lag_boundary_while_update_closes_lag():
    raw_cls = _raw_worker_class()
    lookup = OperatorSpec(
        name="lookup",
        async_policy=AsyncPolicy.bounded_stale(
            depth=8,
            max_staleness=7,
            schedule=SchedulePolicy.deadline_boost(boost_priority=100),
        ),
    )
    update = OperatorSpec(
        name="update",
        async_policy=AsyncPolicy.bounded_stale(
            depth=8,
            max_staleness=7,
            schedule=SchedulePolicy.deadline_boost(boost_priority=200),
        ),
    )
    metas = {"lookup": lookup, "update": update}
    op_manager = MagicMock()
    op_manager._get_operator_spec.side_effect = lambda name: metas[name]
    stub = types.SimpleNamespace(
        op_manager=op_manager,
        config=types.SimpleNamespace(operator_specs=metas),
        _operator_specs=metas,
        _completed_step_count={"lookup": 7, "update": 1},
    )
    stub._bounded_async_ops = types.MethodType(raw_cls._bounded_async_ops, stub)
    stub._async_lag_limit_reached = types.MethodType(
        raw_cls._async_lag_limit_reached,
        stub,
    )

    assert not stub._async_lag_limit_reached("lookup", 2)
    assert not stub._async_lag_limit_reached("lookup", 8)
    assert stub._async_lag_limit_reached("lookup", 9)
    assert not stub._async_lag_limit_reached("update", 1)


@pytest.mark.asyncio
async def test_bounded_stale_lag_release_requeues_lookup_after_update_progress():
    lookup = OperatorSpec(
        name="lookup",
        async_policy=AsyncPolicy.bounded_stale(
            depth=8,
            max_staleness=7,
            schedule=SchedulePolicy.deadline_boost(boost_priority=100),
        ),
    )
    update = OperatorSpec(
        name="update",
        async_policy=AsyncPolicy.bounded_stale(
            depth=8,
            max_staleness=7,
            schedule=SchedulePolicy.deadline_boost(boost_priority=200),
        ),
    )
    stub = _make_pipeline_stub(lookup)
    metas = {"lookup": lookup, "update": update}
    stub.config.operator_specs = metas
    stub._operator_specs = metas
    stub.op_manager._get_operator_spec.side_effect = lambda name: metas[name]
    stub.op_manager.get_operator_spec.side_effect = lambda name: metas[name]
    stub._completed_step_count = {"lookup": 7, "update": 1}
    stub._scheduled_step_count = {"lookup": 9, "update": 1}
    stub._pending_async_lag_steps = {"lookup": [9], "update": []}
    stub._in_op_step_queue = set()
    _attach_admission_state(stub)

    await stub._admission_state.release_async_lag(stub._enqueue)
    assert stub._pending_async_lag_steps["lookup"] == [9]
    assert stub._execution_queue.empty()

    stub._completed_step_count["update"] = 2
    await stub._admission_state.release_async_lag(stub._enqueue)
    assert stub._pending_async_lag_steps["lookup"] == []
    queued = await stub._execution_queue.get()
    assert queued == (-100, 0, 9, "lookup")


def test_phase_order_schedule_places_forward_before_backward_same_micro_step():
    raw_cls = _raw_worker_class()
    order = ("forward", "backward", "optimizer")
    forward = OperatorSpec(
        name="forward",
        async_policy=AsyncPolicy.pipelined(
            depth=2,
            schedule=SchedulePolicy.phase_order(phase="forward", phase_order=order),
        ),
    )
    backward = OperatorSpec(
        name="backward",
        async_policy=AsyncPolicy.pipelined(
            depth=2,
            schedule=SchedulePolicy.phase_order(phase="backward", phase_order=order),
        ),
    )
    op_manager = MagicMock()
    op_manager._get_operator_spec.side_effect = lambda name: {
        "forward": forward,
        "backward": backward,
    }[name]
    stub = types.SimpleNamespace(
        op_manager=op_manager,
        config=types.SimpleNamespace(
            operator_specs={"forward": forward, "backward": backward}
        ),
        _operator_specs={"forward": forward, "backward": backward},
        _completed_step_count={"forward": 0, "backward": 0},
    )
    stub._effective_schedule_key = types.MethodType(
        raw_cls._effective_schedule_key, stub
    )
    stub._prepare_schedule_state = types.MethodType(
        raw_cls._prepare_schedule_state, stub
    )
    stub._deadline_boost_active = types.MethodType(raw_cls._deadline_boost_active, stub)
    _attach_schedule_key_methods(stub, raw_cls)

    assert stub._effective_schedule_key("forward", 4) < stub._effective_schedule_key(
        "backward",
        4,
    )
    assert stub._effective_schedule_key("backward", 4) < stub._effective_schedule_key(
        "forward",
        5,
    )


@pytest.mark.asyncio
async def test_phase_order_gate_blocks_train_until_previous_update_commits():
    raw_cls = _raw_worker_class()
    order = (
        "prepare_dense_inputs",
        "train_core",
        "scatter_sparse_grads",
        "update",
    )

    def meta(name):
        policy = SchedulePolicy.phase_order(phase=name, phase_order=order)
        return OperatorSpec(
            name=name,
            async_policy=AsyncPolicy.pipelined(depth=2, schedule=policy),
        )

    metas = {name: meta(name) for name in order}
    op_manager = MagicMock()
    op_manager._get_operator_spec.side_effect = lambda name: metas[name]
    op_manager.get_operator_spec.side_effect = lambda name: metas[name]
    stub = types.SimpleNamespace(
        op_manager=op_manager,
        config=types.SimpleNamespace(operator_specs=metas),
        _operator_specs=metas,
        _completed_step_count={
            "prepare_dense_inputs": 2,
            "train_core": 1,
            "scatter_sparse_grads": 1,
            "update": 0,
        },
        _scheduled_step_count={
            "prepare_dense_inputs": 2,
            "train_core": 1,
            "scatter_sparse_grads": 1,
            "update": 0,
        },
        _pending_phase_order_steps={"train_core": [1]},
        _execution_queue=asyncio.PriorityQueue(),
        _in_op_step_queue=set(),
        _running=True,
        _profiler_runtime=types.SimpleNamespace(enabled=False),
        spmd_id="dense",
        rank=0,
    )
    stub._phase_schedule = compile_phase_schedule(
        {name: spec.runtime_policy.schedule for name, spec in metas.items()}
    )
    stub._phase_order_step_blocked = types.MethodType(
        raw_cls._phase_order_step_blocked,
        stub,
    )
    stub._enqueue = types.MethodType(raw_cls._enqueue, stub)
    stub._record_local_plan_scheduled = types.MethodType(
        raw_cls._record_local_plan_scheduled, stub
    )
    _attach_execution_queue_methods(stub, raw_cls)
    stub._effective_schedule_key = types.MethodType(
        raw_cls._effective_schedule_key,
        stub,
    )
    stub._prepare_schedule_state = types.MethodType(
        raw_cls._prepare_schedule_state,
        stub,
    )
    stub._deadline_boost_active = types.MethodType(raw_cls._deadline_boost_active, stub)
    stub._should_prepare_inputs_on_ready = lambda _op_name: False
    stub._pending_blocked = {name: [] for name in metas}
    stub._pending_future_steps = {name: [] for name in metas}
    stub._pending_async_lag_steps = {name: [] for name in metas}
    stub._ready_pending_pipeline_steps = {name: [] for name in metas}
    stub._mailbox_ready_steps = {name: set() for name in metas}
    stub._running_plan_steps = {name: set() for name in metas}
    stub._finished_plan_steps = {name: {} for name in metas}
    stub._can_start_local_plan = lambda _op_name: True
    stub._async_lag_limit_reached = lambda _op_name, _step: False
    _attach_admission_state(stub)

    assert not stub._phase_order_step_blocked("prepare_dense_inputs", 1)
    assert stub._phase_order_step_blocked("train_core", 1)

    stub._completed_step_count["update"] = 1
    assert not stub._phase_order_step_blocked("train_core", 1)

    await stub._admission_state.release_phase_order(stub._enqueue)
    assert stub._pending_phase_order_steps["train_core"] == []
    assert await stub._execution_queue.get() == (5, 0, 1, "train_core")


@pytest.mark.asyncio
async def test_phase_graph_allows_scatter_and_update_parallel_but_blocks_next_train():
    raw_cls = _raw_worker_class()
    order = (
        "prepare_dense_inputs",
        "train_core",
        "scatter_sparse_grads",
        "update",
    )
    dependencies = (
        ("prepare_dense_inputs", "train_core", 0),
        ("train_core", "scatter_sparse_grads", 0),
        ("train_core", "update", 0),
        ("update", "train_core", 1),
    )

    def meta(name):
        policy = SchedulePolicy.phase_graph(
            phase=name,
            phase_order=order,
            dependencies=dependencies,
        )
        return OperatorSpec(
            name=name,
            async_policy=AsyncPolicy.pipelined(depth=2, schedule=policy),
        )

    metas = {name: meta(name) for name in order}
    op_manager = MagicMock()
    op_manager._get_operator_spec.side_effect = lambda name: metas[name]
    op_manager.get_operator_spec.side_effect = lambda name: metas[name]
    stub = types.SimpleNamespace(
        op_manager=op_manager,
        config=types.SimpleNamespace(operator_specs=metas),
        _operator_specs=metas,
        _completed_step_count={
            "prepare_dense_inputs": 2,
            "train_core": 1,
            "scatter_sparse_grads": 0,
            "update": 0,
        },
        _scheduled_step_count={
            "prepare_dense_inputs": 2,
            "train_core": 1,
            "scatter_sparse_grads": 0,
            "update": 0,
        },
        _pending_phase_order_steps={"update": [0], "train_core": [1]},
        _execution_queue=asyncio.PriorityQueue(),
        _in_op_step_queue=set(),
        _running=True,
        _profiler_runtime=types.SimpleNamespace(enabled=False),
        spmd_id="dense",
        rank=0,
    )
    stub._phase_schedule = compile_phase_schedule(
        {name: spec.runtime_policy.schedule for name, spec in metas.items()}
    )
    stub._phase_order_step_blocked = types.MethodType(
        raw_cls._phase_order_step_blocked,
        stub,
    )
    stub._enqueue = types.MethodType(raw_cls._enqueue, stub)
    stub._record_local_plan_scheduled = types.MethodType(
        raw_cls._record_local_plan_scheduled, stub
    )
    _attach_execution_queue_methods(stub, raw_cls)
    stub._effective_schedule_key = types.MethodType(
        raw_cls._effective_schedule_key,
        stub,
    )
    stub._prepare_schedule_state = types.MethodType(
        raw_cls._prepare_schedule_state,
        stub,
    )
    stub._deadline_boost_active = types.MethodType(raw_cls._deadline_boost_active, stub)
    stub._should_prepare_inputs_on_ready = lambda _op_name: False
    stub._pending_blocked = {name: [] for name in metas}
    stub._pending_future_steps = {name: [] for name in metas}
    stub._pending_async_lag_steps = {name: [] for name in metas}
    stub._ready_pending_pipeline_steps = {name: [] for name in metas}
    stub._mailbox_ready_steps = {name: set() for name in metas}
    stub._running_plan_steps = {name: set() for name in metas}
    stub._finished_plan_steps = {name: {} for name in metas}
    stub._can_start_local_plan = lambda _op_name: True
    stub._async_lag_limit_reached = lambda _op_name, _step: False
    _attach_admission_state(stub)

    assert not stub._phase_order_step_blocked("scatter_sparse_grads", 0)
    assert not stub._phase_order_step_blocked("update", 0)
    assert stub._phase_order_step_blocked("train_core", 1)

    await stub._admission_state.release_phase_order(stub._enqueue)
    assert stub._pending_phase_order_steps["update"] == []
    assert stub._pending_phase_order_steps["train_core"] == [1]
    assert await stub._execution_queue.get() == (3, 0, 0, "update")

    stub._completed_step_count["update"] = 1
    await stub._admission_state.release_phase_order(stub._enqueue)
    assert stub._pending_phase_order_steps["train_core"] == []
    assert await stub._execution_queue.get() == (5, 0, 1, "train_core")


@pytest.mark.asyncio
async def test_relaxed_finish_does_not_report_non_contiguous_step():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=8, commit="relaxed"),
        )
    )
    result = ExecutionResult(status=ExecutionStatus.SUCCESS, data="ok")

    await stub._commit_local_plan_result("op", 1, result)
    assert stub._completed_step_count["op"] == 0
    stub._report_op_completed.assert_not_called()

    await stub._commit_local_plan_result("op", 0, result)
    assert stub._completed_step_count["op"] == 2
    assert stub._report_op_completed.call_args_list[0].args == ("op", 1)
    assert stub._report_op_completed.call_args_list[1].args == ("op", 2)


@pytest.mark.asyncio
async def test_pending_future_steps_move_into_pipeline_window_by_scheduled_watermark():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=2, commit="relaxed"),
        )
    )
    started: list[int] = []

    def start_local_step_plan(op_name: str, step: int) -> None:
        assert op_name == "op"
        stub._record_local_plan_scheduled(op_name, step)
        stub._running_plan_steps[op_name].add(step)
        started.append(step)

    stub._start_local_step_plan = start_local_step_plan
    stub._pending_future_steps["op"].extend([1, 2])

    start_local_step_plan("op", 0)
    await stub._admission_state.start_pending(
        "op",
        running=lambda: stub._running,
        source_exhausted=stub._source_is_exhausted,
        start_plan=stub._start_local_step_plan,
    )

    assert started == [0, 1]
    assert stub._ready_pending_pipeline_steps["op"] == [2]
    assert stub._pending_future_steps["op"] == []


@pytest.mark.asyncio
async def test_pipelined_wait_consumes_completed_prepare_future():
    stub = _make_pipeline_stub(
        OperatorSpec(name="op", async_policy=AsyncPolicy.pipelined(depth=2))
    )
    prepared = asyncio.Event()
    calls = 0

    async def prepare_input(op_name: str, step: int):
        nonlocal calls
        calls += 1
        assert op_name == "op"
        assert step == 0
        await asyncio.sleep(0.05)
        prepared.set()
        return {"x": 1}

    stub._execute_operator_prepare_input = prepare_input

    stub._prepare_inputs_if_policy_allows("op", 0)
    await prepared.wait()

    start = time.perf_counter()
    result = await stub._local_plan_wait_inputs("op", 0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert result == {"x": 1}
    assert elapsed_ms < 20.0
    assert calls == 1


@pytest.mark.asyncio
async def test_start_prepare_and_enqueue_token_enqueues_before_prepare_done():
    stub = _make_pipeline_stub(
        OperatorSpec(name="op", async_policy=AsyncPolicy.pipelined(depth=2))
    )
    release = asyncio.Event()
    started = asyncio.Event()

    async def prepare_input(op_name: str, step: int):
        started.set()
        await release.wait()
        return {"x": 1}

    stub._execute_operator_prepare_input = prepare_input

    await stub._start_prepare_and_enqueue_token("op", 0)
    await started.wait()

    assert ("op", 0) in stub._input_prepare_runtime._tasks
    assert ("op", 0) in stub._in_op_step_queue
    queued = await stub._execution_queue.get()
    assert queued == (0, 1, 0, "op")

    release.set()
    await stub._input_prepare_runtime.consume(
        "op",
        0,
        lambda: prepare_input("op", 0),
    )


@pytest.mark.asyncio
async def test_source_trigger_fills_pipeline_window_from_async_policy_depth():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=3),
        )
    )
    stub.op_manager.get_source_operators.return_value = ["op"]
    stub.op_manager.is_source_operator.return_value = True

    await stub._trigger_source_operators()

    queued = [await stub._execution_queue.get() for _ in range(3)]
    assert queued == [
        (0, 0, 0, "op"),
        (0, 0, 1, "op"),
        (0, 0, 2, "op"),
    ]
    assert stub._in_op_step_queue == {("op", 0), ("op", 1), ("op", 2)}


@pytest.mark.asyncio
async def test_source_trigger_window_respects_max_steps():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=4),
        )
    )
    stub.config.max_steps = 2
    stub.op_manager.get_source_operators.return_value = ["op"]
    stub.op_manager.is_source_operator.return_value = True

    await stub._trigger_source_operators()

    queued = [await stub._execution_queue.get() for _ in range(2)]
    assert queued == [
        (0, 0, 0, "op"),
        (0, 0, 1, "op"),
    ]
    assert stub._execution_queue.empty()


@pytest.mark.asyncio
async def test_source_pipeline_maintenance_requests_share_one_owner():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=2),
        )
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fill(op_name: str):
        nonlocal calls
        assert op_name == "op"
        calls += 1
        entered.set()
        await release.wait()

    stub._source_maintenance.fill = fill
    stub._source_maintenance.fill = fill
    first = stub._source_maintenance.request("op")
    second = stub._source_maintenance.request("op")
    assert first is second
    await entered.wait()
    assert calls == 1
    release.set()
    await first
    await asyncio.sleep(0)
    assert stub._source_maintenance_tasks == {}


@pytest.mark.asyncio
async def test_source_pipeline_refill_stops_at_drain_ceiling():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=4),
        )
    )
    stub.op_manager.is_source_operator.return_value = True
    stub._drain_step_count = 3
    await stub._source_maintenance.fill("op")
    assert sorted(step for _, step in stub._in_op_step_queue) == [0, 1, 2]


@pytest.mark.asyncio
async def test_clear_drain_boundary_requests_source_maintenance():
    raw_cls = _raw_worker_class()
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=2),
        )
    )
    requested = []

    def request(op_name: str):
        requested.append(op_name)
        return asyncio.create_task(asyncio.sleep(0))

    stub._source_maintenance.request = request
    stub._source_operator_names = ("op",)
    stub._set_drain_boundary = types.MethodType(
        lambda self, _step_count, _terminal_cut=None: asyncio.sleep(0),
        stub,
    )
    stub.clear_drain_boundary = types.MethodType(
        raw_cls.clear_drain_boundary,
        stub,
    )
    await stub.clear_drain_boundary()
    assert requested == ["op"]


@pytest.mark.asyncio
async def test_prepare_ready_refresh_adds_ready_queue_key():
    stub = _make_pipeline_stub(
        OperatorSpec(name="op", async_policy=AsyncPolicy.pipelined(depth=2))
    )

    async def prepare_input(op_name: str, step: int):
        return {"x": 1}

    stub._execute_operator_prepare_input = prepare_input

    await stub._start_prepare_and_enqueue_token("op", 0)
    first = await stub._execution_queue.get()
    assert first == (0, 1, 0, "op")

    await stub._input_prepare_runtime.await_ready(
        "op",
        0,
        lambda: prepare_input("op", 0),
    )
    await asyncio.sleep(0)

    refreshed = await stub._execution_queue.get()
    assert refreshed == (0, 0, 0, "op")


@pytest.mark.asyncio
async def test_prepare_ready_tie_breaker_prefers_ready_same_priority():
    stub = _make_pipeline_stub(
        OperatorSpec(name="op", async_policy=AsyncPolicy.pipelined(depth=2))
    )

    async def pending_prepare():
        await asyncio.sleep(10)

    async def ready_prepare():
        return {"x": 1}

    stub._input_prepare_runtime.start("op", 0, pending_prepare)
    ready_task = stub._input_prepare_runtime.start("op", 1, ready_prepare)
    await ready_task

    assert stub._effective_schedule_key("op", 1) < stub._effective_schedule_key("op", 0)
    for task in list(stub._background_tasks):
        task.cancel()
    await asyncio.gather(*stub._background_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_prepare_pending_token_holds_until_ready_for_fifo_policy():
    stub = _make_pipeline_stub(
        OperatorSpec(name="op", async_policy=AsyncPolicy.pipelined(depth=2))
    )
    release = asyncio.Event()
    started = asyncio.Event()

    async def prepare_input(op_name: str, step: int):
        started.set()
        await release.wait()
        return {"x": 1}

    stub._execute_operator_prepare_input = prepare_input
    stub._prepare_inputs_if_policy_allows("op", 0)
    await started.wait()

    assert stub._should_hold_prepare_pending_token("op", 0)
    release.set()
    await stub._input_prepare_runtime.await_ready(
        "op",
        0,
        lambda: prepare_input("op", 0),
    )
    assert not stub._should_hold_prepare_pending_token("op", 0)


@pytest.mark.asyncio
async def test_deadline_boosted_prepare_pending_token_does_not_hold():
    meta = OperatorSpec(
        name="op",
        async_policy=AsyncPolicy.bounded_stale(
            depth=4,
            max_staleness=3,
            schedule=SchedulePolicy.deadline_boost(boost_priority=100),
        ),
    )
    stub = _make_pipeline_stub(meta)
    release = asyncio.Event()
    started = asyncio.Event()

    async def prepare_input(op_name: str, step: int):
        started.set()
        await release.wait()
        return {"x": 1}

    stub._execute_operator_prepare_input = prepare_input
    stub._prepare_inputs_if_policy_allows("op", 3)
    await started.wait()

    assert not stub._should_hold_prepare_pending_token("op", 3)
    release.set()
    await stub._input_prepare_runtime.await_ready(
        "op",
        3,
        lambda: prepare_input("op", 3),
    )


@pytest.mark.asyncio
async def test_sync_wait_does_not_start_background_prepare_future():
    stub = _make_pipeline_stub(OperatorSpec(name="op", async_policy=AsyncPolicy.sync()))
    calls = 0

    async def prepare_input(op_name: str, step: int):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return {"x": 1}

    stub._execute_operator_prepare_input = prepare_input

    stub._prepare_inputs_if_policy_allows("op", 0)
    assert not stub._input_prepare_runtime._tasks

    result = await stub._local_plan_wait_inputs("op", 0)
    assert result == {"x": 1}
    assert calls == 1


@pytest.mark.asyncio
async def test_prepare_skips_step_at_active_drain_target():
    stub = _make_pipeline_stub(
        OperatorSpec(name="op", async_policy=AsyncPolicy.pipelined(depth=2))
    )
    stub._drain_step_count = 1

    async def prepare_input(op_name: str, step: int):
        raise AssertionError("prepare should not start while drain target blocks step")

    stub._execute_operator_prepare_input = prepare_input

    stub._prepare_inputs_if_policy_allows("op", 1)
    assert not stub._input_prepare_runtime._tasks


@pytest.mark.asyncio
async def test_setting_drain_target_cancels_speculative_prepare_past_target():
    stub = _make_pipeline_stub(
        OperatorSpec(name="op", async_policy=AsyncPolicy.pipelined(depth=2))
    )
    started = asyncio.Event()

    async def prepare_input(op_name: str, step: int):
        started.set()
        await asyncio.sleep(10)

    stub._execute_operator_prepare_input = prepare_input

    stub._prepare_inputs_if_policy_allows("op", 3)
    await started.wait()
    assert ("op", 3) in stub._input_prepare_runtime._tasks

    await stub.set_drain_boundary(2)
    assert ("op", 3) not in stub._input_prepare_runtime._tasks


@pytest.mark.asyncio
async def test_checkpoint_outbound_fence_ignores_parked_future_plans():
    stub = _make_pipeline_stub(
        OperatorSpec(name="op", async_policy=AsyncPolicy.pipelined(depth=4))
    )
    parked = asyncio.Event()

    async def future_prepare():
        await parked.wait()

    stub._input_prepare_runtime.start("op", 2, future_prepare)
    future_plan = asyncio.create_task(parked.wait())
    stub._plan_tasks.add(future_plan)
    stub._delivery_runtime = types.SimpleNamespace(drain=AsyncMock())
    stub._source_acknowledgements = types.SimpleNamespace(flush=AsyncMock())
    try:
        await asyncio.wait_for(
            _raw_worker_class()._drain_outbound_tasks(stub), timeout=0.1
        )
        assert stub._input_prepare_runtime.is_pending("op", 2)
        assert not future_plan.done()
        stub._delivery_runtime.drain.assert_awaited_once_with("worker_drain_outbound")
        stub._source_acknowledgements.flush.assert_awaited_once_with()
    finally:
        future_plan.cancel()
        await stub._input_prepare_runtime.cancel_pending()
        await asyncio.gather(future_plan, return_exceptions=True)


@pytest.mark.asyncio
async def test_local_barrier_waits_for_frame_build_task():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=2, commit="relaxed"),
        )
    )

    frame_build_done = asyncio.Event()

    async def fake_send_output(op_name, result, *, exec_step, step_transfer_cache):
        assert op_name == "op"
        assert result.status is ExecutionStatus.SUCCESS
        assert exec_step == 0
        await asyncio.sleep(0.05)
        frame_build_done.set()
        return ()

    stub._send_output = fake_send_output

    send_token = await stub._local_plan_send_output("op", 0, _prepared(), "payload")
    start = time.perf_counter()
    await stub._local_plan_await_send(send_token)
    await stub._local_plan_commit("op", 0, send_token)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert elapsed_ms >= 50.0
    assert send_token.send_task is not None
    assert frame_build_done.is_set()


@pytest.mark.asyncio
async def test_local_send_node_task_does_not_wait_delivery_future():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=2, commit="relaxed"),
        )
    )

    delivery_done = asyncio.Event()

    async def slow_delivery():
        await asyncio.sleep(0.05)
        delivery_done.set()

    async def fake_send_output(op_name, result, *, exec_step, step_transfer_cache):
        assert op_name == "op"
        assert result.status is ExecutionStatus.SUCCESS
        assert exec_step == 0
        return (asyncio.create_task(slow_delivery()),)

    stub._send_output = fake_send_output

    send_token = await stub._local_plan_send_output("op", 0, _prepared(), "payload")
    await send_token.send_task
    assert not delivery_done.is_set()
    await stub._local_plan_commit("op", 0, send_token)
    assert not delivery_done.is_set()
    await delivery_done.wait()


@pytest.mark.asyncio
async def test_local_barrier_observes_immediate_send_registration_failure():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=2, commit="relaxed"),
        )
    )

    async def failing_send_output(op_name, result, *, exec_step, step_transfer_cache):
        raise RuntimeError("frame build failed")

    stub._send_output = failing_send_output

    send_token = await stub._local_plan_send_output("op", 0, _prepared(), "payload")
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="frame build failed"):
        await stub._local_plan_await_send(send_token)


@pytest.mark.asyncio
async def test_local_barrier_propagates_late_send_failure():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=2, commit="relaxed"),
        )
    )

    async def failing_send_output(op_name, result, *, exec_step, step_transfer_cache):
        await asyncio.sleep(0.01)
        raise RuntimeError("late frame build failed")

    stub._send_output = failing_send_output

    send_token = await stub._local_plan_send_output("op", 0, _prepared(), "payload")
    with pytest.raises(RuntimeError, match="late frame build failed"):
        await stub._local_plan_await_send(send_token)
    assert stub._failure is not None
    assert stub._failure.phase == "SEND"
    assert stub._failure.message == "late frame build failed"


@pytest.mark.asyncio
async def test_skipped_local_send_token_registers_completion_send_task():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=2, commit="relaxed"),
        )
    )
    stub._send_output = AsyncMock(return_value=())

    result = ExecutionResult(status=ExecutionStatus.SKIPPED)
    send_token = await stub._local_plan_send_output("op", 0, result, result)
    assert send_token.send_task is not None
    await send_token.send_task
    stub._send_output.assert_awaited_once_with(
        "op",
        result,
        exec_step=0,
        step_transfer_cache={},
    )
    await stub._local_plan_commit("op", 0, send_token)


@pytest.mark.asyncio
async def test_local_send_uses_wait_cache_before_background_send_runs():
    stub = _make_pipeline_stub(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=2, commit="relaxed"),
        )
    )
    prepared = _prepared({16: "mirror-source"})
    release = asyncio.Event()
    observed = {}

    async def fake_send_output(op_name, result, *, exec_step, step_transfer_cache):
        await release.wait()
        observed.update(step_transfer_cache)
        return ()

    stub._send_output = fake_send_output

    send_token = await stub._local_plan_send_output("op", 0, prepared, "payload")
    prepared.step_transfer_cache.clear()
    release.set()
    await send_token.send_task

    assert observed == {16: "mirror-source"}


@pytest.mark.asyncio
async def test_local_send_commit_profiler_spans_enqueue_wait_not_delivery_wait():
    stub = _make_pipeline_stub_with_profiler(
        OperatorSpec(
            name="op",
            async_policy=AsyncPolicy.pipelined(depth=2, commit="relaxed"),
        ),
        profiler_enabled=True,
    )

    delivery_done = asyncio.Event()

    async def slow_outbound():
        await asyncio.sleep(0.05)
        delivery_done.set()

    async def fake_send_output(op_name, result, *, exec_step, step_transfer_cache):
        assert op_name == "op"
        assert result.status is ExecutionStatus.SUCCESS
        assert exec_step == 0
        return (asyncio.create_task(slow_outbound()),)

    stub._send_output = fake_send_output

    send_token = await stub._local_plan_send_output("op", 0, _prepared(), "payload")
    await stub._local_plan_await_send(send_token)
    await stub._local_plan_commit("op", 0, send_token)

    recorder = stub._profiler_runtime.recorder

    def durations(name: str) -> list[float]:
        name_id = recorder.intern(name)
        return [
            int(row[2]) / 1e6
            for row in recorder.arena.snapshot()
            if int(row[4]) == name_id
        ]

    send_ms = durations("local_step.send_output")
    await_ms = durations("local_step.await_send")
    commit_ms = durations("local_step.commit")

    assert len(send_ms) == 1
    assert len(await_ms) == 1
    assert len(commit_ms) == 1
    assert send_ms[0] < 20.0
    assert await_ms[0] < 20.0
    assert commit_ms[0] < 20.0
    await delivery_done.wait()
