from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast


import numpy as np
import pytest

import fluxway as fw
from fluxway.common.compiled_graph import (
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.profiler import (
    CausalClockViolation,
    MissingAxonNativeHlc,
    MissingTraceMetadata,
    ProfilerBufferExhausted,
    ProfilerConfig,
    ProfilerDiagnostic,
    ProfilerLevel,
    ProfilerRuntimeFactory,
)
from fluxway.profiler.clock import AxonNativeHlcBridge, RayFluxwayHlcBridge
from fluxway.profiler.hierarchy import (
    TRACE_ROOT_TRACK_NAME,
    TraceTrackHierarchy,
)
from fluxway.profiler.graph_symbols import GraphTraceSymbols
from fluxway.profiler.perfetto_writer import PerfettoTraceWriter
from fluxway.profiler.record import (
    PAYLOAD_START_COLUMN,
    RECORD_WIDTH,
    ProfilerSnapshot,
    TraceArena,
    TraceCarrier,
    TraceEventKind,
    TraceMetadata,
)
from fluxway.profiler.recorder import (
    LOCAL_STEP_WAIT_INPUTS,
    MAILBOX_DECLARE_STEP,
    MAILBOX_DEPOSIT_LOCAL_TENSOR,
    WORKER_QUEUE_ENQUEUE,
    WORKER_RUN_LOOP,
    TraceRecorder,
)
from fluxway.profiler.user_api import activate_user_profiler
from fluxway.profiler.wrappers import (
    TracingLocalStepPlanExecutor,
    TracingWorkerMailbox,
)
from fluxway.profiler.snapshots import merge_profiler_snapshots
from fluxway.profiler.filtering import filter_steady_state_records
from fluxway.profiler.filtering import StepFilter
from fluxway.control_plane.controller import ControllerActor
from fluxway.rpc.protocol import DataPlaneCallable, ReceiverIngressCompletion
from fluxway.rpc.protocol import TransportFrameEnvelope
from tests.compiled_graph_fixtures import compiled_spmd_spec
from fluxway.spmd.worker.mailbox import WorkerMailbox
from fluxway.spmd.worker.local_plan import LocalStepContext, LocalStepPlan
from fluxway.spmd.worker.runtime import WorkerProcessCallRuntime
from fluxway.spmd.worker.user_callable_executor import (
    USER_CALLABLE_DEFAULT_OPERATOR_LANE,
    SPMDObjectScope,
    UserCallableExecutor,
    UserCallableSpec,
)
from fluxway.transport.manifest import DependencyDecision, TransportFrameHeader


_TRACE_OPERATOR_SPEC = OperatorSpec(
    name="lookup",
    method_name="tests.Embedding.lookup",
    input_arguments={"data": ("ids", "weights"), "control": ("lr",)},
    output_arguments={"data": ("embeddings",)},
)
_TRACE_PLAN = compile_graph(
    name="profiler-test",
    spmds=(
        compiled_spmd_spec(
            spmd_id="embedding",
            mesh=DeviceMesh(("data",), (1,)),
            operators={"lookup": _TRACE_OPERATOR_SPEC},
        ),
    ),
    dependencies=(),
)
_TRACE_COMPILED_OPERATOR = _TRACE_PLAN.spmd("embedding").operator("lookup")
_TRACE_GRAPH_SYMBOLS = GraphTraceSymbols.from_compiled_graph(_TRACE_PLAN)


def test_disabled_factory_returns_canonical_objects():
    factory = ProfilerRuntimeFactory.from_config(
        None, process_name="worker/spmd/rank:0"
    )
    executor = object()
    handlers = {0: object()}
    mailbox = factory.create_worker_mailbox()

    assert factory.enabled is False
    assert factory.data_plane_backend_executor(executor) is executor
    assert factory.local_step_plan_executor(executor) is executor
    assert factory.user_callable_executor(executor) is executor
    assert type(mailbox) is WorkerMailbox
    assert factory.wrap_data_plane_handlers("remote", handlers) is handlers


def test_enabled_factory_creates_tracing_mailbox(tmp_path):
    config = ProfilerConfig(trace_dir=str(tmp_path), run_id="run")
    factory = ProfilerRuntimeFactory.from_config(
        config, process_name="worker/spmd/rank:0"
    )
    executor = object()
    handlers = {0: _RayLikeDataPlane()}
    mailbox = factory.create_worker_mailbox()

    assert factory.enabled is True
    assert factory.data_plane_backend_executor(executor) is not executor
    assert type(mailbox) is TracingWorkerMailbox
    assert isinstance(mailbox, WorkerMailbox)
    assert not hasattr(mailbox, "_delegate")
    wrapped_handlers = factory.wrap_data_plane_handlers("remote", handlers)
    assert wrapped_handlers is not handlers
    assert wrapped_handlers[0] is not handlers[0]


def test_tracing_mailbox_inherits_uninstrumented_worker_mailbox_contract():
    traced_methods = {
        "clear_step",
        "declare_step",
        "deposit_control_arguments",
        "deposit_local_tensor",
        "deposit_transport_frame",
        "set_ready_callback",
        "wait_step_ready",
    }
    base_public = {
        name
        for name, member in inspect.getmembers(WorkerMailbox, inspect.isfunction)
        if not name.startswith("_")
    }
    tracing_public = {
        name
        for name, member in inspect.getmembers(TracingWorkerMailbox, inspect.isfunction)
        if not name.startswith("_")
    }
    tracing_own_public = {
        name
        for name, member in TracingWorkerMailbox.__dict__.items()
        if inspect.isfunction(member) and not name.startswith("_")
    }

    assert tracing_public == base_public
    assert tracing_own_public == traced_methods
    assert all(
        getattr(TracingWorkerMailbox, name) is getattr(WorkerMailbox, name)
        for name in base_public - traced_methods
    )


@pytest.mark.asyncio
async def test_tracing_mailbox_preserves_dependency_decision_barrier(tmp_path):
    config = ProfilerConfig(trace_dir=str(tmp_path), run_id="run")
    factory = ProfilerRuntimeFactory.from_config(
        config, process_name="worker/spmd/rank:0"
    )
    mailbox = factory.create_worker_mailbox()
    ready: list[tuple[str, int]] = []
    mailbox.set_ready_callback(lambda op_name, step: ready.append((op_name, step)))

    mailbox.declare_step(
        "op",
        3,
        {11: 1},
        [],
        transfer_dependencies={11: 7},
        expected_dependency_decisions={7: 1},
    )
    await mailbox.deposit_dependency_decision(
        "op",
        7,
        3,
        sender_id="Input",
        rank=0,
        expected_count=1,
        decision=DependencyDecision.SKIP,
    )

    assert ready == [("op", 3)]
    assert mailbox.is_step_ready("op", 3)
    assert mailbox.is_step_skipped("op", 3)
    snapshot = factory.recorder.snapshot()
    recorded_names = {
        snapshot["metadata"].name_for(int(row[4])) for row in snapshot["records"]
    }
    assert {"mailbox.declare_step", "mailbox.ready_fire"} <= recorded_names


def test_mailbox_trace_schema_uses_transfer_identity():
    assert MAILBOX_DECLARE_STEP.payload_fields == (
        "op_id",
        "step",
        "transfer_count",
        "state_count",
    )
    assert MAILBOX_DEPOSIT_LOCAL_TENSOR.payload_fields == (
        "op_id",
        "transfer_code",
        "step",
        "data_count",
    )


def test_heartbeat_diagnostics_are_opt_in(tmp_path):
    default_runtime = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="default"),
        process_name="controller",
    )
    diagnostic_runtime = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(
            trace_dir=str(tmp_path),
            run_id="diagnostic",
            diagnostics=frozenset({ProfilerDiagnostic.HEARTBEAT}),
        ),
        process_name="controller",
    )

    assert default_runtime.heartbeat_observer() is None
    assert default_runtime.heartbeat_diagnostics_enabled() is False
    assert diagnostic_runtime.heartbeat_observer() is not None
    assert diagnostic_runtime.heartbeat_diagnostics_enabled() is True


@pytest.mark.asyncio
async def test_user_callable_execute_records_compiled_operator_identity(tmp_path):
    runtime = WorkerProcessCallRuntime(max_workers=1)
    executor = UserCallableExecutor(worker_process_runtime=runtime)
    factory = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="run"),
        process_name="worker/spmd/rank:0",
    )
    wrapped = factory.user_callable_executor(executor)

    async def fn() -> int:
        return 11

    result = await wrapped.execute(
        UserCallableSpec(
            callable_id="lookup",
            fn=fn,
            resource_id=1,
            operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
            operator_is_coroutine_function=True,
            operator_step=9,
            object_scope=SPMDObjectScope(spmd_id="trace", rank=0),
            graph_id=_TRACE_PLAN.graph_id,
            compiled_operator=_TRACE_COMPILED_OPERATOR,
        )
    )
    await runtime.drain()
    runtime.close()
    assert result == 11

    snapshot = factory.recorder.snapshot()
    metadata = snapshot["metadata"]
    execute_name_id = next(
        ident
        for ident, name in metadata.names_by_id.items()
        if name == "user_callable.execute"
    )
    row = next(row for row in snapshot["records"] if int(row[4]) == execute_name_id)
    fields = metadata.payload_fields_for(execute_name_id)
    payload = row[PAYLOAD_START_COLUMN : PAYLOAD_START_COLUMN + len(fields)]
    values = dict(zip(fields, payload, strict=True))

    assert len(fields) == 5
    assert metadata.name_for(int(values["graph_id"])) == _TRACE_PLAN.graph_id
    assert metadata.name_for(int(values["spmd_id"])) == "embedding"
    assert metadata.name_for(int(values["op_id"])) == "lookup"
    assert int(values["step"]) == 9
    assert metadata.name_for(int(values["execution_mode"])) == "coroutine_async"
    assert "operator_facts" not in UserCallableSpec.__dataclass_fields__


@pytest.mark.asyncio
async def test_user_callable_execute_requires_compiled_graph_identity(tmp_path):
    runtime = WorkerProcessCallRuntime(max_workers=1)
    executor = UserCallableExecutor(worker_process_runtime=runtime)
    factory = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="run"),
        process_name="worker/spmd/rank:0",
    )
    wrapped = factory.user_callable_executor(executor)

    async def fn() -> int:
        return 11

    with pytest.raises(ValueError, match="compiled_operator"):
        await wrapped.execute(
            UserCallableSpec(
                callable_id="lookup",
                fn=fn,
                resource_id=1,
                operator_lane=USER_CALLABLE_DEFAULT_OPERATOR_LANE,
                operator_is_coroutine_function=True,
                operator_step=9,
                object_scope=SPMDObjectScope(spmd_id="trace", rank=0),
            )
        )
    runtime.close()


@pytest.mark.asyncio
async def test_user_callable_wrapper_preserves_shutdown_contract(tmp_path):
    runtime = WorkerProcessCallRuntime(max_workers=1)
    executor = UserCallableExecutor(worker_process_runtime=runtime)
    factory = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(trace_dir=str(tmp_path), run_id="run"),
        process_name="worker/spmd/rank:0",
    )

    wrapped = factory.user_callable_executor(executor)
    await wrapped.shutdown()
    runtime.close()


def test_profiler_config_rejects_negative_skip_initial_steps(tmp_path):
    with pytest.raises(ValueError, match="skip_initial_steps"):
        ProfilerConfig(
            trace_dir=str(tmp_path),
            run_id="bad",
            skip_initial_steps=-1,
        )


def test_profiler_config_defaults_to_standard_level(tmp_path):
    config = ProfilerConfig(trace_dir=str(tmp_path), run_id="run")

    assert config.level is ProfilerLevel.STANDARD


@pytest.mark.asyncio
async def test_profiler_level_filters_internal_scheduler_events():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=16 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
        level=ProfilerLevel.STANDARD,
    )

    recorder.worker_scheduler_instant(spec=WORKER_RUN_LOOP)
    recorder.worker_scheduler_instant(
        spec=WORKER_QUEUE_ENQUEUE,
        payload=(recorder.intern("op"), 0, 0),
    )
    async with recorder.local_step_node(
        spec=LOCAL_STEP_WAIT_INPUTS,
        op_name="op",
        step=0,
    ):
        pass

    names = [
        recorder.metadata.name_for(int(row[4])) for row in recorder.arena.snapshot()
    ]
    assert "local_step.wait_inputs" in names
    assert "worker.run_loop" not in names
    assert "worker.queue_enqueue" not in names


@pytest.mark.asyncio
async def test_trace_recorder_skips_step_level_records_below_threshold():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
        step_filter=StepFilter(skip_initial_steps=2),
    )

    recorder.worker_scheduler_instant(spec=WORKER_RUN_LOOP)
    async with recorder.local_step_node(
        spec=LOCAL_STEP_WAIT_INPUTS,
        op_name="op",
        step=1,
    ):
        pass
    async with recorder.local_step_node(
        spec=LOCAL_STEP_WAIT_INPUTS,
        op_name="op",
        step=2,
    ):
        pass

    names = [
        recorder.metadata.name_for(int(row[4])) for row in recorder.arena.snapshot()
    ]
    assert names.count("worker.run_loop") == 1
    assert names.count("local_step.wait_inputs") == 1


def test_scheduler_loop_uses_responsibility_track():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=2 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )

    recorder.worker_scheduler_instant(spec=WORKER_RUN_LOOP)

    tracks = {
        recorder.metadata.name_for(int(row[3])) for row in recorder.arena.snapshot()
    }
    assert "worker/spmd/rank:0/scheduler/loop" in tracks


@pytest.mark.asyncio
async def test_checkpoint_storage_uses_responsibility_track():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=4 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )

    async with recorder.worker_checkpoint_save(step=3, operator_count=2):
        pass

    tracks = {
        recorder.metadata.name_for(int(row[3])) for row in recorder.arena.snapshot()
    }
    assert "worker/spmd/rank:0/checkpoint/storage" in tracks


@pytest.mark.asyncio
async def test_tracing_local_step_plan_requires_step_context():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=4 * RECORD_WIDTH * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )
    wrapped = TracingLocalStepPlanExecutor(object(), recorder)

    with pytest.raises(ValueError, match="requires a step context"):
        await wrapped.execute(object(), object())


@pytest.mark.asyncio
async def test_skipped_trace_span_preserves_wrapped_exception():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
        step_filter=StepFilter(skip_initial_steps=1),
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        async with recorder.local_step_node(
            spec=LOCAL_STEP_WAIT_INPUTS,
            op_name="op",
            step=0,
        ):
            raise RuntimeError("prepare failed")

    assert recorder.arena.snapshot().shape[0] == 0


def test_user_profiler_disabled_is_noop_without_attr_validation():
    assert fw.profiler.enabled() is False
    fw.profiler.event("outside", payload=object())
    with fw.profiler.scope("outside", payload=object()):
        pass


def test_user_profiler_scope_event_record_readable_payload():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )
    with activate_user_profiler(
        recorder=recorder,
        op_name="lookup",
        step=3,
        rank=2,
        execution_mode="worker_process",
    ):
        assert fw.profiler.enabled() is True
        with fw.profiler.scope("embedding_lookup", table="item", id_count=17):
            pass
        fw.profiler.event("cache_hit", hit=True)

    metadata = recorder.metadata
    rows = recorder.arena.snapshot()
    names = [metadata.name_for(int(row[4])) for row in rows]
    assert names == ["user.embedding_lookup", "user.cache_hit"]

    event_id = recorder.intern("user.embedding_lookup")
    fields = metadata.payload_fields_for(event_id)
    row = rows[0]
    payload = row[PAYLOAD_START_COLUMN : PAYLOAD_START_COLUMN + len(fields)]
    values = dict(zip(fields, payload, strict=True))
    assert metadata.name_for(int(values["op_id"])) == "lookup"
    assert int(values["step"]) == 3
    assert int(values["rank"]) == 2
    assert metadata.name_for(int(values["execution_mode"])) == "worker_process"
    assert metadata.name_for(int(values["attrs_id"])) == (
        '{"id_count":17,"table":"item"}'
    )


def test_user_profiler_enabled_rejects_invalid_inputs():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )
    with activate_user_profiler(
        recorder=recorder,
        op_name="lookup",
        step=0,
        rank=0,
        execution_mode="worker_process",
    ):
        with pytest.raises(ValueError, match="non-empty"):
            fw.profiler.event("")
        with pytest.raises(ValueError, match="reserved"):
            fw.profiler.event("bad", step=0)
        with pytest.raises(TypeError, match="str, int, float, bool, or None"):
            fw.profiler.event("bad", payload=object())


def test_user_profiler_skip_initial_steps_applies_to_custom_records():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
        step_filter=StepFilter(skip_initial_steps=2),
    )
    for step in (1, 2):
        with activate_user_profiler(
            recorder=recorder,
            op_name="lookup",
            step=step,
            rank=0,
            execution_mode="worker_process",
        ):
            fw.profiler.event("cache_hit", step_value=step)

    names = [
        recorder.metadata.name_for(int(row[4])) for row in recorder.arena.snapshot()
    ]
    assert names == ["user.cache_hit"]


@pytest.mark.asyncio
async def test_steady_state_filter_removes_lifecycle_records():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="driver/spmd/driver",
    )
    async with recorder.driver_initialize(worker_count=1):
        pass
    async with recorder.driver_drain(target_step=3, worker_count=1):
        pass

    filtered = filter_steady_state_records(recorder.arena.snapshot(), recorder.metadata)
    names = [recorder.metadata.name_for(int(row[4])) for row in filtered]

    assert names == ["driver.drain"]


@pytest.mark.asyncio
@pytest.mark.parametrize("steady_state_only", [False, True])
async def test_controller_export_includes_control_plane_snapshot(
    tmp_path, steady_state_only
):
    if importlib.util.find_spec("perfetto") is None:
        pytest.skip("perfetto is not installed")
    from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as proto

    config = ProfilerConfig(
        trace_dir=str(tmp_path),
        run_id="run",
        steady_state_only=steady_state_only,
    )
    controller_recorder = TraceRecorder(
        TraceArena(buffer_bytes=64 * 1024),
        run_id=config.run_id,
        process_name="controller",
    )
    worker_recorder = TraceRecorder(
        TraceArena(buffer_bytes=64 * 1024),
        run_id=config.run_id,
        process_name="worker/embedding/rank:0",
    )
    controller_class = cast(Any, ControllerActor).__ray_actor_class__
    controller = controller_class.__new__(controller_class)
    controller._profiler_config = config
    controller._profiler_runtime = ProfilerRuntimeFactory(controller_recorder)
    controller._graph = SimpleNamespace(require_compiled_graph=lambda: _TRACE_PLAN)

    async with controller_recorder.controller_initialize(spmd_count=1):
        pass
    async with worker_recorder.user_callable_execute(
        graph_id=_TRACE_PLAN.graph_id,
        compiled_operator=_TRACE_COMPILED_OPERATOR,
        step=0,
        execution_mode="coroutine_async",
    ):
        pass

    output = tmp_path / "trace.pftrace"
    await controller.write_profiler_trace(
        str(output), {"embedding": {0: worker_recorder.snapshot()}}
    )
    trace = proto.Trace.FromString(output.read_bytes())
    names = {
        packet.track_event.name
        for packet in trace.packet
        if packet.track_event.type == proto.TrackEvent.TYPE_SLICE_BEGIN
    }
    assert "controller.profiler_export" in names
    assert "user_callable.execute" in names
    assert ("controller.initialize" in names) is not steady_state_only


def test_trace_arena_fixed_capacity_exhaustion():
    arena = TraceArena(buffer_bytes=16 * np.dtype(np.int64).itemsize)
    arena.append(
        kind=TraceEventKind.INSTANT,
        ts_ns=1,
        dur_ns=0,
        track_id=1,
        name_id=1,
        payload=(7,),
    )
    with pytest.raises(ProfilerBufferExhausted):
        arena.append(
            kind=TraceEventKind.INSTANT,
            ts_ns=2,
            dur_ns=0,
            track_id=1,
            name_id=1,
        )


def test_profiler_buffer_exhausted_carries_structured_context():
    arena = TraceArena(buffer_bytes=16 * np.dtype(np.int64).itemsize)
    arena.append(
        kind=TraceEventKind.INSTANT,
        ts_ns=1,
        dur_ns=0,
        track_id=1,
        name_id=1,
    )

    with pytest.raises(ProfilerBufferExhausted) as exc_info:
        arena.append(
            kind=TraceEventKind.INSTANT,
            ts_ns=2,
            dur_ns=0,
            track_id=1,
            name_id=1,
        )

    error = exc_info.value
    assert error.code == "profiler_buffer_exhausted"
    assert error.context == {"capacity": 1, "attempted_index": 1}
    assert error.as_dict()["code"] == "profiler_buffer_exhausted"
    assert "capacity=1" in str(error)


def test_causal_clock_violation_carries_sender_receiver_hlc():
    error = CausalClockViolation(
        sender_hlc_physical_ns=10,
        sender_hlc_logical=2,
        receiver_hlc_physical_ns=10,
        receiver_hlc_logical=2,
    )

    assert error.code == "causal_clock_violation"
    assert error.context == {
        "sender_hlc_physical_ns": 10,
        "sender_hlc_logical": 2,
        "receiver_hlc_physical_ns": 10,
        "receiver_hlc_logical": 2,
    }


def test_ray_hlc_bridge_merges_sender_carrier():
    carrier = TraceCarrier.root("run").child("send")
    received = RayFluxwayHlcBridge().receive(carrier)

    assert received.receiver_hlc_physical_ns >= carrier.sender_hlc_physical_ns
    assert (
        received.receiver_hlc_physical_ns,
        received.receiver_hlc_logical,
    ) > (carrier.sender_hlc_physical_ns, carrier.sender_hlc_logical)


def test_axon_native_hlc_bridge_uses_native_metadata():
    carrier = TraceCarrier.root("run").child("send")
    result = _AxonResult(native_hlc_physical_ns=123, native_hlc_logical=4)
    received = AxonNativeHlcBridge().receive_result(carrier, result)

    assert received.receiver_hlc_physical_ns == 123
    assert received.receiver_hlc_logical == 4


def test_axon_native_hlc_bridge_rejects_missing_metadata():
    carrier = TraceCarrier.root("run").child("send")
    with pytest.raises(MissingAxonNativeHlc) as exc_info:
        AxonNativeHlcBridge().receive_result(carrier, object())
    assert exc_info.value.code == "missing_axon_native_hlc"
    assert exc_info.value.context == {
        "result_type": "object",
        "has_metadata": False,
    }


@pytest.mark.asyncio
async def test_tracing_data_plane_callable_delegates_and_records(tmp_path):
    config = ProfilerConfig(trace_dir=str(tmp_path), run_id="run")
    factory = ProfilerRuntimeFactory.from_config(
        config, process_name="worker/spmd/rank:0"
    )
    delegate = _RayLikeDataPlane()
    wrapped = factory.wrap_data_plane_handlers("remote", {3: delegate})[3]
    header = TransportFrameHeader(
        frame_i32=np.array([1, 4, 0, 0, 1, 0, 0, 0], dtype=np.int32),
        entry_i32=np.zeros((0, 11), dtype=np.int32),
        payload_i32=np.zeros((0, 8), dtype=np.int32),
        shape_i32=np.zeros((0,), dtype=np.int32),
        global_shape_i64=np.zeros((0,), dtype=np.int64),
        compact_i64=np.zeros((0,), dtype=np.int64),
    )

    result = await wrapped.receive_transport_frames(
        (TransportFrameEnvelope(header=header, payload_segments=(np.ones(2),)),)
    )

    assert result.result is True
    assert delegate.transport_calls == 1
    assert factory.recorder.arena.size >= 2


def test_recorder_metadata_contains_semantic_payload_schema():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=4 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )
    recorder.data_plane_hlc_receive(
        carrier=recorder.root_carrier,
        receiver_hlc_physical_ns=123,
        receiver_hlc_logical=4,
        clock_source=1,
    )

    metadata = recorder.metadata
    event_id = recorder.intern("data_plane.hlc_receive")

    assert metadata.name_for(event_id) == "data_plane.hlc_receive"
    assert metadata.payload_fields_for(event_id) == (
        "receiver_hlc_physical_ns",
        "receiver_hlc_logical",
        "clock_source",
    )
    with pytest.raises(MissingTraceMetadata):
        metadata.name_for(1)


def test_trace_hierarchy_models_fluxway_abstractions():
    worker = TraceTrackHierarchy.from_track_name(
        "worker/InputProcessor/rank:0/transport/send"
    )
    driver = TraceTrackHierarchy.from_track_name(
        "driver/InputProcessor/driver/checkpoint"
    )
    controller = TraceTrackHierarchy.from_track_name("controller/fsm/transition")
    scheduler = TraceTrackHierarchy.from_track_name(
        "worker/InputProcessor/rank:0/scheduler/queue"
    )
    checkpoint = TraceTrackHierarchy.from_track_name(
        "worker/InputProcessor/rank:0/checkpoint/storage"
    )

    assert worker.groups == (
        "SPMDs",
        "InputProcessor",
        "rank:0",
        "transport",
    )
    assert worker.leaf_name == "send"
    assert driver.groups == ("SPMDs", "InputProcessor", "driver")
    assert driver.leaf_name == "checkpoint"
    assert controller.groups == ("Controller", "fsm")
    assert controller.leaf_name == "transition"
    assert scheduler.groups == (
        "SPMDs",
        "InputProcessor",
        "rank:0",
        "scheduler",
    )
    assert scheduler.leaf_name == "queue"
    assert checkpoint.groups == (
        "SPMDs",
        "InputProcessor",
        "rank:0",
        "checkpoint",
    )
    assert checkpoint.leaf_name == "storage"
    assert worker.to_layout(leaf_uuid=7).nodes[0].name == TRACE_ROOT_TRACK_NAME
    with pytest.raises(ValueError):
        TraceTrackHierarchy.from_track_name("unknown/path")


def test_perfetto_writer_dependency_contract(tmp_path):
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=4 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )
    recorder.data_plane_hlc_receive(
        carrier=recorder.root_carrier,
        receiver_hlc_physical_ns=123,
        receiver_hlc_logical=4,
        clock_source=1,
    )
    if importlib.util.find_spec("perfetto") is None:
        with pytest.raises(ImportError, match="perfetto==0.57.1"):
            PerfettoTraceWriter()
        return

    path = Path(tmp_path) / "trace.pftrace"
    PerfettoTraceWriter().write(
        recorder.arena.snapshot(),
        path,
        metadata=recorder.metadata,
        graph_symbols=_TRACE_GRAPH_SYMBOLS,
    )
    assert path.exists()
    assert path.stat().st_size > 0


@pytest.mark.asyncio
async def test_perfetto_writer_decodes_interned_debug_annotations(tmp_path):
    if importlib.util.find_spec("perfetto") is None:
        pytest.skip("perfetto is not installed")
    # pyrefly: ignore [missing-import]
    from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as proto

    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/embedding/rank:0",
    )

    class Delegate:
        async def execute(self, plan, registry, step_context):
            return None

    wrapped = TracingLocalStepPlanExecutor(Delegate(), recorder)
    await wrapped.execute(
        LocalStepPlan.single_call("lookup"),
        object(),
        LocalStepContext(op_name="lookup", step=4),
    )
    async with recorder.user_callable_execute(
        graph_id=_TRACE_PLAN.graph_id,
        compiled_operator=_TRACE_COMPILED_OPERATOR,
        step=4,
        execution_mode="coroutine_async",
    ):
        pass

    path = Path(tmp_path) / "trace.pftrace"
    PerfettoTraceWriter().write(
        recorder.arena.snapshot(),
        path,
        metadata=recorder.metadata,
        graph_symbols=_TRACE_GRAPH_SYMBOLS,
    )

    trace = proto.Trace()
    trace.ParseFromString(path.read_bytes())
    annotations = {}
    for packet in trace.packet:
        event = packet.track_event
        if event.name != "user_callable.execute":
            continue
        for annotation in event.debug_annotations:
            if annotation.HasField("string_value"):
                annotations[annotation.name] = annotation.string_value
            elif annotation.HasField("int_value"):
                annotations[annotation.name] = int(annotation.int_value)
        break

    assert annotations["graph_id"] == _TRACE_PLAN.graph_id
    assert annotations["graph_generation"] == str(_TRACE_GRAPH_SYMBOLS.generation)
    assert annotations["spmd_id"] == "embedding"
    assert annotations["operator.function"] == "tests.Embedding.lookup"
    assert annotations["op_id"] == "lookup"
    assert annotations["step"] == 4
    assert annotations["execution_mode"] == "coroutine_async"
    assert "function_id" not in annotations
    assert "input_signature_id" not in annotations
    assert "output_signature_id" not in annotations

    local_step_annotations = {}
    for packet in trace.packet:
        event = packet.track_event
        if event.name != "local_step_plan.execute":
            continue
        for annotation in event.debug_annotations:
            if annotation.HasField("string_value"):
                local_step_annotations[annotation.name] = annotation.string_value
            elif annotation.HasField("int_value"):
                local_step_annotations[annotation.name] = int(annotation.int_value)
        break

    assert local_step_annotations["op_id"] == "lookup"
    assert local_step_annotations["step"] == 4
    assert local_step_annotations["node_count"] == 1
    assert local_step_annotations["operator.function"] == "tests.Embedding.lookup"
    assert not any(key.startswith("operator.input.") for key in local_step_annotations)
    assert not any(key.startswith("operator.output.") for key in local_step_annotations)


@pytest.mark.asyncio
async def test_perfetto_writer_assigns_overlapping_slices_to_lanes(
    tmp_path, monkeypatch
):
    if importlib.util.find_spec("perfetto") is None:
        pytest.skip("perfetto is not installed")
    # pyrefly: ignore [missing-import]
    from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as proto

    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * RECORD_WIDTH * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/embedding/rank:3",
    )
    timestamps = iter((100, 200, 300, 400))
    monkeypatch.setattr(
        "fluxway.profiler.recorder.time.perf_counter_ns",
        lambda: next(timestamps),
    )
    first = recorder.local_step_plan_execute(op_name="lookup", step=7, node_count=1)
    second = recorder.local_step_plan_execute(op_name="lookup", step=8, node_count=1)
    await first.__aenter__()
    await second.__aenter__()
    await first.__aexit__(None, None, None)
    await second.__aexit__(None, None, None)

    path = Path(tmp_path) / "overlapping.pftrace"
    PerfettoTraceWriter().write(
        recorder.arena.snapshot(),
        path,
        metadata=recorder.metadata,
        graph_symbols=_TRACE_GRAPH_SYMBOLS,
    )
    trace = proto.Trace()
    trace.ParseFromString(path.read_bytes())

    descriptors = {
        packet.track_descriptor.uuid: packet.track_descriptor
        for packet in trace.packet
        if packet.HasField("track_descriptor")
    }
    events_by_step = {7: [], 8: []}
    active_by_track = {}
    for packet in trace.packet:
        event = packet.track_event
        if event.type == proto.TrackEvent.TYPE_SLICE_BEGIN:
            if event.name != "local_step_plan.execute":
                continue
            step = next(
                int(annotation.int_value)
                for annotation in event.debug_annotations
                if annotation.name == "step"
            )
            active_by_track[int(event.track_uuid)] = step
        elif event.type == proto.TrackEvent.TYPE_SLICE_END:
            assert event.name == ""
            assert not event.debug_annotations
            step = active_by_track.pop(int(event.track_uuid))
        else:
            continue
        events_by_step[step].append(
            (int(event.type), int(packet.timestamp), int(event.track_uuid))
        )
    assert not active_by_track

    lane_uuids = set()
    for events in events_by_step.values():
        assert len(events) == 2
        begin, end = events
        assert begin[0] == int(proto.TrackEvent.TYPE_SLICE_BEGIN)
        assert end[0] == int(proto.TrackEvent.TYPE_SLICE_END)
        assert end[1] - begin[1] == 200
        assert begin[2] == end[2]
        lane_uuids.add(begin[2])

    assert len(lane_uuids) == 2
    for lane_uuid in lane_uuids:
        lane = descriptors[lane_uuid]
        assert lane.name in {"lane:0", "lane:1"}
        assert descriptors[lane.parent_uuid].name == "local_plan"


def test_perfetto_writer_expands_user_profiler_attrs(tmp_path):
    if importlib.util.find_spec("perfetto") is None:
        pytest.skip("perfetto is not installed")
    # pyrefly: ignore [missing-import]
    from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as proto

    recorder = TraceRecorder(
        TraceArena(buffer_bytes=8 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/embedding/rank:0",
    )
    with activate_user_profiler(
        recorder=recorder,
        op_name="lookup",
        step=4,
        rank=1,
        execution_mode="worker_process",
    ):
        with fw.profiler.scope(
            "embedding_lookup",
            table="item",
            id_count=19,
            hit=True,
            ratio=0.5,
            optional=None,
        ):
            pass

    path = Path(tmp_path) / "trace.pftrace"
    PerfettoTraceWriter().write(
        recorder.arena.snapshot(),
        path,
        metadata=recorder.metadata,
        graph_symbols=_TRACE_GRAPH_SYMBOLS,
    )

    trace = proto.Trace()
    trace.ParseFromString(path.read_bytes())
    annotations = {}
    for packet in trace.packet:
        event = packet.track_event
        if event.name != "user.embedding_lookup":
            continue
        for annotation in event.debug_annotations:
            if annotation.HasField("string_value"):
                annotations[annotation.name] = annotation.string_value
            elif annotation.HasField("int_value"):
                annotations[annotation.name] = int(annotation.int_value)
            elif annotation.HasField("bool_value"):
                annotations[annotation.name] = bool(annotation.bool_value)
            elif annotation.HasField("double_value"):
                annotations[annotation.name] = float(annotation.double_value)
        break

    assert annotations["op_id"] == "lookup"
    assert annotations["step"] == 4
    assert annotations["rank"] == 1
    assert annotations["execution_mode"] == "worker_process"
    assert annotations["table"] == "item"
    assert annotations["id_count"] == 19
    assert annotations["hit"] is True
    assert annotations["ratio"] == 0.5
    assert annotations["optional"] == "null"
    assert "attrs_id" not in annotations


def test_merge_profiler_snapshots_applies_offsets_and_rejects_metadata_conflicts():
    first = TraceRecorder(
        TraceArena(buffer_bytes=4 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )
    second = TraceRecorder(
        TraceArena(buffer_bytes=4 * 16 * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/spmd/rank:0",
    )
    first.worker_scheduler_instant(spec=WORKER_RUN_LOOP)
    second.worker_scheduler_instant(spec=WORKER_RUN_LOOP)
    first_snapshot = first.snapshot()
    second_snapshot = second.snapshot()
    second_snapshot["monotonic_to_epoch_offset_ns"] = 100

    merged = merge_profiler_snapshots(
        (first_snapshot, second_snapshot),
        process_name="worker/spmd/rank:0",
    )

    assert merged is not None
    assert merged["monotonic_to_epoch_offset_ns"] == 0
    assert len(merged["records"]) == 2
    assert int(merged["records"][1, 1]) == int(second_snapshot["records"][0, 1]) + 100

    conflict_a = cast(ProfilerSnapshot, dict(first_snapshot))
    conflict_b = cast(ProfilerSnapshot, dict(second_snapshot))
    conflict_a["metadata"] = TraceMetadata(
        names_by_id={1: "first"},
        payload_fields_by_name_id={},
    )
    conflict_b["metadata"] = TraceMetadata(
        names_by_id={1: "second"},
        payload_fields_by_name_id={},
    )
    with pytest.raises(ValueError, match="metadata name id conflict"):
        merge_profiler_snapshots(
            (conflict_a, conflict_b),
            process_name="worker/spmd/rank:0",
        )


@dataclass
class _AxonResult:
    native_hlc_physical_ns: int
    native_hlc_logical: int


class _RayLikeDataPlane(DataPlaneCallable):
    def __init__(self) -> None:
        self.transport_calls = 0
        self.control_calls = 0

    async def receive_transport_frames(self, frames):
        self.transport_calls += 1
        return ReceiverIngressCompletion(True)

    async def receive_control(self, op_name, step, names, values):
        self.control_calls += 1
        return True


@pytest.mark.asyncio
async def test_driver_checkpoint_spans_record_canonical_checkpoint_id():
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=64 * 1024),
        run_id="run",
        process_name="driver/spmd/driver",
    )
    async with recorder.driver_snapshot(
        checkpoint_id="chk_snap_test", target_step=42, worker_count=3
    ):
        pass
    async with recorder.driver_restore(
        checkpoint_id="chk_restore_test", worker_count=3
    ):
        pass

    rows = recorder.arena.snapshot()
    names = [recorder.metadata.name_for(int(row[4])) for row in rows]
    assert names == ["driver.snapshot", "driver.restore"]

    snap_row = rows[0]
    snap_fields = recorder.metadata.payload_fields_for(int(snap_row[4]))
    assert snap_fields == ("checkpoint_id", "target_step", "worker_count")
    assert (
        recorder.metadata.name_for(int(snap_row[PAYLOAD_START_COLUMN]))
        == "chk_snap_test"
    )
    assert int(snap_row[PAYLOAD_START_COLUMN + 1]) == 42
    assert int(snap_row[PAYLOAD_START_COLUMN + 2]) == 3

    restore_row = rows[1]
    restore_fields = recorder.metadata.payload_fields_for(int(restore_row[4]))
    assert restore_fields == ("checkpoint_id", "worker_count")
    assert (
        recorder.metadata.name_for(int(restore_row[PAYLOAD_START_COLUMN]))
        == "chk_restore_test"
    )
    assert int(restore_row[PAYLOAD_START_COLUMN + 1]) == 3
