import asyncio
import types
from typing import Any

import numpy as np
import pytest

from fluxway.common.decorators import operator, spmd
from fluxway.common.operator_instances import OperatorInstanceContainer
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.control_plane.graph.builder import Builder
from fluxway.profiler import ProfilerConfig, ProfilerRuntimeFactory
from fluxway.rpc.protocol import LocalWorkerDataPlane, RayActorDataPlane
from fluxway.resources import ResourceProfile
from fluxway.spmd.worker.data_plane import (
    CompiledDependencyCondition,
    CompiledDependencyDecisionRoute,
    CompiledSendPlan,
    CompiledSuccessorRoute,
)
from fluxway.spmd.worker.mailbox import WorkerMailbox
from fluxway.spmd.worker.data_plane_executor import _validate_local_only_batch
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionResult,
    ExecutionStatus,
)
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.spmd.worker.worker import SPMDWorker, _compile_transport_send_context
from fluxway.common.coherence import (
    CoherenceDirectory,
    CoherenceSubject,
    CoherenceInstance,
)
from fluxway.spmd.worker.coherence_sequencer import CoherenceSequencer
from fluxway.transport.communication import TransportFrameBatch
from fluxway.transport.manifest import DependencyDecision

_LOCAL_SPEC = {"type": "static", "axis": []}


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("dp",),
    mesh_axis_shapes=(2,),
)
class _LocalOnlyTransportSPMD:
    @operator(name="produce", output_sharding={"x": _LOCAL_SPEC})
    def produce(self) -> dict:
        return {"x": 1}

    @operator(name="consume", input_sharding={"x": _LOCAL_SPEC})
    def consume(self, x: Any):
        return x


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("dp",),
    mesh_axis_shapes=(1,),
)
class _LocalFanoutTransportSPMD:
    @operator(name="produce", output_sharding={"x": _LOCAL_SPEC})
    def produce(self) -> dict:
        return {"x": 1}

    @operator(name="consume_a", input_sharding={"x": _LOCAL_SPEC})
    def consume_a(self, x: Any):
        return x

    @operator(name="consume_b", input_sharding={"x": _LOCAL_SPEC})
    def consume_b(self, x: Any):
        return x


async def _receive_transport_frames(_frames):
    return ()


async def _receive_control(_op_name, _step, _names, _values):
    return None


def _local_only_config(*, condition_fn=None):
    builder = Builder()
    builder.add_node("A", _LocalOnlyTransportSPMD)
    builder.add_func_edge(
        "A",
        "produce",
        "A",
        "consume",
        source_output="x",
        target_input="x",
        local_only=True,
        condition_fn=condition_fn,
    )
    return builder.build("local_only_transport").nodes["A"].spmd_config


def _local_fanout_config():
    builder = Builder()
    builder.add_node("A", _LocalFanoutTransportSPMD)
    for consumer in ("consume_a", "consume_b"):
        builder.add_func_edge(
            "A",
            "produce",
            "A",
            consumer,
            source_output="x",
            target_input="x",
            local_only=True,
        )
    return builder.build("local_fanout_transport").nodes["A"].spmd_config


def _only_expected_transfer_code(op_manager, op_name):
    expected = op_manager.get_expected_transfer_shards(op_name)
    assert len(expected) == 1
    return next(iter(expected))


def _bind_direct_local_fast_path(stub, raw_cls):
    stub._background_tasks = set()
    stub._assemble_plan_cache = {}
    stub._direct_local_input_slots = {}
    stub._direct_local_ready_steps = {"produce": set(), "consume": set()}
    stub._on_step_ready = lambda _op_name, _step: None
    for name in (
        "_compiled_assemble_plan",
        "_direct_local_input_transfer_codes",
        "_uses_direct_local_input_fast_path",
        "_get_operator_mailbox",
        "_deposit_direct_local_input",
        "_deposit_local_only_orderings",
        "_deposit_local_only_dependency_decisions",
        "_prepare_local_only_outputs",
        "_prepare_local_only_outputs_impl",
        "_spawn_background_task",
        "_create_supervised_task",
        "_on_background_task_done",
    ):
        setattr(stub, name, types.MethodType(getattr(raw_cls, name), stub))
    stub._coherence_output_intents = {}

    async def admit(instance):
        stub._coherence_directory.admit(instance)

    stub._coherence_sequencer = CoherenceSequencer(
        lambda _instances, payload: payload(),
        admit,
    )
    send_context = stub._operator_send_context("produce")
    instances = tuple(
        sorted(
            {
                CoherenceInstance(
                    CoherenceSubject(
                        stub._local_spmd_code,
                        stub.rank,
                        route.logical_tuuid,
                    ),
                    0,
                )
                for successor in send_context.compiled_plan.successor_routes
                for route in successor.tensor_routes
            }
        )
    )
    for instance in instances:
        stub._coherence_sequencer.issue_intent(instance)
    stub._coherence_output_intents[("produce", 0)] = instances
    for name in (
        "_close_step_coherence_intents",
        "_abort_step_coherence_intents",
    ):
        setattr(stub, name, types.MethodType(getattr(raw_cls, name), stub))


def test_local_only_successor_uses_direct_worker_data_plane():
    raw_cls = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]
    config = _local_only_config()
    op_manager = OperatorManager(
        config,
        rank=0,
        spmd_id="A",
        static_args={},
        instance_container=OperatorInstanceContainer(),
    )
    stub = types.SimpleNamespace(
        spmd_id="A",
        rank=0,
        config=config,
        _graph_view=config.require_graph_view(),
        _world_size=config.num_workers,
        _mesh=config.mesh,
        _local_spmd_code=3,
        _successor_handlers={},
        _operator_send_context_cache={},
        _profiler_runtime=ProfilerRuntimeFactory.from_config(
            None,
            process_name="worker/test/rank:0",
        ),
        op_manager=op_manager,
        receive_transport_frames=_receive_transport_frames,
        receive_control=_receive_control,
        _coherence_directory=CoherenceDirectory(),
    )
    stub._require_local_spmd_code = types.MethodType(
        raw_cls._require_local_spmd_code,
        stub,
    )
    stub._coherence_subject = types.MethodType(raw_cls._coherence_subject, stub)
    stub._local_only_successor_handlers = types.MethodType(
        raw_cls._local_only_successor_handlers,
        stub,
    )
    stub._operator_send_context = types.MethodType(
        raw_cls._operator_send_context,
        stub,
    )

    send_context = stub._operator_send_context("produce")

    handler = send_context.successor_handlers[0][0]
    assert isinstance(handler, LocalWorkerDataPlane)
    assert not isinstance(handler, RayActorDataPlane)
    assert send_context.compiled_plan.successor_routes[0].local_only is True


def test_conditional_local_only_input_uses_mailbox_causal_barrier():
    config = _local_only_config(condition_fn=lambda _result: True)
    op_manager = OperatorManager(
        config,
        rank=0,
        spmd_id="A",
        static_args={},
        instance_container=OperatorInstanceContainer(),
    )
    raw_cls = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]
    stub = types.SimpleNamespace(
        spmd_id="A",
        rank=0,
        _world_size=config.num_workers,
        _op_manager=op_manager,
        op_manager=op_manager,
        _assemble_plan_cache={},
    )
    stub._compiled_assemble_plan = types.MethodType(
        raw_cls._compiled_assemble_plan,
        stub,
    )
    stub._direct_local_input_transfer_codes = types.MethodType(
        raw_cls._direct_local_input_transfer_codes,
        stub,
    )

    assert stub._direct_local_input_transfer_codes("consume") == set()


def test_local_only_batch_rejects_non_local_rank():
    route = CompiledSuccessorRoute(
        context_index=0,
        spmd_id="A",
        effective_rank=0,
        tensor_routes=(),
        local_only=True,
    )
    plan = CompiledSendPlan.create(
        op_name="produce",
        source_spmd_id="A",
        rank=0,
        sender_spmd_code=3,
        sender_world_size=2,
        successor_routes=(route,),
        output_count=1,
        transfer_code_to_name={},
    )
    batch = TransportFrameBatch(
        context_index=0,
        target_rank=1,
        batch_seq=0,
        frames=(),
        delegations=(),
    )

    with pytest.raises(ValueError, match="same-rank routing"):
        _validate_local_only_batch(route, plan, batch)


@pytest.mark.asyncio
async def test_local_only_direct_deposit_preserves_transport_tensor_receipt():
    raw_cls = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]
    config = _local_only_config()
    op_manager = OperatorManager(
        config,
        rank=1,
        spmd_id="A",
        static_args={},
        instance_container=OperatorInstanceContainer(),
    )
    mailbox = WorkerMailbox()
    stub = types.SimpleNamespace(
        spmd_id="A",
        rank=1,
        config=config,
        _graph_view=config.require_graph_view(),
        _world_size=config.num_workers,
        _mesh=config.mesh,
        _local_spmd_code=3,
        _successor_handlers={},
        _operator_send_context_cache={},
        _profiler_runtime=ProfilerRuntimeFactory.from_config(
            None,
            process_name="worker/test/rank:1",
        ),
        _mailbox=mailbox,
        _barriers_declared={},
        _op_manager=op_manager,
        op_manager=op_manager,
        receive_transport_frames=_receive_transport_frames,
        receive_control=_receive_control,
        _coherence_directory=CoherenceDirectory(),
    )
    stub._require_local_spmd_code = types.MethodType(
        raw_cls._require_local_spmd_code,
        stub,
    )
    stub._local_only_successor_handlers = types.MethodType(
        raw_cls._local_only_successor_handlers,
        stub,
    )
    stub._operator_send_context = types.MethodType(
        raw_cls._operator_send_context,
        stub,
    )
    stub._ensure_step_declared = types.MethodType(
        raw_cls._ensure_step_declared,
        stub,
    )
    stub._coherence_subject = types.MethodType(raw_cls._coherence_subject, stub)
    stub._deposit_local_only_outputs = types.MethodType(
        raw_cls._deposit_local_only_outputs,
        stub,
    )
    _bind_direct_local_fast_path(stub, raw_cls)

    send_context = stub._operator_send_context("produce")
    original = TransportTensor(
        np.asarray([1, 2, 3], dtype=np.int32),
        global_shape=(3,),
        receipt=SymmetryReceipt(7, 4),
    )
    await stub._deposit_local_only_outputs(
        [original],
        send_context,
        {},
        0,
        dependency_decisions=None,
    )

    transfer_code = _only_expected_transfer_code(op_manager, "consume")
    stored = stub._direct_local_input_slots["consume"][0][transfer_code].tensor
    assert isinstance(stored, TransportTensor)
    assert stored.receipt is not None
    assert stored.receipt.world_size == 7
    assert int(stored.receipt.sender_rank_tags) == 4
    stored.acquire_consumer_lease().release()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_local_only_deposit_records_worker_local_delivery_span(tmp_path):
    raw_cls = SPMDWorker.__ray_actor_class__  # pyrefly: ignore[missing-attribute]
    config = _local_only_config()
    op_manager = OperatorManager(
        config,
        rank=0,
        spmd_id="A",
        static_args={},
        instance_container=OperatorInstanceContainer(),
    )
    mailbox = WorkerMailbox()
    profiler_config = ProfilerConfig(
        trace_dir=str(tmp_path),
        run_id="test_run",
        buffer_bytes_per_process=1024 * 1024,
    )
    profiler_runtime = ProfilerRuntimeFactory.from_config(
        profiler_config,
        process_name="worker/A/rank:0/worker",
    )
    stub = types.SimpleNamespace(
        spmd_id="A",
        rank=0,
        config=config,
        _graph_view=config.require_graph_view(),
        _world_size=config.num_workers,
        _mesh=config.mesh,
        _local_spmd_code=3,
        _successor_handlers={},
        _operator_send_context_cache={},
        _profiler_runtime=profiler_runtime,
        _mailbox=mailbox,
        _barriers_declared={},
        _op_manager=op_manager,
        op_manager=op_manager,
        receive_transport_frames=_receive_transport_frames,
        receive_control=_receive_control,
        _coherence_directory=CoherenceDirectory(),
    )
    stub._require_local_spmd_code = types.MethodType(
        raw_cls._require_local_spmd_code,
        stub,
    )
    stub._local_only_successor_handlers = types.MethodType(
        raw_cls._local_only_successor_handlers,
        stub,
    )
    stub._operator_send_context = types.MethodType(
        raw_cls._operator_send_context,
        stub,
    )
    stub._ensure_step_declared = types.MethodType(
        raw_cls._ensure_step_declared,
        stub,
    )
    stub._coherence_subject = types.MethodType(raw_cls._coherence_subject, stub)
    stub._deposit_local_only_outputs = types.MethodType(
        raw_cls._deposit_local_only_outputs,
        stub,
    )
    _bind_direct_local_fast_path(stub, raw_cls)

    send_context = stub._operator_send_context("produce")
    tensor0 = TransportTensor(np.asarray([1, 2, 3], dtype=np.int32))

    await stub._deposit_local_only_outputs(
        [tensor0],
        send_context,
        {},
        0,
        dependency_decisions=None,
    )

    transfer_code = _only_expected_transfer_code(op_manager, "consume")
    stored0 = stub._direct_local_input_slots["consume"][0][transfer_code].tensor
    lease0 = stored0.acquire_consumer_lease()

    inst1 = CoherenceInstance(CoherenceSubject(stub._local_spmd_code, stub.rank, 1), 1)
    stub._coherence_sequencer.issue_intent(inst1)
    stub._coherence_output_intents[("produce", 1)] = (inst1,)
    tensor1 = TransportTensor(np.asarray([4, 5, 6], dtype=np.int32))

    task1 = asyncio.create_task(
        stub._deposit_local_only_outputs(
            [tensor1],
            send_context,
            {},
            1,
            dependency_decisions=None,
        )
    )
    await task1
    assert task1.done()

    lease0.release()

    stored1 = stub._direct_local_input_slots["consume"][1][transfer_code].tensor
    stored1.acquire_consumer_lease().release()

    snap = profiler_runtime.recorder.snapshot()
    records = snap["records"]
    meta = snap["metadata"]
    delivery_records = [
        r for r in records if meta.name_for(int(r[4])) == "worker.local_delivery"
    ]
    assert len(delivery_records) == 2

    p0 = delivery_records[0][8:14]
    status0 = meta.name_for(int(p0[5]))
    assert p0[1] == transfer_code
    assert p0[3] == 0
    assert p0[4] == 0
    assert status0 == "ready"

    p1 = delivery_records[1][8:14]
    status1 = meta.name_for(int(p1[5]))
    assert p1[1] == transfer_code
    assert p1[3] == 1
    assert p1[4] == 1
    assert status1 == "consumer_lease_wait"


@pytest.mark.asyncio
async def test_local_only_fanout_uses_distinct_zero_copy_aliases():
    raw_cls = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]
    config = _local_fanout_config()
    op_manager = OperatorManager(
        config,
        rank=0,
        spmd_id="A",
        static_args={},
        instance_container=OperatorInstanceContainer(),
    )
    mailbox = WorkerMailbox()
    stub = types.SimpleNamespace(
        spmd_id="A",
        rank=0,
        config=config,
        _graph_view=config.require_graph_view(),
        _world_size=config.num_workers,
        _mesh=config.mesh,
        _local_spmd_code=3,
        _successor_handlers={},
        _operator_send_context_cache={},
        _profiler_runtime=ProfilerRuntimeFactory.from_config(
            None,
            process_name="worker/test/fanout/rank:0",
        ),
        _mailbox=mailbox,
        _barriers_declared={},
        _op_manager=op_manager,
        op_manager=op_manager,
        receive_transport_frames=_receive_transport_frames,
        receive_control=_receive_control,
        _coherence_directory=CoherenceDirectory(),
    )
    for name in (
        "_require_local_spmd_code",
        "_local_only_successor_handlers",
        "_operator_send_context",
        "_ensure_step_declared",
        "_deposit_local_only_outputs",
        "_prepare_local_only_outputs",
        "_prepare_local_only_outputs_impl",
        "_deposit_direct_local_input",
    ):
        setattr(stub, name, types.MethodType(getattr(raw_cls, name), stub))
    stub._coherence_subject = types.MethodType(raw_cls._coherence_subject, stub)
    _bind_direct_local_fast_path(stub, raw_cls)

    original = TransportTensor(np.arange(4, dtype=np.float32), source_step=0)
    send_context = stub._operator_send_context("produce")
    await stub._deposit_local_only_outputs(
        [original], send_context, {}, 0, dependency_decisions=None
    )

    transfer_codes = {
        consumer: _only_expected_transfer_code(op_manager, consumer)
        for consumer in ("consume_a", "consume_b")
    }
    first = stub._direct_local_input_slots["consume_a"][0][
        transfer_codes["consume_a"]
    ].tensor
    second = stub._direct_local_input_slots["consume_b"][0][
        transfer_codes["consume_b"]
    ].tensor
    assert first is not second
    assert (
        first.flat_values.__array_interface__["data"][0]
        == second.flat_values.__array_interface__["data"][0]
        == original.flat_values.__array_interface__["data"][0]
    )
    first_lease = first.acquire_consumer_lease()
    second_lease = second.acquire_consumer_lease()
    first_lease.release()
    second_lease.release()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_send_output_all_local_only_returns_no_background_tasks_with_profiler():
    raw_cls = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]
    config = _local_only_config()
    op_manager = OperatorManager(
        config,
        rank=1,
        spmd_id="A",
        static_args={},
        instance_container=OperatorInstanceContainer(),
    )
    profiler_runtime = ProfilerRuntimeFactory.from_config(
        ProfilerConfig(
            trace_dir="/tmp/fluxway_test_send_output_all_local_only",
            run_id="profiler_enabled",
            buffer_bytes_per_process=1024 * 1024,
        ),
        process_name="worker/test/rank:1",
    )
    stub = types.SimpleNamespace(
        spmd_id="A",
        rank=1,
        config=config,
        _graph_view=config.require_graph_view(),
        _world_size=config.num_workers,
        _mesh=config.mesh,
        _local_spmd_code=3,
        _successor_handlers={},
        _operator_send_context_cache={},
        _profiler_runtime=profiler_runtime,
        _mailbox=profiler_runtime.create_worker_mailbox(),
        _barriers_declared={},
        _op_manager=op_manager,
        op_manager=op_manager,
        _data_plane_batch_max_frames=0,
        _data_plane_batch_flush_us=0,
        _data_plane_executor=None,
        _delivery_runtime=types.SimpleNamespace(submit=None),
        _driver_handle=None,
        receive_transport_frames=_receive_transport_frames,
        receive_control=_receive_control,
        _coherence_directory=CoherenceDirectory(),
    )
    for name in (
        "_require_local_spmd_code",
        "_local_only_successor_handlers",
        "_operator_send_context",
        "_send_output",
        "_send_step_data_plane_frames_background",
        "_deposit_local_only_outputs",
        "_prepare_local_only_outputs",
        "_prepare_local_only_outputs_impl",
        "_ensure_step_declared",
        "_send_control_arguments",
    ):
        setattr(stub, name, types.MethodType(getattr(raw_cls, name), stub))
    stub._coherence_subject = types.MethodType(raw_cls._coherence_subject, stub)
    _bind_direct_local_fast_path(stub, raw_cls)

    result = ExecutionResult(
        status=ExecutionStatus.SUCCESS,
        data={"x": TransportTensor.from_tensor(np.asarray([1, 2, 3], dtype=np.int32))},
    )

    send_tasks = await stub._send_output(
        "produce",
        result,
        exec_step=0,
        step_transfer_cache={},
    )

    transfer_code = _only_expected_transfer_code(op_manager, "consume")
    stored = stub._direct_local_input_slots["consume"][0][transfer_code].tensor
    assert send_tasks == ()
    assert isinstance(stored, TransportTensor)
    stored.acquire_consumer_lease().release()
    await asyncio.sleep(0)


def test_transport_send_context_is_compiled_once_for_mixed_local_routes():
    def condition(_result):
        return True

    routes = (
        CompiledSuccessorRoute(
            context_index=0,
            spmd_id="source",
            effective_rank=0,
            tensor_routes=(),
            dependency_decision_routes=(
                CompiledDependencyDecisionRoute(11, "local", (0,), 1),
            ),
            local_only=True,
        ),
        CompiledSuccessorRoute(
            context_index=1,
            spmd_id="target",
            effective_rank=0,
            tensor_routes=(),
            dependency_decision_routes=(
                CompiledDependencyDecisionRoute(22, "remote", (0,), 1),
            ),
        ),
    )
    plan = CompiledSendPlan.create(
        op_name="source",
        source_spmd_id="source",
        rank=0,
        sender_spmd_code=1,
        sender_world_size=1,
        successor_routes=routes,
        output_count=0,
        transfer_code_to_name={},
        dependency_conditions=(
            CompiledDependencyCondition(11, condition),
            CompiledDependencyCondition(22, condition),
        ),
    )
    local_handlers = {0: object()}
    remote_handlers = {0: object()}

    transport = _compile_transport_send_context(
        plan,
        (local_handlers, remote_handlers),
    )

    assert transport is not None
    assert transport.compiled_plan is not plan
    assert transport.successor_handlers == (remote_handlers,)
    assert tuple(
        route.context_index for route in transport.compiled_plan.successor_routes
    ) == (0,)
    assert tuple(
        item.dependency_code for item in transport.compiled_plan.dependency_conditions
    ) == (22,)
    assert transport.select_dependency_decisions(
        {
            11: DependencyDecision.PROCEED,
            22: DependencyDecision.SKIP,
        }
    ) == {22: DependencyDecision.SKIP}


def test_transport_send_context_reuses_all_remote_plan():
    route = CompiledSuccessorRoute(
        context_index=0,
        spmd_id="target",
        effective_rank=0,
        tensor_routes=(),
    )
    plan = CompiledSendPlan.create(
        op_name="source",
        source_spmd_id="source",
        rank=0,
        sender_spmd_code=1,
        sender_world_size=1,
        successor_routes=(route,),
        output_count=0,
        transfer_code_to_name={},
    )
    handlers = ({0: object()},)

    transport = _compile_transport_send_context(plan, handlers)

    assert transport is not None
    assert transport.compiled_plan is plan
    assert transport.successor_handlers is handlers
    assert transport.select_dependency_decisions(None) is None
