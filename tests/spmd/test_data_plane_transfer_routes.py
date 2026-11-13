import asyncio
from unittest.mock import patch

import numpy as np
import pytest

from fluxway.common.compiled_graph import (
    CompiledDependency,
    DependencySpec,
    GraphCompiler,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
    compile_graph,
)
from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.operator_instances import OperatorInstanceContainer
from fluxway.common.types.configs import SPMDConfig
from fluxway.common.types.sharding import P, PartitionSpec, ShardingSpec
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.spmd.worker.data_plane import (
    SuccessorRouteContext,
    build_completed_without_tensor_step_frame_groups,
    build_step_frame_groups,
    compile_send_plan,
    prepare_route_tensor,
)
from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.spmd.worker.operator_manager.operator_manager import OperatorManager
from fluxway.transport.communication import TransportRoutingMode
from fluxway.transport.manifest import ENTRY_EXPECTED_COUNT, ENTRY_ROUTE_CODE
from tests.compiled_graph_fixtures import compiled_spmd_spec


def _dependency(
    source_op: str,
    target_op: str,
    source_port: str,
    target_port: str,
) -> DependencySpec:
    return DependencySpec(
        source=OperatorKey("source", source_op),
        target=OperatorKey("target", target_op),
        transfers=(
            TensorTransfer(
                source=OperatorPort(source_port),
                target=OperatorPort(target_port),
            ),
        ),
    )


class _HeterogeneousSourceOwner:
    def produce(self) -> dict[str, object]:
        raise RuntimeError("transport route test operators are not executable")


class _HeterogeneousTargetOwner:
    def consume(self, dense: object, scalar: object) -> None:
        del dense, scalar


def _compiled_sender_plan(graph, *, source_spmd_id: str, source_op_id: str, rank: int):
    source_view = graph.spmd(source_spmd_id)
    dependencies = tuple(
        dependency
        for dependency in source_view.outgoing
        if dependency.source_operator.key.op_id == source_op_id
    )
    return compile_send_plan(
        graph_view=source_view,
        source_op_id=source_op_id,
        rank=rank,
        successor_contexts=(
            SuccessorRouteContext(0, dependencies, effective_rank=rank),
        ),
        sender_spmd_code=1,
        interleave_unit_size=None,
    )


def _build_single_frame(
    *,
    compiled_plan,
    flat_results: list[TransportTensor],
    exec_step: int,
):
    groups = build_step_frame_groups(
        compiled_plan=compiled_plan,
        flat_results=flat_results,
        step_transfer_cache={},
        exec_step=exec_step,
    )
    assert len(groups) == 1
    _context_index, frame_arg = groups[0]
    return frame_arg


def _frame_expected_counts(frame_arg) -> dict[int, int]:
    _target_rank, header, payloads = frame_arg
    frame = header.to_frame(payloads)
    return {
        int(entry[ENTRY_ROUTE_CODE]): int(entry[ENTRY_EXPECTED_COUNT])
        for entry in frame.entry_i32
    }


def _target_manager(graph, *, target_spmd_id: str) -> OperatorManager:
    graph_view = graph.spmd(target_spmd_id)
    return OperatorManager(
        SPMDConfig(
            spmd_id=target_spmd_id,
            graph_view=graph_view,
            mesh=graph_view.mesh,
            operator_specs={
                operator.key.op_id: operator.spec for operator in graph_view.operators
            },
        ),
        rank=0,
        spmd_id=target_spmd_id,
        static_args={},
        instance_container=OperatorInstanceContainer(),
    )


def _transfer_codes_by_target_port(graph, *, target_spmd_id: str) -> dict[str, int]:
    return {
        transfer.spec.target.name: transfer.transfer_code
        for dependency in graph.spmd(target_spmd_id).incoming
        for transfer in dependency.transfers
    }


def test_same_source_operator_heterogeneous_outputs_preserve_route_abi():
    chunked = ShardingSpec.chunk(P("data"))
    replicated = ShardingSpec.chunk(PartitionSpec())
    source = OperatorSpec(
        name="produce",
        owner=_HeterogeneousSourceOwner,
        output_arguments={"data": ("dense", "scalar")},
        output_sharding={"dense": chunked, "scalar": replicated},
    )
    target = OperatorSpec(
        name="consume",
        owner=_HeterogeneousTargetOwner,
        input_arguments={"data": ("dense", "scalar")},
        input_sharding={"dense": chunked, "scalar": replicated},
    )
    graph = compile_graph(
        name="same-source-heterogeneous-outputs",
        spmds=(
            compiled_spmd_spec(
                "source", DeviceMesh(("data",), (2,)), {"produce": source}
            ),
            compiled_spmd_spec(
                "target", DeviceMesh(("data",), (1,)), {"consume": target}
            ),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "produce"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(OperatorPort("dense"), OperatorPort("dense")),
                    TensorTransfer(OperatorPort("scalar"), OperatorPort("scalar")),
                ),
            ),
        ),
    )
    transfer_codes = _transfer_codes_by_target_port(graph, target_spmd_id="target")
    expected = _target_manager(
        graph, target_spmd_id="target"
    ).get_expected_transfer_shards("consume")
    assert expected == {
        transfer_codes["dense"]: 2,
        transfer_codes["scalar"]: 1,
    }

    rank_zero_plan = _compiled_sender_plan(
        graph, source_spmd_id="source", source_op_id="produce", rank=0
    )
    rank_one_plan = _compiled_sender_plan(
        graph, source_spmd_id="source", source_op_id="produce", rank=1
    )
    rank_zero_frame = _build_single_frame(
        compiled_plan=rank_zero_plan,
        flat_results=[
            TransportTensor(np.array([1, 2]), global_shape=(4,), source_step=0),
            TransportTensor(np.asarray(7), global_shape=(), source_step=0),
        ],
        exec_step=0,
    )
    rank_one_frame = _build_single_frame(
        compiled_plan=rank_one_plan,
        flat_results=[
            TransportTensor(np.array([3, 4]), global_shape=(4,), source_step=0),
            TransportTensor(np.asarray(8), global_shape=(), source_step=0),
        ],
        exec_step=0,
    )

    assert _frame_expected_counts(rank_zero_frame) == expected
    assert _frame_expected_counts(rank_one_frame) == {transfer_codes["dense"]: 2}
    warm_frame = _build_single_frame(
        compiled_plan=rank_zero_plan,
        flat_results=[
            TransportTensor(np.array([1, 2]), global_shape=(4,), source_step=1),
            TransportTensor(np.asarray(7), global_shape=(), source_step=1),
        ],
        exec_step=1,
    )
    assert _frame_expected_counts(warm_frame) == expected
    assert len(rank_zero_plan.frame_template_cache) == 1

    mailbox = WorkerMailbox()
    mailbox.declare_step("consume", 0, expected, [])
    for sender_id, frame_arg in (
        ("source", rank_zero_frame),
        ("source", rank_one_frame),
    ):
        _target_rank, header, payloads = frame_arg
        asyncio.run(
            mailbox.deposit_transport_frame(
                "consume", sender_id, header.to_frame(payloads)
            )
        )
    assert mailbox.is_step_ready("consume", 0)
    assert (
        len(
            asyncio.run(
                mailbox.get_tensor_shard_views(
                    "consume", transfer_codes["dense"], step=0
                )
            )
        )
        == 2
    )
    assert (
        len(
            asyncio.run(
                mailbox.get_tensor_shard_views(
                    "consume", transfer_codes["scalar"], step=0
                )
            )
        )
        == 1
    )


def test_same_target_operator_heterogeneous_inputs_preserve_route_abi():
    chunked = ShardingSpec.chunk(P("data"))
    replicated = ShardingSpec.chunk(PartitionSpec())
    dense_source = OperatorSpec(
        name="produce_dense",
        owner=_HeterogeneousSourceOwner,
        output_arguments={"data": ("dense",)},
        output_sharding={"dense": chunked},
    )
    scalar_source = OperatorSpec(
        name="produce_scalar",
        owner=_HeterogeneousSourceOwner,
        output_arguments={"data": ("scalar",)},
        output_sharding={"scalar": replicated},
    )
    target = OperatorSpec(
        name="consume",
        owner=_HeterogeneousTargetOwner,
        input_arguments={"data": ("dense", "scalar")},
        input_sharding={"dense": chunked, "scalar": replicated},
    )
    graph = compile_graph(
        name="same-target-heterogeneous-inputs",
        spmds=(
            compiled_spmd_spec(
                "dense_source",
                DeviceMesh(("data",), (2,)),
                {"produce_dense": dense_source},
            ),
            compiled_spmd_spec(
                "scalar_source",
                DeviceMesh(("data",), (2,)),
                {"produce_scalar": scalar_source},
            ),
            compiled_spmd_spec(
                "target", DeviceMesh(("data",), (1,)), {"consume": target}
            ),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("dense_source", "produce_dense"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(OperatorPort("dense"), OperatorPort("dense")),
                ),
            ),
            DependencySpec(
                source=OperatorKey("scalar_source", "produce_scalar"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(OperatorPort("scalar"), OperatorPort("scalar")),
                ),
            ),
        ),
    )
    transfer_codes = _transfer_codes_by_target_port(graph, target_spmd_id="target")
    expected = _target_manager(
        graph, target_spmd_id="target"
    ).get_expected_transfer_shards("consume")
    assert expected == {
        transfer_codes["dense"]: 2,
        transfer_codes["scalar"]: 1,
    }

    dense_rank_zero = _build_single_frame(
        compiled_plan=_compiled_sender_plan(
            graph,
            source_spmd_id="dense_source",
            source_op_id="produce_dense",
            rank=0,
        ),
        flat_results=[
            TransportTensor(np.array([1, 2]), global_shape=(4,), source_step=0)
        ],
        exec_step=0,
    )
    dense_rank_one = _build_single_frame(
        compiled_plan=_compiled_sender_plan(
            graph,
            source_spmd_id="dense_source",
            source_op_id="produce_dense",
            rank=1,
        ),
        flat_results=[
            TransportTensor(np.array([3, 4]), global_shape=(4,), source_step=0)
        ],
        exec_step=0,
    )
    scalar_rank_zero = _build_single_frame(
        compiled_plan=_compiled_sender_plan(
            graph,
            source_spmd_id="scalar_source",
            source_op_id="produce_scalar",
            rank=0,
        ),
        flat_results=[TransportTensor(np.asarray(7), global_shape=(), source_step=0)],
        exec_step=0,
    )
    scalar_rank_one_plan = _compiled_sender_plan(
        graph,
        source_spmd_id="scalar_source",
        source_op_id="produce_scalar",
        rank=1,
    )
    assert (
        build_step_frame_groups(
            compiled_plan=scalar_rank_one_plan,
            flat_results=[
                TransportTensor(np.asarray(8), global_shape=(), source_step=0)
            ],
            step_transfer_cache={},
            exec_step=0,
        )
        == []
    )

    assert _frame_expected_counts(dense_rank_zero) == {transfer_codes["dense"]: 2}
    assert _frame_expected_counts(dense_rank_one) == {transfer_codes["dense"]: 2}
    assert _frame_expected_counts(scalar_rank_zero) == {transfer_codes["scalar"]: 1}

    mailbox = WorkerMailbox()
    mailbox.declare_step("consume", 0, expected, [])
    for sender_id, frame_arg in (
        ("dense_source", dense_rank_zero),
        ("dense_source", dense_rank_one),
        ("scalar_source", scalar_rank_zero),
    ):
        _target_rank, header, payloads = frame_arg
        asyncio.run(
            mailbox.deposit_transport_frame(
                "consume", sender_id, header.to_frame(payloads)
            )
        )
    assert mailbox.is_step_ready("consume", 0)
    assert (
        len(
            asyncio.run(
                mailbox.get_tensor_shard_views(
                    "consume", transfer_codes["dense"], step=0
                )
            )
        )
        == 2
    )
    assert (
        len(
            asyncio.run(
                mailbox.get_tensor_shard_views(
                    "consume", transfer_codes["scalar"], step=0
                )
            )
        )
        == 1
    )


def test_sender_plan_uses_append_only_graph_view_generation():
    replicated = ShardingSpec.chunk(PartitionSpec())
    produce = OperatorSpec(
        name="produce",
        output_arguments={"data": ("value",)},
        output_sharding={"value": replicated},
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("value",)},
        input_sharding={"value": replicated},
    )
    source_mesh = DeviceMesh(("data",), (1,))
    target_mesh = DeviceMesh(("data",), (1,))
    compiler = GraphCompiler(graph_id="eager-append-test")
    initial = compiler.compile(
        name="eager-append",
        spmds=(
            compiled_spmd_spec("source", source_mesh, {"produce": produce}),
            compiled_spmd_spec("target", target_mesh, {}),
        ),
        dependencies=(),
    )
    dependency = _dependency("produce", "consume", "value", "value")
    extended = compiler.compile(
        name="eager-append",
        spmds=(
            compiled_spmd_spec("source", source_mesh, {"produce": produce}),
            compiled_spmd_spec("target", target_mesh, {"consume": consume}),
        ),
        dependencies=(dependency,),
    )
    source_view = extended.spmd("source")
    send_plan = compile_send_plan(
        graph_view=source_view,
        source_op_id="produce",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(0, source_view.outgoing, effective_rank=0),
        ),
        sender_spmd_code=1,
        interleave_unit_size=None,
    )

    assert initial.generation == 0
    assert extended.generation == 1
    assert initial.spmd("source").operator("produce").output_uuids == (1,)
    assert source_view.operator("produce").output_uuids == (1,)
    assert send_plan.successor_routes[0].tensor_routes[0].logical_tuuid == 1


def _compile_fanout_plan(
    *,
    second_target_sharding: ShardingSpec,
):
    replicated = ShardingSpec.chunk(PartitionSpec())
    source = OperatorSpec(
        name="produce",
        output_arguments={"data": ("value",)},
        output_sharding={"value": replicated},
    )
    target_a = OperatorSpec(
        name="consume_a",
        input_arguments={"data": ("left",)},
        input_sharding={"left": replicated},
    )
    target_b = OperatorSpec(
        name="consume_b",
        input_arguments={"data": ("right",)},
        input_sharding={"right": second_target_sharding},
    )
    source_mesh = DeviceMesh(("data",), (1,))
    target_mesh = DeviceMesh(("data",), (1,))
    plan = compile_graph(
        name="sender-transfer-routes",
        spmds=(
            compiled_spmd_spec("source", source_mesh, {"produce": source}),
            compiled_spmd_spec(
                "target",
                target_mesh,
                {"consume_a": target_a, "consume_b": target_b},
            ),
        ),
        dependencies=(
            _dependency("produce", "consume_a", "value", "left"),
            _dependency("produce", "consume_b", "value", "right"),
        ),
    )
    source_plan = plan.spmd("source")
    compiled = compile_send_plan(
        graph_view=source_plan,
        source_op_id="produce",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(
                context_index=0,
                dependencies=source_plan.outgoing,
                effective_rank=0,
            ),
        ),
        sender_spmd_code=3,
        interleave_unit_size=None,
    )
    return compiled, source_plan.outgoing


def test_compile_send_plan_preserves_each_canonical_transfer_route():
    partitioned = ShardingSpec.chunk(PartitionSpec("data"))
    compiled, dependencies = _compile_fanout_plan(second_target_sharding=partitioned)

    routes = compiled.successor_routes[0].tensor_routes
    expected_transfers = tuple(
        transfer for dependency in dependencies for transfer in dependency.transfers
    )
    assert [route.transfer_code for route in routes] == [
        transfer.transfer_code for transfer in expected_transfers
    ]
    assert [route.dependency_code for route in routes] == [
        dependency.code for dependency in dependencies
    ]
    assert {route.logical_tuuid for route in routes} == {1}
    assert [route.target_op for route in routes] == ["consume_a", "consume_b"]
    assert [route.target_port for route in routes] == ["left", "right"]
    assert [route.output_index for route in routes] == [0, 0]
    assert all(route.routing_mode is TransportRoutingMode.ALGEBRAIC for route in routes)
    assert routes[0].target_spec != routes[1].target_spec
    assert routes[0].child_slots == routes[1].child_slots == (0,)
    assert {
        compiled.coherence_logical_tuuid(route.transfer_code) for route in routes
    } == {1}
    assert {
        compiled.coherence_child_slot(route.transfer_code, 0) for route in routes
    } == {0}


def test_coherence_child_slots_are_canonical_bijection():
    from types import SimpleNamespace
    from typing import Any, cast

    from fluxway.spmd.worker.data_plane import _compile_coherence_child_slots

    transfer = SimpleNamespace(logical_tuuid=7)
    contexts = (
        SimpleNamespace(
            spmd_id="target_b",
            target_world_size=2,
            dependencies=(SimpleNamespace(transfers=(transfer,)),),
        ),
        SimpleNamespace(
            spmd_id="target_a",
            target_world_size=2,
            dependencies=(SimpleNamespace(transfers=(transfer,)),),
        ),
    )

    slots = _compile_coherence_child_slots(cast(Any, contexts))

    assert slots == {
        (7, "target_a", 0): 0,
        (7, "target_a", 1): 1,
        (7, "target_b", 0): 2,
        (7, "target_b", 1): 3,
    }
    assert set(slots.values()) == set(range(len(slots)))


def test_sender_batches_shared_logical_tensor_by_transfer_code_without_rewrapping():
    replicated = ShardingSpec.chunk(PartitionSpec())
    compiled, _ = _compile_fanout_plan(second_target_sharding=replicated)
    raw = np.arange(4, dtype=np.float32)

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[raw],
            step_transfer_cache={},
            exec_step=5,
        )

    distributor.plan_and_shard.assert_called_once()
    call = distributor.plan_and_shard.call_args
    tensors = call.args[0]
    assert len(tensors) == 2
    assert tensors[0] is tensors[1]
    assert call.kwargs["transfer_codes"] == [1, 2]


def test_sender_batches_multiple_outputs_without_payload_copy():
    replicated = ShardingSpec.chunk(PartitionSpec())
    source = OperatorSpec(
        name="produce",
        output_arguments={"data": ("first", "second")},
        output_sharding={"first": replicated, "second": replicated},
    )
    target = OperatorSpec(
        name="consume",
        input_arguments={"data": ("left", "right")},
        input_sharding={"left": replicated, "right": replicated},
    )
    mesh = DeviceMesh(("data",), (1,))
    plan = compile_graph(
        name="sender-multiple-outputs",
        spmds=(
            compiled_spmd_spec("source", mesh, {"produce": source}),
            compiled_spmd_spec("target", mesh, {"consume": target}),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "produce"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(OperatorPort("first"), OperatorPort("left")),
                    TensorTransfer(OperatorPort("second"), OperatorPort("right")),
                ),
            ),
        ),
    )
    source_plan = plan.spmd("source")
    compiled = compile_send_plan(
        graph_view=source_plan,
        source_op_id="produce",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(0, source_plan.outgoing, effective_rank=0),
        ),
        sender_spmd_code=3,
        interleave_unit_size=None,
    )
    first = np.arange(4, dtype=np.float32)
    second = np.arange(4, 8, dtype=np.float32)

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[first, second],
            step_transfer_cache={},
            exec_step=5,
        )

    call = distributor.plan_and_shard.call_args
    tensors = call.args[0]
    assert [
        route.output_index for route in compiled.successor_routes[0].tensor_routes
    ] == [
        0,
        1,
    ]
    assert call.kwargs["transfer_codes"] == [1, 2]
    assert np.shares_memory(tensors[0].flat_values, first)
    assert np.shares_memory(tensors[1].flat_values, second)


def test_sender_separates_content_routed_outputs_with_distinct_values():
    dynamic = ShardingSpec.hash(P("data"))
    source = OperatorSpec(
        name="produce",
        output_arguments={"data": ("first", "second")},
        output_sharding={"first": dynamic, "second": dynamic},
    )
    target = OperatorSpec(
        name="consume",
        input_arguments={"data": ("left", "right")},
        input_sharding={"left": dynamic, "right": dynamic},
    )
    source_mesh = DeviceMesh(("data",), (1,))
    target_mesh = DeviceMesh(("data",), (2,))
    plan = compile_graph(
        name="sender-distinct-content-routing",
        spmds=(
            compiled_spmd_spec("source", source_mesh, {"produce": source}),
            compiled_spmd_spec("target", target_mesh, {"consume": target}),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "produce"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(OperatorPort("first"), OperatorPort("left")),
                    TensorTransfer(OperatorPort("second"), OperatorPort("right")),
                ),
            ),
        ),
    )
    source_plan = plan.spmd("source")
    compiled = compile_send_plan(
        graph_view=source_plan,
        source_op_id="produce",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(0, source_plan.outgoing, effective_rank=0),
        ),
        sender_spmd_code=3,
        interleave_unit_size=None,
    )
    first = np.array([0, 2], dtype=np.int64)
    second = np.array([1, 3], dtype=np.int64)

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[first, second],
            step_transfer_cache={},
            exec_step=5,
        )

    assert distributor.plan_and_shard.call_count == 2
    routed_values = [
        call.args[0][0].flat_values
        for call in distributor.plan_and_shard.call_args_list
    ]
    assert np.shares_memory(routed_values[0], first)
    assert np.shares_memory(routed_values[1], second)


def test_completed_without_tensor_rows_use_transfer_codes():
    replicated = ShardingSpec.chunk(PartitionSpec())
    compiled, _ = _compile_fanout_plan(second_target_sharding=replicated)

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        build_completed_without_tensor_step_frame_groups(
            compiled_plan=compiled,
            empty_values=None,
            exec_step=9,
        )

    assert [
        call.kwargs["transfer_code"]
        for call in distributor.plan_completed_without_tensor.call_args_list
    ] == [1, 2]


def test_declared_transfer_rejects_none_leaf_instead_of_silent_drop():
    replicated = ShardingSpec.chunk(PartitionSpec())
    compiled, _ = _compile_fanout_plan(second_target_sharding=replicated)

    with pytest.raises(ValueError, match="Declared transfer code 1"):
        build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[None],
            step_transfer_cache={},
            exec_step=9,
        )


def _compile_mirror_send_plan(*, bind_mirror_input: bool = True):
    dynamic = ShardingSpec.hash(P("data"))
    mirror = ShardingSpec.follow("lookup.unique_ids")
    upstream = OperatorSpec(
        name="upstream",
        output_arguments={"data": ("ids",)},
        output_sharding={"ids": dynamic},
    )
    lookup = OperatorSpec(
        name="lookup",
        input_arguments={"data": ("unique_ids",)},
        output_arguments={"data": ("embedding",)},
        input_sharding={"unique_ids": dynamic},
        output_sharding={"embedding": mirror},
    )
    target = OperatorSpec(
        name="consume",
        input_arguments={"data": ("embedding",)},
        input_sharding={"embedding": dynamic},
    )
    source_mesh = DeviceMesh(("data",), (2,))
    target_mesh = DeviceMesh(("data",), (2,))
    incoming = DependencySpec(
        source=OperatorKey("source", "upstream"),
        target=OperatorKey("source", "lookup"),
        transfers=(
            TensorTransfer(
                source=OperatorPort("ids"),
                target=OperatorPort("unique_ids"),
            ),
        ),
        local_only=True,
    )
    plan = compile_graph(
        name="sender-mirror-route",
        spmds=(
            compiled_spmd_spec(
                "source",
                source_mesh,
                {"lookup": lookup, "upstream": upstream},
            ),
            compiled_spmd_spec("target", target_mesh, {"consume": target}),
        ),
        dependencies=(
            *((incoming,) if bind_mirror_input else ()),
            _dependency("lookup", "consume", "embedding", "embedding"),
        ),
    )
    source_plan = plan.spmd("source")
    outgoing = tuple(
        dependency
        for dependency in source_plan.outgoing
        if dependency.source_operator.key.op_id == "lookup"
    )
    input_transfer_code = source_plan.incoming[0].transfers[0].transfer_code
    compiled = compile_send_plan(
        graph_view=source_plan,
        source_op_id="lookup",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(
                context_index=0,
                dependencies=outgoing,
                effective_rank=0,
            ),
        ),
        sender_spmd_code=4,
        interleave_unit_size=None,
    )
    return compiled, input_transfer_code


@pytest.mark.parametrize(
    ("source_world_size", "target_world_size", "receipt_world_size", "valid"),
    ((4, 2, 2, True), (2, 3, 2, False)),
)
def test_receipt_rank_domain_comes_from_runtime_lineage(
    source_world_size, target_world_size, receipt_world_size, valid
):
    dynamic = ShardingSpec.hash(P("data"))
    source_mesh = DeviceMesh(("data",), (source_world_size,))
    target_mesh = DeviceMesh(("data",), (target_world_size,))
    upstream = OperatorSpec(
        name="upstream",
        output_arguments={"data": ("ids",)},
        output_sharding={"ids": dynamic},
    )
    lookup = OperatorSpec(
        name="lookup",
        input_arguments={"data": ("ids",)},
        output_arguments={"data": ("embedding",)},
        input_sharding={"ids": dynamic},
        output_sharding={"embedding": ShardingSpec.follow("lookup.ids")},
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("embedding",)},
        input_sharding={"embedding": dynamic},
    )
    graph = compile_graph(
        name="receipt-rank-domain-mismatch",
        spmds=(
            compiled_spmd_spec(
                "source",
                source_mesh,
                {"upstream": upstream, "lookup": lookup},
            ),
            compiled_spmd_spec("target", target_mesh, {"consume": consume}),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "upstream"),
                target=OperatorKey("source", "lookup"),
                transfers=(TensorTransfer(OperatorPort("ids"), OperatorPort("ids")),),
                local_only=True,
            ),
            DependencySpec(
                source=OperatorKey("source", "lookup"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(
                        OperatorPort("embedding"),
                        OperatorPort("embedding"),
                    ),
                ),
            ),
        ),
    )
    source_view = graph.spmd("source")
    outgoing = tuple(
        dependency
        for dependency in source_view.outgoing
        if dependency.spec.source.op_id == "lookup"
    )

    compiled = compile_send_plan(
        graph_view=source_view,
        source_op_id="lookup",
        rank=0,
        successor_contexts=(SuccessorRouteContext(0, outgoing, effective_rank=0),),
        sender_spmd_code=1,
        interleave_unit_size=None,
    )
    route = compiled.successor_routes[0].tensor_routes[0]
    assert route.receipt_routing is not None
    assert route.receipt_routing.rank_count == target_world_size

    input_transfer_code = source_view.incoming[0].transfers[0].transfer_code
    reference = TransportTensor(
        np.ones((2,), dtype=np.int64),
        receipt=SymmetryReceipt(receipt_world_size, np.array([0, 1], dtype=np.int32)),
    )

    def build():
        return build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[np.ones((2, 8), dtype=np.float32)],
            step_transfer_cache={input_transfer_code: reference},
            exec_step=0,
        )

    if valid:
        assert build()
    else:
        with pytest.raises(ValueError, match="receipt world size does not match"):
            build()


def test_receipt_redirects_values_despite_identical_static_geometry():
    chunk = ShardingSpec.chunk(P("dp"))
    mesh = DeviceMesh(("dp",), (2,))
    upstream = OperatorSpec(
        name="produce",
        output_arguments={"data": ("value",)},
        output_sharding={"value": chunk},
    )
    forward = OperatorSpec(
        name="forward",
        input_arguments={"data": ("value",)},
        input_sharding={"value": chunk},
        output_arguments={"data": ("result",)},
        output_sharding={"result": ShardingSpec.follow("forward.value")},
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("value",)},
        input_sharding={"value": chunk},
    )
    graph = compile_graph(
        name="receipt-precedence",
        spmds=(
            compiled_spmd_spec("upstream", mesh, {"produce": upstream}),
            compiled_spmd_spec("source", mesh, {"forward": forward}),
            compiled_spmd_spec("target", mesh, {"consume": consume}),
        ),
        dependencies=(
            DependencySpec(
                OperatorKey("upstream", "produce"),
                OperatorKey("source", "forward"),
                (TensorTransfer(OperatorPort("value"), OperatorPort("value")),),
            ),
            _dependency("forward", "consume", "result", "value"),
        ),
    )
    view = graph.spmd("source")
    plan = compile_send_plan(
        graph_view=view,
        source_op_id="forward",
        rank=0,
        successor_contexts=(SuccessorRouteContext(0, view.outgoing, 0),),
        sender_spmd_code=2,
        interleave_unit_size=None,
    )
    route = plan.successor_routes[0].tensor_routes[0]
    assert route.receipt_routing is not None
    assert route.direct_targets is None
    values = np.arange(2)
    reference = TransportTensor(
        values,
        receipt=SymmetryReceipt(2, np.array([1, 1], dtype=np.int32)),
    )
    frames = build_step_frame_groups(
        compiled_plan=plan,
        flat_results=[values],
        step_transfer_cache={view.incoming[0].transfers[0].transfer_code: reference},
        exec_step=0,
    )
    destinations = []
    for _, (target, _, payloads) in frames:
        if any(payload.size for payload in payloads):
            destinations.append(target)
            np.testing.assert_array_equal(payloads[0], values)
    assert destinations == [1]


def test_mirror_route_uses_exact_input_transfer_code_cache_key():
    compiled, input_transfer_code = _compile_mirror_send_plan()
    route = compiled.successor_routes[0].tensor_routes[0]
    assert route.receipt_routing is not None
    assert route.direct_targets is None
    assert route.receipt_routing.input_transfer_code == input_transfer_code
    assert route.receipt_routing.reference_path == "lookup.unique_ids"
    assert route.receipt_routing.rank_count == 2
    assert route.routing_mode is TransportRoutingMode.RECEIPT
    assert route.source_spec.spec.is_dynamic
    assert not route.source_spec.spec.is_referential
    assert route.target_spec.spec.is_dynamic
    assert not route.target_spec.spec.is_referential
    source_ref = TransportTensor(
        np.array([3, 7], dtype=np.int64),
        origin_indices=ExplicitIndex(np.array([20, 10], dtype=np.int64)),
        global_shape=(64,),
        receipt=SymmetryReceipt(2, np.array([1, 0], dtype=np.int32)),
    )
    output = TransportTensor(np.ones((2, 8), dtype=np.float32), global_shape=(2, 8))
    output._bind_coherence_ticket_id(41)
    output_alias = output.zero_copy_readonly_alias()

    inherited = prepare_route_tensor(
        compiled_plan=compiled,
        route=route,
        raw_leaf=output_alias,
        step_transfer_cache={input_transfer_code: source_ref},
        exec_step=3,
        wrapped_results={},
    )

    assert inherited is not None
    assert inherited.receipt is source_ref.receipt
    assert inherited.origin_indices is source_ref.origin_indices
    assert inherited.coherence_ticket_id is None
    assert output.coherence_ticket_id == 41

    frame_groups = build_step_frame_groups(
        compiled_plan=compiled,
        flat_results=[output_alias],
        step_transfer_cache={input_transfer_code: source_ref},
        exec_step=3,
    )
    assert frame_groups
    assert output_alias.coherence_ticket_id is None
    assert output.coherence_ticket_id == 41


def test_mirror_route_rejects_unknown_path_without_logical_tuuid_fallback():
    with pytest.raises(ValueError, match="lookup.unique_ids"):
        _compile_mirror_send_plan(bind_mirror_input=False)


def test_mirror_route_rejects_input_without_required_receipt():
    compiled, input_transfer_code = _compile_mirror_send_plan()
    route = compiled.successor_routes[0].tensor_routes[0]
    source_ref = TransportTensor(
        np.array([3, 7], dtype=np.int64),
        global_shape=(64,),
    )

    with pytest.raises(ValueError, match="requires a SymmetryReceipt"):
        prepare_route_tensor(
            compiled_plan=compiled,
            route=route,
            raw_leaf=np.ones((2, 8), dtype=np.float32),
            step_transfer_cache={input_transfer_code: source_ref},
            exec_step=3,
            wrapped_results={},
        )


def test_mirror_route_preserves_transformed_tensor_lineage():
    compiled, input_transfer_code = _compile_mirror_send_plan()
    route = compiled.successor_routes[0].tensor_routes[0]
    source_ref = TransportTensor(
        np.array([3, 7], dtype=np.int64),
        origin_indices=ExplicitIndex(np.array([0, 1], dtype=np.int64)),
        receipt=SymmetryReceipt(2, np.array([0, 1], dtype=np.int32)),
    )
    embedding = TransportTensor(
        np.array([[10.0], [20.0]], dtype=np.float32),
        origin_indices=source_ref.origin_indices,
        receipt=source_ref.receipt,
    )
    transformed = embedding.gather(np.array([1, 0], dtype=np.int32))

    inherited = prepare_route_tensor(
        compiled_plan=compiled,
        route=route,
        raw_leaf=transformed,
        step_transfer_cache={input_transfer_code: source_ref},
        exec_step=3,
        wrapped_results={},
    )

    assert inherited.receipt is not None
    np.testing.assert_array_equal(
        inherited.receipt.sender_rank_tags,
        np.array([1, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        inherited.origin_indices_materialized,
        np.array([1, 0], dtype=np.int64),
    )


def test_sender_separates_mirror_outputs_with_distinct_receipts():
    dynamic = ShardingSpec.hash(P("data"))
    source_mesh = DeviceMesh(("data",), (2,))
    target_mesh = DeviceMesh(("data",), (2,))
    upstream = OperatorSpec(
        name="upstream",
        output_arguments={"data": ("ids_a", "ids_b")},
        output_sharding={"ids_a": dynamic, "ids_b": dynamic},
    )
    lookup = OperatorSpec(
        name="lookup",
        input_arguments={"data": ("ids_a", "ids_b")},
        output_arguments={"data": ("embedding_a", "embedding_b")},
        input_sharding={"ids_a": dynamic, "ids_b": dynamic},
        output_sharding={
            "embedding_a": ShardingSpec.follow("lookup.ids_a"),
            "embedding_b": ShardingSpec.follow("lookup.ids_b"),
        },
    )
    consume = OperatorSpec(
        name="consume",
        input_arguments={"data": ("embedding_a", "embedding_b")},
        input_sharding={"embedding_a": dynamic, "embedding_b": dynamic},
    )
    graph = compile_graph(
        name="sender-distinct-receipts",
        spmds=(
            compiled_spmd_spec(
                "source",
                source_mesh,
                {"upstream": upstream, "lookup": lookup},
            ),
            compiled_spmd_spec("target", target_mesh, {"consume": consume}),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "upstream"),
                target=OperatorKey("source", "lookup"),
                transfers=(
                    TensorTransfer(OperatorPort("ids_a"), OperatorPort("ids_a")),
                    TensorTransfer(OperatorPort("ids_b"), OperatorPort("ids_b")),
                ),
                local_only=True,
            ),
            DependencySpec(
                source=OperatorKey("source", "lookup"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(
                        OperatorPort("embedding_a"),
                        OperatorPort("embedding_a"),
                    ),
                    TensorTransfer(
                        OperatorPort("embedding_b"),
                        OperatorPort("embedding_b"),
                    ),
                ),
            ),
        ),
    )
    source_view = graph.spmd("source")
    outgoing = tuple(
        dependency
        for dependency in source_view.outgoing
        if dependency.spec.source.op_id == "lookup"
    )
    compiled = compile_send_plan(
        graph_view=source_view,
        source_op_id="lookup",
        rank=0,
        successor_contexts=(SuccessorRouteContext(0, outgoing, effective_rank=0),),
        sender_spmd_code=4,
        interleave_unit_size=None,
    )
    shared_origin = ExplicitIndex(np.array([0, 1], dtype=np.int64))
    input_transfers = source_view.incoming[0].transfers
    step_transfer_cache = {
        input_transfers[0].transfer_code: TransportTensor(
            np.array([2, 4], dtype=np.int64),
            origin_indices=shared_origin,
            receipt=SymmetryReceipt(2, 0),
        ),
        input_transfers[1].transfer_code: TransportTensor(
            np.array([3, 5], dtype=np.int64),
            origin_indices=shared_origin,
            receipt=SymmetryReceipt(2, 1),
        ),
    }

    with patch(
        "fluxway.spmd.worker.data_plane.AsyncDataDistributor"
    ) as distributor_cls:
        distributor = distributor_cls.return_value
        distributor.build_transport_frame_args.return_value = []
        build_step_frame_groups(
            compiled_plan=compiled,
            flat_results=[
                np.ones((2, 4), dtype=np.float32),
                np.full((2, 4), 2.0, dtype=np.float32),
            ],
            step_transfer_cache=step_transfer_cache,
            exec_step=5,
        )

    assert distributor.plan_and_shard.call_count == 2
    receipts = [
        call.args[0][0].receipt for call in distributor.plan_and_shard.call_args_list
    ]
    assert receipts == [
        step_transfer_cache[input_transfers[0].transfer_code].receipt,
        step_transfer_cache[input_transfers[1].transfer_code].receipt,
    ]


def test_successor_context_rejects_mixed_peer_dependencies():
    replicated = ShardingSpec.chunk(PartitionSpec())
    compiled, dependencies = _compile_fanout_plan(second_target_sharding=replicated)
    foreign = CompiledDependency.create(
        code=99,
        spec=DependencySpec(
            source=dependencies[0].spec.source,
            target=OperatorKey("foreign", "consume"),
        ),
        source_operator=dependencies[0].source_operator,
        target_operator=dependencies[0].target_operator,
        transfers=(),
        source_world_size=1,
        target_world_size=1,
        source_mesh=dependencies[0].source_mesh,
        target_mesh=dependencies[0].target_mesh,
        _unit_relations=__import__(
            "fluxway.common.frozen_mapping", fromlist=["freeze_mapping"]
        ).freeze_mapping({}),
    )

    with pytest.raises(ValueError, match="target SPMD"):
        SuccessorRouteContext(
            context_index=compiled.successor_routes[0].context_index,
            dependencies=(dependencies[0], foreign),
            effective_rank=0,
        )
