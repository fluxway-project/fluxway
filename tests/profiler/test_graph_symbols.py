from __future__ import annotations

import importlib.util
import inspect
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from fluxway.common.compiled_graph import (
    DependencySpec,
    GraphCompiler,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types.sharding import PartitionSpec, ShardingSpec
from fluxway.profiler.errors import GraphTraceSymbolConflict, MissingGraphTraceSymbol
from fluxway.profiler.graph_symbols import GraphTraceSymbols
from fluxway.profiler.perfetto_writer import (
    PerfettoTraceWriter,
    _GraphAnnotationResolver,
)
from fluxway.profiler.record import PAYLOAD_START_COLUMN, RECORD_WIDTH, TraceArena
from fluxway.profiler.recorder import TraceRecorder
from tests.compiled_graph_fixtures import compiled_spmd_spec


def _graph():
    replicated = ShardingSpec.chunk(PartitionSpec())
    source = OperatorSpec(
        name="emit",
        method_name="training.Source.emit",
        output_arguments={"data": ("batch.ids",), "control": ("step",)},
        output_param_map={
            "return": ("batch.ids", "step"),
            "batch": ("batch.ids",),
        },
        output_sharding={"batch.ids": replicated},
    )
    target = OperatorSpec(
        name="consume",
        method_name="training.Target.consume",
        input_arguments={
            "data": ("left.ids", "right.ids"),
            "control": ("step",),
            "static": ("training",),
        },
        input_param_map={
            "left": ("left.ids",),
            "right": ("right.ids",),
            "step": ("step",),
            "training": ("training",),
        },
        input_sharding={"left.ids": replicated, "right.ids": replicated},
    )
    dependency = DependencySpec(
        source=OperatorKey("source", "emit"),
        target=OperatorKey("target", "consume"),
        transfers=(
            TensorTransfer(
                source=OperatorPort("batch.ids"),
                target=OperatorPort("left.ids"),
            ),
            TensorTransfer(
                source=OperatorPort("batch.ids"),
                target=OperatorPort("right.ids"),
            ),
        ),
    )
    return compile_graph(
        name="profiler-symbols",
        spmds=(
            compiled_spmd_spec("source", DeviceMesh(("data",), (1,)), {"emit": source}),
            compiled_spmd_spec(
                "target", DeviceMesh(("data",), (1,)), {"consume": target}
            ),
        ),
        dependencies=(dependency,),
    )


def test_graph_trace_symbols_preserve_fanout_endpoints_and_data_ports():
    graph = _graph()
    symbols = GraphTraceSymbols.from_compiled_graph(graph)
    first, second = graph.dependencies[0].transfers

    assert first.logical_tuuid == second.logical_tuuid
    assert symbols.transfer(first.transfer_code).target_port == "left.ids"
    assert symbols.transfer(second.transfer_code).target_port == "right.ids"
    assert symbols.tensor(first.logical_tuuid).source_port == "batch.ids"

    operator = symbols.operator(OperatorKey("target", "consume"))
    assert operator.function_id == "training.Target.consume"
    assert operator.input_data_ports == ("left.ids", "right.ids")
    assert operator.output_data_ports == ()


def test_graph_trace_symbols_raise_typed_errors_for_unknown_identity():
    symbols = GraphTraceSymbols.from_compiled_graph(_graph())

    with pytest.raises(MissingGraphTraceSymbol, match="transfer"):
        symbols.transfer(999)
    with pytest.raises(MissingGraphTraceSymbol, match="operator"):
        symbols.operator(OperatorKey("target", "missing"))
    with pytest.raises(MissingGraphTraceSymbol, match="logical_tensor"):
        symbols.tensor(999)


def test_graph_trace_symbols_reject_mapping_identity_mismatches():
    symbols = GraphTraceSymbols.from_compiled_graph(_graph())
    operator_key, operator = next(iter(symbols.operators.items()))
    transfer_code, transfer = next(iter(symbols.transfers.items()))
    logical_tuuid, tensor = next(iter(symbols.logical_tensors.items()))

    with pytest.raises(GraphTraceSymbolConflict, match="operator"):
        replace(
            symbols,
            operators={OperatorKey(operator_key.spmd_id, "different"): operator},
        )
    with pytest.raises(GraphTraceSymbolConflict, match="transfer"):
        replace(symbols, transfers={transfer_code + 100: transfer})
    with pytest.raises(GraphTraceSymbolConflict, match="logical_tensor"):
        replace(symbols, logical_tensors={logical_tuuid + 100: tensor})


def test_graph_trace_symbols_reject_cross_reference_drift():
    symbols = GraphTraceSymbols.from_compiled_graph(_graph())
    transfer_code, transfer = next(iter(symbols.transfers.items()))
    logical_tuuid, tensor = next(iter(symbols.logical_tensors.items()))

    with pytest.raises(GraphTraceSymbolConflict, match="target input port"):
        replace(
            symbols,
            transfers={
                **symbols.transfers,
                transfer_code: replace(transfer, target_port="not-a-port"),
            },
        )
    with pytest.raises(GraphTraceSymbolConflict, match="source output port"):
        replace(
            symbols,
            logical_tensors={
                **symbols.logical_tensors,
                logical_tuuid: replace(tensor, source_port="not-a-port"),
            },
        )
    with pytest.raises(GraphTraceSymbolConflict, match="unknown logical TUUID"):
        replace(
            symbols,
            transfers={
                **symbols.transfers,
                transfer_code: replace(transfer, logical_tuuid=999),
            },
        )


def test_graph_trace_symbols_follow_append_only_graph_generations():
    full_graph = _graph()
    full_dependency = full_graph.dependencies[0].spec
    compiler = GraphCompiler(graph_id="profiler-generations")
    spmds = tuple(
        compiled_spmd_spec(
            spmd.spmd_id,
            spmd.mesh,
            {operator.key.op_id: operator.spec for operator in spmd.operators},
            spmd_class=spmd.spmd_class,
            resource_profile=spmd.resource_profile,
        )
        for spmd in full_graph.spmds
    )
    initial = compiler.compile(
        name=full_graph.name,
        spmds=spmds,
        dependencies=(
            replace(full_dependency, transfers=full_dependency.transfers[:1]),
        ),
    )
    extended = compiler.compile(
        name=full_graph.name,
        spmds=spmds,
        dependencies=(full_dependency,),
    )
    initial_symbols = GraphTraceSymbols.from_compiled_graph(initial)
    extended_symbols = GraphTraceSymbols.from_compiled_graph(extended)
    initial_transfer = initial.dependencies[0].transfers[0]
    extended_transfers = extended.dependencies[0].transfers

    assert initial_symbols.generation == 0
    assert extended_symbols.generation == 1
    assert extended_transfers[0].transfer_code == initial_transfer.transfer_code
    assert extended_transfers[1].transfer_code > initial_transfer.transfer_code
    assert extended_symbols.transfer(
        initial_transfer.transfer_code
    ) == initial_symbols.transfer(initial_transfer.transfer_code)
    assert extended_symbols.transfer(
        extended_transfers[1].transfer_code
    ).target_port == ("right.ids")


@pytest.mark.asyncio
async def test_perfetto_writer_adds_graph_symbols_without_widening_records(tmp_path):
    if importlib.util.find_spec("perfetto") is None:
        pytest.skip("perfetto is not installed")
    from perfetto.protos.perfetto.trace import perfetto_trace_pb2 as proto

    graph = _graph()
    symbols = GraphTraceSymbols.from_compiled_graph(graph)
    first, second = graph.dependencies[0].transfers
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=4 * RECORD_WIDTH * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/target/rank:0",
    )
    for transfer in (first, second):
        async with recorder.worker_assemble_tensor(
            op_name="consume",
            transfer_code=transfer.transfer_code,
            logical_tuuid=transfer.logical_tuuid,
            step=3,
            sender_count=1,
            shard_count=1,
        ):
            pass

    records = recorder.arena.snapshot()
    assert records.shape == (2, RECORD_WIDTH)
    path = tmp_path / "symbols.pftrace"
    PerfettoTraceWriter().write(
        records,
        path,
        metadata=recorder.metadata,
        graph_symbols=symbols,
    )

    trace = proto.Trace()
    trace.ParseFromString(path.read_bytes())
    target_ports = set()
    source_ports = set()
    source_functions = set()
    operator_functions = set()
    logical_tensor_producers = set()
    for packet in trace.packet:
        event = packet.track_event
        if (
            event.type != proto.TrackEvent.TYPE_SLICE_BEGIN
            or event.name != "worker.assemble_tensor"
        ):
            continue
        values = {
            annotation.name: annotation.string_value
            for annotation in event.debug_annotations
            if annotation.HasField("string_value")
        }
        target_ports.add(values["transfer.target.port"])
        source_ports.add(values["transfer.source.port"])
        source_functions.add(values["transfer.source.function"])
        operator_functions.add(values["operator.function"])
        assert not any(key.startswith("operator.input.") for key in values)
        assert not any(key.startswith("operator.output.") for key in values)
        logical_tensor_producers.add(values["logical_tensor.producer"])

    assert target_ports == {"left.ids", "right.ids"}
    assert source_ports == {"batch.ids"}
    assert source_functions == {"training.Source.emit"}
    assert operator_functions == {"training.Target.consume"}
    assert logical_tensor_producers == {"source.emit:batch.ids"}


def test_profiler_graph_symbols_do_not_enter_recorder_hot_path():
    source = inspect.getsource(TraceRecorder)
    resolver_source = inspect.getsource(_GraphAnnotationResolver.annotations)

    assert RECORD_WIDTH == 16
    assert "GraphTraceSymbols" not in source
    assert "graph_symbols" not in source
    assert "source_port" not in source
    assert "target_port" not in source
    assert "payload = {" not in resolver_source


@pytest.mark.asyncio
async def test_graph_annotation_resolver_reuses_cold_path_projection(monkeypatch):
    graph = _graph()
    symbols = GraphTraceSymbols.from_compiled_graph(graph)
    transfer = graph.dependencies[0].transfers[0]
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=4 * RECORD_WIDTH * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/target/rank:0",
    )
    for step in range(3):
        async with recorder.worker_assemble_tensor(
            op_name="consume",
            transfer_code=transfer.transfer_code,
            logical_tuuid=transfer.logical_tuuid,
            step=step,
            sender_count=1,
            shard_count=1,
        ):
            pass

    class CountingSymbols:
        def __init__(self) -> None:
            self.graph_id = symbols.graph_id
            self.calls = {
                "operator_by_id": 0,
                "operator": 0,
                "transfer": 0,
                "tensor": 0,
            }

        def operator_by_id(self, spmd_id: str, op_id: str):
            self.calls["operator_by_id"] += 1
            return symbols.operator_by_id(spmd_id, op_id)

        def operator(self, key: OperatorKey):
            self.calls["operator"] += 1
            return symbols.operator(key)

        def transfer(self, transfer_code: int):
            self.calls["transfer"] += 1
            return symbols.transfer(transfer_code)

        def tensor(self, logical_tuuid: int):
            self.calls["tensor"] += 1
            return symbols.tensor(logical_tuuid)

    counting_symbols = CountingSymbols()
    hierarchy_calls = 0
    from fluxway.profiler import perfetto_writer

    original_hierarchy = perfetto_writer.TraceTrackHierarchy.from_track_name

    def counted_hierarchy(track_name: str):
        nonlocal hierarchy_calls
        hierarchy_calls += 1
        return original_hierarchy(track_name)

    monkeypatch.setattr(
        perfetto_writer.TraceTrackHierarchy,
        "from_track_name",
        counted_hierarchy,
    )
    resolver = _GraphAnnotationResolver(
        metadata=recorder.metadata,
        graph_symbols=cast(GraphTraceSymbols, counting_symbols),
    )
    resolved = []
    for row in recorder.arena.snapshot():
        name_id = int(row[4])
        fields = recorder.metadata.payload_fields_for(name_id)
        resolved.append(
            resolver.annotations(
                track_uuid=int(row[3]),
                name_id=name_id,
                payload_values=row[
                    PAYLOAD_START_COLUMN : PAYLOAD_START_COLUMN + len(fields)
                ],
            )
        )

    assert resolved[0] is resolved[1] is resolved[2]
    assert counting_symbols.calls == {
        "operator_by_id": 1,
        "operator": 2,
        "transfer": 1,
        "tensor": 1,
    }
    assert hierarchy_calls == 1


@pytest.mark.asyncio
async def test_perfetto_writer_rejects_missing_operator_symbol(tmp_path):
    if importlib.util.find_spec("perfetto") is None:
        pytest.skip("perfetto is not installed")
    symbols = GraphTraceSymbols.from_compiled_graph(_graph())
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=RECORD_WIDTH * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/target/rank:0",
    )
    async with recorder.local_step_plan_execute(
        op_name="missing",
        step=1,
        node_count=1,
    ):
        pass

    with pytest.raises(MissingGraphTraceSymbol, match="missing"):
        PerfettoTraceWriter().write(
            recorder.arena.snapshot(),
            tmp_path / "missing.pftrace",
            metadata=recorder.metadata,
            graph_symbols=symbols,
        )


@pytest.mark.asyncio
async def test_perfetto_writer_allows_optional_execution_phase_operator_id(tmp_path):
    if importlib.util.find_spec("perfetto") is None:
        pytest.skip("perfetto is not installed")
    symbols = GraphTraceSymbols.from_compiled_graph(_graph())
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=RECORD_WIDTH * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/TrainMetrics/rank:0",
    )
    recorder.execution_phase(
        phase="execution.activate",
        op_name=None,
        step=None,
        epoch=0,
        sequence=1,
        execution_generation=1,
    )

    output = tmp_path / "execution-phase.pftrace"
    PerfettoTraceWriter().write(
        recorder.arena.snapshot(),
        output,
        metadata=recorder.metadata,
        graph_symbols=symbols,
    )
    assert output.is_file()


@pytest.mark.asyncio
async def test_perfetto_writer_uses_controller_execution_phase_track(tmp_path):
    if importlib.util.find_spec("perfetto") is None:
        pytest.skip("perfetto is not installed")
    symbols = GraphTraceSymbols.from_compiled_graph(_graph())
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=RECORD_WIDTH * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="controller",
    )
    recorder.controller_execution_phase(
        phase="activation.publish",
        op_name="read",
        step=0,
        epoch=0,
        sequence=1,
        execution_generation=1,
    )

    output = tmp_path / "controller-execution-phase.pftrace"
    PerfettoTraceWriter().write(
        recorder.arena.snapshot(),
        output,
        metadata=recorder.metadata,
        graph_symbols=symbols,
    )
    assert output.is_file()


@pytest.mark.asyncio
async def test_perfetto_writer_rejects_unknown_spmd_execution_phase_operator(
    tmp_path,
):
    if importlib.util.find_spec("perfetto") is None:
        pytest.skip("perfetto is not installed")
    symbols = GraphTraceSymbols.from_compiled_graph(_graph())
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=RECORD_WIDTH * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/target/rank:0",
    )
    recorder.execution_phase(
        phase="operator.invocation.begin",
        op_name="missing",
        step=0,
        epoch=0,
        sequence=1,
        execution_generation=1,
    )

    with pytest.raises(MissingGraphTraceSymbol, match="missing"):
        PerfettoTraceWriter().write(
            recorder.arena.snapshot(),
            tmp_path / "unknown-spmd-execution-phase.pftrace",
            metadata=recorder.metadata,
            graph_symbols=symbols,
        )


@pytest.mark.asyncio
async def test_graph_annotation_resolver_rejects_trace_graph_mismatch():
    graph = _graph()
    symbols = GraphTraceSymbols.from_compiled_graph(graph)
    compiled_operator = graph.spmd("target").operator("consume")
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=RECORD_WIDTH * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/target/rank:0",
    )
    async with recorder.user_callable_execute(
        graph_id="different-graph",
        compiled_operator=compiled_operator,
        step=1,
        execution_mode="coroutine_async",
    ):
        pass
    row = recorder.arena.snapshot()[0]
    name_id = int(row[4])
    fields = recorder.metadata.payload_fields_for(name_id)
    resolver = _GraphAnnotationResolver(
        metadata=recorder.metadata,
        graph_symbols=symbols,
    )

    with pytest.raises(GraphTraceSymbolConflict, match="trace_graph"):
        resolver.annotations(
            track_uuid=int(row[3]),
            name_id=name_id,
            payload_values=row[
                PAYLOAD_START_COLUMN : PAYLOAD_START_COLUMN + len(fields)
            ],
        )


@pytest.mark.asyncio
async def test_graph_annotation_resolver_rejects_track_spmd_mismatch():
    graph = _graph()
    symbols = GraphTraceSymbols.from_compiled_graph(graph)
    compiled_operator = graph.spmd("source").operator("emit")
    recorder = TraceRecorder(
        TraceArena(buffer_bytes=RECORD_WIDTH * np.dtype(np.int64).itemsize),
        run_id="run",
        process_name="worker/target/rank:0",
    )
    async with recorder.user_callable_execute(
        graph_id=graph.graph_id,
        compiled_operator=compiled_operator,
        step=1,
        execution_mode="coroutine_async",
    ):
        pass
    row = recorder.arena.snapshot()[0]
    name_id = int(row[4])
    fields = recorder.metadata.payload_fields_for(name_id)
    resolver = _GraphAnnotationResolver(
        metadata=recorder.metadata,
        graph_symbols=symbols,
    )

    with pytest.raises(GraphTraceSymbolConflict, match="trace_operator_spmd"):
        resolver.annotations(
            track_uuid=int(row[3]),
            name_id=name_id,
            payload_values=row[
                PAYLOAD_START_COLUMN : PAYLOAD_START_COLUMN + len(fields)
            ],
        )
