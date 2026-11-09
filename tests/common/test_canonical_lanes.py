from __future__ import annotations

import fluxway as fw
from fluxway.common.canonical_lanes import (
    CANONICAL_LANE_COMPUTE,
    CANONICAL_LANE_IO,
    infer_canonical_lane,
    is_coroutine_operator,
    is_io_operator,
)
from fluxway.common.decorators import operator, spmd
from fluxway.common.operator_policy import AsyncPolicy, SchedulePolicy
from fluxway.common.operator_spec import OperatorSpec
from fluxway.reader.base_reader import BaseReader
from fluxway.spmd.worker.worker import SPMDWorker


@spmd(
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(1,),
)
class DummyReader(BaseReader):
    @operator("read", output_sharding={"data": {"type": "static", "axis": []}})
    def read(self) -> dict[str, list[int]]:
        return {"data": [1, 2, 3]}


@spmd(
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(1,),
)
class DummyTrainer:
    @operator("train", input_sharding={"data": {"type": "static", "axis": []}})
    def train(self, data: list[int]) -> int:
        return len(data)

    @operator("async_fetch")
    async def async_fetch(self) -> str:
        return "async_data"

    @operator(
        "custom_op",
        async_policy=AsyncPolicy.pipelined(
            depth=2,
            schedule=SchedulePolicy.fifo(
                lane="custom_lane", lane_order=("custom_lane",)
            ),
        ),
    )
    def custom_op(self) -> str:
        return "custom"


def test_operator_classification():
    read_spec = OperatorSpec(name="read", method_name="read", owner=DummyReader)
    assert is_io_operator(read_spec)
    assert not is_coroutine_operator(read_spec)
    assert infer_canonical_lane(read_spec) == CANONICAL_LANE_IO

    source_spec = OperatorSpec(name="src", is_source=True)
    assert is_io_operator(source_spec)
    assert infer_canonical_lane(source_spec) == CANONICAL_LANE_IO

    train_spec = OperatorSpec(name="train", method_name="train", owner=DummyTrainer)
    assert not is_io_operator(train_spec)
    assert not is_coroutine_operator(train_spec)
    assert infer_canonical_lane(train_spec) == CANONICAL_LANE_COMPUTE

    async_spec = OperatorSpec(
        name="async_fetch", method_name="async_fetch", owner=DummyTrainer
    )
    assert not is_io_operator(async_spec)
    assert is_coroutine_operator(async_spec)
    assert infer_canonical_lane(async_spec) is None


def test_builder_canonical_lane_integration():
    builder = fw.Builder()
    builder.add_node("reader", DummyReader)
    builder.add_node("trainer", DummyTrainer)
    builder.add_func_edge(
        "reader",
        "read",
        "trainer",
        "train",
        tensor_map={"data": "data"},
    )
    builder.add_func_edge(
        "trainer",
        "train",
        "trainer",
        "async_fetch",
        local_only=True,
    )
    graph = builder.build("canonical_graph")

    reader_specs = graph.nodes["reader"].spmd_config.operator_specs
    trainer_specs = graph.nodes["trainer"].spmd_config.operator_specs

    # Reader op gets inferred 'io' lane and collaborative priority
    assert reader_specs["read"].async_policy.schedule.lane == "io"
    assert reader_specs["read"].async_policy.schedule.lane_ordering == ("compute", "io")

    # Trainer sync op gets inferred 'compute' lane and collaborative priority
    assert trainer_specs["train"].async_policy.schedule.lane == "compute"
    assert trainer_specs["train"].async_policy.schedule.lane_ordering == (
        "compute",
        "io",
    )

    # Async coroutine op keeps lane None
    assert trainer_specs["async_fetch"].async_policy.schedule.lane is None
    assert trainer_specs["async_fetch"].async_policy.schedule.lane_ordering == ()

    # Explicit user configuration is strictly preserved
    assert trainer_specs["custom_op"].async_policy.schedule.lane == "custom_lane"
    assert trainer_specs["custom_op"].async_policy.schedule.lane_ordering == (
        "custom_lane",
    )


def test_canonical_lane_single_lane_no_ordering():
    builder = fw.Builder()
    builder.add_node("trainer", DummyTrainer)
    builder.add_func_edge(
        "trainer",
        "train",
        "trainer",
        "custom_op",
        local_only=True,
    )
    graph = builder.build("single_node_graph")

    train_op = graph.nodes["trainer"].spmd_config.operator_specs["train"]
    assert train_op.async_policy.schedule.lane == CANONICAL_LANE_COMPUTE
    # When io is absent, ordering is not injected
    assert train_op.async_policy.schedule.lane_ordering == ()


def test_worker_priority_cooperation():
    schedule_compute = SchedulePolicy.fifo(lane="compute", lane_order=("compute", "io"))
    schedule_io = SchedulePolicy.fifo(lane="io", lane_order=("compute", "io"))

    p_compute = SPMDWorker._lane_priority(schedule_compute, "compute")
    p_io = SPMDWorker._lane_priority(schedule_io, "io")

    assert p_compute == 1
    assert p_io == 0
    assert p_compute > p_io
