from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError
from ray import cloudpickle

import fluxway as fw
from fluxway.common.decorators import operator, spmd
from fluxway.common.operator_policy import AsyncPolicy, SchedulePolicy, StatePolicy


_ASYNC_POLICY = AsyncPolicy.bounded_stale(
    depth=3,
    max_staleness=2,
    schedule=SchedulePolicy.deadline_boost(
        base_priority=7,
        boost_priority=101,
    ),
    state=StatePolicy.thread_safe(),
)


@spmd(
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(1,),
)
class _AsyncPolicySPMD:
    @operator(name="step", async_policy=_ASYNC_POLICY)
    def step(self) -> None:
        return None

    @operator(name="after")
    def after(self) -> None:
        return None


def _operator_spec():
    return _AsyncPolicySPMD.step._fluxway_operator_spec


def test_async_policy_is_the_only_operator_async_fact_source():
    spec = _operator_spec()

    assert spec.async_policy is _ASYNC_POLICY
    assert {
        "priority",
        "max_inflight_steps",
        "state_policy",
        "schedule_policy",
        "max_async_lag",
        "local_pipeline_depth",
        "commit_policy",
        "state_access",
        "state_resource",
    }.isdisjoint(type(spec).model_fields)

    runtime = spec.runtime_policy
    assert runtime.mode == "bounded_stale"
    assert runtime.pipeline_depth == 3
    assert runtime.max_async_lag == 2
    assert runtime.max_inflight_steps == 2
    assert runtime.commit_policy == "relaxed"
    assert runtime.state_access == "thread_safe"
    assert runtime.state_resource is None
    assert runtime.schedule is _ASYNC_POLICY.schedule
    assert runtime.priority == 7
    assert runtime.max_staleness == 2
    assert runtime.is_pipelined is True
    assert runtime.has_bounded_staleness is True

    with pytest.raises(FrozenInstanceError):
        runtime.pipeline_depth = 4


def test_async_policy_and_nested_policies_are_immutable():
    with pytest.raises(ValidationError, match="frozen"):
        _ASYNC_POLICY.depth = 4  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        _ASYNC_POLICY.schedule.base_priority = 8  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        _ASYNC_POLICY.state.access = "exclusive"  # type: ignore[misc]


def test_compiled_graph_reuses_operator_spec_and_ray_preserves_semantics():
    builder = fw.Builder()
    builder.add_node("pipeline", _AsyncPolicySPMD)
    builder.add_func_edge(
        "pipeline",
        "step",
        "pipeline",
        "after",
        local_only=True,
    )
    graph = builder.build("async-policy-ssot")
    runtime_spec = graph.nodes["pipeline"].spmd_config.operator_specs["step"]
    compiled = graph.require_compiled_graph().spmd("pipeline").operator("step")
    assert compiled.spec is runtime_spec
    assert runtime_spec.async_policy is compiled.spec.async_policy
    assert runtime_spec.runtime_policy.schedule is compiled.spec.async_policy.schedule

    restored = cloudpickle.loads(cloudpickle.dumps(graph))
    restored_spec = restored.nodes["pipeline"].spmd_config.operator_specs["step"]
    restored_compiled = (
        restored.require_compiled_graph().spmd("pipeline").operator("step")
    )
    assert restored_compiled.spec is restored_spec
    assert restored_spec.async_policy == runtime_spec.async_policy
    assert restored_spec.runtime_policy == runtime_spec.runtime_policy
