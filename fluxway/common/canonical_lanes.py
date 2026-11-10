from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fluxway.common.frozen_mapping import freeze_mapping
from fluxway.common.operator_spec import OperatorExecutionTrait, OperatorSpec

if TYPE_CHECKING:
    from fluxway.common.compiled_graph import SPMDGraphSpec
    from fluxway.common.operator_policy import SchedulePolicy

CANONICAL_LANE_COMPUTE: str = "compute"
CANONICAL_LANE_IO: str = "io"
CANONICAL_LANE_ORDER: tuple[str, ...] = (CANONICAL_LANE_COMPUTE, CANONICAL_LANE_IO)


_BASE_READER_CLS: type | None = None


def _get_base_reader() -> type:
    global _BASE_READER_CLS
    if _BASE_READER_CLS is None:
        from fluxway.reader.base_reader import BaseReader

        _BASE_READER_CLS = BaseReader
    return _BASE_READER_CLS


def compute_lane_priority(policy: SchedulePolicy, lane: str | None) -> int:
    """Compute the effective priority score for a lane under a SchedulePolicy.

    Single source of truth for lane ordering and priority calculation.
    """
    if lane is None:
        return 0
    priorities = policy.lane_priorities
    if lane in priorities:
        return priorities[lane]
    lane_order = policy.lane_ordering
    if not lane_order:
        return 0
    if lane not in lane_order:
        raise ValueError(f"lane {lane!r} is not in lane_order={lane_order!r}.")
    return len(lane_order) - lane_order.index(lane) - 1


def resolve_operator_execution_trait(
    spec: OperatorSpec, spmd_class: type | None = None
) -> OperatorExecutionTrait:
    """Resolve the execution trait of an operator cleanly from static metadata."""
    if spec.execution_trait is not None:
        return spec.execution_trait
    if spec.is_source is True or spec.source is not None:
        return OperatorExecutionTrait.IO_SOURCE
    owner = spec.owner or spmd_class
    if (
        owner is not None
        and isinstance(owner, type)
        and issubclass(owner, _get_base_reader())
    ):
        return OperatorExecutionTrait.IO_SOURCE
    return OperatorExecutionTrait.SYNC_COMPUTE


def is_io_operator(spec: OperatorSpec, spmd_class: type | None = None) -> bool:
    """Return True if the operator is an I/O source or reader operator."""
    if spec.execution_trait == OperatorExecutionTrait.IO_SOURCE:
        return True
    if spec.is_source or spec.source is not None:
        return True
    owner = spec.owner or spmd_class
    if (
        owner is not None
        and isinstance(owner, type)
        and issubclass(owner, _get_base_reader())
    ):
        return True
    return False


def is_coroutine_operator(spec: OperatorSpec, spmd_class: type | None = None) -> bool:
    """Return True if the operator implementation is an async coroutine function."""
    if spec.execution_trait is not None:
        return spec.execution_trait == OperatorExecutionTrait.ASYNC_COROUTINE
    return False


def infer_canonical_lane(
    spec: OperatorSpec, spmd_class: type | None = None
) -> str | None:
    """Infer the default canonical lane for an operator based on its metadata.

    If the operator already declares an explicit lane, the declared lane is returned.
    Otherwise:
    - I/O / source operators default to 'io'.
    - Synchronous compute operators default to 'compute'.
    - Coroutine operators return None (unassigned lane, executed in event loop).
    """
    if spec.async_policy.schedule.lane is not None:
        return spec.async_policy.schedule.lane
    trait = resolve_operator_execution_trait(spec, spmd_class)
    if trait == OperatorExecutionTrait.IO_SOURCE:
        return CANONICAL_LANE_IO
    if trait == OperatorExecutionTrait.SYNC_COMPUTE:
        return CANONICAL_LANE_COMPUTE
    if trait == OperatorExecutionTrait.ASYNC_COROUTINE:
        return None
    raise ValueError(f"Unknown operator execution trait: {trait!r}")


def apply_canonical_lane_presets(
    spmds: tuple[SPMDGraphSpec, ...],
) -> tuple[SPMDGraphSpec, ...]:
    """Inspect SPMD operator metadata and apply canonical lane and ordering presets.

    1. Operators without an explicit lane are classified into 'io' or 'compute'.
    2. Execution traits are resolved and populated onto OperatorSpec.
    3. When both canonical lanes ('compute' and 'io') are identified across the graph,
       operators in canonical lanes without custom lane_ordering/lane_priorities
       are injected with lane_order=('compute', 'io') to prioritize compute over I/O.
    """
    inferred_lanes: dict[tuple[int, str], str | None] = {}
    resolved_traits: dict[tuple[int, str], OperatorExecutionTrait] = {}
    has_compute = False
    has_io = False
    for spmd_idx, spmd in enumerate(spmds):
        for op_id, spec in spmd.operators.items():
            trait = resolve_operator_execution_trait(spec, spmd.spmd_class)
            resolved_traits[(spmd_idx, op_id)] = trait
            target_lane = infer_canonical_lane(spec, spmd.spmd_class)
            inferred_lanes[(spmd_idx, op_id)] = target_lane
            effective_lane = spec.async_policy.schedule.lane or target_lane
            if effective_lane == CANONICAL_LANE_COMPUTE:
                has_compute = True
            elif effective_lane == CANONICAL_LANE_IO:
                has_io = True

    should_inject_ordering = has_compute and has_io

    updated_spmds: list[SPMDGraphSpec] = []
    for spmd_idx, spmd in enumerate(spmds):
        spmd_changed = False
        new_operators: dict[str, OperatorSpec] = {}
        for op_id, spec in spmd.operators.items():
            schedule = spec.async_policy.schedule
            target_lane = inferred_lanes[(spmd_idx, op_id)]
            trait = resolved_traits[(spmd_idx, op_id)]
            effective_lane = schedule.lane if schedule.lane is not None else target_lane

            updates: dict[str, Any] = {}
            if schedule.lane is None and target_lane is not None:
                updates["lane"] = target_lane
            if (
                should_inject_ordering
                and effective_lane in (CANONICAL_LANE_COMPUTE, CANONICAL_LANE_IO)
                and not schedule.lane_ordering
                and not schedule.lane_priorities
            ):
                updates["lane_ordering"] = CANONICAL_LANE_ORDER

            spec_updates: dict[str, Any] = {}
            if spec.execution_trait is None:
                spec_updates["execution_trait"] = trait
            if updates:
                new_schedule = schedule.model_copy(update=updates)
                new_async = spec.async_policy.model_copy(
                    update={"schedule": new_schedule}
                )
                spec_updates["async_policy"] = new_async

            if spec_updates:
                new_operators[op_id] = spec.updated(**spec_updates)
                spmd_changed = True
            else:
                new_operators[op_id] = spec

        if spmd_changed:
            updated_spmds.append(replace(spmd, operators=freeze_mapping(new_operators)))
        else:
            updated_spmds.append(spmd)

    return tuple(updated_spmds)
