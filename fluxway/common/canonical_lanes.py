from __future__ import annotations

import inspect
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fluxway.common.frozen_mapping import freeze_mapping
from fluxway.common.operator_spec import OperatorSpec

if TYPE_CHECKING:
    from fluxway.common.compiled_graph import SPMDGraphSpec

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


def is_io_operator(spec: OperatorSpec, spmd_class: type | None = None) -> bool:
    """Return True if the operator is an I/O source or reader operator."""
    if spec.is_source:
        return True
    if spec.source is not None:
        return True
    owner = spec.owner or spmd_class
    if owner is not None and isinstance(owner, type):
        if issubclass(owner, _get_base_reader()):
            return True
    return False


def is_coroutine_operator(spec: OperatorSpec, spmd_class: type | None = None) -> bool:
    """Return True if the operator implementation is an async coroutine function."""
    owner = spec.owner or spmd_class
    method_name = spec.method_name or spec.name
    if owner is not None and method_name:
        method = getattr(owner, method_name, None)
        if method is not None:
            if inspect.iscoroutinefunction(method):
                return True
            if hasattr(method, "func") and inspect.iscoroutinefunction(method.func):
                return True
            if hasattr(method, "__wrapped__") and inspect.iscoroutinefunction(
                method.__wrapped__
            ):
                return True
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
    if is_io_operator(spec, spmd_class):
        return CANONICAL_LANE_IO
    if not is_coroutine_operator(spec, spmd_class):
        return CANONICAL_LANE_COMPUTE
    return None


def apply_canonical_lane_presets(
    spmds: tuple[SPMDGraphSpec, ...],
) -> tuple[SPMDGraphSpec, ...]:
    """Inspect SPMD operator metadata and apply canonical lane and ordering presets.

    1. Operators without an explicit lane are classified into 'io' or 'compute'.
    2. When both canonical lanes ('compute' and 'io') are identified across the graph,
       operators in canonical lanes without custom lane_ordering/lane_priorities
       are injected with lane_order=('compute', 'io') to prioritize compute over I/O.
    """
    inferred_lanes: dict[tuple[int, str], str | None] = {}
    has_compute = False
    has_io = False
    for spmd_idx, spmd in enumerate(spmds):
        for op_id, spec in spmd.operators.items():
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

            if updates:
                new_schedule = schedule.model_copy(update=updates)
                new_async = spec.async_policy.model_copy(
                    update={"schedule": new_schedule}
                )
                new_operators[op_id] = spec.updated(async_policy=new_async)
                spmd_changed = True
            else:
                new_operators[op_id] = spec

        if spmd_changed:
            updated_spmds.append(replace(spmd, operators=freeze_mapping(new_operators)))
        else:
            updated_spmds.append(spmd)

    return tuple(updated_spmds)
