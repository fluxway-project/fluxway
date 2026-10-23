from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from dataclasses import replace

import pytest

from fluxway.common.ingestion import LayoutUnitDescriptor
from fluxway.common.ingestion.source_scan import (
    OpaqueSourceCursor,
    SourceScanBudget,
    SourceScanResult,
    SourceScanStatus,
)
from fluxway.common.metadata import metadata_digest
from fluxway.common.types import CheckpointCommitPermit, CheckpointCut
from fluxway.control_plane.ingestion import (
    AlgebraicLayoutCompiler,
    AssignmentPolicy,
    IngestionLayoutPlanningComponent,
    LayoutCompileRequest,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
    TailLayoutMode,
)
from tests.control_plane.source_scan_fixtures import InMemorySourceScanAdapter

from fluxway.control_plane.ingestion.planning_component import (
    SourcePlanningBudgetExceeded,
)
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger


def _fixed_initial_cursor(cursor):
    return lambda: cursor


def _memory_source(request):
    return InMemorySourceScanAdapter(
        (
            SourceScanResult(
                source_identity=request.source_identity,
                view_identity=request.view_identity,
                schema_digest=request.schema_digest,
                lineage_digest=request.lineage_digest,
                units=request.units,
                unit_cursors=tuple(
                    InMemorySourceScanAdapter.cursor(0) for _ in request.units
                ),
                records=request.cardinality,
                next_cursor=InMemorySourceScanAdapter.cursor(request.cardinality),
                status=SourceScanStatus.READY
                if request.tail_layout_mode is TailLayoutMode.RETAIN_PENDING
                else SourceScanStatus.TERMINAL,
            ),
        )
    )


async def _scan_request(planner, ledger, request, *, max_steps=None):
    return await planner.plan_source_scan(
        _memory_source(request),
        _fixed_initial_cursor(InMemorySourceScanAdapter.cursor(0)),
        ledger,
        InMemorySourceScanAdapter.cursor(0),
        budget=SourceScanBudget(request.cardinality, len(request.units)),
        topology=request.topology,
        tail_layout_mode=request.tail_layout_mode,
        assignment_policy=AssignmentPolicy.default(),
        epoch=0,
        start_position=0,
        max_steps=max_steps,
    )


@pytest.mark.asyncio
async def test_failed_publication_does_not_install_planner_or_cursor(
    source_layout, monkeypatch
) -> None:
    compiled = source_layout(demands=(1,), cardinality=2, terminal=True)
    planner = IngestionLayoutPlanningComponent(
        "atomic", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)

    def reject(*args, **kwargs):
        raise RuntimeError("prepare rejected")

    monkeypatch.setattr(ledger, "prepare_publication", reject)
    with pytest.raises(RuntimeError, match="prepare rejected"):
        await planner.plan_and_publish(
            ledger,
            compiled.request,
            epoch=0,
            start_position=0,
        )
    assert planner.windows == ()
    image = planner.checkpoint_state(CheckpointCut.for_position("cut", 0))
    assert image["cursor"] is None
    assert ledger.epochs == ()
    await planner.close()


@pytest.mark.asyncio
async def test_descriptor_budget_failure_reports_redacted_measurements(
    source_layout,
) -> None:
    compiled = source_layout(demands=(1, 1), cardinality=2, terminal=True)
    planner = IngestionLayoutPlanningComponent(
        "descriptor-budget", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)

    try:
        with pytest.raises(SourcePlanningBudgetExceeded) as raised:
            await planner.plan_and_publish(
                ledger,
                compiled.request,
                epoch=0,
                start_position=0,
                max_descriptor_bytes=1,
            )

        error = raised.value
        assert error.budget_name == "encoded_descriptor_bytes"
        assert error.limit == 1
        assert error.observed == sum(
            len(item.encoded) for item in compiled.encoded_worker_layouts()
        )
        assert error.unit_count == len(compiled.request.units)
        assert error.row_count == compiled.request.cardinality
        assert error.complete_steps == 1
        assert error.window_start_step == 0
        assert error.window_end_step == 1
        assert error.reader_count == len(compiled.request.topology.groups)
        assert error.source_digest
        assert "source_identity=" not in str(error)
        assert "view_identity=" not in str(error)
        assert planner.windows == ()
        assert ledger.epochs == ()
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_planner_publishes_bounded_complete_steps_and_keeps_tail(
    source_layout,
) -> None:
    compiled = source_layout(demands=(2,), cardinality=6, terminal=True)
    planner = IngestionLayoutPlanningComponent(
        "bounded-window", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=4)

    try:
        activation = await planner.plan_and_publish(
            ledger,
            compiled.request,
            epoch=0,
            start_position=0,
            max_steps=1,
            max_records=6,
            max_descriptor_bytes=64 * 1024 * 1024,
        )

        assert activation.epoch.start_position == 0
        assert activation.epoch.end_position == 1
        assert sum(unit.cardinality for unit in planner._pending_units) == 4
        assert activation.epoch.terminal is False
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_planner_shrinks_complete_steps_to_descriptor_bound(
    source_layout,
) -> None:
    compiled = source_layout(demands=(2,), cardinality=4, terminal=True)
    source_unit = compiled.request.units[0]
    one_step_request = replace(
        compiled.request,
        units=(source_unit.slice(0, 2),),
        tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
    )
    one_step = AlgebraicLayoutCompiler().compile(
        one_step_request, epoch=0, start_step=0
    )
    one_step_bound = (
        sum(len(item.encoded) for item in one_step.encoded_worker_layouts()) + 10
    )
    planner = IngestionLayoutPlanningComponent(
        "shrink-window", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=4)

    try:
        activation = await planner.plan_and_publish(
            ledger,
            compiled.request,
            epoch=0,
            start_position=0,
            max_descriptor_bytes=one_step_bound,
        )
        assert activation.epoch.end_position == 1
        assert sum(unit.cardinality for unit in planner._pending_units) == 2
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_descriptor_budget_failure_reports_final_one_step_measurement(
    source_layout,
) -> None:
    compiled = source_layout(demands=(2,), cardinality=4, terminal=True)
    source_unit = compiled.request.units[0]
    one_step_request = replace(
        compiled.request,
        units=(source_unit.slice(0, 2),),
        tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
    )
    one_step = AlgebraicLayoutCompiler().compile(
        one_step_request, epoch=0, start_step=0
    )
    one_step_reader_bytes = tuple(
        len(item.encoded) for item in one_step.encoded_worker_layouts()
    )
    planner = IngestionLayoutPlanningComponent(
        "one-step-budget", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=4)

    try:
        with pytest.raises(SourcePlanningBudgetExceeded) as raised:
            await planner.plan_and_publish(
                ledger,
                compiled.request,
                epoch=0,
                start_position=0,
                max_descriptor_bytes=sum(one_step_reader_bytes) - 1,
            )

        error = raised.value
        assert error.complete_steps == 1
        assert error.row_count == 2
        assert error.unit_count == 1
        assert error.reader_encoded_bytes == one_step_reader_bytes
        assert error.observed == sum(one_step_reader_bytes)
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_unit_budget_failure_reports_minimal_complete_step() -> None:
    base = LayoutUnitDescriptor(
        "unit-0",
        "revision",
        1,
        "rows",
        "SplittableSpan",
        "node",
        "file",
        0,
        b"metadata-0",
    )
    request = LayoutCompileRequest(
        source_identity="source",
        view_identity="revision",
        schema_digest="schema",
        lineage_digest="lineage",
        units=(
            base,
            replace(
                base,
                unit_id="unit-1",
                logical_start=1,
                opaque_token=b"metadata-1",
            ),
        ),
        topology=ReaderTopologyDescriptor(
            2, (ReaderGroupDemand("reader", (RankDemand("reader", 2),)),)
        ),
        tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
        assignment_policy=AssignmentPolicy.default(),
    )
    planner = IngestionLayoutPlanningComponent("unit-budget", topology=request.topology)
    ledger = LogicalWorkLedger(max_in_flight_steps=2)

    try:
        with pytest.raises(SourcePlanningBudgetExceeded) as raised:
            await planner.plan_and_publish(
                ledger,
                request,
                epoch=0,
                start_position=0,
                max_units=1,
            )

        error = raised.value
        assert error.budget_name == "units"
        assert error.observed == 2
        assert error.limit == 1
        assert error.complete_steps == 1
        assert error.row_count == 2
        assert error.unit_count == 2
        assert planner.windows == ()
        assert ledger.epochs == ()
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_planner_advances_next_window_from_pending_tail(source_layout) -> None:
    compiled = source_layout(demands=(2,), cardinality=6, terminal=True)
    planner = IngestionLayoutPlanningComponent(
        "next-window", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=4)

    try:
        first = await planner.plan_and_publish(
            ledger,
            compiled.request,
            epoch=0,
            start_position=0,
            max_steps=1,
            max_descriptor_bytes=64 * 1024 * 1024,
        )
        second = await planner.plan_next_window(
            ledger,
            max_steps=1,
            max_descriptor_bytes=64 * 1024 * 1024,
        )
        assert first.epoch.end_position == second.epoch.start_position == 1
        assert second.epoch.end_position == 2
        assert second.epoch.terminal is False
        assert tuple(window.work_epoch.epoch for window in planner.windows) == (0, 1)
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_planner_reclaim_keeps_only_latest_window_anchor(source_layout) -> None:
    compiled = source_layout(demands=(1,), cardinality=6, terminal=True)
    planner = IngestionLayoutPlanningComponent(
        "reclaim", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=4)
    try:
        await planner.plan_and_publish(
            ledger,
            compiled.request,
            epoch=0,
            start_position=0,
            max_steps=1,
            max_descriptor_bytes=64 * 1024 * 1024,
        )
        await planner.plan_next_window(
            ledger, max_steps=1, max_descriptor_bytes=64 * 1024 * 1024
        )
        assert tuple(window.work_epoch.epoch for window in planner.windows) == (0, 1)
        planner.reclaim_before(1)
        assert tuple(window.work_epoch.epoch for window in planner.windows) == (1,)
        assert len(planner.windows) == 1
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_planner_checkpoint_uses_component_transaction_capability(
    source_layout,
) -> None:
    compiled = source_layout(demands=(1,), cardinality=2, terminal=True)
    planner = IngestionLayoutPlanningComponent(
        "transactional-planner", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    try:
        await _scan_request(planner, ledger, compiled.request)
        cut = CheckpointCut.for_position("cut", 2)
        state = planner.checkpoint_state(cut)
        permit = CheckpointCommitPermit("attempt", 0, b"token")
        prepared = planner.prepare_checkpoint(
            "checkpoint", permit, cut, metadata_digest(state)
        )
        planner.apply_prepared(prepared)
        planner.apply_prepared(prepared)
        planner.abort_prepared("checkpoint", permit)
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_scan_result_planner_assembles_complete_step_across_ready_scan_results(
    source_layout,
) -> None:
    compiled = source_layout(demands=(2,), cardinality=4, terminal=True)
    first_unit = replace(compiled.request.units[0], cardinality=2)
    second_unit = replace(
        compiled.request.units[0],
        unit_id="unit-1",
        logical_start=2,
        cardinality=2,
    )
    cursors = tuple(
        OpaqueSourceCursor.from_payload(index.to_bytes(8, "big")) for index in range(3)
    )
    scan_results = (
        SourceScanResult(
            source_identity="source",
            view_identity="r",
            schema_digest="schema",
            lineage_digest="r",
            units=(first_unit,),
            unit_cursors=tuple(
                OpaqueSourceCursor.from_payload(unit.unit_id.encode())
                for unit in (first_unit,)
            ),
            records=2,
            next_cursor=cursors[1],
            status=SourceScanStatus.READY,
        ),
        SourceScanResult(
            source_identity="source",
            view_identity="r",
            schema_digest="schema",
            lineage_digest="r",
            units=(second_unit,),
            unit_cursors=tuple(
                OpaqueSourceCursor.from_payload(unit.unit_id.encode())
                for unit in (second_unit,)
            ),
            records=2,
            next_cursor=cursors[2],
            status=SourceScanStatus.TERMINAL,
        ),
    )
    adapter = InMemorySourceScanAdapter(scan_results)
    initial_cursor_factory = _fixed_initial_cursor(cursors[0])
    planner = IngestionLayoutPlanningComponent(
        "bounded-scan", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    budget = SourceScanBudget(2, 1)
    try:
        first_activation = await planner.plan_source_scan(
            adapter,
            initial_cursor_factory,
            ledger,
            cursors[0],
            budget=budget,
            topology=compiled.request.topology,
            tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=0,
            max_steps=1,
            max_descriptor_bytes=64 * 1024 * 1024,
        )
        assert first_activation is not None
        assert first_activation.epoch.terminal is False
        assert first_activation.epoch.assigned_cardinality == 2
        assert planner.resume_cursor == InMemorySourceScanAdapter.cursor(2)
        second_activation = await planner.plan_source_scan(
            adapter,
            initial_cursor_factory,
            ledger,
            cursors[1],
            budget=budget,
            topology=compiled.request.topology,
            tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
            epoch=1,
            start_position=1,
            max_steps=1,
            max_descriptor_bytes=64 * 1024 * 1024,
        )
        assert second_activation is not None
        assert second_activation.epoch.terminal is True
        assert second_activation.epoch.assigned_cardinality == 2
        assert planner.resume_cursor == InMemorySourceScanAdapter.cursor(4)
        assert planner._pending_units == ()
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_scan_result_planner_idle_preserves_cursor_and_pending_tail(
    source_layout,
) -> None:
    compiled = source_layout(demands=(2,), cardinality=2, terminal=False)
    cursor = OpaqueSourceCursor.from_payload(b"idle")

    class IdleAdapter:
        def close(self) -> None:
            pass

        def scan_next(self, requested, budget):
            assert requested == cursor
            return SourceScanResult(
                source_identity="source",
                view_identity="r",
                schema_digest="schema",
                lineage_digest="r",
                units=(),
                unit_cursors=(),
                records=0,
                next_cursor=cursor,
                status=SourceScanStatus.IDLE,
            )

        def advance_cursor(self, cursor, records):
            raise AssertionError(
                "Scripted scan fixture cannot derive consumption positions"
            )

    planner = IngestionLayoutPlanningComponent(
        "bounded-scan-idle", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    planner._pending_units = compiled.request.units
    try:
        result = await planner.plan_source_scan(
            IdleAdapter(),
            _fixed_initial_cursor(cursor),
            ledger,
            cursor,
            budget=SourceScanBudget(2, 1),
            topology=compiled.request.topology,
            tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=0,
        )
        assert result is None
        assert planner.resume_cursor == cursor
        assert planner._pending_units == compiled.request.units
        assert planner.windows == ()
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_scan_result_planner_allows_empty_continuous_view_transition(
    source_layout,
) -> None:
    compiled = source_layout(demands=(1,), cardinality=1, terminal=False)
    first_cursor = OpaqueSourceCursor.from_payload(b"first")
    second_cursor = OpaqueSourceCursor.from_payload(b"second")
    terminal_cursor = OpaqueSourceCursor.from_payload(b"terminal")
    unit = compiled.request.units[0]

    class ContinuousAdapter:
        def close(self) -> None:
            pass

        def scan_next(self, requested, budget):
            if requested == first_cursor:
                return SourceScanResult(
                    source_identity="source",
                    view_identity="snapshot-1",
                    schema_digest="schema",
                    lineage_digest="snapshot-1",
                    units=(),
                    unit_cursors=(),
                    records=0,
                    next_cursor=second_cursor,
                    status=SourceScanStatus.PROGRESS,
                )
            if requested == second_cursor:
                return SourceScanResult(
                    source_identity="source",
                    view_identity="snapshot-2",
                    schema_digest="schema",
                    lineage_digest="snapshot-2",
                    units=(unit,),
                    unit_cursors=(OpaqueSourceCursor.from_payload(b"unit"),),
                    records=unit.cardinality,
                    next_cursor=terminal_cursor,
                    status=SourceScanStatus.TERMINAL,
                )
            raise AssertionError(f"unexpected scan cursor: {requested!r}")

        def advance_cursor(self, requested, records):
            raise AssertionError("continuous transition fixture has no partial unit")

    planner = IngestionLayoutPlanningComponent(
        "continuous-view-transition", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    try:
        activation = await planner.plan_source_scan(
            ContinuousAdapter(),
            _fixed_initial_cursor(first_cursor),
            ledger,
            first_cursor,
            budget=SourceScanBudget(1, 1),
            topology=compiled.request.topology,
            tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=0,
        )
        assert activation is not None
        descriptor = activation.decoded_worker_layouts[0]
        assert descriptor.view_identity == "snapshot-2"
        assert descriptor.lineage_digest == "snapshot-2"
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_scan_result_planner_rejects_view_transition_with_pending_units(
    source_layout,
) -> None:
    compiled = source_layout(demands=(2,), cardinality=2, terminal=False)
    first_cursor = OpaqueSourceCursor.from_payload(b"first")
    second_cursor = OpaqueSourceCursor.from_payload(b"second")
    third_cursor = OpaqueSourceCursor.from_payload(b"third")
    unit = replace(compiled.request.units[0], cardinality=1)

    class ContinuousAdapter:
        def close(self) -> None:
            pass

        def scan_next(self, requested, budget):
            if requested == first_cursor:
                return SourceScanResult(
                    source_identity="source",
                    view_identity="snapshot-1",
                    schema_digest="schema",
                    lineage_digest="snapshot-1",
                    units=(unit,),
                    unit_cursors=(OpaqueSourceCursor.from_payload(b"unit"),),
                    records=unit.cardinality,
                    next_cursor=second_cursor,
                    status=SourceScanStatus.READY,
                )
            return SourceScanResult(
                source_identity="source",
                view_identity="snapshot-2",
                schema_digest="schema",
                lineage_digest="snapshot-2",
                units=(),
                unit_cursors=(),
                records=0,
                next_cursor=third_cursor,
                status=SourceScanStatus.PROGRESS,
            )

        def advance_cursor(self, requested, records):
            raise AssertionError("continuous transition fixture has no partial unit")

    planner = IngestionLayoutPlanningComponent(
        "continuous-view-transition-pending", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    try:
        with pytest.raises(ValueError, match="identity or schema changed"):
            await planner.plan_source_scan(
                ContinuousAdapter(),
                _fixed_initial_cursor(first_cursor),
                ledger,
                first_cursor,
                budget=SourceScanBudget(2, 1),
                topology=compiled.request.topology,
                tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
                assignment_policy=AssignmentPolicy.default(),
                epoch=0,
                start_position=0,
            )
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_scan_result_planner_rejects_scan_budget_before_publication(
    source_layout,
) -> None:
    compiled = source_layout(demands=(2,), cardinality=2, terminal=True)
    cursor = OpaqueSourceCursor.from_payload((0).to_bytes(8, "big"))
    scan_result = SourceScanResult(
        source_identity="source",
        view_identity="r",
        schema_digest="schema",
        lineage_digest="r",
        units=compiled.request.units,
        unit_cursors=tuple(
            OpaqueSourceCursor.from_payload(unit.unit_id.encode())
            for unit in compiled.request.units
        ),
        records=2,
        next_cursor=OpaqueSourceCursor.from_payload((1).to_bytes(8, "big")),
        status=SourceScanStatus.TERMINAL,
    )

    class OversizedSource(InMemorySourceScanAdapter):
        def scan_next(self, cursor, budget):
            return scan_result

    adapter = OversizedSource((scan_result,))
    planner = IngestionLayoutPlanningComponent(
        "bounded-scan-budget", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    try:
        with pytest.raises(ValueError, match="record budget"):
            await planner.plan_source_scan(
                adapter,
                _fixed_initial_cursor(cursor),
                ledger,
                cursor,
                budget=SourceScanBudget(1, 1),
                topology=compiled.request.topology,
                tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
                assignment_policy=AssignmentPolicy.default(),
                epoch=0,
                start_position=0,
            )
        assert planner.windows == ()
        assert planner.resume_cursor is None
        assert ledger.epochs == ()
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_scan_result_planner_publication_failure_does_not_advance_cursor(
    source_layout,
):
    compiled = source_layout(demands=(2,), cardinality=2, terminal=True)
    cursor = OpaqueSourceCursor.from_payload((0).to_bytes(8, "big"))
    scan_result = SourceScanResult(
        source_identity="source",
        view_identity="r",
        schema_digest="schema",
        lineage_digest="r",
        units=compiled.request.units,
        unit_cursors=tuple(
            OpaqueSourceCursor.from_payload(unit.unit_id.encode())
            for unit in compiled.request.units
        ),
        records=2,
        next_cursor=OpaqueSourceCursor.from_payload((1).to_bytes(8, "big")),
        status=SourceScanStatus.TERMINAL,
    )
    adapter = InMemorySourceScanAdapter((scan_result,))
    planner = IngestionLayoutPlanningComponent(
        "bounded-scan-failure", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)

    def reject(*args, **kwargs):
        raise RuntimeError("publication rejected")

    ledger.prepare_publication = reject
    try:
        with pytest.raises(RuntimeError, match="publication rejected"):
            await planner.plan_source_scan(
                adapter,
                _fixed_initial_cursor(cursor),
                ledger,
                cursor,
                budget=SourceScanBudget(2, 1),
                topology=compiled.request.topology,
                tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
                assignment_policy=AssignmentPolicy.default(),
                epoch=0,
                start_position=0,
            )
        assert planner.windows == ()
        assert planner.resume_cursor is None
        assert planner._pending_units == ()
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_terminal_padding_publishes_one_complete_step(source_layout) -> None:
    compiled = source_layout(demands=(2,), cardinality=2, terminal=True)
    request = replace(
        compiled.request,
        units=(replace(compiled.request.units[0], cardinality=1),),
        tail_layout_mode=TailLayoutMode.SYNTHESIZE_PADDING,
    )
    planner = IngestionLayoutPlanningComponent(
        "terminal-pad", topology=request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)

    try:
        activation = await planner.plan_and_publish(
            ledger,
            request,
            epoch=0,
            start_position=0,
            max_steps=1,
            max_records=1,
            max_descriptor_bytes=64 * 1024 * 1024,
        )

        assert activation.epoch.end_position == 1
        assert activation.epoch.terminal is True
        assert planner._pending_units == ()
        assert planner.windows[0].cardinality == 2
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_terminal_drop_is_applied_by_final_complete_step_window(
    source_layout,
) -> None:
    compiled = source_layout(demands=(2,), cardinality=4, terminal=True)
    request = replace(
        compiled.request,
        units=(replace(compiled.request.units[0], cardinality=3),),
        tail_layout_mode=TailLayoutMode.DISCARD_REMAINDER,
    )
    planner = IngestionLayoutPlanningComponent(
        "terminal-drop", topology=request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)

    try:
        activation = await planner.plan_and_publish(
            ledger,
            request,
            epoch=0,
            start_position=0,
            max_steps=1,
            max_descriptor_bytes=64 * 1024 * 1024,
        )

        assert activation.epoch.end_position == 1
        assert activation.epoch.terminal is True
        assert planner._pending_units == ()
        assert planner.windows[0].discarded_cardinality == 1
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_terminal_require_rejects_tail_before_publication(source_layout) -> None:
    compiled = source_layout(demands=(2,), cardinality=4, terminal=True)
    request = replace(
        compiled.request,
        units=(replace(compiled.request.units[0], cardinality=3),),
        tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
    )
    planner = IngestionLayoutPlanningComponent(
        "terminal-require", topology=request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)

    try:
        with pytest.raises(ValueError, match="incomplete terminal global cardinality"):
            await planner.plan_and_publish(
                ledger,
                request,
                epoch=0,
                start_position=0,
                max_steps=1,
                max_descriptor_bytes=64 * 1024 * 1024,
            )

        assert planner.windows == ()
        assert ledger.epochs == ()
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_scan_result_publication_installs_cursor_with_window(
    source_layout,
) -> None:
    compiled = source_layout(demands=(1,), cardinality=2, terminal=True)
    cursor = OpaqueSourceCursor.from_payload((0).to_bytes(8, "big"))
    next_cursor = OpaqueSourceCursor.from_payload((1).to_bytes(8, "big"))
    scan_result = SourceScanResult(
        source_identity=compiled.request.source_identity,
        view_identity=compiled.request.view_identity,
        schema_digest=compiled.request.schema_digest,
        lineage_digest=compiled.request.lineage_digest,
        units=compiled.request.units,
        unit_cursors=tuple(
            OpaqueSourceCursor.from_payload(unit.unit_id.encode())
            for unit in compiled.request.units
        ),
        records=2,
        next_cursor=next_cursor,
        status=SourceScanStatus.TERMINAL,
    )
    adapter = InMemorySourceScanAdapter((scan_result,))
    planner = IngestionLayoutPlanningComponent(
        "cursor", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    await planner.plan_source_scan(
        adapter,
        _fixed_initial_cursor(cursor),
        ledger,
        cursor,
        budget=SourceScanBudget(
            max_records=2,
            max_units=1,
        ),
        topology=compiled.request.topology,
        tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
        assignment_policy=AssignmentPolicy.default(),
        epoch=0,
        start_position=0,
    )
    image = planner.checkpoint_state(CheckpointCut.for_position("cut", 2))
    assert image["cursor"] == InMemorySourceScanAdapter.cursor(2).to_checkpoint()
    assert image["initial_cursor"] == cursor.to_checkpoint()
    assert (
        image["source_identity"],
        image["view_identity"],
        image["schema_digest"],
        image["lineage_digest"],
    ) == (
        compiled.request.source_identity,
        compiled.request.view_identity,
        compiled.request.schema_digest,
        compiled.request.lineage_digest,
    )
    assert len(planner.windows) == 1
    await planner.close()


@pytest.mark.asyncio
async def test_close_joins_accepted_publication(source_layout) -> None:
    compiled = source_layout(demands=(1,), cardinality=2, terminal=True)

    class GatedPlanner(IngestionLayoutPlanningComponent):
        def __init__(self) -> None:
            super().__init__("close", topology=compiled.request.topology)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def _compile_prepared(self, request, *, epoch, start_position):
            self.entered.set()
            await self.release.wait()
            return compiled

    planner = GatedPlanner()
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    publication = asyncio.create_task(
        planner.plan_and_publish(ledger, compiled.request, epoch=0, start_position=0)
    )
    await planner.entered.wait()
    closing = asyncio.create_task(planner.close())
    await asyncio.sleep(0)
    assert not closing.done()
    planner.release.set()
    await publication
    await closing
    assert len(planner.windows) == 1
    with pytest.raises(RuntimeError, match="closing"):
        await planner.plan_and_publish(
            ledger, compiled.request, epoch=0, start_position=0
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_close_settles_capture_and_publication(source_layout, cancel_waiter):
    compiled = source_layout(demands=(1,), cardinality=1)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    class Adapter:
        def close(self) -> None:
            pass

        def scan_next(self, cursor, budget):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(20):
                raise TimeoutError("test release barrier was abandoned")
            return SourceScanResult(
                source_identity="source",
                view_identity="r",
                schema_digest="schema",
                lineage_digest="r",
                units=compiled.request.units,
                unit_cursors=tuple(
                    InMemorySourceScanAdapter.cursor(0)
                    for unit in compiled.request.units
                ),
                records=sum(unit.cardinality for unit in compiled.request.units),
                next_cursor=OpaqueSourceCursor.from_payload((1).to_bytes(8, "big")),
                status=SourceScanStatus.TERMINAL,
            )

        def advance_cursor(self, cursor, records):
            raise AssertionError(
                "Scripted scan fixture cannot derive consumption positions"
            )

    planner = IngestionLayoutPlanningComponent("capture-close")
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    caller = asyncio.create_task(
        planner.plan_source_scan(
            Adapter(),
            _fixed_initial_cursor(
                OpaqueSourceCursor.from_payload((0).to_bytes(8, "big"))
            ),
            ledger,
            OpaqueSourceCursor.from_payload((0).to_bytes(8, "big")),
            budget=SourceScanBudget(
                max_records=2,
                max_units=1,
            ),
            topology=compiled.request.topology,
            tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=0,
        )
    )
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if cancel_waiter:
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
        closing = asyncio.create_task(planner.close())
        done, _ = await asyncio.wait((closing,), timeout=0.05)
        assert not done
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        release.set()
        if not cancel_waiter:
            await caller
        await asyncio.wait_for(planner.close(), 20)
        image = planner.checkpoint_state(CheckpointCut.for_position("cut", 0))
        assert (
            image["cursor"]
            == OpaqueSourceCursor.from_payload((0).to_bytes(8, "big")).to_checkpoint()
        )
        assert len(planner.windows) == len(ledger.epochs) == 1
        with pytest.raises(RuntimeError, match="closing"):
            await planner.plan_and_publish(
                ledger, compiled.request, epoch=1, start_position=1
            )
        assert not planner._adapter_lane._thread.is_alive()
        assert not planner._lifecycle_lane._thread.is_alive()
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await planner.close()


@pytest.mark.asyncio
async def test_cancelled_scan_result_waiter_failure_is_reported_after_resource_cleanup(
    source_layout,
):
    entered = asyncio.Event()
    compiled = source_layout(demands=(1,), cardinality=1, terminal=True)

    class Adapter:
        def close(self) -> None:
            pass

        async def scan_next(self, cursor, budget):
            loop.call_soon_threadsafe(entered.set)
            await asyncio.wrap_future(released)
            raise ValueError("scan result adapter failed")

        def advance_cursor(self, cursor, records):
            raise AssertionError(
                "Scripted scan fixture cannot derive consumption positions"
            )

    loop = asyncio.get_running_loop()
    released = Future()
    planner = IngestionLayoutPlanningComponent("failure-close")
    caller = asyncio.create_task(
        planner.plan_source_scan(
            Adapter(),
            _fixed_initial_cursor(OpaqueSourceCursor.from_payload(b"start")),
            LogicalWorkLedger(max_in_flight_steps=2),
            OpaqueSourceCursor.from_payload(b"start"),
            budget=SourceScanBudget(2, 1),
            topology=compiled.request.topology,
            tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=0,
        )
    )
    await asyncio.wait_for(entered.wait(), 5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    released.set_result(None)
    with pytest.raises(ExceptionGroup) as caught:
        await planner.close()
    assert len(caught.value.exceptions) == 1
    assert str(caught.value.exceptions[0]) == "scan result adapter failed"
    assert not planner._adapter_lane._thread.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("window_steps", [1, 5, 13])
async def test_planner_bookmark_does_not_depend_on_window_partition(
    source_layout, window_steps
):
    request = source_layout(demands=(1,), cardinality=20, terminal=False).request
    planner = IngestionLayoutPlanningComponent("bookmarks", topology=request.topology)
    ledger = LogicalWorkLedger(max_in_flight_steps=32)
    try:
        await _scan_request(planner, ledger, request, max_steps=window_steps)
        while (
            await planner.plan_next_window(ledger, max_steps=window_steps) is not None
        ):
            pass
        for position in (0, 1, 5, 13, 20):
            state = planner.checkpoint_state(
                CheckpointCut.for_position("cut", position)
            )
            assert (
                state["cursor"]
                == InMemorySourceScanAdapter.cursor(position).to_checkpoint()
            )
            assert set(state) == {
                "cut_position",
                "cursor",
                "initial_cursor",
                "binding_identity",
                "source_identity",
                "view_identity",
                "schema_digest",
                "lineage_digest",
                "global_cardinality",
                "tail_layout_mode",
            }
            assert (
                state["initial_cursor"]
                == InMemorySourceScanAdapter.cursor(0).to_checkpoint()
            )
            assert state["binding_identity"] is None
        planner.reclaim_before(20)
        assert planner._capture.observations == ()
        assert planner.checkpoint_state(CheckpointCut.for_position("cut", 20))[
            "cursor"
        ] == (InMemorySourceScanAdapter.cursor(20).to_checkpoint())
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_direct_compile_cannot_invent_a_source_bookmark(source_layout):
    compiled = source_layout(demands=(1,), cardinality=3, terminal=False)
    planner = IngestionLayoutPlanningComponent(
        "compile", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=8)
    try:
        await planner.plan_and_publish(
            ledger, compiled.request, epoch=0, start_position=0
        )
        with pytest.raises(ValueError, match="source consumption cursor"):
            planner.checkpoint_state(CheckpointCut.for_position("cut", 1))
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_checkpoint_bookmark_is_stable_when_prefetch_advances(source_layout):
    seed = source_layout(demands=(1,), cardinality=1, terminal=False)
    cursors = {
        name: OpaqueSourceCursor.from_payload(name.encode())
        for name in (
            "initial",
            "before-a",
            "after-a",
            "before-b",
            "after-b",
        )
    }

    class Source:
        def close(self) -> None:
            pass

        def scan_next(self, cursor, budget):
            index = 0 if cursor == cursors["initial"] else 1
            assert cursor == cursors["initial" if index == 0 else "after-a"]
            letter = "a" if index == 0 else "b"
            return SourceScanResult(
                source_identity="source",
                view_identity="view",
                schema_digest="schema",
                lineage_digest="lineage",
                units=(
                    replace(seed.request.units[0], unit_id=letter, logical_start=index),
                ),
                unit_cursors=(cursors[f"before-{letter}"],),
                records=1,
                next_cursor=cursors[f"after-{letter}"],
                status=SourceScanStatus.READY,
            )

        def advance_cursor(self, cursor, records):
            raise AssertionError("These cuts lie on record boundaries")

    source = Source()
    planner = IngestionLayoutPlanningComponent("stable", topology=seed.request.topology)
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    cut = CheckpointCut.for_position("cut", 1)
    initial_cursor_factory = _fixed_initial_cursor(cursors["initial"])
    try:

        async def plan(epoch):
            return await planner.plan_source_scan(
                source,
                initial_cursor_factory,
                ledger,
                cursors["initial"],
                budget=SourceScanBudget(1, 1),
                topology=seed.request.topology,
                tail_layout_mode=TailLayoutMode.RETAIN_PENDING,
                assignment_policy=AssignmentPolicy.default(),
                epoch=epoch,
                start_position=epoch,
            )

        await plan(0)
        before = planner.checkpoint_state(cut)
        assert before["cursor"] == cursors["after-a"].to_checkpoint()
        await plan(1)
        assert planner.checkpoint_state(cut) == before
        permit = CheckpointCommitPermit("attempt", 0, b"token")
        prepared = planner.prepare_checkpoint(
            "checkpoint", permit, cut, metadata_digest(before)
        )
        planner.apply_prepared(prepared)
        planner.reclaim_before(1)
        assert planner.checkpoint_state(cut) == before
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_restore_rejects_changed_consumption_contract_before_publication(
    source_layout,
):
    request = source_layout(demands=(2, 2), cardinality=8, terminal=False).request
    original = IngestionLayoutPlanningComponent("contract", topology=request.topology)
    try:
        await _scan_request(original, LogicalWorkLedger(max_in_flight_steps=4), request)
        cut = CheckpointCut.for_position("cut", 1)
        state = original.checkpoint_state(cut)
        different_batch = IngestionLayoutPlanningComponent(
            "contract",
            topology=source_layout(demands=(3,), cardinality=9).request.topology,
        )
        try:
            with pytest.raises(ValueError, match="logical batch"):
                different_batch.restore_state(state, cut)
        finally:
            await different_batch.close()
        for changed in (
            replace(request, source_identity="changed"),
            replace(request, schema_digest="changed"),
            replace(request, tail_layout_mode=TailLayoutMode.REJECT_REMAINDER),
        ):
            restored = IngestionLayoutPlanningComponent(
                "contract", topology=request.topology
            )
            ledger = LogicalWorkLedger(max_in_flight_steps=4)
            ledger.restore_state({"committed_cut": 1})
            restored.restore_state(state, cut)
            try:
                with pytest.raises(ValueError, match="identity or schema|tail policy"):
                    await restored.plan_source_scan(
                        _memory_source(changed),
                        _fixed_initial_cursor(restored.resume_cursor),
                        ledger,
                        restored.resume_cursor,
                        budget=SourceScanBudget(4, 1),
                        topology=request.topology,
                        tail_layout_mode=changed.tail_layout_mode,
                        assignment_policy=AssignmentPolicy.default(),
                        epoch=0,
                        start_position=1,
                    )
                assert restored.windows == ()
                assert ledger.epochs == ()
                assert restored.checkpoint_state(cut) == state
            finally:
                await restored.close()
    finally:
        await original.close()
