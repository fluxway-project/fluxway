import asyncio
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from fluxway.common.ingestion import (
    OpaqueSourceCursor,
    SourceBindingIdentity,
    SourceLayoutActivation,
    SourceScanResult,
    SourceScanStatus,
    SourceWorkAcknowledgement,
    WorkerLayoutDescriptor,
)
from fluxway.common.types import CheckpointCommitPermit, CheckpointCut
from fluxway.common.types.configs import SourceControlConfig
from fluxway.control_plane.controller import (
    ControllerActor,
    _ensure_source_scan_binding,
)
from fluxway.control_plane.ingestion import (
    IngestionLayoutPlanningComponent,
    LayoutUnitDescriptor,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
    TailLayoutMode,
    SourceScanBinding,
)
from fluxway.control_plane.source_control import SourceControlRuntime
from fluxway.control_plane.source_progress import SourceProgressSupervisor


def test_source_control_rejects_unknown_window_size() -> None:
    with pytest.raises(ValidationError, match="window_size"):
        SourceControlConfig(window_size=3)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_source_control_planning_window_steps_is_strictly_positive(value) -> None:
    with pytest.raises(ValidationError, match="planning_window_steps"):
        SourceControlConfig(planning_window_steps=value)


def test_source_control_planning_window_steps_has_a_validated_default() -> None:
    assert SourceControlConfig().planning_window_steps > 0


@pytest.mark.parametrize(
    "field_name",
    [
        "max_in_flight_steps",
        "max_planning_records",
        "max_planning_units",
        "max_planning_descriptor_bytes",
    ],
)
def test_source_control_limits_reject_boolean_values(field_name) -> None:
    with pytest.raises(ValidationError, match=field_name):
        SourceControlConfig(**{field_name: True})


@pytest.mark.asyncio
async def test_runtime_publishes_and_acknowledges_logical_work(source_layout) -> None:
    runtime = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=2))
    compiled = source_layout(cardinality=4)
    bindings = tuple(
        WorkerLayoutDescriptor.decode(value)
        for value in compiled.encoded_worker_layouts()
    )
    await runtime.publish_work_epoch(compiled.work_epoch, bindings)
    leases = await runtime.acquire_work(
        0, tuple(segment for binding in bindings for segment in binding.segments)
    )
    await runtime.acknowledge_work(
        tuple(
            SourceWorkAcknowledgement(lease.item.key, lease.lease_id)
            for lease in leases
        )
    )
    assert await runtime.candidate_work_cut() == 1


@pytest.mark.asyncio
async def test_runtime_waits_for_logical_cut_without_polling(source_layout) -> None:
    runtime = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=2))
    compiled = source_layout(cardinality=4)
    bindings = tuple(
        WorkerLayoutDescriptor.decode(value)
        for value in compiled.encoded_worker_layouts()
    )
    await runtime.publish_work_epoch(compiled.work_epoch, bindings)
    waiting = asyncio.create_task(runtime.wait_for_work_cut(1))
    await asyncio.sleep(0)
    assert not waiting.done()
    leases = await runtime.acquire_work(
        0, tuple(segment for binding in bindings for segment in binding.segments)
    )
    await runtime.acknowledge_work(
        tuple(
            SourceWorkAcknowledgement(lease.item.key, lease.lease_id)
            for lease in leases
        )
    )
    assert await waiting == 1


def test_runtime_checkpoint_is_logical_work_state() -> None:
    runtime = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=2))
    state = runtime.checkpoint_state(CheckpointCut.for_position("cut", 0))
    assert set(state) == {
        "committed_cut",
    }


def test_runtime_scan_budget_is_derived_from_one_config_owner() -> None:
    config = SourceControlConfig(
        planning_window_steps=7,
        max_planning_records=13,
        max_planning_units=5,
    )
    runtime = SourceControlRuntime(config)

    assert runtime.planning_window_steps == 7
    assert runtime.scan_budget.max_records == 13
    assert runtime.scan_budget.max_units == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("restore_position", [None, 7])
async def test_controller_first_activation_uses_scan_result_capabilities(
    restore_position,
) -> None:
    modified_controller = ControllerActor.__ray_metadata__.modified_class
    topology = ReaderTopologyDescriptor(
        2, (ReaderGroupDemand("reader:read:0", (RankDemand("rank", 2),)),)
    )
    unit = LayoutUnitDescriptor(
        "unit",
        "revision",
        2,
        "rows",
        "SplittableSpan",
        "node",
        "file",
        0,
        b"metadata",
    )
    cursor = OpaqueSourceCursor.from_payload(b"initial")
    next_cursor = OpaqueSourceCursor.from_payload(b"terminal")

    class ScanAdapter:
        def close(self) -> None:
            pass

        def scan_next(self, requested, budget):
            assert requested == cursor
            return SourceScanResult(
                source_identity="source",
                view_identity="view",
                schema_digest="schema",
                lineage_digest="lineage",
                units=(unit,),
                unit_cursors=tuple(
                    OpaqueSourceCursor.from_payload(unit.unit_id.encode())
                    for unit in (unit,)
                ),
                records=2,
                next_cursor=next_cursor,
                status=SourceScanStatus.TERMINAL,
            )

        def advance_cursor(self, cursor, records):
            raise AssertionError(
                "Scripted scan fixture cannot derive consumption positions"
            )

    class SourceDescriptor:
        requires_source_control_runtime = True

        def _build_reader_topology(self, **kwargs):
            return topology

        def _build_source_scan_adapter(self, **kwargs):
            return ScanAdapter()

        def _build_source_scan_tail_layout(self, **kwargs):
            return TailLayoutMode.DISCARD_REMAINDER

        def build_source_scan_binding(self, **kwargs):
            adapter = self._build_source_scan_adapter(**kwargs)

            def initial_cursor_factory():
                return cursor

            return SourceScanBinding(
                adapter=adapter,
                identity=SourceBindingIdentity(
                    configuration_identity="config",
                    source_identity="source",
                    mode_identity="finite",
                    native_identity="native",
                ),
                initial_cursor_factory=initial_cursor_factory,
                topology=self._build_reader_topology(**kwargs),
                tail_layout_mode=self._build_source_scan_tail_layout(**kwargs),
            )

    config = SimpleNamespace(
        operator_specs={"read": SimpleNamespace(source=SourceDescriptor())}
    )
    runtime = SourceControlRuntime(
        SourceControlConfig(
            planning_window_steps=1,
            max_planning_records=2,
            max_planning_units=1,
        )
    )
    planner = IngestionLayoutPlanningComponent(
        "controller-scan result", topology=topology
    )
    dispatched: list[SourceLayoutActivation] = []

    async def dispatch(*args):
        dispatched.append(args[-1])

    if restore_position is not None:
        cut = CheckpointCut.for_position("restore", restore_position)
        runtime.restore_state({"committed_cut": restore_position}, cut)
        planner.restore_state(
            {
                "cut_position": restore_position,
                "cursor": cursor.to_checkpoint(),
                "initial_cursor": cursor.to_checkpoint(),
                "binding_identity": SourceBindingIdentity(
                    configuration_identity="config",
                    source_identity="source",
                    mode_identity="finite",
                    native_identity="native",
                ).to_checkpoint(),
                "source_identity": "source",
                "view_identity": "view",
                "schema_digest": "schema",
                "lineage_digest": "lineage",
                "global_cardinality": 2,
                "tail_layout_mode": TailLayoutMode.DISCARD_REMAINDER.value,
            },
            cut,
        )
    controller = SimpleNamespace(
        _source_layout_target=lambda: ("reader", "read", config),
        _require_ingestion_planner=lambda: planner,
        _require_source_control=lambda: runtime,
        _source_window_activation_lock=asyncio.Lock(),
        dispatch_driver_component=dispatch,
    )
    try:
        await modified_controller.activate_source_scan_layout_from_descriptor(
            controller
        )
        assert len(planner.windows) == 1
        assert planner.windows[0].work_epoch.start_step == (restore_position or 0)
        assert runtime.work_ledger.epochs
        assert len(dispatched) == 1
        assert isinstance(dispatched[0], SourceLayoutActivation)
        assert dispatched[0].epoch.terminal is True
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_controller_reuses_binding_identity_across_activation(monkeypatch):
    modified_controller = ControllerActor.__ray_metadata__.modified_class
    topology = ReaderTopologyDescriptor(
        1, (ReaderGroupDemand("reader:read:0", (RankDemand("rank", 1),)),)
    )
    unit = LayoutUnitDescriptor(
        "unit", "revision", 1, "rows", "AtomicItem", "node", "file", 0, b"m"
    )
    cursor = OpaqueSourceCursor.from_payload(b"initial")
    result = SourceScanResult(
        source_identity="source",
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        units=(unit,),
        unit_cursors=(OpaqueSourceCursor.from_payload(b"unit"),),
        records=1,
        next_cursor=OpaqueSourceCursor.from_payload(b"terminal"),
        status=SourceScanStatus.TERMINAL,
    )

    class Adapter:
        def close(self) -> None:
            pass

        def scan_next(self, requested, budget):
            return result

        def advance_cursor(self, requested, records):
            raise AssertionError("not needed")

    class Descriptor:
        requires_source_control_runtime = True
        calls = 0

        def build_source_scan_binding(self, **kwargs):
            type(self).calls += 1
            adapter = Adapter()

            def initial_cursor_factory():
                return cursor

            return SourceScanBinding(
                adapter=adapter,
                identity=SourceBindingIdentity("config", "source", "finite", "native"),
                initial_cursor_factory=initial_cursor_factory,
                topology=topology,
                tail_layout_mode=TailLayoutMode.DISCARD_REMAINDER,
            )

    descriptor = Descriptor()
    config = SimpleNamespace(
        operator_specs={"read": SimpleNamespace(source=descriptor)}
    )
    runtime = SourceControlRuntime(
        SourceControlConfig(max_planning_records=1, max_planning_units=1)
    )
    planner = IngestionLayoutPlanningComponent("binding-once", topology=topology)

    async def dispatch(*args):
        return None

    controller = SimpleNamespace(
        _source_layout_target=lambda: ("reader", "read", config),
        _require_ingestion_planner=lambda: planner,
        _require_source_control=lambda: runtime,
        _source_window_activation_lock=asyncio.Lock(),
        dispatch_driver_component=dispatch,
    )
    try:
        await modified_controller.activate_source_scan_layout_from_descriptor(
            controller
        )
        await modified_controller.activate_source_scan_layout_from_descriptor(
            controller
        )
        assert descriptor.calls == 1
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_controller_retries_empty_initial_source_without_fabricating_cursor():
    modified_controller = ControllerActor.__ray_metadata__.modified_class
    topology = ReaderTopologyDescriptor(
        1, (ReaderGroupDemand("reader:read:0", (RankDemand("rank", 1),)),)
    )
    unit = LayoutUnitDescriptor(
        "unit", "revision", 1, "rows", "AtomicItem", "node", "file", 0, b"m"
    )
    cursor = OpaqueSourceCursor.from_payload(b"initial")
    result = SourceScanResult(
        source_identity="source",
        view_identity="view",
        schema_digest="schema",
        lineage_digest="lineage",
        units=(unit,),
        unit_cursors=(OpaqueSourceCursor.from_payload(b"unit"),),
        records=1,
        next_cursor=OpaqueSourceCursor.from_payload(b"terminal"),
        status=SourceScanStatus.TERMINAL,
    )

    class Adapter:
        def close(self) -> None:
            pass

        def scan_next(self, requested, budget):
            assert requested == cursor
            return result

        def advance_cursor(self, requested, records):
            raise AssertionError("not needed")

    class Descriptor:
        requires_source_control_runtime = True
        initial_calls = 0

        def build_source_scan_binding(self, **kwargs):
            adapter = Adapter()

            def initial_cursor_factory():
                type(self).initial_calls += 1
                return None if type(self).initial_calls == 1 else cursor

            return SourceScanBinding(
                adapter=adapter,
                identity=SourceBindingIdentity(
                    "config", "source", "continuous", "native"
                ),
                initial_cursor_factory=initial_cursor_factory,
                topology=topology,
                tail_layout_mode=TailLayoutMode.DISCARD_REMAINDER,
            )

    descriptor = Descriptor()
    config = SimpleNamespace(
        operator_specs={"read": SimpleNamespace(source=descriptor)}
    )
    runtime = SourceControlRuntime(
        SourceControlConfig(max_planning_records=1, max_planning_units=1)
    )
    planner = IngestionLayoutPlanningComponent("empty-initial", topology=topology)
    dispatched: list[SourceLayoutActivation] = []

    async def dispatch(*args):
        dispatched.append(args[-1])

    controller = SimpleNamespace(
        _source_layout_target=lambda: ("reader", "read", config),
        _require_ingestion_planner=lambda: planner,
        _require_source_control=lambda: runtime,
        _source_window_activation_lock=asyncio.Lock(),
        dispatch_driver_component=dispatch,
    )
    try:
        await modified_controller.activate_source_scan_layout_from_descriptor(
            controller
        )
        assert planner.windows == ()
        assert planner.resume_cursor is None
        await modified_controller.activate_source_scan_layout_from_descriptor(
            controller
        )
        assert len(planner.windows) == 1
        assert len(dispatched) == 1
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_controller_binding_helper_serializes_construction():
    topology = ReaderTopologyDescriptor(
        1, (ReaderGroupDemand("reader:read:0", (RankDemand("rank", 1),)),)
    )
    cursor = OpaqueSourceCursor.from_payload(b"initial")

    class Adapter:
        def close(self) -> None:
            pass

        def scan_next(self, requested, budget):
            raise AssertionError("scan is not part of binding construction")

        def advance_cursor(self, requested, records):
            raise AssertionError(
                "cursor advancement is not part of binding construction"
            )

    class Descriptor:
        calls = 0

        def build_source_scan_binding(self, **kwargs):
            type(self).calls += 1
            time.sleep(0.05)
            adapter = Adapter()

            def initial_cursor_factory():
                return cursor

            return SourceScanBinding(
                adapter=adapter,
                identity=SourceBindingIdentity("config", "source", "finite", "native"),
                initial_cursor_factory=initial_cursor_factory,
                topology=topology,
                tail_layout_mode=TailLayoutMode.DISCARD_REMAINDER,
            )

    descriptor = Descriptor()
    config = SimpleNamespace(
        operator_specs={"read": SimpleNamespace(source=descriptor)}
    )
    planner = IngestionLayoutPlanningComponent("binding-construction")
    owner = SimpleNamespace(_require_ingestion_planner=lambda: planner)
    first, second = await asyncio.gather(
        _ensure_source_scan_binding(
            owner, node_id="reader", operator_name="read", config=config
        ),
        _ensure_source_scan_binding(
            owner, node_id="reader", operator_name="read", config=config
        ),
    )

    assert first is second
    assert descriptor.calls == 1
    await planner.close()


@pytest.mark.asyncio
async def test_source_progress_supervisor_serializes_wakeups():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def progress() -> None:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()

    supervisor = SourceProgressSupervisor(progress)
    try:
        await supervisor.start()
        await asyncio.wait_for(entered.wait(), timeout=1)
        supervisor.wake()
        supervisor.wake()
        await asyncio.sleep(0)
        assert calls == 1
        release.set()
    finally:
        await supervisor.stop()


@pytest.mark.asyncio
async def test_source_progress_supervisor_propagates_failure():
    async def progress() -> None:
        raise ValueError("planner failed")

    supervisor = SourceProgressSupervisor(progress)
    await supervisor.start()
    with pytest.raises(RuntimeError, match="source progress supervisor failed"):
        await asyncio.sleep(0.02)
        await supervisor.raise_if_failed()


def test_runtime_checkpoint_transaction_uses_logical_cut() -> None:
    runtime = SourceControlRuntime(SourceControlConfig(max_in_flight_steps=2))
    permit = CheckpointCommitPermit("attempt", 0, b"token")
    with pytest.raises(ValueError, match="digest"):
        runtime.prepare_checkpoint(
            "checkpoint", permit, CheckpointCut.for_position("cut", 0), "bad"
        )
