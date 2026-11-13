"""
Unit tests for SPMDSynchronizer drain-to-count barrier and Worker drain target.

Design:
- Workers are self-blocked at drain_step_count by Worker._drain_resume_event (set_drain_boundary RPC).
- Workers therefore only ever REPORT completion up to step_count.
- The Synchronizer's job is to count those ACKs and fire the on_drained callback.
- Synchronizer no longer manages blocking for drain (that's the Worker's responsibility).

Tests are split into:
- TestDrainSetup: setup/teardown invariants
- TestDrainAckCallback: ACK counting and callback triggering
- TestDrainAndAllowedStepIndependence: drain ACK must not be affected by allowed_step
- TestWorkerDrainTarget: Worker.set/clear_drain_boundary state correctness
"""

import asyncio
from itertools import permutations
import threading
import types
import typing

import pytest
from fluxway.common.types import (
    DrainBoundary,
    DrainEvent,
    DrainEventKind,
    DrainPhase,
    reduce_drain_boundary,
)
from fluxway.spmd.driver.synchronizer import (
    MaxRankStepSkewSyncPolicy,
    SPMDSynchronizer,
)


def _make_sync(num_workers: int = 2) -> SPMDSynchronizer:
    return SPMDSynchronizer(
        spmd_id="test_spmd",
        num_workers=num_workers,
        sync_policy=MaxRankStepSkewSyncPolicy(
            max_rank_step_skew=100
        ),  # disable policy blocking
    )


def _register_op(sync: SPMDSynchronizer, op_name: str, step: int = 0) -> None:
    """Simulate workers reporting completions so the op appears in _spmd_steps."""
    for rank in range(sync.num_workers):
        sync.report_worker_op_completed(rank, op_name, step)


class TestDrainSetup:
    def test_double_drain_raises(self):
        """Calling drain_to_step_count while already draining must raise RuntimeError."""
        sync = _make_sync()
        _register_op(sync, "op_a", step=0)
        sync.drain_to_step_count(step_count=5, on_drained=lambda: None)
        with pytest.raises(RuntimeError, match="already draining"):
            sync.drain_to_step_count(step_count=10, on_drained=lambda: None)


def _reduce_events(events):
    boundary = DrainBoundary()
    dispatches = []
    for event in events:
        transition = reduce_drain_boundary(boundary, event)
        boundary = transition.boundary
        if transition.dispatch_target is not None:
            dispatches.append(transition.dispatch_target)
    return boundary, dispatches


def test_boundary_observation_order_is_idempotent_and_minimum_wins() -> None:
    events = (
        DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 14),
        DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 13),
        DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 15),
        DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 13),
    )
    outcomes = {
        _reduce_events(order)[0].observed_cut for order in permutations(events[:2])
    }
    assert outcomes == {13}
    boundary, dispatches = _reduce_events(events)
    assert boundary.observed_cut == 13
    assert boundary.admitted_cut == 13
    assert dispatches == [14, 13]


def test_boundary_rejects_durable_rollback_after_commit() -> None:
    boundary, _ = _reduce_events(
        (
            DrainEvent(DrainEventKind.PERIODIC_REQUEST, 20),
            DrainEvent(DrainEventKind.DRAIN_COMPLETE, 20),
            DrainEvent(DrainEventKind.SNAPSHOT_BEGIN, 20),
            DrainEvent(DrainEventKind.CHECKPOINT_COMMITTED, 20),
        )
    )
    assert boundary.committed_cut == 20
    with pytest.raises(ValueError, match="published checkpoint"):
        reduce_drain_boundary(
            boundary, DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 13)
        )


def test_boundary_restore_terminal_state_is_completed_and_clamps_late_report() -> None:
    boundary = DrainBoundary(
        phase=DrainPhase.COMPLETED,
        observed_cut=13,
        admitted_cut=13,
        drained_cut=13,
        committed_cut=13,
        revision=7,
    )
    restored = DrainBoundary.from_mapping(boundary.to_dict())
    assert restored.phase is DrainPhase.COMPLETED
    assert restored.committed_cut == 13
    assert (
        reduce_drain_boundary(
            boundary, DrainEvent(DrainEventKind.LATE_COMPLETION, 15)
        ).clamped_completion
        == 13
    )


def test_restore_projection_discards_transient_drain_state() -> None:
    historical = DrainBoundary(
        phase=DrainPhase.COMPLETED,
        observed_cut=9,
        admitted_cut=9,
        drained_cut=9,
        committed_cut=9,
        revision=41,
    )

    restored = DrainBoundary.from_committed_cut(historical.committed_cut)

    assert restored == DrainBoundary(
        phase=DrainPhase.OPEN,
        drained_cut=9,
        committed_cut=9,
    )
    transition = reduce_drain_boundary(
        restored, DrainEvent(DrainEventKind.LATE_COMPLETION, 10)
    )
    assert transition.clamped_completion == 10

    def test_is_draining_property(self):
        sync = _make_sync()
        _register_op(sync, "op_a", step=0)

        assert not sync.is_draining
        sync.drain_to_step_count(step_count=5, on_drained=lambda: None)
        assert sync.is_draining
        sync.clear_drain()
        assert not sync.is_draining

    def test_clear_drain_idempotent(self):
        """Calling clear_drain when not draining must not raise."""
        sync = _make_sync()
        sync.clear_drain()

    def test_shorten_drain_recomputes_existing_acknowledgements(self):
        fired = threading.Event()
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=3)
        sync.drain_to_step_count(step_count=5, on_drained=fired.set)

        sync.shorten_drain_to_step_count(3)

        assert fired.is_set()

    def test_shorten_drain_rejects_target_extension(self):
        sync = _make_sync(num_workers=1)
        _register_op(sync, "op_a", step=0)
        sync.drain_to_step_count(step_count=3, on_drained=lambda: None)

        with pytest.raises(ValueError, match="only shrink"):
            sync.shorten_drain_to_step_count(4)
        sync.clear_drain()


class TestDrainAckCallback:
    """
    Workers are self-blocked at step_count, so they report exactly step_count.
    The callback fires when all workers have reported step_count.
    """

    def test_callback_fires_at_step_count_single_op(self):
        fired = threading.Event()
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=0)
        fired.clear()
        sync.drain_to_step_count(step_count=5, on_drained=fired.set)

        sync.report_worker_op_completed(0, "op_a", 5)
        assert not fired.is_set(), "Callback should not fire until all workers ACK"
        sync.report_worker_op_completed(1, "op_a", 5)
        assert fired.is_set(), "Callback must fire when all workers reach step_count"

    def test_callback_fires_with_multiple_ops(self):
        """All ops must be ACKed (at step_count) before on_drained fires."""
        fired = threading.Event()
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=0)
        _register_op(sync, "op_b", step=0)
        fired.clear()
        sync.drain_to_step_count(step_count=3, on_drained=fired.set)

        sync.report_worker_op_completed(0, "op_a", 3)
        sync.report_worker_op_completed(1, "op_a", 3)
        assert not fired.is_set(), "op_b not yet drained"

        sync.report_worker_op_completed(0, "op_b", 3)
        sync.report_worker_op_completed(1, "op_b", 3)
        assert fired.is_set()

    def test_registered_op_without_prior_completion_blocks_drain(self):
        sync = _make_sync(num_workers=1)
        sync.register_expected_ops(["lookup", "update"])
        sync.report_worker_op_completed(0, "lookup", 2)

        fired = threading.Event()
        sync.drain_to_step_count(step_count=2, on_drained=fired.set)
        assert not fired.is_set()

        sync.report_worker_op_completed(0, "update", 1)
        assert not fired.is_set()

        sync.report_worker_op_completed(0, "update", 2)
        assert fired.is_set()

    def test_partial_worker_completion_does_not_fire_early(self):
        fired = threading.Event()
        sync = _make_sync(num_workers=3)
        _register_op(sync, "op_a", step=0)
        fired.clear()
        sync.drain_to_step_count(step_count=2, on_drained=fired.set)

        # step_count = 2; only 2 of 3 workers reported
        sync.report_worker_op_completed(0, "op_a", 2)
        sync.report_worker_op_completed(1, "op_a", 2)
        assert not fired.is_set()

        sync.report_worker_op_completed(2, "op_a", 2)
        assert fired.is_set()


class TestDrainAndAllowedStepIndependence:
    """
    drain_step_count (ACK fence) and update_allowed_step (flow-control ceiling)
    must not interfere with each other.
    """

    def test_allowed_step_does_not_block_drain_ack(self):
        """A restrictive allowed_step must not prevent the drain callback from firing."""
        fired = threading.Event()
        sync = _make_sync(num_workers=2)
        sync.set_callbacks(
            block_worker_op=lambda r, op, version: None,
            unblock_worker_op=lambda r, op, version: None,
        )
        _register_op(sync, "op_a", step=0)
        sync.update_allowed_step("op_a", 3)
        sync.drain_to_step_count(step_count=5, on_drained=fired.set)

        # Workers reach step_count=5; drain ACK must fire regardless of allowed_step=3
        sync.report_worker_op_completed(0, "op_a", 5)
        sync.report_worker_op_completed(1, "op_a", 5)
        assert fired.is_set(), "drain ACK must fire even when allowed_step is set"

    def test_drain_clear_does_not_reset_allowed_step(self):
        """clear_drain must leave update_allowed_step unchanged."""
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=0)
        sync.update_allowed_step("op_a", 10)
        sync.drain_to_step_count(step_count=5, on_drained=lambda: None)
        sync.clear_drain()
        assert sync._allowed_steps.get("op_a") == 10


@pytest.mark.parametrize("observations", [(14, 13), (13, 14)])
def test_terminal_cut_converges_physical_rank_observations(
    observations: tuple[int, int],
) -> None:
    sync = _make_sync(num_workers=2)
    sync.register_expected_ops(("read", "update"))
    for rank, count in enumerate(observations):
        sync.report_worker_op_completed(rank, "read", count)
        sync.report_worker_op_completed(rank, "update", count)

    sync.apply_boundary_event(DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 13))

    assert sync.get_spmd_steps() == {"read": 13, "update": 13}
    assert sync._worker_completed_count == {
        0: {"read": observations[0], "update": observations[0]},
        1: {"read": observations[1], "update": observations[1]},
    }


def test_terminal_cut_shortens_active_drain_and_clamps_late_reports() -> None:
    fired = threading.Event()
    sync = _make_sync(num_workers=2)
    sync.register_expected_ops(("read", "update"))
    sync.report_worker_op_completed(0, "read", 14)
    sync.report_worker_op_completed(1, "read", 13)
    sync.report_worker_op_completed(0, "update", 14)
    sync.report_worker_op_completed(1, "update", 13)
    sync.drain_to_step_count(20, on_drained=fired.set)

    sync.apply_boundary_event(DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 13))
    assert not fired.is_set()
    sync.shorten_drain_to_step_count(13)
    assert fired.is_set()

    sync.report_worker_op_completed(0, "read", 15)
    sync.report_worker_op_completed(0, "update", 15)
    assert sync.get_spmd_steps() == {"read": 13, "update": 13}


def test_late_completion_keeps_raw_cursor_and_clamps_report_view() -> None:
    reports: list[int] = []
    sync = _make_sync(num_workers=2)
    sync.register_expected_op("read")
    sync.set_callbacks(
        on_spmd_op_completed=lambda _spmd, _op, count: reports.append(count)
    )
    sync.report_worker_op_completed(0, "read", 14)
    sync.report_worker_op_completed(1, "read", 14)
    sync.apply_boundary_event(DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 13))
    reports.clear()

    sync.report_worker_op_completed(0, "read", 15)
    sync.report_worker_op_completed(1, "read", 15)

    assert sync.get_all_worker_steps() == {0: {"read": 15}, 1: {"read": 15}}
    assert sync.get_spmd_steps() == {"read": 13}
    assert reports == []


def test_terminal_cut_after_completed_drain_is_monotone() -> None:
    sync = _make_sync(num_workers=2)
    sync.register_expected_ops(("read",))
    _register_op(sync, "read", step=20)
    fired = threading.Event()
    sync.drain_to_step_count(20, on_drained=fired.set)
    assert fired.is_set()
    sync.clear_drain()

    sync.apply_boundary_event(DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 13))
    sync.apply_boundary_event(DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 15))
    assert sync.get_spmd_steps() == {"read": 13}


def test_terminal_cut_after_drain_ack_rearms_shortened_completion() -> None:
    callbacks: list[int] = []
    sync = _make_sync(num_workers=2)
    sync.register_expected_ops(("read",))
    _register_op(sync, "read", step=0)
    sync.drain_to_step_count(20, on_drained=lambda: callbacks.append(20))
    sync.report_worker_op_completed(0, "read", 20)
    sync.report_worker_op_completed(1, "read", 20)
    assert callbacks == [20]

    sync.apply_boundary_event(DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 13))
    sync.shorten_drain_to_step_count(13, on_drained=lambda: callbacks.append(13))
    assert callbacks == [20, 13]


def test_terminal_shrink_invalidates_old_drained_cut() -> None:
    boundary, _ = _reduce_events(
        (
            DrainEvent(DrainEventKind.PERIODIC_REQUEST, 20),
            DrainEvent(DrainEventKind.DRAIN_COMPLETE, 20),
            DrainEvent(DrainEventKind.SNAPSHOT_BEGIN, 20),
            DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 13),
        )
    )
    assert boundary.phase is DrainPhase.DRAINING
    assert boundary.admitted_cut == 13
    assert boundary.drained_cut is None


def test_zero_terminal_cut_converges_all_virtual_source_lanes() -> None:
    syncs = [
        _make_sync(num_workers=2),
        _make_sync(num_workers=2),
    ]
    for sync in syncs:
        sync.register_expected_ops(("read", "update"))
        _register_op(sync, "read", step=4)
        _register_op(sync, "update", step=4)
        sync.apply_boundary_event(DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, 0))
        assert sync.get_spmd_steps() == {"read": 0, "update": 0}

    def test_restore_does_not_create_credit_ceiling_for_uncontrolled_op(self):
        sync = _make_sync(num_workers=1)
        blocked = []
        sync.set_callbacks(
            block_worker_op=lambda rank, op, epoch: blocked.append((rank, op, epoch)),
            unblock_worker_op=lambda rank, op, epoch: None,
        )

        sync.load_spmd_state(
            {
                "drain_boundary": DrainBoundary(committed_cut=2).to_dict(),
            }
        )
        sync.report_worker_op_completed(0, "train_core", 3)

        assert "train_core" not in sync._allowed_steps
        assert blocked == []

    def test_restore_raises_existing_credit_to_restored_count(self):
        sync = _make_sync(num_workers=1)
        sync.update_allowed_step("controlled", 1)

        sync.load_spmd_state(
            {
                "drain_boundary": DrainBoundary(committed_cut=2).to_dict(),
            }
        )

        assert sync._allowed_steps == {"controlled": 2}

    def test_restore_discards_transient_drain_boundary_state(self):
        sync = _make_sync(num_workers=1)
        sync.register_expected_ops(("controlled",))
        sync.load_spmd_state(
            {
                "drain_boundary": DrainBoundary(
                    phase=DrainPhase.COMPLETED,
                    observed_cut=9,
                    admitted_cut=9,
                    drained_cut=9,
                    committed_cut=9,
                    revision=41,
                ).to_dict(),
            }
        )

        assert sync.drain_boundary == DrainBoundary(
            phase=DrainPhase.OPEN,
            drained_cut=9,
            committed_cut=9,
        )
        sync.report_worker_op_completed(0, "controlled", 10)
        assert sync.get_spmd_steps()["controlled"] == 10


class TestWorkerDrainTarget:
    """
    Verify SPMDWorker.set_drain_boundary / clear_drain_boundary state correctness.
    """

    def _make_stub(self):
        from fluxway.spmd.worker.worker import SPMDWorker as _ActorCls
        from fluxway.spmd.worker.worker import LocalInputPrepareRuntime

        raw_cls: typing.Any = getattr(_ActorCls, "__ray_actor_class__", _ActorCls)
        stub = types.SimpleNamespace(
            _drain_step_count=None,
            _drain_cond=asyncio.Condition(),
            _running=True,
            _running_plan_steps={},
            _background_tasks=set(),
            spmd_id="test_spmd",
            rank=0,
            _drain_boundary=DrainBoundary(),
            _source_operator_names=(),
        )
        stub._input_prepare_runtime = LocalInputPrepareRuntime(
            background_tasks=stub._background_tasks,
            on_task_done=lambda _task: None,
            schedule_key=lambda op_name, step: (0, 0, step, op_name),
        )
        stub.set_drain_boundary = types.MethodType(raw_cls.set_drain_boundary, stub)
        stub._set_drain_boundary = types.MethodType(raw_cls._set_drain_boundary, stub)
        stub._drain_blocks_step = types.MethodType(raw_cls._drain_blocks_step, stub)
        stub.op_manager = types.SimpleNamespace(
            get_operator_spec=lambda _op: types.SimpleNamespace(
                is_source=False,
                is_interleaved=False,
                has_source_descriptor=False,
            )
        )
        stub.clear_drain_boundary = types.MethodType(raw_cls.clear_drain_boundary, stub)
        return stub

    @pytest.mark.asyncio
    async def test_set_drain_target_updates_state(self):
        w = self._make_stub()
        await w.set_drain_boundary(42)
        assert w._drain_step_count == 42

    @pytest.mark.asyncio
    async def test_clear_drain_resets_state(self):
        w = self._make_stub()
        await w.set_drain_boundary(42)
        await w.clear_drain_boundary()
        assert w._drain_step_count is None

    @pytest.mark.asyncio
    async def test_set_overrides_previous_target(self):
        w = self._make_stub()
        await w.set_drain_boundary(10)
        await w.set_drain_boundary(20)
        assert w._drain_step_count == 20

    @pytest.mark.asyncio
    async def test_clear_is_idempotent(self):
        w = self._make_stub()
        await w.clear_drain_boundary()
        assert w._drain_step_count is None


class TestDrainEdgeCases:
    """
    Edge-case coverage for drain_to_step_count instant-check logic.

    When no ops have been registered yet (known_ops is empty), drain must NOT
    fire immediately because there is no evidence that any work has completed.
    Firing prematurely would produce a vacuous snapshot with no real state.
    """

    def test_drain_does_not_fire_when_no_ops_registered(self):
        """drain_to_step_count with zero registered ops must NOT fire on the spot."""
        sync = _make_sync(num_workers=2)
        fired = threading.Event()
        # No _register_op call — known_ops is empty
        sync.drain_to_step_count(step_count=0, on_drained=fired.set)
        assert not fired.is_set(), (
            "drain must not fire when no ops have been registered: "
            "an empty snapshot would be produced with no actual state"
        )

    def test_drain_fires_immediately_when_already_at_target(self):
        """drain_to_step_count fires immediately if all workers already satisfy the target."""
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=5)  # both workers at count=5
        fired = threading.Event()
        sync.drain_to_step_count(step_count=3, on_drained=fired.set)
        assert fired.is_set(), (
            "drain must fire immediately when workers already exceed step_count"
        )

    def test_drain_does_not_fire_when_workers_below_target(self):
        """drain_to_step_count must NOT fire if workers are below step_count."""
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=2)  # both workers at count=2
        fired = threading.Event()
        sync.drain_to_step_count(step_count=5, on_drained=fired.set)
        assert not fired.is_set(), (
            "drain must not fire when workers have not reached step_count"
        )
