"""
Unit tests for SPMDSynchronizer drain-to-count barrier and Worker drain target.

Design:
- Workers are self-blocked at drain_target_count by Worker._drain_resume_event (set_drain_target_count RPC).
- Workers therefore only ever REPORT completion up to target_count.
- The Synchronizer's job is to count those ACKs and fire the on_drained callback.
- Synchronizer no longer manages blocking for drain (that's the Worker's responsibility).

Tests are split into:
- TestDrainSetup: setup/teardown invariants
- TestDrainAckCallback: ACK counting and callback triggering
- TestDrainAndAllowedStepIndependence: drain ACK must not be affected by allowed_step
- TestWorkerDrainTarget: Worker.set/clear_drain_target_count state correctness
"""

import asyncio
import threading
import types

import pytest
from fluxway.spmd.driver.synchronizer import SPMDSynchronizer, MaxSkewSyncPolicy


def _make_sync(num_workers: int = 2) -> SPMDSynchronizer:
    return SPMDSynchronizer(
        spmd_id="test_spmd",
        num_workers=num_workers,
        sync_policy=MaxSkewSyncPolicy(max_skew=100),  # disable policy blocking
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
        sync.drain_to_step_count(target_count=5, on_drained=lambda: None)

        with pytest.raises(RuntimeError, match="already draining"):
            sync.drain_to_step_count(target_count=10, on_drained=lambda: None)

    def test_is_draining_property(self):
        sync = _make_sync()
        _register_op(sync, "op_a", step=0)

        assert not sync.is_draining
        sync.drain_to_step_count(target_count=5, on_drained=lambda: None)
        assert sync.is_draining
        sync.clear_drain()
        assert not sync.is_draining

    def test_clear_drain_idempotent(self):
        """Calling clear_drain when not draining must not raise."""
        sync = _make_sync()
        sync.clear_drain()
        sync.clear_drain()


class TestDrainAckCallback:
    """
    Workers are self-blocked at target_count, so they report exactly target_count.
    The callback fires when all workers have reported target_count.
    """

    def test_callback_fires_at_target_count_single_op(self):
        fired = threading.Event()
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=0)
        fired.clear()
        sync.drain_to_step_count(target_count=5, on_drained=fired.set)

        sync.report_worker_op_completed(0, "op_a", 5)
        assert not fired.is_set(), "Callback should not fire until all workers ACK"
        sync.report_worker_op_completed(1, "op_a", 5)
        assert fired.is_set(), "Callback must fire when all workers reach target_count"

    def test_callback_fires_with_multiple_ops(self):
        """All ops must be ACKed (at target_count) before on_drained fires."""
        fired = threading.Event()
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=0)
        _register_op(sync, "op_b", step=0)
        fired.clear()
        sync.drain_to_step_count(target_count=3, on_drained=fired.set)

        sync.report_worker_op_completed(0, "op_a", 3)
        sync.report_worker_op_completed(1, "op_a", 3)
        assert not fired.is_set(), "op_b not yet drained"

        sync.report_worker_op_completed(0, "op_b", 3)
        sync.report_worker_op_completed(1, "op_b", 3)
        assert fired.is_set()

    def test_partial_worker_completion_does_not_fire_early(self):
        fired = threading.Event()
        sync = _make_sync(num_workers=3)
        _register_op(sync, "op_a", step=0)
        fired.clear()
        sync.drain_to_step_count(target_count=2, on_drained=fired.set)

        # target_count = 2; only 2 of 3 workers reported
        sync.report_worker_op_completed(0, "op_a", 2)
        sync.report_worker_op_completed(1, "op_a", 2)
        assert not fired.is_set()

        sync.report_worker_op_completed(2, "op_a", 2)
        assert fired.is_set()


class TestDrainAndAllowedStepIndependence:
    """
    drain_target_count (ACK fence) and update_allowed_step (flow-control ceiling)
    must not interfere with each other.
    """

    def test_allowed_step_does_not_block_drain_ack(self):
        """A restrictive allowed_step must not prevent the drain callback from firing."""
        fired = threading.Event()
        sync = _make_sync(num_workers=2)
        sync.set_callbacks(
            block_worker_op=lambda r, op: None, unblock_worker_op=lambda r, op: None
        )
        _register_op(sync, "op_a", step=0)
        sync.update_allowed_step("op_a", 3)
        sync.drain_to_step_count(target_count=5, on_drained=fired.set)

        # Workers reach target_count=5; drain ACK must fire regardless of allowed_step=3
        sync.report_worker_op_completed(0, "op_a", 5)
        sync.report_worker_op_completed(1, "op_a", 5)
        assert fired.is_set(), "drain ACK must fire even when allowed_step is set"

    def test_drain_clear_does_not_reset_allowed_step(self):
        """clear_drain must leave update_allowed_step unchanged."""
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=0)
        sync.update_allowed_step("op_a", 10)
        sync.drain_to_step_count(target_count=5, on_drained=lambda: None)
        sync.clear_drain()
        assert sync._allowed_steps.get("op_a") == 10


class TestWorkerDrainTarget:
    """
    Verify SPMDWorker.set_drain_target_count / clear_drain_target_count state correctness.
    """

    def _make_stub(self):
        from fluxway.spmd.worker.worker import SPMDWorker as _ActorCls

        raw_cls = _ActorCls.__ray_actor_class__
        stub = types.SimpleNamespace(
            _drain_target_count=None,
            _drain_cond=asyncio.Condition(),
            spmd_id="test_spmd",
            rank=0,
        )
        stub.set_drain_target_count = types.MethodType(
            raw_cls.set_drain_target_count, stub
        )
        stub.clear_drain_target_count = types.MethodType(
            raw_cls.clear_drain_target_count, stub
        )
        return stub

    @pytest.mark.asyncio
    async def test_set_drain_target_updates_state(self):
        w = self._make_stub()
        await w.set_drain_target_count(42)
        assert w._drain_target_count == 42

    @pytest.mark.asyncio
    async def test_clear_drain_resets_state(self):
        w = self._make_stub()
        await w.set_drain_target_count(42)
        await w.clear_drain_target_count()
        assert w._drain_target_count is None

    @pytest.mark.asyncio
    async def test_set_overrides_previous_target(self):
        w = self._make_stub()
        await w.set_drain_target_count(10)
        await w.set_drain_target_count(20)
        assert w._drain_target_count == 20

    @pytest.mark.asyncio
    async def test_clear_is_idempotent(self):
        w = self._make_stub()
        await w.clear_drain_target_count()
        assert w._drain_target_count is None


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
        sync.drain_to_step_count(target_count=0, on_drained=fired.set)
        assert not fired.is_set(), (
            "drain must not fire when no ops have been registered: "
            "an empty snapshot would be produced with no actual state"
        )

    def test_drain_fires_immediately_when_already_at_target(self):
        """drain_to_step_count fires immediately if all workers already satisfy the target."""
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=5)  # both workers at count=5
        fired = threading.Event()
        sync.drain_to_step_count(target_count=3, on_drained=fired.set)
        assert (
            fired.is_set()
        ), "drain must fire immediately when workers already exceed target_count"

    def test_drain_does_not_fire_when_workers_below_target(self):
        """drain_to_step_count must NOT fire if workers are below target_count."""
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=2)  # both workers at count=2
        fired = threading.Event()
        sync.drain_to_step_count(target_count=5, on_drained=fired.set)
        assert (
            not fired.is_set()
        ), "drain must not fire when workers have not reached target_count"
