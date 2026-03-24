"""
Unit tests for SPMDSynchronizer drain-to-step barrier and Worker drain ceiling.

Design:
- Workers are self-blocked at drain_ceiling by Worker._drain_resume_event (set_drain_ceiling RPC).
- Workers therefore only ever REPORT completion up to ceiling-1.
- The Synchronizer's job is to count those ACKs and fire the on_drained callback.
- Synchronizer no longer manages blocking for drain (that's the Worker's responsibility).

Tests are split into:
- TestDrainSetup: setup/teardown invariants
- TestDrainAckCallback: ACK counting and callback triggering
- TestDrainAndAllowedStepIndependence: drain ACK must not be affected by allowed_step
- TestWorkerDrainCeiling: Worker.set/clear_drain_ceiling state correctness
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
        """Calling drain_to_step while already draining must raise RuntimeError."""
        sync = _make_sync()
        _register_op(sync, "op_a", step=0)
        sync.drain_to_step(target_step=5, on_drained=lambda: None)

        with pytest.raises(RuntimeError, match="already draining"):
            sync.drain_to_step(target_step=10, on_drained=lambda: None)

    def test_is_draining_property(self):
        sync = _make_sync()
        _register_op(sync, "op_a", step=0)

        assert not sync.is_draining
        sync.drain_to_step(target_step=5, on_drained=lambda: None)
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
    Workers are self-blocked at ceiling, so they only report up to ceiling-1.
    The callback fires when all workers have reported ceiling-1.
    """

    def test_callback_fires_at_ceiling_minus_one_single_op(self):
        fired = threading.Event()
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=0)
        fired.clear()
        sync.drain_to_step(target_step=5, on_drained=fired.set)

        sync.report_worker_op_completed(0, "op_a", 4)  # ceiling-1 = 4
        assert not fired.is_set(), "Callback should not fire until all workers ACK"
        sync.report_worker_op_completed(1, "op_a", 4)
        assert fired.is_set(), "Callback must fire when all workers reach ceiling-1"

    def test_callback_fires_with_multiple_ops(self):
        """All ops must be ACKed (at ceiling-1) before on_drained fires."""
        fired = threading.Event()
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=0)
        _register_op(sync, "op_b", step=0)
        fired.clear()
        sync.drain_to_step(target_step=3, on_drained=fired.set)

        sync.report_worker_op_completed(0, "op_a", 2)
        sync.report_worker_op_completed(1, "op_a", 2)
        assert not fired.is_set(), "op_b not yet drained"

        sync.report_worker_op_completed(0, "op_b", 2)
        sync.report_worker_op_completed(1, "op_b", 2)
        assert fired.is_set()

    def test_partial_worker_completion_does_not_fire_early(self):
        fired = threading.Event()
        sync = _make_sync(num_workers=3)
        _register_op(sync, "op_a", step=0)
        fired.clear()
        sync.drain_to_step(target_step=2, on_drained=fired.set)

        # ceiling-1 = 1; only 2 of 3 workers reported
        sync.report_worker_op_completed(0, "op_a", 1)
        sync.report_worker_op_completed(1, "op_a", 1)
        assert not fired.is_set()

        sync.report_worker_op_completed(2, "op_a", 1)
        assert fired.is_set()


class TestDrainAndAllowedStepIndependence:
    """
    drain_ceiling (ACK fence) and update_allowed_step (flow-control ceiling)
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
        sync.drain_to_step(target_step=5, on_drained=fired.set)

        # Workers reach ceiling-1=4; drain ACK must fire regardless of allowed_step=3
        sync.report_worker_op_completed(0, "op_a", 4)
        sync.report_worker_op_completed(1, "op_a", 4)
        assert fired.is_set(), "drain ACK must fire even when allowed_step is set"

    def test_drain_clear_does_not_reset_allowed_step(self):
        """clear_drain must leave update_allowed_step unchanged."""
        sync = _make_sync(num_workers=2)
        _register_op(sync, "op_a", step=0)
        sync.update_allowed_step("op_a", 10)
        sync.drain_to_step(target_step=5, on_drained=lambda: None)
        sync.clear_drain()
        assert sync._allowed_steps.get("op_a") == 10


class TestWorkerDrainCeiling:
    """
    Verify SPMDWorker.set_drain_ceiling / clear_drain_ceiling field logic.

    SPMDWorker is a @ray.remote actor so we cannot instantiate it.  We instead
    build a plain stub and bind the methods via __ray_actor_class__.
    """

    def _make_stub(self):
        from fluxway.spmd.worker.worker import SPMDWorker as _ActorCls

        raw_cls = _ActorCls.__ray_actor_class__
        stub = types.SimpleNamespace(
            _drain_ceiling=None, _drain_resume_event=asyncio.Event()
        )
        stub._drain_resume_event.set()
        stub.set_drain_ceiling = types.MethodType(raw_cls.set_drain_ceiling, stub)
        stub.clear_drain_ceiling = types.MethodType(raw_cls.clear_drain_ceiling, stub)
        return stub

    def test_set_drain_ceiling_updates_state(self):
        w = self._make_stub()
        w.set_drain_ceiling(42)
        assert w._drain_ceiling == 42
        assert not w._drain_resume_event.is_set()

    def test_clear_drain_ceiling_resets_state(self):
        w = self._make_stub()
        w.set_drain_ceiling(42)
        w.clear_drain_ceiling()
        assert w._drain_ceiling is None
        assert w._drain_resume_event.is_set()

    def test_set_overrides_previous_ceiling(self):
        w = self._make_stub()
        w.set_drain_ceiling(10)
        w.set_drain_ceiling(20)
        assert w._drain_ceiling == 20
        assert not w._drain_resume_event.is_set()

    def test_clear_is_idempotent(self):
        w = self._make_stub()
        w.clear_drain_ceiling()
        assert w._drain_ceiling is None
        assert w._drain_resume_event.is_set()
