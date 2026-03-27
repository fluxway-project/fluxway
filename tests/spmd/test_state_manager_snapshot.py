"""
Tests for StateManager snapshot API.
"""

import pytest
from fluxway.control_plane.state_manager.manager import StateManager


def _make_payload() -> dict:
    return {
        "spmd_a": {
            0: {
                "spmd_id": "spmd_a",
                "rank": 0,
                "completed_step_count": {"op": 5},
                "static_args": {},
                "operator_states": {},
            },
            1: {
                "spmd_id": "spmd_a",
                "rank": 1,
                "completed_step_count": {"op": 5},
                "static_args": {},
                "operator_states": {},
            },
        }
    }


class TestSnapshotRoundTrip:
    def test_save_and_load_snapshot(self):
        manager = StateManager()
        payload = _make_payload()
        manager.save_snapshot("snap_001", payload)
        loaded = manager.load_snapshot("snap_001")
        assert loaded == payload

    def test_load_nonexistent_snapshot_raises(self):
        manager = StateManager()
        with pytest.raises(KeyError, match="snap_999"):
            manager.load_snapshot("snap_999")

    def test_list_snapshots(self):
        manager = StateManager()
        manager.save_snapshot("snap_1", _make_payload())
        manager.save_snapshot("snap_2", _make_payload())
        assert set(manager.list_snapshots()) == {"snap_1", "snap_2"}

    def test_delete_snapshot(self):
        manager = StateManager()
        manager.save_snapshot("snap_del", _make_payload())
        manager.delete_snapshot("snap_del")
        assert "snap_del" not in manager.list_snapshots()

    def test_delete_nonexistent_raises(self):
        manager = StateManager()
        with pytest.raises(KeyError):
            manager.delete_snapshot("nonexistent")

    def test_overwrite_snapshot(self):
        manager = StateManager()
        manager.save_snapshot("snap_x", _make_payload())
        new_payload = {"spmd_b": {0: {"rank": 0}}}
        manager.save_snapshot("snap_x", new_payload)
        assert manager.load_snapshot("snap_x") == new_payload

    def test_isolated_across_instances(self):
        """Two StateManager instances must not share snapshots."""
        m1 = StateManager()
        m2 = StateManager()
        m1.save_snapshot("snap_iso", _make_payload())
        assert "snap_iso" not in m2.list_snapshots()
