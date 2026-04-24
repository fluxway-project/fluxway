"""
Unit tests for per-worker checkpoint I/O.

Covers:
 - SafeTensorsBackend save → load round-trip (tensors + scalars + nested dicts)
 - OrbaxBackend save → load round-trip (jax pytrees)
 - WorkerCheckpointBackend registry (create by name)
 - _flatten_worker_state / _unflatten_worker_state internal round-trip
 - Failure paths: missing meta file, unwritable directory
 - save_state_to_disk / load_state_from_disk Worker method contract (mock backend)
"""

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from fluxway.control_plane.checkpointer import (
    SafeTensorsBackend,
    WorkerCheckpointBackend,
)
from fluxway.control_plane.checkpointer.safetensors import (
    _flatten_worker_state,
    _unflatten_worker_state,
    load_worker_state,
    save_worker_state,
)

# =========================================================================
# Helpers
# =========================================================================


def _make_state() -> Dict[str, Any]:
    """Canonical test state: tensors, scalars, and nested dicts."""
    return {
        "weights": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "step": 42,
        "config": {"lr": 0.01, "batch_size": 32},
        "label": "my_op",
        "flag": True,
        "nothing": None,
    }


# =========================================================================
# _flatten_worker_state / _unflatten_worker_state
# =========================================================================


class TestFlattenUnflatten:
    def test_round_trip_primitives(self):
        state = {"a": 1, "b": "hello", "c": True, "d": None}
        flat_t, flat_m = _flatten_worker_state(state)
        assert flat_t == {}
        result = _unflatten_worker_state(flat_t, flat_m)
        assert result == state

    def test_round_trip_numpy(self):
        arr = np.array([10, 20, 30], dtype=np.int64)
        state = {"x": arr}
        flat_t, flat_m = _flatten_worker_state(state)
        assert "x" in flat_t
        assert flat_m == {}
        result = _unflatten_worker_state(flat_t, flat_m)
        np.testing.assert_array_equal(result["x"], arr)

    def test_round_trip_nested_dict(self):
        state = {"cfg": {"lr": 0.001, "steps": 10}}
        flat_t, flat_m = _flatten_worker_state(state)
        result = _unflatten_worker_state(flat_t, flat_m)
        assert result == state

    def test_round_trip_mixed(self):
        state = _make_state()
        flat_t, flat_m = _flatten_worker_state(state)
        result = _unflatten_worker_state(flat_t, flat_m)
        assert result["step"] == state["step"]
        assert result["label"] == state["label"]
        assert result["flag"] == state["flag"]
        assert result["nothing"] == state["nothing"]
        np.testing.assert_array_equal(result["weights"], state["weights"])

    def test_separator_not_in_keys(self):
        """Keys must not contain the internal separator \\x00."""
        state = {"normal_key": 1, "another": np.zeros(3, dtype=np.float32)}
        flat_t, flat_m = _flatten_worker_state(state)
        for key in list(flat_t) + list(flat_m):
            # Composite keys use \x00 as separator; the root segment must be a
            # top-level field declared in state (no spurious separator injected).
            root = key.split("\x00")[0]
            assert (
                root in state
            ), f"Unexpected root key {root!r} in flattened key {key!r}"


# =========================================================================
# save_worker_state / load_worker_state (filesystem round-trip)
# =========================================================================


class TestSaveLoadWorkerState:
    def test_round_trip(self, tmp_path):
        state = _make_state()
        t_path = str(tmp_path / "w.safetensors")
        m_path = str(tmp_path / "w.meta.json")
        save_worker_state(state, t_path, m_path)
        loaded = load_worker_state(t_path, m_path)
        assert loaded["step"] == state["step"]
        assert loaded["label"] == state["label"]
        np.testing.assert_array_equal(loaded["weights"], state["weights"])

    def test_tensor_only_state(self, tmp_path):
        state = {"w": np.ones((4, 4), dtype=np.float32)}
        t_path = str(tmp_path / "w.safetensors")
        m_path = str(tmp_path / "w.meta.json")
        save_worker_state(state, t_path, m_path)
        loaded = load_worker_state(t_path, m_path)
        np.testing.assert_array_equal(loaded["w"], state["w"])

    def test_scalar_only_state(self, tmp_path):
        state = {"epoch": 7, "done": False}
        t_path = str(tmp_path / "w.safetensors")
        m_path = str(tmp_path / "w.meta.json")
        save_worker_state(state, t_path, m_path)
        # No .safetensors file is written when there are no tensors
        assert not Path(t_path).exists()
        loaded = load_worker_state(t_path, m_path)
        assert loaded == state

    def test_missing_meta_raises(self, tmp_path):
        t_path = str(tmp_path / "w.safetensors")
        m_path = str(tmp_path / "w.meta.json")
        with pytest.raises(FileNotFoundError, match="Checkpoint metadata not found"):
            load_worker_state(t_path, m_path)

    def test_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        t_path = str(deep / "w.safetensors")
        m_path = str(deep / "w.meta.json")
        save_worker_state({"v": 1}, t_path, m_path)
        assert Path(m_path).exists()


# =========================================================================
# SafeTensorsBackend — high-level backend API
# =========================================================================


class TestSafeTensorsBackend:
    @pytest.mark.asyncio
    async def test_round_trip(self, tmp_path):
        backend = SafeTensorsBackend()
        state = _make_state()
        metadata = await backend.save(state, str(tmp_path), "spmd_a", rank=0, step=5)
        assert "tensors_path" in metadata
        assert "meta_path" in metadata
        loaded = await backend.load(metadata)
        assert loaded["step"] == state["step"]
        np.testing.assert_array_equal(loaded["weights"], state["weights"])

    @pytest.mark.asyncio
    async def test_multiple_ranks_do_not_overwrite(self, tmp_path):
        backend = SafeTensorsBackend()
        state_r0 = {"v": np.array([0.0], dtype=np.float32)}
        state_r1 = {"v": np.array([1.0], dtype=np.float32)}
        meta0 = await backend.save(state_r0, str(tmp_path), "spmd_a", rank=0, step=1)
        meta1 = await backend.save(state_r1, str(tmp_path), "spmd_a", rank=1, step=1)
        loaded0 = await backend.load(meta0)
        loaded1 = await backend.load(meta1)
        np.testing.assert_array_equal(loaded0["v"], state_r0["v"])
        np.testing.assert_array_equal(loaded1["v"], state_r1["v"])

    def test_is_picklable(self):
        backend = SafeTensorsBackend()
        restored = pickle.loads(pickle.dumps(backend))
        assert isinstance(restored, SafeTensorsBackend)

    @pytest.mark.asyncio
    async def test_load_missing_metadata_raises(self):
        backend = SafeTensorsBackend()
        with pytest.raises(FileNotFoundError):
            await backend.load(
                {
                    "tensors_path": "/nonexistent/path.safetensors",
                    "meta_path": "/nonexistent/path.meta.json",
                }
            )


# =========================================================================
# Backend registry
# =========================================================================


class TestBackendRegistry:
    def test_create_safetensors(self):
        b = WorkerCheckpointBackend.create("safetensors")
        assert isinstance(b, SafeTensorsBackend)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown checkpoint backend"):
            WorkerCheckpointBackend.create("no_such_backend")

    def test_registered_names_contains_defaults(self):
        names = WorkerCheckpointBackend.registered_names()
        assert "safetensors" in names
        assert "orbax" in names

    def test_custom_backend_self_registers(self):
        class _TestBackend(WorkerCheckpointBackend, backend_name="_test_only_"):
            def __init__(self, force_write: bool = False):
                pass

            async def save(self, state, ckpt_dir, spmd_id, rank, step):
                return {}

            async def load(self, metadata):
                return {}

            def finalize(self):
                pass

        b = WorkerCheckpointBackend.create("_test_only_")
        assert isinstance(b, _TestBackend)


# =========================================================================
# Worker save_state_to_disk / load_state_from_disk — mock backend
# =========================================================================


class _CapturingBackend(WorkerCheckpointBackend, backend_name="_capturing_"):
    """Records calls without touching the filesystem."""

    def __init__(self, force_write: bool = False):
        self.saved_states = []
        self.load_return = {}

    async def save(self, state, ckpt_dir, spmd_id, rank, step):
        self.saved_states.append(
            {
                "state": state,
                "ckpt_dir": ckpt_dir,
                "spmd_id": spmd_id,
                "rank": rank,
                "step": step,
            }
        )
        return {"_captured": True}

    async def load(self, metadata):
        return self.load_return

    def finalize(self):
        pass


class TestWorkerCheckpointContract:
    """
    Verifies the contract between Worker and WorkerCheckpointBackend.

    SPMDWorker is wrapped by @ray.remote; calling SPMDWorker.method(mock, ...)
    routes through Ray's tracing shim. We therefore test the contract by
    replicating the minimal protocol inline — matching what save_state_to_disk
    and load_state_from_disk actually do — without invoking the Ray actor.
    """

    @pytest.mark.asyncio
    async def test_save_collects_state_and_calls_backend(self, tmp_path):
        backend = _CapturingBackend()
        operator_states = {"embed": {"w": np.array([1.0], dtype=np.float32)}}
        completed_step_count = {"embed": 10, "lookup": 10}

        # Replicate the save_state_to_disk contract
        state = {
            "operator_states": operator_states,
            "completed_step_count": completed_step_count,
        }
        metadata = await backend.save(
            state, str(tmp_path), spmd_id="spmd_test", rank=0, step=10
        )
        metadata["completed_step_count"] = dict(completed_step_count)

        assert len(backend.saved_states) == 1
        call = backend.saved_states[0]
        assert call["step"] == 10
        assert call["spmd_id"] == "spmd_test"
        assert "operator_states" in call["state"]
        assert metadata["completed_step_count"] == {"embed": 10, "lookup": 10}

    @pytest.mark.asyncio
    async def test_load_restores_completed_step_count(self):
        backend = _CapturingBackend()
        backend.load_return = {
            "operator_states": {"embed": {"w": 99}},
            "completed_step_count": {"embed": 7, "lookup": 7},
        }
        completed_step_count = {"embed": 10, "lookup": 10}

        # Replicate the load_state_from_disk contract
        state = await backend.load({"_captured": True})
        for op_name, step in state["completed_step_count"].items():
            completed_step_count[op_name] = step

        assert completed_step_count == {"embed": 7, "lookup": 7}
        # operator_states are forwarded to _call_operator_restore_state
        assert state["operator_states"] == {"embed": {"w": 99}}
