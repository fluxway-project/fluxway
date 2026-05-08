"""
Tests for the unified completion notification mechanism and SPMD Worker infrastructure.

Covers:
- ArrivalBarrier: N-arrival primitive shared across transport and mailbox layers
- StepBarrier: per-step composite barrier in WorkerMailbox
- WorkerMailbox: declare_step, deposit_shards/put_state, wait_step_ready, on_ready callback
- TransportReceiver: async wait() via ArrivalBarrier
- Worker integration: mesh, config, data reception
"""

import sys
import os
import asyncio
from unittest.mock import MagicMock
import numpy as np
import pytest

sys.path.append(os.getcwd())

from fluxway.common.sync import ArrivalBarrier
from fluxway.spmd.worker.mailbox.mailbox import StepBarrier, WorkerMailbox
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.common.types.configs import SPMDConfig, OperatorMeta, ArgumentType
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.communication import TransportReceiver
from fluxway.common.types.sharding import ShardingSpec, PartitionSpec

# ---------------------------------------------------------------------------
# Helpers: build minimal SPMDConfig with registered operators
# ---------------------------------------------------------------------------


def _make_null_meta(name: str) -> OperatorMeta:
    """Create a minimal OperatorMeta for unit tests (no sharding, no args)."""
    return OperatorMeta(name=name, input_arguments=None, output_arguments=None)


def _make_config(mesh: DeviceMesh, *op_names: str, **extra_ops) -> SPMDConfig:
    """
    Build SPMDConfig with the given operator names (each with a null lambda + meta).
    extra_ops allows passing named op->callable overrides.
    """
    ops = {op: (extra_ops.get(op) or (lambda *a, **kw: None)) for op in op_names}
    metas = {op: _make_null_meta(op) for op in op_names}
    return SPMDConfig(mesh=mesh, operators=ops, operator_metas=metas)


# Generic dummy functions for testing
def compute_fn(state, batch):
    return state


def optimize_fn(state, grad):
    return state


def apply_fn(state, grad):
    return state


def _worker_cls():
    return SPMDWorker.__ray_metadata__.modified_class


# ===========================================================================
# ArrivalBarrier — shared primitive
# ===========================================================================


def test_arrival_barrier_zero_immediately_done():
    """n=0 means the barrier fires immediately."""
    b = ArrivalBarrier(0)
    assert b.is_done()


def test_arrival_barrier_n1():
    """n=1: first arrive() fires the barrier."""
    b = ArrivalBarrier(1)
    assert not b.is_done()
    fired = b.arrive()
    assert fired
    assert b.is_done()


def test_arrival_barrier_n3():
    """n=3: barrier fires on exactly the 3rd arrive()."""
    b = ArrivalBarrier(3)
    assert b.arrive() is False
    assert b.arrive() is False
    fired = b.arrive()
    assert fired
    assert b.is_done()
    # Extra arrives do not return True (already done)
    assert b.arrive() is False


def test_arrival_barrier_reset():
    """reset() clears the event and counter."""
    b = ArrivalBarrier(2)
    b.arrive()
    b.arrive()
    assert b.is_done()
    b.reset()
    assert not b.is_done()
    assert b.count == 0


@pytest.mark.asyncio
async def test_arrival_barrier_async_wait():
    """await wait() resolves after N arrivals triggered from a task."""
    b = ArrivalBarrier(2)

    async def trigger():
        await asyncio.sleep(0)
        b.arrive()
        await asyncio.sleep(0)
        b.arrive()

    asyncio.create_task(trigger())
    await b.wait()
    assert b.is_done()


def test_arrival_barrier_periodic_batches():
    """
    10 total arrivals, batch=3 → fires at counts 3, 6, 9.
    close() flushes the remainder (1) → fires at count 10.
    Total: 4 firings.
    """
    b = ArrivalBarrier(3, periodic=True)
    fires = []

    for i in range(10):
        fired = b.arrive()
        if fired:
            fires.append(b.count)

    remainder = b.close()

    assert fires == [3, 6, 9], f"expected [3,6,9], got {fires}"
    assert remainder is True
    assert b.count == 10


@pytest.mark.asyncio
async def test_arrival_barrier_periodic_async():
    """Async wait_batch() gets each of the 4 firings for 10/3 scenario."""
    b = ArrivalBarrier(3, periodic=True)
    received = []

    async def producer():
        for _ in range(10):
            await asyncio.sleep(0)
            b.arrive()
        b.close()

    asyncio.create_task(producer())

    async for count in b:
        received.append(count)

    assert received == [3, 6, 9, 10], f"expected [3,6,9,10], got {received}"


# ===========================================================================
# TransportReceiver — now backed by ArrivalBarrier
# ===========================================================================


@pytest.mark.asyncio
async def test_transport_receiver_async_wait():
    """TransportReceiver.wait() resolves when wait_for_n_slices payloads arrive."""
    receiver = TransportReceiver(wait_for_n_slices=2)
    assert not receiver.is_done()

    async def send():
        await asyncio.sleep(0)
        receiver.receive_slices(0, [np.zeros(4)], [4])
        await asyncio.sleep(0)
        receiver.receive_slices(1, [np.zeros(4)], [4])

    asyncio.create_task(send())
    await receiver.wait()
    assert receiver.is_done()
    slices = receiver.get_buffered_slices()
    assert len(slices) == 2


def test_transport_receiver_buffered():
    """Synchronous usage: receive_slices accumulates; get_buffered_slices consumes."""
    receiver = TransportReceiver(wait_for_n_slices=3)
    for i in range(3):
        receiver.receive_slices(i, [np.eye(2)], [2])
    assert receiver.is_done()
    slices = receiver.get_buffered_slices()
    assert len(slices) == 3
    assert receiver.get_buffered_slices() == []  # consumed


@pytest.mark.asyncio
async def test_transport_receiver_periodic_10_batch3():
    """
    10 slices total, periodic batch_size=3.
    Expect 4 triggers: at cumulative counts 3, 6, 9, and 10 (remainder).
    """
    receiver = TransportReceiver(wait_for_n_slices=3, periodic=True)
    trigger_counts = []

    async def producer():
        for _ in range(10):
            await asyncio.sleep(0)
            receiver.receive_slices(0, [np.zeros(4)], [4])
        receiver.close()  # flushes remainder (1 slice)

    asyncio.create_task(producer())

    async for count in receiver:
        trigger_counts.append(count)

    assert trigger_counts == [3, 6, 9, 10], f"got {trigger_counts}"


def test_transport_receiver_zero_copy_offset():
    """
    Offset (shared buffer) zero-copy: two senders each write 2 rows
    into a shared 4-row buffer at different offsets.  No concat needed.
    """
    shared = np.zeros((4, 3), dtype=np.float32)
    receiver = TransportReceiver(wait_for_n_slices=2)
    receiver.register_zero_copy_buffers(
        target_buffers=[shared],
        expect_offsets={0: 0, 1: 2},  # rank 0 → rows 0-1, rank 1 → rows 2-3
    )

    data0 = np.ones((2, 3), dtype=np.float32) * 1.0
    data1 = np.ones((2, 3), dtype=np.float32) * 2.0
    receiver.receive_slices(0, [data0], [2])
    receiver.receive_slices(1, [data1], [2])

    assert receiver.is_done()
    # shared buffer should have rank-0 data in rows 0-1, rank-1 in rows 2-3
    np.testing.assert_array_equal(receiver.target_buffers[0][:2], data0)
    np.testing.assert_array_equal(receiver.target_buffers[0][2:], data1)
    # verify actual written-row counts are recorded
    wc = receiver.get_written_counts()
    assert wc[0] == [2], f"rank-0 written_counts={wc[0]}"
    assert wc[1] == [2], f"rank-1 written_counts={wc[1]}"


def test_transport_receiver_zero_copy_dlpack_and_no_concat():
    """
    Verifies two essential properties of offset zero-copy mode:

    1. The underlying buffer's base memory address does NOT change after
       either write — pure in-place update, no realloc.

    2. Both senders' data is readable directly from the single contiguous
       buffer without any concat/stack step.

    Note: get_dlpack_slice belongs to the registration phase, not the
    receive path.  The receive path writes directly into the pre-allocated
    slice via index assignment, which already avoids any final concat.
    """
    shared = np.zeros((6, 4), dtype=np.float32)
    from fluxway.common.backend import get_backend

    backend = get_backend()

    # Capture memory address before writes (numpy __array_interface__)
    ptr_before = shared.__array_interface__["data"][0]

    receiver = TransportReceiver(wait_for_n_slices=2)
    receiver.register_zero_copy_buffers(
        target_buffers=[shared],
        # rank 0 writes rows 0-2, rank 1 writes rows 3-5
        expect_offsets={0: 0, 1: 3},
    )

    data0 = np.ones((3, 4), dtype=np.float32) * 10.0
    data1 = np.ones((3, 4), dtype=np.float32) * 20.0
    receiver.receive_slices(0, [data0], [3])
    receiver.receive_slices(1, [data1], [3])
    assert receiver.is_done()

    # --- Property 1: No realloc — same underlying buffer ---
    # (JAX's dynamic_update_slice always returns a new jax.Array,
    #  so we only check identity for non-JAX backends.)
    if backend.name() != "jax":
        ptr_after = receiver.target_buffers[0].__array_interface__["data"][0]
        assert ptr_after == ptr_before, (
            f"Buffer was reallocated! Before: {ptr_before:#x}, After: {ptr_after:#x}"
        )

    # --- Property 2: No concat needed — read directly from the single buffer ---
    buf = receiver.target_buffers[0]  # single contiguous pre-allocated array
    np.testing.assert_array_equal(buf[:3], data0)  # rank-0 rows
    np.testing.assert_array_equal(buf[3:], data1)  # rank-1 rows


def test_transport_receiver_zero_copy_per_sender():
    """
    No-offset (per-sender) zero-copy: each sender writes into its own
    independent buffer at row 0.  No shared buffer, no concat.
    """
    buf0 = np.zeros((3, 2), dtype=np.float32)
    buf1 = np.zeros((5, 2), dtype=np.float32)

    receiver = TransportReceiver(wait_for_n_slices=2)
    receiver.register_per_sender_buffers({0: [buf0], 1: [buf1]})

    data0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    data1 = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)

    receiver.receive_slices(0, [data0], [3])  # fill all 3 rows of buf0
    receiver.receive_slices(1, [data1], [2])  # fill first 2 rows of buf1

    assert receiver.is_done()
    per_sender = receiver.get_per_sender_buffers()
    np.testing.assert_array_equal(per_sender[0][0], data0)
    np.testing.assert_array_equal(per_sender[1][0][:2], data1)
    # verify actual written-row counts (buf1 has 5 rows but only 2 were written)
    wc = receiver.get_written_counts()
    assert wc[0] == [3], f"rank-0 written_counts={wc[0]}"
    assert wc[1] == [2], f"rank-1 written_counts={wc[1]}"


def test_worker_phase_one_does_not_start_runtime(monkeypatch):
    mesh = DeviceMesh(axis_names=("dp",), device_shape=(1,))
    config = _make_config(mesh, "compute")
    worker_cls = _worker_cls()
    started = {"heartbeat": 0, "run_loop": 0}

    monkeypatch.setattr(
        worker_cls,
        "_start_heartbeat",
        lambda self: started.__setitem__("heartbeat", started["heartbeat"] + 1),
    )
    monkeypatch.setattr(
        worker_cls,
        "_start_run_loop",
        lambda self: started.__setitem__("run_loop", started["run_loop"] + 1),
    )

    worker = worker_cls(
        config=config,
        rank=0,
        spmd_id="spmd-a",
        driver_handle=object(),
        checkpoint_backend=MagicMock(),
    )

    assert started == {"heartbeat": 0, "run_loop": 0}
    assert worker.local_rank is None
    assert worker.local_world_size is None
    assert worker.node_rank is None
    assert worker.num_nodes is None


def test_worker_setup_and_start_initializes_runtime(monkeypatch):
    mesh = DeviceMesh(axis_names=("dp",), device_shape=(1,))
    config = _make_config(mesh, "compute")
    worker_cls = _worker_cls()
    calls = {"preinstantiate": [], "heartbeat": 0, "run_loop": 0}

    class FakeOperatorManager:
        def __init__(
            self,
            config,
            rank,
            spmd_id,
            *,
            local_rank=None,
            local_world_size=None,
            node_rank=0,
            num_nodes=1,
        ):
            self.payload = (
                config,
                rank,
                spmd_id,
                local_rank,
                local_world_size,
                node_rank,
                num_nodes,
            )

        def is_source_operator(self, op_name):
            return False

    monkeypatch.setattr(
        worker_cls,
        "_pre_instantiate_operators",
        staticmethod(
            lambda config, rank, **kwargs: calls["preinstantiate"].append(
                (rank, kwargs)
            )
        ),
    )
    monkeypatch.setattr(
        worker_cls,
        "_start_heartbeat",
        lambda self: calls.__setitem__("heartbeat", calls["heartbeat"] + 1),
    )
    monkeypatch.setattr(
        worker_cls,
        "_start_run_loop",
        lambda self: calls.__setitem__("run_loop", calls["run_loop"] + 1),
    )
    monkeypatch.setattr(
        "fluxway.spmd.worker.worker.OperatorManager", FakeOperatorManager
    )

    worker = worker_cls(
        config=config,
        rank=2,
        spmd_id="spmd-a",
        driver_handle=object(),
        checkpoint_backend=MagicMock(),
    )
    worker.setup_and_start(local_rank=1, local_world_size=3, node_rank=0, num_nodes=2)

    assert calls["preinstantiate"] == [
        (2, {"local_rank": 1, "local_world_size": 3, "node_rank": 0, "num_nodes": 2})
    ]
    assert calls["heartbeat"] == 1
    assert calls["run_loop"] == 1


@pytest.mark.asyncio
async def test_worker_stop_clears_driver_handle_and_waits_for_tasks():
    mesh = DeviceMesh(axis_names=("dp",), device_shape=(1,))
    config = _make_config(mesh, "compute")
    worker_cls = _worker_cls()
    worker = worker_cls(
        config=config,
        rank=0,
        spmd_id="spmd-a",
        driver_handle=object(),
        checkpoint_backend=MagicMock(),
    )

    async def run_forever():
        await asyncio.Future()

    background_task_ran = asyncio.Event()

    async def background_task():
        background_task_ran.set()
        await asyncio.Future()

    worker._execution_queue = asyncio.Queue()
    worker._run_task = asyncio.create_task(run_forever())
    worker._heartbeat_task = asyncio.create_task(run_forever())
    worker._spawn_background_task(background_task())
    await background_task_ran.wait()

    await worker.stop()

    assert worker.get_driver_handle() is None
    assert worker._run_task.cancelled()
    assert worker._heartbeat_task.cancelled()
    assert worker._background_tasks == set()
    sentinel = await worker._execution_queue.get()
    assert sentinel == (999, 0, "__stop__")


def test_worker_setup_and_start_logs_phase_markers(monkeypatch, capsys):
    mesh = DeviceMesh(axis_names=("dp",), device_shape=(1,))
    config = _make_config(mesh, "compute")
    worker_cls = _worker_cls()
    monkeypatch.setenv("FLUXWAY_LOG_LEVEL", "DEBUG")

    monkeypatch.setattr(
        worker_cls,
        "_pre_instantiate_operators",
        staticmethod(lambda config, rank, **kwargs: None),
    )
    monkeypatch.setattr(worker_cls, "_start_heartbeat", lambda self: None)
    monkeypatch.setattr(worker_cls, "_start_run_loop", lambda self: None)

    class FakeOperatorManager:
        def __init__(self, *args, **kwargs):
            pass

        def is_source_operator(self, op_name):
            return False

    monkeypatch.setattr(
        "fluxway.spmd.worker.worker.OperatorManager", FakeOperatorManager
    )

    worker = worker_cls(
        config=config,
        rank=0,
        spmd_id="spmd-log",
        driver_handle=object(),
        checkpoint_backend=MagicMock(),
    )

    worker.setup_and_start(local_rank=0, local_world_size=1, node_rank=0, num_nodes=1)

    captured = capsys.readouterr()
    assert "[DEBUG]" in captured.err
    assert "setup_and_start_begin" in captured.err
    assert "setup_and_start_preinstantiate_complete" in captured.err
    assert "setup_and_start_runtime_started" in captured.err


# ===========================================================================
# StepBarrier
# ===========================================================================


def test_step_barrier_fires_on_completion():
    """Fires when all tensor shards AND all states have arrived."""
    b = StepBarrier(
        "test", 0, expected_tensors={"emb": 2, "attn": 1}, expected_states=["lr"]
    )
    assert not b.is_ready()
    b.tensor_arrived("emb")
    b.tensor_arrived("emb")
    b.tensor_arrived("attn")
    assert not b.is_ready()  # state "lr" still missing
    b.control_arg_arrived("lr")
    assert b.is_ready()


def test_step_barrier_partial_not_ready():
    """Partial arrivals do not fire the barrier."""
    b = StepBarrier("test", 0, expected_tensors={"x": 3}, expected_states=[])
    b.tensor_arrived("x")
    b.tensor_arrived("x")
    assert not b.is_ready()


def test_step_barrier_no_expectations_fires_immediately():
    """Empty expectations → already ready."""
    b = StepBarrier("test", 0, expected_tensors={}, expected_states=[])
    assert b.is_ready()


def test_step_barrier_unknown_name_ignored():
    """Arrivals for unexpected names are silently ignored."""
    b = StepBarrier("test", 0, expected_tensors={"x": 1}, expected_states=[])
    assert b.tensor_arrived("unknown_tensor") is False
    assert not b.is_ready()
    b.tensor_arrived("x")
    assert b.is_ready()


@pytest.mark.asyncio
async def test_step_barrier_periodic():
    """Verify StepBarrier in periodic mode fires for every arrival."""
    # Eager op expects 1 shard for 'x'
    b = StepBarrier(
        "test", 0, expected_tensors={"x": 1}, expected_states=[], periodic=True
    )

    # First arrival
    fired = b.tensor_arrived("x")
    assert fired is True

    # Second arrival
    fired = b.tensor_arrived("x")
    assert fired is True


@pytest.mark.asyncio
async def test_step_barrier_async_wait():
    b = StepBarrier("test", 0, expected_tensors={"x": 1}, expected_states=["s"])

    async def arrive():
        await asyncio.sleep(0)
        b.tensor_arrived("x")
        await asyncio.sleep(0)
        b.control_arg_arrived("s")

    asyncio.create_task(arrive())
    await b.wait()
    assert b.is_ready()


# ===========================================================================
# WorkerMailbox — declare_step / on_ready / wait_step_ready
# ===========================================================================


@pytest.mark.asyncio
async def test_mailbox_declare_step_and_on_ready():
    """on_ready callback fires exactly once when the step barrier completes."""
    mb = WorkerMailbox()
    fired = []

    def cb(op, step):
        fired.append((op, step))

    mb.set_ready_callback(cb)
    mb.declare_step("compute", step=0, expected_tensors={101: 2}, expected_states=[])

    tensor = TransportTensor.from_tensor(np.zeros(4))
    await mb.deposit_shards(
        "compute",
        [101],
        0,
        "sender0",
        rank=0,
        sender_world_size=4,
        payloads=[tensor.flat_values],
        counts=[4],
        meta_tensor_state_list=[0],
    )
    assert not fired
    await mb.deposit_shards(
        "compute",
        [101],
        0,
        "sender1",
        rank=1,
        sender_world_size=4,
        payloads=[tensor.flat_values],
        counts=[4],
        meta_tensor_state_list=[0],
    )
    assert fired == [("compute", 0)]


@pytest.mark.asyncio
async def test_mailbox_wait_step_ready():
    """wait_step_ready() coroutine resolves when the step is done."""
    mb = WorkerMailbox()
    mb.declare_step(
        "compute", step=5, expected_tensors={201: 1}, expected_states=["cfg"]
    )

    async def feed():
        await asyncio.sleep(0)
        tensor = TransportTensor.from_tensor(np.ones(3))
        await mb.deposit_shards(
            "compute",
            [201],
            5,
            "sender0",
            rank=0,
            sender_world_size=4,
            payloads=[tensor.flat_values],
            counts=[3],
            meta_tensor_state_list=[0],
        )
        await asyncio.sleep(0)
        await mb.deposit_control_arguments(
            "compute", 5, names=["cfg"], values=[{"lr": 0.1}]
        )

    asyncio.create_task(feed())
    await mb.wait_step_ready("compute", step=5)
    assert mb.is_step_ready("compute", step=5)


@pytest.mark.asyncio
async def test_mailbox_periodic_trigger():
    """
    Verify WorkerMailbox handles periodic operators with virtual steps.

    Virtual Step Mapping Flow:
    ```mermaid
    sequenceDiagram
        participant Sender
        participant SPMDWorker
        participant WorkerMailbox
        participant ExecutionLoop

        Sender->>SPMDWorker: receive_shards(op, name, step=100)
        SPMDWorker->>WorkerMailbox: deposit_shards(op, name, step=100)
        Note over WorkerMailbox: Virtual step mapping: 100 -> 0
        WorkerMailbox->>SPMDWorker: on_ready(op, step=0)
        SPMDWorker->>ExecutionLoop: Enqueue(op, step=0)

        Sender->>SPMDWorker: receive_shards(op, name, step=101)
        SPMDWorker->>WorkerMailbox: deposit_shards(op, name, step=101)
        Note over WorkerMailbox: Virtual step mapping: 101 -> 1
        WorkerMailbox->>SPMDWorker: on_ready(op, step=1)
        SPMDWorker->>ExecutionLoop: Enqueue(op, step=1)
    ```
    """
    mb = WorkerMailbox()
    ready_steps = []

    def on_ready(op, step):
        ready_steps.append((op, step))

    mb.set_ready_callback(on_ready)

    from fluxway.common.types.configs import PeriodicBarrierConfig

    # Declare periodic op
    mb.declare_step(
        "periodic_op",
        step=0,
        expected_tensors={301: 1},
        expected_states=[],
        barrier_config=PeriodicBarrierConfig(n=1),
    )

    # Deposit first shard (sender says step 100)
    tensor = TransportTensor.from_tensor(np.array([1.0]))
    await mb.deposit_shards(
        "periodic_op",
        [301],
        step=100,
        sender_id="s0",
        rank=0,
        sender_world_size=4,
        payloads=[tensor.flat_values],
        counts=[1],
        meta_tensor_state_list=[0],
    )

    # Deposit second shard (sender says step 101)
    await mb.deposit_shards(
        "periodic_op",
        [301],
        step=101,
        sender_id="s1",
        rank=1,
        sender_world_size=4,
        payloads=[tensor.flat_values],
        counts=[1],
        meta_tensor_state_list=[0],
    )

    # Should have two ready events with sequential virtual steps 0 and 1
    assert ready_steps == [("periodic_op", 0), ("periodic_op", 1)]

    # Verify data storage
    shards0 = await mb.get_tensor_shards("periodic_op", 301, 0)
    assert ("s0", 0) in shards0
    shards1 = await mb.get_tensor_shards("periodic_op", 301, 1)
    assert ("s1", 1) in shards1


@pytest.mark.asyncio
async def test_mailbox_is_step_ready_before_declare():
    """is_step_ready returns False when declare_step was not called."""
    mb = WorkerMailbox()
    assert not mb.is_step_ready("optimize", step=99)


@pytest.mark.asyncio
async def test_mailbox_no_barrier_without_declare():
    """wait_step_ready raises if declare_step was never called."""
    mb = WorkerMailbox()
    with pytest.raises(RuntimeError, match="declare_step"):
        await mb.wait_step_ready("compute", step=7)


# ===========================================================================
# WorkerMailbox — state data
# ===========================================================================


@pytest.mark.asyncio
async def test_mailbox_state_put_get():
    mb = WorkerMailbox()
    await mb.deposit_control_arguments("compute", 1, names=["lr"], values=[0.001])
    assert await mb.has_control_argument("compute", "lr", 1)
    assert await mb.get_control_argument("compute", "lr", 1) == 0.001


@pytest.mark.asyncio
async def test_mailbox_get_all_states():
    mb = WorkerMailbox()
    await mb.deposit_control_arguments("compute", 0, names=["a"], values=[1])
    await mb.deposit_control_arguments("compute", 0, names=["b"], values=[2])
    states = await mb.get_all_control_arguments("compute", 0)
    assert states == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_mailbox_periodic_state():
    """Verify WorkerMailbox handles periodic operators with states."""
    mb = WorkerMailbox()
    ready_steps = []
    mb.set_ready_callback(lambda op, step: ready_steps.append((op, step)))

    from fluxway.common.types.configs import PeriodicBarrierConfig

    mb.declare_step(
        "periodic_op",
        step=0,
        expected_tensors={},
        expected_states=["s"],
        barrier_config=PeriodicBarrierConfig(n=1),
    )

    await mb.deposit_control_arguments(
        "periodic_op", step=50, names=["s"], values=["v1"]
    )
    await mb.deposit_control_arguments(
        "periodic_op", step=51, names=["s"], values=["v2"]
    )

    assert ready_steps == [("periodic_op", 0), ("periodic_op", 1)]
    assert await mb.get_control_argument("periodic_op", "s", 0) == "v1"
    assert await mb.get_control_argument("periodic_op", "s", 1) == "v2"


# ===========================================================================
# WorkerMailbox — tensor shards
# ===========================================================================


@pytest.mark.asyncio
async def test_mailbox_tensor_put_get():
    mb = WorkerMailbox()
    d = np.array([1.0, 2.0])
    tensor = TransportTensor.from_tensor(d)
    await mb.deposit_shards(
        "compute",
        [401],
        0,
        "sender0",
        rank=0,
        sender_world_size=4,
        payloads=[tensor.flat_values],
        counts=[2],
        meta_tensor_state_list=[0],
    )
    shards = await mb.get_tensor_shards("compute", 401, 0)
    assert ("sender0", 0) in shards
    np.testing.assert_array_equal(shards[("sender0", 0)][0].flat_values, d)


@pytest.mark.asyncio
async def test_mailbox_tensor_multi_rank():
    mb = WorkerMailbox()
    tensor_zeros = TransportTensor.from_tensor(np.zeros(4))
    await mb.deposit_shards(
        "compute",
        [501],
        1,
        "sender0",
        rank=0,
        sender_world_size=4,
        payloads=[tensor_zeros.flat_values],
        counts=[4],
        meta_tensor_state_list=[0],
    )
    tensor_ones = TransportTensor.from_tensor(np.ones(4))
    await mb.deposit_shards(
        "compute",
        [501],
        1,
        "sender1",
        rank=1,
        sender_world_size=4,
        payloads=[tensor_ones.flat_values],
        counts=[4],
        meta_tensor_state_list=[0],
    )
    assert await mb.get_tensor_shard_count("compute", 501, 1) == 2


@pytest.mark.asyncio
async def test_mailbox_op_isolation():
    mb = WorkerMailbox()
    await mb.deposit_shards(
        "compute",
        [601],
        0,
        "sender0",
        rank=0,
        sender_world_size=4,
        payloads=[np.array([1.0])],
        counts=[1],
        meta_tensor_state_list=[0],
    )
    await mb.deposit_shards(
        "optimize",
        [601],
        0,
        "sender0",
        rank=0,
        sender_world_size=4,
        payloads=[np.array([2.0])],
        counts=[1],
        meta_tensor_state_list=[0],
    )
    fwd = await mb.get_tensor_shards("compute", 601, 0)
    bwd = await mb.get_tensor_shards("optimize", 601, 0)
    np.testing.assert_array_equal(fwd[("sender0", 0)][0].flat_values, [1.0])
    np.testing.assert_array_equal(bwd[("sender0", 0)][0].flat_values, [2.0])


# ===========================================================================
# WorkerMailbox — zero-copy
# ===========================================================================


@pytest.mark.asyncio
async def test_mailbox_zero_copy_writes_into_buffer():
    mb = WorkerMailbox()
    buf = np.zeros((4, 2), dtype=np.float32)
    await mb.register_zero_copy_slot(
        "compute", 701, step=0, target_buffers=[buf], expect_offsets={0: 0}
    )
    await mb.deposit_shards(
        "compute",
        [701],
        step=0,
        sender_id="s0",
        rank=0,
        sender_world_size=4,
        payloads=[np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
        counts=[2],
        meta_tensor_state_list=[0],
    )
    buffers = await mb.get_zero_copy_buffers("compute", 701, step=0)
    assert buffers is not None
    np.testing.assert_array_equal(buffers[0][:2], [[1.0, 2.0], [3.0, 4.0]])


# ===========================================================================
# WorkerMailbox — cleanup / backpressure
# ===========================================================================


@pytest.mark.asyncio
async def test_mailbox_clear_step():
    mb = WorkerMailbox()
    await mb.deposit_control_arguments("compute", 1, names=["s"], values=["x"])
    await mb.deposit_control_arguments("compute", 2, names=["s"], values=["y"])
    tensor = TransportTensor.from_tensor(np.zeros(2))
    await mb.deposit_shards(
        "compute",
        [801],
        1,
        "sender0",
        0,
        4,
        payloads=[tensor.flat_values],
        counts=[2],
        meta_tensor_state_list=[0],
    )
    mb.declare_step("compute", 1, {}, [])

    await mb.clear_step("compute", 1)

    assert not await mb.has_control_argument("compute", "s", 1)
    assert await mb.has_control_argument("compute", "s", 2)
    assert await mb.get_tensor_shard_count("compute", 801, 1) == 0
    assert not mb.is_step_ready("compute", 1)  # barrier also cleared


@pytest.mark.asyncio
async def test_mailbox_get_size():
    mb = WorkerMailbox()
    assert await mb.get_size() == 0
    await mb.deposit_control_arguments("compute", 0, names=["s"], values=[None])
    tensor = TransportTensor.from_tensor(np.zeros(2))
    await mb.deposit_shards(
        "compute",
        [901],
        0,
        "s0",
        0,
        1,
        payloads=[tensor.flat_values],
        counts=[2],
        meta_tensor_state_list=[0],
    )
    await mb.deposit_shards(
        "compute",
        [901],
        0,
        "s1",
        1,
        1,
        payloads=[tensor.flat_values],
        counts=[2],
        meta_tensor_state_list=[0],
    )
    assert await mb.get_size() == 3


# ===========================================================================
# Worker / OperatorManager integration
# ===========================================================================


def test_layout_mesh():
    mesh = DeviceMesh(axis_names=("data", "model"), device_shape=(4, 2))
    assert mesh.size == 8
    assert mesh.get_coordinate(5) == (2, 1)
    assert mesh.get_rank((2, 1)) == 5


@pytest.mark.asyncio
async def test_spmd_config_with_mesh():
    mesh = DeviceMesh(axis_names=("data", "model"), device_shape=(4, 2))
    config = _make_config(mesh, "compute")
    assert config.num_workers == 8


def test_operator_manager_mesh():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    config = _make_config(mesh, "compute", "optimize", "apply")
    assert OperatorManager(config, rank=0)._mesh is mesh


@pytest.mark.asyncio
async def test_spmd_worker_init():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    config = _make_config(mesh, "compute", "optimize", "apply")
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(config, rank=0)
    worker.setup_and_start(local_rank=0, local_world_size=1, node_rank=0, num_nodes=1)
    assert worker._mesh is mesh
    assert worker._op_manager._mesh is mesh


@pytest.mark.asyncio
async def test_receive_multi_tensor_data():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    config = _make_config(mesh, "compute")
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(config, rank=0)
    worker.setup_and_start(local_rank=0, local_world_size=1, node_rank=0, num_nodes=1)

    worker._mailbox.declare_step(
        "compute", step=1, expected_tensors={998: 1, 999: 1}, expected_states=[]
    )
    worker._op_manager.get_operators_for_tuuid = MagicMock(return_value=["compute"])
    worker._op_manager.get_sender_world_size = MagicMock(return_value=2)

    t1 = TransportTensor.from_tensor(np.array([1, 2]))
    t2 = TransportTensor.from_tensor(np.array([3, 4]))

    # Two TUUIDs in one RPC call: flat payloads/counts for each
    await worker.receive_shards(
        tuuids=[998, 999],
        step=1,
        sender_id="s0",
        sender_rank=0,
        payloads=[t1.flat_values, t2.flat_values],
        counts=[2, 2],
        meta_tensor_state_list=[0, 0],
    )
    s1 = await worker._mailbox.get_tensor_shards("compute", 998, 1)
    s2 = await worker._mailbox.get_tensor_shards("compute", 999, 1)
    np.testing.assert_array_equal(s1[("s0", 0)][0].flat_values, [1, 2])
    np.testing.assert_array_equal(s2[("s0", 0)][0].flat_values, [3, 4])


@pytest.mark.asyncio
async def test_mailbox_zero_copy_ragged():
    mb = WorkerMailbox()
    # For ragged, we expect 2 buffers: data and lengths
    data_buf = np.zeros((10,), dtype=np.float32)
    len_buf = np.zeros((4,), dtype=np.int32)

    await mb.register_zero_copy_slot(
        "compute",
        1001,
        step=0,
        target_buffers=[data_buf, len_buf],
        expect_offsets={0: [0, 0]},
    )

    # Create a ragged tensor: 2 elements, counts [3, 2], values [1,1,1, 2,2]
    flat_values = np.array([1, 1, 1, 2, 2], dtype=np.float32)
    splits = np.array([0, 3, 5], dtype=np.int32)
    tt = TransportTensor(flat_values, nested_splits=[splits])

    await mb.deposit_shards(
        "compute",
        [1001],
        0,
        "s0",
        0,
        1,
        payloads=[tt.flat_values, np.array([3, 2], dtype=np.int32)],
        counts=[5, 2],
        meta_tensor_state_list=[1],  # meta_state 1 for ragged
    )

    # Verify buffers from mailbox (handles JAX-style non-inplace updates)
    buffers = await mb.get_zero_copy_buffers("compute", 1001, step=0)
    assert buffers is not None
    # Verify data buffer
    np.testing.assert_array_equal(buffers[0][:5], flat_values)
    # Verify lengths buffer
    np.testing.assert_array_equal(buffers[1][:2], [3, 2])


def test_replicate_operator_manager():
    from fluxway.common.types.configs import OperatorMeta

    mesh = DeviceMesh(axis_names=("x", "y"), device_shape=(2, 2))

    replicated_spec = ShardingSpec.chunk(PartitionSpec())
    meta = OperatorMeta(
        name="test_op",
        input_arguments={ArgumentType.DATA.value: ["a"]},
        output_arguments={},
        input_sharding={"a": replicated_spec},
        output_sharding={},
    )

    def dummy_op(a):
        pass

    dummy_op._flux_operator_meta = meta

    from fluxway.common.types.configs import SPMDTopology, SPMDTopologyInfo

    config = SPMDConfig(
        spmd_id="test",
        num_workers=4,
        mesh=mesh,
        operators={"test_op": dummy_op},
        operator_metas={"test_op": meta},
        topology={
            "test_op": SPMDTopology(
                predecessors={
                    "test": SPMDTopologyInfo(
                        spmd_id="test",
                        tuuid_sharding={1001: replicated_spec},
                        tuuid_map={"a": 1001},
                    )
                }
            )
        },
    )

    manager = OperatorManager(config, rank=0)

    # Query by the registered op name
    spec = manager.get_input_sharding("test_op")
    assert 1001 in spec
    # get_input_sharding returns unbound ShardingSpec; bind to mesh to get replica_size
    bound = spec[1001].bind(mesh)
    assert bound.replica_size == 4


@pytest.mark.asyncio
async def test_send_output_none_mesh():
    # Test fallback to default mesh when config.mesh is None
    # This specifically verifies the fix for ShardingSpec.chunk(...).bind(None)
    # Use a dummy mesh for config to satisfy OperatorManager initialization
    config = _make_config(DeviceMesh(axis_names=("dp",), device_shape=(1,)), "compute")
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(config, rank=0)

    # Mock successor info to trigger _send_output logic
    from unittest.mock import MagicMock

    worker._op_manager = MagicMock()
    test_mesh = DeviceMesh(axis_names=("data",), device_shape=(1,))
    # _send_output calls source_logical.bind(self._mesh) so these must be unbound ShardingSpecs
    unbound_spec = ShardingSpec.chunk(PartitionSpec())

    worker._op_manager.get_output_sharding.return_value = {123: unbound_spec}
    worker._op_manager.get_output_tuuids.return_value = [123]
    worker._op_manager.get_successor_ids.return_value = ["succ"]
    worker._op_manager.get_output_state_names.return_value = []
    worker._op_manager.get_output_tuuid_to_name.return_value = {123: "out"}

    op_meta_mock = MagicMock()
    op_meta_mock.output_treedef = None
    worker._op_manager.get_operator_metadata.return_value = op_meta_mock

    disp_mock = MagicMock()
    disp_mock.interleave_unit_size = None
    worker._op_manager.get_dispatcher.return_value = disp_mock

    succ_info = MagicMock()
    succ_info.is_self_loop = False
    succ_info.tuuid_sharding = {123: unbound_spec}
    succ_info.mesh = test_mesh
    succ_info.is_interleaved = False
    succ_info.interleave_unit_size = None
    worker._op_manager.get_successor_info.return_value = succ_info

    # Mock successor handlers
    handler = MagicMock()
    from fluxway.rpc.protocol import DataPlaneCallable

    worker._successor_handlers = {"succ": {0: DataPlaneCallable(handler)}}

    from fluxway.spmd.worker.operator_manager.dispatcher import (
        ExecutionStatus,
        ExecutionResult,
    )

    data = [np.array([1, 2, 3])]
    result = ExecutionResult(status=ExecutionStatus.SUCCESS, data=data)
    await worker._send_output("compute", result, exec_step=0)

    assert handler.receive_shards.remote.called
