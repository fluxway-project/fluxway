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
import numpy as np
import pytest

sys.path.append(os.getcwd())

from fluxway.common.sync import ArrivalBarrier
from fluxway.spmd.worker.mailbox.mailbox import StepBarrier, WorkerMailbox
from fluxway.spmd.worker.mailbox import OpType
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.common.types.configs import SPMDConfig
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.communication import TransportReceiver


# ---------------------------------------------------------------------------
# Dummy user functions
# ---------------------------------------------------------------------------


def forward_fn(state, batch):
    return state


def backward_fn(state, grad):
    return state


def update_fn(state, grad):
    return state


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


# ===========================================================================
# StepBarrier
# ===========================================================================


def test_step_barrier_fires_on_completion():
    """Fires when all tensor shards AND all states have arrived."""
    b = StepBarrier(expected_tensors={"emb": 2, "attn": 1}, expected_states=["lr"])
    assert not b.is_ready()
    b.tensor_arrived("emb")
    b.tensor_arrived("emb")
    b.tensor_arrived("attn")
    assert not b.is_ready()  # state "lr" still missing
    b.state_arrived("lr")
    assert b.is_ready()


def test_step_barrier_partial_not_ready():
    """Partial arrivals do not fire the barrier."""
    b = StepBarrier(expected_tensors={"x": 3}, expected_states=[])
    b.tensor_arrived("x")
    b.tensor_arrived("x")
    assert not b.is_ready()


def test_step_barrier_no_expectations_fires_immediately():
    """Empty expectations → already ready."""
    b = StepBarrier(expected_tensors={}, expected_states=[])
    assert b.is_ready()


def test_step_barrier_unknown_name_ignored():
    """Arrivals for unexpected names are silently ignored."""
    b = StepBarrier(expected_tensors={"x": 1}, expected_states=[])
    assert b.tensor_arrived("unknown_tensor") is False
    assert not b.is_ready()
    b.tensor_arrived("x")
    assert b.is_ready()


@pytest.mark.asyncio
async def test_step_barrier_async_wait():
    b = StepBarrier(expected_tensors={"x": 1}, expected_states=["s"])

    async def arrive():
        await asyncio.sleep(0)
        b.tensor_arrived("x")
        await asyncio.sleep(0)
        b.state_arrived("s")

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
    mb.declare_step(
        OpType.FORWARD, step=0, expected_tensors={"x": 2}, expected_states=[]
    )

    tensor = TransportTensor.from_tensor(np.zeros(4))
    await mb.deposit_shards(
        OpType.FORWARD, "x", 0, rank=0, payloads=[tensor.flat_values], counts=[4]
    )
    assert not fired
    await mb.deposit_shards(
        OpType.FORWARD, "x", 0, rank=1, payloads=[tensor.flat_values], counts=[4]
    )
    assert fired == [(OpType.FORWARD, 0)]


@pytest.mark.asyncio
async def test_mailbox_wait_step_ready():
    """wait_step_ready() coroutine resolves when the step is done."""
    mb = WorkerMailbox()
    mb.declare_step(
        OpType.FORWARD, step=5, expected_tensors={"a": 1}, expected_states=["cfg"]
    )

    async def feed():
        await asyncio.sleep(0)
        tensor = TransportTensor.from_tensor(np.ones(3))
        await mb.deposit_shards(
            OpType.FORWARD, "a", 5, rank=0, payloads=[tensor.flat_values], counts=[3]
        )
        await asyncio.sleep(0)
        await mb.deposit_states(OpType.FORWARD, 5, names=["cfg"], values=[{"lr": 0.1}])

    asyncio.create_task(feed())
    await mb.wait_step_ready(OpType.FORWARD, step=5)
    assert mb.is_step_ready(OpType.FORWARD, step=5)


@pytest.mark.asyncio
async def test_mailbox_is_step_ready_before_declare():
    """is_step_ready returns False when declare_step was not called."""
    mb = WorkerMailbox()
    assert not mb.is_step_ready(OpType.BACKWARD, step=99)


@pytest.mark.asyncio
async def test_mailbox_no_barrier_without_declare():
    """wait_step_ready raises if declare_step was never called."""
    mb = WorkerMailbox()
    with pytest.raises(RuntimeError, match="declare_step"):
        await mb.wait_step_ready(OpType.FORWARD, step=7)


# ===========================================================================
# WorkerMailbox — state data
# ===========================================================================


@pytest.mark.asyncio
async def test_mailbox_state_put_get():
    mb = WorkerMailbox()
    await mb.deposit_states(OpType.FORWARD, 1, names=["lr"], values=[0.001])
    assert await mb.has_state(OpType.FORWARD, "lr", 1)
    assert await mb.get_state(OpType.FORWARD, "lr", 1) == 0.001


@pytest.mark.asyncio
async def test_mailbox_get_all_states():
    mb = WorkerMailbox()
    await mb.deposit_states(OpType.FORWARD, 0, names=["a"], values=[1])
    await mb.deposit_states(OpType.FORWARD, 0, names=["b"], values=[2])
    states = await mb.get_all_states(OpType.FORWARD, 0)
    assert states == {"a": 1, "b": 2}


# ===========================================================================
# WorkerMailbox — tensor shards
# ===========================================================================


@pytest.mark.asyncio
async def test_mailbox_tensor_put_get():
    mb = WorkerMailbox()
    d = np.array([1.0, 2.0])
    tensor = TransportTensor.from_tensor(d)
    await mb.deposit_shards(
        OpType.FORWARD, "x", 0, rank=0, payloads=[tensor.flat_values], counts=[2]
    )
    shards = await mb.get_tensor_shards(OpType.FORWARD, "x", 0)
    assert 0 in shards
    np.testing.assert_array_equal(shards[0].flat_values, d)


@pytest.mark.asyncio
async def test_mailbox_tensor_multi_rank():
    mb = WorkerMailbox()
    tensor_zeros = TransportTensor.from_tensor(np.zeros(4))
    await mb.deposit_shards(
        OpType.FORWARD, "e", 1, rank=0, payloads=[tensor_zeros.flat_values], counts=[4]
    )
    tensor_ones = TransportTensor.from_tensor(np.ones(4))
    await mb.deposit_shards(
        OpType.FORWARD, "e", 1, rank=1, payloads=[tensor_ones.flat_values], counts=[4]
    )
    assert await mb.get_tensor_shard_count(OpType.FORWARD, "e", 1) == 2


@pytest.mark.asyncio
async def test_mailbox_op_isolation():
    mb = WorkerMailbox()
    await mb.deposit_shards(
        OpType.FORWARD, "g", 0, rank=0, payloads=[np.array([1.0])], counts=[1]
    )
    await mb.deposit_shards(
        OpType.BACKWARD, "g", 0, rank=0, payloads=[np.array([2.0])], counts=[1]
    )
    fwd = await mb.get_tensor_shards(OpType.FORWARD, "g", 0)
    bwd = await mb.get_tensor_shards(OpType.BACKWARD, "g", 0)
    np.testing.assert_array_equal(fwd[0].flat_values, [1.0])
    np.testing.assert_array_equal(bwd[0].flat_values, [2.0])


# ===========================================================================
# WorkerMailbox — zero-copy
# ===========================================================================


@pytest.mark.asyncio
async def test_mailbox_zero_copy_writes_into_buffer():
    mb = WorkerMailbox()
    buf = np.zeros((4, 2), dtype=np.float32)
    await mb.register_zero_copy_slot(
        OpType.FORWARD, "emb", step=0, target_buffers=[buf], expect_offsets={0: 0}
    )
    await mb.deposit_shards(
        OpType.FORWARD,
        "emb",
        step=0,
        rank=0,
        payloads=[np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)],
        counts=[2],
    )
    buffers = await mb.get_zero_copy_buffers(OpType.FORWARD, "emb", step=0)
    assert buffers is not None
    np.testing.assert_array_equal(buffers[0][:2], [[1.0, 2.0], [3.0, 4.0]])


# ===========================================================================
# WorkerMailbox — cleanup / backpressure
# ===========================================================================


@pytest.mark.asyncio
async def test_mailbox_clear_step():
    mb = WorkerMailbox()
    await mb.deposit_states(OpType.FORWARD, 1, names=["s"], values=["x"])
    await mb.deposit_states(OpType.FORWARD, 2, names=["s"], values=["y"])
    tensor = TransportTensor.from_tensor(np.zeros(2))
    await mb.deposit_shards(
        OpType.FORWARD, "t", 1, rank=0, payloads=[tensor.flat_values], counts=[2]
    )
    mb.declare_step(OpType.FORWARD, 1, {}, [])

    await mb.clear_step(OpType.FORWARD, 1)

    assert not await mb.has_state(OpType.FORWARD, "s", 1)
    assert await mb.has_state(OpType.FORWARD, "s", 2)
    assert await mb.get_tensor_shard_count(OpType.FORWARD, "t", 1) == 0
    assert not mb.is_step_ready(OpType.FORWARD, 1)  # barrier also cleared


@pytest.mark.asyncio
async def test_mailbox_get_size():
    mb = WorkerMailbox()
    assert await mb.get_size() == 0
    await mb.deposit_states(OpType.FORWARD, 0, names=["s"], values=[None])
    tensor = TransportTensor.from_tensor(np.zeros(2))
    await mb.deposit_shards(
        OpType.FORWARD, "t", 0, rank=0, payloads=[tensor.flat_values], counts=[2]
    )
    await mb.deposit_shards(
        OpType.FORWARD, "t", 0, rank=1, payloads=[tensor.flat_values], counts=[2]
    )
    assert await mb.get_size() == 3


# ===========================================================================
# Worker / OperatorManager integration
# ===========================================================================


def test_layout_mesh():
    mesh = DeviceMesh((4, 2), axis_names=("data", "model"))
    assert mesh.size == 8
    assert mesh.get_coordinate(5) == (2, 1)
    assert mesh.get_rank((2, 1)) == 5


def test_spmd_config_with_mesh():
    mesh = DeviceMesh((4, 2), axis_names=("data", "model"))
    config = SPMDConfig(mesh=mesh, user_forward_fn=forward_fn)
    assert config.has_mesh and config.get_mesh_size() == 8


def test_operator_manager_mesh():
    mesh = DeviceMesh((2,), axis_names=("data",))
    config = SPMDConfig(
        mesh=mesh,
        user_forward_fn=forward_fn,
        user_backward_fn=backward_fn,
        user_update_fn=update_fn,
    )
    assert OperatorManager(config, rank=0)._mesh is mesh


@pytest.mark.asyncio
async def test_spmd_worker_init():
    mesh = DeviceMesh((2,), axis_names=("data",))
    config = SPMDConfig(
        mesh=mesh,
        user_forward_fn=forward_fn,
        user_backward_fn=backward_fn,
        user_update_fn=update_fn,
    )
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(config, rank=0)
    assert worker._mesh is mesh
    assert worker._op_manager._mesh is mesh


@pytest.mark.asyncio
async def test_receive_multi_tensor_data():
    mesh = DeviceMesh((2,), axis_names=("data",))
    config = SPMDConfig(mesh=mesh, user_forward_fn=forward_fn)
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(config, rank=0)

    t1 = TransportTensor.from_tensor(np.array([1, 2]))
    t2 = TransportTensor.from_tensor(np.array([3, 4]))

    await worker.receive_shards(
        OpType.FORWARD,
        "multi",
        step=1,
        sender_rank=0,
        payloads=[t1.flat_values, t2.flat_values],
        counts=[2, 2],
        is_ragged=False,
        has_indices=False,
    )
    shards = await worker._mailbox.get_tensor_shards(OpType.FORWARD, "multi", 1)
    received = shards[0]
    assert isinstance(received, list)
    assert len(received) == 2
    np.testing.assert_array_equal(received[0].flat_values, [1, 2])
    np.testing.assert_array_equal(received[1].flat_values, [3, 4])


@pytest.mark.asyncio
async def test_mailbox_zero_copy_ragged():
    mb = WorkerMailbox()
    # For ragged, we expect 2 buffers: data and lengths
    data_buf = np.zeros((10,), dtype=np.float32)
    len_buf = np.zeros((4,), dtype=np.int32)

    await mb.register_zero_copy_slot(
        OpType.FORWARD,
        "ragged",
        step=0,
        target_buffers=[data_buf, len_buf],
        expect_offsets={0: [0, 0]},
    )

    # Create a ragged tensor: 2 elements, counts [3, 2], values [1,1,1, 2,2]
    flat_values = np.array([1, 1, 1, 2, 2], dtype=np.float32)
    splits = np.array([0, 3, 5], dtype=np.int32)
    tt = TransportTensor(flat_values, nested_splits=[splits])

    await mb.deposit_shards(
        OpType.FORWARD,
        "ragged",
        step=0,
        rank=0,
        payloads=[tt.flat_values, np.array([3, 2], dtype=np.int32)],
        counts=[5, 2],
        is_ragged=True,
        has_indices=False,
    )

    # Verify buffers from mailbox (handles JAX-style non-inplace updates)
    buffers = await mb.get_zero_copy_buffers(OpType.FORWARD, "ragged", step=0)
    assert buffers is not None
    # Verify data buffer
    np.testing.assert_array_equal(buffers[0][:5], flat_values)
    # Verify lengths buffer
    np.testing.assert_array_equal(buffers[1][:2], [3, 2])


def test_replicate_operator_manager():
    from fluxway.common.types.configs import OperatorMeta, ArgumentType

    mesh = DeviceMesh((2, 2), axis_names=("x", "y"))

    meta = OperatorMeta(
        name="test_op",
        input_arguments={ArgumentType.TENSOR.value: ["a"]},
        output_arguments={},
        input_sharding={"a": {"type": "replicated"}},
        output_sharding={},
    )

    def my_forward(a):
        pass

    my_forward._flux_operator_meta = meta

    config = SPMDConfig(
        spmd_id="test",
        worker_count=4,
        mesh=mesh,
        user_forward_fn=my_forward,
        user_backward_fn=None,
        user_update_fn=None,
    )

    manager = OperatorManager(config, rank=0)

    spec = manager.get_input_sharding(OpType.FORWARD)
    assert "a" in spec
    assert spec["a"].replica_size == 4
