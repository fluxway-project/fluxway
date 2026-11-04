"""
Tests for the unified completion notification mechanism and SPMD Worker infrastructure.

Covers:
- ArrivalBarrier: N-arrival primitive shared across transport and mailbox layers
- StepBarrier: per-step composite barrier in WorkerMailbox
- WorkerMailbox: declared tensor-frame/control arrivals and exact-step readiness
- Worker integration: mesh, config, data reception
"""

import sys
import os
import asyncio
from fluxway.spmd.worker.lifecycle import WorkerExecutionLifecycle
import numpy as np
import pytest

sys.path.append(os.getcwd())

from fluxway.common.sync import ArrivalBarrier
from fluxway.common.compiled_graph import (
    DependencySpec,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
)
from fluxway.spmd.worker.mailbox.mailbox import StepBarrier, WorkerMailbox
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.spmd.worker.source_scope import SourceExecutionScope
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.common.types.configs import SPMDConfig
from fluxway.common.types import DrainBoundary, DrainPhase
from fluxway.common.operator_policy import ArgumentType, PeriodicBarrierConfig
from fluxway.common.operator_instances import OperatorInstanceContainer
from fluxway import OperatorSpec
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.sharding import ShardingSpec, PartitionSpec
from fluxway.transport.manifest import PayloadRole
from tests.compiled_graph_fixtures import (
    compile_test_graph,
    single_spmd_config,
    spmd_config_from_graph,
)
from tests.spmd.mailbox_frame_helpers import TensorFrameEntry, deposit_tensor_frame

# ---------------------------------------------------------------------------
# Helpers: build minimal SPMDConfig with registered operators
# ---------------------------------------------------------------------------


class _WorkerOperatorOwner:
    def produce(self):
        return None

    def consume(self, value):
        return value

    def compute(self):
        return None

    def optimize(self):
        return None

    def apply(self):
        return None

    def test_op(self):
        return None


def _make_null_meta(name: str) -> OperatorSpec:
    return OperatorSpec(
        name=name,
        owner=_WorkerOperatorOwner,
        method_name=name,
    )


def _make_config(mesh: DeviceMesh, *op_names: str) -> SPMDConfig:
    operators = {op: _make_null_meta(op) for op in op_names}
    return single_spmd_config(
        name="spmd-worker",
        spmd_id="test",
        mesh=mesh,
        operators=operators,
    )


# Generic dummy functions for testing
def compute_fn(state, batch):
    return state


def optimize_fn(state, grad):
    return state


def apply_fn(state, grad):
    return state


def _worker_cls():
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    return cls.modified_class if cls else SPMDWorker


@pytest.mark.asyncio
async def test_checkpoint_restore_restores_scheduler_watermark() -> None:
    worker_type = SPMDWorker.__ray_actor_class__  # type: ignore[attr-defined]
    worker = worker_type.__new__(worker_type)
    worker._execution_lifecycle = WorkerExecutionLifecycle()
    worker._source_scope = SourceExecutionScope(
        acquire=worker._acquire_source_work,
        spawn=worker._spawn_background_task,
        completed=lambda op: worker._completed_step_count[op],
        depth=worker._operator_pipeline_depth,
    )
    worker._completed_step_count = {"Reader": 0, "Dense": 0}
    worker._scheduled_step_count = {"Reader": 0, "Dense": 0}
    worker._drain_boundary = DrainBoundary()
    worker._source_operator_names = ("Reader",)
    worker.spmd_id = "Reader"
    worker.rank = 0
    worker.config = type("Config", (), {"max_steps": None})()
    worker._op_manager = type(
        "OperatorManager",
        (),
        {"is_source_operator": lambda _self, name: name == "Reader"},
    )()
    worker._restore_checkpoint_runtime_state(
        {
            "drain_boundary": DrainBoundary(committed_cut=1).to_dict(),
        }
    )

    assert worker._scheduled_step_count == {"Reader": 1, "Dense": 1}


def test_checkpoint_restore_discards_transient_drain_boundary_state() -> None:
    worker_type = SPMDWorker.__ray_actor_class__  # type: ignore[attr-defined]
    worker = worker_type.__new__(worker_type)
    worker._execution_lifecycle = WorkerExecutionLifecycle()
    worker._source_scope = SourceExecutionScope(
        acquire=worker._acquire_source_work,
        spawn=worker._spawn_background_task,
        completed=lambda op: worker._completed_step_count[op],
        depth=worker._operator_pipeline_depth,
    )
    worker._completed_step_count = {"Reader": 0}
    worker._scheduled_step_count = {"Reader": 0}
    worker._drain_boundary = DrainBoundary()
    worker._source_operator_names = ("Reader",)
    worker.spmd_id = "Reader"
    worker.rank = 0
    worker.config = type("Config", (), {"max_steps": None})()
    worker._op_manager = type(
        "OperatorManager",
        (),
        {"is_source_operator": lambda _self, name: name == "Reader"},
    )()

    worker._restore_checkpoint_runtime_state(
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

    assert worker._drain_boundary == DrainBoundary(
        phase=DrainPhase.OPEN,
        drained_cut=9,
        committed_cut=9,
    )
    assert worker._completed_step_count == {"Reader": 9}
    assert worker._scheduled_step_count == {"Reader": 9}


def test_checkpoint_restore_rejects_duplicate_runtime_cursor() -> None:
    worker_type = SPMDWorker.__ray_actor_class__  # type: ignore[attr-defined]
    worker = worker_type.__new__(worker_type)
    worker._execution_lifecycle = WorkerExecutionLifecycle()
    worker._source_scope = SourceExecutionScope(
        acquire=worker._acquire_source_work,
        spawn=worker._spawn_background_task,
        completed=lambda op: worker._completed_step_count[op],
        depth=worker._operator_pipeline_depth,
    )
    worker._completed_step_count = {"lookup": 0}
    worker._scheduled_step_count = {"lookup": 0}
    worker._drain_boundary = DrainBoundary()
    worker._source_operator_names = ("lookup",)
    worker.spmd_id = "Embedding"
    worker.rank = 0
    worker.config = type("Config", (), {"max_steps": None})()
    worker._op_manager = type(
        "OperatorManager",
        (),
        {"is_source_operator": lambda _self, name: name == "lookup"},
    )()

    with pytest.raises(ValueError, match="only drain_boundary"):
        worker._restore_checkpoint_runtime_state(
            {
                "completed_step_count": {"lookup": 4, "update": 4},
                "drain_boundary": DrainBoundary(committed_cut=4).to_dict(),
            }
        )


def _values_frame_entry(
    transfer_code: int,
    values,
    valid_count: int,
    *,
    expected_count: int = 1,
) -> TensorFrameEntry:
    return TensorFrameEntry(
        transfer_code=transfer_code,
        payloads=((PayloadRole.VALUES, values, valid_count),),
        expected_count=expected_count,
    )


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
    calls = {
        "preinstantiate": [],
        "manager_static_args": None,
        "heartbeat": 0,
        "run_loop": 0,
    }

    class FakeOperatorManager:
        def __init__(
            self,
            config,
            rank,
            spmd_id,
            *,
            static_args,
            instance_container=None,
            local_rank=None,
            local_world_size=None,
            node_rank=0,
            num_nodes=1,
            worker_process_runtime=None,
        ):
            self.payload = (
                config,
                rank,
                spmd_id,
                local_rank,
                local_world_size,
                node_rank,
                num_nodes,
                instance_container,
                worker_process_runtime,
            )
            calls["manager_static_args"] = static_args

        def is_source_operator(self, op_name):
            return False

        def get_operator_spec(self, op_name):
            return config.operator_specs[op_name]

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
    )
    worker.setup_and_start(local_rank=1, local_world_size=3, node_rank=0, num_nodes=2)

    assert len(calls["preinstantiate"]) == 1
    rank, kwargs = calls["preinstantiate"][0]
    assert rank == 2
    static_args = kwargs.pop("static_args")
    assert static_args == {}
    assert kwargs.pop("instance_container") is worker._instance_container
    assert calls["manager_static_args"] is static_args
    assert kwargs == {
        "local_rank": 1,
        "local_world_size": 3,
        "node_rank": 0,
        "num_nodes": 2,
        "rank0_address": None,
    }
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
    )

    async def run_forever():
        await asyncio.Future()

    background_task_ran = asyncio.Event()

    async def background_task():
        background_task_ran.set()
        await asyncio.Future()

    worker._execution_queue = asyncio.Queue()  # type: ignore[assignment]
    worker._execution_queues = {"__default__": worker._execution_queue}  # type: ignore[assignment]
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
    assert sentinel == (999, 0, 0, "__stop__")


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

        def get_operator_spec(self, op_name):
            return config.operator_specs[op_name]

    monkeypatch.setattr(
        "fluxway.spmd.worker.worker.OperatorManager", FakeOperatorManager
    )

    worker = worker_cls(
        config=config,
        rank=0,
        spmd_id="spmd-log",
        driver_handle=object(),
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
        "test", 0, expected_transfers={101: 2, 102: 1}, expected_states=["lr"]
    )
    assert not b.is_ready()
    b.transfer_arrived(101)
    b.transfer_arrived(101)
    b.transfer_arrived(102)
    assert not b.is_ready()  # state "lr" still missing
    b.control_arg_arrived("lr")
    assert b.is_ready()


def test_step_barrier_partial_not_ready():
    """Partial arrivals do not fire the barrier."""
    b = StepBarrier("test", 0, expected_transfers={103: 3}, expected_states=[])
    b.transfer_arrived(103)
    b.transfer_arrived(103)
    assert not b.is_ready()


def test_step_barrier_no_expectations_fires_immediately():
    """Empty expectations → already ready."""
    b = StepBarrier("test", 0, expected_transfers={}, expected_states=[])
    assert b.is_ready()


def test_step_barrier_unknown_name_ignored():
    """Arrivals for unexpected names are silently ignored."""
    b = StepBarrier("test", 0, expected_transfers={103: 1}, expected_states=[])
    assert b.transfer_arrived(999) is False
    assert not b.is_ready()
    b.transfer_arrived(103)
    assert b.is_ready()


@pytest.mark.asyncio
async def test_step_barrier_periodic():
    """Verify StepBarrier in periodic mode fires for every arrival."""
    # Eager op expects 1 shard for 'x'
    b = StepBarrier(
        "test", 0, expected_transfers={103: 1}, expected_states=[], periodic=True
    )

    # First arrival
    fired = b.transfer_arrived(103)
    assert fired is True

    # Second arrival
    fired = b.transfer_arrived(103)
    assert fired is True


@pytest.mark.asyncio
async def test_step_barrier_async_wait():
    b = StepBarrier("test", 0, expected_transfers={103: 1}, expected_states=["s"])

    async def arrive():
        await asyncio.sleep(0)
        b.transfer_arrived(103)
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
    mb.declare_step("compute", step=0, expected_transfers={101: 2}, expected_states=[])

    tensor = TransportTensor.from_tensor(np.zeros(4))
    await deposit_tensor_frame(
        mb,
        "compute",
        "sender0",
        step=0,
        sender_rank=0,
        sender_world_size=4,
        entries=(
            _values_frame_entry(
                101,
                tensor.flat_values,
                4,
                expected_count=2,
            ),
        ),
    )
    assert not fired
    await deposit_tensor_frame(
        mb,
        "compute",
        "sender1",
        step=0,
        sender_rank=1,
        sender_world_size=4,
        entries=(
            _values_frame_entry(
                101,
                tensor.flat_values,
                4,
                expected_count=2,
            ),
        ),
    )
    assert fired == [("compute", 0)]


@pytest.mark.asyncio
async def test_mailbox_wait_step_ready():
    """wait_step_ready() coroutine resolves when the step is done."""
    mb = WorkerMailbox()
    mb.declare_step(
        "compute", step=5, expected_transfers={201: 1}, expected_states=["cfg"]
    )

    async def feed():
        await asyncio.sleep(0)
        tensor = TransportTensor.from_tensor(np.ones(3))
        await deposit_tensor_frame(
            mb,
            "compute",
            "sender0",
            step=5,
            sender_rank=0,
            sender_world_size=4,
            entries=(_values_frame_entry(201, tensor.flat_values, 3),),
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

        Sender->>SPMDWorker: receive_transport_frame(step=100)
        SPMDWorker->>WorkerMailbox: deposit_transport_frame(step=100)
        Note over WorkerMailbox: Virtual step mapping: 100 -> 0
        WorkerMailbox->>SPMDWorker: on_ready(op, step=0)
        SPMDWorker->>ExecutionLoop: Enqueue(op, step=0)

        Sender->>SPMDWorker: receive_transport_frame(step=101)
        SPMDWorker->>WorkerMailbox: deposit_transport_frame(step=101)
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

    # Declare periodic op
    mb.declare_step(
        "periodic_op",
        step=0,
        expected_transfers={301: 1},
        expected_states=[],
        barrier_config=PeriodicBarrierConfig(n=1),
    )

    # Deposit first shard (sender says step 100)
    tensor = TransportTensor.from_tensor(np.array([1.0]))
    await deposit_tensor_frame(
        mb,
        "periodic_op",
        "s0",
        step=100,
        sender_rank=0,
        sender_world_size=4,
        entries=(_values_frame_entry(301, tensor.flat_values, 1),),
    )

    # Deposit second shard (sender says step 101)
    await deposit_tensor_frame(
        mb,
        "periodic_op",
        "s1",
        step=101,
        sender_rank=1,
        sender_world_size=4,
        entries=(_values_frame_entry(301, tensor.flat_values, 1),),
    )

    # Should have two ready events with sequential virtual steps 0 and 1
    assert ready_steps == [("periodic_op", 0), ("periodic_op", 1)]

    # Verify data storage
    shards0 = await mb.get_tensor_shard_views("periodic_op", 301, 0)
    assert ("s0", 0) in shards0
    shards1 = await mb.get_tensor_shard_views("periodic_op", 301, 1)
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

    mb.declare_step(
        "periodic_op",
        step=0,
        expected_transfers={},
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
    mb.declare_step("compute", 0, expected_transfers={401: 1}, expected_states=[])
    d = np.array([1.0, 2.0])
    tensor = TransportTensor.from_tensor(d)
    await deposit_tensor_frame(
        mb,
        "compute",
        "sender0",
        step=0,
        sender_rank=0,
        sender_world_size=4,
        entries=(_values_frame_entry(401, tensor.flat_values, 2),),
    )
    shards = await mb.get_tensor_shard_views("compute", 401, 0)
    assert ("sender0", 0) in shards
    np.testing.assert_array_equal(shards[("sender0", 0)][0].flat_values, d)


@pytest.mark.asyncio
async def test_mailbox_tensor_multi_rank():
    mb = WorkerMailbox()
    mb.declare_step("compute", 1, expected_transfers={501: 2}, expected_states=[])
    tensor_zeros = TransportTensor.from_tensor(np.zeros(4))
    await deposit_tensor_frame(
        mb,
        "compute",
        "sender0",
        step=1,
        sender_rank=0,
        sender_world_size=4,
        entries=(
            _values_frame_entry(
                501,
                tensor_zeros.flat_values,
                4,
                expected_count=2,
            ),
        ),
    )
    tensor_ones = TransportTensor.from_tensor(np.ones(4))
    await deposit_tensor_frame(
        mb,
        "compute",
        "sender1",
        step=1,
        sender_rank=1,
        sender_world_size=4,
        entries=(
            _values_frame_entry(
                501,
                tensor_ones.flat_values,
                4,
                expected_count=2,
            ),
        ),
    )
    assert await mb.get_tensor_shard_count("compute", 501, 1) == 2


@pytest.mark.asyncio
async def test_mailbox_op_isolation():
    mb = WorkerMailbox()
    mb.declare_step("compute", 0, expected_transfers={601: 1}, expected_states=[])
    mb.declare_step("optimize", 0, expected_transfers={601: 1}, expected_states=[])
    await deposit_tensor_frame(
        mb,
        "compute",
        "sender0",
        step=0,
        sender_rank=0,
        sender_world_size=4,
        entries=(_values_frame_entry(601, np.array([1.0]), 1),),
    )
    await deposit_tensor_frame(
        mb,
        "optimize",
        "sender0",
        step=0,
        sender_rank=0,
        sender_world_size=4,
        entries=(_values_frame_entry(601, np.array([2.0]), 1),),
    )
    fwd = await mb.get_tensor_shard_views("compute", 601, 0)
    bwd = await mb.get_tensor_shard_views("optimize", 601, 0)
    np.testing.assert_array_equal(fwd[("sender0", 0)][0].flat_values, [1.0])
    np.testing.assert_array_equal(bwd[("sender0", 0)][0].flat_values, [2.0])


# ===========================================================================
# WorkerMailbox — cleanup / backpressure
# ===========================================================================


@pytest.mark.asyncio
async def test_mailbox_clear_step():
    mb = WorkerMailbox()
    mb.declare_step("compute", 1, expected_transfers={801: 1}, expected_states=["s"])
    await mb.deposit_control_arguments("compute", 1, names=["s"], values=["x"])
    await mb.deposit_control_arguments("compute", 2, names=["s"], values=["y"])
    tensor = TransportTensor.from_tensor(np.zeros(2))
    await deposit_tensor_frame(
        mb,
        "compute",
        "sender0",
        step=1,
        sender_rank=0,
        sender_world_size=4,
        entries=(_values_frame_entry(801, tensor.flat_values, 2),),
    )

    await mb.clear_step("compute", 1)

    assert not await mb.has_control_argument("compute", "s", 1)
    assert await mb.has_control_argument("compute", "s", 2)
    assert await mb.get_tensor_shard_count("compute", 801, 1) == 0
    assert not mb.is_step_ready("compute", 1)  # barrier also cleared


@pytest.mark.asyncio
async def test_mailbox_get_size():
    mb = WorkerMailbox()
    assert await mb.get_size() == 0
    mb.declare_step("compute", 0, expected_transfers={901: 2}, expected_states=["s"])
    await mb.deposit_control_arguments("compute", 0, names=["s"], values=[None])
    tensor = TransportTensor.from_tensor(np.zeros(2))
    await deposit_tensor_frame(
        mb,
        "compute",
        "s0",
        step=0,
        sender_rank=0,
        sender_world_size=2,
        entries=(
            _values_frame_entry(
                901,
                tensor.flat_values,
                2,
                expected_count=2,
            ),
        ),
    )
    await deposit_tensor_frame(
        mb,
        "compute",
        "s1",
        step=0,
        sender_rank=1,
        sender_world_size=2,
        entries=(
            _values_frame_entry(
                901,
                tensor.flat_values,
                2,
                expected_count=2,
            ),
        ),
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
    manager = OperatorManager(
        config,
        rank=0,
        static_args={},
        instance_container=OperatorInstanceContainer(),
    )
    assert manager._mesh is config.require_graph_view().mesh


@pytest.mark.asyncio
async def test_spmd_worker_init():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    config = _make_config(mesh, "compute", "optimize", "apply")
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(config, rank=0)
    worker.setup_and_start(local_rank=0, local_world_size=1, node_rank=0, num_nodes=1)
    assert worker._mesh is config.require_graph_view().mesh
    assert worker.op_manager._mesh is config.require_graph_view().mesh


def test_replicate_operator_manager():
    mesh = DeviceMesh(axis_names=("x", "y"), device_shape=(2, 2))

    replicated_spec = ShardingSpec.chunk(PartitionSpec())
    source_meta = OperatorSpec(
        name="produce",
        owner=_WorkerOperatorOwner,
        output_arguments={ArgumentType.DATA.value: ["value"]},
        output_sharding={"value": replicated_spec},
    )
    target_meta = OperatorSpec(
        name="test_op",
        owner=_WorkerOperatorOwner,
        method_name="test_op",
        input_arguments={ArgumentType.DATA.value: ["a"]},
        output_arguments={},
        input_sharding={"a": replicated_spec},
        output_sharding={},
    )
    graph = compile_test_graph(
        name="replicated-input",
        spmds={
            "source": (mesh, {"produce": source_meta}),
            "target": (mesh, {"test_op": target_meta}),
        },
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "produce"),
                target=OperatorKey("target", "test_op"),
                transfers=(TensorTransfer(OperatorPort("value"), OperatorPort("a")),),
            ),
        ),
    )
    config = spmd_config_from_graph(graph, "target")
    manager = OperatorManager(
        config,
        rank=0,
        static_args={},
        instance_container=OperatorInstanceContainer(),
    )

    spec = manager.get_input_sharding("test_op")
    transfer_code = graph.dependencies[0].transfers[0].transfer_code
    assert transfer_code in spec
    bound = spec[transfer_code].bind(config.require_graph_view().mesh)
    assert bound.replica_size == 4


@pytest.mark.asyncio
async def test_send_output_uses_compiled_mesh():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(1,))
    unbound_spec = ShardingSpec.chunk(PartitionSpec())
    source_meta = OperatorSpec(
        name="compute",
        owner=_WorkerOperatorOwner,
        output_arguments={ArgumentType.DATA.value: ["value"]},
        output_sharding={"value": unbound_spec},
    )
    target_meta = OperatorSpec(
        name="consume",
        owner=_WorkerOperatorOwner,
        input_arguments={ArgumentType.DATA.value: ["value"]},
        input_sharding={"value": unbound_spec},
    )
    graph = compile_test_graph(
        name="send-output-compiled-mesh",
        spmds={
            "source": (mesh, {"compute": source_meta}),
            "target": (mesh, {"consume": target_meta}),
        },
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "compute"),
                target=OperatorKey("target", "consume"),
                transfers=(
                    TensorTransfer(OperatorPort("value"), OperatorPort("value")),
                ),
            ),
        ),
    )
    config = spmd_config_from_graph(graph, "source")
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker
    worker = cls(config, rank=0, spmd_id="source")
    worker._op_manager = OperatorManager(
        config,
        rank=0,
        spmd_id="source",
        static_args={},
        instance_container=OperatorInstanceContainer(),
    )

    class RecordingDataPlaneExecutor:
        def __init__(self):
            self.calls = []

        async def send_tensor_frames(self, **kwargs):
            self.calls.append(kwargs)
            return [asyncio.sleep(0)]

    worker.register_spmd_code_map(4, {4: "source"})
    worker._successor_handlers = {"target": {0: object()}}
    recorder = RecordingDataPlaneExecutor()
    worker._data_plane_executor = recorder

    from fluxway.spmd.worker.operator_manager.dispatcher import (
        ExecutionStatus,
        ExecutionResult,
    )

    data = [np.array([1, 2, 3])]
    result = ExecutionResult(status=ExecutionStatus.SUCCESS, data=data)
    await worker._send_output("compute", result, exec_step=0, step_transfer_cache={})
    await worker._drain_outbound_tasks()

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["compiled_plan"].op_name == "compute"
    assert call["exec_step"] == 0
    assert call["successor_handlers"] == ({0: worker._successor_handlers["target"][0]},)
