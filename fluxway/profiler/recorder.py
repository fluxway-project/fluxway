from __future__ import annotations

from dataclasses import dataclass
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
import time
import threading
from collections.abc import AsyncGenerator, Generator
from typing import Any, TYPE_CHECKING

from fluxway.profiler.record import (
    EMPTY_TRACE_PAYLOAD,
    ProfilerSnapshot,
    TraceArena,
    TraceCarrier,
    TraceEventKind,
    TraceMetadata,
    stable_u64,
)
from fluxway.profiler.clock import ClockCalibration
from fluxway.profiler.filtering import StepFilter

if TYPE_CHECKING:
    from fluxway.common.compiled_graph import CompiledOperator


EMPTY_PAYLOAD_FIELDS: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TraceEventSpec:
    name: str
    track: str
    payload_fields: tuple[str, ...] = EMPTY_PAYLOAD_FIELDS
    step_payload_field: str | None = None


DATA_PLANE_RECEIVE_TRANSPORT_FRAME = TraceEventSpec(
    "data_plane.receive_transport_frame",
    "transport/send",
    ("remote_rank", "step"),
    "step",
)
DATA_PLANE_RECEIVE_CONTROL = TraceEventSpec(
    "data_plane.receive_control",
    "control/send",
    ("step", "field_count"),
    "step",
)
DATA_PLANE_HLC_RECEIVE = TraceEventSpec(
    "data_plane.hlc_receive",
    "transport/clock",
    ("receiver_hlc_physical_ns", "receiver_hlc_logical", "clock_source"),
)
DATA_PLANE_SEND_STEP_FRAMES = TraceEventSpec(
    "data_plane.send_step_frames",
    "data_plane/build",
    ("rank", "exec_step"),
    "exec_step",
)
DATA_PLANE_FRAME_BUILD = TraceEventSpec(
    "data_plane.frame_build",
    "data_plane/build",
    ("op_id", "rank", "step", "successor_count", "output_count"),
    "step",
)
DATA_PLANE_FRAME_SUBMIT = TraceEventSpec(
    "data_plane.frame_submit",
    "data_plane/enqueue",
    ("source_rank", "output_step", "frame_count"),
    "output_step",
)
DATA_PLANE_ENQUEUE_BATCH = TraceEventSpec(
    "data_plane.enqueue_batch",
    "data_plane/enqueue",
    (
        "op_id",
        "step",
        "source_spmd_code",
        "source_rank",
        "target_spmd_id",
        "target_rank",
        "batch_seq",
        "frame_count",
    ),
    "step",
)
DATA_PLANE_BATCH_RPC = TraceEventSpec(
    "data_plane.batch_rpc",
    "data_plane/batch_rpc",
    (
        "op_id",
        "step",
        "source_spmd_code",
        "source_rank",
        "target_spmd_id",
        "target_rank",
        "batch_seq",
        "frame_count",
    ),
    "step",
)
DATA_PLANE_DELIVERY_FENCE = TraceEventSpec(
    "data_plane.delivery_fence",
    "data_plane/delivery",
    ("fence_id", "pending_delivery_count"),
)
DELIVERY_REGISTER_TOKEN = TraceEventSpec(
    "delivery.register_token",
    "data_plane/delivery",
    ("pending_delivery_count",),
)
DELIVERY_SEMAPHORE_WAIT = TraceEventSpec(
    "delivery.semaphore_wait",
    "data_plane/delivery",
    ("pending_delivery_count",),
)
DELIVERY_RUN = TraceEventSpec(
    "delivery.run",
    "data_plane/delivery",
    ("pending_delivery_count",),
)
DELIVERY_POISON = TraceEventSpec(
    "delivery.poison",
    "data_plane/delivery",
    ("error_id",),
)
DELIVERY_FENCE = TraceEventSpec(
    "delivery.fence",
    "data_plane/delivery",
    ("fence_id", "pending_delivery_count"),
)
PREPARE_REGISTER = TraceEventSpec(
    "prepare.register",
    "compute/prepare",
    ("op_id", "step", "pending_prepare_count"),
    "step",
)
PREPARE_RUN = TraceEventSpec(
    "prepare.run",
    "compute/prepare",
    ("op_id", "step"),
    "step",
)
PREPARE_SNAPSHOT = TraceEventSpec(
    "prepare.snapshot",
    "compute/prepare",
    ("op_id", "step"),
    "step",
)
PREPARE_ASSEMBLE_THREAD = TraceEventSpec(
    "prepare.assemble_thread",
    "compute/prepare",
    ("op_id", "step"),
    "step",
)
PREPARE_ASSEMBLE_QUEUE_WAIT = TraceEventSpec(
    "prepare.assemble_queue_wait",
    "compute/prepare",
    ("op_id", "step", "priority", "prepare_state", "schedule_step"),
    "step",
)
PREPARE_CONSUME = TraceEventSpec(
    "prepare.consume",
    "compute/prepare",
    ("op_id", "step"),
    "step",
)
PREPARE_FENCE = TraceEventSpec(
    "prepare.fence",
    "compute/prepare",
    ("pending_prepare_count",),
)
DATA_PLANE_RECEIVER_WAIT = TraceEventSpec(
    "data_plane.receiver_wait",
    "transport/recv",
    ("remote_rank", "step"),
    "step",
)
DATA_PLANE_TENSOR_SPLIT = TraceEventSpec(
    "data_plane.tensor_split",
    "data_plane/build",
    ("op_id", "rank", "step", "target_rank", "payload_count"),
    "step",
)
WORKER_RECEIVE_TRANSPORT_FRAME = TraceEventSpec(
    "worker.receive_transport_frame",
    "transport/recv",
    ("step", "sender_spmd_code", "entry_count", "payload_count"),
    "step",
)
WORKER_DEPOSIT_TRANSPORT_FRAME = TraceEventSpec(
    "worker.deposit_transport_frame",
    "mailbox/deposit",
    ("op_id", "step", "entry_count", "watermark"),
    "step",
)
WORKER_ASSEMBLE_INPUT = TraceEventSpec(
    "worker.assemble_input",
    "tensor/reassemble",
    ("op_id", "step", "transfer_count", "state_count"),
    "step",
)
WORKER_ASSEMBLE_TENSOR = TraceEventSpec(
    "worker.assemble_tensor",
    "tensor/reassemble",
    (
        "op_id",
        "transfer_code",
        "logical_tuuid",
        "step",
        "sender_count",
        "shard_count",
    ),
    "step",
)
WORKER_ASSEMBLE_FILTER_SHARDS = TraceEventSpec(
    "worker.assemble_tensor.filter_shards",
    "tensor/reassemble/detail",
    (
        "op_id",
        "transfer_code",
        "logical_tuuid",
        "step",
        "sender_count",
        "shard_count",
    ),
    "step",
)
WORKER_ASSEMBLE_CAT_VALUES = TraceEventSpec(
    "worker.assemble_tensor.cat_values",
    "tensor/reassemble/detail",
    ("op_id", "transfer_code", "logical_tuuid", "step", "value_count"),
    "step",
)
WORKER_ASSEMBLE_MERGE_METADATA = TraceEventSpec(
    "worker.assemble_tensor.merge_metadata",
    "tensor/reassemble/detail",
    ("op_id", "transfer_code", "logical_tuuid", "step", "shard_count"),
    "step",
)
WORKER_ASSEMBLE_SORT = TraceEventSpec(
    "worker.assemble_tensor.sort",
    "tensor/reassemble/detail",
    (
        "op_id",
        "transfer_code",
        "logical_tuuid",
        "step",
        "actual_len",
        "capacity",
    ),
    "step",
)
WORKER_ASSEMBLE_SCATTER = TraceEventSpec(
    "worker.assemble_tensor.scatter",
    "tensor/reassemble/detail",
    (
        "op_id",
        "transfer_code",
        "logical_tuuid",
        "step",
        "logical_len",
        "capacity",
    ),
    "step",
)
WORKER_SEND_OUTPUT = TraceEventSpec(
    "worker.send_output",
    "tensor/split",
    ("op_id", "step", "output_count", "successor_count"),
    "step",
)
DATA_PLANE_RAY_PAYLOAD_SERIALIZE = TraceEventSpec(
    "data_plane.ray_payload_serialize",
    "data_plane/batch_rpc/detail",
    ("frame_count", "payload_count"),
)
DATA_PLANE_RAY_RPC_SUBMIT = TraceEventSpec(
    "data_plane.ray_rpc_submit",
    "data_plane/batch_rpc/detail",
    ("frame_count",),
)
DATA_PLANE_RAY_RPC_WAIT = TraceEventSpec(
    "data_plane.ray_rpc_wait",
    "data_plane/batch_rpc/detail",
    ("frame_count",),
)
DATA_PLANE_RECEIVER_BATCH = TraceEventSpec(
    "data_plane.receiver_batch",
    "transport/recv/detail",
    ("frame_count",),
)
DATA_PLANE_RECEIVER_DEPOSIT = TraceEventSpec(
    "data_plane.receiver_deposit",
    "transport/recv/detail",
    ("step", "sender_spmd_code", "entry_count", "payload_count"),
    "step",
)
LOCAL_STEP_PLAN_EXECUTE = TraceEventSpec(
    "local_step_plan.execute",
    "compute/local_plan",
    ("op_id", "step", "node_count"),
    "step",
)
LOCAL_STEP_WAIT_INPUTS = TraceEventSpec(
    "local_step.wait_inputs",
    "compute/local_plan",
    ("op_id", "step"),
    "step",
)
LOCAL_STEP_CALL_USER = TraceEventSpec(
    "local_step.call_user",
    "compute/local_plan",
    ("op_id", "step"),
    "step",
)
LOCAL_STEP_SEND_OUTPUT = TraceEventSpec(
    "local_step.send_output",
    "compute/local_plan",
    ("op_id", "step", "source_spmd_code", "source_rank"),
    "step",
)
LOCAL_STEP_AWAIT_SEND = TraceEventSpec(
    "local_step.await_send",
    "compute/local_plan",
    ("op_id", "step", "source_spmd_code", "source_rank"),
    "step",
)
LOCAL_STEP_COMMIT = TraceEventSpec(
    "local_step.commit",
    "compute/local_plan",
    ("op_id", "step"),
    "step",
)
EXECUTION_PHASE = TraceEventSpec(
    "execution.phase",
    "compute/lifecycle",
    ("phase_id", "op_id", "step", "epoch", "sequence", "execution_generation"),
    "step",
)
CONTROLLER_EXECUTION_PHASE = TraceEventSpec(
    "execution.phase",
    "lifecycle/execution_phase",
    ("phase_id", "op_id", "step", "epoch", "sequence", "execution_generation"),
    "step",
)
USER_CALLABLE_EXECUTE = TraceEventSpec(
    "user_callable.execute",
    "compute/user_callable",
    (
        "graph_id",
        "spmd_id",
        "op_id",
        "step",
        "execution_mode",
    ),
    "step",
)
USER_CALLABLE_FUNCTION_RUN = TraceEventSpec(
    "user_callable.function_run",
    "compute/user_callable",
    ("op_id", "step", "rank", "execution_mode"),
    "step",
)
USER_CUSTOM_PAYLOAD_FIELDS = (
    "op_id",
    "step",
    "rank",
    "execution_mode",
    "attrs_id",
)
WORKER_PROCESS_CALL_SUBMIT = TraceEventSpec(
    "worker_process_call.submit",
    "compute/user_callable_runtime",
    ("callable_id", "resource_id", "pending_call_count"),
)
WORKER_PROCESS_CALL_QUEUE_WAIT = TraceEventSpec(
    "worker_process_call.queue_wait",
    "compute/user_callable_runtime",
    ("callable_id", "resource_id", "pending_call_count"),
)
WORKER_PROCESS_CALL_THREAD_RUN = TraceEventSpec(
    "worker_process_call.thread_run",
    "compute/user_callable_runtime",
    ("callable_id", "resource_id"),
)
WORKER_PROCESS_CALL_FUTURE_WAIT = TraceEventSpec(
    "worker_process_call.future_wait",
    "compute/user_callable_runtime",
    ("callable_id", "resource_id", "pending_call_count"),
)
USER_CALLABLE_SNAPSHOT_HOOK = TraceEventSpec(
    "user_callable.snapshot_hook",
    "checkpoint/user_callable",
    ("hook_id", "operator_count"),
)
DATA_PLANE_SEND_SEMAPHORE_WAIT = TraceEventSpec(
    "data_plane.send_semaphore_wait",
    "transport/send",
    ("source_rank", "output_step"),
    "output_step",
)
TRANSPORT_DISTRIBUTE = TraceEventSpec(
    "transport.distribute",
    "transport/distribute",
    ("op_id", "rank", "step", "target_spmd_id"),
    "step",
)
TRANSPORT_RECEIVER_WAIT = TraceEventSpec(
    "transport.receiver_wait",
    "transport/receiver",
    ("expected_slice_count", "periodic"),
)
CONTROLLER_INITIALIZE = TraceEventSpec(
    "controller.initialize",
    "lifecycle/initialize",
    ("spmd_count",),
)
CONTROLLER_TOPOLOGY_REGISTRATION = TraceEventSpec(
    "controller.topology_registration",
    "topology/registration",
    ("spmd_count", "edge_count"),
)
CONTROLLER_DRAIN = TraceEventSpec(
    "controller.drain",
    "drain/admission",
    ("target_step", "spmd_count"),
    "target_step",
)
CONTROLLER_CHECKPOINT = TraceEventSpec(
    "controller.checkpoint",
    "checkpoint/save",
    ("target_step",),
    "target_step",
)
CONTROLLER_RESTORE = TraceEventSpec(
    "controller.restore",
    "checkpoint/restore",
    ("checkpoint_id",),
)
CONTROLLER_BACKPRESSURE = TraceEventSpec(
    "controller.backpressure",
    "backpressure/flow",
    ("busy_spmd_count", "paused_reader_count"),
)
CONTROLLER_COMPLETION = TraceEventSpec(
    "controller.completion",
    "lifecycle/completion",
    ("spmd_count",),
)
CONTROLLER_CLEANUP = TraceEventSpec(
    "controller.reclaim",
    "lifecycle/reclaim",
    ("spmd_count",),
)
CONTROLLER_PROFILER_EXPORT = TraceEventSpec(
    "controller.profiler_export",
    "profiler/export",
    ("spmd_count", "record_count"),
)
CONTROLLER_FSM_TRANSITION = TraceEventSpec(
    "controller_fsm.transition",
    "fsm/transition",
    ("from_state_id", "event_id", "to_state_id"),
)
CONTROLLER_FSM_ENTRY_CALLBACK = TraceEventSpec(
    "controller_fsm.entry_callback",
    "fsm/entry_callback",
    ("state_id",),
)
HEARTBEAT_RECEIVE = TraceEventSpec(
    "heartbeat.receive",
    "heartbeat/receive",
    ("spmd_id", "state_id", "healthy_workers", "num_workers"),
)
HEARTBEAT_MONITOR_LOOP = TraceEventSpec(
    "heartbeat.monitor_loop",
    "heartbeat/monitor_loop",
    ("spmd_count",),
)
HEARTBEAT_TIMEOUT_CHECK = TraceEventSpec(
    "heartbeat.timeout_check",
    "heartbeat/timeout_check",
    ("spmd_id", "elapsed_ms"),
)
HEARTBEAT_BLOCKED_WATCHDOG = TraceEventSpec(
    "heartbeat.blocked_watchdog",
    "heartbeat/blocked_watchdog",
    ("timeout_ms",),
)
SPMD_MANAGER_PLACEMENT_GROUP_CREATE = TraceEventSpec(
    "spmd_manager.placement_group_create",
    "spmd_manager/placement_group_create",
    ("spmd_id", "worker_count"),
)
SPMD_MANAGER_DRIVER_START = TraceEventSpec(
    "spmd_manager.driver_start",
    "spmd_manager/driver_start",
    ("spmd_id", "worker_count"),
)
SPMD_MANAGER_DRIVER_STOP = TraceEventSpec(
    "spmd_manager.driver_stop",
    "spmd_manager/driver_stop",
    ("spmd_id",),
)
DRIVER_INITIALIZE = TraceEventSpec(
    "driver.initialize",
    "driver/lifecycle",
    ("worker_count",),
)
DRIVER_START_WORKERS = TraceEventSpec(
    "driver.start_workers",
    "driver/lifecycle",
    ("worker_count",),
)
DRIVER_REGISTER_TOPOLOGY = TraceEventSpec(
    "driver.register_topology",
    "driver/topology",
    ("predecessor_count", "successor_count"),
)
DRIVER_DRAIN = TraceEventSpec(
    "driver.drain",
    "driver/drain",
    ("target_step", "worker_count"),
    "target_step",
)
DRIVER_SNAPSHOT = TraceEventSpec(
    "driver.snapshot",
    "driver/checkpoint",
    ("checkpoint_id", "target_step", "worker_count"),
    "target_step",
)
DRIVER_RESTORE = TraceEventSpec(
    "driver.restore",
    "driver/checkpoint",
    ("checkpoint_id", "worker_count"),
)
DRIVER_HEARTBEAT_THREAD_JOIN = TraceEventSpec(
    "driver.heartbeat_thread_join",
    "driver/heartbeat",
    ("timeout_ms",),
)
DRIVER_SOURCE_START = TraceEventSpec(
    "driver.source_start",
    "driver/source",
    ("worker_count",),
)
DRIVER_WORKER_HEARTBEAT = TraceEventSpec(
    "driver.worker_heartbeat",
    "driver/heartbeat",
    ("rank", "status_id"),
)
DRIVER_OP_COMPLETION = TraceEventSpec(
    "driver.op_completion",
    "driver/completion",
    ("rank", "op_id", "step"),
    "step",
)
WORKER_BACKGROUND_TASK = TraceEventSpec(
    "worker.background_task",
    "scheduler/task",
    ("task_id",),
)
WORKER_OUTBOUND_TASK = TraceEventSpec(
    "worker.outbound_task",
    "scheduler/task",
    ("task_id",),
)
WORKER_QUEUE_ENQUEUE = TraceEventSpec(
    "worker.queue_enqueue",
    "scheduler/queue",
    ("op_id", "step", "priority"),
    "step",
)
WORKER_QUEUE_DEQUEUE = TraceEventSpec(
    "worker.queue_dequeue",
    "scheduler/queue",
    ("op_id", "step", "priority"),
    "step",
)
WORKER_BLOCKED_TASK = TraceEventSpec(
    "worker.blocked_task",
    "scheduler/queue",
    ("op_id", "step"),
    "step",
)
WORKER_FUTURE_TASK = TraceEventSpec(
    "worker.future_task",
    "scheduler/queue",
    ("op_id", "step", "completed_count"),
    "step",
)
WORKER_STALE_TASK = TraceEventSpec(
    "worker.stale_task",
    "scheduler/queue",
    ("op_id", "step", "completed_count"),
    "step",
)
WORKER_RUN_LOOP = TraceEventSpec(
    "worker.run_loop",
    "scheduler/loop",
    EMPTY_PAYLOAD_FIELDS,
)
WORKER_ADVANCE_STEP = TraceEventSpec(
    "worker.advance_step",
    "scheduler/step",
    ("op_id", "step", "next_step"),
    "step",
)
WORKER_DRAIN_WAIT = TraceEventSpec(
    "worker.drain_wait",
    "scheduler/step",
    ("op_id", "step", "target_step"),
    "step",
)
WORKER_CHECKPOINT_SAVE = TraceEventSpec(
    "checkpoint.async_save",
    "checkpoint/storage",
    ("step", "operator_count"),
    "step",
)
WORKER_CHECKPOINT_LOAD = TraceEventSpec(
    "checkpoint.async_load",
    "checkpoint/storage",
    ("operator_count", "has_rank_storage"),
)
WORKER_SNAPSHOT_HOOK = TraceEventSpec(
    "worker.snapshot_hook",
    "checkpoint/hook",
    ("hook_id", "step"),
    "step",
)
MAILBOX_DECLARE_STEP = TraceEventSpec(
    "mailbox.declare_step",
    "mailbox/barrier",
    ("op_id", "step", "transfer_count", "state_count"),
    "step",
)
MAILBOX_DEPOSIT_CONTROL = TraceEventSpec(
    "mailbox.deposit_control",
    "mailbox/deposit",
    ("op_id", "step", "field_count"),
    "step",
)
MAILBOX_DEPOSIT_FRAME = TraceEventSpec(
    "mailbox.deposit_frame",
    "mailbox/deposit",
    ("op_id", "step", "entry_count", "payload_count"),
    "step",
)
MAILBOX_DEPOSIT_LOCAL_TENSOR = TraceEventSpec(
    "mailbox.deposit_local_tensor",
    "mailbox/deposit",
    ("op_id", "transfer_code", "step", "data_count"),
    "step",
)
MAILBOX_READY_FIRE = TraceEventSpec(
    "mailbox.ready_fire",
    "mailbox/barrier",
    ("op_id", "step"),
    "step",
)
MAILBOX_WAIT_STEP_READY = TraceEventSpec(
    "mailbox.wait_step_ready",
    "mailbox/barrier",
    ("op_id", "step"),
    "step",
)
MAILBOX_CLEAR_STEP = TraceEventSpec(
    "mailbox.clear_step",
    "mailbox/barrier",
    ("op_id", "step"),
    "step",
)


class TraceRecorder:
    def __init__(
        self,
        arena: TraceArena,
        *,
        run_id: str,
        process_name: str,
        clock_calibration: ClockCalibration | None = None,
        step_filter: StepFilter | None = None,
    ) -> None:
        if step_filter is not None and not isinstance(step_filter, StepFilter):
            raise TypeError("step_filter must be a StepFilter or None.")
        self._arena = arena
        self._run_id = run_id
        self._process_name = process_name
        self._step_filter = StepFilter() if step_filter is None else step_filter
        self._clock_calibration = clock_calibration or ClockCalibration(
            monotonic_to_epoch_offset_ns=0,
            uncertainty_ns=0,
        )
        self._interned: dict[str, int] = {}
        self._names_by_id: dict[int, str] = {}
        self._payload_fields_by_name_id: dict[int, tuple[str, ...]] = {}
        self._intern_lock = threading.RLock()
        self._root_carrier = TraceCarrier.root(run_id)

    def is_step_profiled(self, step: int) -> bool:
        return self._step_filter.is_step_profiled(step)

    @property
    def arena(self) -> TraceArena:
        return self._arena

    @property
    def root_carrier(self) -> TraceCarrier:
        return self._root_carrier

    @property
    def metadata(self) -> TraceMetadata:
        with self._intern_lock:
            return TraceMetadata(
                names_by_id=dict(self._names_by_id),
                payload_fields_by_name_id=dict(self._payload_fields_by_name_id),
            )

    def snapshot(self) -> ProfilerSnapshot:
        return {
            "records": self._arena.snapshot(),
            "metadata": self.metadata,
            "process_name": self._process_name,
            "monotonic_to_epoch_offset_ns": self._clock_calibration.monotonic_to_epoch_offset_ns,
            "clock_uncertainty_ns": self._clock_calibration.uncertainty_ns,
        }

    def intern(self, value: str) -> int:
        existing = self._interned.get(value)
        if existing is not None:
            return existing
        with self._intern_lock:
            existing = self._interned.get(value)
            if existing is not None:
                return existing
            ident = stable_u64(self._run_id, self._process_name, value)
            self._interned[value] = ident
            self._names_by_id[ident] = value
            return ident

    def data_plane_hlc_receive(
        self,
        *,
        carrier: TraceCarrier,
        receiver_hlc_physical_ns: int,
        receiver_hlc_logical: int,
        clock_source: int,
    ) -> None:
        self._instant(
            DATA_PLANE_HLC_RECEIVE,
            carrier=carrier,
            payload=(
                receiver_hlc_physical_ns,
                receiver_hlc_logical,
                clock_source,
            ),
        )

    def data_plane_receive_transport_frame(
        self, *, carrier: TraceCarrier, remote_rank: int, step: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DATA_PLANE_RECEIVE_TRANSPORT_FRAME,
            carrier=carrier,
            payload=(remote_rank, step),
        )

    def data_plane_receive_control(
        self, *, carrier: TraceCarrier, step: int, field_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DATA_PLANE_RECEIVE_CONTROL,
            carrier=carrier,
            payload=(step, field_count),
        )

    def data_plane_send_step_frames(
        self, *, rank: int, exec_step: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DATA_PLANE_SEND_STEP_FRAMES,
            payload=(rank, exec_step),
        )

    def data_plane_frame_build_sync(
        self,
        *,
        op_name: str,
        rank: int,
        step: int,
        successor_count: int,
        output_count: int,
    ) -> AbstractContextManager[None]:
        return self._span(
            DATA_PLANE_FRAME_BUILD,
            payload=(
                self.intern(op_name),
                rank,
                step,
                successor_count,
                output_count,
            ),
        )

    def data_plane_frame_submit(
        self, *, source_rank: int, output_step: int, frame_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DATA_PLANE_FRAME_SUBMIT,
            payload=(source_rank, output_step, frame_count),
        )

    def data_plane_enqueue_batch(
        self,
        *,
        op_name: str,
        step: int,
        source_spmd_code: int,
        source_rank: int,
        target_spmd_id: str,
        target_rank: int,
        batch_seq: int,
        frame_count: int,
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DATA_PLANE_ENQUEUE_BATCH,
            payload=(
                self.intern(op_name),
                step,
                source_spmd_code,
                source_rank,
                self.intern(target_spmd_id),
                target_rank,
                batch_seq,
                frame_count,
            ),
        )

    def data_plane_batch_rpc(
        self,
        *,
        op_name: str,
        step: int,
        source_spmd_code: int,
        source_rank: int,
        target_spmd_id: str,
        target_rank: int,
        batch_seq: int,
        frame_count: int,
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DATA_PLANE_BATCH_RPC,
            payload=(
                self.intern(op_name),
                step,
                source_spmd_code,
                source_rank,
                self.intern(target_spmd_id),
                target_rank,
                batch_seq,
                frame_count,
            ),
        )

    def data_plane_delivery_fence(
        self, *, fence_id: str, pending_delivery_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DATA_PLANE_DELIVERY_FENCE,
            payload=(self.intern(fence_id), pending_delivery_count),
        )

    def delivery_register_token(
        self, *, pending_delivery_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DELIVERY_REGISTER_TOKEN,
            payload=(pending_delivery_count,),
        )

    def delivery_semaphore_wait(
        self, *, pending_delivery_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DELIVERY_SEMAPHORE_WAIT,
            payload=(pending_delivery_count,),
        )

    def delivery_run(
        self, *, pending_delivery_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DELIVERY_RUN,
            payload=(pending_delivery_count,),
        )

    def delivery_poison(self, *, error: str) -> AbstractContextManager[None]:
        return self._span(
            DELIVERY_POISON,
            payload=(self.intern(error),),
        )

    def delivery_fence(
        self, *, fence_id: str, pending_delivery_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DELIVERY_FENCE,
            payload=(self.intern(fence_id), pending_delivery_count),
        )

    def prepare_register(
        self, *, op_name: str, step: int, pending_prepare_count: int
    ) -> AbstractContextManager[None]:
        return self._span(
            PREPARE_REGISTER,
            payload=(self.intern(op_name), step, pending_prepare_count),
        )

    def prepare_run(
        self, *, op_name: str, step: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            PREPARE_RUN,
            payload=(self.intern(op_name), step),
        )

    def prepare_snapshot(
        self, *, op_name: str, step: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            PREPARE_SNAPSHOT,
            payload=(self.intern(op_name), step),
        )

    def prepare_assemble_thread(
        self, *, op_name: str, step: int
    ) -> AbstractContextManager[None]:
        return self._span(
            PREPARE_ASSEMBLE_THREAD,
            payload=(self.intern(op_name), step),
        )

    def prepare_assemble_queue_wait(
        self,
        *,
        op_name: str,
        step: int,
        schedule_key: tuple[int, int, int, str],
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            PREPARE_ASSEMBLE_QUEUE_WAIT,
            payload=(
                self.intern(op_name),
                step,
                schedule_key[0],
                schedule_key[1],
                schedule_key[2],
            ),
        )

    def prepare_consume(
        self, *, op_name: str, step: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            PREPARE_CONSUME,
            payload=(self.intern(op_name), step),
        )

    def prepare_fence(
        self, *, pending_prepare_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            PREPARE_FENCE,
            payload=(pending_prepare_count,),
        )

    def data_plane_receiver_wait(
        self, *, remote_rank: int, step: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DATA_PLANE_RECEIVER_WAIT,
            payload=(remote_rank, step),
        )

    def data_plane_tensor_split(
        self,
        *,
        op_name: str,
        rank: int,
        step: int,
        target_rank: int,
        payload_count: int,
    ) -> None:
        self._instant(
            DATA_PLANE_TENSOR_SPLIT,
            payload=(
                self.intern(op_name),
                rank,
                step,
                target_rank,
                payload_count,
            ),
        )

    def worker_receive_transport_frame(
        self,
        *,
        step: int,
        sender_spmd_code: int,
        entry_count: int,
        payload_count: int,
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            WORKER_RECEIVE_TRANSPORT_FRAME,
            payload=(step, sender_spmd_code, entry_count, payload_count),
        )

    def worker_deposit_transport_frame(
        self,
        *,
        op_name: str,
        step: int,
        entry_count: int,
        watermark: int,
    ) -> None:
        self._instant(
            WORKER_DEPOSIT_TRANSPORT_FRAME,
            payload=(self.intern(op_name), step, entry_count, watermark),
        )

    def worker_assemble_input(
        self, *, op_name: str, step: int, transfer_count: int, state_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            WORKER_ASSEMBLE_INPUT,
            payload=(self.intern(op_name), step, transfer_count, state_count),
        )

    def worker_assemble_tensor(
        self,
        *,
        op_name: str,
        transfer_code: int,
        logical_tuuid: int,
        step: int,
        sender_count: int,
        shard_count: int,
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            WORKER_ASSEMBLE_TENSOR,
            payload=(
                self.intern(op_name),
                transfer_code,
                logical_tuuid,
                step,
                sender_count,
                shard_count,
            ),
        )

    def worker_assemble_tensor_sync(
        self,
        *,
        op_name: str,
        transfer_code: int,
        logical_tuuid: int,
        step: int,
        sender_count: int,
        shard_count: int,
    ) -> AbstractContextManager[None]:
        return self._span(
            WORKER_ASSEMBLE_TENSOR,
            payload=(
                self.intern(op_name),
                transfer_code,
                logical_tuuid,
                step,
                sender_count,
                shard_count,
            ),
        )

    def worker_assemble_detail(
        self,
        *,
        spec: TraceEventSpec,
        op_name: str,
        transfer_code: int,
        logical_tuuid: int,
        step: int,
        first_count: int,
        second_count: int = 0,
    ) -> AbstractContextManager[None]:
        spec_name = spec.name
        payload: tuple[Any, ...]
        if spec_name == WORKER_ASSEMBLE_FILTER_SHARDS.name:
            payload = (
                self.intern(op_name),
                transfer_code,
                logical_tuuid,
                step,
                first_count,
                second_count,
            )
        elif spec_name in (
            WORKER_ASSEMBLE_CAT_VALUES.name,
            WORKER_ASSEMBLE_MERGE_METADATA.name,
        ):
            payload = (
                self.intern(op_name),
                transfer_code,
                logical_tuuid,
                step,
                first_count,
            )
        elif spec_name in (WORKER_ASSEMBLE_SORT.name, WORKER_ASSEMBLE_SCATTER.name):
            payload = (
                self.intern(op_name),
                transfer_code,
                logical_tuuid,
                step,
                first_count,
                second_count,
            )
        else:
            raise ValueError(f"Unsupported assemble detail spec {spec.name!r}.")
        return self._span(spec, payload=payload)

    def worker_send_output(
        self, *, op_name: str, step: int, output_count: int, successor_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            WORKER_SEND_OUTPUT,
            payload=(self.intern(op_name), step, output_count, successor_count),
        )

    def local_step_plan_execute(
        self, *, op_name: str, step: int, node_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            LOCAL_STEP_PLAN_EXECUTE,
            payload=(self.intern(op_name), step, node_count),
        )

    def execution_phase(
        self,
        *,
        phase: str,
        op_name: str | None,
        step: int | None,
        epoch: int,
        sequence: int,
        execution_generation: int,
    ) -> None:
        """Record opaque lifecycle metadata shared by all execution backends."""
        self._record_execution_phase(
            EXECUTION_PHASE,
            phase=phase,
            op_name=op_name,
            step=step,
            epoch=epoch,
            sequence=sequence,
            execution_generation=execution_generation,
        )

    def controller_execution_phase(
        self,
        *,
        phase: str,
        op_name: str | None,
        step: int | None,
        epoch: int,
        sequence: int,
        execution_generation: int,
    ) -> None:
        self._record_execution_phase(
            CONTROLLER_EXECUTION_PHASE,
            phase=phase,
            op_name=op_name,
            step=step,
            epoch=epoch,
            sequence=sequence,
            execution_generation=execution_generation,
        )

    def _record_execution_phase(
        self,
        spec: TraceEventSpec,
        *,
        phase: str,
        op_name: str | None,
        step: int | None,
        epoch: int,
        sequence: int,
        execution_generation: int,
    ) -> None:
        self._instant(
            spec,
            payload=(
                self.intern(phase),
                self.intern(op_name or ""),
                -1 if step is None else step,
                epoch,
                sequence,
                execution_generation,
            ),
        )

    def local_step_node(
        self,
        *,
        spec: TraceEventSpec,
        op_name: str,
        step: int,
        source_spmd_code: int | None = None,
        source_rank: int | None = None,
    ) -> AbstractAsyncContextManager[None]:
        if spec in (LOCAL_STEP_SEND_OUTPUT, LOCAL_STEP_AWAIT_SEND):
            if source_spmd_code is None or source_rank is None:
                raise ValueError(
                    f"{spec.name} requires source_spmd_code and source_rank."
                )
            return self._async_span(
                spec,
                payload=(self.intern(op_name), step, source_spmd_code, source_rank),
            )
        if source_spmd_code is not None or source_rank is not None:
            raise ValueError(
                f"{spec.name} does not accept source_spmd_code/source_rank."
            )
        return self._async_span(
            spec,
            payload=(self.intern(op_name), step),
        )

    def user_callable_execute(
        self,
        *,
        graph_id: str,
        compiled_operator: CompiledOperator,
        step: int,
        execution_mode: str,
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            USER_CALLABLE_EXECUTE,
            payload=(
                self.intern(graph_id),
                self.intern(compiled_operator.key.spmd_id),
                self.intern(compiled_operator.key.op_id),
                step,
                self.intern(execution_mode),
            ),
        )

    def user_callable_function_run(
        self,
        *,
        op_name: str,
        step: int,
        rank: int,
        execution_mode: str,
    ) -> AbstractContextManager[None]:
        return self._span(
            USER_CALLABLE_FUNCTION_RUN,
            payload=(
                self.intern(op_name),
                step,
                rank,
                self.intern(execution_mode),
            ),
        )

    def user_custom_slice(
        self,
        *,
        name: str,
        op_name: str,
        step: int,
        rank: int,
        execution_mode: str,
        attrs_json: str,
        start_ns: int,
        dur_ns: int,
    ) -> None:
        self._slice(
            self._user_custom_spec(name=name, execution_mode=execution_mode),
            start_ns=start_ns,
            dur_ns=dur_ns,
            payload=(
                self.intern(op_name),
                step,
                rank,
                self.intern(execution_mode),
                self.intern(attrs_json),
            ),
        )

    def user_custom_event(
        self,
        *,
        name: str,
        op_name: str,
        step: int,
        rank: int,
        execution_mode: str,
        attrs_json: str,
    ) -> None:
        self._instant(
            self._user_custom_spec(name=name, execution_mode=execution_mode),
            payload=(
                self.intern(op_name),
                step,
                rank,
                self.intern(execution_mode),
                self.intern(attrs_json),
            ),
        )

    @staticmethod
    def _user_custom_spec(*, name: str, execution_mode: str) -> TraceEventSpec:
        return TraceEventSpec(
            name,
            f"compute/user_custom/{execution_mode}",
            USER_CUSTOM_PAYLOAD_FIELDS,
            "step",
        )

    def worker_process_submit(
        self, *, callable_id: str, resource_id: int, pending_call_count: int
    ) -> AbstractContextManager[None]:
        return self._span(
            WORKER_PROCESS_CALL_SUBMIT,
            payload=(self.intern(callable_id), resource_id, pending_call_count),
        )

    def worker_process_queue_wait(
        self, *, callable_id: str, resource_id: int, pending_call_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            WORKER_PROCESS_CALL_QUEUE_WAIT,
            payload=(self.intern(callable_id), resource_id, pending_call_count),
        )

    def worker_process_thread_run(
        self, *, callable_id: str, resource_id: int
    ) -> AbstractContextManager[None]:
        return self._span(
            WORKER_PROCESS_CALL_THREAD_RUN,
            payload=(self.intern(callable_id), resource_id),
        )

    def worker_process_future_wait(
        self, *, callable_id: str, resource_id: int, pending_call_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            WORKER_PROCESS_CALL_FUTURE_WAIT,
            payload=(self.intern(callable_id), resource_id, pending_call_count),
        )

    def user_callable_snapshot_hook(
        self, *, hook_name: str, operator_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            USER_CALLABLE_SNAPSHOT_HOOK,
            payload=(self.intern(hook_name), operator_count),
        )

    def data_plane_send_semaphore_wait(
        self, *, source_rank: int, output_step: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DATA_PLANE_SEND_SEMAPHORE_WAIT,
            payload=(source_rank, output_step),
        )

    def data_plane_ray_payload_serialize(
        self, *, frame_count: int, payload_count: int
    ) -> AbstractContextManager[None]:
        return self._span(
            DATA_PLANE_RAY_PAYLOAD_SERIALIZE,
            payload=(frame_count, payload_count),
        )

    def data_plane_ray_rpc_submit(
        self, *, frame_count: int
    ) -> AbstractContextManager[None]:
        return self._span(DATA_PLANE_RAY_RPC_SUBMIT, payload=(frame_count,))

    def data_plane_ray_rpc_wait(
        self, *, frame_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(DATA_PLANE_RAY_RPC_WAIT, payload=(frame_count,))

    def data_plane_receiver_batch(
        self, *, frame_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(DATA_PLANE_RECEIVER_BATCH, payload=(frame_count,))

    def data_plane_receiver_deposit(
        self,
        *,
        step: int,
        sender_spmd_code: int,
        entry_count: int,
        payload_count: int,
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DATA_PLANE_RECEIVER_DEPOSIT,
            payload=(step, sender_spmd_code, entry_count, payload_count),
        )

    def transport_distribute(
        self, *, op_name: str, rank: int, step: int, target_spmd_id: str
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            TRANSPORT_DISTRIBUTE,
            payload=(self.intern(op_name), rank, step, self.intern(target_spmd_id)),
        )

    def transport_distribute_sync(
        self, *, op_name: str, rank: int, step: int, target_spmd_id: str
    ) -> AbstractContextManager[None]:
        return self._span(
            TRANSPORT_DISTRIBUTE,
            payload=(self.intern(op_name), rank, step, self.intern(target_spmd_id)),
        )

    def transport_receiver_wait(
        self, *, expected_slice_count: int, periodic: bool
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            TRANSPORT_RECEIVER_WAIT,
            payload=(expected_slice_count, int(periodic)),
        )

    def controller_initialize(
        self, *, spmd_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(CONTROLLER_INITIALIZE, payload=(spmd_count,))

    def controller_topology_registration(
        self, *, spmd_count: int, edge_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            CONTROLLER_TOPOLOGY_REGISTRATION,
            payload=(spmd_count, edge_count),
        )

    def controller_drain(
        self, *, target_step: int, spmd_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(CONTROLLER_DRAIN, payload=(target_step, spmd_count))

    def controller_checkpoint(
        self, *, target_step: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(CONTROLLER_CHECKPOINT, payload=(target_step,))

    def controller_restore(
        self, *, checkpoint_id: str
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            CONTROLLER_RESTORE,
            payload=(self.intern(checkpoint_id),),
        )

    def controller_backpressure(
        self, *, busy_spmd_count: int, paused_reader_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            CONTROLLER_BACKPRESSURE,
            payload=(busy_spmd_count, paused_reader_count),
        )

    def controller_completion(
        self, *, spmd_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(CONTROLLER_COMPLETION, payload=(spmd_count,))

    def controller_cleanup(
        self, *, spmd_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(CONTROLLER_CLEANUP, payload=(spmd_count,))

    def controller_profiler_export(
        self, *, spmd_count: int, record_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            CONTROLLER_PROFILER_EXPORT,
            payload=(spmd_count, record_count),
        )

    def controller_fsm_transition(
        self, *, from_state: str, event: str, to_state: str
    ) -> None:
        self._instant(
            CONTROLLER_FSM_TRANSITION,
            payload=(
                self.intern(from_state),
                self.intern(event),
                self.intern(to_state),
            ),
        )

    def controller_fsm_entry_callback(
        self, *, state: str
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            CONTROLLER_FSM_ENTRY_CALLBACK,
            payload=(self.intern(state),),
        )

    def heartbeat_receive(
        self, *, spmd_id: str, state: str, healthy_workers: int, num_workers: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            HEARTBEAT_RECEIVE,
            payload=(
                self.intern(spmd_id),
                self.intern(state),
                healthy_workers,
                num_workers,
            ),
        )

    def heartbeat_monitor_loop(
        self, *, spmd_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(HEARTBEAT_MONITOR_LOOP, payload=(spmd_count,))

    def heartbeat_timeout_check(self, *, spmd_id: str, elapsed_ms: int) -> None:
        self._instant(
            HEARTBEAT_TIMEOUT_CHECK,
            payload=(self.intern(spmd_id), elapsed_ms),
        )

    def heartbeat_blocked_watchdog(
        self, *, timeout_ms: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(HEARTBEAT_BLOCKED_WATCHDOG, payload=(timeout_ms,))

    def spmd_manager_placement_group_create(
        self, *, spmd_id: str, worker_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            SPMD_MANAGER_PLACEMENT_GROUP_CREATE,
            payload=(self.intern(spmd_id), worker_count),
        )

    def spmd_manager_driver_start(
        self, *, spmd_id: str, worker_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            SPMD_MANAGER_DRIVER_START,
            payload=(self.intern(spmd_id), worker_count),
        )

    def spmd_manager_driver_stop(
        self, *, spmd_id: str
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            SPMD_MANAGER_DRIVER_STOP,
            payload=(self.intern(spmd_id),),
        )

    def driver_initialize(
        self, *, worker_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(DRIVER_INITIALIZE, payload=(worker_count,))

    def driver_start_workers(
        self, *, worker_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(DRIVER_START_WORKERS, payload=(worker_count,))

    def driver_register_topology(
        self, *, predecessor_count: int, successor_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DRIVER_REGISTER_TOPOLOGY,
            payload=(predecessor_count, successor_count),
        )

    def driver_drain(
        self, *, target_step: int, worker_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(DRIVER_DRAIN, payload=(target_step, worker_count))

    def driver_snapshot(
        self, *, checkpoint_id: str, target_step: int, worker_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DRIVER_SNAPSHOT,
            payload=(self.intern(checkpoint_id), target_step, worker_count),
        )

    def driver_restore(
        self, *, checkpoint_id: str, worker_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            DRIVER_RESTORE,
            payload=(self.intern(checkpoint_id), worker_count),
        )

    def driver_heartbeat_thread_join(
        self, *, timeout_ms: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(DRIVER_HEARTBEAT_THREAD_JOIN, payload=(timeout_ms,))

    def driver_source_start(
        self, *, worker_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(DRIVER_SOURCE_START, payload=(worker_count,))

    def driver_worker_heartbeat(self, *, rank: int, status: str) -> None:
        self._instant(
            DRIVER_WORKER_HEARTBEAT,
            payload=(rank, self.intern(status)),
        )

    def driver_op_completion(self, *, rank: int, op_name: str, step: int) -> None:
        self._instant(
            DRIVER_OP_COMPLETION,
            payload=(rank, self.intern(op_name), step),
        )

    def worker_scheduler_span(
        self, *, spec: TraceEventSpec, payload: tuple[int, ...] = EMPTY_TRACE_PAYLOAD
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(spec, payload=payload)

    def worker_scheduler_instant(
        self, *, spec: TraceEventSpec, payload: tuple[int, ...] = EMPTY_TRACE_PAYLOAD
    ) -> None:
        self._instant(spec, payload=payload)

    def worker_op_payload(self, op_name: str, *values: int) -> tuple[int, ...]:
        return (self.intern(op_name), *values)

    def worker_checkpoint_save(
        self, *, step: int, operator_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            WORKER_CHECKPOINT_SAVE,
            payload=(step, operator_count),
        )

    def worker_checkpoint_load(
        self, *, operator_count: int, has_rank_storage: bool
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            WORKER_CHECKPOINT_LOAD,
            payload=(operator_count, int(has_rank_storage)),
        )

    def mailbox_declare_step(
        self, *, op_name: str, step: int, transfer_count: int, state_count: int
    ) -> None:
        self._instant(
            MAILBOX_DECLARE_STEP,
            payload=(self.intern(op_name), step, transfer_count, state_count),
        )

    def mailbox_deposit_control(
        self, *, op_name: str, step: int, field_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            MAILBOX_DEPOSIT_CONTROL,
            payload=(self.intern(op_name), step, field_count),
        )

    def mailbox_deposit_frame(
        self, *, op_name: str, step: int, entry_count: int, payload_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            MAILBOX_DEPOSIT_FRAME,
            payload=(self.intern(op_name), step, entry_count, payload_count),
        )

    def mailbox_deposit_local_tensor(
        self, *, op_name: str, transfer_code: int, step: int, data_count: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            MAILBOX_DEPOSIT_LOCAL_TENSOR,
            payload=(self.intern(op_name), transfer_code, step, data_count),
        )

    def mailbox_ready_fire(self, *, op_name: str, step: int) -> None:
        self._instant(
            MAILBOX_READY_FIRE,
            payload=(self.intern(op_name), step),
        )

    def mailbox_wait_step_ready(
        self, *, op_name: str, step: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            MAILBOX_WAIT_STEP_READY,
            payload=(self.intern(op_name), step),
        )

    def mailbox_clear_step(
        self, *, op_name: str, step: int
    ) -> AbstractAsyncContextManager[None]:
        return self._async_span(
            MAILBOX_CLEAR_STEP,
            payload=(self.intern(op_name), step),
        )

    def _instant(
        self,
        spec: TraceEventSpec,
        *,
        carrier: TraceCarrier | None = None,
        payload: tuple[int, ...] = EMPTY_TRACE_PAYLOAD,
    ) -> None:
        if self._skip_record(spec, payload):
            return
        active = self._root_carrier if carrier is None else carrier
        track_id, name_id = self._register_spec(spec)
        self._arena.append(
            kind=TraceEventKind.INSTANT,
            ts_ns=time.perf_counter_ns(),
            dur_ns=0,
            track_id=track_id,
            name_id=name_id,
            flow_id=active.flow_id,
            task_id=active.task_id,
            parent_task_id=active.parent_task_id,
            payload=payload,
        )

    @contextmanager
    def _span(
        self,
        spec: TraceEventSpec,
        *,
        carrier: TraceCarrier | None = None,
        payload: tuple[int, ...] = EMPTY_TRACE_PAYLOAD,
    ) -> Generator[None, None, None]:
        active = self._root_carrier if carrier is None else carrier
        track_id, name_id = self._register_spec(spec)
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            end = time.perf_counter_ns()
            if not self._skip_record(spec, payload):
                self._arena.append(
                    kind=TraceEventKind.SLICE_BEGIN,
                    ts_ns=start,
                    dur_ns=end - start,
                    track_id=track_id,
                    name_id=name_id,
                    flow_id=active.flow_id,
                    task_id=active.task_id,
                    parent_task_id=active.parent_task_id,
                    payload=payload,
                )

    def _slice(
        self,
        spec: TraceEventSpec,
        *,
        start_ns: int,
        dur_ns: int,
        carrier: TraceCarrier | None = None,
        payload: tuple[int, ...] = EMPTY_TRACE_PAYLOAD,
    ) -> None:
        if self._skip_record(spec, payload):
            return
        active = self._root_carrier if carrier is None else carrier
        track_id, name_id = self._register_spec(spec)
        self._arena.append(
            kind=TraceEventKind.SLICE_BEGIN,
            ts_ns=start_ns,
            dur_ns=dur_ns,
            track_id=track_id,
            name_id=name_id,
            flow_id=active.flow_id,
            task_id=active.task_id,
            parent_task_id=active.parent_task_id,
            payload=payload,
        )

    def _skip_record(self, spec: TraceEventSpec, payload: tuple[int, ...]) -> bool:
        if not self._step_filter.has_filter:
            return False
        if spec.step_payload_field is None:
            return False
        try:
            step_index = spec.payload_fields.index(spec.step_payload_field)
        except ValueError as exc:
            raise ValueError(
                f"step_payload_field={spec.step_payload_field!r} is not present "
                f"in payload schema for event={spec.name!r}."
            ) from exc
        if step_index >= len(payload):
            raise ValueError(
                f"event={spec.name!r} payload length {len(payload)} does not contain "
                f"step field {spec.step_payload_field!r} at index {step_index}."
            )
        step = payload[step_index]
        return not self._step_filter.is_step_profiled(step)

    @asynccontextmanager
    async def _async_span(
        self,
        spec: TraceEventSpec,
        *,
        carrier: TraceCarrier | None = None,
        payload: tuple[int, ...] = EMPTY_TRACE_PAYLOAD,
    ) -> AsyncGenerator[None, None]:
        with self._span(
            spec,
            carrier=carrier,
            payload=payload,
        ):
            yield

    def _register_spec(self, spec: TraceEventSpec) -> tuple[int, int]:
        track_id = self.intern(f"{self._process_name}/{spec.track}")
        name_id = self.intern(spec.name)
        with self._intern_lock:
            existing = self._payload_fields_by_name_id.get(name_id)
            if existing is None:
                self._payload_fields_by_name_id[name_id] = spec.payload_fields
            elif existing != spec.payload_fields:
                raise ValueError(
                    f"trace payload schema mismatch for event={spec.name!r}: "
                    f"{existing!r} != {spec.payload_fields!r}"
                )
        return track_id, name_id
