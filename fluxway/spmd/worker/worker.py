"""
SPMD Worker: Ray Actor for computation execution.

Based on ARCHITECTURE.md Section 3.2

Each SPMD Worker runs user-defined forward/backward/update logic
and implements dual-stream convergence with mailbox-based execution.

Key responsibilities:
1. Provide handler registration interface (handlers only, topology from config)
2. Provide data reception interface
3. Execute operators from priority queue
4. Send outputs to successors using computed routing
5. Provide op-level blocking (block one op without affecting others)
6. Report op completion to driver
7. Handle control_argument reception and propagate to successors

This implementation uses Ray's async actor API for concurrency.
See: https://docs.ray.io/en/latest/ray-core/actors/async_api.html
"""

import asyncio
from functools import partial
from contextlib import nullcontext
from dataclasses import dataclass, replace
import os
import socket
import time
import inspect
from collections.abc import Awaitable, Callable, Mapping


import ray

from fluxway.common.tensor_backend import (
    get_backend,
    force_writable,
    ensure_cpu_serializable,
    set_hybrid_device,
)
from fluxway.common.decorators import (
    operator_context,
    get_current_context,
    _inject_context_properties,
    _STATIC_CONTEXT_ATTR_NAMES,
    _validate_worker_process_marker,
)
from fluxway.common.canonical_lanes import compute_lane_priority
from fluxway.common.operator_instances import OperatorInstanceContainer
from fluxway.common.execution import compile
from fluxway.common.compiled_graph import CompiledDependency
from fluxway.common.ingestion import (
    SourceWorkAcknowledgement,
    SourceWorkItem,
    SourceWorkLease,
)
from fluxway.common.ingestion.segment import CompiledSegment
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types import (
    CheckpointComponentPublication,
    CheckpointCommitPermit,
    CheckpointCut,
    checkpoint_position,
    ExecutionFailure,
    WorkerSaveResult,
    SPMDConfig,
    TopologySnapshot,
    WorkerHeartbeat,
    WorkerStatus,
    ExecutionWaitState,
    CheckpointActivityPhase,
    DrainBoundary,
    DrainEvent,
    DrainEventKind,
    DrainPhase,
    reduce_drain_boundary,
    RestoreOptionsByComponent,
)
from fluxway.common.coherence import (
    CoherenceAction,
    CoherenceDirectory,
    InvalidationResult,
    CoherenceState,
    CoherenceSubject,
    CoherenceStateTransitionError,
    ExecutionFenceState,
    ExecutionToken,
    GrantKind,
    GrantPending,
    GrantReady,
    GrantRejected,
    GrantToken,
    TensorAccessIntent,
    CoherenceInstance,
    UnsupportedAllocationHandoffError,
)
from fluxway.common.allocation import AllocationRegistry, AllocationRelation
from fluxway.common.types.tensor import (
    SymmetryReceipt,
    TransportTensor,
    runtime_allocation_descriptor,
)
from fluxway.common.resource_registry import (
    ResourceRegistry,
    set_active_resource_registry,
    reset_active_resource_registry,
)
from fluxway.common.logical_partition_binding import LogicalPartitionBinding
from fluxway.control_plane.ledger_client import ControlPlaneClient
from fluxway.common.types.sharding import (
    ShardingSpec,
    PhysicalLayout,
)
from fluxway.common.index_descriptor import ContinuousIndex, IndexDescriptor
from fluxway.common.operator_initialization import operator_initialization_arguments
from fluxway.common.operator_policy import SchedulePolicy
from fluxway.common.schedule_semantics import (
    CompiledPhaseSchedule,
    compile_phase_schedule,
)
from fluxway.transport.manifest import (
    ENTRY_AUX,
    ENTRY_EXPECTED_COUNT,
    ENTRY_ROUTE_CODE,
    ENTRY_STATE_FLAGS,
    DependencyDecision,
    FRAME_ENTRY_COUNT,
    FRAME_FLAGS,
    FRAME_SENDER_SPMD_CODE,
    FRAME_SENDER_RANK,
    FRAME_STEP,
    FrameFlag,
    PayloadRole,
    TensorEntryKind,
    TransportFrame,
    TransportFrameHeader,
)
from fluxway.transport.coherence import entry_coherence_action, entry_tensor_kind
from fluxway.common.utils import ensure_logging_configured
from fluxway.common.utils.logging import get_logger
from fluxway.common.utils.tree import tree_flatten, tree_unflatten
from fluxway.diagnostics import register_system_signal_metadata
from fluxway.rpc.protocol import (
    DataPlaneCallable,
    LocalWorkerDataPlane,
    TransportFrameEnvelope,
)
from fluxway.spmd.worker.mailbox import WorkerMailbox
from fluxway.spmd.worker.mailbox.mailbox import TransportShardView
from fluxway.spmd.worker.checkpoint import WorkerCheckpointCoordinator
from fluxway.spmd.worker.checkpoint_activity import (
    CheckpointActivity,
)
from fluxway.spmd.worker.data_plane import (
    CompiledSendPlan,
    DependencyDecisions,
    SuccessorRouteContext,
    build_coherence_frame,
    compile_send_plan,
    evaluate_dependency_decisions,
    ensure_transport_source_step,
    prepare_route_tensor,
    skipped_dependency_decisions,
)
from fluxway.spmd.worker.data_plane_executor import BatchingDataPlaneExecutor
from fluxway.spmd.worker.coherence_runtime import (
    CoherenceAdmissionRuntime,
    CoherenceBatchRetireRuntime,
    InputCoherenceRelease,
    InputCoherenceReleasePlan,
    ParentGrantReturnRuntime,
    ParentReturnRoute,
)
from fluxway.spmd.worker.coherence_sequencer import CoherenceSequencer
from fluxway.spmd.worker.assembly import (
    CompiledAssemblePlan,
    compile_assemble_plan,
)
from fluxway.spmd.worker.mailbox import (
    DirectLocalInputSlot,
    DirectLocalMailbox,
    DelegatedWorkerMailbox,
    OperatorMailbox,
    create_operator_mailbox,
)
from fluxway.spmd.worker.local_plan import (
    LocalNodeSpec,
    LocalStepContext,
    LocalStepNodeKind,
    LocalStepPlan,
    LocalStepPlanExecutor,
    MappingCallableRegistry,
)
from fluxway.spmd.worker.user_callable_executor import (
    USER_CALLABLE_DEFAULT_OPERATOR_LANE,
    SPMDObjectScope,
    UserCallableExecutor,
    UserCallableSpec,
)
from fluxway.spmd.worker.runtime import (
    SourceAcknowledgementRuntime,
    LocalInputPrepareRuntime,
    OutboundDeliveryRuntime,
    WorkerProcessCallRuntime,
)
from fluxway.spmd.worker.operator_manager.operator_manager import (
    OperatorManager,
    ResolvedDependencyRoute,
)
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionStatus,
    ExecutionResult,
)
from fluxway.spmd.worker.source_scope import (
    SourceExecutionScope,
    SourceExhaustedError,
)
from fluxway.spmd.worker.lifecycle import WorkerExecutionLifecycle, WorkerPhase
from fluxway.spmd.worker.admission import AdmissionAction, StepAdmissionState
from fluxway.spmd.worker.source_maintenance import SourcePipelineMaintenance
from fluxway.profiler import ProfilerRuntimeFactory
from fluxway.profiler.record import ProfilerSnapshot
from fluxway.profiler.recorder import (
    LOCAL_STEP_AWAIT_SEND,
    LOCAL_STEP_CALL_USER,
    LOCAL_STEP_COMMIT,
    LOCAL_STEP_SEND_OUTPUT,
    LOCAL_STEP_WAIT_INPUTS,
    WORKER_ASSEMBLE_CAT_VALUES,
    WORKER_ASSEMBLE_FILTER_SHARDS,
    WORKER_ASSEMBLE_MERGE_METADATA,
    WORKER_ASSEMBLE_SCATTER,
    WORKER_ASSEMBLE_SORT,
    WORKER_ADVANCE_STEP,
    WORKER_ADMISSION,
    WORKER_BACKGROUND_TASK,
    WORKER_DRAIN_WAIT,
    WORKER_QUEUE_DEQUEUE,
    WORKER_QUEUE_ENQUEUE,
    WORKER_RUN_LOOP,
)
from fluxway.common.types.drain import DrainAdmission
from typing import Any, Optional, get_type_hints, cast

_WAIT_INPUTS_CALLABLE = "__fluxway_wait_inputs__"
_CALL_USER_OP_CALLABLE = "__fluxway_call_user_op__"
_SEND_OUTPUT_CALLABLE = "__fluxway_send_output__"
_AWAIT_SEND_CALLABLE = "__fluxway_await_send__"
_COMMIT_STEP_CALLABLE = "__fluxway_commit_step__"
_WAIT_INPUTS_RESOURCE = -1
_SEND_OUTPUT_RESOURCE = -1
_AWAIT_SEND_RESOURCE = -1
_COMMIT_STEP_RESOURCE_SALT = "__fluxway_commit_step_resource__"
_EXECUTION_DEFAULT_LANE = USER_CALLABLE_DEFAULT_OPERATOR_LANE
_EXECUTION_STOP_OP = "__stop__"
_EXECUTION_STOP_ITEM = (999, 0, 0, _EXECUTION_STOP_OP)


def _classify_execution_wait_state(
    *,
    execution_phase: str | None,
    execution_phase_active: bool,
    active_plan_steps: bool,
    pending_plan_tasks: bool,
    pending_send_tasks: bool,
    pending_input_prepare_tasks: bool,
    queued_work: bool,
    pending_admission: bool,
) -> ExecutionWaitState:
    if execution_phase_active and execution_phase is not None:
        if execution_phase.startswith("source_scope."):
            return ExecutionWaitState.SOURCE_ADMISSION_WAIT
        return ExecutionWaitState.OPERATOR_EXECUTION_ACTIVE
    if active_plan_steps or pending_plan_tasks:
        return ExecutionWaitState.OPERATOR_EXECUTION_ACTIVE
    if pending_send_tasks:
        return ExecutionWaitState.OUTBOUND_TRANSPORT_WAIT
    if pending_input_prepare_tasks:
        return ExecutionWaitState.INPUT_PREPARE_WAIT
    if queued_work:
        return ExecutionWaitState.EXECUTION_QUEUE_WAIT
    if pending_admission:
        return ExecutionWaitState.OPERATOR_ADMISSION_WAIT
    return ExecutionWaitState.COMPLETION_BARRIER_OR_DRIVER_WAIT


def _assemble_detail_span(
    worker: Any,
    *,
    spec: Any,
    op_name: str,
    transfer_code: int,
    logical_tuuid: int,
    step: int,
    first_count: int,
    second_count: int = 0,
) -> Any:
    profiler_runtime = getattr(worker, "_profiler_runtime", None)
    if profiler_runtime is None or not getattr(profiler_runtime, "enabled", False):
        return nullcontext()
    return profiler_runtime.recorder.worker_assemble_detail(
        spec=spec,
        op_name=op_name,
        transfer_code=transfer_code,
        logical_tuuid=logical_tuuid,
        step=step,
        first_count=first_count,
        second_count=second_count,
    )


def _profiler_enabled(worker: Any) -> bool:
    profiler_runtime = getattr(worker, "_profiler_runtime", None)
    return bool(
        profiler_runtime is not None and getattr(profiler_runtime, "enabled", False)
    )


def _route_inspection_snapshot(
    route: ResolvedDependencyRoute,
) -> dict[str, Any]:
    dependency = route.dependency
    transfer = route.transfer
    source = dependency.spec.source
    target = dependency.spec.target
    return {
        "dependency_code": dependency.code,
        "transfer_code": transfer.transfer_code,
        "logical_tuuid": transfer.logical_tuuid,
        "source": f"{source.spmd_id}.{source.op_id}.{transfer.spec.source.name}",
        "target": f"{target.spmd_id}.{target.op_id}.{transfer.spec.target.name}",
        "peer_spmd_id": route.spmd_id,
        "peer_world_size": route.world_size,
        "local_only": dependency.spec.local_only,
        "source_sharding": str(transfer.source_sharding),
        "target_sharding": str(transfer.target_sharding),
    }


def _nonzero_1d_indices(backend: Any, mask: Any) -> Any:
    nonzero_result = backend.nonzero(mask)
    if isinstance(nonzero_result, tuple):
        if len(nonzero_result) != 1:
            raise ValueError(
                "Expected a one-dimensional mask for dynamic-to-static assembly."
            )
        return nonzero_result[0]

    result_shape = backend.shape(nonzero_result)
    if len(result_shape) == 1:
        return nonzero_result
    if len(result_shape) == 2 and result_shape[1] == 1:
        return nonzero_result.flatten()
    raise ValueError("Expected a one-dimensional mask for dynamic-to-static assembly.")


@compile(static_argnums=())
def _jit_sort_fused(values, origin_indices, routing_indices, receipt_tags):
    backend = get_backend()
    sort_idx = backend.argsort(origin_indices)
    return (
        backend.take(values, sort_idx),
        backend.take(origin_indices, sort_idx),
        backend.take(routing_indices, sort_idx),
        backend.take(receipt_tags, sort_idx),
        sort_idx,
    )


def _pad_tensor_to_power_of_2(tensor, original_length: int, cap: int, pad_value=0):
    if cap == original_length:
        return tensor
    backend = get_backend()
    pad_shape = list(backend.shape(tensor))
    pad_shape[0] = cap - original_length
    if pad_value == 0:
        zeros = backend.zeros(tuple(pad_shape), dtype=backend.dtype(tensor))
    else:
        zeros = backend.full(tuple(pad_shape), pad_value, dtype=backend.dtype(tensor))
    return backend.cat([tensor, zeros], dim=0)


def _is_strictly_increasing_1d(tensor: Any) -> bool:
    backend = get_backend()
    shape = backend.shape(tensor)
    if not shape or int(shape[0]) <= 1:
        return True
    return bool(backend.to_numpy(backend.min(tensor[1:] > tensor[:-1])))


def _continuous_index_is_strictly_increasing(index: ContinuousIndex) -> bool:
    return index.count <= 1 or index.step > 0


@compile(static_argnums=(1,))
def _jit_reduce_scalars(stacked, op: str):
    backend = get_backend()
    if op == "sum":
        return backend.sum(stacked)
    if op == "mean":
        return backend.mean(stacked)
    if op == "min":
        return backend.min(stacked)
    if op == "max":
        return backend.max(stacked)
    raise ValueError(f"Unsupported reduction op: {op}")


def _describe_layout_for_error(layout: ShardingSpec | PhysicalLayout | None) -> str:
    if layout is None:
        return "None"

    if isinstance(layout, ShardingSpec):
        strategy_name = type(layout.strategy).__name__
        return (
            f"ShardingSpec(strategy={strategy_name}, "
            f"content={layout.is_content_addressable}, "
            f"static={layout.is_static}, "
            f"partitions={layout.partitions}, "
            f"routing_dim={layout.routing_dim})"
        )

    if isinstance(layout, PhysicalLayout):
        strategy_name = type(layout.strategy).__name__
        return (
            f"PhysicalLayout(strategy={strategy_name}, "
            f"content={layout.is_content_addressable}, "
            f"static={layout.is_static}, "
            f"partitions={layout.partitions}, "
            f"routing_dim={layout.routing_dim}, "
            f"global_shape={layout.global_shape}, "
            f"mesh_shape={layout.mesh.shape}, "
            f"mesh_axes={layout.mesh.axis_names})"
        )

    raise TypeError(
        f"[Fluxway Protocol Error] Unsupported layout descriptor "
        f"{type(layout).__name__}; expected ShardingSpec, PhysicalLayout, or None."
    )


logger = get_logger(__name__)


def _resolve_logical_partition_bindings(
    value: Any, logical_partition_id: int, logical_mesh: Any
) -> Any:
    if isinstance(value, LogicalPartitionBinding):
        return value.resolve(logical_partition_id, logical_mesh=logical_mesh)
    if isinstance(value, Mapping):
        return {
            key: _resolve_logical_partition_bindings(
                item, logical_partition_id, logical_mesh
            )
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _resolve_logical_partition_bindings(
                item, logical_partition_id, logical_mesh
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            _resolve_logical_partition_bindings(
                item, logical_partition_id, logical_mesh
            )
            for item in value
        ]
    return value


@dataclass(frozen=True)
class _SuccessorSendContext:
    route_context: SuccessorRouteContext
    handlers: dict[int, Any]

    @property
    def spmd_id(self) -> str:
        return self.route_context.spmd_id

    @property
    def dependencies(self) -> tuple[CompiledDependency, ...]:
        return self.route_context.dependencies


@dataclass(frozen=True)
class _OperatorSendContext:
    compiled_plan: CompiledSendPlan
    successor_contexts: tuple[_SuccessorSendContext, ...]
    successor_handlers: tuple[dict[int, Any], ...]
    transport: "_TransportSendContext | None"


@dataclass(frozen=True)
class _TransportSendContext:
    compiled_plan: CompiledSendPlan
    successor_handlers: tuple[dict[int, Any], ...]
    condition_codes: tuple[int, ...] | None

    def select_dependency_decisions(
        self,
        dependency_decisions: DependencyDecisions,
    ) -> DependencyDecisions:
        if self.condition_codes is None:
            return dependency_decisions
        if not self.condition_codes:
            return None
        if dependency_decisions is None:
            raise ValueError(
                f"Operator {self.compiled_plan.op_name!r} requires transport "
                "dependency decisions."
            )
        return {code: dependency_decisions[code] for code in self.condition_codes}


def _compile_transport_send_context(
    compiled_plan: CompiledSendPlan,
    successor_handlers: tuple[dict[int, Any], ...],
) -> _TransportSendContext | None:
    if len(successor_handlers) != len(compiled_plan.successor_routes):
        raise ValueError(
            f"Operator {compiled_plan.op_name!r} has "
            f"{len(compiled_plan.successor_routes)} successor routes but "
            f"{len(successor_handlers)} handler groups."
        )
    context_indices = tuple(
        index
        for index, route in enumerate(compiled_plan.successor_routes)
        if not route.local_only
    )
    if not context_indices:
        return None
    transport_plan = compiled_plan.project_successors(context_indices)
    condition_codes = (
        None
        if transport_plan is compiled_plan
        else tuple(
            condition.dependency_code
            for condition in transport_plan.dependency_conditions
        )
    )
    return _TransportSendContext(
        compiled_plan=transport_plan,
        successor_handlers=(
            successor_handlers
            if transport_plan is compiled_plan
            else tuple(successor_handlers[index] for index in context_indices)
        ),
        condition_codes=condition_codes,
    )


@dataclass(frozen=True)
class LocalSendToken:
    """SEND registration token; delivery completion is fenced by OutboundDeliveryRuntime."""

    op_name: str
    step: int
    result: ExecutionResult
    send_task: asyncio.Task[ExecutionResult] | None


@dataclass(frozen=True)
class _PreparedOperatorInput:
    value: Any
    step_transfer_cache: dict[int, Any]
    tensor_bindings: tuple["_PreparedTensorBinding", ...] = ()
    source_work_items: tuple[SourceWorkItem, ...] | None = None


@dataclass(frozen=True)
class _PreparedTensorBinding:
    port: str
    transfer_code: int
    tensor: TransportTensor
    grants: tuple["_InputCoherenceGrant", ...]


_DirectLocalInputSlot = DirectLocalInputSlot


@dataclass(frozen=True)
class _InputCoherenceGrant:
    parent_spmd_id: str | None
    parent_rank: int
    subject: CoherenceSubject
    route_code: int
    epoch: int
    local_access_token: GrantToken
    parent_return_required: bool = True

    @property
    def identity(self) -> tuple[str | None, int, CoherenceSubject, int, int]:
        return (
            self.parent_spmd_id,
            self.parent_rank,
            self.subject,
            self.route_code,
            self.epoch,
        )


@dataclass(slots=True)
class _OwnershipReservation:
    grants: tuple[_InputCoherenceGrant, ...]
    released_read_tokens: list[GrantToken]
    write_token: GrantToken | None = None
    pending: GrantPending | None = None
    ownership_granted: bool = False

    @property
    def canonical(self) -> _InputCoherenceGrant:
        return min(self.grants, key=lambda grant: grant.route_code)


@dataclass(frozen=True, slots=True)
class _AllocationHandoffIntent:
    input_tensor: TransportTensor
    output_tensor: Any
    old_instance: CoherenceInstance
    new_instance: CoherenceInstance
    output_descriptor: Any


@ray.remote
class SPMDWorker:
    """
    SPMD Worker (Ray Async Actor).

    Implements dual-stream convergence with priority queue execution:
    - Data is received by WorkerMailbox (async-safe)
    - Executes operators based on priority metadata
    - Supports op-level blocking without affecting other ops
    - Reports op completion to driver
    - Handles control_argument for cross-SPMD state transfer

    Uses asyncio for concurrency instead of threading.
    Ray will automatically detect this as an async actor due to async method definitions.
    """

    _completed_step_count: dict[str, int]
    _scheduled_step_count: dict[str, int]
    _pending_future_steps: dict[str, list[int]]
    _pending_phase_order_steps: dict[str, list[int]]
    _phase_schedule: CompiledPhaseSchedule
    _successor_handlers: dict[str, dict[int, Any]]
    _operator_send_context_cache: dict[str, _OperatorSendContext]
    _heartbeat_interval: float
    _failure: ExecutionFailure | None
    _op_runnable: dict[str, "asyncio.Event"]
    _mailbox: WorkerMailbox
    _drain_cond: "asyncio.Condition"

    @property
    def op_manager(self) -> OperatorManager:
        if self._op_manager is None:
            raise RuntimeError("OperatorManager is not initialized yet.")
        return self._op_manager

    def __init__(
        self,
        config: SPMDConfig,
        rank: int,
        spmd_id: str = "",
        driver_handle: Any | None = None,
        tensor_device: str = "cpu",
        control_plane_handle: Any | None = None,
    ):
        """
        Initialize worker.

        Args:
            config: SPMD configuration with user functions
            rank: This worker's rank in the SPMD
            spmd_id: ID of this SPMD (for metadata tracking)
            driver_handle: Reference to the SPMD driver for callbacks (Native Ray)
            node_type: Type of node (SPMD or READER)
            reader_task: File read task (for READER nodes)
            reader_config: Reader configuration (for READER nodes)
        """
        # Ray actors run in separate worker processes; ensure they respect
        # FLUXWAY_LOG_LEVEL / FLUXWAY_LOG_FORCE so DEBUG logs are not dropped.
        ensure_logging_configured()
        set_hybrid_device(tensor_device)
        logger.debug("phase_one_init_begin spmd_id=%s rank=%s", spmd_id, rank)
        self.config = config
        self.rank = rank
        self.spmd_id = spmd_id or ""
        register_system_signal_metadata(spmd_id=self.spmd_id, rank=self.rank)

        self._resource_registry = ResourceRegistry()

        self._tensor_device = tensor_device
        self._execution_lifecycle = WorkerExecutionLifecycle()
        self._execution_generation = 0
        self._execution_epoch = 0
        self._execution_sequence = 0
        self._execution_operator: str | None = None
        self._execution_step: int | None = None
        self._execution_phase: str | None = None
        self._execution_phase_active = False
        self._execution_phase_started_timestamp = time.time()
        self._last_execution_activity_timestamp = time.time()
        self._graph_view = config.require_graph_view()
        self._operator_specs = {
            op_id: config.operator_specs[op_id]
            for op_id in self._graph_view.operator_ids
        }
        self._phase_schedule = compile_phase_schedule(
            {
                op_id: spec.runtime_policy.schedule
                for op_id, spec in self._operator_specs.items()
            }
        )

        self._checkpoint_coordinator: WorkerCheckpointCoordinator | None = None
        self._checkpoint_repository_root: str | None = None

        # Driver handle for callbacks (Control Plane - Native Ray)
        self._driver_handle: Any | None = driver_handle
        self._control_plane_handle: Any | None = control_plane_handle

        # Mesh from config
        self._mesh = self._graph_view.mesh
        self._world_size = self._graph_view.world_size
        self.local_rank: int | None = None
        self.local_world_size: int | None = None
        self.node_rank: int | None = None
        self.num_nodes: int | None = None
        self.rank0_address: str | None = None

        self._op_manager: OperatorManager | None = None
        self._source_scope = SourceExecutionScope(
            acquire=self._acquire_source_work,
            spawn=self._spawn_background_task,
            completed=lambda operator: self._completed_step_count[operator],
            depth=self._operator_pipeline_depth,
        )
        self._instance_container = OperatorInstanceContainer()
        operator_names = self._graph_view.operator_ids

        # Completed step counters {op_name: count} - tracks how many steps were executed
        self._completed_step_count: dict[str, int] = dict.fromkeys(operator_names, 0)
        self._scheduled_step_count: dict[str, int] = dict.fromkeys(operator_names, 0)

        # Op-level blocking: asyncio.Event (set=runnable, clear=blocked)
        self._op_runnable = {op_name: asyncio.Event() for op_name in operator_names}
        self._latest_op_runnable_epoch: dict[str, int] = dict.fromkeys(
            operator_names, -1
        )
        for event in self._op_runnable.values():
            event.set()  # Initially all ops are runnable

        # Pending tasks when blocked (to avoid priority queue reentry loop)
        self._pending_blocked: dict[str, list[int]] = {
            op_name: [] for op_name in operator_names
        }

        # Pending tasks for future steps (arrived early, waiting for current step to complete)
        # Structure: {op_name: List[step]} - steps waiting to be executed
        self._pending_future_steps: dict[str, list[int]] = {
            op_name: [] for op_name in operator_names
        }
        self._pending_async_lag_steps: dict[str, list[int]] = {
            op_name: [] for op_name in operator_names
        }
        self._pending_phase_order_steps: dict[str, list[int]] = {
            op_name: [] for op_name in operator_names
        }
        self._running_plan_steps: dict[str, set[int]] = {
            op_name: set() for op_name in operator_names
        }
        self._finished_plan_steps: dict[str, dict[int, ExecutionResult]] = {
            op_name: {} for op_name in operator_names
        }
        self._ready_pending_pipeline_steps: dict[str, list[int]] = {
            op_name: [] for op_name in operator_names
        }
        self._mailbox_ready_steps: dict[str, set[int]] = {
            op_name: set() for op_name in operator_names
        }
        self._admission_state = StepAdmissionState(
            completed=self._completed_step_count,
            scheduled=self._scheduled_step_count,
            queue_membership=set(),
            pending_blocked=self._pending_blocked,
            pending_future=self._pending_future_steps,
            pending_async_lag=self._pending_async_lag_steps,
            pending_phase_order=self._pending_phase_order_steps,
            ready_pipeline=self._ready_pending_pipeline_steps,
            mailbox_ready=self._mailbox_ready_steps,
            running=self._running_plan_steps,
            finished=self._finished_plan_steps,
            is_periodic=lambda op_name: self.op_manager.is_periodic_trigger(op_name),
            is_runnable=lambda op_name: self._op_runnable[op_name].is_set(),
            hold_prepare_token=self._should_hold_prepare_pending_token,
            async_lag_blocked=self._async_lag_limit_reached,
            phase_order_blocked=self._phase_order_step_blocked,
            can_start=self._can_start_local_plan,
        )
        self._in_op_step_queue = self._admission_state.queue_membership
        self._plan_tasks: set[asyncio.Task] = set()
        self._plan_task_steps: dict[asyncio.Task, tuple[str, int]] = {}
        self._send_tasks: dict[asyncio.Task, tuple[str, int]] = {}

        # Priority queue: asyncio.PriorityQueue with (priority, prepare_state, step, op_name).
        # All cross-operator ordering goes through _effective_schedule_key().
        self._execution_queue: asyncio.PriorityQueue[tuple[int, int, int, str]] = (
            asyncio.PriorityQueue()
        )
        self._execution_queues: dict[
            str, asyncio.PriorityQueue[tuple[int, int, int, str]]
        ] = {_EXECUTION_DEFAULT_LANE: self._execution_queue}
        self._initialize_execution_lanes()

        # --- Backpressure components ---
        self._max_in_flight = int(getattr(self.config, "data_plane_max_in_flight", 50))
        if self._max_in_flight <= 0:
            raise ValueError(
                f"data_plane_max_in_flight must be positive, got {self._max_in_flight}."
            )
        self._data_plane_batch_max_frames = int(
            getattr(self.config, "data_plane_batch_max_frames", 8)
        )
        if self._data_plane_batch_max_frames <= 0:
            raise ValueError(
                "data_plane_batch_max_frames must be positive, "
                f"got {self._data_plane_batch_max_frames}."
            )
        self._data_plane_batch_flush_us = int(
            getattr(self.config, "data_plane_batch_flush_us", 100)
        )
        if self._data_plane_batch_flush_us < 0:
            raise ValueError(
                "data_plane_batch_flush_us must be non-negative, "
                f"got {self._data_plane_batch_flush_us}."
            )
        self._mailbox_capacity = 100  # Default value
        self._high_watermark = int(self._mailbox_capacity * 0.8)

        # Handler storage (Data Plane - wrapped in DataPlaneCallable)
        # {spmd_id: {rank: DataPlaneCallable}}
        self._predecessor_handlers: dict[str, dict[int, DataPlaneCallable]] = {}
        self._successor_handlers = {}
        self._operator_send_context_cache = {}
        self._local_step_plan_cache: dict[str, LocalStepPlan] = {}
        self._assemble_plan_cache: dict[tuple[str, int], CompiledAssemblePlan] = {}
        self._operator_mailboxes: dict[str, OperatorMailbox] = {}
        self._direct_local_input_slots: dict[
            str, dict[int, dict[int, _DirectLocalInputSlot]]
        ] = {}
        self._direct_local_ready_steps: dict[str, set[int]] = {
            op_name: set() for op_name in operator_names
        }
        self._local_input_origins: dict[tuple[str, int, int], tuple[str, int]] = {}
        self._local_spmd_code: int | None = None
        self._spmd_id_by_code: dict[int, str] = {}
        self._coherence_input_grants: dict[
            tuple[str, int],
            dict[
                tuple[str | None, int, CoherenceSubject, int, int], _InputCoherenceGrant
            ],
        ] = {}
        self._coherence_revocations: set[CoherenceInstance] = set()
        self._coherence_input_tensors: dict[
            tuple[str, int],
            set[TransportTensor],
        ] = {}
        self._coherence_input_bindings: dict[
            tuple[str, int], tuple[_PreparedTensorBinding, ...]
        ] = {}
        self._coherence_alias_dependencies: dict[
            tuple[str, int], set[CoherenceInstance]
        ] = {}
        self._coherence_output_intents: dict[
            tuple[str, int], tuple[CoherenceInstance, ...]
        ] = {}

        # Heartbeat configuration
        self._heartbeat_interval: float = self.config.heartbeat_interval_s
        self._heartbeat_running: bool = False
        self._heartbeat_task: asyncio.Task | None = None
        self._checkpoint_activity = CheckpointActivity()

        # State tracking
        self._failure: ExecutionFailure | None = None
        self._fence_state = ExecutionFenceState.OPEN
        self._last_progress_timestamp = time.time()
        self._last_stall_warning_timestamp: float | None = None

        # Run loop synchronization
        self._run_task: asyncio.Task | None = None
        self._lane_run_tasks: dict[str, asyncio.Task] = {}
        self._background_tasks: set[asyncio.Task] = set()
        profiler_config = getattr(self.config, "__dict__", {}).get("profiler_config")
        profiler_process_name = f"worker/{self.spmd_id}/rank:{self.rank}"
        self._profiler_runtime = ProfilerRuntimeFactory.from_config(
            profiler_config,
            process_name=profiler_process_name,
        )
        self._mailbox = self._profiler_runtime.create_worker_mailbox()
        self._mailbox.set_ready_callback(self._on_step_ready)
        self._coherence_directory = CoherenceDirectory()
        self._coherence_admission_runtime = CoherenceAdmissionRuntime(
            self._coherence_directory
        )
        self._coherence_sequencer = CoherenceSequencer(
            self._retire_coherence_payload,
            self._coherence_admission_runtime.admit,
        )
        self._coherence_batch_retire_runtime = CoherenceBatchRetireRuntime(
            self._coherence_directory,
            self._coherence_sequencer,
        )
        self._parent_grant_return_runtime = ParentGrantReturnRuntime(
            self._coherence_directory,
            self._send_parent_return_group,
        )
        self._allocation_registry = AllocationRegistry()
        self._data_plane_executor = self._profiler_runtime.data_plane_backend_executor(
            BatchingDataPlaneExecutor()
        )
        self._worker_process_call_runtime = WorkerProcessCallRuntime(
            max_workers=self._worker_process_call_max_workers(),
            profiler_runtime=self._profiler_runtime,
            thread_name_prefix=f"fluxway-call-{self.spmd_id}-{self.rank}",
        )
        user_callable_executor = self._profiler_runtime.user_callable_executor(
            UserCallableExecutor(
                worker_process_runtime=self._worker_process_call_runtime,
            )
        )
        self._user_callable_executor = user_callable_executor
        self._local_step_plan_executor = (
            self._profiler_runtime.local_step_plan_executor(
                LocalStepPlanExecutor(user_callable_executor)
            )
        )
        self._input_prepare_runtime = LocalInputPrepareRuntime(
            background_tasks=self._background_tasks,
            on_task_done=self._on_background_task_done,
            profiler_runtime=self._profiler_runtime,
            on_prepare_ready=self._on_prepare_ready,
            max_workers=self._input_prepare_max_workers(),
            schedule_key=self._effective_schedule_key,
        )
        self._delivery_runtime = OutboundDeliveryRuntime(
            max_in_flight=self._max_in_flight,
            profiler_runtime=self._profiler_runtime,
            background_tasks=self._background_tasks,
            on_task_done=self._on_background_task_done,
        )

        # Source operator pause control
        self._source_pause_events: dict[str, asyncio.Event] = {}
        self._source_maintenance_tasks: dict[str, asyncio.Task] = {}
        self._source_maintenance = SourcePipelineMaintenance(
            tasks=self._source_maintenance_tasks,
            spawn=self._spawn_background_task,
            is_source=lambda op_name: self.op_manager.is_source_operator(op_name),
            running=lambda: self._running,
            exhausted=self._source_is_exhausted,
            depth=self._operator_pipeline_depth,
            completed=lambda op_name: self._completed_step_count.get(op_name, 0),
            max_steps=lambda: self.config.max_steps,
            drain_blocks=self._drain_blocks_step,
            known_steps=self._admission_state.known_steps,
            enqueue=self._enqueue,
        )
        self._source_operator_names = tuple(
            op_name for op_name, spec in self._operator_specs.items() if spec.is_source
        )
        self._source_control_operator_names = tuple(
            op_name
            for op_name, spec in self._operator_specs.items()
            if spec.is_source and spec.requires_source_control_runtime
        )
        self._source_acknowledgements = SourceAcknowledgementRuntime(
            capacity=max(
                1,
                sum(
                    self._operator_pipeline_depth(name)
                    for name in self._source_control_operator_names
                ),
            ),
            send=self._send_source_acknowledgements,
            spawn=self._spawn_background_task,
        )
        self._drain_boundary = DrainBoundary()
        self._admission: DrainAdmission | None = None
        self._pending_terminal_observation: tuple[str, int] | None = None

        # Drain-to-step state
        self._drain_step_count: int | None = None
        self._drain_cond: asyncio.Condition = asyncio.Condition()
        if self.config is not None and getattr(
            self.config, "requires_drain_at_startup", False
        ):
            # Block at step 0 until Controller issues the initial drain target!
            self._drain_step_count = 0

        # Expected topology counts for topology readiness
        # Initialize to None to strictly wait for registration from driver
        self._expected_predecessor_count: int | None = None
        self._expected_successor_count: int | None = None

        # Tracks which (op_name, step) barriers have been registered to avoid double-declaring.
        self._barriers_declared: dict[tuple[str, int], bool] = {}

        logger.debug(
            "phase_one_init_complete spmd_id=%s rank=%s world_size=%s",
            self.spmd_id,
            self.rank,
            self._world_size,
        )

    # =========================================================================
    # Operator Pre-Instantiation
    # =========================================================================

    @staticmethod
    def _pre_instantiate_operators(
        config: "SPMDConfig",
        rank: int,
        *,
        static_args: dict[str, Any],
        instance_container: OperatorInstanceContainer,
        local_rank: int | None = None,
        local_world_size: int | None = None,
        node_rank: int | None = None,
        num_nodes: int | None = None,
        rank0_address: str | None = None,
    ) -> None:
        """Eagerly instantiate all SPMD operator class instances.

        This runs inside SPMDWorker.__init__, before OperatorManager is created
        and before the Ray RPC event loop starts. Operator classes get their
        __init__ called while JAX's XLA backend is still uninitialised, which
        allows users to call jax.distributed.initialize (or any other
        per-process setup) safely from __init__ via self.rank / self.world_size.

        The static_args declared for this SPMD node are filtered by each class's
        __init__ signature and forwarded as keyword arguments, so users can
        declare custom init parameters and have them populated from static_args.
        """
        graph_view = config.require_graph_view()
        world_size = graph_view.world_size

        if not isinstance(instance_container, OperatorInstanceContainer):
            raise TypeError("operator prewarm requires an OperatorInstanceContainer")
        container = instance_container
        seen: set = set()
        with operator_context(
            static_args=static_args,
            rank=rank,
            world_size=world_size,
            spmd_id=config.spmd_id,
            local_rank=local_rank,
            local_world_size=local_world_size,
            node_rank=0 if node_rank is None else node_rank,
            num_nodes=1 if num_nodes is None else num_nodes,
            rank0_address=rank0_address,
            instance_container=container,
            ray_component_limits=config.require_graph_view().resource_profile.ray_component_limits,
        ):
            for op_id in graph_view.operator_ids:
                try:
                    op_meta = config.operator_specs[op_id]
                    compiled_op = graph_view.operator(op_id)
                except KeyError as exc:
                    raise RuntimeError(
                        f"SPMDConfig {config.spmd_id!r} is missing active operator "
                        f"{op_id!r}."
                    ) from exc
                cls = op_meta.owner
                if cls is None:
                    continue
                _validate_worker_process_marker(cls)
                if id(cls) in seen:
                    continue
                seen.add(id(cls))
                if container.get(cls) is not None:
                    continue

                execution_group = compiled_op.execution_layout.group_for_rank(rank)
                logical_partition_id = execution_group.logical_rank(rank)
                resolved_static_args = _resolve_logical_partition_bindings(
                    static_args,
                    logical_partition_id,
                    execution_group.logical_mesh,
                )
                init_kwargs = operator_initialization_arguments(
                    cls, resolved_static_args
                )

                instance = object.__new__(cls)
                _inject_context_properties(cls)
                ctx = get_current_context()
                for _attr in _STATIC_CONTEXT_ATTR_NAMES:
                    setattr(instance, _attr, getattr(ctx, _attr))
                logger.debug(
                    "operator_init_begin owner_class=%s rank=%s",
                    cls.__qualname__,
                    rank,
                )
                try:
                    instance.__init__(**init_kwargs)
                except Exception:
                    logger.exception(
                        "operator_init_failed owner_class=%s rank=%s world_size=%s "
                        "local_rank=%s local_world_size=%s node_rank=%s num_nodes=%s",
                        cls.__qualname__,
                        rank,
                        world_size,
                        local_rank,
                        local_world_size,
                        0 if node_rank is None else node_rank,
                        1 if num_nodes is None else num_nodes,
                    )
                    raise
                logger.debug(
                    "operator_init_complete owner_class=%s rank=%s",
                    cls.__qualname__,
                    rank,
                )
                container.get_or_create(cls, lambda instance=instance: instance)

    # =========================================================================
    # Driver Handle Management
    # =========================================================================

    def get_node_id(self) -> str:
        """Return the current Ray node identifier for topology-aware startup."""
        return ray.get_runtime_context().get_node_id()

    def get_node_ip(self) -> str:
        """Return the current Ray node IP address for distributed rendezvous."""
        if "POD_IP" in os.environ:
            return os.environ["POD_IP"]

        try:
            # Connect to Ray's GCS server to determine the cluster-routable IP
            gcs_address = ray.get_runtime_context().gcs_address
            if gcs_address:
                head_ip, head_port = gcs_address.split(":")
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    # Doesn't actually connect, just uses OS routing table
                    s.connect((head_ip, int(head_port)))
                    return s.getsockname()[0]
        except Exception:
            pass

        return ray.util.get_node_ip_address()

    def find_free_port(self, host: str | None = None) -> int:
        """Find an available port on the current node."""
        target_host = host if host is not None else self.get_node_ip()
        addr_info = socket.getaddrinfo(target_host, 0, type=socket.SOCK_STREAM)
        if not addr_info:
            raise OSError(f"Could not resolve address for host {target_host!r}")
        last_err: OSError | None = None
        for family, socktype, proto, _, sockaddr in addr_info:
            try:
                with socket.socket(family, socktype, proto) as s:
                    s.bind(sockaddr)
                    return s.getsockname()[1]
            except OSError as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        raise OSError(f"Failed to bind any resolved address for host {target_host!r}")

    def _max_operator_pipeline_depth(self) -> int:
        max_depth = 1
        for op_meta in self._operator_specs.values():
            depth = op_meta.runtime_policy.pipeline_depth
            if depth <= 0:
                raise ValueError(
                    f"local_pipeline_depth must be positive for op={op_meta.name}, "
                    f"got {depth}."
                )
            max_depth = max(max_depth, depth)
        return max_depth

    def _worker_process_call_max_workers(self) -> int:
        return self._max_operator_pipeline_depth()

    def _input_prepare_max_workers(self) -> int:
        return self._max_operator_pipeline_depth()

    async def acquire_source_operator_work(
        self, operator_name: str, step: int
    ) -> tuple[SourceWorkLease, ...]:
        self._record_execution_phase(
            "source_scope.wait", op_name=operator_name, step=step
        )
        leases = await self._source_scope.take(operator_name, step)
        self._record_execution_phase(
            "source_scope.take", op_name=operator_name, step=step
        )
        return leases

    async def _acquire_source_work(
        self, epoch: int, segments: tuple[CompiledSegment, ...]
    ) -> tuple[SourceWorkLease, ...]:
        if self._control_plane_handle is None:
            raise RuntimeError(
                "source work acquisition requires a control-plane handle"
            )
        return await ControlPlaneClient(self._control_plane_handle).acquire_source_work(
            epoch, segments
        )

    def setup_and_start(
        self,
        local_rank: int,
        local_world_size: int,
        node_rank: int,
        num_nodes: int,
        rank0_address: str | None = None,
    ) -> None:
        """Finish topology-aware startup after the driver has all worker node IDs."""
        logger.debug(
            "setup_and_start_begin spmd_id=%s rank=%s local_rank=%s "
            "local_world_size=%s node_rank=%s num_nodes=%s",
            self.spmd_id,
            self.rank,
            local_rank,
            local_world_size,
            node_rank,
            num_nodes,
        )
        try:
            self.local_rank = local_rank
            self.local_world_size = local_world_size
            self.node_rank = node_rank
            self.num_nodes = num_nodes
            self.rank0_address = rank0_address
            static_args = self.config.static_args

            preinstantiate_kwargs = {
                "static_args": static_args,
                "instance_container": self._instance_container,
                "local_rank": local_rank,
                "local_world_size": local_world_size,
                "node_rank": node_rank,
                "num_nodes": num_nodes,
                "rank0_address": rank0_address,
            }
            resource_registry_token = set_active_resource_registry(
                self._resource_registry
            )
            try:
                self._pre_instantiate_operators(
                    self.config,
                    self.rank,
                    **preinstantiate_kwargs,
                )
            finally:
                reset_active_resource_registry(resource_registry_token)
            logger.debug(
                "setup_and_start_preinstantiate_complete spmd_id=%s rank=%s",
                self.spmd_id,
                self.rank,
            )
            manager_kwargs = dict(
                static_args=static_args,
                instance_container=self._instance_container,
                local_rank=local_rank,
                local_world_size=local_world_size,
                node_rank=node_rank,
                num_nodes=num_nodes,
                worker_process_runtime=self._worker_process_call_runtime,
            )
            if self._control_plane_handle is not None:
                manager_kwargs["control_plane_handle"] = self._control_plane_handle
            self._op_manager = OperatorManager(
                self.config,
                self.rank,
                self.spmd_id,
                **manager_kwargs,
            )
            self._local_step_plan_cache = {
                op_name: self._compile_operator_local_plan(op_name)
                for op_name in self._graph_view.operator_ids
            }
            logger.debug(
                "setup_and_start_operator_manager_complete spmd_id=%s rank=%s",
                self.spmd_id,
                self.rank,
            )
            self._source_pause_events = {
                op_name: asyncio.Event()
                for op_name in self._operator_specs
                if self.op_manager.is_source_operator(op_name)
            }
            for event in self._source_pause_events.values():
                event.set()

            if self._driver_handle is not None:
                self._start_heartbeat()
            self._start_run_loop()
            logger.info(
                "setup_and_start_completed spmd_id=%s rank=%s",
                self.spmd_id,
                self.rank,
            )
            logger.debug(
                "setup_and_start_runtime_started spmd_id=%s rank=%s",
                self.spmd_id,
                self.rank,
            )
        except Exception:
            logger.exception(
                "setup_and_start_failed spmd_id=%s rank=%s local_rank=%s "
                "local_world_size=%s node_rank=%s num_nodes=%s",
                self.spmd_id,
                self.rank,
                local_rank,
                local_world_size,
                node_rank,
                num_nodes,
            )
            raise

    def set_driver_handle(self, driver_handle: Any | None) -> None:
        """
        Set the driver handle for callbacks.

        Args:
            driver_handle: Reference to the SPMD driver
        """
        self._driver_handle = driver_handle

    def get_driver_handle(self) -> Any | None:
        """Get the driver handle."""
        return self._driver_handle

    def register_spmd_code_map(
        self, local_spmd_code: int, spmd_id_by_code: dict[int, str]
    ) -> None:
        if not isinstance(local_spmd_code, int) or local_spmd_code < 0:
            raise ValueError(
                f"local_spmd_code must be a non-negative int, got {local_spmd_code!r}."
            )
        if local_spmd_code not in spmd_id_by_code:
            raise ValueError(
                f"local_spmd_code={local_spmd_code} is missing from spmd_id_by_code."
            )
        self._local_spmd_code = local_spmd_code
        self._spmd_id_by_code = dict(spmd_id_by_code)
        self._operator_send_context_cache.clear()

    def _require_local_spmd_code(self) -> int:
        if self._local_spmd_code is None:
            raise ValueError(
                f"[SPMDWorker] local_spmd_code is not registered "
                f"spmd_id={self.spmd_id} rank={self.rank}."
            )
        return self._local_spmd_code

    def _coherence_subject(
        self,
        owner_spmd_code: int,
        owner_rank: int,
        transfer_code: int,
    ) -> CoherenceSubject:
        transfer = self._graph_view.transfer(transfer_code)
        return CoherenceSubject(
            owner_spmd_code=owner_spmd_code,
            owner_rank=owner_rank,
            logical_tuuid=transfer.logical_tuuid,
        )

    def _coherence_send_plan(self, transfer_code: int) -> CompiledSendPlan:
        transfer = self._graph_view.transfer(transfer_code)
        dependency = self._graph_view.dependency(transfer.dependency_code)
        plan = self._operator_send_context(
            dependency.source_operator.key.op_id
        ).compiled_plan
        if plan.coherence_logical_tuuid(transfer_code) != transfer.logical_tuuid:
            raise CoherenceStateTransitionError(
                "compiled send plan coherence identity differs from graph logical TUUID."
            )
        return plan

    def _coherence_child_slot(self, transfer_code: int, target_rank: int) -> int:
        return self._coherence_send_plan(transfer_code).coherence_child_slot(
            transfer_code,
            target_rank,
        )

    # =========================================================================
    # Handler Registration (handlers only, topology from config)
    # =========================================================================

    def register_predecessor_handlers(
        self, spmd_id: str, handlers: dict[int, DataPlaneCallable]
    ) -> None:
        """
        Register handlers for a predecessor SPMD.

        Topology information is already compiled into the SPMDConfig.
        Only handlers need runtime registration.

        Args:
            spmd_id: Predecessor SPMD ID
            handlers: {rank: DataPlaneCallable}
        """
        self._predecessor_handlers[spmd_id] = (
            self._profiler_runtime.wrap_data_plane_handlers(spmd_id, handlers)
        )
        logger.debug(
            f"Registered predecessor handlers predecessor_spmd_id={spmd_id} "
            f"ranks={list(handlers.keys())} spmd_id={self.spmd_id} rank={self.rank}"
        )

        # Update topology readiness from registered handlers
        self._check_topology_ready()

    def register_successor_handlers(
        self, spmd_id: str, handlers: dict[int, DataPlaneCallable]
    ) -> None:
        """
        Register handlers for a successor SPMD.

        Topology information is already compiled into the SPMDConfig.
        Only handlers need runtime registration.

        Args:
            spmd_id: Successor SPMD ID
            handlers: {rank: DataPlaneCallable}
        """
        self._successor_handlers[spmd_id] = (
            self._profiler_runtime.wrap_data_plane_handlers(spmd_id, handlers)
        )
        self._operator_send_context_cache.clear()
        logger.debug(
            f"Registered successor handlers successor_spmd_id={spmd_id} "
            f"ranks={list(handlers.keys())} spmd_id={self.spmd_id} rank={self.rank}"
        )

        # Update topology readiness from registered handlers
        self._check_topology_ready()

    def set_expected_topology(
        self, predecessor_count: int, successor_count: int
    ) -> None:
        """
        Set expected predecessor/successor counts for topology readiness.

        Called by driver before topology registration begins.

        Args:
            predecessor_count: Expected number of predecessor SPMDs
            successor_count: Expected number of successor SPMDs
        """
        self._expected_predecessor_count = predecessor_count
        self._expected_successor_count = successor_count
        logger.debug(
            f"Topology expectation set predecessor_count={predecessor_count} "
            f"successor_count={successor_count} spmd_id={self.spmd_id} rank={self.rank}"
        )

        self._check_topology_ready()

    def _check_topology_ready(self) -> None:
        if (
            self._expected_predecessor_count is None
            or self._expected_successor_count is None
        ):
            logger.debug(
                f"Waiting for topology counts spmd_id={self.spmd_id} rank={self.rank}"
            )
            return

        # Check if all expected handlers are registered
        pred_registered = (
            len(self._predecessor_handlers) >= self._expected_predecessor_count
        )
        succ_registered = (
            len(self._successor_handlers) >= self._expected_successor_count
        )

        logger.debug(
            f"Handler registration status "
            f"pred={len(self._predecessor_handlers)}/{self._expected_predecessor_count} "
            f"succ={len(self._successor_handlers)}/{self._expected_successor_count} "
            f"spmd_id={self.spmd_id} rank={self.rank}"
        )

        if pred_registered and succ_registered:
            logger.debug(
                f"Topology handlers ready "
                f"pred={len(self._predecessor_handlers)}/{self._expected_predecessor_count} "
                f"succ={len(self._successor_handlers)}/{self._expected_successor_count} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            self._execution_lifecycle.topology_ready()

    def activate_execution(self) -> None:
        if not self._execution_lifecycle.activate():
            return

        self._execution_generation = getattr(self, "_execution_generation", 0) + 1
        self._record_execution_phase("execution.activate")

        if self.config.auto_start_source_operators:
            self._spawn_background_task(self._trigger_source_operators())
        for op_name, steps in self._mailbox_ready_steps.items():
            if steps:
                self._spawn_background_task(self._schedule_mailbox_ready_steps(op_name))

    def _spawn_background_task(self, coro: Any) -> asyncio.Task:
        task = self._create_supervised_task(coro)
        self._background_tasks.add(task)
        return task

    def _create_supervised_task(self, coro: Any) -> asyncio.Task:
        if self._profiler_runtime.enabled:
            recorder = self._profiler_runtime.recorder
            original_coro = coro
            task_name = getattr(
                getattr(original_coro, "cr_code", None),
                "co_name",
                type(original_coro).__name__,
            )

            async def traced_coro() -> Any:
                async with recorder.worker_scheduler_span(
                    spec=WORKER_BACKGROUND_TASK,
                    payload=(recorder.intern(task_name),),
                ):
                    return await original_coro

            coro = traced_coro()
        task = asyncio.create_task(coro)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        self._record_failure(exc, phase="BACKGROUND")

    def _record_failure(
        self,
        exception: BaseException,
        *,
        phase: str,
        op_name: str | None = None,
        step: int | None = None,
    ) -> ExecutionFailure:
        if self._failure is not None:
            return self._failure

        failure = ExecutionFailure.from_exception(
            spmd_id=self.spmd_id,
            rank=self.rank,
            op_name=op_name,
            step=step,
            phase=phase,
            exception=exception,
        )
        self._failure = failure
        self._execution_lifecycle.fail()
        self._fence_state = ExecutionFenceState.CLOSING
        asyncio.create_task(self._source_scope.fail_all(exception))

        for execution_queue in self._execution_queues.values():
            execution_queue.put_nowait(_EXECUTION_STOP_ITEM)
        for tasks in (self._plan_tasks, self._send_tasks, self._background_tasks):
            for task in tuple(tasks):
                if not task.done():
                    task.cancel()

        if self._driver_handle is not None:
            self._driver_handle.receive_worker_failure.remote(self.rank, failure)

        logger.error(
            "Worker execution failed spmd_id=%s rank=%s op=%s step=%s phase=%s error=%s",
            self.spmd_id,
            self.rank,
            op_name,
            step,
            phase,
            exception,
            exc_info=(type(exception), exception, exception.__traceback__),
        )
        return failure

    async def _drain_outbound_tasks(self) -> None:
        while True:
            send_tasks = [task for task in self._send_tasks if not task.done()]
            if send_tasks:
                await asyncio.gather(*send_tasks)
                continue
            break
        await self._delivery_runtime.drain("worker_drain_outbound")
        await self._source_acknowledgements.flush()
        if self._failure is not None:
            raise RuntimeError(f"Worker execution failed: {self._failure}")

    async def _trigger_source_operators(self) -> None:
        """
        Trigger initial steps for source operators (those with 0 input tensors).
        """
        source_ops = self.op_manager.get_source_operators()
        logger.debug(
            f"Source operators identified ops={source_ops} spmd_id={self.spmd_id} rank={self.rank}"
        )
        for op_name in source_ops:
            await self._source_maintenance.request(op_name)

    async def start_source_operators(self) -> None:
        await self._trigger_source_operators()

    # =========================================================================
    # Op-Level Blocking
    # =========================================================================

    async def set_op_runnable(
        self, op_name: str, runnable: bool, op_runnable_epoch: int
    ) -> None:
        latest_epoch = self._latest_op_runnable_epoch[op_name]
        if op_runnable_epoch < latest_epoch:
            logger.debug(
                f"Stale op runnable control ignored op={op_name} runnable={runnable} "
                f"epoch={op_runnable_epoch} latest_epoch={latest_epoch} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            return
        self._latest_op_runnable_epoch[op_name] = op_runnable_epoch
        if runnable:
            self._op_runnable[op_name].set()
            pending = self._pending_blocked[op_name]
            self._pending_blocked[op_name] = []
            for step in pending:
                await self._enqueue(op_name, step)
        else:
            self._op_runnable[op_name].clear()
        logger.debug(
            f"Op runnable updated op={op_name} runnable={runnable} "
            f"epoch={op_runnable_epoch} spmd_id={self.spmd_id} rank={self.rank}"
        )

    def is_op_blocked(self, op_name: str) -> bool:
        """Check if an operator is blocked."""
        return not self._op_runnable[op_name].is_set()

    # =========================================================================
    # Data Reception
    # =========================================================================

    async def receive_control(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> None:
        """
        Receive state data from Driver (control plane).

        Args:
            op_name: Operator name
            step: Step number
            names: State variable names
            values: State data values
        """
        # Normal mode: step-ordered execution
        completed_step_count = self._completed_step_count.get(op_name, 0)

        # Drop stale data (data for steps already executed)
        # Note: periodic ops don't have stale steps as they use virtual sequence
        is_periodic = (
            self.op_manager.is_periodic_trigger(op_name) if self._op_manager else False
        )
        if not is_periodic and step < completed_step_count:
            logger.warning(
                f"Dropped stale state data op={op_name} names={names} "
                f"step={step} completed_step_count={completed_step_count} spmd_id={self.spmd_id} rank={self.rank}"
            )
            return

        # Lazily declare the step barrier on first data arrival
        self._ensure_step_declared(op_name, step)

        logger.debug(
            f"Received states op={op_name} step={step} names={names} "
            f"spmd_id={self.spmd_id} rank={self.rank}"
        )

        # Cache in mailbox — barrier fires on_ready automatically
        await self._mailbox.deposit_control_arguments(
            op_name, step=step, names=names, values=values
        )

    async def dispatch_control(self, request: Any) -> object:
        from fluxway.common.driver_control import IngestionLayoutActivationRequest
        from fluxway.common.ingestion import (
            IngestionLayoutActivation,
            IngestionLayoutReceiver,
            ingestion_worker_id,
        )

        if not isinstance(request, IngestionLayoutActivationRequest):
            raise TypeError("worker control requires a registered request")
        op_name = request.operator_name
        activation = request.activation
        if not isinstance(activation, IngestionLayoutActivation):
            raise TypeError("worker ingestion activation requires layout metadata")
        spec = self.op_manager.get_operator_spec(op_name)
        if spec.is_source is not True:
            raise ValueError("ingestion activation target must be a source operator")
        expected_worker_id = ingestion_worker_id(self.spmd_id, op_name, self.rank)
        if activation.descriptor.worker_id != expected_worker_id:
            raise ValueError(
                "ingestion activation worker identity does not match target rank"
            )
        owner = self.op_manager.require_operator_instance(op_name)
        if not isinstance(owner, IngestionLayoutReceiver):
            raise TypeError(
                f"source operator {op_name!r} does not implement "
                "activate_ingestion_layout"
            )
        self._execution_epoch = activation.epoch
        self._record_execution_phase(
            "activation.dispatch.begin",
            op_name=op_name,
            step=activation.start_position,
        )
        await self._source_scope.activate(
            op_name, activation, owner.activate_ingestion_layout
        )
        self._record_execution_phase(
            "activation.install.end",
            op_name=op_name,
            step=activation.end_position - 1,
        )
        return None

    async def receive_transport_frames(
        self,
        frames: tuple[TransportFrameEnvelope, ...],
    ) -> tuple[int, ...]:
        if not frames:
            raise ValueError("receive_transport_frames requires at least one frame.")
        if self._fence_state is not ExecutionFenceState.OPEN:
            raise CoherenceStateTransitionError("worker frame admission is fenced.")
        recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        watermarks = []
        if recorder is None:
            for frame in frames:
                watermarks.append(await self._receive_transport_frame_envelope(frame))
        else:
            async with recorder.data_plane_receiver_batch(frame_count=len(frames)):
                for frame in frames:
                    watermarks.append(
                        await self._receive_transport_frame_envelope(frame)
                    )
        return tuple(watermarks)

    async def _receive_transport_frame_envelope(
        self,
        envelope: TransportFrameEnvelope,
    ) -> int:
        header = envelope.header
        payload_iovs = envelope.payload_segments
        if not isinstance(header, TransportFrameHeader):
            raise TypeError(
                f"header must be TransportFrameHeader, got {type(header).__name__}."
            )
        frame = header.to_frame(payload_iovs)
        sender_spmd_code = int(frame.frame_i32[FRAME_SENDER_SPMD_CODE])
        step = int(frame.frame_i32[FRAME_STEP])
        entry_count = int(frame.frame_i32[FRAME_ENTRY_COUNT])
        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        try:
            sender_id = self._spmd_id_by_code[sender_spmd_code]
        except KeyError as exc:
            raise ValueError(
                f"[SPMDWorker] Unknown sender_spmd_code={sender_spmd_code} "
                f"spmd_id={self.spmd_id} rank={self.rank}."
            ) from exc

        frame_flag = FrameFlag(int(frame.frame_i32[FRAME_FLAGS]))
        if frame_flag is FrameFlag.ORDERING_COMPLETION:
            return await self._receive_ordering_frame(
                sender_id=sender_id,
                frame=frame,
                step=step,
            )
        if frame_flag is FrameFlag.DEPENDENCY_DECISION:
            return await self._receive_dependency_decision_frame(
                sender_id=sender_id,
                frame=frame,
                step=step,
            )
        if frame_flag is FrameFlag.COHERENCE:
            return await self._receive_coherence_frame(
                sender_spmd_code=sender_spmd_code,
                frame=frame,
            )

        watermark = 0
        op_to_entry_indices: dict[str, list[int]] = {}
        for entry_index, entry in enumerate(frame.entry_i32):
            transfer_code = int(entry[ENTRY_ROUTE_CODE])
            state_flags = int(entry[ENTRY_STATE_FLAGS])
            action = entry_coherence_action(state_flags)
            tensor_kind = TensorEntryKind(entry_tensor_kind(state_flags))
            sender_rank = int(frame.frame_i32[FRAME_SENDER_RANK])
            epoch = int(entry[ENTRY_AUX])
            op_name = self.op_manager.get_transfer_target_op(transfer_code)
            if (
                tensor_kind in (TensorEntryKind.TENSOR, TensorEntryKind.ROUTING_MISS)
                and action is CoherenceAction.BUS_RD
            ):
                subject = self._coherence_subject(
                    sender_spmd_code,
                    sender_rank,
                    transfer_code,
                )
                grants = self._coherence_input_grants.setdefault((op_name, step), {})
                identity = (
                    sender_id,
                    sender_rank,
                    subject,
                    transfer_code,
                    epoch,
                )
                if identity not in grants:
                    self._coherence_directory.accept_parent_read(subject, epoch)
                    local_read = self._coherence_directory.request_read(
                        subject,
                        epoch,
                        ExecutionToken(step),
                    )
                    if not isinstance(local_read, GrantReady):
                        raise CoherenceStateTransitionError(
                            "BUS_RD did not produce an immediate local read grant."
                        )
                    grant = _InputCoherenceGrant(
                        parent_spmd_id=sender_id,
                        parent_rank=sender_rank,
                        subject=subject,
                        route_code=transfer_code,
                        epoch=epoch,
                        local_access_token=local_read.token,
                        parent_return_required=(tensor_kind is TensorEntryKind.TENSOR),
                    )
                    grants[grant.identity] = grant
            op_to_entry_indices.setdefault(op_name, []).append(entry_index)

        async def deposit_frame_entries() -> int:
            deposited_watermark = 0
            for op_name, entry_indices in op_to_entry_indices.items():
                self._ensure_step_declared(op_name, step)
                op_frame = TransportFrame(
                    frame_i32=frame.frame_i32.copy(),
                    entry_i32=frame.entry_i32[entry_indices],
                    payload_i32=frame.payload_i32,
                    shape_i32=frame.shape_i32,
                    global_shape_i64=frame.global_shape_i64,
                    compact_i64=frame.compact_i64,
                    payload_segments=frame.payload_segments,
                )
                op_frame.frame_i32[FRAME_ENTRY_COUNT] = len(entry_indices)
                deposited_watermark = await self._mailbox.deposit_transport_frame(
                    op_name,
                    sender_id,
                    op_frame,
                )
                if profiler_recorder is not None:
                    profiler_recorder.worker_deposit_transport_frame(
                        op_name=op_name,
                        step=step,
                        entry_count=len(entry_indices),
                        watermark=deposited_watermark,
                    )
            return deposited_watermark

        if profiler_recorder is None:
            watermark = await deposit_frame_entries()
        else:
            async with profiler_recorder.worker_receive_transport_frame(
                step=step,
                sender_spmd_code=sender_spmd_code,
                entry_count=entry_count,
                payload_count=len(payload_iovs),
            ):
                async with profiler_recorder.data_plane_receiver_deposit(
                    step=step,
                    sender_spmd_code=sender_spmd_code,
                    entry_count=entry_count,
                    payload_count=len(payload_iovs),
                ):
                    watermark = await deposit_frame_entries()

        return watermark

    async def _receive_coherence_frame(
        self,
        *,
        sender_spmd_code: int,
        frame: TransportFrame,
    ) -> int:
        sender_rank = int(frame.frame_i32[FRAME_SENDER_RANK])
        for entry in frame.entry_i32:
            route_code = int(entry[ENTRY_ROUTE_CODE])
            epoch = int(entry[ENTRY_AUX])
            action = entry_coherence_action(int(entry[ENTRY_STATE_FLAGS]))
            incoming_subject = self._coherence_subject(
                sender_spmd_code,
                sender_rank,
                route_code,
            )
            if action is CoherenceAction.BUS_RD:
                self._coherence_directory.accept_parent_read(
                    incoming_subject,
                    epoch,
                )
                continue
            if action is CoherenceAction.SNOOP_INVALIDATE:
                await self._invalidate_parent_grant(
                    incoming_subject,
                    route_code,
                    epoch,
                    sender_rank,
                )
                continue
            local_code = self._require_local_spmd_code()
            local_subject = self._coherence_subject(
                local_code,
                self.rank,
                route_code,
            )
            if action is CoherenceAction.GRANT_RETURN:
                child_slot = self._coherence_child_slot(route_code, sender_rank)
                entry_state = self._coherence_directory.admit(
                    CoherenceInstance(local_subject, epoch)
                )
                if entry_state.ownership_child_id == child_slot:
                    self._coherence_directory.return_ownership_child(
                        local_subject,
                        epoch,
                        child_id=child_slot,
                        data_valid=False,
                    )
                else:
                    self._coherence_directory.return_child(
                        local_subject,
                        epoch,
                        child_id=child_slot,
                    )
                continue
            if action is CoherenceAction.OWNERSHIP_CANCEL:
                child_slot = self._coherence_child_slot(route_code, sender_rank)
                self._coherence_directory.return_ownership_child(
                    local_subject,
                    epoch,
                    child_id=child_slot,
                    data_valid=True,
                )
                continue
            if action is CoherenceAction.RFO:
                await self._grant_child_ownership(
                    subject=local_subject,
                    epoch=epoch,
                    requester_spmd_code=sender_spmd_code,
                    requester_rank=sender_rank,
                    route_code=route_code,
                )
                continue
            if action is CoherenceAction.OWNERSHIP_GRANT:
                self._coherence_directory.accept_parent_ownership(
                    incoming_subject,
                    epoch,
                )
                continue
            raise CoherenceStateTransitionError(
                f"unsupported coherence action={action.name}."
            )
        return 0

    async def _invalidate_parent_grant(
        self,
        subject: CoherenceSubject,
        route_code: int,
        epoch: int,
        sender_rank: int,
    ) -> None:
        revocation = CoherenceInstance(subject, epoch)
        if revocation in self._coherence_revocations:
            raise CoherenceStateTransitionError(
                f"duplicate invalidation for subject={subject} epoch={epoch}."
            )
        self._coherence_revocations.add(revocation)
        try:
            invalidation = self._coherence_directory.begin_invalidation(subject, epoch)
            if invalidation is InvalidationResult.SATISFIED_BY_PARENT_RETURN:
                return
            entry = self._coherence_directory.admit(CoherenceInstance(subject, epoch))
            await self._coherence_admission_runtime.await_deliveries_complete(
                CoherenceInstance(subject, epoch), entry.child_share_bitmap
            )
            if entry.child_share_bitmap:
                plan = self._coherence_send_plan(route_code)
                await asyncio.gather(
                    *(
                        self._send_coherence_to_child(
                            child.transfer_code,
                            child.target_rank,
                            epoch,
                            CoherenceAction.SNOOP_INVALIDATE,
                        )
                        for child_slot in self._bitmap_indices(entry.child_share_bitmap)
                        for child in (
                            plan.coherence_child(
                                subject.logical_tuuid,
                                child_slot,
                            ),
                        )
                    )
                )
            await self._coherence_admission_runtime.await_quiescent(
                CoherenceInstance(subject, epoch)
            )
            self._coherence_directory.finish_invalidation(subject, epoch)
            await self._parent_grant_return_runtime.return_parent_grants(
                (
                    ParentReturnRoute(
                        subject=subject,
                        route_code=route_code,
                        epoch=epoch,
                        parent_spmd_id=self._spmd_id_by_code[subject.owner_spmd_code],
                        parent_rank=subject.owner_rank,
                    ),
                )
            )
        finally:
            self._coherence_revocations.remove(revocation)

    async def _grant_child_ownership(
        self,
        *,
        subject: CoherenceSubject,
        epoch: int,
        requester_spmd_code: int,
        requester_rank: int,
        route_code: int,
    ) -> None:
        plan = self._coherence_send_plan(route_code)
        requester_slot = plan.coherence_child_slot(route_code, requester_rank)
        entry = self._coherence_directory.admit(CoherenceInstance(subject, epoch))
        grant = self._coherence_directory.request_write(
            subject,
            epoch,
            ExecutionToken(requester_slot),
        )
        await self._coherence_admission_runtime.await_deliveries_complete(
            CoherenceInstance(subject, epoch), entry.child_share_bitmap
        )
        entry = self._coherence_directory.admit(CoherenceInstance(subject, epoch))
        if entry.child_share_bitmap & (1 << requester_slot):
            self._coherence_directory.return_child(
                subject,
                epoch,
                requester_slot,
            )
        sibling_bitmap = entry.child_share_bitmap & ~(1 << requester_slot)
        await asyncio.gather(
            *(
                self._send_coherence_to_child(
                    child.transfer_code,
                    child.target_rank,
                    epoch,
                    CoherenceAction.SNOOP_INVALIDATE,
                )
                for child_slot in self._bitmap_indices(sibling_bitmap)
                for child in (plan.coherence_child(subject.logical_tuuid, child_slot),)
            )
        )
        if isinstance(grant, GrantPending):
            token = await self._coherence_admission_runtime.await_grant(grant)
        elif isinstance(grant, GrantReady):
            token = grant.token
        elif isinstance(grant, GrantRejected):
            raise grant.error
        else:
            raise TypeError(f"unsupported grant result={type(grant).__name__}.")
        self._coherence_directory.delegate_ownership(token, requester_slot)
        await self._send_coherence_to_child(
            route_code,
            requester_rank,
            epoch,
            CoherenceAction.OWNERSHIP_GRANT,
            target_spmd_code=requester_spmd_code,
        )

    async def _send_coherence_to_child(
        self,
        route_code: int,
        child_rank: int,
        epoch: int,
        action: CoherenceAction,
        *,
        target_spmd_code: int | None = None,
    ) -> None:
        transfer = self._graph_view.transfer(route_code)
        dependency = self._graph_view.dependency(transfer.dependency_code)
        target_spmd_id = dependency.spec.target.spmd_id
        if target_spmd_code is not None:
            known_target = self._spmd_id_by_code.get(target_spmd_code)
            if known_target != target_spmd_id:
                raise CoherenceStateTransitionError(
                    f"coherence target code={target_spmd_code} resolves to "
                    f"{known_target!r}, expected {target_spmd_id!r}."
                )
        handlers = self._successor_handlers.get(target_spmd_id)
        if handlers is None or child_rank not in handlers:
            raise ValueError(
                f"missing direct child handler target={target_spmd_id}:{child_rank}."
            )
        await self._send_coherence_frame(
            handlers[child_rank], (route_code,), epoch, action
        )

    async def _send_coherence_to_parent(
        self,
        subject: CoherenceSubject,
        route_code: int,
        epoch: int,
        action: CoherenceAction,
    ) -> None:
        sender_id = self._spmd_id_by_code[subject.owner_spmd_code]
        handlers = self._predecessor_handlers.get(sender_id)
        if handlers is None or subject.owner_rank not in handlers:
            raise ValueError(
                "missing direct parent handler "
                f"sender={sender_id}:{subject.owner_rank}."
            )
        await self._send_coherence_frame(
            handlers[subject.owner_rank],
            (route_code,),
            epoch,
            action,
        )

    async def _send_parent_return_group(
        self,
        routes: tuple[ParentReturnRoute, ...],
    ) -> None:
        if not routes:
            raise ValueError("parent return group cannot be empty.")
        canonical = routes[0]
        if any(
            route.parent_spmd_id != canonical.parent_spmd_id
            or route.parent_rank != canonical.parent_rank
            or route.epoch != canonical.epoch
            for route in routes
        ):
            raise ValueError("parent return group contains incompatible routes.")
        handlers = self._predecessor_handlers.get(canonical.parent_spmd_id)
        if handlers is None or canonical.parent_rank not in handlers:
            raise ValueError(
                "missing direct parent handler "
                f"sender={canonical.parent_spmd_id}:{canonical.parent_rank}."
            )
        await self._send_coherence_frame(
            handlers[canonical.parent_rank],
            tuple(route.route_code for route in routes),
            canonical.epoch,
            CoherenceAction.GRANT_RETURN,
        )

    async def _send_coherence_batch_to_parent(
        self,
        subject: CoherenceSubject,
        route_codes: tuple[int, ...],
        epoch: int,
        action: CoherenceAction,
    ) -> None:
        sender_id = self._spmd_id_by_code[subject.owner_spmd_code]
        handlers = self._predecessor_handlers.get(sender_id)
        if handlers is None or subject.owner_rank not in handlers:
            raise ValueError(
                "missing direct parent handler "
                f"sender={sender_id}:{subject.owner_rank}."
            )
        await self._send_coherence_frame(
            handlers[subject.owner_rank],
            route_codes,
            epoch,
            action,
        )

    async def _send_coherence_frame(
        self,
        handler: DataPlaneCallable,
        route_codes: tuple[int, ...],
        epoch: int,
        action: CoherenceAction,
    ) -> None:
        header = build_coherence_frame(
            step=epoch,
            sender_spmd_code=self._require_local_spmd_code(),
            sender_rank=self.rank,
            sender_world_size=self._world_size,
            epoch=epoch,
            action=action,
            route_codes=route_codes,
        )
        await handler.receive_transport_frames(
            (TransportFrameEnvelope(header=header, payload_segments=()),)
        )

    @staticmethod
    def _bitmap_indices(bitmap: int):
        while bitmap:
            least_bit = bitmap & -bitmap
            yield least_bit.bit_length() - 1
            bitmap ^= least_bit

    async def _receive_ordering_frame(
        self,
        *,
        sender_id: str,
        frame: TransportFrame,
        step: int,
    ) -> int:
        watermark = 0
        for entry in frame.entry_i32:
            dependency_code = int(entry[ENTRY_ROUTE_CODE])
            expected_count = int(entry[ENTRY_EXPECTED_COUNT])
            if expected_count <= 0:
                raise ValueError(
                    f"[SPMDWorker] Ordering dependency code={dependency_code} "
                    f"has invalid expected_count={expected_count}."
                )
            target_op = self.op_manager.get_ordering_target_op(dependency_code)
            self._ensure_step_declared(target_op, step)
            watermark = await self._mailbox.deposit_ordering(
                target_op,
                dependency_code,
                step,
                sender_id=sender_id,
                rank=int(frame.frame_i32[FRAME_SENDER_RANK]),
                expected_count=expected_count,
            )
        return watermark

    async def _receive_dependency_decision_frame(
        self,
        *,
        sender_id: str,
        frame: TransportFrame,
        step: int,
    ) -> int:
        watermark = 0
        sender_rank = int(frame.frame_i32[FRAME_SENDER_RANK])
        for entry in frame.entry_i32:
            dependency_code = int(entry[ENTRY_ROUTE_CODE])
            expected_count = int(entry[ENTRY_EXPECTED_COUNT])
            canonical_count = self.op_manager.get_dependency_decision_expected_count(
                dependency_code
            )
            if expected_count != canonical_count:
                raise ValueError(
                    f"[SPMDWorker] Dependency decision code={dependency_code} "
                    f"expected_count={expected_count} does not match canonical "
                    f"source count={canonical_count}."
                )
            decision = DependencyDecision(int(entry[ENTRY_AUX]))
            target_op = self.op_manager.get_dependency_decision_target_op(
                dependency_code
            )
            self._ensure_step_declared(target_op, step)
            watermark = await self._mailbox.deposit_dependency_decision(
                target_op,
                dependency_code,
                step,
                sender_id=sender_id,
                rank=sender_rank,
                expected_count=expected_count,
                decision=decision,
            )
        return watermark

    def _ensure_step_declared(self, op_name: str, step: int) -> None:
        """
        Lazily register a StepBarrier for (op_name, step) on first data arrival.

        Expected counts are read from OperatorManager so the barrier knows
        exactly how many shards/states to wait for before firing.
        """
        if self._mailbox.is_step_ready(op_name, step):
            return  # Already done (or re-fired from a previous run)
        if self._barriers_declared.get((op_name, step)):
            return  # Already declared

        expected_transfers = self.op_manager.get_expected_transfer_shards(op_name)
        expected_states = self.op_manager.get_expected_control_arg_names(op_name)
        meta = self.op_manager.get_operator_spec(op_name)
        self._mailbox.declare_step(
            op_name,
            step,
            expected_transfers,
            expected_states,
            barrier_config=meta.barrier_config,
            expected_orderings=self.op_manager.get_expected_ordering_counts(op_name),
            transfer_dependencies=(
                self.op_manager.get_transfer_dependency_codes(op_name)
                if expected_transfers
                else {}
            ),
            expected_dependency_decisions=(
                self.op_manager.get_expected_dependency_decisions(op_name)
                if expected_transfers
                else {}
            ),
        )
        self._barriers_declared[(op_name, step)] = True

    def _on_step_ready(self, op_name: str, step: int) -> None:
        """
        Called by WorkerMailbox when a step's StepBarrier fires.

        Records the ready step and wakes the scheduler asynchronously so mailbox
        deposits can return without waiting on queue insertion.
        """
        logger.debug(
            f"Step ready op={op_name} step={step} spmd_id={self.spmd_id} rank={self.rank}"
        )
        if not self._running or self._source_is_exhausted(op_name):
            return
        if not self.op_manager.is_periodic_trigger(op_name):
            if self._scheduler_knows_step(op_name, step):
                return
            self._mailbox_ready_steps[op_name].add(step)
            self._spawn_background_task(self._schedule_mailbox_ready_steps(op_name))
            return
        if self._should_start_prepare_on_ready(op_name, step):
            self._spawn_background_task(
                self._start_prepare_and_enqueue_token(op_name, step)
            )
        else:
            self._spawn_background_task(self._enqueue(op_name, step))

    def _get_operator_mailbox(self, op_name: str) -> OperatorMailbox:
        operator_mailboxes = getattr(self, "_operator_mailboxes", None)
        if operator_mailboxes is None:
            operator_mailboxes = {}
            self._operator_mailboxes = operator_mailboxes
        mailbox = operator_mailboxes.get(op_name)
        if mailbox is not None:
            return mailbox
        slots = None
        ready_steps = None
        if hasattr(self, "_direct_local_input_slots"):
            slots = self._direct_local_input_slots.setdefault(op_name, {})
        if hasattr(self, "_direct_local_ready_steps"):
            ready_steps = self._direct_local_ready_steps.setdefault(op_name, set())
        mailbox = create_operator_mailbox(
            op_name=op_name,
            expected_direct_transfer_codes=self._direct_local_input_transfer_codes(
                op_name
            ),
            on_step_ready=self._on_step_ready,
            worker_mailbox=getattr(self, "_mailbox", None),
            slots=slots,
            ready_steps=ready_steps,
        )
        operator_mailboxes[op_name] = mailbox
        return mailbox

    def _direct_local_input_transfer_codes(self, op_name: str) -> set[int]:
        expected = self.op_manager.get_expected_transfer_shards(op_name)
        if not expected or any(count != 1 for count in expected.values()):
            return set()
        if self.op_manager.get_expected_control_arg_names(op_name):
            return set()
        if self.op_manager.get_expected_ordering_counts(op_name):
            return set()
        if self.op_manager.get_expected_dependency_decisions(op_name):
            return set()

        direct_codes: set[int] = set()
        for transfer_code in expected:
            route = self.op_manager.get_predecessor_route(op_name, transfer_code)
            if (
                not route.local_only
                or route.spmd_id != self.spmd_id
                or route.world_size != self._world_size
            ):
                return set()
            plan = self._compiled_assemble_plan(op_name, transfer_code)
            if plan.assembles_to_dynamic or plan.has_dynamic_predecessor:
                return set()
            direct_codes.add(transfer_code)
        return direct_codes

    def _uses_direct_local_input_fast_path(self, op_name: str) -> bool:
        return self._get_operator_mailbox(op_name).is_direct_fast_path

    def _deposit_direct_local_input(
        self,
        *,
        op_name: str,
        transfer_code: int,
        step: int,
        sender_id: str,
        rank: int,
        tensor: TransportTensor,
    ) -> None:
        mailbox = self._get_operator_mailbox(op_name)
        if not isinstance(mailbox, DirectLocalMailbox):
            raise ValueError(
                f"Direct local deposit invoked for operator without DirectLocalMailbox: op={op_name}"
            )
        plan = self._compiled_assemble_plan(op_name, transfer_code)
        ensure_transport_source_step(
            op_name=op_name,
            tensor_name=(
                f"transfer_code={transfer_code},"
                f"logical_tensor={plan.logical_tensor.name!r},"
                f"logical_tuuid={plan.logical_tensor.logical_tuuid}"
            ),
            tensor=tensor,
            exec_step=step,
        )
        mailbox.deposit(
            transfer_code,
            step,
            sender_id,
            rank,
            tensor,
        )

    def _direct_local_slots_ready(self, op_name: str, step: int) -> bool:
        return self._get_operator_mailbox(op_name).is_ready(step)

    def _clear_direct_local_step(self, op_name: str, step: int) -> None:
        mailbox = self._get_operator_mailbox(op_name)
        if isinstance(mailbox, DirectLocalMailbox):
            mailbox.clear_step_sync(step)
        else:
            self._direct_local_input_slots.get(op_name, {}).pop(step, None)
            self._direct_local_ready_steps.get(op_name, set()).discard(step)

    def _should_start_prepare_on_ready(self, op_name: str, step: int) -> bool:
        if not self._should_prepare_inputs_on_ready(op_name):
            return False
        if self._drain_blocks_step(op_name, step):
            return False
        return True

    async def _start_prepare_and_enqueue_token(self, op_name: str, step: int) -> None:
        self._prepare_inputs_if_policy_allows(op_name, step)
        await self._enqueue(op_name, step)

    def _prepare_inputs_if_policy_allows(self, op_name: str, step: int) -> None:
        if not self._should_prepare_inputs_on_ready(op_name):
            return
        if self._drain_blocks_step(op_name, step):
            return
        self._input_prepare_runtime.start(
            op_name,
            step,
            lambda: self._execute_operator_prepare_input(op_name, step),
        )

    def _on_prepare_ready(self, op_name: str, step: int) -> None:
        if not self._running:
            return
        if (op_name, step) not in self._in_op_step_queue:
            return
        try:
            self._enqueue_prepare_ready_refresh(op_name, step)
        except Exception as error:
            self._record_failure(
                error,
                phase="BACKGROUND",
                op_name=op_name,
                step=step,
            )

    def _enqueue_prepare_ready_refresh(self, op_name: str, step: int) -> None:
        schedule_key = self._effective_schedule_key(op_name, step)
        self._publish_execution_item(op_name, schedule_key)
        if self._profiler_runtime.enabled:
            recorder = self._profiler_runtime.recorder
            if recorder.event_enabled(WORKER_QUEUE_ENQUEUE, step=step):
                recorder.worker_scheduler_instant(
                    spec=WORKER_QUEUE_ENQUEUE,
                    payload=recorder.worker_op_payload(
                        op_name,
                        step,
                        schedule_key[0],
                    ),
                )

    def _should_prepare_inputs_on_ready(self, op_name: str) -> bool:
        meta = self.op_manager.get_operator_spec(op_name)
        return meta.runtime_policy.is_pipelined

    async def _schedule_mailbox_ready_steps(self, op_name: str) -> None:
        if self.op_manager.is_periodic_trigger(op_name):
            return
        ready_steps = self._mailbox_ready_steps[op_name]
        while ready_steps:
            step = min(ready_steps)
            ready_steps.discard(step)
            completed = self._completed_step_count[op_name]
            if step < completed:
                continue
            if self._scheduler_knows_step(op_name, step):
                continue
            self._prepare_inputs_if_policy_allows(op_name, step)
            await self._enqueue(op_name, step)

    def _scheduler_knows_step(self, op_name: str, step: int) -> bool:
        completed = self._completed_step_count[op_name]
        if step < completed:
            return False
        scheduled = self._scheduled_step_count[op_name]
        if step < scheduled:
            return True
        if (op_name, step) in self._in_op_step_queue:
            return True
        if step in self._pending_future_steps[op_name]:
            return True
        if step in self._pending_async_lag_steps[op_name]:
            return True
        if step in self._pending_phase_order_steps[op_name]:
            return True
        if step in self._ready_pending_pipeline_steps[op_name]:
            return True
        if step in self._mailbox_ready_steps[op_name]:
            return True
        if step in self._running_plan_steps[op_name]:
            return True
        if step in self._finished_plan_steps[op_name]:
            return True
        return False

    async def _enqueue(self, op_name: str, step: int) -> None:
        """Enqueue a task for execution."""
        if not self._running:
            logger.debug(
                f"Skipping enqueue after stop op={op_name} step={step} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            return

        key = (op_name, step)
        if key in self._in_op_step_queue:
            logger.debug(
                f"Idempotency guard: already in queue op={op_name} step={step}"
            )
            return
        self._in_op_step_queue.add(key)

        logger.debug(
            f"Enqueue called op={op_name} step={step} spmd_id={self.spmd_id} rank={self.rank}"
        )
        schedule_key = self._effective_schedule_key(op_name, step)
        priority_key = schedule_key[0]
        if self._profiler_runtime.enabled:
            recorder = self._profiler_runtime.recorder
            if recorder.event_enabled(WORKER_QUEUE_ENQUEUE, step=step):
                async with recorder.worker_scheduler_span(
                    spec=WORKER_QUEUE_ENQUEUE,
                    payload=recorder.worker_op_payload(op_name, step, priority_key),
                ):
                    await self._put_execution_item(op_name, schedule_key)
            else:
                await self._put_execution_item(op_name, schedule_key)
        else:
            await self._put_execution_item(op_name, schedule_key)
        logger.debug(
            f"Task enqueued op={op_name} step={step} schedule_key={schedule_key} spmd_id={self.spmd_id} rank={self.rank}"
        )

    def _initialize_execution_lanes(self) -> None:
        self._execution_queues = {_EXECUTION_DEFAULT_LANE: self._execution_queue}
        for meta in self._operator_specs.values():
            lane = meta.runtime_policy.schedule.lane
            if lane is not None:
                self._execution_queue_for_lane(lane)

    def _execution_lane_for_op(self, op_name: str) -> str:
        meta = self.op_manager._get_operator_spec(op_name)
        lane = meta.runtime_policy.schedule.lane
        if lane is None:
            return _EXECUTION_DEFAULT_LANE
        return lane

    def _execution_queue_for_lane(
        self,
        lane: str,
    ) -> asyncio.PriorityQueue[tuple[int, int, int, str]]:
        if lane not in self._execution_queues:
            self._execution_queues[lane] = asyncio.PriorityQueue()
        return self._execution_queues[lane]

    def _execution_queue_for_op(
        self,
        op_name: str,
    ) -> asyncio.PriorityQueue[tuple[int, int, int, str]]:
        return self._execution_queue_for_lane(self._execution_lane_for_op(op_name))

    def _publish_execution_item(
        self,
        op_name: str,
        item: tuple[int, int, int, str],
    ) -> None:
        self._execution_queue_for_op(op_name).put_nowait(item)

    async def _put_execution_item(
        self,
        op_name: str,
        item: tuple[int, int, int, str],
    ) -> None:
        self._publish_execution_item(op_name, item)

    def _effective_schedule_key(
        self, op_name: str, step: int
    ) -> tuple[int, int, int, str]:
        meta = self.op_manager._get_operator_spec(op_name)
        policy = meta.runtime_policy.schedule
        base_priority = meta.runtime_policy.priority
        kind = policy.kind
        prepare_state = self._prepare_schedule_state(op_name, step)
        if kind == "deadline_boost":
            effective_priority = base_priority
            if self._deadline_boost_active(op_name, step):
                effective_priority += policy.boost_priority
            if self._uses_lane_aware_phase_priority(policy):
                effective_priority += self._lane_priority(policy, policy.lane)
            return (-effective_priority, prepare_state, step, op_name)
        if kind in ("phase_order", "phase_graph"):
            phase_order = policy.phase_ordering
            phase = policy.phase
            if phase is None or not phase_order or phase not in phase_order:
                raise ValueError(
                    f"{kind} schedule for op={op_name} has invalid phase={phase!r} "
                    f"phase_order={phase_order!r}."
                )
            phase_index = phase_order.index(phase)
            if self._uses_lane_aware_phase_priority(policy):
                lane = policy.lane
                lane_priority = self._lane_priority(policy, lane)
                phase_priority = self._phase_priority(policy, phase)
                return (
                    -base_priority - lane_priority - phase_priority,
                    prepare_state,
                    step,
                    op_name,
                )
            return (
                step * len(phase_order) + phase_index - base_priority,
                prepare_state,
                step,
                op_name,
            )
        if kind != "fifo":
            raise ValueError(f"Unsupported schedule policy {kind!r} for op={op_name}.")
        if self._uses_lane_aware_phase_priority(policy):
            lane_priority = self._lane_priority(policy, policy.lane)
            return (-base_priority - lane_priority, prepare_state, step, op_name)
        return (-base_priority, prepare_state, step, op_name)

    @staticmethod
    def _uses_lane_aware_phase_priority(policy: SchedulePolicy) -> bool:
        return bool(
            policy.lane
            or policy.lane_ordering
            or policy.lane_priorities
            or policy.phase_priorities
        )

    @staticmethod
    def _lane_priority(policy: SchedulePolicy, lane: str | None) -> int:
        return compute_lane_priority(policy, lane)

    @staticmethod
    def _phase_priority(policy: SchedulePolicy, phase: str) -> int:
        return policy.phase_priorities.get(phase, 0)

    def _prepare_schedule_state(self, op_name: str, step: int) -> int:
        runtime = getattr(self, "_input_prepare_runtime", None)
        if runtime is None:
            return 0
        if not self._should_prepare_inputs_on_ready(op_name):
            return 0
        if runtime.is_ready(op_name, step):
            return 0
        if runtime.is_pending(op_name, step):
            return 1
        return 0

    def _should_hold_prepare_pending_token(self, op_name: str, step: int) -> bool:
        runtime = getattr(self, "_input_prepare_runtime", None)
        if runtime is None or not runtime.is_pending(op_name, step):
            return False
        return not self._deadline_boost_active(op_name, step)

    def _deadline_boost_active(self, op_name: str, step: int) -> bool:
        meta = self.op_manager._get_operator_spec(op_name)
        policy = meta.runtime_policy.schedule
        if policy.kind != "deadline_boost":
            return False
        max_async_lag = meta.runtime_policy.max_async_lag
        if max_async_lag is None:
            raise ValueError(
                f"deadline_boost schedule for op={op_name} requires max_async_lag."
            )
        committed = [
            count
            for name, count in self._completed_step_count.items()
            if name in self._operator_specs
        ]
        slowest_committed = min(
            committed,
            default=self._completed_step_count.get(op_name, 0),
        )
        return step - slowest_committed >= max_async_lag

    def _operator_send_context(self, op_name: str) -> _OperatorSendContext:
        cached = self._operator_send_context_cache.get(op_name)
        if cached is not None:
            return cached
        if self._local_spmd_code is None:
            raise ValueError(
                f"[SPMDWorker] local_spmd_code is not registered "
                f"spmd_id={self.spmd_id} rank={self.rank}."
            )
        source_operator = self.config.require_graph_view().operator(op_name)
        dependencies = tuple(
            sorted(
                (
                    *self.op_manager.get_outgoing_tensor_dependencies(op_name),
                    *self.op_manager.get_outgoing_ordering_dependencies(op_name),
                ),
                key=lambda dependency: dependency.code,
            )
        )
        grouped_dependencies: dict[tuple[str, bool], list[CompiledDependency]] = {}
        for dependency in dependencies:
            key = (
                dependency.spec.target.spmd_id,
                dependency.spec.local_only,
            )
            grouped_dependencies.setdefault(key, []).append(dependency)

        interleave_unit_size = (
            source_operator.spec.interleave_unit_size
            if source_operator.spec.is_interleaved
            else None
        )
        effective_rank = (
            self.op_manager.get_dispatcher(op_name)
            .bind_execution_group(self._graph_view.mesh, self.rank)
            .logical_rank(self.rank)
        )
        successor_contexts: list[_SuccessorSendContext] = []
        for context_index, key in enumerate(sorted(grouped_dependencies)):
            spmd_id, local_only = key
            route_context = SuccessorRouteContext(
                context_index=context_index,
                dependencies=tuple(grouped_dependencies[key]),
                effective_rank=effective_rank,
            )
            if local_only:
                handlers = self._local_only_successor_handlers(
                    op_name,
                    route_context,
                )
            else:
                handlers = self._successor_handlers.get(spmd_id, {})
                if not handlers:
                    raise ValueError(
                        f"[SPMDWorker] Missing successor handlers for op={op_name} "
                        f"successor={spmd_id} source={self.spmd_id}:{self.rank}."
                    )
            successor_contexts.append(
                _SuccessorSendContext(
                    route_context=route_context,
                    handlers=handlers,
                )
            )

        route_contexts = tuple(context.route_context for context in successor_contexts)
        compiled_plan = compile_send_plan(
            graph_view=self._graph_view,
            source_op_id=source_operator.key.op_id,
            rank=self.rank,
            successor_contexts=route_contexts,
            sender_spmd_code=self._local_spmd_code,
            interleave_unit_size=interleave_unit_size,
        )
        successor_handlers = tuple(
            successor.handlers for successor in successor_contexts
        )
        context = _OperatorSendContext(
            compiled_plan=compiled_plan,
            successor_contexts=tuple(successor_contexts),
            successor_handlers=successor_handlers,
            transport=_compile_transport_send_context(
                compiled_plan,
                successor_handlers,
            ),
        )
        self._operator_send_context_cache[op_name] = context
        return context

    def _local_only_successor_handlers(
        self,
        op_name: str,
        route_context: SuccessorRouteContext,
    ) -> dict[int, Any]:
        if route_context.spmd_id != self.spmd_id:
            raise ValueError(
                "local_only successors must target the current SPMD, "
                f"op={op_name} source={self.spmd_id}:{self.rank} "
                f"successor={route_context.spmd_id}."
            )
        source_spec = route_context.dependencies[0].source_operator.spec
        if source_spec.is_interleaved:
            raise ValueError(
                f"local_only successors do not support interleaved routing, op={op_name}."
            )
        if route_context.target_world_size != self._world_size:
            raise ValueError(
                "local_only successors require matching world_size, "
                f"op={op_name} source_world_size={self._world_size} "
                f"target_world_size={route_context.target_world_size}."
            )
        handler = LocalWorkerDataPlane(
            self,
            sender_spmd_code=self._require_local_spmd_code(),
        )
        return self._profiler_runtime.wrap_data_plane_handlers(
            route_context.spmd_id,
            {self.rank: handler},
        )

    # =========================================================================
    # Execution Logic
    # =========================================================================

    def _start_run_loop(self) -> None:
        """
        Start the run loop as an asyncio task.

        The loop will block until start() is called.
        """
        self._initialize_execution_lanes()
        event_loop = asyncio.get_event_loop()
        self._lane_run_tasks = {}
        self._run_task = event_loop.create_task(self._run())
        self._lane_run_tasks[_EXECUTION_DEFAULT_LANE] = self._run_task
        for lane in self._execution_queues:
            if lane == _EXECUTION_DEFAULT_LANE:
                continue
            self._lane_run_tasks[lane] = event_loop.create_task(
                self._run_execution_lane_loop(lane)
            )
        logger.debug(
            f"Run loop task started waiting for start signal spmd_id={self.spmd_id} rank={self.rank}"
        )

    async def _run(self) -> None:
        await self._run_execution_lane_loop(_EXECUTION_DEFAULT_LANE)

    async def _run_execution_lane_loop(self, lane: str) -> None:
        token = set_active_resource_registry(self._resource_registry)
        try:
            await self._run_execution_lane_loop_bound(lane)
        finally:
            reset_active_resource_registry(token)

    async def _run_execution_lane_loop_bound(self, lane: str) -> None:
        """
        Internal main execution loop (async).

        Blocks on PriorityQueue until tasks are available.
        Respects op-level blocking.
        """
        # Wait for start signal from driver
        logger.debug(
            f"Waiting for start signal spmd_id={self.spmd_id} rank={self.rank}"
        )
        await self._execution_lifecycle.wait_for_activation()

        logger.debug(f"Execution loop started spmd_id={self.spmd_id} rank={self.rank}")
        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if profiler_recorder is not None:
            profiler_recorder.worker_scheduler_instant(spec=WORKER_RUN_LOOP)

        execution_queue = self._execution_queue_for_lane(lane)

        # Normal SPMD mode: data-driven execution loop
        while self._running:
            try:
                # Blocking wait for next task (async)
                logger.debug(
                    f"Waiting for next task in execution queue lane={lane} spmd_id={self.spmd_id} rank={self.rank}"
                )
                priority, _prepare_state, step, op_name = await execution_queue.get()
                try:
                    queue_key = (op_name, step)
                    if (
                        op_name != _EXECUTION_STOP_OP
                        and queue_key not in self._in_op_step_queue
                    ):
                        continue
                    if (
                        priority,
                        _prepare_state,
                        step,
                        op_name,
                    ) == _EXECUTION_STOP_ITEM:
                        break
                    if (
                        profiler_recorder is not None
                        and profiler_recorder.event_enabled(
                            WORKER_QUEUE_DEQUEUE, step=step
                        )
                    ):
                        profiler_recorder.worker_scheduler_instant(
                            spec=WORKER_QUEUE_DEQUEUE,
                            payload=profiler_recorder.worker_op_payload(
                                op_name,
                                step,
                                priority,
                            ),
                        )
                    logger.debug(
                        f"Execution loop picked up task op={op_name} step={step} priority={priority} "
                        f"spmd_id={self.spmd_id} rank={self.rank}"
                    )

                    # Check if we should stop
                    if not self._running:
                        break

                    decision = self._admission_state.evaluate(op_name, step)
                    if (
                        profiler_recorder is not None
                        and profiler_recorder.event_enabled(WORKER_ADMISSION, step=step)
                    ):
                        profiler_recorder.worker_admission(
                            op_name=op_name,
                            step=step,
                            action=decision.action.value,
                            reason=decision.reason,
                        )
                    if decision.action is not AdmissionAction.DISPATCH:
                        continue
                    self._start_local_step_plan(op_name, step)
                    self._spawn_background_task(
                        self._admission_state.start_pending(
                            op_name,
                            running=lambda: self._running,
                            source_exhausted=self._source_is_exhausted,
                            start_plan=self._start_local_step_plan,
                        )
                    )
                finally:
                    execution_queue.task_done()

            except asyncio.CancelledError:
                logger.info(
                    f"Run loop cancelled spmd_id={self.spmd_id} rank={self.rank}"
                )
                break
            except Exception as error:
                self._record_failure(error, phase="SCHEDULER")
                break

    async def stop(self) -> None:
        await self._execution_lifecycle.close(self._shutdown)

    async def _shutdown(self) -> None:
        self._fence_state = ExecutionFenceState.CLOSING
        self._driver_handle = None
        await self._source_scope.cancel()
        # Wake any worker blocked on drain condition
        async with self._drain_cond:
            self._drain_step_count = None
            self._drain_cond.notify_all()
        # Put sentinels to unblock all execution lanes.
        await self._execution_queue.put(_EXECUTION_STOP_ITEM)
        for lane, execution_queue in self._execution_queues.items():
            if lane == _EXECUTION_DEFAULT_LANE:
                continue
            await execution_queue.put(_EXECUTION_STOP_ITEM)

        pending_tasks = []
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            pending_tasks.append(self._run_task)
        for lane, task in self._lane_run_tasks.items():
            if lane == _EXECUTION_DEFAULT_LANE:
                continue
            if not task.done():
                task.cancel()
                pending_tasks.append(task)
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            pending_tasks.append(self._heartbeat_task)
        shutdown_errors: list[Exception] = []
        cancel_results = await asyncio.gather(
            self._input_prepare_runtime.cancel_pending(),
            self._worker_process_call_runtime.cancel_pending(),
            self._delivery_runtime.cancel_pending(),
            return_exceptions=True,
        )
        shutdown_errors.extend(
            result for result in cancel_results if isinstance(result, Exception)
        )
        send_tasks = [task for task in self._send_tasks if not task.done()]
        for task in send_tasks:
            task.cancel()
        if send_tasks:
            send_results = await asyncio.gather(*send_tasks, return_exceptions=True)
            shutdown_errors.extend(
                result for result in send_results if isinstance(result, Exception)
            )
        plan_tasks = [
            task for task in getattr(self, "_plan_tasks", set()) if not task.done()
        ]
        for task in plan_tasks:
            task.cancel()
        pending_tasks.extend(plan_tasks)
        if pending_tasks:
            pending_results = await asyncio.gather(
                *pending_tasks, return_exceptions=True
            )
            shutdown_errors.extend(
                result for result in pending_results if isinstance(result, Exception)
            )
        background_tasks = [task for task in self._background_tasks if not task.done()]
        for task in background_tasks:
            task.cancel()
        if background_tasks:
            background_results = await asyncio.gather(
                *background_tasks, return_exceptions=True
            )
            shutdown_errors.extend(
                result for result in background_results if isinstance(result, Exception)
            )
        self._plan_tasks.clear()
        self._plan_task_steps.clear()
        if hasattr(self, "_send_tasks"):
            self._send_tasks.clear()
        self._background_tasks.clear()
        user_shutdown_results = await asyncio.gather(
            self._shutdown_checkpoint_coordinator(),
            self._shutdown_local_operator_instances(),
            self._local_step_plan_executor.user_callable_executor.shutdown(),
            return_exceptions=True,
        )
        shutdown_errors.extend(
            result for result in user_shutdown_results if isinstance(result, Exception)
        )
        for close in (
            self._input_prepare_runtime.close,
            self._worker_process_call_runtime.close,
        ):
            try:
                close()
            except Exception as error:
                shutdown_errors.append(error)
        for tensors in self._coherence_input_tensors.values():
            for tensor in tensors:
                tensor._invalidate_runtime_access()
                tensor._release_runtime_payload()
        self._coherence_input_tensors.clear()
        self._coherence_output_intents.clear()
        self._coherence_sequencer.fence()
        self._coherence_directory.fence()
        self._allocation_registry.fence()
        self._fence_state = ExecutionFenceState.FENCED
        if len(shutdown_errors) == 1:
            raise shutdown_errors[0]
        if shutdown_errors:
            raise ExceptionGroup("worker shutdown failed", shutdown_errors)

        logger.debug(f"Worker stopping spmd_id={self.spmd_id} rank={self.rank}")

    async def _shutdown_checkpoint_coordinator(self) -> None:
        coordinator = self._checkpoint_coordinator
        if coordinator is None:
            return
        errors: list[BaseException] = []
        try:
            await coordinator.rollback_active()
        except BaseException as error:
            errors.append(error)
        try:
            await coordinator.close()
        except BaseException as error:
            errors.append(error)
        self._checkpoint_coordinator = None
        self._checkpoint_repository_root = None
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("checkpoint coordinator shutdown failed", errors)

    async def _propagate_completed_without_tensor(
        self,
        op_name: str,
        step: int,
        dependency_decisions: DependencyDecisions,
    ) -> tuple[asyncio.Task, ...]:
        """Propagate protocol-empty tensor arrivals to successor SPMDs."""
        if self._op_manager is None:
            raise RuntimeError("Cannot propagate empty packets before worker setup.")

        send_context = self._operator_send_context(op_name)
        if not send_context.successor_contexts:
            return ()

        await self._deposit_local_only_orderings(
            send_context.compiled_plan,
            step,
            dependency_decisions,
        )
        await self._deposit_local_only_dependency_decisions(
            send_context.compiled_plan,
            step,
            dependency_decisions,
        )

        backend = get_backend()
        empty_values = backend.zeros((0,), dtype=backend.int32())
        tasks = await self._data_plane_executor.send_completed_without_tensor_frames(
            compiled_plan=send_context.compiled_plan,
            exec_step=step,
            successor_handlers=send_context.successor_handlers,
            source_spmd_id=self.spmd_id,
            batch_max_frames=self._data_plane_batch_max_frames,
            batch_flush_us=self._data_plane_batch_flush_us,
            submit_delivery=self._delivery_submitter(op_name, step),
            delivery_schedule_key=self._effective_schedule_key(op_name, step),
            batch_submit=partial(
                self._coherence_batch_retire_runtime.retire_publication,
                send_context.compiled_plan,
            ),
            dependency_decisions=dependency_decisions,
            empty_values=empty_values,
        )
        self._close_step_coherence_intents(op_name, step)
        return tuple(tasks)

    async def _propagate_skipped_completion(
        self,
        op_name: str,
        step: int,
        dependency_decisions: DependencyDecisions,
    ) -> tuple[asyncio.Task, ...]:
        if self._op_manager is None:
            raise RuntimeError(
                "Cannot propagate skipped completion before worker setup."
            )
        send_context = self._operator_send_context(op_name)
        if not send_context.successor_contexts:
            return ()
        await self._deposit_local_only_orderings(
            send_context.compiled_plan,
            step,
            dependency_decisions,
        )
        await self._deposit_local_only_dependency_decisions(
            send_context.compiled_plan,
            step,
            dependency_decisions,
        )
        tasks = await self._data_plane_executor.send_dependency_decision_frames(
            compiled_plan=send_context.compiled_plan,
            flat_results=[],
            step_transfer_cache={},
            exec_step=step,
            successor_handlers=send_context.successor_handlers,
            source_spmd_id=self.spmd_id,
            batch_max_frames=self._data_plane_batch_max_frames,
            batch_flush_us=self._data_plane_batch_flush_us,
            submit_delivery=self._delivery_submitter(op_name, step),
            delivery_schedule_key=self._effective_schedule_key(op_name, step),
            batch_submit=partial(
                self._coherence_batch_retire_runtime.retire_publication,
                send_context.compiled_plan,
            ),
            dependency_decisions=dependency_decisions,
        )
        self._close_step_coherence_intents(op_name, step)
        return tuple(tasks)

    def _operator_pipeline_depth(self, op_name: str) -> int:
        try:
            meta = self._operator_specs[op_name]
        except KeyError as error:
            raise KeyError(f"unknown operator pipeline depth: {op_name!r}") from error
        depth = meta.runtime_policy.pipeline_depth
        if depth <= 0:
            raise ValueError(
                f"local_pipeline_depth must be positive for op={op_name}, got {depth}."
            )
        return depth

    def _can_start_local_plan(self, op_name: str) -> bool:
        return len(self._running_plan_steps[op_name]) < self._operator_pipeline_depth(
            op_name
        )

    def _start_local_step_plan(self, op_name: str, step: int) -> None:
        if self._failure is not None or not self._running:
            return
        self._declare_step_coherence_intents(op_name, step)
        self._record_local_plan_scheduled(op_name, step)
        self._record_execution_phase("admission.accepted", op_name=op_name, step=step)
        self._running_plan_steps[op_name].add(step)
        task = asyncio.create_task(self._run_local_step_plan(op_name, step))
        self._plan_tasks.add(task)
        self._plan_task_steps[task] = (op_name, step)

        def _done(done_task: asyncio.Task) -> None:
            self._plan_tasks.discard(done_task)
            self._plan_task_steps.pop(done_task, None)
            self._running_plan_steps[op_name].discard(step)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc is not None:
                self._record_failure(
                    exc,
                    phase="LOCAL_STEP_PLAN",
                    op_name=op_name,
                    step=step,
                )
                return
            if self._running:
                self._spawn_background_task(
                    self._admission_state.start_pending(
                        op_name,
                        running=lambda: self._running,
                        source_exhausted=self._source_is_exhausted,
                        start_plan=self._start_local_step_plan,
                    )
                )

        task.add_done_callback(_done)

    def _declare_step_coherence_intents(self, op_name: str, step: int) -> None:
        key = (op_name, step)
        if key in self._coherence_output_intents:
            raise CoherenceStateTransitionError(
                f"duplicate output coherence intents for op={op_name} step={step}."
            )
        send_context = self._operator_send_context(op_name)
        local_code = self._require_local_spmd_code()
        instances_list: list[CoherenceInstance] = []
        instances_seen: set[CoherenceInstance] = set()
        for successor in send_context.compiled_plan.successor_routes:
            for route in successor.tensor_routes:
                instance = CoherenceInstance(
                    CoherenceSubject(local_code, self.rank, route.logical_tuuid),
                    step,
                )
                if instance not in instances_seen:
                    instances_seen.add(instance)
                    instances_list.append(instance)
        instances = tuple(instances_list)
        for instance in instances:
            self._coherence_sequencer.issue_intent(instance)
        self._coherence_output_intents[key] = instances

    async def _retire_coherence_payload(
        self,
        instances: tuple[CoherenceInstance, ...],
        payload: Callable[[], Awaitable[Any]],
    ) -> Any:
        del instances
        if not callable(payload):
            raise TypeError(
                "coherence sequencer payload must be an async retirement callable."
            )
        return await payload()

    def _close_step_coherence_intents(self, op_name: str, step: int) -> None:
        instances = self._coherence_output_intents.pop((op_name, step), ())
        for instance in instances:
            self._coherence_sequencer.close_intent(instance)

    def _abort_step_coherence_intents(
        self,
        op_name: str,
        step: int,
        error: BaseException,
    ) -> None:
        instances = self._coherence_output_intents.pop((op_name, step), ())
        for instance in instances:
            self._coherence_sequencer.abort_intent(instance, error)

    def _record_local_plan_scheduled(self, op_name: str, step: int) -> None:
        is_periodic = self.op_manager.is_periodic_trigger(op_name)
        if is_periodic:
            return
        scheduled_count = self._scheduled_step_count[op_name]
        if step != scheduled_count:
            raise RuntimeError(
                f"local plan schedule order violation op={op_name} step={step} "
                f"scheduled_count={scheduled_count}."
            )
        self._scheduled_step_count[op_name] = scheduled_count + 1

    def _bounded_async_ops(self) -> tuple[str, ...]:
        metas = self._operator_specs
        result = [
            name
            for name, meta in metas.items()
            if meta.runtime_policy.has_bounded_staleness
        ]
        return tuple(sorted(result))

    def _async_lag_limit_reached(self, op_name: str, step: int) -> bool:
        meta = self.op_manager._get_operator_spec(op_name)
        if not meta.runtime_policy.has_bounded_staleness:
            return False
        max_staleness = meta.runtime_policy.max_staleness
        if max_staleness is None:
            raise RuntimeError(
                f"Operator {op_name!r} has bounded staleness but max_staleness is None."
            )
        peers = self._bounded_async_ops()
        if len(peers) < 2:
            return False
        slowest_completed = min(
            self._completed_step_count.get(name, 0) for name in peers
        )
        own_completed = self._completed_step_count.get(op_name, 0)
        if own_completed <= slowest_completed:
            return False
        return step - slowest_completed > max_staleness

    def _phase_order_step_blocked(self, op_name: str, step: int) -> bool:
        gate = self._phase_schedule.gates.get(op_name)
        if gate is None:
            return False
        return gate.is_blocked(self._completed_step_count, step=step)

    def _state_resource_id(self, op_name: str) -> int:
        meta = self.op_manager.get_operator_spec(op_name)
        is_stateful = bool(meta.owner_class_qualname)
        if not is_stateful:
            return -1
        state_access = meta.runtime_policy.state_access
        if state_access == "thread_safe":
            return -1
        if state_access != "exclusive":
            raise ValueError(
                f"state_access for op={op_name} must be 'exclusive' or 'thread_safe', "
                f"got {state_access!r}."
            )
        state_resource = meta.runtime_policy.state_resource
        owner_key = (
            f"{meta.owner_class_module}."
            f"{meta.owner_class_qualname}:"
            f"{state_resource or '__owner__'}"
        )
        from fluxway.profiler.record import stable_u64

        return stable_u64(self.spmd_id, owner_key) % ((1 << 31) - 1)

    def _commit_resource_id(self, op_name: str) -> int:
        from fluxway.profiler.record import stable_u64

        return stable_u64(self.spmd_id, _COMMIT_STEP_RESOURCE_SALT, op_name) % (
            (1 << 31) - 1
        )

    def _compile_operator_local_plan(self, op_name: str) -> LocalStepPlan:
        return LocalStepPlan(
            node_kinds=(
                int(LocalStepNodeKind.WAIT),
                int(LocalStepNodeKind.CALL),
                int(LocalStepNodeKind.SEND),
                int(LocalStepNodeKind.AWAIT_SEND),
                int(LocalStepNodeKind.BARRIER),
            ),
            callable_ids=(
                _WAIT_INPUTS_CALLABLE,
                _CALL_USER_OP_CALLABLE,
                _SEND_OUTPUT_CALLABLE,
                _AWAIT_SEND_CALLABLE,
                _COMMIT_STEP_CALLABLE,
            ),
            resource_ids=(
                _WAIT_INPUTS_RESOURCE,
                self._state_resource_id(op_name),
                _SEND_OUTPUT_RESOURCE,
                _AWAIT_SEND_RESOURCE,
                self._commit_resource_id(op_name),
            ),
            dependency_offsets=(0, 0, 1, 3, 4, 5),
            dependencies=(0, 0, 1, 2, 3),
            result_node=4,
        )

    def _record_execution_phase(
        self,
        phase: str,
        *,
        op_name: str | None = None,
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Record opaque execution metadata without interpreting a backend."""
        now = time.time()
        self._last_execution_activity_timestamp = now
        self._execution_phase = phase
        self._execution_phase_started_timestamp = now
        self._execution_phase_active = phase.endswith((".begin", ".wait")) or (
            phase == "admission.accepted"
        )
        self._execution_operator = op_name
        self._execution_step = step
        if epoch is not None:
            self._execution_epoch = epoch
        self._execution_sequence = getattr(self, "_execution_sequence", 0) + 1
        if not getattr(self._profiler_runtime, "enabled", False):
            return
        self._profiler_runtime.recorder.execution_phase(
            phase=phase,
            op_name=op_name,
            step=step,
            epoch=getattr(self, "_execution_epoch", 0),
            sequence=self._execution_sequence,
            execution_generation=getattr(self, "_execution_generation", 0),
        )

    async def _run_local_step_plan(self, op_name: str, step: int) -> ExecutionResult:
        self._record_execution_phase("local_plan.begin", op_name=op_name, step=step)
        try:
            plan = self._local_step_plan_cache[op_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Operator {op_name!r} has no installed local step plan."
            ) from exc
        registry = self._operator_local_node_registry(op_name, step)
        context = LocalStepContext(op_name=op_name, step=step)
        try:
            result = await self._local_step_plan_executor.execute(
                plan, registry, context
            )
        except BaseException:
            self._record_execution_phase("local_plan.error", op_name=op_name, step=step)
            raise
        self._record_execution_phase("local_plan.end", op_name=op_name, step=step)
        return result

    def _operator_local_node_registry(
        self, op_name: str, step: int
    ) -> MappingCallableRegistry:
        worker_self = self

        class OperatorLocalNodeRegistry(MappingCallableRegistry):
            def __init__(self) -> None:
                super().__init__({})

            def spec_for(
                self,
                kind: LocalStepNodeKind,
                callable_id: str,
                resource_id: int,
                step_context: LocalStepContext = LocalStepContext(),
            ) -> LocalNodeSpec:
                del resource_id, step_context
                if callable_id == _WAIT_INPUTS_CALLABLE:
                    return LocalNodeSpec(
                        callable_id=callable_id,
                        fn=lambda: worker_self._local_plan_wait_inputs(op_name, step),
                    )
                if callable_id == _CALL_USER_OP_CALLABLE:
                    return LocalNodeSpec(
                        callable_id=callable_id,
                        user_runner=lambda dependency_results: (
                            worker_self._local_plan_call_user(
                                op_name,
                                step,
                                dependency_results[0],
                            )
                        ),
                    )
                if callable_id == _SEND_OUTPUT_CALLABLE:
                    return LocalNodeSpec(
                        callable_id=callable_id,
                        fn=lambda prepared_input, op_result: (
                            worker_self._local_plan_send_output(
                                op_name,
                                step,
                                prepared_input,
                                op_result,
                            )
                        ),
                    )
                if callable_id == _COMMIT_STEP_CALLABLE:
                    return LocalNodeSpec(
                        callable_id=callable_id,
                        fn=lambda result: worker_self._local_plan_commit(
                            op_name, step, result
                        ),
                    )
                if callable_id == _AWAIT_SEND_CALLABLE:
                    return LocalNodeSpec(
                        callable_id=callable_id,
                        fn=worker_self._local_plan_await_send,
                    )
                raise KeyError(f"Unknown local plan callable_id {callable_id!r}.")

        return OperatorLocalNodeRegistry()

    async def _local_plan_wait_inputs(self, op_name: str, step: int) -> Any:
        recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )

        def prepare_save():
            if self._should_prepare_inputs_on_ready(op_name):
                return self._input_prepare_runtime.consume(
                    op_name,
                    step,
                    lambda: self._execute_operator_prepare_input(op_name, step),
                )
            return self._execute_operator_prepare_input(op_name, step)

        if recorder is None:
            return await prepare_save()
        async with recorder.local_step_node(
            spec=LOCAL_STEP_WAIT_INPUTS,
            op_name=op_name,
            step=step,
        ):
            return await prepare_save()

    async def _local_plan_call_user(
        self, op_name: str, step: int, input_data: Any
    ) -> Any:
        if isinstance(input_data, ExecutionResult):
            return input_data
        if not isinstance(input_data, _PreparedOperatorInput):
            raise TypeError(
                f"Local CALL for op={op_name} step={step} requires prepared input "
                f"metadata from WAIT; actual_type={type(input_data).__name__}."
            )
        operator_input = input_data.value
        self._record_execution_phase(
            "operator.invocation.begin", op_name=op_name, step=step
        )
        await self._apply_operator_tensor_access(op_name, step, input_data)
        recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        try:
            if recorder is None:
                spec = self._operator_user_callable_spec(
                    op_name,
                    step,
                    operator_input,
                    source_work_items=input_data.source_work_items,
                )
                result = (
                    await self._local_step_plan_executor.user_callable_executor.execute(
                        spec
                    )
                )
            else:
                async with recorder.local_step_node(
                    spec=LOCAL_STEP_CALL_USER,
                    op_name=op_name,
                    step=step,
                ):
                    spec = self._operator_user_callable_spec(
                        op_name,
                        step,
                        operator_input,
                        source_work_items=input_data.source_work_items,
                    )
                    result = await self._local_step_plan_executor.user_callable_executor.execute(
                        spec
                    )
            normalized = self._normalize_interleaved_source_result(
                op_name, step, result
            )
        except BaseException:
            self._record_execution_phase(
                "operator.invocation.error", op_name=op_name, step=step
            )
            raise
        self._record_execution_phase(
            "operator.invocation.end", op_name=op_name, step=step
        )
        return normalized

    async def _apply_operator_tensor_access(
        self,
        op_name: str,
        step: int,
        prepared_input: _PreparedOperatorInput,
    ) -> None:
        spec = self.config.require_graph_view().operator(op_name).spec
        intents: dict[TransportTensor, TensorAccessIntent] = {}
        selected_grants: dict[
            tuple[str | None, int, CoherenceSubject, int, int],
            _InputCoherenceGrant,
        ] = {}
        requires_ownership = False
        for binding in prepared_input.tensor_bindings:
            intent = spec.tensor_access[binding.port]
            previous = intents.setdefault(binding.tensor, intent)
            if previous is not intent:
                raise CoherenceStateTransitionError(
                    f"tensor input {binding.port!r} has conflicting access intents."
                )
            if intent in (TensorAccessIntent.WRITABLE, TensorAccessIntent.DONATE):
                requires_ownership = True
                for grant in binding.grants:
                    selected_grants[grant.identity] = grant
        self._coherence_input_tensors.setdefault((op_name, step), set()).update(intents)
        for tensor in intents:
            tensor._bind_consumer_lease_owner(self._coherence_sequencer)
        if requires_ownership:
            if not selected_grants:
                raise CoherenceStateTransitionError(
                    f"writable tensor access has no grants for op={op_name} step={step}."
                )
            await self._acquire_owned_inputs(
                op_name,
                step,
                selected_grants,
            )
        for tensor, intent in intents.items():
            try:
                tensor._set_runtime_access_intent(intent)
            except Exception as error:
                ports = tuple(
                    binding.port
                    for binding in prepared_input.tensor_bindings
                    if binding.tensor is tensor
                )
                raise RuntimeError(
                    "cannot grant operator tensor access: "
                    f"op={op_name!r} step={step} ports={ports!r} "
                    f"tensor_id={id(tensor)} source_step={tensor.source_step!r} "
                    f"coherence_instances={tensor._coherence_instances!r} "
                    f"intent={intent!r}: {error.__class__.__name__}({error})"
                ) from error

    async def _acquire_owned_inputs(
        self,
        op_name: str,
        step: int,
        selected_grants: Mapping[
            tuple[str | None, int, CoherenceSubject, int, int], _InputCoherenceGrant
        ],
    ) -> None:
        grants = self._coherence_input_grants.get((op_name, step))
        if not grants:
            raise CoherenceStateTransitionError(
                "writable tensor access has no input grants for "
                f"op={op_name} step={step}."
            )
        grouped: dict[
            tuple[CoherenceSubject, int],
            list[_InputCoherenceGrant],
        ] = {}
        for grant in selected_grants.values():
            if grant.local_access_token.kind is not GrantKind.READ:
                raise CoherenceStateTransitionError(
                    "donation requires active local read grants."
                )
            grouped.setdefault((grant.subject, grant.epoch), []).append(grant)
        ordered_groups = tuple(grouped.values())
        reservations: list[_OwnershipReservation] = []
        parent_dispatch_started = False
        try:
            for group in ordered_groups:
                reservation = _OwnershipReservation(tuple(group), [])
                reservations.append(reservation)
                for grant in group:
                    self._coherence_directory.release(grant.local_access_token)
                    reservation.released_read_tokens.append(grant.local_access_token)
                request = self._coherence_directory.request_write(
                    group[0].subject,
                    group[0].epoch,
                    ExecutionToken(step),
                )
                if isinstance(request, GrantPending):
                    reservation.pending = request
                elif isinstance(request, GrantReady):
                    reservation.write_token = request.token
                elif isinstance(request, GrantRejected):
                    raise request.error
                else:
                    raise TypeError(
                        f"unsupported grant result={type(request).__name__}."
                    )
            parent_requests = tuple(
                reservation
                for reservation in reservations
                if reservation.pending is not None
                and reservation.pending.requires_parent_ownership
            )
            parent_request_groups: dict[
                tuple[int, int, int],
                list[_OwnershipReservation],
            ] = {}
            for reservation in parent_requests:
                canonical = reservation.canonical
                parent_request_groups.setdefault(
                    (
                        canonical.subject.owner_spmd_code,
                        canonical.subject.owner_rank,
                        canonical.epoch,
                    ),
                    [],
                ).append(reservation)
            parent_dispatch_started = bool(parent_request_groups)
            await asyncio.gather(
                *(
                    self._send_coherence_batch_to_parent(
                        group[0].canonical.subject,
                        tuple(item.canonical.route_code for item in group),
                        group[0].canonical.epoch,
                        CoherenceAction.RFO,
                    )
                    for group in parent_request_groups.values()
                )
            )
            for reservation in parent_requests:
                canonical = reservation.canonical
                if not self._coherence_directory.has_parent_ownership(
                    canonical.subject,
                    canonical.epoch,
                ):
                    raise CoherenceStateTransitionError(
                        "RFO completed without per-subject ownership authority."
                    )
                reservation.ownership_granted = True
            pending_reservations = tuple(
                reservation
                for reservation in reservations
                if reservation.pending is not None
            )
            if pending_reservations:
                pending_grants = tuple(
                    reservation.pending
                    for reservation in pending_reservations
                    if reservation.pending is not None
                )
                if len(pending_grants) != len(pending_reservations):
                    raise CoherenceStateTransitionError(
                        "pending reservation lost its request handle."
                    )
                tokens = await asyncio.gather(
                    *(
                        self._coherence_admission_runtime.await_grant(pending)
                        for pending in pending_grants
                    )
                )
                for reservation, token in zip(
                    pending_reservations,
                    tokens,
                    strict=True,
                ):
                    reservation.write_token = token
        except BaseException:
            if parent_dispatch_started:
                self._record_failure(
                    CoherenceStateTransitionError(
                        "batched RFO failed after remote effects may have occurred."
                    ),
                    phase="COHERENCE_RFO",
                    op_name=op_name,
                    step=step,
                )
                raise
            await self._rollback_ownership_reservations(
                grants,
                reservations,
            )
            raise

        for reservation in reservations:
            if reservation.write_token is None:
                raise CoherenceStateTransitionError(
                    "ownership reservation completed without a write token."
                )
            for grant in reservation.grants:
                grants.pop(grant.identity)
            updated = replace(
                reservation.canonical,
                local_access_token=reservation.write_token,
            )
            grants[updated.identity] = updated

    def _ensure_local_input_grant(
        self,
        op_name: str,
        step: int,
        transfer_code: int,
        tensor: TransportTensor,
    ) -> _InputCoherenceGrant:
        try:
            source_spmd_id, source_rank = self._local_input_origins[
                (op_name, step, transfer_code)
            ]
        except KeyError as exc:
            raise CoherenceStateTransitionError(
                f"local-only input origin is missing for transfer_code={transfer_code}."
            ) from exc
        if source_spmd_id == self.spmd_id:
            owner_code = self._require_local_spmd_code()
        else:
            owner_codes = [
                code
                for code, spmd_id in self._spmd_id_by_code.items()
                if spmd_id == source_spmd_id
            ]
            if len(owner_codes) != 1:
                raise CoherenceStateTransitionError(
                    f"local-only input origin {source_spmd_id!r} has no unique SPMD code."
                )
            owner_code = owner_codes[0]
        subject = self._coherence_subject(owner_code, source_rank, transfer_code)
        expected_subject = subject
        bound_instances = tensor._coherence_instances
        if len(bound_instances) > 1:
            matching_instances = tuple(
                instance
                for instance in bound_instances
                if instance.subject == expected_subject
            )
            if len(matching_instances) != 1:
                raise CoherenceStateTransitionError(
                    "local-only input has ambiguous coherence instances: "
                    f"transfer_code={transfer_code} instances={bound_instances!r}."
                )
            epoch = matching_instances[0].epoch
        elif bound_instances:
            instance = bound_instances[0]
            if instance.subject != expected_subject:
                raise CoherenceStateTransitionError(
                    "local-only input coherence subject mismatch: "
                    f"expected={expected_subject} actual={instance.subject}."
                )
            epoch = instance.epoch
        else:
            raise CoherenceStateTransitionError(
                "local-only input has no canonical coherence epoch: "
                f"transfer_code={transfer_code}."
            )
        grants = self._coherence_input_grants.setdefault((op_name, step), {})
        identity = (None, source_rank, subject, transfer_code, epoch)
        existing = grants.get(identity)
        if existing is not None:
            return existing
        entry = self._coherence_directory.admit(CoherenceInstance(subject, epoch))
        if entry.state is CoherenceState.INVALID:
            self._coherence_directory.publish_local_payload(subject, epoch)
        local_read = self._coherence_directory.request_read(
            subject,
            epoch,
            ExecutionToken(step),
        )
        if not isinstance(local_read, GrantReady):
            raise CoherenceStateTransitionError(
                "local-only input did not produce an immediate local read grant."
            )
        grant = _InputCoherenceGrant(
            parent_spmd_id=None,
            parent_rank=source_rank,
            subject=subject,
            route_code=transfer_code,
            epoch=epoch,
            local_access_token=local_read.token,
            parent_return_required=False,
        )
        grants[grant.identity] = grant
        tensor_instance = CoherenceInstance(subject, epoch)
        if not tensor._coherence_instances:
            tensor._bind_coherence_instances((tensor_instance,))
            if not self._allocation_registry.is_bound(tensor_instance):
                self._allocation_registry.bind(
                    tensor_instance,
                    runtime_allocation_descriptor(tensor),
                )
        return grant

    async def _rollback_ownership_reservations(
        self,
        grants: dict[
            tuple[str | None, int, CoherenceSubject, int, int],
            _InputCoherenceGrant,
        ],
        reservations: list[_OwnershipReservation],
    ) -> None:
        try:
            for reservation in reservations:
                canonical = reservation.canonical
                if reservation.pending is not None:
                    self._coherence_directory.cancel_pending_request(
                        reservation.pending.request_id
                    )
                if reservation.write_token is not None:
                    self._coherence_directory.release(reservation.write_token)
                else:
                    released = set(reservation.released_read_tokens)
                    for grant in reservation.grants:
                        if grant.local_access_token in released:
                            continue
                        self._coherence_directory.release(grant.local_access_token)
                        reservation.released_read_tokens.append(
                            grant.local_access_token
                        )
                if reservation.ownership_granted:
                    self._coherence_directory.cancel_parent_ownership(
                        canonical.subject,
                        canonical.epoch,
                    )
                    if canonical.parent_return_required:
                        await self._send_coherence_to_parent(
                            canonical.subject,
                            canonical.route_code,
                            canonical.epoch,
                            CoherenceAction.OWNERSHIP_CANCEL,
                        )
                if self._coherence_directory.parent_return_ready(
                    canonical.subject,
                    canonical.epoch,
                ):
                    if (
                        canonical.parent_return_required
                        and not reservation.ownership_granted
                    ):
                        assert canonical.parent_spmd_id is not None
                        await self._parent_grant_return_runtime.return_parent_grants(
                            (
                                ParentReturnRoute(
                                    subject=canonical.subject,
                                    route_code=canonical.route_code,
                                    epoch=canonical.epoch,
                                    parent_spmd_id=canonical.parent_spmd_id,
                                    parent_rank=canonical.parent_rank,
                                ),
                            )
                        )
        finally:
            for reservation in reservations:
                for grant in reservation.grants:
                    grants.pop(cast(Any, grant.identity), None)

    def _normalize_interleaved_source_result(
        self,
        op_name: str,
        step: int,
        result: Any,
    ) -> Any:
        operator = self.config.require_graph_view().operator(op_name)
        if not operator.spec.is_source or not operator.spec.is_interleaved:
            return result
        if self._source_blocks_step(op_name, step):
            return ExecutionResult(status=ExecutionStatus.SOURCE_EXHAUSTED)
        if result is None:
            self._record_source_terminal(op_name, step)
            return ExecutionResult(status=ExecutionStatus.SOURCE_EXHAUSTED)
        return result

    def _record_source_terminal(
        self, op_name: str, step: int, *, notify_controller: bool = True
    ) -> None:
        transition = reduce_drain_boundary(
            self._drain_boundary,
            DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, step),
        )
        self._drain_boundary = transition.boundary
        if notify_controller and (
            self._pending_terminal_observation is None
            or step < self._pending_terminal_observation[1]
        ):
            self._pending_terminal_observation = (op_name, step)

    def _notify_source_terminal(self) -> None:
        observation = self._pending_terminal_observation
        if observation is None:
            return
        op_name, terminal_step = observation
        self._pending_terminal_observation = None
        driver = self._driver_handle
        if driver is None:
            raise RuntimeError(
                f"Worker {self.spmd_id!r} rank {self.rank} has no Driver handle "
                "for a terminal boundary event"
            )
        driver.report_boundary_event.remote(
            op_name,
            DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, terminal_step),
        )

    async def _forward_control_requests(self, requests: tuple[Any, ...]) -> None:
        if not isinstance(requests, tuple):
            raise TypeError("worker control requests must be a tuple")
        if not requests:
            return
        driver = self._driver_handle
        if driver is None:
            raise RuntimeError(
                f"Worker {self.spmd_id!r} rank {self.rank} has no Driver handle "
                "for control requests"
            )
        for request in requests:
            if request is None:
                raise ValueError("worker control requests cannot contain None")
            await driver.forward_control_plane_request.remote(request)

    async def _local_plan_send_output(
        self,
        op_name: str,
        step: int,
        prepared_input: Any,
        op_result: Any,
    ) -> LocalSendToken:
        recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        step_transfer_cache = self._step_transfer_cache_from_prepared_input(
            op_name,
            step,
            prepared_input,
            op_result,
        )
        handoffs = self._register_coherence_alias_dependencies(
            op_name,
            step,
            op_result,
        )
        if recorder is None:
            token = self._register_local_send(
                op_name, step, op_result, step_transfer_cache
            )
            self._commit_allocation_handoffs(handoffs, step)
            return token
        source_spmd_code = self._require_local_spmd_code()
        async with recorder.local_step_node(
            spec=LOCAL_STEP_SEND_OUTPUT,
            op_name=op_name,
            step=step,
            source_spmd_code=source_spmd_code,
            source_rank=self.rank,
        ):
            token = self._register_local_send(
                op_name,
                step,
                op_result,
                step_transfer_cache,
            )
            self._commit_allocation_handoffs(handoffs, step)
            return token

    def _register_coherence_alias_dependencies(
        self,
        op_name: str,
        step: int,
        op_result: Any,
    ) -> tuple[_AllocationHandoffIntent, ...]:
        input_tensors = self._coherence_input_tensors.get((op_name, step))
        if not input_tensors:
            return ()
        result = self._normalize_local_plan_result(op_name, step, op_result)
        if not self._result_requires_send(result) or result.data is None:
            return ()
        meta = self.op_manager.get_operator_spec(op_name)
        flat_results, _treedef = tree_flatten(
            result.data,
            treedef_template=(
                None
                if meta.output_treedef is None
                else meta.output_treedef.materialize()
            ),
        )
        plan = self._operator_send_context(op_name).compiled_plan
        local_code = self._require_local_spmd_code()
        routes = {
            (route.output_index, route.logical_tuuid): route
            for context in plan.successor_routes
            for route in context.tensor_routes
        }.values()
        outputs = tuple(
            (
                route,
                flat_results[route.output_index],
                CoherenceInstance(
                    CoherenceSubject(
                        owner_spmd_code=local_code,
                        owner_rank=self.rank,
                        logical_tuuid=route.logical_tuuid,
                    ),
                    step,
                ),
                runtime_allocation_descriptor(flat_results[route.output_index]),
            )
            for route in routes
        )
        outputs_by_root: dict[
            tuple[str, str, int, int],
            list[tuple[Any, CoherenceInstance, Any]],
        ] = {}
        for _route, output, instance, descriptor in outputs:
            if (
                isinstance(output, TransportTensor)
                and output.source_step is not None
                and output.source_step != step
            ):
                raise ValueError(
                    f"Operator {op_name!r} output has source_step="
                    f"{output.source_step}, but canonical coherence epoch is "
                    f"operator step={step}."
                )
            identity = descriptor.identity
            outputs_by_root.setdefault(
                (
                    identity.backend,
                    identity.device,
                    identity.root_address,
                    identity.root_nbytes,
                ),
                [],
            ).append((output, instance, descriptor))
        spec = self.config.require_graph_view().operator(op_name).spec
        bindings = self._coherence_input_bindings.get((op_name, step), ())
        donated_bindings = tuple(
            binding
            for binding in bindings
            if spec.tensor_access[binding.port] is TensorAccessIntent.DONATE
        )
        claimed_outputs: set[CoherenceInstance] = set()
        handoffs: list[_AllocationHandoffIntent] = []
        for input_tensor in input_tensors:
            input_descriptor = runtime_allocation_descriptor(input_tensor)
            input_identity = input_descriptor.identity
            matches = tuple(
                (output, instance, descriptor)
                for output, instance, descriptor in outputs_by_root.get(
                    (
                        input_identity.backend,
                        input_identity.device,
                        input_identity.root_address,
                        input_identity.root_nbytes,
                    ),
                    (),
                )
                if input_identity.relation(descriptor.identity)
                is not AllocationRelation.DISJOINT
            )
            for _output, instance, _descriptor in matches:
                self._coherence_alias_dependencies.setdefault(
                    (op_name, step),
                    set(),
                ).add(instance)
            donation_binding = next(
                (
                    binding
                    for binding in donated_bindings
                    if binding.tensor is input_tensor
                ),
                None,
            )
            if donation_binding is None:
                continue
            if len(input_tensor._coherence_instances) != 1:
                raise UnsupportedAllocationHandoffError(
                    "donate requires one input coherence instance."
                )
            exact_matches = tuple(
                (output, instance, descriptor)
                for output, instance, descriptor in matches
                if input_identity.relation(descriptor.identity)
                is AllocationRelation.EXACT
            )
            if len(exact_matches) != 1:
                raise UnsupportedAllocationHandoffError(
                    "donate requires one exact SSA output allocation."
                )
            output, new_instance, descriptor = exact_matches[0]
            if new_instance in claimed_outputs:
                raise UnsupportedAllocationHandoffError(
                    "multiple donated inputs cannot claim one SSA output."
                )
            old_instance = input_tensor._coherence_instances[0]
            self._coherence_directory.require_local_write(
                old_instance,
                ExecutionToken(step),
            )
            if not self._allocation_registry.is_bound(old_instance):
                raise UnsupportedAllocationHandoffError(
                    "donate input has no allocation binding."
                )
            self._coherence_directory.require_local_handoff(
                old_instance,
                new_instance,
                ExecutionToken(step),
            )
            self._allocation_registry.require_handoff(
                old_instance,
                new_instance,
                descriptor,
            )
            if (
                isinstance(output, TransportTensor)
                and output is not input_tensor
                and output._coherence_instances
            ):
                raise UnsupportedAllocationHandoffError(
                    "donated output already has a coherence instance."
                )
            handoffs.append(
                _AllocationHandoffIntent(
                    input_tensor,
                    output,
                    old_instance,
                    new_instance,
                    descriptor,
                )
            )
            claimed_outputs.add(new_instance)
        return tuple(handoffs)

    def _commit_allocation_handoffs(
        self,
        handoffs: tuple[_AllocationHandoffIntent, ...],
        step: int,
    ) -> None:
        for handoff in handoffs:
            self._coherence_directory.handoff_local_write(
                handoff.old_instance,
                handoff.new_instance,
                ExecutionToken(step),
            )
            self._allocation_registry.handoff(
                handoff.old_instance,
                handoff.new_instance,
                handoff.output_descriptor,
            )
            output = handoff.output_tensor
            if isinstance(output, TransportTensor):
                if output is handoff.input_tensor:
                    output._handoff_coherence_instance(
                        handoff.old_instance,
                        handoff.new_instance,
                    )
                else:
                    output._bind_coherence_instances((handoff.new_instance,))

    def _step_transfer_cache_from_prepared_input(
        self,
        op_name: str,
        step: int,
        prepared_input: Any,
        op_result: Any,
    ) -> dict[int, Any]:
        result = self._normalize_local_plan_result(op_name, step, op_result)
        if isinstance(prepared_input, _PreparedOperatorInput):
            return dict(prepared_input.step_transfer_cache)
        if isinstance(prepared_input, ExecutionResult):
            if not self._result_requires_send(result):
                return {}
            if result.status in (
                ExecutionStatus.SKIPPED,
                ExecutionStatus.INACTIVE,
                ExecutionStatus.COMPLETED_WITHOUT_TENSOR,
                ExecutionStatus.SOURCE_EXHAUSTED,
            ):
                return {}
        raise TypeError(
            f"Local SEND for op={op_name} step={step} requires prepared input "
            "metadata from WAIT."
        )

    def _register_local_send(
        self,
        op_name: str,
        step: int,
        op_result: Any,
        step_transfer_cache: dict[int, Any],
    ) -> LocalSendToken:
        result = self._normalize_local_plan_result(op_name, step, op_result)
        send_task = None
        if self._result_requires_send(result):
            send_task = self._spawn_send_task(
                op_name,
                step,
                self._send_local_plan_result(
                    op_name,
                    step,
                    result,
                    step_transfer_cache,
                ),
            )
        else:
            self._close_step_coherence_intents(op_name, step)
        return LocalSendToken(
            op_name=op_name,
            step=step,
            result=result,
            send_task=send_task,
        )

    def _normalize_local_plan_result(
        self,
        op_name: str,
        step: int,
        op_result: Any,
    ) -> ExecutionResult:
        if not isinstance(op_result, ExecutionResult):
            return ExecutionResult(status=ExecutionStatus.SUCCESS, data=op_result)
        if op_result.status in (
            ExecutionStatus.SUCCESS,
            ExecutionStatus.SKIPPED,
            ExecutionStatus.INACTIVE,
            ExecutionStatus.COMPLETED_WITHOUT_TENSOR,
            ExecutionStatus.SOURCE_EXHAUSTED,
            ExecutionStatus.STALE_AFTER_RESTORE,
        ):
            return op_result
        raise RuntimeError(
            f"Unsupported ExecutionResult status {op_result.status!r} "
            f"for op={op_name} step={step}."
        )

    def _result_requires_send(self, result: ExecutionResult) -> bool:
        if result.status in (
            ExecutionStatus.INACTIVE,
            ExecutionStatus.SOURCE_EXHAUSTED,
            ExecutionStatus.STALE_AFTER_RESTORE,
        ):
            return False
        return True

    async def _send_local_plan_result(
        self,
        op_name: str,
        step: int,
        result: ExecutionResult,
        step_transfer_cache: dict[int, Any],
    ) -> ExecutionResult:
        await self._send_output(
            op_name,
            result,
            exec_step=step,
            step_transfer_cache=step_transfer_cache,
        )
        return result

    def _spawn_send_task(
        self,
        op_name: str,
        step: int,
        coro: Any,
    ) -> asyncio.Task[ExecutionResult]:
        task = asyncio.create_task(coro)
        self._send_tasks[task] = (op_name, step)

        def _done(done_task: asyncio.Task) -> None:
            self._send_tasks.pop(done_task, None)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc is not None:
                self._record_failure(
                    exc,
                    phase="SEND",
                    op_name=op_name,
                    step=step,
                )

        task.add_done_callback(_done)
        return task

    async def _local_plan_commit(
        self, op_name: str, step: int, send_token: LocalSendToken
    ) -> ExecutionResult:
        if send_token.op_name != op_name or send_token.step != step:
            raise RuntimeError(
                f"LocalSendToken mismatch for op={op_name} step={step}: "
                f"token_op={send_token.op_name} token_step={send_token.step}."
            )
        recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if recorder is None:
            return await self._commit_local_plan_result(
                op_name,
                step,
                send_token.result,
            )
        async with recorder.local_step_node(
            spec=LOCAL_STEP_COMMIT, op_name=op_name, step=step
        ):
            return await self._commit_local_plan_result(
                op_name, step, send_token.result
            )

    async def _local_plan_await_send(
        self, send_token: LocalSendToken
    ) -> LocalSendToken:
        recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if recorder is None:
            await self._await_send_registration(send_token)
            self._observe_local_send_registration(send_token)
            return send_token
        source_spmd_code = self._require_local_spmd_code()
        async with recorder.local_step_node(
            spec=LOCAL_STEP_AWAIT_SEND,
            op_name=send_token.op_name,
            step=send_token.step,
            source_spmd_code=source_spmd_code,
            source_rank=self.rank,
        ):
            await self._await_send_registration(send_token)
            self._observe_local_send_registration(send_token)
        return send_token

    async def _await_send_registration(
        self,
        send_token: LocalSendToken,
    ) -> None:
        if send_token.send_task is None:
            return
        await send_token.send_task

    def _observe_local_send_registration(self, send_token: LocalSendToken) -> None:
        send_task = send_token.send_task
        if send_task is None or not send_task.done():
            return
        if send_task.cancelled():
            raise RuntimeError(
                f"Local SEND task was cancelled before commit_save "
                f"op={send_token.op_name} step={send_token.step}."
            )
        exc = send_task.exception()
        if exc is not None:
            raise exc

    async def _execute_operator_prepare_input(self, op_name: str, step: int) -> Any:
        if self._source_blocks_step(op_name, step):
            return ExecutionResult(status=ExecutionStatus.SOURCE_EXHAUSTED)
        async with self._drain_cond:
            while self._drain_blocks_step(op_name, step):
                recorder = (
                    self._profiler_runtime.recorder
                    if self._profiler_runtime.enabled
                    else None
                )
                if recorder is None:
                    await self._drain_cond.wait()
                else:
                    assert self._drain_step_count is not None
                    async with recorder.worker_scheduler_span(
                        spec=WORKER_DRAIN_WAIT,
                        payload=recorder.worker_op_payload(
                            op_name,
                            step,
                            self._drain_step_count,
                        ),
                    ):
                        await self._drain_cond.wait()

        current_completed = self._completed_step_count.get(op_name, 0)
        if step < current_completed:
            self._barriers_declared.pop((op_name, step), None)
            return ExecutionResult(status=ExecutionStatus.STALE_AFTER_RESTORE)

        if self._mailbox.is_step_skipped(op_name, step):
            return ExecutionResult(status=ExecutionStatus.SKIPPED)

        dispatcher = self.op_manager.get_dispatcher(op_name)
        if not dispatcher.should_execute(self.rank, self._world_size, step):
            self._barriers_declared.pop((op_name, step), None)
            return ExecutionResult(status=ExecutionStatus.INACTIVE)

        meta = self.config.require_graph_view().operator(op_name).spec
        source_work_items: tuple[SourceWorkItem, ...] | None = None
        if meta.is_source is True and meta.requires_source_control_runtime:
            try:
                leases = await self.acquire_source_operator_work(op_name, step)
            except SourceExhaustedError:
                return ExecutionResult(status=ExecutionStatus.SOURCE_EXHAUSTED)
            source_work_items = tuple(lease.item for lease in leases)

        if self._uses_direct_local_input_fast_path(op_name):
            if not self._direct_local_slots_ready(op_name, step):
                raise ValueError(
                    "Direct local input fast path selected but inputs are not ready: "
                    f"op={op_name} step={step}."
                )
            prepared = self._assemble_direct_local_input(op_name, step)
            if isinstance(prepared, _PreparedOperatorInput):
                import dataclasses

                prepared = dataclasses.replace(
                    prepared,
                    source_work_items=source_work_items,
                )
            return prepared

        prepared = await self._assemble_input(op_name, step)
        if isinstance(prepared, _PreparedOperatorInput):
            import dataclasses

            prepared = dataclasses.replace(
                prepared,
                source_work_items=source_work_items,
            )
        return prepared

    async def _send_source_acknowledgements(
        self, acknowledgements: tuple[SourceWorkAcknowledgement, ...]
    ) -> None:
        if self._control_plane_handle is None:
            raise RuntimeError("source acknowledgement requires a control-plane handle")
        client = ControlPlaneClient(self._control_plane_handle)
        await client.acknowledge_source_work(acknowledgements)

    async def _commit_local_plan_result(
        self, op_name: str, step: int, result: ExecutionResult
    ) -> ExecutionResult:
        if result.status is ExecutionStatus.STALE_AFTER_RESTORE:
            return result
        if (
            self._source_blocks_step(op_name, step)
            and result.status is not ExecutionStatus.SOURCE_EXHAUSTED
        ):
            return ExecutionResult(status=ExecutionStatus.SOURCE_EXHAUSTED)
        self._finished_plan_steps[op_name][step] = result
        await self._commit_finished_steps(op_name)
        return result

    async def _commit_finished_steps(self, op_name: str) -> None:
        pending = self._finished_plan_steps[op_name]
        while self._completed_step_count[op_name] in pending:
            step = self._completed_step_count[op_name]
            result = pending.pop(step)
            self._spawn_background_task(self._return_input_coherence(op_name, step))
            await self._mailbox.clear_step(op_name, step)
            self._clear_direct_local_step(op_name, step)
            self._barriers_declared.pop((op_name, step), None)
            if (
                result.status is ExecutionStatus.SUCCESS
                and op_name in self._source_control_operator_names
            ):
                leases = self._source_scope.finish(op_name, step)
                await self._source_acknowledgements.submit(
                    tuple(
                        SourceWorkAcknowledgement(lease.item.key, lease.lease_id)
                        for lease in leases
                    )
                )
            if result.status is ExecutionStatus.SOURCE_EXHAUSTED:
                await self._forward_control_requests(result.control_requests)
                self._record_source_terminal(op_name, step)
                self._discard_source_pipeline(op_name, step)
                self._notify_source_terminal()
                return
            await self._advance_step(op_name, result=result)

    def _discard_source_pipeline(self, op_name: str, terminal_step: int) -> None:
        finished = self._finished_plan_steps[op_name]
        for step in tuple(finished):
            if step >= terminal_step:
                finished.pop(step)
        pending_steps_by_name = (
            self._pending_blocked[op_name],
            self._pending_future_steps[op_name],
            self._ready_pending_pipeline_steps[op_name],
            self._pending_async_lag_steps[op_name],
            self._pending_phase_order_steps[op_name],
        )
        for pending_steps in pending_steps_by_name:
            pending_steps[:] = [step for step in pending_steps if step < terminal_step]
        self._admission_state.discard_at_or_after(op_name, terminal_step)
        running_steps = self._running_plan_steps[op_name]
        for task, (task_op_name, task_step) in tuple(self._plan_task_steps.items()):
            if (
                task_op_name == op_name
                and task_step >= terminal_step
                and not task.done()
            ):
                task.cancel()
        for intent_op, intent_step in tuple(self._coherence_output_intents):
            if intent_op == op_name and intent_step >= terminal_step:
                self._abort_step_coherence_intents(
                    intent_op,
                    intent_step,
                    RuntimeError(
                        f"source pipeline discarded op={intent_op} step={intent_step}."
                    ),
                )
        running_steps.difference_update(
            step for step in tuple(running_steps) if step >= terminal_step
        )
        mailbox_ready_steps = self._mailbox_ready_steps[op_name]
        mailbox_ready_steps.difference_update(
            step for step in tuple(mailbox_ready_steps) if step >= terminal_step
        )
        direct_mailbox_ready_steps = self._direct_local_ready_steps.get(op_name)
        if direct_mailbox_ready_steps is not None:
            direct_mailbox_ready_steps.difference_update(
                step
                for step in tuple(direct_mailbox_ready_steps)
                if step >= terminal_step
            )
        self._in_op_step_queue.difference_update(
            {
                (queued_op, step)
                for queued_op, step in self._in_op_step_queue
                if queued_op == op_name and step >= terminal_step
            }
        )
        for key in tuple(self._local_input_origins):
            if key[0] == op_name and key[1] >= terminal_step:
                del self._local_input_origins[key]
        for task, key in tuple(self._send_tasks.items()):
            if key[0] == op_name and key[1] >= terminal_step and not task.done():
                task.cancel()
        self._spawn_background_task(
            self._delivery_runtime.cancel_matching(
                lambda schedule_key: (
                    schedule_key[3] == op_name and schedule_key[2] >= terminal_step
                )
            )
        )

    async def _return_input_coherence(self, op_name: str, step: int) -> None:
        grants = self._coherence_input_grants.pop((op_name, step), {})
        alias_dependencies = self._coherence_alias_dependencies.pop(
            (op_name, step),
            set(),
        )
        input_tensors = self._coherence_input_tensors.pop((op_name, step), set())
        self._coherence_input_bindings.pop((op_name, step), None)
        for key in tuple(self._local_input_origins):
            if key[:2] == (op_name, step):
                del self._local_input_origins[key]

        if not grants and not alias_dependencies and not input_tensors:
            return

        plan = InputCoherenceReleasePlan.create(
            grants=tuple(grants.values()),
            alias_dependencies=alias_dependencies,
            revoked_instances=frozenset(self._coherence_revocations),
            parent_handlers=self._predecessor_handlers,
            tensors=input_tensors,
        )
        await InputCoherenceRelease(
            directory=self._coherence_directory,
            admission_runtime=self._coherence_admission_runtime,
            parent_return_runtime=self._parent_grant_return_runtime,
            allocation_registry=self._allocation_registry,
            plan=plan,
        ).complete()

    def _operator_user_callable_spec(
        self,
        op_name: str,
        step: int,
        input_data: Any,
        source_work_items: tuple[SourceWorkItem, ...] | None = None,
    ) -> UserCallableSpec:
        graph_view = self.config.require_graph_view()
        compiled_operator = graph_view.operator(op_name)
        meta = compiled_operator.spec
        if isinstance(input_data, dict):
            operator_args: tuple[Any, ...] = ()
            operator_kwargs = input_data
        else:
            operator_args = (input_data,)
            operator_kwargs = {}
        operator_is_coroutine_function = self.op_manager.operator_is_coroutine_function(
            op_name
        )
        source_kwargs: dict[str, Any] = {}
        if source_work_items is not None:
            source_kwargs["source_work_items"] = source_work_items

        def execute_regular_operator() -> Any:
            return self.op_manager.execute(
                op_name,
                step,
                *operator_args,
                **source_kwargs,
                **operator_kwargs,
            )

        async def execute_coroutine_operator() -> Any:
            return await self.op_manager.execute_async(
                op_name,
                step,
                *operator_args,
                **source_kwargs,
                **operator_kwargs,
            )

        return UserCallableSpec(
            callable_id=compiled_operator.key.op_id,
            fn=(
                execute_coroutine_operator
                if operator_is_coroutine_function
                else execute_regular_operator
            ),
            resource_id=self._state_resource_id(op_name),
            operator_lane=(
                _EXECUTION_DEFAULT_LANE
                if meta.runtime_policy.schedule.lane is None
                else meta.runtime_policy.schedule.lane
            ),
            operator_is_coroutine_function=operator_is_coroutine_function,
            operator_step=step,
            object_scope=SPMDObjectScope(
                spmd_id=compiled_operator.key.spmd_id,
                rank=self.rank,
            ),
            graph_id=graph_view.graph_id,
            compiled_operator=compiled_operator,
        )

    def _operator_input_transfer_codes(
        self, op_name: str
    ) -> tuple[Any, dict[str, list[int | None]]]:
        meta = self.op_manager.get_operator_spec(op_name)
        input_types = self.op_manager.get_operator_input_types(op_name)
        input_transfer_codes = self.op_manager.get_input_transfer_codes(op_name)
        parameter_codes: dict[str, list[int | None]] = {}
        for parameter_name in input_types:
            if parameter_name in self.op_manager.get_static_args():
                continue
            paths = meta.input_param_map.get(parameter_name, (parameter_name,))
            codes: list[int | None] = []
            for path in paths:
                codes.append(input_transfer_codes.get(f"{op_name}.{path}"))
            parameter_codes[parameter_name] = codes
        return meta, parameter_codes

    async def _assemble_input(self, op_name: str, step: int) -> Any:
        background_prepare = self._should_prepare_inputs_on_ready(op_name)
        static_args = self.op_manager.get_static_args()
        input_types = self.op_manager.get_operator_input_types(op_name)
        expected_state_names = self.op_manager.get_expected_control_arg_names(op_name)
        all_states = await self._mailbox.get_all_control_arguments(op_name, step)
        state_args = {
            name: all_states[name]
            for name in expected_state_names
            if name in all_states
        }

        meta, parameter_codes = self._operator_input_transfer_codes(op_name)
        transfer_codes = list(
            dict.fromkeys(
                code
                for codes in parameter_codes.values()
                for code in codes
                if code is not None
            )
        )

        async def snapshot_tensors() -> list[Any]:
            return await self._mailbox.get_tensor_shard_views_and_arrivals_many(
                op_name,
                transfer_codes,
                step,
            )

        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if profiler_recorder is not None and background_prepare:
            async with profiler_recorder.prepare_snapshot(op_name=op_name, step=step):
                tensor_input_list = await snapshot_tensors()
        else:
            tensor_input_list = await snapshot_tensors()
        all_source_steps = [item[0] for item in tensor_input_list]
        all_shards_list = [item[1] for item in tensor_input_list]
        all_arrivals_list = [item[2] for item in tensor_input_list]
        transfer_to_source_step = dict(
            zip(transfer_codes, all_source_steps, strict=True)
        )
        transfer_to_shards = dict(zip(transfer_codes, all_shards_list, strict=True))  # type: ignore
        transfer_to_arrivals = dict(zip(transfer_codes, all_arrivals_list, strict=True))  # type: ignore

        for transfer_code in transfer_codes:
            if transfer_to_shards[transfer_code]:
                continue
            assemble_plan = self._compiled_assemble_plan(op_name, transfer_code)
            if assemble_plan.completed_without_tensor(
                transfer_to_arrivals[transfer_code]
            ):
                return ExecutionResult(status=ExecutionStatus.COMPLETED_WITHOUT_TENSOR)

        async def assemble_tensors() -> list[Any]:
            assemble_tasks = [
                self._assemble_tensor_for_prepare_policy(
                    op_name,
                    transfer_code,
                    step,
                    transfer_to_shards[transfer_code],
                    transfer_to_arrivals[transfer_code],
                    background_prepare=background_prepare,
                )
                for transfer_code in transfer_codes
            ]
            return await asyncio.gather(*assemble_tasks)

        if profiler_recorder is None:
            assembled_list = await assemble_tensors()
        else:
            async with profiler_recorder.worker_assemble_input(
                op_name=op_name,
                step=step,
                transfer_count=len(transfer_codes),
                state_count=len(state_args),
            ):
                assembled_list = await assemble_tensors()
        for transfer_code, assembled in zip(
            transfer_codes, assembled_list, strict=True
        ):
            arrivals = transfer_to_arrivals[transfer_code]
            if arrivals:
                sender_id, sender_rank = next(iter(arrivals))
                self._local_input_origins[(op_name, step, transfer_code)] = (
                    sender_id,
                    sender_rank,
                )
            if isinstance(assembled, TransportTensor):
                assembled.source_step = transfer_to_source_step[transfer_code]
                instances = tuple(
                    CoherenceInstance(grant.subject, grant.epoch)
                    for grant in self._coherence_input_grants.get(
                        (op_name, step), {}
                    ).values()
                    if grant.route_code == transfer_code
                )
                if instances:
                    assembled._bind_coherence_instances(instances)
                    if len(instances) == 1:
                        self._allocation_registry.bind(
                            instances[0],
                            runtime_allocation_descriptor(assembled),
                        )
        transfer_to_assembled = dict(zip(transfer_codes, assembled_list, strict=True))  # type: ignore
        tensor_bindings = self._prepare_tensor_bindings(
            op_name,
            step,
            meta,
            parameter_codes,
            transfer_to_assembled,
        )

        input_kwargs = {}
        for p_name, p_codes in parameter_codes.items():
            if not p_codes or p_codes[0] is None:
                continue
            if any(code is None for code in p_codes):
                raise ValueError(
                    f"Operator {op_name!r} input {p_name!r} mixes tensor and "
                    "non-tensor leaves; declare separate parameters."
                )

            tensor_codes = [code for code in p_codes if code is not None]
            p_leaves = [transfer_to_assembled[code] for code in tensor_codes]
            tdef = meta.input_treedefs.get(p_name)
            runtime_treedef = None if tdef is None else tdef.materialize()
            if runtime_treedef is not None:
                input_kwargs[p_name] = tree_unflatten(runtime_treedef, p_leaves)
            else:
                input_kwargs[p_name] = p_leaves[0]

        final_args = {}
        for p_name in input_types:
            if p_name in static_args:
                final_args[p_name] = static_args[p_name]
        final_args.update(state_args)
        final_args.update(input_kwargs)
        return _PreparedOperatorInput(
            value=final_args,
            step_transfer_cache=dict(transfer_to_assembled),
            tensor_bindings=tensor_bindings,
        )

    def _assemble_direct_local_input(
        self, op_name: str, step: int
    ) -> _PreparedOperatorInput:
        static_args = self.op_manager.get_static_args()
        input_types = self.op_manager.get_operator_input_types(op_name)
        meta, parameter_codes = self._operator_input_transfer_codes(op_name)
        mailbox = self._get_operator_mailbox(op_name)
        if not isinstance(mailbox, DirectLocalMailbox):
            raise ValueError(
                f"Direct local assemble invoked for operator without DirectLocalMailbox: op={op_name}"
            )
        step_slots = mailbox.get_step_slots(step)

        transfer_to_assembled = {
            transfer_code: slot.tensor for transfer_code, slot in step_slots.items()
        }
        for transfer_code, slot in step_slots.items():
            self._local_input_origins[(op_name, step, transfer_code)] = (
                slot.sender_id,
                slot.rank,
            )
        tensor_bindings = self._prepare_tensor_bindings(
            op_name,
            step,
            meta,
            parameter_codes,
            transfer_to_assembled,
        )
        input_kwargs = {}
        for p_name, p_codes in parameter_codes.items():
            if not p_codes or p_codes[0] is None:
                continue
            if any(code is None for code in p_codes):
                raise ValueError(
                    f"Operator {op_name!r} input {p_name!r} mixes tensor and "
                    "non-tensor leaves; declare separate parameters."
                )
            tensor_codes = [code for code in p_codes if code is not None]
            missing = [
                code for code in tensor_codes if code not in transfer_to_assembled
            ]
            if missing:
                raise ValueError(
                    "Direct local input fast path cannot satisfy operator input: "
                    f"op={op_name} step={step} param={p_name} missing={missing}."
                )
            p_leaves = [transfer_to_assembled[code] for code in tensor_codes]
            tdef = meta.input_treedefs.get(p_name)
            runtime_treedef = None if tdef is None else tdef.materialize()
            if runtime_treedef is not None:
                input_kwargs[p_name] = tree_unflatten(runtime_treedef, p_leaves)
            else:
                input_kwargs[p_name] = p_leaves[0]

        final_args = {}
        for p_name in input_types:
            if p_name in static_args:
                final_args[p_name] = static_args[p_name]
        final_args.update(input_kwargs)
        return _PreparedOperatorInput(
            value=final_args,
            step_transfer_cache=dict(transfer_to_assembled),
            tensor_bindings=tensor_bindings,
        )

    def _prepare_tensor_bindings(
        self,
        op_name: str,
        step: int,
        meta: OperatorSpec,
        parameter_codes: Mapping[str, list[int | None]],
        transfer_to_assembled: Mapping[int, Any],
    ) -> tuple[_PreparedTensorBinding, ...]:
        grants = self._coherence_input_grants.setdefault((op_name, step), {})
        bindings: list[_PreparedTensorBinding] = []
        for parameter, codes in parameter_codes.items():
            ports = tuple(meta.input_param_map.get(parameter, (parameter,)))
            if len(ports) != len(codes):
                raise CoherenceStateTransitionError(
                    f"operator {op_name!r} input port metadata does not match "
                    f"transfer codes for parameter {parameter!r}."
                )
            for port, code in zip(ports, codes, strict=True):
                if code is None:
                    continue
                tensor = transfer_to_assembled[code]
                if not isinstance(tensor, TransportTensor):
                    continue
                route = self.op_manager.get_predecessor_route(op_name, code)
                if route.local_only:
                    self._ensure_local_input_grant(op_name, step, code, tensor)
                route_grants = tuple(
                    grant for grant in grants.values() if grant.route_code == code
                )
                if not route_grants:
                    raise CoherenceStateTransitionError(
                        f"tensor input port {port!r} has no coherence grant."
                    )
                bindings.append(
                    _PreparedTensorBinding(
                        port=port,
                        transfer_code=code,
                        tensor=tensor,
                        grants=route_grants,
                    )
                )
        result = tuple(bindings)
        self._coherence_input_bindings[(op_name, step)] = result
        return result

    async def _assemble_tensor_for_prepare_policy(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        shards: dict[Any, list[Any]],
        arrivals: dict[Any, dict[str, Any]],
        *,
        background_prepare: bool,
    ) -> "TransportTensor":
        r"""Assemble tensor shards according to the operator prepare policy.

        Execution Flow:
        +-------------------------------------------------------------+
        |                _assemble_tensor_for_prepare_policy          |
        +-------------------------------------------------------------+
                                       |
                       [background_prepare == True?]
                                      / \
                                     /   \
                             (Yes)  /     \  (No: sync operator)
                                   v       v
            +------------------------------+ +-----------------------+
            |  _input_prepare_runtime      | |  _assemble_tensor     |
            |  (PriorityThreadExecutor)    | |  (Event loop inline)  |
            +------------------------------+ +-----------------------+
                           |                             |
                           +-------------> <-------------+
                                          |
                                          v
                           +------------------------------+
                           |  _assemble_tensor_sync       |
                           |  (Numba concatenate/assemble)|
                           +------------------------------+
        """
        if not background_prepare:
            return await self._assemble_tensor(
                op_name, transfer_code, step, shards, arrivals
            )
        return await self._input_prepare_runtime.run_assemble(
            op_name,
            step,
            lambda: self._assemble_tensor_sync_traced(
                op_name,
                transfer_code,
                step,
                shards,
                arrivals,
            ),
        )

    def _get_values(self, obj: Any) -> Any:
        if isinstance(obj, (TransportTensor, TransportShardView)):
            return obj.flat_values
        return obj

    def _get_routing_indices(self, obj: Any) -> Any | None:
        if isinstance(obj, (TransportTensor, TransportShardView)):
            return obj.routing_indices_materialized
        return None

    def _flatten_shards(
        self, shards: dict[tuple[str, int], list[Any]], index: int = 0
    ) -> list[Any]:
        """
        Extract the N-th tensor from each multi-tensor packet.
        """
        tensors = []
        for sender_res in shards.values():
            if index < len(sender_res):
                tensors.append(sender_res[index])
        return tensors

    def _flatten_shards_with_ranks(
        self, shards: dict[Any, list[Any]], index: int = 0
    ) -> list[tuple[Any, Any]]:
        """Flatten shards and include their sender keys (ranks)."""
        flat = []
        for key in sorted(shards.keys()):
            tensor_list = shards[key]
            if index < len(tensor_list):
                flat.append((key, tensor_list[index]))
        return flat

    def _normalize_and_cat_shards(self, tensor_list: list[Any], dim: int = 0) -> Any:
        backend = get_backend()
        count = len(tensor_list)
        if count == 0:
            raise ValueError("Cannot concatenate an empty shard list.")

        if count == 1:
            return backend.as_tensor(force_writable(tensor_list[0]))

        # Find the archetype (first non-empty tensor with proper rank)
        prototype = None
        for t in tensor_list:
            if all(extent > 0 for extent in backend.shape(t)):
                prototype = t
                break

        if prototype is None:
            proto_shape = tuple(backend.shape(tensor_list[0]))
            proto_dtype = backend.dtype(tensor_list[0])
            for t in tensor_list[1:]:
                if tuple(backend.shape(t)) != proto_shape:
                    raise ValueError(
                        "Cannot concatenate all-empty shards with different shapes."
                    )
                if backend.dtype(t) != proto_dtype:
                    raise ValueError(
                        "Cannot concatenate all-empty shards with different dtypes."
                    )
            return backend.cat(tensor_list, dim=dim)

        # Check for mismatch against prototype to decide if we need the slow-path
        needs_normalization = False
        proto_shape = backend.shape(prototype)
        proto_dtype = backend.dtype(prototype)
        for t in tensor_list:
            if backend.shape(t) != proto_shape or backend.dtype(t) != proto_dtype:
                needs_normalization = True
                break

        # Homogeneous Fast Path: If no mismatches found, skip alignment loop
        if not needs_normalization:
            return backend.cat(tensor_list, dim=dim)

        proto_shape = backend.shape(prototype)
        proto_dtype = backend.dtype(prototype)
        proto_ndims = len(proto_shape)

        normalized = []
        for t in tensor_list:
            t_shape = backend.shape(t)
            if len(t_shape) != proto_ndims:
                new_shape = list(proto_shape)
                new_shape[dim] = 0
                normalized.append(backend.zeros(tuple(new_shape), dtype=proto_dtype))
            elif backend.dtype(t) != proto_dtype:
                normalized.append(backend.cast(t, proto_dtype))
            else:
                normalized.append(t)

        return backend.cat(normalized, dim=dim)

    # =========================================================================
    # Shard Assembly Helpers (shared by all _assemble_* functions)
    # =========================================================================

    def _merge_index_descriptors(
        self,
        shard_list: list[Any],
        property: str = "origin_indices",
        *,
        require_all_if_any: bool = False,
    ) -> Optional["IndexDescriptor"]:
        """Merge IndexDescriptors from shards only when ALL shards carry one."""
        descriptors: list[IndexDescriptor] = [
            getattr(s, property) for s in shard_list if getattr(s, property) is not None
        ]
        if not descriptors:
            return None
        if len(descriptors) != len(shard_list):
            if require_all_if_any:
                raise ValueError(
                    f"Partial {property} metadata cannot be merged; "
                    f"got {len(descriptors)} descriptors for {len(shard_list)} shards."
                )
            return None
        return IndexDescriptor.concatenate(descriptors)

    def _merge_materialized_index_descriptors(
        self,
        shards: list[Any],
        property: str = "origin_indices_materialized",
    ) -> Any | None:
        """Merge materialized indices from shards."""
        if not any(
            hasattr(s, property) and getattr(s, property) is not None for s in shards
        ):
            return None
        index_list = []
        for s in shards:
            idx = getattr(s, property, None)
            if idx is None and len(s) > 0:
                raise ValueError(
                    f"Shard {s} is missing materialized index descriptor {property}"
                )
            if idx is None:
                raise ValueError(
                    f"Shard {s} is missing {property}; partial metadata cannot be merged."
                )
            index_list.append(idx)
        return self._normalize_and_cat_shards(index_list, dim=0)

    def _maybe_merge_receipts(self, shards: list[Any]) -> Optional["SymmetryReceipt"]:
        """Merge SymmetryReceipts from input shards into a single receipt."""
        receipts = [s.receipt for s in shards]
        if all(r is not None for r in receipts) and receipts:
            lengths = [self._routing_entity_count(s) for s in shards]
            return SymmetryReceipt.concatenate(receipts, lengths)  # type: ignore[arg-type]
        return None

    def _routing_entity_count(self, shard: Any) -> int:
        backend = get_backend()
        if getattr(shard, "is_ragged", False):
            splits = shard.nested_splits
            if not splits:
                raise ValueError("Ragged shard is missing nested_splits.")
            return int(backend.shape(splits[0])[0]) - 1
        values = self._get_values(shard)
        shape = backend.shape(values)
        if not shape:
            return 1
        return int(shape[0])

    def _merge_nested_splits(self, shards: list[Any]) -> list[Any]:
        if not any(getattr(s, "is_ragged", False) for s in shards):
            return []
        if not all(getattr(s, "is_ragged", False) for s in shards):
            raise ValueError("Cannot merge mixed dense and ragged shards.")

        backend = get_backend()
        level_count = len(shards[0].nested_splits)
        if level_count == 0:
            raise ValueError("Ragged shard is missing nested_splits.")
        for shard in shards[1:]:
            if len(shard.nested_splits) != level_count:
                raise ValueError(
                    "Cannot merge ragged shards with different nested level counts."
                )

        merged_splits = []
        for level in range(level_count):
            lengths = []
            for shard in shards:
                splits = shard.nested_splits[level]
                lengths.append(splits[1:] - splits[:-1])
            merged_lengths = self._normalize_and_cat_shards(lengths, dim=0)
            zero = backend.zeros((1,), dtype=backend.dtype(merged_lengths))
            merged_splits.append(
                backend.cat([zero, backend.cumsum(merged_lengths, dim=0)], dim=0)
            )
        return merged_splits

    def _wrap_shard_for_operator(self, shard: Any) -> TransportTensor:
        if isinstance(shard, TransportTensor):
            return shard
        return TransportTensor(
            self._get_values(shard),
            nested_splits=shard.nested_splits,
            routing_indices=shard.routing_indices,
            origin_indices=shard.origin_indices,
            global_shape=shard.global_shape,
            receipt=shard.receipt,
            routing_axes=shard.routing_axes,
        )

    def _describe_shard_flow_for_error(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        shards_dict: dict[Any, list[Any]],
        filtered_shards: list[Any],
        input_layout: "PhysicalLayout | None",
        assemble_plan: CompiledAssemblePlan,
    ) -> str:
        backend = get_backend()
        logical_tuuid = assemble_plan.transfer.logical_tuuid
        source_spmd_id = assemble_plan.source_spmd_id
        source_keys = [
            key
            for key in shards_dict
            if isinstance(key, tuple) and len(key) == 2 and key[0] == source_spmd_id
        ]

        shard_lines = []
        for i, shard in enumerate(filtered_shards[:16]):
            values = self._get_values(shard)
            shape = tuple(backend.shape(values))
            origin = shard.origin_indices_materialized
            routing = self._get_routing_indices(shard)
            if origin is not None and int(backend.shape(origin)[0]) > 0:
                origin_np = backend.to_numpy(origin)
                origin_text = (
                    f"origin_len={len(origin_np)}, "
                    f"origin_min={int(origin_np.min())}, "
                    f"origin_max={int(origin_np.max())}"
                )
            else:
                origin_text = "origin_len=0"
            if routing is not None and int(backend.shape(routing)[0]) > 0:
                routing_np = backend.to_numpy(routing)
                routing_text = (
                    f"routing_len={len(routing_np)}, "
                    f"routing_min={int(routing_np.min())}, "
                    f"routing_max={int(routing_np.max())}"
                )
            else:
                routing_text = "routing_len=0"
            shard_lines.append(
                f"shard[{i}]: shape={shape}, global_shape={shard.global_shape}, "
                f"{origin_text}, {routing_text}, ragged_levels={len(shard.nested_splits)}"
            )

        shard_summary = "; ".join(shard_lines) if shard_lines else "none"
        return (
            f"op={op_name}, transfer_code={transfer_code}, "
            f"logical_tuuid={logical_tuuid}, source_spmd={source_spmd_id}, "
            f"step={step}, "
            f"consumer_rank={self.rank}, consumer={_describe_layout_for_error(input_layout)}, "
            f"mailbox_keys={list(shards_dict.keys())}, "
            f"source_mailbox_keys={source_keys}, "
            f"source_layout={_describe_layout_for_error(assemble_plan.source_layout)}, "
            f"filtered_shards={len(filtered_shards)}, "
            f"filtered_detail=[{shard_summary}]"
        )

    # =========================================================================
    # Tensor Assembly
    # =========================================================================

    async def _assemble_tensor(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        shards: dict[Any, list[Any]],
        arrivals: dict[Any, dict[str, Any]],
    ) -> "TransportTensor":
        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if profiler_recorder is None:
            return self._assemble_tensor_sync(
                op_name, transfer_code, step, shards, arrivals
            )
        shard_count = (
            sum(len(items) for items in shards.values())
            if _profiler_enabled(self)
            else 0
        )
        async with profiler_recorder.worker_assemble_tensor(
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=self._compiled_assemble_plan(
                op_name, transfer_code
            ).transfer.logical_tuuid,
            step=step,
            sender_count=len(shards),
            shard_count=shard_count,
        ):
            return self._assemble_tensor_sync(
                op_name, transfer_code, step, shards, arrivals
            )

    def _assemble_tensor_sync_traced(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        shards: dict[Any, list[Any]],
        arrivals: dict[Any, dict[str, Any]],
    ) -> "TransportTensor":
        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if profiler_recorder is None:
            return self._assemble_tensor_sync(
                op_name, transfer_code, step, shards, arrivals
            )
        shard_count = (
            sum(len(items) for items in shards.values())
            if _profiler_enabled(self)
            else 0
        )
        with profiler_recorder.worker_assemble_tensor_sync(
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=self._compiled_assemble_plan(
                op_name, transfer_code
            ).transfer.logical_tuuid,
            step=step,
            sender_count=len(shards),
            shard_count=shard_count,
        ):
            return self._assemble_tensor_sync(
                op_name, transfer_code, step, shards, arrivals
            )

    def _assemble_tensor_sync(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        shards: dict[Any, list[Any]],
        arrivals: dict[Any, dict[str, Any]],
    ) -> "TransportTensor":
        assemble_plan = self._compiled_assemble_plan(op_name, transfer_code)
        input_layout = assemble_plan.target_layout

        if assemble_plan.assembles_to_dynamic:
            return self._assemble_to_dynamic_tensor(
                op_name,
                transfer_code,
                step,
                shards,
                arrivals,
                assemble_plan=assemble_plan,
            )

        if assemble_plan.has_dynamic_predecessor:
            return self._assemble_dynamic_to_static_tensor(
                op_name,
                transfer_code,
                step,
                shards,
                input_layout,
                assemble_plan=assemble_plan,
            )
        return self._assemble_static_to_static_tensor(
            op_name,
            transfer_code,
            step,
            shards,
            input_layout,
            assemble_plan=assemble_plan,
        )

    def _compiled_assemble_plan(
        self,
        op_name: str,
        transfer_code: int,
    ) -> CompiledAssemblePlan:
        key = (op_name, transfer_code)
        plan = self._assemble_plan_cache.get(key)
        if plan is None:
            plan = compile_assemble_plan(
                op_manager=self.op_manager,
                op_name=op_name,
                transfer_code=transfer_code,
            )
            self._assemble_plan_cache[key] = plan
        return plan

    def _filter_shards_by_routing(
        self,
        *,
        step: int,
        shards_dict: dict[Any, list[Any]],
        assemble_plan: CompiledAssemblePlan,
    ) -> list[Any]:
        return assemble_plan.filter_shards(
            op_manager=self.op_manager,
            worker_rank=self.rank,
            step=step,
            shards_dict=shards_dict,
            flatten_shards=self._flatten_shards,
        )

    def _routing_miss_tensor(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        arrivals: dict[Any, dict[str, Any]],
    ) -> "TransportTensor":
        assemble_plan = self._compiled_assemble_plan(op_name, transfer_code)
        logical_tensor = assemble_plan.logical_tensor
        tensor_identity = (
            f"logical_tensor={logical_tensor.name!r} "
            f"logical_tuuid={logical_tensor.logical_tuuid}"
        )
        expected_values = {
            int(meta.get("expected_count", 0)) for meta in arrivals.values()
        }
        if len(expected_values) != 1:
            raise ValueError(
                f"[Fluxway Protocol Error] Conflicting expected_count metadata "
                f"for routing-miss tensor op={op_name} "
                f"transfer_code={transfer_code} {tensor_identity} "
                f"step={step}: {sorted(expected_values)}."
            )
        expected = next(iter(expected_values), 0)
        if expected <= 0:
            raise ValueError(
                f"[Fluxway Protocol Error] Missing expected_count for routing-miss "
                f"tensor op={op_name} transfer_code={transfer_code} "
                f"{tensor_identity} step={step}."
            )
        if len(arrivals) != expected:
            raise ValueError(
                f"[Fluxway Protocol Error] Incomplete routing-miss tensor "
                f"arrival metadata op={op_name} transfer_code={transfer_code} "
                f"{tensor_identity} step={step}: "
                f"arrived={len(arrivals)} expected={expected}."
            )

        routing_miss_arrivals = {}
        for key, meta in arrivals.items():
            raw_tensor_kind = meta.get("tensor_kind")
            if raw_tensor_kind is None:
                raise ValueError(
                    f"[Fluxway Protocol Error] Tensor arrival lacks tensor_kind "
                    f"op={op_name} transfer_code={transfer_code} "
                    f"{tensor_identity} step={step} sender={key}."
                )
            tensor_kind = TensorEntryKind(int(raw_tensor_kind))
            if tensor_kind is TensorEntryKind.ROUTING_MISS:
                routing_miss_arrivals[key] = meta
                continue
            if tensor_kind is TensorEntryKind.COMPLETED_WITHOUT_TENSOR:
                continue
            raise ValueError(
                f"[Fluxway Protocol Error] Tensor kind {tensor_kind.name} has no "
                f"business shard op={op_name} transfer_code={transfer_code} "
                f"{tensor_identity} step={step} sender={key}."
            )
        if not routing_miss_arrivals:
            raise ValueError(
                f"[Fluxway Protocol Error] Transfer completed without tensor but "
                f"reached dynamic assembly op={op_name} transfer_code={transfer_code} "
                f"{tensor_identity} step={step}."
            )

        first_meta = next(iter(routing_miss_arrivals.values()))
        first_schema = tuple(first_meta.get("payload_schema") or ())
        if not first_schema:
            raise ValueError(
                f"[Fluxway Protocol Error] Routing-miss tensor lacks payload schema "
                f"op={op_name} transfer_code={transfer_code} "
                f"{tensor_identity} step={step}."
            )
        for key, meta in routing_miss_arrivals.items():
            schema = tuple(meta.get("payload_schema") or ())
            if schema != first_schema:
                raise ValueError(
                    f"[Fluxway Protocol Error] Routing-miss schema mismatch "
                    f"op={op_name} transfer_code={transfer_code} "
                    f"{tensor_identity} step={step} sender={key}."
                )

        value_schema = [
            s for s in first_schema if int(s["role"]) == int(PayloadRole.VALUES)
        ]
        if len(value_schema) != 1:
            raise ValueError(
                f"[Fluxway Protocol Error] Routing-miss tensor requires exactly "
                f"one VALUES payload schema op={op_name} "
                f"transfer_code={transfer_code} {tensor_identity} "
                f"step={step}; "
                f"got {len(value_schema)}."
            )
        shape = list(value_schema[0]["shape"])
        if not shape:
            shape = [0]
        shape[0] = 0
        from fluxway.common.types.tensor import SymmetryReceipt

        backend = get_backend()
        values = backend.zeros(tuple(shape), dtype=value_schema[0]["dtype"])
        global_shapes = {
            None if meta.get("global_shape") is None else tuple(meta["global_shape"])
            for meta in routing_miss_arrivals.values()
        }
        if len(global_shapes) != 1:
            raise ValueError(
                f"[Fluxway Protocol Error] Conflicting global_shape metadata for "
                f"routing-miss tensor op={op_name} transfer_code={transfer_code} "
                f"{tensor_identity} step={step}: {global_shapes}."
            )
        global_shape = next(iter(global_shapes))
        sender_world_sizes = {
            int(meta.get("sender_world_size", 0)) for meta in arrivals.values()
        }
        if len(sender_world_sizes) != 1:
            raise ValueError(
                f"[Fluxway Protocol Error] Conflicting sender_world_size metadata "
                f"for routing-miss tensor op={op_name} "
                f"transfer_code={transfer_code} {tensor_identity} "
                f"step={step}: {sorted(sender_world_sizes)}."
            )
        sender_world_size = next(iter(sender_world_sizes), 0)
        if sender_world_size <= 0:
            raise ValueError(
                f"[Fluxway Protocol Error] Missing sender_world_size for routing-miss "
                f"tensor op={op_name} transfer_code={transfer_code} "
                f"{tensor_identity} step={step}."
            )
        receipt = SymmetryReceipt(
            sender_world_size,
            backend.zeros((0,), dtype=backend.int32()),
        )
        return TransportTensor(values, global_shape=global_shape, receipt=receipt)

    def _assemble_static_to_static_tensor(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        shards: dict[Any, list[Any]],
        input_layout: "PhysicalLayout",
        *,
        assemble_plan: CompiledAssemblePlan,
    ) -> "TransportTensor":
        """Assemble static-to-static tensor using exact physical shards."""
        logical_tuuid = assemble_plan.transfer.logical_tuuid
        shard_count = (
            sum(len(items) for items in shards.values())
            if _profiler_enabled(self)
            else 0
        )
        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_FILTER_SHARDS,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(shards),
            second_count=shard_count,
        ):
            shard_list = self._filter_shards_by_routing(
                step=step,
                shards_dict=shards,
                assemble_plan=assemble_plan,
            )

        if not shard_list:
            flow_detail = self._describe_shard_flow_for_error(
                op_name,
                transfer_code,
                step,
                shards,
                shard_list,
                input_layout,
                assemble_plan,
            )
            raise ValueError(
                f"[Fluxway Assembly Error] Static-to-static assembly for op "
                f"'{op_name}' (transfer_code={transfer_code}, "
                f"logical_tuuid={logical_tuuid}, rank {self.rank}, step {step}) "
                f"received no routed shards. Flow detail: {flow_detail}"
            )

        if len(shard_list) == 1:
            return self._wrap_shard_for_operator(shard_list[0])

        flat_values = [self._get_values(s) for s in shard_list]
        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_CAT_VALUES,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(flat_values),
        ):
            merged_values = self._normalize_and_cat_shards(flat_values, dim=0)
        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_MERGE_METADATA,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(shard_list),
        ):
            merged_nested_splits = self._merge_nested_splits(shard_list)
            merged_routing_indices = self._merge_index_descriptors(
                shard_list, "routing_indices"
            )
            merged_origin_indices = self._merge_index_descriptors(
                shard_list, "origin_indices"
            )
            merged_receipt = self._maybe_merge_receipts(shard_list)

        first_shard = shard_list[0]
        if not isinstance(first_shard, (TransportTensor, TransportShardView)):
            raise TypeError("assembled shard must be a typed transport shard")
        global_shape = first_shard.global_shape

        return TransportTensor(
            merged_values,
            nested_splits=merged_nested_splits,
            routing_indices=merged_routing_indices,
            origin_indices=merged_origin_indices,
            global_shape=global_shape,
            receipt=merged_receipt,
        )

    def _assemble_scalar_reduction(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        shards: dict[Any, list[Any]],
        input_layout: "PhysicalLayout",
    ) -> TransportTensor:
        backend = get_backend()
        logical_tuuid = self._compiled_assemble_plan(
            op_name, transfer_code
        ).transfer.logical_tuuid

        all_values = []
        for rank_shards in shards.values():
            for s in rank_shards:
                val = self._get_values(s)
                if backend.shape(val) != ():
                    val = backend.reshape(val, ())
                all_values.append(val)

        if not all_values:
            raise ValueError(
                f"No scalar shards found for transfer_code={transfer_code}, "
                f"logical_tuuid={logical_tuuid} at step {step} for reduction."
            )

        stacked = backend.stack(all_values)
        op = (
            cast(Any, input_layout.reduce_op).value
            if hasattr(input_layout.reduce_op, "value")
            else str(input_layout.reduce_op)
        )
        reduced = _jit_reduce_scalars(stacked, op.lower())
        return TransportTensor.from_tensor(reduced, global_shape=())

    def _assemble_dynamic_to_static_tensor(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        shards: dict[Any, list[Any]],
        input_layout: "PhysicalLayout",
        *,
        assemble_plan: CompiledAssemblePlan,
    ) -> "TransportTensor":
        """Assemble dynamic-to-static (Sparse-to-Dense) tensor using origin_indices."""
        backend = get_backend()
        logical_tuuid = assemble_plan.transfer.logical_tuuid

        shard_count = (
            sum(len(items) for items in shards.values())
            if _profiler_enabled(self)
            else 0
        )
        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_FILTER_SHARDS,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(shards),
            second_count=shard_count,
        ):
            filtered_shards = self._filter_shards_by_routing(
                step=step,
                shards_dict=shards,
                assemble_plan=assemble_plan,
            )

        # Metadata should be extracted from ALL expected shards (even empty ones)
        # to ensure global_shape and receipt provenance is preserved.
        shard_global_shape = None
        for s in filtered_shards:
            if hasattr(s, "global_shape") and s.global_shape is not None:
                shard_global_shape = s.global_shape
                break

        # Align global_shape: use shard metadata, or fallback to layout, or infer.
        if shard_global_shape is not None:
            global_shape = shard_global_shape
        elif input_layout.global_shape is not None:
            global_shape = input_layout.global_shape
        else:
            global_shape = None  # Will infer later if data arrives

        valid_shards = [
            s for s in filtered_shards if backend.shape(self._get_values(s))[0] > 0
        ]

        if not valid_shards:
            flow_detail = self._describe_shard_flow_for_error(
                op_name,
                transfer_code,
                step,
                shards,
                filtered_shards,
                input_layout,
                assemble_plan,
            )
            raise ValueError(
                f"[Fluxway Assembly Error] Dynamic-to-static assembly for op "
                f"'{op_name}' (transfer_code={transfer_code}, "
                f"logical_tuuid={logical_tuuid}, rank {self.rank}, step {step}) "
                "received no non-empty data shards. Dense SPMD inputs must be "
                "slot-addressed by real payload rows; constructing a zero local "
                "buffer would hide upstream routing data loss. "
                f"Flow detail: {flow_detail}"
            )

        if any(getattr(s, "is_ragged", False) for s in valid_shards):
            raise ValueError(
                f"[Fluxway Assembly Error] Dynamic-to-static assembly for op "
                f"'{op_name}' (transfer_code={transfer_code}, "
                f"logical_tuuid={logical_tuuid}, rank {self.rank}, step {step}) "
                "received ragged payload shards. Fluxway transport keeps dynamic "
                "ragged streams compact; consumer-specific padding/pooling/masking "
                "must be implemented in the SPMD operator input adapter."
            )

        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_MERGE_METADATA,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(valid_shards),
        ):
            all_receipt = self._maybe_merge_receipts(valid_shards)

        routing_dim = getattr(input_layout, "routing_dim", 0)
        if isinstance(routing_dim, list | tuple):
            routing_dim = routing_dim[0]

        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_CAT_VALUES,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(valid_shards),
        ):
            all_values = self._normalize_and_cat_shards(
                [self._get_values(s) for s in valid_shards], dim=routing_dim
            )

        # Materialize indices (origin_indices preferred for logical ordering).
        # Both origin_indices and routing_indices encode *global* per-row
        # positions; without them we have no information-theoretic basis to
        # decide which destination slot each row belongs to.  The previous
        # arange(0, num) fallback silently produced sender-local positions
        # that collide across senders (every sender claiming positions 0..N),
        # which is wrong in any multi-sender scenario.  Fail loudly instead
        # of generating undefined scatter behaviour.
        all_indices_list = []
        for s in valid_shards:
            idx = s.origin_indices_materialized
            if idx is None:
                idx = self._get_routing_indices(s)
            if idx is None:
                raise ValueError(
                    f"[Fluxway Protocol Error] Dynamic-to-static assembly for "
                    f"op '{op_name}' (transfer_code={transfer_code}, "
                    f"logical_tuuid={logical_tuuid}) received a shard with "
                    f"neither origin_indices nor routing_indices. The producer "
                    f"must attach per-row global position metadata so each row "
                    f"can be deterministically scattered into the destination "
                    f"buffer; arange-style fallbacks would yield colliding "
                    f"sender-local positions across multiple senders and "
                    f"silently corrupt the result."
                )
            all_indices_list.append(idx)
        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_MERGE_METADATA,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(all_indices_list),
        ):
            all_indices = self._normalize_and_cat_shards(all_indices_list, dim=0)

        # global_shape resolution hierarchy (Strict)
        if shard_global_shape is not None:
            global_shape = shard_global_shape
        elif input_layout.global_shape is not None:
            global_shape = input_layout.global_shape
        else:
            raise ValueError(
                f"[Fluxway Assembly Error] Dynamic-to-Static conversion failed for "
                f"transfer_code={transfer_code}, logical_tuuid={logical_tuuid}. "
                "Global shape information is missing. Inference from raw indices is forbidden. "
                "Please Ensure the producer TransportTensor has global_shape set."
            )

        local_shape = input_layout.get_shard_shape(self.rank, data_shape=global_shape)
        slice_start, slice_end = input_layout.get_shard_bounds(
            self.rank, routing_dim, data_shape=global_shape
        )

        routing_extent = (
            global_shape[routing_dim] if routing_dim < len(global_shape) else None
        )
        num_shards = input_layout.shard_size(routing_dim)
        if (
            routing_extent is not None
            and num_shards > 1
            and routing_extent % num_shards != 0
        ):
            raise ValueError(
                f"[Fluxway Assembly Error] global_shape[{routing_dim}]={routing_extent} "
                f"is not divisible by shard_count={num_shards} "
                f"in op '{op_name}' (transfer_code={transfer_code}, "
                f"logical_tuuid={logical_tuuid}). "
                f"This causes partition boundary misalignment between producers and consumers. "
                f"Ensure the upstream batch_size is divisible by the DP world_size."
            )

        mask = (all_indices >= slice_start) & (all_indices < slice_end)
        local_indices = all_indices[mask] - slice_start

        logical_len_early = int(backend.sum(mask))
        buffer_capacity = local_shape[routing_dim]

        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_SCATTER,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=logical_len_early,
            second_count=buffer_capacity,
        ):
            # When origin_indices have collisions (multiple source rows map to the
            # same target slot via scatter-add), logical_len_early can exceed
            # buffer_capacity even though the actual number of unique target slots
            # fits.  Compute unique slots first and validate against those.
            n_unique_slots = logical_len_early  # default: 1:1 mapping
            if logical_len_early > buffer_capacity and logical_len_early > 0:
                unique_local = backend.unique(local_indices)
                n_unique_slots = int(backend.shape(unique_local)[0])

            if n_unique_slots > buffer_capacity:
                raise ValueError(
                    f"[Fluxway Assembly Error] Dynamic-to-static buffer overflow in op '{op_name}' "
                    f"(transfer_code={transfer_code}, logical_tuuid={logical_tuuid}, "
                    f"rank {self.rank}, step {step}). "
                    f"Received {logical_len_early} data elements mapping to {n_unique_slots} unique "
                    f"buffer positions, but buffer capacity is {buffer_capacity} "
                    f"(global_shape={global_shape}, local_shape={local_shape}). "
                    f"The producer's global_shape is too small for the actual data volume. "
                    f"This typically means the upstream unique/sparse operation produced more elements "
                    f"than global_shape[0] accounts for (e.g. multi-column features after ravel+unique)."
                )

            if logical_len_early > 0 and n_unique_slots < logical_len_early:
                logger.warning(
                    f"[Fluxway Assembly Warning] origin_indices collision detected in op '{op_name}' "
                    f"(transfer_code={transfer_code}, logical_tuuid={logical_tuuid}, "
                    f"rank {self.rank}, step {step}). "
                    f"{logical_len_early} data elements map to only {n_unique_slots} unique buffer positions. "
                    f"{logical_len_early - n_unique_slots} elements will be silently overwritten by index_update. "
                    f"This usually indicates the producer uses row-level origin_indices for multi-column features, "
                    f"causing multiple unique keys from the same row to collide."
                )

            if routing_dim == 0:
                local_values = all_values[mask]
            else:
                indices_to_take = _nonzero_1d_indices(backend, mask)
                idx_tuple = [slice(None)] * len(local_shape)
                idx_tuple[routing_dim] = indices_to_take
                local_values = all_values[tuple(idx_tuple)]

            assembled_data = backend.zeros(local_shape, dtype=all_values.dtype)
            indices_tuple = [slice(None)] * len(local_shape)
            indices_tuple[routing_dim] = local_indices
            assembled_data = backend.index_update(
                assembled_data, tuple(indices_tuple), local_values
            )

            # Assemble mirror receipt into the static layout if present
            assembled_receipt = all_receipt
            if all_receipt is not None:
                final_ranks = backend.full(
                    (local_shape[routing_dim],),
                    all_receipt.world_size,
                    dtype=backend.int32(),
                )
                sliced_receipt = all_receipt[mask]
                final_ranks = backend.index_update(
                    final_ranks, local_indices, sliced_receipt.sender_rank_tags
                )
                assembled_receipt = SymmetryReceipt(all_receipt.world_size, final_ranks)

        return TransportTensor(
            assembled_data,
            routing_indices=None,
            global_shape=global_shape,
            receipt=assembled_receipt,
        )

    def _assemble_to_dynamic_tensor(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        shards: dict[Any, list[Any]],
        arrivals: dict[Any, dict[str, Any]],
        *,
        assemble_plan: CompiledAssemblePlan,
    ) -> "TransportTensor":
        """
        Assemble to-dynamic (Sparse/Dense -> Sparse) tensor by merging shards.
        Handles both static_to_dynamic and dynamic_to_dynamic scenarios.
        """
        backend = get_backend()
        logical_tuuid = assemble_plan.transfer.logical_tuuid

        shard_count = sum(len(items) for items in shards.values())
        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_FILTER_SHARDS,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(shards),
            second_count=shard_count,
        ):
            filtered_shards = assemble_plan.collect_arrived_shards(
                shards,
                self._flatten_shards,
            )

        if not filtered_shards:
            filtered_arrivals = assemble_plan.collect_arrivals(arrivals)
            if filtered_arrivals:
                return self._routing_miss_tensor(
                    op_name, transfer_code, step, filtered_arrivals
                )
            flow_detail = self._describe_shard_flow_for_error(
                op_name,
                transfer_code,
                step,
                shards,
                filtered_shards,
                None,
                assemble_plan,
            )
            raise ValueError(
                f"[Fluxway Assembly Error] Dynamic assembly for op '{op_name}' "
                f"(transfer_code={transfer_code}, logical_tuuid={logical_tuuid}, "
                f"rank {self.rank}, step {step}) received no "
                "arrived shards. Dynamic inputs require an explicit empty or "
                "non-empty packet from every expected sender; synthesizing an "
                f"empty TransportTensor would hide a transport/barrier bug. "
                f"Flow detail: {flow_detail}"
            )

        all_values_list = [self._get_values(s) for s in filtered_shards]
        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_CAT_VALUES,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(all_values_list),
        ):
            merged_values = self._normalize_and_cat_shards(all_values_list, dim=0)
        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_MERGE_METADATA,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(filtered_shards),
        ):
            merged_nested_splits = self._merge_nested_splits(filtered_shards)

        # Spatial Indices (global keys).  If *any* shard carries
        # routing_indices, *every* shard must — they label per-row global
        # positions and a partial set is meaningless (an arange-style
        # fallback for the missing shards would silently invent sender-local
        # positions that collide across producers).  Reject the heterogeneous
        # case explicitly so the contract violation surfaces at the boundary.
        merged_routing_indices = None
        if any(self._get_routing_indices(s) is not None for s in filtered_shards):
            indices_list = []
            for s in filtered_shards:
                idx = self._get_routing_indices(s)
                if idx is None:
                    raise ValueError(
                        f"[Fluxway Protocol Error] Dynamic assembly for op "
                        f"'{op_name}' (transfer_code={transfer_code}, "
                        f"logical_tuuid={logical_tuuid}) received an inconsistent "
                        f"set of shards: some carry routing_indices and some "
                        f"do not.  routing_indices encode per-row *global* "
                        f"positions; producers must agree on whether to emit "
                        f"them.  An arange fallback would generate "
                        f"sender-local positions that collide across senders "
                        f"and silently corrupt downstream routing."
                    )
                indices_list.append(idx)
            with _assemble_detail_span(
                self,
                spec=WORKER_ASSEMBLE_MERGE_METADATA,
                op_name=op_name,
                transfer_code=transfer_code,
                logical_tuuid=logical_tuuid,
                step=step,
                first_count=len(indices_list),
            ):
                merged_routing_indices = self._normalize_and_cat_shards(
                    indices_list, dim=0
                )

        # Origin Indices (logical offsets for deterministic sorting)
        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_MERGE_METADATA,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(filtered_shards),
        ):
            merged_origin_descriptor = self._merge_index_descriptors(
                filtered_shards,
                "origin_indices",
                require_all_if_any=True,
            )
        merged_origin_indices = None

        with _assemble_detail_span(
            self,
            spec=WORKER_ASSEMBLE_MERGE_METADATA,
            op_name=op_name,
            transfer_code=transfer_code,
            logical_tuuid=logical_tuuid,
            step=step,
            first_count=len(filtered_shards),
        ):
            merged_receipt = self._maybe_merge_receipts(filtered_shards)

        original_global_shape = None
        if filtered_shards:
            original_global_shape = filtered_shards[0].global_shape

        # DETERMINISTIC SORTING:
        if merged_origin_descriptor is not None:
            backend = get_backend()
            shape_merged = backend.shape(merged_values)
            actual_len: int = int(shape_merged[0]) if shape_merged else 0

            requires_static_shapes = backend.requires_static_shapes()
            if isinstance(merged_origin_descriptor, ContinuousIndex):
                origin_is_identity_sorted = _continuous_index_is_strictly_increasing(
                    merged_origin_descriptor
                )
            elif requires_static_shapes:
                origin_is_identity_sorted = False
            else:
                merged_origin_indices = merged_origin_descriptor.materialize()
                origin_is_identity_sorted = _is_strictly_increasing_1d(
                    merged_origin_indices
                )
            needs_sort = requires_static_shapes or not origin_is_identity_sorted

            if needs_sort:
                if merged_origin_indices is None:
                    merged_origin_indices = merged_origin_descriptor.materialize()
                receipt_tags = None
                if merged_receipt is not None and not merged_receipt.is_scalar():
                    receipt_tags = merged_receipt.sender_rank_tags

                dummy_routing = False
                sort_routing = merged_routing_indices
                if sort_routing is None:
                    sort_routing = merged_origin_indices
                    dummy_routing = True

                dummy_receipt = False
                if receipt_tags is None:
                    receipt_tags = merged_origin_indices
                    dummy_receipt = True

                if requires_static_shapes:
                    cap = 1 if actual_len <= 1 else 1 << (actual_len - 1).bit_length()

                    sort_values = _pad_tensor_to_power_of_2(
                        merged_values, actual_len, cap, pad_value=0
                    )
                    sort_origin = _pad_tensor_to_power_of_2(
                        merged_origin_indices, actual_len, cap, pad_value=2147483647
                    )
                    sort_routing = _pad_tensor_to_power_of_2(
                        sort_routing, actual_len, cap, pad_value=0
                    )
                    sort_tags = _pad_tensor_to_power_of_2(
                        receipt_tags, actual_len, cap, pad_value=0
                    )
                else:
                    cap = actual_len
                    sort_values = merged_values
                    sort_origin = merged_origin_indices
                    sort_tags = receipt_tags

                with _assemble_detail_span(
                    self,
                    spec=WORKER_ASSEMBLE_SORT,
                    op_name=op_name,
                    transfer_code=transfer_code,
                    logical_tuuid=logical_tuuid,
                    step=step,
                    first_count=actual_len,
                    second_count=cap,
                ):
                    (
                        sorted_values,
                        sorted_origin,
                        sorted_routing,
                        sorted_tags,
                        sort_idx,
                    ) = _jit_sort_fused(
                        sort_values, sort_origin, sort_routing, sort_tags
                    )

                merged_values = sorted_values[:actual_len]
                merged_origin_indices = sorted_origin[:actual_len]
                merged_routing_indices = (
                    sorted_routing[:actual_len] if not dummy_routing else None
                )

                if merged_receipt is not None and not dummy_receipt:
                    merged_receipt = SymmetryReceipt(
                        merged_receipt.world_size, sorted_tags[:actual_len]
                    )
            elif merged_origin_indices is None:
                merged_origin_indices = merged_origin_descriptor

        return TransportTensor(
            merged_values,
            nested_splits=merged_nested_splits,
            routing_indices=merged_routing_indices,
            global_shape=(
                original_global_shape
                if original_global_shape is not None
                else backend.shape(merged_values)
            ),
            receipt=merged_receipt,
            origin_indices=merged_origin_indices,
        )

    async def _advance_step(self, op_name: str, result: ExecutionResult) -> None:
        """
        Advance session progress after completing a step.
        Explicitly differentiates between Count (Progress Watermark) and Index (Task Sequence).
        """
        self._completed_step_count[op_name] += 1
        self._last_progress_timestamp = time.time()
        completed_step_count = self._completed_step_count[op_name]
        logger.debug(
            f"Step completed op={op_name} total_count={completed_step_count} "
            f"spmd_id={self.spmd_id} rank={self.rank}"
        )

        # Schedule next task: The index of the next task equals the number of already finished tasks.
        # e.g., Finished Step Index 0 -> Completed Step Count 1 -> Next Task Index 1.
        next_step_index = completed_step_count
        if self._profiler_runtime.enabled:
            recorder = self._profiler_runtime.recorder
            recorder.worker_scheduler_instant(
                spec=WORKER_ADVANCE_STEP,
                payload=recorder.worker_op_payload(
                    op_name,
                    completed_step_count - 1,
                    next_step_index,
                ),
            )

        # If it's a source operator, automatically trigger the next step
        if self.op_manager.is_source_operator(op_name):
            # Check for global step limit (max_steps)
            max_steps = self.config.max_steps
            if max_steps is not None and completed_step_count >= max_steps:
                logger.debug(
                    f"Source op reached max_steps stopping op={op_name} max_steps={max_steps} "
                    f"spmd_id={self.spmd_id} rank={self.rank}"
                )
                # Max steps reached, do NOT enqueue next task.
            elif (
                result.status is not ExecutionStatus.SUCCESS or result.data is not None
            ):
                # Wait if paused by backpressure
                if op_name in self._source_pause_events:
                    await self._source_pause_events[op_name].wait()

                self._source_maintenance.request(op_name)
            else:
                logger.debug(
                    f"Source op finished op={op_name} spmd_id={self.spmd_id} rank={self.rank}"
                )

        # Report definitive completed count to Driver for synchronization/drain.
        self._report_op_completed(op_name, completed_step_count)
        await self._admission_state.release_async_lag(self._enqueue)
        await self._admission_state.release_phase_order(self._enqueue)

        # Check if this op is in periodic_trigger mode
        is_periodic = (
            self.op_manager.is_periodic_trigger(op_name) if self._op_manager else False
        )

        if not is_periodic:
            # Normal mode: re-enqueue pending future tasks that match the new step
            self._admission_state.move_future(op_name)
            await self._admission_state.start_pending(
                op_name,
                running=lambda: self._running,
                source_exhausted=self._source_is_exhausted,
                start_plan=self._start_local_step_plan,
            )

    def _report_op_completed(self, op_name: str, step: int) -> None:
        """
        Report op completion to driver.

        Args:
            op_name: Operation name
            step: Completed step number
        """
        self._record_execution_phase("completion.report", op_name=op_name, step=step)
        if self._driver_handle is not None:
            logger.debug(
                f"Reporting op completion op={op_name} step={step} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            self._driver_handle.report_worker_op_completed.remote(
                self.rank, op_name, step
            )

    # =========================================================================
    # Data Sending
    # =========================================================================

    async def _send_output(
        self,
        op_name: str,
        result: ExecutionResult,
        *,
        exec_step: int,
        step_transfer_cache: dict[int, Any],
    ) -> tuple[asyncio.Task, ...]:
        """Send output to successor SPMDs using TransportTensor."""
        if result.status is ExecutionStatus.INACTIVE:
            return ()
        send_context = self._operator_send_context(op_name)
        if not send_context.successor_contexts:
            return ()
        if result.status is ExecutionStatus.SKIPPED:
            dependency_decisions = skipped_dependency_decisions(
                send_context.compiled_plan
            )
            return await self._propagate_skipped_completion(
                op_name,
                exec_step,
                dependency_decisions,
            )
        if send_context.compiled_plan.dependency_conditions:
            dependency_decisions = await evaluate_dependency_decisions(
                send_context.compiled_plan,
                result,
            )
        else:
            dependency_decisions = None
        if result.status is ExecutionStatus.COMPLETED_WITHOUT_TENSOR or (
            result.status is ExecutionStatus.SUCCESS and result.data is None
        ):
            return await self._propagate_completed_without_tensor(
                op_name,
                exec_step,
                dependency_decisions,
            )

        # Flatten the result using guided tree_flatten
        meta = self.op_manager.get_operator_spec(op_name)
        flat_results, treedef = tree_flatten(
            result.data,
            treedef_template=(
                None
                if meta.output_treedef is None
                else meta.output_treedef.materialize()
            ),
        )

        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )

        # Extract state arguments
        state_data = {}
        expected_state_names = self.op_manager.get_output_state_names(op_name)
        if expected_state_names:
            # Use tree_unflatten to reconstruct the original structure for state extraction
            # This is a temporary measure until state is properly integrated into the graph
            reconstructed_result = tree_unflatten(treedef, flat_results)
            for name in expected_state_names:
                val = getattr(reconstructed_result, name, None)
                if val is not None:
                    state_data[name] = val

        async def send_successor_frames() -> tuple[asyncio.Task, ...]:
            if not send_context.successor_contexts:
                return ()
            tasks = await self._send_step_data_plane_frames_background(
                flat_results,
                send_context,
                step_transfer_cache,
                exec_step,
                dependency_decisions,
            )
            return tuple(tasks)

        send_tasks: tuple[asyncio.Task, ...] = ()
        if profiler_recorder is None:
            send_tasks = await send_successor_frames()
        else:
            async with profiler_recorder.worker_send_output(
                op_name=op_name,
                step=exec_step,
                output_count=len(flat_results),
                successor_count=len(send_context.successor_contexts),
            ):
                send_tasks = await send_successor_frames()

        self._send_control_arguments(
            send_context=send_context,
            state_data=state_data,
            exec_step=exec_step,
            dependency_decisions=dependency_decisions,
        )
        return send_tasks

    def _send_control_arguments(
        self,
        *,
        send_context: _OperatorSendContext,
        state_data: dict[str, Any],
        exec_step: int,
        dependency_decisions: DependencyDecisions,
    ) -> None:
        if not state_data:
            return
        names = list(state_data)
        values = [ensure_cpu_serializable(state_data[name]) for name in names]
        seen_codes: set[int] = set()
        for context in send_context.successor_contexts:
            for dependency in context.dependencies:
                if dependency.code in seen_codes:
                    raise ValueError(
                        f"Duplicate control dependency code {dependency.code}."
                    )
                seen_codes.add(dependency.code)
                group = dependency.source_operator.execution_layout.group_for_step(
                    exec_step
                )
                if self.rank != group.physical_rank(0):
                    continue
                if dependency.spec.condition is not None:
                    if (
                        dependency_decisions is None
                        or dependency.code not in dependency_decisions
                    ):
                        raise ValueError(
                            f"Missing condition decision for dependency code "
                            f"{dependency.code}."
                        )
                    decision = dependency_decisions[dependency.code]
                    if decision is DependencyDecision.SKIP:
                        continue
                if self._driver_handle is None:
                    raise RuntimeError(
                        f"Cannot send control values for dependency code "
                        f"{dependency.code} without a driver handle."
                    )
                target = dependency.spec.target
                self._driver_handle.routing_control_argument_to_controller.remote(
                    target.spmd_id,
                    target.op_id,
                    names,
                    exec_step,
                    values,
                )

    async def _send_step_data_plane_frames_background(
        self,
        flat_results: list[Any],
        send_context: _OperatorSendContext,
        step_transfer_cache: dict[int, Any],
        exec_step: int,
        dependency_decisions: DependencyDecisions,
    ) -> list[asyncio.Task[Any]]:
        op_name = send_context.compiled_plan.op_name
        local_deliveries: tuple[asyncio.Task[Any], ...] = ()
        remote_deliveries: tuple[asyncio.Task[Any], ...] = ()
        try:
            local_deliveries = await self._prepare_local_only_outputs(
                flat_results,
                send_context,
                step_transfer_cache,
                exec_step,
                dependency_decisions,
            )
            transport = send_context.transport
            if transport is None:
                self._close_step_coherence_intents(op_name, exec_step)
                if local_deliveries:
                    await asyncio.gather(*local_deliveries)
                return []
            tasks = await self._data_plane_executor.send_tensor_frames(
                compiled_plan=transport.compiled_plan,
                flat_results=flat_results,
                step_transfer_cache=step_transfer_cache,
                exec_step=exec_step,
                successor_handlers=transport.successor_handlers,
                source_spmd_id=self.spmd_id,
                batch_max_frames=self._data_plane_batch_max_frames,
                batch_flush_us=self._data_plane_batch_flush_us,
                submit_delivery=self._delivery_submitter(op_name, exec_step),
                delivery_schedule_key=self._effective_schedule_key(op_name, exec_step),
                dependency_decisions=transport.select_dependency_decisions(
                    dependency_decisions
                ),
                batch_submit=partial(
                    self._coherence_batch_retire_runtime.retire_publication,
                    transport.compiled_plan,
                ),
            )
            remote_deliveries = tuple(tasks)
            self._close_step_coherence_intents(op_name, exec_step)
            await asyncio.gather(*tasks)
            if local_deliveries:
                await asyncio.gather(*local_deliveries)
            return tasks
        except BaseException as exc:
            self._abort_step_coherence_intents(op_name, exec_step, exc)
            pending = tuple(
                task for task in local_deliveries + remote_deliveries if not task.done()
            )
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise

    def _delivery_submitter(self, op_name: str, step: int):
        return partial(
            self._delivery_runtime.submit,
            schedule_key=self._effective_schedule_key(op_name, step),
        )

    async def _deposit_local_only_outputs(
        self,
        flat_results: list[Any],
        send_context: _OperatorSendContext,
        step_transfer_cache: dict[int, Any],
        exec_step: int,
        dependency_decisions: DependencyDecisions,
    ) -> None:
        deliveries = await self._prepare_local_only_outputs(
            flat_results,
            send_context,
            step_transfer_cache,
            exec_step,
            dependency_decisions,
        )
        self._close_step_coherence_intents(
            send_context.compiled_plan.op_name,
            exec_step,
        )
        if deliveries:
            await asyncio.gather(*deliveries)

    async def _prepare_local_only_outputs(
        self,
        flat_results: list[Any],
        send_context: _OperatorSendContext,
        step_transfer_cache: dict[int, Any],
        exec_step: int,
        dependency_decisions: DependencyDecisions,
    ) -> tuple[asyncio.Task[Any], ...]:
        deliveries: list[asyncio.Task[Any]] = []
        try:
            return await self._prepare_local_only_outputs_impl(
                flat_results,
                send_context,
                step_transfer_cache,
                exec_step,
                dependency_decisions,
                deliveries,
            )
        except BaseException as error:
            self._abort_step_coherence_intents(
                send_context.compiled_plan.op_name,
                exec_step,
                error,
            )
            if deliveries:
                await asyncio.gather(*deliveries, return_exceptions=True)
            raise

    async def _prepare_local_only_outputs_impl(
        self,
        flat_results: list[Any],
        send_context: _OperatorSendContext,
        step_transfer_cache: dict[int, Any],
        exec_step: int,
        dependency_decisions: DependencyDecisions,
        deliveries: list[asyncio.Task[Any]],
    ) -> tuple[asyncio.Task[Any], ...]:
        wrapped_results: dict[int, TransportTensor] = {}
        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        await self._deposit_local_only_orderings(
            send_context.compiled_plan,
            exec_step,
            dependency_decisions,
        )
        await self._deposit_local_only_dependency_decisions(
            send_context.compiled_plan,
            exec_step,
            dependency_decisions,
        )
        for route in send_context.compiled_plan.successor_routes:
            if not route.local_only:
                continue
            if route.spmd_id != self.spmd_id:
                raise ValueError(
                    "local_only route must target the current SPMD, "
                    f"source={self.spmd_id}:{self.rank} target={route.spmd_id}."
                )
            for tensor_route in route.tensor_routes:
                if (
                    dependency_decisions is not None
                    and dependency_decisions.get(tensor_route.dependency_code)
                    is DependencyDecision.SKIP
                ):
                    continue
                if tensor_route.output_index >= len(flat_results):
                    raise ValueError(
                        f"Operator {send_context.compiled_plan.op_name!r} produced "
                        f"{len(flat_results)} flat outputs, but transfer code "
                        f"{tensor_route.transfer_code} requires output index "
                        f"{tensor_route.output_index}."
                    )
                raw_leaf = flat_results[tensor_route.output_index]
                leaf_tensor = prepare_route_tensor(
                    compiled_plan=send_context.compiled_plan,
                    route=tensor_route,
                    raw_leaf=raw_leaf,
                    step_transfer_cache=step_transfer_cache,
                    exec_step=exec_step,
                    wrapped_results=wrapped_results,
                ).zero_copy_readonly_alias()
                output_subject = self._coherence_subject(
                    self._require_local_spmd_code(),
                    self.rank,
                    tensor_route.transfer_code,
                )
                output_instance = CoherenceInstance(output_subject, exec_step)
                ticket_id = self._coherence_sequencer.allocate_ticket_id()
                leaf_tensor._bind_coherence_ticket_id(ticket_id)
                deposit_complete = asyncio.get_running_loop().create_future()

                async def retire_local_output(
                    *,
                    leaf_tensor=leaf_tensor,
                    output_subject=output_subject,
                    output_instance=output_instance,
                    tensor_route=tensor_route,
                    deposit_complete=deposit_complete,
                    ticket_id=ticket_id,
                ) -> None:
                    try:
                        output_entry = self._coherence_directory.admit(output_instance)
                        if output_entry.state is CoherenceState.INVALID:
                            self._coherence_directory.publish_local_payload(
                                output_subject,
                                exec_step,
                            )
                        if leaf_tensor._coherence_instances:
                            if leaf_tensor._coherence_instances != (output_instance,):
                                raise CoherenceStateTransitionError(
                                    "local-only output has a conflicting coherence identity: "
                                    f"expected={output_instance} "
                                    f"actual={leaf_tensor._coherence_instances}."
                                )
                        else:
                            leaf_tensor._bind_coherence_instances((output_instance,))
                        leaf_tensor._bind_consumer_lease_owner(
                            self._coherence_sequencer
                        )
                        leaf_tensor._reserve_consumer_lease(ticket_id=ticket_id)
                        target_op = tensor_route.target_op
                        mailbox = self._get_operator_mailbox(target_op)
                        if isinstance(mailbox, DirectLocalMailbox):
                            mailbox.deposit(
                                tensor_route.transfer_code,
                                exec_step,
                                self.spmd_id,
                                self.rank,
                                leaf_tensor,
                            )
                        elif isinstance(mailbox, DelegatedWorkerMailbox):
                            self._ensure_step_declared(target_op, exec_step)
                            await mailbox.deposit_local_tensor(
                                transfer_code=tensor_route.transfer_code,
                                step=exec_step,
                                sender_id=self.spmd_id,
                                rank=self.rank,
                                tensor=leaf_tensor,
                            )
                        else:
                            raise TypeError(
                                f"Unsupported operator mailbox type: {type(mailbox)!r}"
                            )
                        if not deposit_complete.done():
                            deposit_complete.set_result(None)
                    except BaseException as error:
                        leaf_tensor._cleanup_consumer_lease_reservation_ref()
                        if not deposit_complete.done():
                            deposit_complete.set_exception(error)
                        raise

                retirement_task = self._spawn_background_task(
                    self._coherence_sequencer.commit_payload(
                        (output_instance,),
                        retire_local_output,
                        ticket_id=ticket_id,
                    )
                )

                def complete_deposit(
                    done_task: asyncio.Task[Any],
                    deposit_complete=deposit_complete,
                ) -> None:
                    if done_task.cancelled():
                        if not deposit_complete.done():
                            deposit_complete.set_exception(asyncio.CancelledError())
                        return
                    error = done_task.exception()
                    if error is not None and not deposit_complete.done():
                        deposit_complete.set_exception(error)

                retirement_task.add_done_callback(complete_deposit)

                lane_status, active_leases = (
                    self._coherence_sequencer.inspect_lane_status(output_instance)
                )

                async def await_deposit(
                    deposit_complete=deposit_complete,
                    tensor_route=tensor_route,
                    lane_status=lane_status,
                    active_leases=active_leases,
                ) -> None:
                    if profiler_recorder is None:
                        await asyncio.shield(deposit_complete)
                    else:
                        async with profiler_recorder.worker_local_delivery(
                            op_name=send_context.compiled_plan.op_name,
                            transfer_code=tensor_route.transfer_code,
                            logical_tuuid=tensor_route.logical_tuuid,
                            step=exec_step,
                            active_consumer_leases=active_leases,
                            status=lane_status,
                        ):
                            await asyncio.shield(deposit_complete)

                deliveries.append(asyncio.create_task(await_deposit()))
        return tuple(deliveries)

    async def _deposit_local_only_dependency_decisions(
        self,
        compiled_plan: CompiledSendPlan,
        step: int,
        dependency_decisions: DependencyDecisions,
    ) -> None:
        if not dependency_decisions:
            return
        for route in compiled_plan.successor_routes:
            if not route.local_only:
                continue
            for dependency_decision_route in route.dependency_decision_routes:
                decision = dependency_decisions.get(
                    dependency_decision_route.dependency_code
                )
                if decision is None:
                    continue
                if dependency_decision_route.target_ranks != (self.rank,):
                    raise ValueError(
                        "local_only dependency decision "
                        f"code={dependency_decision_route.dependency_code} must target rank "
                        f"{self.rank}, got {dependency_decision_route.target_ranks}."
                    )
                self._ensure_step_declared(dependency_decision_route.target_op, step)
                await self._mailbox.deposit_dependency_decision(
                    dependency_decision_route.target_op,
                    dependency_decision_route.dependency_code,
                    step,
                    sender_id=self.spmd_id,
                    rank=self.rank,
                    expected_count=dependency_decision_route.expected_count,
                    decision=decision,
                )

    async def _deposit_local_only_orderings(
        self,
        compiled_plan: CompiledSendPlan,
        step: int,
        dependency_decisions: DependencyDecisions,
    ) -> None:
        for route in compiled_plan.successor_routes:
            if not route.local_only:
                continue
            for ordering in route.ordering_routes:
                if (
                    dependency_decisions is not None
                    and ordering.dependency_code in dependency_decisions
                ):
                    continue
                if ordering.target_ranks != (self.rank,):
                    raise ValueError(
                        f"local_only ordering dependency code={ordering.dependency_code} "
                        f"must target rank {self.rank}, got {ordering.target_ranks}."
                    )
                self._ensure_step_declared(ordering.target_op, step)
                await self._mailbox.deposit_ordering(
                    ordering.target_op,
                    ordering.dependency_code,
                    step,
                    sender_id=self.spmd_id,
                    rank=self.rank,
                    expected_count=ordering.expected_count,
                )

    def _reconstruct_data_type_from_dict(self, cls: type, data: dict[str, Any]) -> Any:
        """Reconstruct a @data_type or dataclass from a nested dictionary of fields."""
        from fluxway.common.decorators import _is_fluxway_data_type

        if not _is_fluxway_data_type(cls):
            # If not a data_type, it might be a simple type or a raw dict
            return data

        # Get type hints for constructor
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}

        kwargs = {}
        for name, hint in hints.items():
            if name in data:
                val = data[name]
                if isinstance(val, dict) and _is_fluxway_data_type(hint):
                    kwargs[name] = self._reconstruct_data_type_from_dict(hint, val)
                else:
                    kwargs[name] = val

        return cls(**kwargs)

    def _extract_control_arguments(
        self, data: Any, state_names: list[str]
    ) -> dict[str, Any]:
        """Extract named control argument values from operator output data."""
        args: dict[str, Any] = {}

        for name in state_names:
            val = getattr(data, name, None)
            if val is not None:
                args[name] = val

        if len(args) < len(state_names) and isinstance(data, dict):
            for name in state_names:
                if name not in args and name in data:
                    args[name] = data[name]

        return args

    # =========================================================================
    # Static Arguments
    # =========================================================================

    def set_static_arg(self, name: str, value: Any) -> None:
        """Set a static argument value."""
        self.op_manager.set_static_arg(name, value)

    def get_static_arg(self, name: str) -> Any | None:
        """Get a static argument value."""
        return self.op_manager.get_static_arg(name)

    # =========================================================================
    # State Argument Handling
    # =========================================================================

    def receive_control_argument(self, state_name: str, state_value: Any) -> None:
        """
        Receive control_argument from driver (broadcast from Controller).

        State arguments are set as static args for operator execution.

        Args:
            state_name: State variable name
            state_value: State value
        """
        self.set_static_arg(state_name, state_value)
        logger.debug(
            f"Received control_argument name={state_name} spmd_id={self.spmd_id} rank={self.rank}"
        )

    def send_control_argument(
        self,
        target_spmd_id: str,
        target_op_name: str,
        state_name: str,
        step: int,
        state_value: Any,
    ) -> None:
        """
        Send control_argument to target SPMD.

        Only rank 0 worker should call this method.
        Path: rank0-worker -> driver -> controller -> target-driver -> all-workers

        Args:
            target_spmd_id: Target SPMD ID
            target_op_name: Target operator type on the receiving side
            state_name: State variable name
            step: Current step number
            state_value: State value

        Raises:
            RuntimeError: If called from non-rank-0 worker
        """
        if self.rank != 0:
            raise RuntimeError("Only rank 0 worker can send control_argument")

        if self._driver_handle is None:
            logger.error(
                f"No driver handle cannot send control_argument "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            return

        try:
            self._driver_handle.routing_control_argument_to_controller.remote(
                target_spmd_id,
                target_op_name,
                state_name,
                step,
                ensure_cpu_serializable(state_value),
            )
            logger.debug(
                f"Sent control_argument name={state_name} target_spmd_id={target_spmd_id} "
                f"target_op={target_op_name} step={step} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
        except Exception as e:
            logger.error(
                f"Failed to send control_argument error={e} spmd_id={self.spmd_id} rank={self.rank}"
            )

    # =========================================================================
    # Heartbeat
    # =========================================================================

    def _start_heartbeat(self) -> None:
        """Start heartbeat reporting to driver as an asyncio task."""
        if self._heartbeat_running:
            return

        self._heartbeat_running = True
        self._heartbeat_task = asyncio.get_event_loop().create_task(
            self._heartbeat_loop()
        )
        logger.debug(
            f"Worker heartbeat loop started spmd_id={self.spmd_id} rank={self.rank}"
        )

    async def stop_heartbeat(self) -> None:
        """Stop heartbeat reporting."""
        self._heartbeat_running = False
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Heartbeat reporting loop (async)."""
        while self._heartbeat_running:
            try:
                self._warn_if_stalled()
                await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Error sending heartbeat: {e}"
                )

            await asyncio.sleep(self._heartbeat_interval)

    def _warn_if_stalled(self) -> None:
        if self._failure is not None or not self._running:
            return
        now = time.time()
        elapsed = now - self._last_progress_timestamp
        threshold = self.config.execution_progress_warning_s
        if elapsed < threshold:
            return
        if self._last_stall_warning_timestamp == self._last_progress_timestamp:
            return
        self._last_stall_warning_timestamp = self._last_progress_timestamp
        diagnostics = self._execution_diagnostic_snapshot()
        logger.warning(
            "Worker execution has no committed progress for %.3fs; "
            "wait_state=%s phase=%s phase_age=%.3fs activity_age=%.3fs "
            "stall_reason=%s. "
            "spmd_id=%s rank=%s diagnostics=%s",
            elapsed,
            diagnostics.get("execution_wait_state", "unknown"),
            diagnostics.get("execution_phase"),
            diagnostics.get("execution_phase_age_s", 0.0),
            diagnostics.get("execution_activity_age_s", 0.0),
            diagnostics["stall_reason"],
            self.spmd_id,
            self.rank,
            diagnostics,
        )

    def _execution_diagnostic_snapshot(self) -> dict[str, Any]:
        now = time.time()
        checkpoint_activity = self._checkpoint_activity.snapshot()
        active_plan_steps = {
            op_name: sorted(steps)
            for op_name, steps in sorted(self._running_plan_steps.items())
            if steps
        }
        queue_sizes = {
            lane: queue.qsize()
            for lane, queue in sorted(self._execution_queues.items())
        }
        pending_plan_steps = sorted(
            {
                (op_name, step)
                for task, (op_name, step) in self._plan_task_steps.items()
                if not task.done()
            }
        )
        pending_send_steps = sorted(
            {
                (op_name, step)
                for task, (op_name, step) in self._send_tasks.items()
                if not task.done()
            }
        )
        pending_plan_tasks = sum(1 for task in self._plan_tasks if not task.done())
        pending_send_tasks = sum(1 for task in self._send_tasks if not task.done())
        pending_input_prepare_tasks = self._input_prepare_runtime.pending_count()
        pending_admission = any(
            steps
            for pending_by_op in (
                self._pending_blocked,
                self._pending_future_steps,
                self._pending_async_lag_steps,
                self._pending_phase_order_steps,
                self._ready_pending_pipeline_steps,
            )
            for steps in pending_by_op.values()
        ) or any(self._mailbox_ready_steps.values())
        execution_phase = getattr(self, "_execution_phase", None)
        execution_phase_active = bool(getattr(self, "_execution_phase_active", False))
        execution_phase_age_s = (
            max(
                0.0,
                now
                - getattr(
                    self,
                    "_execution_phase_started_timestamp",
                    now,
                ),
            )
            if execution_phase_active
            else 0.0
        )
        execution_activity_age_s = max(
            0.0,
            now
            - getattr(
                self,
                "_last_execution_activity_timestamp",
                self._last_progress_timestamp,
            ),
        )
        committed_progress_age_s = max(0.0, now - self._last_progress_timestamp)
        execution_wait_state = _classify_execution_wait_state(
            execution_phase=execution_phase,
            execution_phase_active=execution_phase_active,
            active_plan_steps=bool(active_plan_steps),
            pending_plan_tasks=bool(pending_plan_tasks),
            pending_send_tasks=bool(pending_send_tasks),
            pending_input_prepare_tasks=bool(pending_input_prepare_tasks),
            queued_work=any(queue_sizes.values()),
            pending_admission=pending_admission,
        )

        if active_plan_steps or pending_plan_tasks:
            stall_reason = "operator_plan"
        elif pending_send_tasks:
            stall_reason = "outbound_transport"
        elif pending_input_prepare_tasks:
            stall_reason = "input_prepare"
        elif any(queue_sizes.values()):
            stall_reason = "execution_queue"
        elif pending_admission:
            stall_reason = "admission_wait"
        else:
            stall_reason = "completion_barrier_or_driver_wait"
        return {
            "execution_generation": getattr(self, "_execution_generation", 0),
            "execution_operator": getattr(self, "_execution_operator", None),
            "execution_step": getattr(self, "_execution_step", None),
            "execution_epoch": getattr(self, "_execution_epoch", 0),
            "execution_sequence": getattr(self, "_execution_sequence", 0),
            "execution_in_flight": bool(active_plan_steps),
            "execution_phase": execution_phase,
            "execution_phase_age_s": execution_phase_age_s,
            "execution_activity_age_s": execution_activity_age_s,
            "committed_progress_age_s": committed_progress_age_s,
            "execution_wait_state": execution_wait_state.value,
            "active_plan_steps": active_plan_steps,
            "pending_plan_steps": [
                {"op_name": op_name, "step": step}
                for op_name, step in pending_plan_steps
            ],
            "pending_send_steps": [
                {"op_name": op_name, "step": step}
                for op_name, step in pending_send_steps
            ],
            "completed_step_count": dict(sorted(self._completed_step_count.items())),
            "scheduled_step_count": dict(sorted(self._scheduled_step_count.items())),
            "execution_queue_sizes": queue_sizes,
            "pending_background_tasks": sum(
                1 for task in self._background_tasks if not task.done()
            ),
            "pending_plan_tasks": pending_plan_tasks,
            "pending_send_tasks": pending_send_tasks,
            "pending_input_prepare_tasks": pending_input_prepare_tasks,
            "pending_phase_order_steps": {
                op_name: list(steps)
                for op_name, steps in sorted(self._pending_phase_order_steps.items())
                if steps
            },
            "pending_future_steps": {
                op_name: list(steps)
                for op_name, steps in sorted(self._pending_future_steps.items())
                if steps
            },
            "pending_async_lag_steps": {
                op_name: list(steps)
                for op_name, steps in sorted(self._pending_async_lag_steps.items())
                if steps
            },
            "checkpoint_phase": checkpoint_activity.phase.value,
            "checkpoint_id": checkpoint_activity.checkpoint_id,
            "checkpoint_progress_timestamp": checkpoint_activity.progress_timestamp,
            "checkpoint_completed_units": checkpoint_activity.completed_units,
            "stall_reason": stall_reason,
        }

    async def _send_heartbeat(self) -> None:
        """Send heartbeat to driver."""
        if self._driver_handle is None:
            return

        # Determine status
        failure = self._failure
        if failure is not None:
            status = WorkerStatus.ERROR
        elif self._execution_lifecycle.phase is WorkerPhase.READY:
            status = WorkerStatus.READY
        elif self._execution_lifecycle.phase is WorkerPhase.RESTORING:
            status = WorkerStatus.RESTORING
        elif self._execution_lifecycle.phase in (
            WorkerPhase.CLOSING,
            WorkerPhase.CLOSED,
        ):
            status = WorkerStatus.STOPPED
        elif self._running and any(
            not event.is_set() for event in self._op_runnable.values()
        ):
            status = WorkerStatus.BLOCKED
        elif self._running:
            status = WorkerStatus.RUNNING
        else:
            status = WorkerStatus.INITIALIZING

        checkpoint_activity = self._checkpoint_activity.snapshot()
        if self._running:
            diagnostics = self._execution_diagnostic_snapshot()
        else:
            now = time.time()
            committed_progress_timestamp = getattr(
                self, "_last_progress_timestamp", now
            )
            execution_activity_timestamp = getattr(
                self,
                "_last_execution_activity_timestamp",
                committed_progress_timestamp,
            )
            diagnostics = {
                "execution_phase": getattr(self, "_execution_phase", None),
                "execution_phase_age_s": 0.0,
                "execution_activity_age_s": max(
                    0.0, now - execution_activity_timestamp
                ),
                "committed_progress_age_s": max(
                    0.0, now - committed_progress_timestamp
                ),
                "execution_wait_state": ExecutionWaitState.IDLE.value,
            }
        execution_phase = cast(str | None, diagnostics["execution_phase"])
        execution_phase_age_s = cast(float, diagnostics["execution_phase_age_s"])
        execution_activity_age_s = cast(float, diagnostics["execution_activity_age_s"])
        committed_progress_age_s = cast(float, diagnostics["committed_progress_age_s"])
        heartbeat = WorkerHeartbeat(
            rank=self.rank,
            timestamp=time.time(),
            status=status,
            failure=failure,
            checkpoint_phase=checkpoint_activity.phase,
            checkpoint_progress_timestamp=checkpoint_activity.progress_timestamp,
            checkpoint_id=checkpoint_activity.checkpoint_id,
            checkpoint_completed_units=checkpoint_activity.completed_units,
            execution_generation=getattr(self, "_execution_generation", 0),
            execution_operator=getattr(self, "_execution_operator", None),
            execution_step=getattr(self, "_execution_step", None),
            execution_epoch=getattr(self, "_execution_epoch", 0),
            execution_sequence=getattr(self, "_execution_sequence", 0),
            execution_in_flight=any(getattr(self, "_running_plan_steps", {}).values()),
            execution_phase=execution_phase,
            execution_phase_age_s=execution_phase_age_s,
            execution_activity_age_s=execution_activity_age_s,
            committed_progress_age_s=committed_progress_age_s,
            execution_wait_state=ExecutionWaitState(
                diagnostics["execution_wait_state"]
            ),
        )

        try:
            self._driver_handle.receive_worker_heartbeat.remote(self.rank, heartbeat)
            logger.heartbeat(
                f"Worker heartbeat sent status={status.name} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
        except Exception as e:
            logger.error(
                f"Failed to send heartbeat error={e} spmd_id={self.spmd_id} rank={self.rank}"
            )

    def _operator_instance(self, op_name: str) -> Any | None:
        op_meta = self._operator_specs.get(op_name)
        if op_meta is None:
            raise RuntimeError(
                f"Operator {op_name} not found in config "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
        owner_cls = op_meta.owner
        if owner_cls is None:
            return None
        instance = self._instance_container.get(owner_cls)
        if instance is None:
            raise RuntimeError(
                f"Operator {op_name} instance is not initialized "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
        return instance

    def _operator_instances(self) -> dict[str, Any]:
        if self._op_manager is None:
            return {}
        instances: dict[str, Any] = {}
        seen: set[int] = set()
        for op_name in self._operator_specs:
            instance = self._operator_instance(op_name)
            if instance is None or id(instance) in seen:
                continue
            seen.add(id(instance))
            instances[op_name] = instance
        return instances

    async def _shutdown_local_operator_instances(self) -> None:
        shutdown_errors: list[Exception] = []
        for instance in self._operator_instances().values():
            hook = getattr(instance, "on_shutdown", None)
            if not callable(hook):
                continue
            try:
                result = hook()
                if inspect.isawaitable(result):
                    await result
            except Exception as error:
                shutdown_errors.append(error)
        if len(shutdown_errors) == 1:
            raise shutdown_errors[0]
        if shutdown_errors:
            raise ExceptionGroup("local operator shutdown failed", shutdown_errors)

    # =========================================================================
    # State Persistence (checkpoint I/O)
    # =========================================================================

    async def checkpoint_prepare_save(
        self,
        repository_root: str,
        checkpoint_id: str,
        attempt_id: str,
        permit: CheckpointCommitPermit,
        cut: CheckpointCut,
    ) -> None:
        self._checkpoint_activity.begin(checkpoint_id)
        try:
            await self._drain_outbound_tasks()
            coordinator = self._get_checkpoint_coordinator(repository_root)
            await coordinator.prepare_save(checkpoint_id, attempt_id, permit, cut)
            self._checkpoint_activity.phase(CheckpointActivityPhase.CAPTURING)
        except BaseException:
            self._checkpoint_activity.failed()
            raise

    async def checkpoint_save(
        self,
    ) -> tuple[dict[str, Any], Mapping[str, WorkerSaveResult]]:
        try:
            await self._drain_outbound_tasks()
            coordinator = self._require_checkpoint_coordinator()
            records = await coordinator.save()
            self._checkpoint_activity.phase(CheckpointActivityPhase.STAGING)
            self._checkpoint_activity.unit_completed()
        except BaseException:
            self._checkpoint_activity.failed()
            raise
        snapshot_boundary = self._drain_boundary
        if self._drain_step_count is not None:
            snapshot_cut = self._drain_step_count
            if snapshot_boundary.observed_cut is not None:
                snapshot_cut = min(snapshot_cut, snapshot_boundary.observed_cut)
            snapshot_boundary = DrainBoundary(
                phase=(
                    DrainPhase.COMPLETED
                    if snapshot_boundary.observed_cut is not None
                    else DrainPhase.OPEN
                ),
                observed_cut=snapshot_boundary.observed_cut,
                admitted_cut=snapshot_cut,
                drained_cut=snapshot_cut,
                committed_cut=snapshot_cut,
                revision=snapshot_boundary.revision + 1,
            )
        runtime_state = {
            "drain_boundary": snapshot_boundary.to_dict(),
        }
        if snapshot_boundary.committed_cut is None:
            raise RuntimeError(
                "checkpoint save completed without a checkpoint position"
            )
        return runtime_state, records

    async def checkpoint_finish_save(self) -> None:
        try:
            await self._drain_outbound_tasks()
            await self._require_checkpoint_coordinator().finish_save()
            self._checkpoint_activity.phase(CheckpointActivityPhase.COMMITTING)
        except BaseException:
            self._checkpoint_activity.failed()
            raise

    async def checkpoint_commit_save(
        self,
        gathered_records: Mapping[str, Mapping[int, WorkerSaveResult]],
    ) -> Mapping[str, CheckpointComponentPublication]:
        try:
            await self._drain_outbound_tasks()
            result = await self._require_checkpoint_coordinator().commit_save(
                gathered_records
            )
            self._checkpoint_activity.unit_completed()
            return result
        except BaseException:
            self._checkpoint_activity.failed()
            raise

    async def checkpoint_rollback_save(self) -> None:
        try:
            await self._drain_outbound_tasks()
            await self._require_checkpoint_coordinator().rollback_active()
        finally:
            if self._checkpoint_activity.snapshot().checkpoint_id is not None:
                self._checkpoint_activity.failed()

    async def checkpoint_publish_save(self) -> bool:
        try:
            has_output = await self._require_checkpoint_coordinator().publish_save()
            self._checkpoint_activity.published()
            return has_output
        except BaseException:
            self._checkpoint_activity.failed()
            raise

    async def checkpoint_restore(
        self,
        repository_root: str,
        checkpoint_id: str,
        cut: CheckpointCut,
        source_topology: TopologySnapshot,
        current_topology: TopologySnapshot,
        runtime_state: Mapping[str, Any],
        publications: Mapping[str, CheckpointComponentPublication],
        restore_options_by_component: RestoreOptionsByComponent | None = None,
    ) -> None:
        self._execution_lifecycle.begin_restore()
        try:
            await self._drain_outbound_tasks()
            self._restore_checkpoint_runtime_state(runtime_state)
            coordinator = self._get_checkpoint_coordinator(repository_root)
            await coordinator.restore(
                checkpoint_id,
                cut,
                publications,
                source_topology,
                current_topology,
                restore_options_by_component,
            )
            self._execution_lifecycle.finish_restore()
        except BaseException as error:
            self._record_failure(error, phase="RESTORE")
            raise

    def _restore_checkpoint_runtime_state(
        self, runtime_state: Mapping[str, Any]
    ) -> None:
        self._execution_lifecycle.require_restore()
        self._source_scope.require_unactivated()
        committed_cut = checkpoint_position(runtime_state)
        boundary_state = runtime_state["drain_boundary"]
        DrainBoundary.from_mapping(boundary_state)
        self._drain_boundary = DrainBoundary.from_committed_cut(committed_cut)
        self._drain_step_count = committed_cut
        for op_name in self._completed_step_count:
            self._completed_step_count[op_name] = committed_cut
            self._scheduled_step_count[op_name] = committed_cut

    def _get_checkpoint_coordinator(
        self, repository_root: str
    ) -> WorkerCheckpointCoordinator:
        coordinator = self._checkpoint_coordinator
        if coordinator is not None:
            if self._checkpoint_repository_root != repository_root:
                raise ValueError(
                    "worker checkpoint repository root changed during its lifetime: "
                    f"current={self._checkpoint_repository_root!r} "
                    f"requested={repository_root!r}"
                )
            return coordinator
        checkpoint_components = tuple(self.config.checkpoint_components)
        operator_instance: Any = None
        if checkpoint_components:
            modules = tuple(self._operator_instances().values())
            if len(modules) != 1:
                raise RuntimeError(
                    "checkpoint components require exactly one SPMD owner instance, "
                    f"got {len(modules)} for spmd_id={self.spmd_id!r}"
                )
            operator_instance = modules[0]
        coordinator = WorkerCheckpointCoordinator(
            components=checkpoint_components,
            operator_instance=operator_instance,
            repository_root=repository_root,
            spmd_id=self.spmd_id,
            rank=self.rank,
            source_topology=self._checkpoint_topology(),
        )
        self._checkpoint_coordinator = coordinator
        self._checkpoint_repository_root = repository_root
        return coordinator

    def _require_checkpoint_coordinator(self) -> WorkerCheckpointCoordinator:
        coordinator = self._checkpoint_coordinator
        if coordinator is None:
            raise RuntimeError("worker checkpoint transaction is not prepared")
        return coordinator

    def _checkpoint_topology(self) -> TopologySnapshot:
        mesh = self._graph_view.mesh
        if mesh is None:
            raise RuntimeError(
                f"checkpoint requires a device mesh for spmd_id={self.spmd_id!r}"
            )
        return TopologySnapshot(
            world_size=self._world_size,
            mesh_axis_names=tuple(mesh.axis_names),
            mesh_axis_shapes=tuple(mesh.shape),
        )

    async def inspect_operator_state(self) -> dict[str, Any]:
        snapshots: dict[str, Any] = {}
        for op_name, instance in self._operator_instances().items():
            snapshot_fn = getattr(instance, "audit_state_snapshot", None)
            if callable(snapshot_fn):
                snapshots[op_name] = snapshot_fn()
        operator_inputs: dict[str, Any] = {}
        transfer_targets: dict[str, str] = {}
        if self._op_manager is not None:
            for op_name in sorted(self._operator_specs):
                input_sharding = self.op_manager.get_input_sharding(op_name)
                input_transfer_codes = self.op_manager.get_input_transfer_codes(op_name)
                expected_transfer_codes = self.op_manager.get_expected_transfer_codes(
                    op_name
                )
                predecessor_routes = self.op_manager.get_predecessor_routes(op_name)
                successor_routes = self.op_manager.get_successor_routes(op_name)
                operator_inputs[op_name] = {
                    "input_transfer_codes": dict(sorted(input_transfer_codes.items())),
                    "input_sharding_transfer_codes": sorted(input_sharding),
                    "expected_transfer_codes": expected_transfer_codes,
                    "expected_transfer_shards": self.op_manager.get_expected_transfer_shards(
                        op_name
                    ),
                    "output_logical_tuuids": self.op_manager.get_output_tuuids(op_name),
                    "predecessors": [
                        _route_inspection_snapshot(route)
                        for route in sorted(
                            predecessor_routes,
                            key=lambda item: item.transfer_code,
                        )
                    ],
                    "successors": [
                        _route_inspection_snapshot(route)
                        for route in sorted(
                            successor_routes,
                            key=lambda item: item.transfer_code,
                        )
                    ],
                }
                for route in predecessor_routes:
                    transfer_targets[str(route.transfer_code)] = route.target_op
        return {
            "spmd_id": self.spmd_id,
            "rank": self.rank,
            "world_size": self._world_size,
            "completed_step_count": dict(self._completed_step_count),
            "operators": snapshots,
            "runtime": {
                "drain_step_count": self._drain_step_count,
                "drain_boundary": self._drain_boundary.to_dict(),
                "scheduled_step_count": dict(self._scheduled_step_count),
                "barriers_declared": [
                    [op_name, step] for op_name, step in sorted(self._barriers_declared)
                ],
                "blocked_ops": [
                    op_name
                    for op_name, event in sorted(self._op_runnable.items())
                    if not event.is_set()
                ],
                "latest_op_runnable_epoch": dict(
                    sorted(self._latest_op_runnable_epoch.items())
                ),
                "pending_blocked": {
                    op_name: list(steps)
                    for op_name, steps in sorted(self._pending_blocked.items())
                    if steps
                },
                "pending_future_steps": {
                    op_name: list(steps)
                    for op_name, steps in sorted(self._pending_future_steps.items())
                    if steps
                },
                "pending_async_lag_steps": {
                    op_name: list(steps)
                    for op_name, steps in sorted(self._pending_async_lag_steps.items())
                    if steps
                },
                "ready_pending_pipeline_steps": {
                    op_name: list(steps)
                    for op_name, steps in sorted(
                        self._ready_pending_pipeline_steps.items()
                    )
                    if steps
                },
                "mailbox_ready_steps": {
                    op_name: sorted(steps)
                    for op_name, steps in sorted(self._mailbox_ready_steps.items())
                    if steps
                },
                "running_plan_steps": {
                    op_name: sorted(steps)
                    for op_name, steps in sorted(self._running_plan_steps.items())
                    if steps
                },
                "finished_plan_steps": {
                    op_name: sorted(steps)
                    for op_name, steps in sorted(self._finished_plan_steps.items())
                    if steps
                },
                "execution_queue_size": sum(
                    queue.qsize() for queue in self._execution_queues.values()
                ),
                "execution_queue_sizes": {
                    lane: queue.qsize()
                    for lane, queue in sorted(self._execution_queues.items())
                },
                "in_op_step_queue": [
                    [op_name, step] for op_name, step in sorted(self._in_op_step_queue)
                ],
                "background_task_count": sum(
                    1 for task in self._background_tasks if not task.done()
                ),
                "plan_task_count": sum(
                    1 for task in self._plan_tasks if not task.done()
                ),
                "send_task_count": sum(
                    1 for task in self._send_tasks if not task.done()
                ),
                "input_prepare_pending_count": self._input_prepare_runtime.pending_count(),
                "mailbox": await self._mailbox.snapshot(),
                "operator_inputs": operator_inputs,
                "transfer_targets": transfer_targets,
            },
        }

    async def export_profiler_snapshot(self) -> ProfilerSnapshot | None:
        await self._drain_outbound_tasks()
        return self._profiler_runtime.snapshot()

    # =========================================================================
    # Utility
    # =========================================================================

    def get_current_step(self, op_name: str) -> int:
        """Get current step for an operator name."""
        return self._completed_step_count[op_name]

    def get_rank(self) -> int:
        """Get this worker's rank."""
        return self.rank

    @property
    def _running(self) -> bool:
        return self._execution_lifecycle.active

    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    # =========================================================================
    # Drain Control (called by Driver before drain_to_step_count)
    # =========================================================================

    def _drain_blocks_step(self, op_name: str, step: int) -> bool:
        del op_name
        target = self._drain_step_count
        return target is not None and step >= target

    async def _set_drain_boundary(
        self, step_count: int | None, terminal_cut: int | None = None
    ) -> None:
        if step_count is not None and (
            isinstance(step_count, bool) or not isinstance(step_count, int)
        ):
            raise TypeError("drain step count must be an integer or None")
        if step_count is not None and step_count < 0:
            raise ValueError(f"drain step count must be non-negative, got {step_count}")
        async with self._drain_cond:
            if terminal_cut is not None:
                transition = reduce_drain_boundary(
                    self._drain_boundary,
                    DrainEvent(DrainEventKind.TERMINAL_OBSERVATION, terminal_cut),
                )
                self._drain_boundary = transition.boundary
            self._drain_step_count = step_count
            self._drain_cond.notify_all()
            logger.debug(
                "Drain step count set to %s spmd_id=%s rank=%s",
                step_count,
                self.spmd_id,
                self.rank,
            )
        if terminal_cut is not None:
            for op_name in self._source_operator_names:
                self._discard_source_pipeline(op_name, terminal_cut)
        await self._input_prepare_runtime.cancel_matching(
            lambda op_name, step: (
                self._drain_blocks_step(op_name, step)
                and step not in self._running_plan_steps.get(op_name, set())
            )
        )
        if step_count is not None:
            for op_name in self._source_operator_names:
                self._source_maintenance.request(op_name)

    def _source_is_exhausted(self, op_name: str) -> bool:
        if op_name not in self._source_operator_names:
            return False
        return self._drain_boundary.observed_cut is not None

    def _source_blocks_step(self, op_name: str, step: int) -> bool:
        del op_name
        terminal_step = self._drain_boundary.observed_cut
        return terminal_step is not None and step >= terminal_step

    def _committed_step(self, op_name: str) -> int:
        execution_step = self._completed_step_count[op_name]
        terminal_cut = self._drain_boundary.observed_cut
        if terminal_cut is None:
            return execution_step
        return min(execution_step, terminal_cut)

    async def install_admission(self, admission: DrainAdmission) -> None:
        if not admission.supersedes(self._admission):
            return
        self._admission = admission
        await self._set_drain_boundary(admission.target, admission.terminal_cut)

    async def set_drain_boundary(
        self, step_count: int, terminal_cut: int | None = None
    ) -> None:
        await self._set_drain_boundary(step_count, terminal_cut)

    async def clear_drain_boundary(self) -> None:
        """Remove the drain target and let deferred executions proceed."""
        await self._set_drain_boundary(None)
        for op_name in self._source_operator_names:
            self._source_maintenance.request(op_name)

    # =========================================================================
    # Source Node Control
    # =========================================================================

    def pause_sources(self) -> None:
        """
        Pause all source operators (called by driver when downstream is busy).
        """
        for event in self._source_pause_events.values():
            event.clear()
        logger.debug(
            f"Sources paused on worker spmd_id={self.spmd_id} rank={self.rank}"
        )

    def resume_sources(self) -> None:
        """
        Resume all source operators (called by driver when backpressure relieved).
        """
        for event in self._source_pause_events.values():
            event.set()
        logger.debug(
            f"Sources resumed on worker spmd_id={self.spmd_id} rank={self.rank}"
        )
