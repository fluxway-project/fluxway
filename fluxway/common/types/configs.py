"""
Core data type definitions for Fluxway framework.

Based on ARCHITECTURE.md sections 2.1.1, 3.2.1, 4.2, 4.3
"""

import os
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Any, TypeVar

from pydantic import ConfigDict, Field, model_validator

from fluxway.common.compiled_graph import (
    CompiledGraph,
    DependencySpec,
    SPMDGraphView,
)
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.schema import FluxwayBaseSchema
from fluxway.common.types.checkpoint import CheckpointRetentionPolicy
from fluxway.common.types.failure import ExecutionFailure
from fluxway.resources import ResourceProfile

T = TypeVar("T")

DEFAULT_SNAPSHOT_TIMEOUT_SECS: float = 86400.0
DEFAULT_INACTIVITY_TIMEOUT_SECS: float = 600.0


# Transport stream markers for fields within a @data_type
class DataFlow:
    """Marker for Worker-to-Worker (Data Plane) transport. Supports sharding."""


class ControlFlow:
    """Marker for Driver-to-Driver (Control Plane) transport. Broadcast to all workers."""


# User-facing type aliases
# Usage: field_name: Data[jnp.ndarray] or field_name: Control[float]
Data = Annotated[T, DataFlow]
Control = Annotated[T, ControlFlow]


class TensorProxy(FluxwayBaseSchema):
    """
    A lightweight proxy representing a tensor in the computational graph.

    It uses a globally unique identifier (int TUUID) to track dependencies and
    sharding specifications independently of the variable names used in
    operator functions.
    """

    tuuid: int
    name: str | None = None  # Original name for debugging
    shape: tuple[int, ...] | None = None
    dtype: str | None = None

    def __repr__(self):
        return f"TensorProxy(tuuid={self.tuuid}, name={self.name})"


class NodeType(StrEnum):
    """Node type identifier."""

    SPMD = "spmd"  # Normal SPMD node


class MailboxStatus(FluxwayBaseSchema):
    """Mailbox status for backpressure monitoring."""

    is_full: bool = False
    capacity: int = 100
    current_size: int = 0
    full_ratio: float = 0.8  # Ratio threshold to consider mailbox as full


class SPMDConfig(FluxwayBaseSchema):
    """
    Configuration for a single SPMD (Single Program, Multiple Data) execution unit.

    This defines how the SPMD operates, including its mesh, operators, and execution
    parameters.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Mesh axis configuration (e.g., ("data", 4), ("model", 2))
    spmd_id: str = ""
    graph_view: SPMDGraphView | None = Field(default=None, frozen=True)
    mesh: Any = None  # DeviceMesh (use Any to avoid circular import)

    operator_specs: dict[str, OperatorSpec] = Field(default_factory=dict)
    checkpoint_components: tuple[Any, ...] = ()

    # Static arguments for operators: {arg_name: arg_value}
    static_args: dict[str, Any] = Field(default_factory=dict)

    max_rank_step_skew: int = Field(
        default=0,
        ge=0,
        strict=True,
        description=(
            "Maximum completed-step difference between ranks in this SPMD. "
            "Zero enforces lockstep."
        ),
    )
    # Global step limit for the SPMD. If set, source operators will self-terminate
    max_steps: int | None = None

    # If true, worker will block identically at step=0 until a drain checkpoint_object is provided.
    requires_drain_at_startup: bool = False

    # If true, workers enqueue source operators immediately after topology startup.
    auto_start_source_operators: bool = True

    # Inactivity timeout threshold for checkpointing watchdog
    inactivity_timeout_s: float = DEFAULT_INACTIVITY_TIMEOUT_SECS

    def require_graph_view(self) -> SPMDGraphView:
        if self.graph_view is None:
            raise RuntimeError(
                "SPMDConfig has no graph view. Compile the graph before creating "
                "workers."
            )
        if self.graph_view.spmd_id != self.spmd_id:
            raise RuntimeError(
                f"SPMDConfig {self.spmd_id!r} received graph view for "
                f"{self.graph_view.spmd_id!r}."
            )
        return self.graph_view

    # Per-node Ray runtime_env for Driver/Worker actors.
    # Passed through to .options(runtime_env=...) when creating actors.
    # Example: {"env_vars": {"JAX_PLATFORMS": "gpu"}}
    runtime_env: dict[str, Any] | None = None

    # Dedicated runtime_env for the SPMDDriver actor.
    # The driver is a control-plane actor and MUST NOT inherit GPU-related
    # env_vars (CUDA_VISIBLE_DEVICES / JAX_PLATFORMS=cuda,...) from the
    # worker side, otherwise pickle.loads of jax.Array on the driver will
    # eagerly initialize a CUDA backend in a process that holds no GPU.
    # When None, SPMDManager applies a safe default that explicitly hides
    # GPUs from the driver actor.
    driver_runtime_env: dict[str, Any] | None = None

    # Transport backend for data plane RPC (e.g. "ray", "axon")
    transport_type: str = "ray"

    data_plane_batch_max_frames: int = 8
    data_plane_batch_flush_us: int = 100
    data_plane_max_in_flight: int = 50

    heartbeat_interval_s: float = 5.0
    # Diagnostic only. A quiet progress interval must not affect liveness.
    execution_progress_warning_s: float = 90.0
    heartbeat_timeout_s: float = 90.0

    # Optional profiler runtime configuration. ``None`` is the zero-overhead path:
    # workers construct the original executors and data-plane handlers directly.
    profiler_config: Any | None = None

    @property
    def num_workers(self) -> int:
        """Derive world size (total workers) from mesh size."""
        if self.graph_view is not None:
            return self.graph_view.world_size
        if self.mesh is None:
            return 1
        return int(self.mesh.size)


# =============================================================================
# Resource and Scheduling Types (Section 2.1.2)
# =============================================================================


# =============================================================================
# Graph Builder Types (Based on graph_example.py)
# =============================================================================


class Node(FluxwayBaseSchema):
    """
    Represents a node in the task graph.

    Each node corresponds to an SPMD compute unit with resource requirements
    and connections to other nodes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    resource_profile: ResourceProfile | None = None
    spmd_config: SPMDConfig = Field(default_factory=SPMDConfig)
    node_type: NodeType = NodeType.SPMD  # Node type: SPMD
    spmd_class: type


class OperatorContext(FluxwayBaseSchema):
    """Context provided to an operator during execution."""

    model_config = ConfigDict(frozen=True)

    physical_rank: int
    physical_world_size: int
    logical_rank: int
    logical_world_size: int
    step: int
    global_source_offset: int | None = None
    is_interleaved: bool = False
    source_work_items: Any | None = None

    async def run_in_lane(
        self,
        fn: Callable[..., Any],
        *args: Any,
        lane: str = "default",
        **kwargs: Any,
    ) -> Any:
        """Execute a blocking or CPU-bound callable in a worker-managed thread lane."""
        from fluxway.common.decorators import get_current_context

        state = get_current_context()
        if state.lane_runner is None:
            raise RuntimeError(
                f"Cannot run in lane {lane!r}: no active lane runner configured "
                "in current operator context."
            )
        return await state.lane_runner(fn, *args, lane=lane, **kwargs)


class Graph(FluxwayBaseSchema):
    """
    Complete task graph with nodes and edges.

    Built using the Builder class for programmatic construction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""
    nodes: dict[str, Node] = Field(default_factory=dict)
    compiled_graph: CompiledGraph | None = Field(default=None, frozen=True)

    def require_compiled_graph(self) -> CompiledGraph:
        if self.compiled_graph is None:
            raise RuntimeError(
                "Graph has no compiled graph. Build it with fluxway.Builder before "
                "starting execution or querying topology."
            )
        return self.compiled_graph

    @property
    def dependency_specs(self) -> tuple[DependencySpec, ...]:
        return self.require_compiled_graph().dependency_specs

    @property
    def source_nodes(self) -> tuple[str, ...]:
        return self.require_compiled_graph().source_spmd_ids

    @property
    def sink_nodes(self) -> tuple[str, ...]:
        return self.require_compiled_graph().sink_spmd_ids

    @property
    def requires_source_control_runtime(self) -> bool:
        """Whether this graph contains a descriptor-backed source operator."""
        return self.require_compiled_graph().requires_source_control_runtime

    @property
    def source_control_spmd_ids(self) -> tuple[str, ...]:
        """Canonical SPMD owners for ordered source-control publication."""
        return self.require_compiled_graph().source_control_spmd_ids


class GraphValidationResult(FluxwayBaseSchema):
    """
    Result of graph validation.

    Contains validation status and any errors/warnings found.
    """

    is_valid: bool = True
    is_acyclic: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unreachable_nodes: list[str] = Field(default_factory=list)


# =============================================================================
# State Management Types (Section 2.1.4)
# =============================================================================


class GlobalState(FluxwayBaseSchema):
    """Global training state aggregated from all SPMDs."""

    current_step: int = 0
    # TODO: Add global metrics (loss, throughput, etc.)
    # TODO: Add checkpoint metadata
    # TODO: Add resource usage statistics


class SPMDState(FluxwayBaseSchema):
    """State of a single SPMD instance."""

    spmd_id: str = ""
    current_step: int = 0
    # TODO: Add SPMD-specific metrics
    # TODO: Add model parameters reference
    # TODO: Add optimizer state reference


class WorkerTelemetry(FluxwayBaseSchema):
    """Telemetry data from a single worker."""

    worker_id: str = ""
    # TODO: Add GPU utilization
    # TODO: Add memory usage
    # TODO: Add network I/O statistics
    # TODO: Add heartbeat timestamp


# =============================================================================
# Handle Types (Section 3.2.1, 6.1)
# =============================================================================


class LogicalHandle(FluxwayBaseSchema):
    """
    Logical handle providing a stable interface.

    Logical handles don't change when nodes are evicted and restored.
    """

    logical_id: str
    spmd_id: str
    worker_id: str


class ActualHandle(FluxwayBaseSchema):
    """
    Actual handle corresponding to a Ray Actor.

    May change when nodes are evicted and rebuilt.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    actor_handle: Any  # ray.actor.ActorHandle
    worker_id: str
    created_at: float = 0.0  # Creation timestamp


class NodeView(FluxwayBaseSchema):
    """
    Worker's view of the SPMD topology.

    Maintains logical handle mappings for predecessors and successors.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_spmd_id: str = ""
    current_worker_id: str = ""
    predecessor_logical_handles: dict[str, list[LogicalHandle]] = Field(
        default_factory=dict
    )
    successor_logical_handles: dict[str, list[LogicalHandle]] = Field(
        default_factory=dict
    )
    handle_mapping: dict[str, ActualHandle] = Field(default_factory=dict)

    def get_predecessor_handles(self, spmd_id: str) -> list[LogicalHandle]:
        """Get all logical handles for a predecessor SPMD."""
        # TODO: Implement handle lookup
        return self.predecessor_logical_handles.get(spmd_id, [])

    def get_successor_handles(self, spmd_id: str) -> list[LogicalHandle]:
        """Get all logical handles for a successor SPMD."""
        # TODO: Implement handle lookup
        return self.successor_logical_handles.get(spmd_id, [])

    def call_logical_handle(self, logical_id: str, method: str, *args, **kwargs):
        """
        Call a method through logical handle, auto-mapping to actual handle.

        TODO: Implement auto-mapping logic
        TODO: Handle node eviction and recovery
        TODO: Implement retry mechanism
        """

    def update_handle_mapping(self, logical_id: str, actual_handle: ActualHandle):
        """Update handle mapping when node changes."""
        # TODO: Implement mapping update

    def handle_node_eviction(self, spmd_id: str):
        """Handle node eviction and trigger recovery."""
        # TODO: Implement eviction handling


# =============================================================================
# Driver and Worker Status Types (for heartbeat and state management)
# =============================================================================


class DriverStatus(StrEnum):
    """Driver health status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    TIMEOUT = "timeout"


class WorkerStatus(StrEnum):
    """Worker operational status."""

    INITIALIZING = "initializing"
    READY = "ready"
    RESTORING = "restoring"
    STOPPED = "stopped"
    RUNNING = "running"
    BLOCKED = "blocked"
    ERROR = "error"


class ExecutionWaitState(StrEnum):
    """Worker execution wait classification for diagnostics only."""

    IDLE = "idle"
    OPERATOR_EXECUTION_ACTIVE = "operator_execution_active"
    OPERATOR_ADMISSION_WAIT = "operator_admission_wait"
    SOURCE_ADMISSION_WAIT = "source_admission_wait"
    OUTBOUND_TRANSPORT_WAIT = "outbound_transport_wait"
    INPUT_PREPARE_WAIT = "input_prepare_wait"
    EXECUTION_QUEUE_WAIT = "execution_queue_wait"
    COMPLETION_BARRIER_OR_DRIVER_WAIT = "completion_barrier_or_driver_wait"


class CheckpointActivityPhase(StrEnum):
    """Worker-local checkpoint activity reported in heartbeats."""

    IDLE = "idle"
    CUTTING = "cutting"
    CAPTURING = "capturing"
    STAGING = "staging"
    COMMITTING = "committing"
    PUBLISHED = "published"
    FAILED = "failed"


class ControllerState(StrEnum):
    """Controller lifecycle phases coordinated by ExecutionLifecycle."""

    INITIALIZING = "initializing"  # Allocating resources, starting SPMDs
    PENDING = "pending"  # Waiting for all drivers to init (workers_ready)
    TOPOLOGY_REGISTERING = "topology_registering"  # topology registration in progress
    READY = "ready"  # Topology ready, waiting for start
    ACTIVATING = "activating"
    RESUMING = "resuming"
    RUNNING = "running"  # Normal execution
    COMPLETING = "completing"  # Final cleanup after successful execution
    FAILING = "failing"  # Final cleanup after unrecoverable failure
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"  # Task completed
    FAILED = "failed"  # Task failed


class ControllerEvent(StrEnum):
    """Events that trigger controller state transitions."""

    INIT_COMPLETE = "init_complete"  # Resources allocated, SPMDs started
    ALL_DRIVERS_READY = "all_drivers_ready"  # All drivers report workers_ready
    ALL_SPMDS_TOPOLOGY_READY = (
        "all_spmds_topology_ready"  # All worker topologies ready after topology
    )
    START = "start"  # Start execution
    ACTIVATION_COMPLETE = "activation_complete"
    RESUME = "resume"
    DRIVER_ERROR = "driver_error"  # Driver heartbeat timeout or error
    ALL_DRIVERS_COMPLETED = "all_drivers_completed"  # All drivers report completion
    CLEANUP_FINISHED = "cleanup_finished"  # Terminal cleanup completed
    CLEANUP_FAILED = "cleanup_failed"  # Terminal cleanup failed
    CANCEL = "cancel"


class DriverState(StrEnum):
    """SPMD Driver state machine states."""

    INITIALIZING = "initializing"  # Initializing workers
    READY = "ready"  # Ready, waiting for Controller command
    RESTORING = "restoring"  # Restoring a published checkpoint
    ACTIVATING = "activating"
    RUNNING = "running"  # Running
    BLOCKED = "blocked"  # Blocked
    DRAINING = "draining"  # Draining to checkpoint_object step
    DRAINED = "drained"  # Reached the drain checkpoint_object step
    SNAPSHOTTING = "snapshotting"  # Extracting operator states via save_state
    ERROR = "error"  # Error state
    COMPLETED = "completed"  # Completed
    STOPPING = "stopping"
    STOPPED = "stopped"


class DriverEvent(StrEnum):
    """Events that trigger driver state transitions."""

    INIT_COMPLETE = "init_complete"
    INIT_FAILED = "init_failed"
    START = "start"
    ACTIVATION_COMPLETE = "activation_complete"
    WORKER_ERROR = "worker_error"
    COMPONENT_ERROR = "component_error"
    BLOCK_REQUESTED = "block_requested"
    UNBLOCK_REQUESTED = "unblock_requested"
    ALL_WORKERS_COMPLETED = "all_workers_completed"
    COMPLETE = "complete"
    DRAIN_REQUESTED = "drain_requested"
    DRAIN_COMPLETE = "drain_complete"
    SNAPSHOT_REQUESTED = "snapshot_requested"
    RESUME = "resume"
    SNAPSHOT_FAILED = "snapshot_failed"
    RESTORE_REQUESTED = "restore_requested"
    RESTORE_COMPLETE = "restore_complete"
    RESTORE_FAILED = "restore_failed"
    STOP = "stop"
    STOPPED = "stopped"


class _FailureHeartbeat[StatusT](FluxwayBaseSchema):
    status: StatusT
    failure: ExecutionFailure | None = None

    @model_validator(mode="after")
    def validate_failure_status(self):
        is_error = self.status in (WorkerStatus.ERROR, DriverStatus.ERROR)
        if is_error != (self.failure is not None):
            raise ValueError("ERROR heartbeat status and failure must occur together.")
        return self


class WorkerHeartbeat(_FailureHeartbeat[WorkerStatus]):
    """Worker heartbeat data."""

    rank: int
    timestamp: float
    status: WorkerStatus = WorkerStatus.RUNNING
    checkpoint_phase: CheckpointActivityPhase = CheckpointActivityPhase.IDLE
    checkpoint_progress_timestamp: float = 0.0
    checkpoint_id: str | None = None
    checkpoint_completed_units: int = 0
    # Opaque execution metadata for cross-framework diagnostics. Fluxway records
    # it and never interprets framework-specific phases or compilation states.
    execution_generation: int = 0
    execution_operator: str | None = None
    execution_step: int | None = None
    execution_epoch: int = 0
    execution_sequence: int = 0
    execution_in_flight: bool = False
    execution_phase: str | None = None
    execution_phase_age_s: float = 0.0
    execution_activity_age_s: float = 0.0
    committed_progress_age_s: float = 0.0
    execution_wait_state: ExecutionWaitState = ExecutionWaitState.IDLE


class DriverHeartbeat(_FailureHeartbeat[DriverStatus]):
    """Driver heartbeat data reported to Controller."""

    spmd_id: str
    timestamp: float
    status: DriverStatus = DriverStatus.HEALTHY
    state: DriverState = DriverState.INITIALIZING
    num_workers: int = 0
    healthy_num_workers: int = 0
    workers_ready: bool = False  # All workers have reported heartbeat
    topology_ready: bool = False


class FlowControlState(FluxwayBaseSchema):
    """Driver flow-control state reported outside the heartbeat channel."""

    spmd_id: str
    timestamp: float
    paused: bool = False
    epoch: int = 0


class HeartbeatConfig(FluxwayBaseSchema):
    """Heartbeat configuration."""

    heartbeat_interval: float = 10.0  # Heartbeat interval (seconds)
    heartbeat_timeout: float = 600.0  # Heartbeat timeout (seconds)
    max_missed_heartbeats: int = 60  # Maximum missed heartbeats before error


DEFAULT_SOURCE_MAX_IN_FLIGHT_STEPS: int = 100
DEFAULT_SOURCE_PLANNING_WINDOW_STEPS: int = 100
DEFAULT_SOURCE_MAX_PLANNING_RECORDS: int = 1_000_000
DEFAULT_SOURCE_MAX_PLANNING_UNITS: int = 100_000
DEFAULT_SOURCE_MAX_PLANNING_DESCRIPTOR_BYTES: int = 64 * 1024 * 1024


class SourceControlConfig(FluxwayBaseSchema):
    """Configuration for bounded logical source work acquisition."""

    model_config = ConfigDict(extra="forbid")

    max_in_flight_steps: int = Field(
        default=DEFAULT_SOURCE_MAX_IN_FLIGHT_STEPS, gt=0, strict=True
    )
    planning_window_steps: int = Field(
        default=DEFAULT_SOURCE_PLANNING_WINDOW_STEPS, gt=0, strict=True
    )
    max_planning_records: int = Field(
        default=DEFAULT_SOURCE_MAX_PLANNING_RECORDS, gt=0, strict=True
    )
    max_planning_units: int = Field(
        default=DEFAULT_SOURCE_MAX_PLANNING_UNITS, gt=0, strict=True
    )
    max_planning_descriptor_bytes: int = Field(
        default=DEFAULT_SOURCE_MAX_PLANNING_DESCRIPTOR_BYTES, gt=0, strict=True
    )


class CheckpointConfig(FluxwayBaseSchema):
    """Configuration for checkpoint manager."""

    repository_dir: str
    interval_steps: int
    snapshot_timeout_s: float = Field(
        default_factory=lambda: float(
            os.environ.get(
                "FLUXWAY_SNAPSHOT_TIMEOUT_SECS",
                str(DEFAULT_SNAPSHOT_TIMEOUT_SECS),
            )
        )
    )
    inactivity_timeout_s: float = DEFAULT_INACTIVITY_TIMEOUT_SECS
    retention_policy: CheckpointRetentionPolicy = Field(
        default_factory=CheckpointRetentionPolicy
    )
    retention_domains: dict[str, CheckpointRetentionPolicy] = Field(
        default_factory=dict
    )
    restore_scope: tuple[str, ...] | None = None
    restore_spmd_bindings: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_checkpoint_config(self):
        if not self.repository_dir:
            raise ValueError("repository_dir must be non-empty")
        if self.interval_steps < 1:
            raise ValueError("interval_steps must be positive")
        if self.snapshot_timeout_s <= 0:
            raise ValueError("snapshot_timeout_s must be positive")
        if self.inactivity_timeout_s <= 0:
            raise ValueError("inactivity_timeout_s must be positive")
        if any(not domain for domain in self.retention_domains):
            raise ValueError("retention_domains keys must be non-empty")
        if self.restore_scope is not None:
            if not self.restore_scope:
                raise ValueError("restore_scope must not be empty when provided")
            if any(not value for value in self.restore_scope):
                raise ValueError("restore_scope entries must be non-empty")
            if len(set(self.restore_scope)) != len(self.restore_scope):
                raise ValueError("restore_scope entries must be unique")
        if any(
            not source or not target
            for source, target in self.restore_spmd_bindings.items()
        ):
            raise ValueError("restore_spmd_bindings requires non-empty IDs")
        if len(set(self.restore_spmd_bindings.values())) != len(
            self.restore_spmd_bindings
        ):
            raise ValueError("restore_spmd_bindings targets must be unique")
        if self.restore_scope is not None and not set(
            self.restore_spmd_bindings
        ).issubset(self.restore_scope):
            raise ValueError(
                "restore_spmd_bindings sources must belong to restore_scope"
            )
        return self


class AsyncStepConfig(FluxwayBaseSchema):
    """Configuration for async step difference control."""

    max_steps: int | None = None  # Maximum steps to run (global limit)
    node_max_steps: dict[str, int] = Field(
        default_factory=dict
    )  # Per-node step overrides
    default_max_inflight_steps: int = (
        1  # Maximum generic step difference for flow control
    )
    min_lock_duration: float = (
        1.0  # Minimum lock duration (prevent frequent lock/unlock)
    )


class DriverResourceSpec(FluxwayBaseSchema):
    """Driver management resource configuration (system default)."""

    num_cpus: float = 1.0
    memory_gb: float = 1.0


class SPMDOpStep(FluxwayBaseSchema):
    """SPMD operation step tracking."""

    spmd_id: str
    completed_step_count: dict[str, int] = Field(
        default_factory=dict
    )  # op_name -> completed count


class OpLockState(FluxwayBaseSchema):
    """Operation lock state."""

    spmd_id: str
    op_name: str  # Operator name
    locked_at: float
    reason: str
