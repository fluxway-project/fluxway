"""
Core data type definitions for Fluxway framework.

Based on ARCHITECTURE.md sections 2.1.1, 3.2.1, 4.2, 4.3
"""

from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Any,
    Type,
    Annotated,
    TypeVar,
    Union,
    Tuple,
)

from pydantic import BaseModel, Field, ConfigDict


from enum import Enum

T = TypeVar("T")


# Transport stream markers for fields within a @data_type
class DataFlow:
    """Marker for Worker-to-Worker (Data Plane) transport. Supports sharding."""

    pass


class ControlFlow:
    """Marker for Driver-to-Driver (Control Plane) transport. Broadcast to all workers."""

    pass


# User-facing type aliases
# Usage: field_name: Data[jnp.ndarray] or field_name: Control[float]
Data = Annotated[T, DataFlow]
Control = Annotated[T, ControlFlow]


class FluxwayBaseSchema(BaseModel):
    """Base model for all Fluxway configuration and metadata types."""

    def as_dict(self, **kwargs) -> Dict[str, Any]:
        """
        Convert model to dictionary.
        Renamed from Pydantic's model_dump for better domain alignment.
        """
        return self.model_dump(**kwargs)


class TensorProxy(FluxwayBaseSchema):
    """
    A lightweight proxy representing a tensor in the computational graph.

    It uses a globally unique identifier (int TUUID) to track dependencies and
    sharding specifications independently of the variable names used in
    operator functions.
    """

    tuuid: int
    name: Optional[str] = None  # Original name for debugging
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None

    def __repr__(self):
        return f"TensorProxy(tuuid={self.tuuid}, name={self.name})"


class NodeType(str, Enum):
    """Node type identifier."""

    SPMD = "spmd"  # Normal SPMD node


class ReaderState(str, Enum):
    """Reader state for data reading nodes."""

    INITIALIZING = "initializing"
    READING = "reading"
    PAUSED = "paused"  # Paused due to backpressure
    COMPLETED = "completed"
    ERROR = "error"


class MailboxStatus(FluxwayBaseSchema):
    """Mailbox status for backpressure monitoring."""

    is_full: bool = False
    capacity: int = 100
    current_size: int = 0
    full_ratio: float = 0.8  # Ratio threshold to consider mailbox as full


class SPMDTopologyInfo(FluxwayBaseSchema):
    """Information about a predecessor/successor SPMD in the graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spmd_id: str
    # Physical scale of the remote node
    world_size: int = 1
    mesh: Any = None
    # TUUID-based sharding mapping: {tuuid: ShardingSpec}
    tuuid_sharding: Dict[int, Any] = Field(default_factory=dict)
    # Map of local parameter/argument names to their assigned UUIDs.
    # {param_name: tuuid}
    tuuid_map: Dict[str, int] = Field(default_factory=dict)
    # Optional condition function for conditional edges
    condition_fn: Optional[Callable[..., bool]] = None
    # Link to the symmetric edge (src_node, src_op, dst_node, dst_op)
    symmetric_to: Optional[Tuple[str, str, str, str]] = None
    # Whether this remote node is an interleaved source
    is_interleaved: bool = False
    # Size of the interleaved group (if applicable)
    interleave_unit_size: int = 1


class SPMDTopology(FluxwayBaseSchema):
    """Topology view for a single operator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Predecessors of this operator: {spmd_id: info}
    predecessors: Dict[str, SPMDTopologyInfo] = Field(default_factory=dict)
    # Successors of this operator: {spmd_id: info}
    successors: Dict[str, SPMDTopologyInfo] = Field(default_factory=dict)


class SPMDConfig(FluxwayBaseSchema):
    """
    Configuration for a single SPMD (Single Program, Multiple Data) execution unit.

    This defines how the SPMD operates, including its mesh, operators, and execution
    parameters.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Mesh axis configuration (e.g., ("data", 4), ("model", 2))
    spmd_id: str = ""
    mesh: Any = None  # DeviceMesh (use Any to avoid circular import)

    # Generic operator registration
    operators: Dict[str, Callable] = Field(default_factory=dict)
    # Explicit OperatorMeta storage - ensures metadata survives Ray serialization
    operator_metas: Dict[str, "OperatorMeta"] = Field(default_factory=dict)

    # Topology of the graph for each operator
    # {op_name: SPMDTopology}
    topology: Dict[str, "SPMDTopology"] = Field(default_factory=dict)

    # Static arguments for operators: {arg_name: arg_value}
    static_args: Dict[str, Any] = Field(default_factory=dict)

    max_worker_step_skew: int = (
        0  # Max allowed step diff between workers in a mesh (intra-SPMD sync)
    )
    # Global step limit for the SPMD. If set, source operators will self-terminate
    max_steps: Optional[int] = None

    # If true, worker will block identically at step=0 until a drain target is provided.
    requires_drain_at_startup: bool = False

    @property
    def num_workers(self) -> int:
        """Derive world size (total workers) from mesh size."""
        if self.mesh is None:
            return 1
        return int(self.mesh.size)


# =============================================================================
# Resource and Scheduling Types (Section 2.1.2)
# =============================================================================


class ResourceSpec(FluxwayBaseSchema):
    """Resource specification for SPMD scheduling."""

    num_gpus: int = 0
    num_cpus: int = 1
    memory_gb: float = 0.0
    count: int = 1
    # TODO: Add network bandwidth requirements
    # TODO: Add storage requirements


class PlacementStrategy(str, Enum):
    """Placement strategy for resource allocation."""

    PACK = "PACK"  # Pack bundles on as few nodes as possible
    SPREAD = "SPREAD"  # Spread bundles across nodes
    STRICT_PACK = "STRICT_PACK"  # Strict packing requirement
    STRICT_SPREAD = "STRICT_SPREAD"  # Strict spreading requirement


class ResourcePriority(str, Enum):
    """Resource priority for allocation and preemption decisions."""

    HIGH = "HIGH"  # High priority, less likely to be preempted
    MID = "MID"  # Medium priority (default)
    LOW = "LOW"  # Low priority, may be preempted first


class PlacementPlan(FluxwayBaseSchema):
    """Resource placement plan generated by scheduler."""

    # Mapping of SPMD ID to resource requirements
    spmd_placements: Dict[str, "ResourceSpec"] = Field(default_factory=dict)

    # Mapping of SPMD ID to resource priority
    spmd_priorities: Dict[str, ResourcePriority] = Field(default_factory=dict)

    # Shared placement groups for conditional branches
    # Key: shared_pg_id, Value: list of SPMD IDs sharing this PG
    shared_pgs: Dict[str, List[str]] = Field(default_factory=dict)

    # Placement strategy
    strategy: PlacementStrategy = PlacementStrategy.PACK


# =============================================================================
# Graph Builder Types (Based on graph_example.py)
# =============================================================================


class Edge(FluxwayBaseSchema):
    """
    Represents an edge in the task graph.

    Edges connect nodes and can specify source/target op types.
    This allows flexible connections like forward->backward for sink nodes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_node: str
    target_node: str
    source_op: Optional[str] = None  # Source node's op name
    target_op: Optional[str] = None  # Target node's op name
    condition_fn: Optional[Callable[..., bool]] = None

    # Link to the symmetric edge (for automatic mirror-sharding based routing)
    # Format: (source_node, source_op, target_node, target_op)
    symmetric_to: Optional[Tuple[str, str, str, str]] = None


class Node(FluxwayBaseSchema):
    """
    Represents a node in the task graph.

    Each node corresponds to an SPMD execution unit with resource requirements
    and connections to other nodes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    resource: Optional[ResourceSpec] = None
    priority: Optional[ResourcePriority] = None  # Resource priority for allocation
    spmd_config: SPMDConfig = Field(default_factory=SPMDConfig)
    node_type: NodeType = NodeType.SPMD  # Node type: SPMD


class ArgumentType(str, Enum):
    """Argument type for fluxway transfer data."""

    CONTROL = "control"  # Control argument
    DATA = "data"  # Data argument
    STATIC = "static"  # Static argument


class EdgeShardingInfo(FluxwayBaseSchema):
    """Sharding information for a single edge between SPMDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    world_size: int = 1
    mesh: Any = None
    # New TUUID-based sharding mapping: {tuuid: ShardingSpec}
    tuuid_sharding: Dict[int, Any] = Field(default_factory=dict)
    # Map of local parameter/argument names to their assigned UUIDs.
    tuuid_map: Dict[str, int] = Field(default_factory=dict)
    # Condition function for conditional edges
    condition_fn: Optional[Callable[..., bool]] = None
    # output_param_map: source op's output mapping (sender side)
    # input_param_map: target op's input mapping (receiver side)
    output_param_map: Dict[str, List[str]] = Field(default_factory=dict)
    input_param_map: Dict[str, List[str]] = Field(default_factory=dict)
    is_interleaved: bool = False
    interleave_unit_size: int = 1


class BarrierType(str, Enum):
    """Execution barrier types for operators."""

    STANDARD = "standard"  # Wait for all expected shards/states (default)
    PERIODIC = "periodic"  # Fire every 'n' arrivals; used for streaming
    TOLERANT = "tolerant"  # Wait for bounded stale data, used for off-policy


class BarrierConfig(FluxwayBaseSchema):
    """Base configuration for operator execution barriers."""

    type: BarrierType = BarrierType.STANDARD


class StandardBarrierConfig(BarrierConfig):
    """Standard barrier: fires when all expected data for a step arrives."""

    type: BarrierType = BarrierType.STANDARD


class PeriodicBarrierConfig(BarrierConfig):
    """
    Periodic barrier: fires every 'n' arrivals regardless of global step boundaries.
    Commonly used for streaming or 'eager-like' processing where n=1.
    """

    type: BarrierType = BarrierType.PERIODIC
    n: int = 1


class TolerantBarrierConfig(BarrierConfig):
    """
    Tolerant barrier: fires when data within allowable staleness window is available.
    Commonly used for Off-Policy / SSP training.
    """

    type: BarrierType = BarrierType.TOLERANT
    tensor_staleness: Dict[int, int] = Field(default_factory=dict)
    state_staleness: Dict[str, int] = Field(default_factory=dict)
    debug_names: Dict[int, str] = Field(default_factory=dict, exclude=True)
    retrieval_policy: str = "latest"  # "latest", "oldest", "closest"

    # Input field for logical paths (resolved during initialization)
    staleness: Dict[str, int] = Field(default_factory=dict, exclude=True)

    def resolve(self, resolver_fn: Callable[[str], Optional[int]]) -> None:
        """Resolve all stale paths to TUUIDs or Control States."""
        for key, val in self.staleness.items():
            tuuid = resolver_fn(key)
            if tuuid is not None:
                self.tensor_staleness[tuuid] = val
                self.debug_names[tuuid] = key
            else:
                # Might be a control state, preserve for runtime check in OperatorManager
                self.state_staleness[key] = val


class OperatorContext(FluxwayBaseSchema):
    """Context provided to an operator during execution."""

    physical_rank: int
    physical_world_size: int
    logical_rank: int
    logical_world_size: int
    step: int
    is_interleaved: bool = False


class OperatorMeta(FluxwayBaseSchema):
    """Metadata attached to an operator function."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    owner_class_qualname: str = ""
    owner_class_module: str = ""

    is_source: Optional[bool] = None  # Explicitly mark as source (producer)
    # Whether this is an interleaved source.
    is_interleaved: bool = False
    # Size of the interleaved group (e.g., 1 for single-worker).
    interleave_unit_size: int = 1

    priority: int = 0  # Priority for execution order (lower value -> higher priority)
    max_inflight_steps: int = (
        0  # Max steps predecessors can lead (backpressure_threshold)
    )
    # Keys use ArgumentType.value (i.e. "state" | "tensor" | "static") to avoid enum
    # serialization / circular import surprises across control-plane and worker.
    input_arguments: Optional[Dict[str, List[str]]] = None
    output_arguments: Optional[Dict[str, List[str]]] = None
    input_sharding: Dict[Any, Any] = Field(default_factory=dict)
    output_sharding: Dict[Any, Any] = Field(default_factory=dict)
    output_uuids: List[int] = Field(default_factory=list)

    # Implicit parameter-to-field mappings
    input_param_map: Dict[str, List[str]] = Field(default_factory=dict)
    output_param_map: Dict[str, List[str]] = Field(default_factory=dict)
    input_treedefs: Dict[str, Any] = Field(default_factory=dict)
    output_treedef: Optional[Any] = None

    # Linking metadata (populated by Builder)
    input_pre_sharding: Dict[str, List[EdgeShardingInfo]] = Field(default_factory=dict)
    output_post_sharding: Dict[str, List[EdgeShardingInfo]] = Field(
        default_factory=dict
    )

    # Execution barrier configuration
    barrier_config: Union[
        StandardBarrierConfig, PeriodicBarrierConfig, TolerantBarrierConfig
    ] = Field(default_factory=StandardBarrierConfig)

    def resolve_sharding_topology(
        self, resolver_fn: Callable[[str], Optional[int]]
    ) -> None:
        for sharding in [self.input_sharding, self.output_sharding]:
            for spec in sharding.values():
                # ShardingSpec handles both referential and algebraic logic
                if hasattr(spec, "routing") and hasattr(spec.routing, "resolve"):
                    spec.routing.resolve(resolver_fn)

    def resolve_async_strategy(
        self, resolver_fn: Callable[[str], Optional[int]]
    ) -> None:
        if hasattr(self.barrier_config, "resolve"):
            self.barrier_config.resolve(resolver_fn)

    def get_arguments(self, is_output: bool, arg_type: "ArgumentType") -> List[str]:
        args_map = self.output_arguments if is_output else self.input_arguments

        if args_map and arg_type.value in args_map:
            return list(args_map[arg_type.value])

        if arg_type == ArgumentType.DATA:
            sharding = self.output_sharding if is_output else self.input_sharding
            return list(sharding.keys())

        return []

    # ---------------------------------------------------------------------
    # Compare / validate helpers (used by control-plane graph builder)
    # ---------------------------------------------------------------------

    def state_names(self, is_output: bool) -> List[str]:
        """Get state argument names from input/output arguments."""
        return self.get_arguments(is_output=is_output, arg_type=ArgumentType.CONTROL)

    def validate_and_update_state_output_match_dest_input(
        self,
        dst_meta: "OperatorMeta",
        src_spmd_class: Type,
        src_method_name: str,
        ctx: str,
    ) -> None:
        """
            Validate state arguments between src output and dst input.

            Logic:
        - Compare src.output_arguments.state with dst.input_arguments.state (must match exactly)
        - If src.output_arguments.state is missing, infer from src return type fields,
          require that output fields contain all dst.input_arguments.state,
          then update src.output_arguments.state = dst.input_arguments.state
        """
        from fluxway.common.utils import infer_fluxway_output_fields

        dst_input_state = set(dst_meta.state_names(is_output=False))
        src_output_state_list = self.state_names(is_output=True)

        if src_output_state_list:
            src_output_state = set(src_output_state_list)
            if src_output_state != dst_input_state:
                raise ValueError(
                    f"State names mismatch in {ctx}: "
                    f"src_output_state={sorted(src_output_state)}, "
                    f"dst_input_state={sorted(dst_input_state)}"
                )
            return

        # src.output_state not specified: infer from return type fields
        output_fields = set(
            infer_fluxway_output_fields(src_spmd_class, src_method_name)
        )
        missing = dst_input_state - output_fields
        if missing:
            raise ValueError(
                f"State coverage mismatch in {ctx}: operator output type of "
                f"{src_spmd_class.__name__}.{src_method_name} is missing state fields "
                f"{sorted(missing)} required by destination input"
            )

        # Update src.output_arguments.state = dst.input_state
        if self.output_arguments is None:
            self.output_arguments = {}
        self.output_arguments[ArgumentType.CONTROL.value] = list(dst_input_state)


class Graph(FluxwayBaseSchema):
    """
    Complete task graph with nodes and edges.

    Built using the Builder class for programmatic construction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""
    nodes: Dict[str, Node] = Field(default_factory=dict)
    edges: List[Edge] = Field(default_factory=list)
    source_nodes: List[str] = Field(default_factory=list)  # Entry nodes (no upstream)
    sink_nodes: List[str] = Field(default_factory=list)  # Exit nodes (no downstream)


class GraphValidationResult(FluxwayBaseSchema):
    """
    Result of graph validation.

    Contains validation status and any errors/warnings found.
    """

    is_valid: bool = True
    is_acyclic: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    unreachable_nodes: List[str] = Field(default_factory=list)


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
    predecessor_logical_handles: Dict[str, List[LogicalHandle]] = Field(
        default_factory=dict
    )
    successor_logical_handles: Dict[str, List[LogicalHandle]] = Field(
        default_factory=dict
    )
    handle_mapping: Dict[str, ActualHandle] = Field(default_factory=dict)

    def get_predecessor_handles(self, spmd_id: str) -> List[LogicalHandle]:
        """Get all logical handles for a predecessor SPMD."""
        # TODO: Implement handle lookup
        return self.predecessor_logical_handles.get(spmd_id, [])

    def get_successor_handles(self, spmd_id: str) -> List[LogicalHandle]:
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
        pass

    def update_handle_mapping(self, logical_id: str, actual_handle: ActualHandle):
        """Update handle mapping when node changes."""
        # TODO: Implement mapping update
        pass

    def handle_node_eviction(self, spmd_id: str):
        """Handle node eviction and trigger recovery."""
        # TODO: Implement eviction handling
        pass


# =============================================================================
# Driver and Worker Status Types (for heartbeat and state management)
# =============================================================================


class DriverStatus(str, Enum):
    """Driver health status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    TIMEOUT = "timeout"


class WorkerStatus(str, Enum):
    """Worker operational status."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    BLOCKED = "blocked"
    ERROR = "error"


class ControllerState(str, Enum):
    """Controller state machine states.

    Lifecycle:
      INITIALIZING → PENDING → TOPOLOGY_REGISTERING → RUNNING
                                                       ↓
                                                    BLOCKED ↔ RECOVERING
                                                       ↓
                                                  COMPLETED / FAILED
    """

    INITIALIZING = "initializing"  # Allocating resources, starting SPMDs
    PENDING = "pending"  # Waiting for all drivers to init (workers_ready)
    TOPOLOGY_REGISTERING = "topology_registering"  # topology registration in progress
    RUNNING = "running"  # Normal execution
    BLOCKED = "blocked"  # Task blocked (handling exceptions)
    RECOVERING = "recovering"  # Recovering from errors
    COMPLETED = "completed"  # Task completed
    FAILED = "failed"  # Task failed


class ControllerEvent(str, Enum):
    """Events that trigger controller state transitions."""

    INIT_COMPLETE = "init_complete"  # Resources allocated, SPMDs started
    ALL_DRIVERS_READY = "all_drivers_ready"  # All drivers report workers_ready
    ALL_SPMDS_TOPOLOGY_READY = (
        "all_spmds_topology_ready"  # All workers running after topology
    )
    DRIVER_ERROR = "driver_error"  # Driver heartbeat timeout or error
    DRIVER_RECOVERED = "driver_recovered"  # Abnormal driver recovered
    ALL_DRIVERS_COMPLETED = "all_drivers_completed"  # All drivers report completion
    ALL_DRIVERS_DRAINED = "all_drivers_drained"  # All drivers reached DRAINED state
    RECOVERY_FAILED = "recovery_failed"  # Recovery failed


class DriverState(str, Enum):
    """SPMD Driver state machine states."""

    INITIALIZING = "initializing"  # Initializing workers
    READY = "ready"  # Ready, waiting for Controller command
    RUNNING = "running"  # Running
    BLOCKED = "blocked"  # Blocked
    DRAINING = "draining"  # Draining to target step
    DRAINED = "drained"  # Reached the drain target step
    SNAPSHOTTING = "snapshotting"  # Extracting operator states via save_state
    ERROR = "error"  # Error state
    RECOVERING = "recovering"  # Recovering
    COMPLETED = "completed"  # Completed


class DriverEvent(str, Enum):
    """Events that trigger driver state transitions."""

    INIT_COMPLETE = "init_complete"
    INIT_FAILED = "init_failed"
    START = "start"
    WORKER_ERROR = "worker_error"
    WORKER_RECOVERED = "worker_recovered"
    BLOCK_REQUESTED = "block_requested"
    UNBLOCK_REQUESTED = "unblock_requested"
    ALL_WORKERS_COMPLETED = "all_workers_completed"
    COMPLETE = "complete"
    DRAIN_REQUESTED = "drain_requested"
    DRAIN_COMPLETE = "drain_complete"
    SNAPSHOT_REQUESTED = "snapshot_requested"
    SNAPSHOT_COMPLETE = "snapshot_complete"
    SNAPSHOT_FAILED = "snapshot_failed"


class WorkerHeartbeat(FluxwayBaseSchema):
    """Worker heartbeat data."""

    rank: int
    timestamp: float
    completed_step_count: Dict[str, int] = Field(
        default_factory=dict
    )  # op_type.name -> completed count
    status: WorkerStatus = WorkerStatus.RUNNING
    error_message: Optional[str] = None
    telemetry: Optional["WorkerTelemetry"] = None
    mailbox_status: Optional[MailboxStatus] = None  # Mailbox status for backpressure


class DriverHeartbeat(FluxwayBaseSchema):
    """Driver heartbeat data reported to Controller."""

    spmd_id: str
    timestamp: float
    status: DriverStatus = DriverStatus.HEALTHY
    state: DriverState = DriverState.INITIALIZING
    error_message: Optional[str] = None
    completed_step_count: Dict[str, int] = Field(
        default_factory=dict
    )  # op_type.name -> completed step count
    num_workers: int = 0
    healthy_num_workers: int = 0
    blocked_ops: List[str] = Field(default_factory=list)
    workers_ready: bool = False  # All workers have reported heartbeat
    workers_running: bool = False  # All workers are running main loop
    is_downstream_busy: bool = (
        False  # Whether downstream SPMDs are busy (for backpressure)
    )
    source_state: Optional[ReaderState] = None  # Generic source state
    busy_worker_ratio: float = 0.0  # Ratio of workers with full mailbox


class HeartbeatConfig(FluxwayBaseSchema):
    """Heartbeat configuration."""

    heartbeat_interval: float = 5.0  # Heartbeat interval (seconds)
    heartbeat_timeout: float = 15.0  # Heartbeat timeout (seconds)
    max_missed_heartbeats: int = 3  # Maximum missed heartbeats before error


class CheckpointConfig(FluxwayBaseSchema):
    """Configuration for checkpoint manager."""

    backend: str  # Validated at runtime by WorkerCheckpointBackend.create()
    ckpt_dir: str
    interval_steps: int
    force_write: bool = False  # Force write even if directory already exists


class AsyncStepConfig(FluxwayBaseSchema):
    """Configuration for async step difference control."""

    max_steps: Optional[int] = None  # Maximum steps to run (global limit)
    node_max_steps: Dict[str, int] = Field(
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

    num_cpus: int = 1
    memory_gb: float = 1.0


class SPMDOpStep(FluxwayBaseSchema):
    """SPMD operation step tracking."""

    spmd_id: str
    completed_step_count: Dict[str, int] = Field(
        default_factory=dict
    )  # op_name -> completed count


class OpLockState(FluxwayBaseSchema):
    """Operation lock state."""

    spmd_id: str
    op_name: str  # Operator name
    locked_at: float
    reason: str
