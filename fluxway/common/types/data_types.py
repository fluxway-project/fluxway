"""
Core data type definitions for Fluxway framework.

Based on ARCHITECTURE.md sections 2.1.1, 3.2.1, 4.2, 4.3
"""

from enum import IntEnum
from typing import Callable, Dict, List, Optional, Any, Tuple, Type

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator


# =============================================================================
# Graph Related Types (Section 2.1.1)
# =============================================================================

from enum import Enum

from fluxway.common.types.sharding import ShardingSpec


class OpType(IntEnum):
    """Operator type for execution phases.

    name - op type
    value - priority for execution order (lower value -> higher priority)
    """

    # TODO(Chiave): Support user-defined name and priority for each op_type
    FORWARD = 0
    BACKWARD = 1
    UPDATE = 2


class NodeType(str, Enum):
    """Node type identifier."""

    SPMD = "spmd"  # Normal SPMD node
    READER = "reader"  # Data reader node


class ReaderState(str, Enum):
    """Reader state for data reading nodes."""

    INITIALIZING = "initializing"
    READING = "reading"
    PAUSED = "paused"  # Paused due to backpressure
    COMPLETED = "completed"
    ERROR = "error"


class ReaderConfig(BaseModel):
    """Reader configuration for data reading nodes."""

    src_dir: str  # Source data directory
    data_type: str = "json"  # Data type: json, parquet, csv, etc.
    batch_size: int = 1000  # Batch size for reading
    file_pattern: str = "*"  # File pattern for matching
    worker_count: int = 1  # Number of reader workers
    encoding: str = "utf-8"  # File encoding
    tensor_name: Optional[str] = (
        None  # Tensor field name extracted from output_data_type (set during build)
    )
    tensor_type: Optional[Type] = (
        None  # Tensor field type extracted from output_data_type (set during build)
    )


class MailboxStatus(BaseModel):
    """Mailbox status for backpressure monitoring."""

    is_full: bool = False
    capacity: int = 100
    current_size: int = 0
    full_ratio: float = 0.8  # Ratio threshold to consider mailbox as full


class SPMDConfig(BaseModel):
    """SPMD configuration including worker count, resource requirements and execution params."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    worker_count: int = 1
    # Mesh is now stored as DeviceMesh directly
    mesh: Optional[Any] = None  # DeviceMesh (use Any to avoid circular import)
    user_forward_fn: Optional[Callable] = None
    user_backward_fn: Optional[Callable] = None
    user_update_fn: Optional[Callable] = None  # User-defined parameter update function
    # Explicit OperatorMeta storage - ensures metadata survives Ray serialization
    # (function attributes are lost when pickled by reference across processes)
    forward_meta: Optional["OperatorMeta"] = None
    backward_meta: Optional["OperatorMeta"] = None
    update_meta: Optional["OperatorMeta"] = None
    max_async_steps: int = 1  # Maximum async steps between workers in the same SPMD
    params_update_steps: int = 1  # Sync barrier every N steps for parameter updates
    # Whether to remap output step numbers.
    # - Default False: output step should be consistent with the input/execution step.
    # - If True:
    #   - eager_trigger operators: output_step = total_rank * _current_step + local_rank
    #   - non-eager operators: output_step = _current_step
    enable_output_step_remap: bool = False

    @model_validator(mode="after")
    def _sync_worker_count_from_mesh(self) -> "SPMDConfig":
        """
        Keep worker_count derived from mesh.

        worker_count = product(mesh axis sizes) = mesh.num_devices
        """
        if self.mesh is None:
            self.worker_count = 1
            return self
        self.worker_count = int(getattr(self.mesh, "size", 1))
        return self

    @property
    def has_mesh(self) -> bool:
        """Check if mesh is defined."""
        return self.mesh is not None

    def get_mesh_size(self) -> int:
        """Get total number of devices in mesh."""
        if not self.has_mesh or self.mesh is None:
            return 1
        return self.mesh.size


# =============================================================================
# Resource and Scheduling Types (Section 2.1.2)
# =============================================================================


class ResourceSpec(BaseModel):
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


class PlacementPlan(BaseModel):
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


class Edge(BaseModel):
    """
    Represents an edge in the task graph.

    Edges connect nodes and can specify source/target op types.
    This allows flexible connections like forward->backward for sink nodes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_node: str
    target_node: str
    source_op: OpType = OpType.FORWARD  # Source node's op type
    target_op: OpType = OpType.FORWARD  # Target node's op type
    condition_fn: Optional[Callable[..., bool]] = None


class Node(BaseModel):
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
    node_type: NodeType = NodeType.SPMD  # Node type: SPMD or READER
    reader_config: Optional[ReaderConfig] = (
        None  # Reader configuration (only for READER nodes)
    )


class ArgumentType(str, Enum):
    """Argument type for fluxway transfer data."""

    STATE = "state"  # State argument
    TENSOR = "tensor"  # Tensor argument
    STATIC = "static"  # Static argument


class EdgeShardingInfo(BaseModel):
    """
    Sharding information for a single logical edge (including conditional edges).

    一个 SPMD operator 的前驱 / 后继边，可能是条件边，也可能是非条件边。
    - 对于同一对 (source_spmd_id, target_spmd_id)，可能存在多条条件边，
      因此需要用 List[EdgeShardingInfo] 来描述每一条边。
    - 条件信息直接记录为 condition_fn，便于在运行时根据实际执行路径选择。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 条件函数，如果为 None 表示无条件边（始终生效）
    condition_fn: Optional[Callable[..., bool]] = None
    # 对应这条边上，当前 operator 视角下相关 tensor 的 sharding 规格
    # 例如：前驱边写入到 input_pre_sharding 时，是「前驱输出 tensor 的 sharding」；
    #       后继边写入到 output_post_sharding 时，是「后继输入 tensor 的 sharding」。
    tensor_sharding: Dict[str, ShardingSpec] = Field(default_factory=dict)
    # 目标节点的 op 类型（用于跨 op 类型连边，如 forward->backward）
    target_op: OpType = OpType.FORWARD


class OperatorMeta(BaseModel):
    """Metadata attached to an operator function."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    # Raw sharding specs (from @operator decorator, mesh-independent)
    # Each value is a dict with keys: type ("dense"|"sparse"), axis (list), fn (optional)
    input_sharding_raw: Dict[str, Any] = Field(default_factory=dict)
    output_sharding_raw: Dict[str, Any] = Field(default_factory=dict)
    # Resolved sharding specs (from @spmd decorator, mesh-aware): new algebraic ShardingSpec
    input_sharding: Optional[Dict[str, ShardingSpec]] = None
    output_sharding: Optional[Dict[str, ShardingSpec]] = None
    # Keys use ArgumentType.value (i.e. "state" | "tensor" | "static") to avoid enum
    # serialization / circular import surprises across control-plane and worker.
    input_arguments: Optional[Dict[str, List[str]]]
    output_arguments: Optional[Dict[str, List[str]]]
    # Linking sharding specs (populated by Builder when constructing the graph)
    # 对于当前 operator：
    # - input_pre_sharding：记录所有「前驱 SPMD -> 当前 SPMD」的边及其 sharding，
    #   包含条件边；key 为前驱 spmd_id，value 为该前驱到当前的若干条边。
    #   结构：{pred_spmd_id: List[EdgeShardingInfo]}
    # - output_post_sharding：记录所有「当前 SPMD -> 后继 SPMD」的边及其 sharding，
    #   同样包含条件边；key 为后继 spmd_id。
    #   结构：{succ_spmd_id: List[EdgeShardingInfo]}
    input_pre_sharding: Dict[str, List[EdgeShardingInfo]] = Field(default_factory=dict)
    output_post_sharding: Dict[str, List[EdgeShardingInfo]] = Field(
        default_factory=dict
    )

    # Eager trigger mode: when True, the operator triggers as soon as any data arrives
    # (expected tensor shards = 1 for all tensors, ignoring sharding topology)
    eager_trigger: bool = False

    def get_arguments(self, is_output: bool, arg_type: "ArgumentType") -> List[str]:
        """
        Get input/output arguments list for a given arg type.

        Note: input_arguments/output_arguments store keys as ArgumentType.value (str).
        """
        args_map = self.output_arguments if is_output else self.input_arguments
        if not args_map:
            return []
        return list(args_map.get(arg_type.value) or [])

    # ---------------------------------------------------------------------
    # Compare / validate helpers (used by control-plane graph builder)
    # ---------------------------------------------------------------------

    def tensor_names(self, is_output: bool) -> List[str]:
        """Get tensor argument names from input/output arguments."""
        return self.get_arguments(is_output=is_output, arg_type=ArgumentType.TENSOR)

    def state_names(self, is_output: bool) -> List[str]:
        """Get state argument names from input/output arguments."""
        return self.get_arguments(is_output=is_output, arg_type=ArgumentType.STATE)

    def validate_tensor_io_match(
        self, dst_meta: "OperatorMeta", src_is_output: bool, ctx: str
    ) -> None:
        """
        Validate tensor variable names match between src and dst (io semantics).

        - If src_is_output=True: compare src.output tensor vs dst.input tensor
        - If src_is_output=False: compare src.input tensor vs dst.output tensor
        """
        src_names = set(self.tensor_names(is_output=src_is_output))
        dst_names = set(dst_meta.tensor_names(is_output=not src_is_output))
        if src_names != dst_names:
            raise ValueError(
                f"Tensor names mismatch in {ctx}: "
                f"source={sorted(src_names)}, target={sorted(dst_names)}"
            )

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
        self.output_arguments[ArgumentType.STATE.value] = list(dst_input_state)


class Graph(BaseModel):
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


class GraphValidationResult(BaseModel):
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


class GlobalState(BaseModel):
    """Global training state aggregated from all SPMDs."""

    current_step: int = 0
    # TODO: Add global metrics (loss, throughput, etc.)
    # TODO: Add checkpoint metadata
    # TODO: Add resource usage statistics


class SPMDState(BaseModel):
    """State of a single SPMD instance."""

    spmd_id: str = ""
    current_step: int = 0
    # TODO: Add SPMD-specific metrics
    # TODO: Add model parameters reference
    # TODO: Add optimizer state reference


class WorkerTelemetry(BaseModel):
    """Telemetry data from a single worker."""

    worker_id: str = ""
    # TODO: Add GPU utilization
    # TODO: Add memory usage
    # TODO: Add network I/O statistics
    # TODO: Add heartbeat timestamp


# =============================================================================
# Handle Types (Section 3.2.1, 6.1)
# =============================================================================


class LogicalHandle(BaseModel):
    """
    Logical handle providing a stable interface.

    Logical handles don't change when nodes are evicted and restored.
    """

    logical_id: str
    spmd_id: str
    worker_id: str


class ActualHandle(BaseModel):
    """
    Actual handle corresponding to a Ray Actor.

    May change when nodes are evicted and rebuilt.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    actor_handle: Any  # ray.actor.ActorHandle
    worker_id: str
    created_at: float = 0.0  # Creation timestamp


class NodeView(BaseModel):
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
# Dual-Stream Convergence Types (Section 3.2.2)
# =============================================================================


class ControlConfig(BaseModel):
    """
    Control configuration for a single batch.

    Contains all control plane parameters needed for computation.
    """

    batch_id: str = ""
    global_step: int = 0
    learning_rate: float = 0.001
    clip_norm: float = 1.0
    is_training: bool = True
    # TODO: Add more control parameters as needed


# =============================================================================
# ULP Protocol Types (Section 4.2, 4.3)
# =============================================================================


class PartitionOp(IntEnum):
    """
    Partition operator enum for routing computation.

    Determines how data is partitioned and routed to target workers.
    """

    DIV_MOD = 0  # Dense/Slicing: PID = (val // params[0]) % params[1]
    HASH_MOD = 1  # Sparse/Embedding: PID = hash(val) % params[0]
    LOOKUP = 2  # MoE/Irregular: PID = params[val]
    REPLICATE = 3  # Broadcast: PID = All Target Ranks


class LayoutSpec(BaseModel):
    """
    Sender-side view defining how data maps to physical nodes.

    All fields must be static native arrays (NumPy/JAX/Torch Tensor),
    no Python List/Dict allowed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    op_type: PartitionOp = PartitionOp.DIV_MOD
    op_params: Optional[np.ndarray] = None  # Static parameters for the operator
    topology_map: Optional[np.ndarray] = None  # Logical PID -> Physical Rank mapping
    target_ranks: Optional[np.ndarray] = None  # Only for REPLICATE mode


class DependencySpec(BaseModel):
    """
    Receiver-side view defining expected signals for a batch.

    Used by Ticket Barrier to determine when all shards have arrived.
    """

    expected_shards: int = 0  # Ticket threshold (usually = sender world size)


class RPCPayload(BaseModel):
    """
    RPC payload for data transmission.

    Kept minimal to reduce transmission overhead.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_id: str = ""
    data: Optional[np.ndarray] = None  # Data shard (Tensor)
    global_indices: Optional[np.ndarray] = (
        None  # Original indices for Sparse->Dense restoration
    )
    # Note: No sender_id or complex metadata needed


# =============================================================================
# Tensor Metadata Types (for routing and backward dependency tracking)
# =============================================================================


class TensorMeta(BaseModel):
    """
    Tensor metadata — the single descriptor that accompanies every tensor/shard.

    Lifecycle (four phases):

    1. **Routing meta** (``_build_routing_meta``):
       Describes the **full output tensor** before routing.  Passed to the
       routing function so it can decide where to send each slice.
       - ``shape``   = full local tensor shape (pre-slicing)
       - ``global_shape`` = global tensor shape (dense: inferred from spec;
                            sparse output: propagated from input meta)
       - ``global_indices`` = per-row global index (dense: from spec offset;
                              sparse: from operator-produced indices)

    2. **Shard meta** (``_build_shard_meta``):
       Derived from routing meta + one route's ``data_mask``.
       Describes the **sliced shard** actually sent to a receiver.
       - ``shape`` = shard shape (computed from data_mask)
       - ``global_shape`` = inherited from routing meta (unchanged)
       - ``global_indices`` = axis-0 slice of routing meta's global_indices

    3. **Assembled meta** (after reception / ``_assemble_*``):
       Describes the **reconstructed local tensor** that the operator will consume.
       - dense-to-dense: ``global_shape`` inferred from input spec
       - sparse-to-dense: ``global_shape`` carried from sender's meta
       - to-dynamic:     ``global_shape`` propagated, ``global_indices`` merged

    4. **Backward lookup** (``get_forward_step_meta``):
       Backward uses ``forward_step`` to look up the corresponding forward
       assembled meta from ``_meta_history`` for sparse reverse routing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Basic tensor description
    tensor_name: str = ""
    shape: Tuple[int, ...] = ()  # Shape of this tensor/shard
    dtype: str = "float32"

    # Global routing info — required when sparse routing is involved
    global_indices: Optional[np.ndarray] = None  # Per-row global indices
    global_shape: Optional[Tuple[int, ...]] = None  # Global tensor shape

    # Source tracking — required for backward reverse routing
    source_spmd_id: str = ""
    source_rank: int = -1
    forward_step: int = -1

    # Flags
    is_sparse: bool = False  # True if output via dynamic (indexed/hash) ShardingSpec
    is_empty: bool = False  # Empty packet for sparse routing sync


class ShardOrigin(BaseModel):
    """
    Records the origin of a single received shard for backward reverse routing.

    During forward assembly, each non-empty received shard is recorded so that
    backward can reverse-route gradients back to the exact source workers
    without requiring a user-provided routing function.

    Key design (to support dynamic→static reordering):
    - ``global_indices`` 记录该 shard 自己携带的全局索引（稀疏语义）。
    - ``local_indices`` 记录在 **重排序之后**，该 shard 对应的行在本地
      assembled tensor 中的下标位置（axis-0）。
      这样反向时可以直接用 ``local_indices`` 在本地 dense tensor 上做切片/
      gather，而不需要假设 shard 在重排序后仍然是连续区间。

    Fields:
        source_spmd_id: SPMD that sent this shard.
        source_rank:    Rank within the source SPMD.
        global_indices: Global indices carried by this shard (sparse tensors).
        shard_shape:    Shape of the received shard data.
        local_indices:  1D array of row indices in the assembled local tensor
                        (after any reordering).  ``None`` means “use full range”.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_spmd_id: str = ""
    source_rank: int = -1
    global_indices: Optional[np.ndarray] = None
    shard_shape: Tuple[int, ...] = ()
    local_indices: Optional[np.ndarray] = None


class StepTensorMeta(BaseModel):
    """
    Per-step per-tensor metadata preserving both the assembled global view
    and the per-shard origin info needed for backward reverse routing.

    Stored in ``Worker._meta_history[op_type][step][tensor_name]``.

    Fields:
        assembled_meta: The assembled TensorMeta (merged global view).
        shard_origins:  Per-shard origin records.  For backward reverse routing
                        without a user route-fn, the worker iterates these to
                        determine which slice of the backward output goes to
                        which source worker.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    assembled_meta: Optional[TensorMeta] = None
    shard_origins: List[ShardOrigin] = Field(default_factory=list)


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
    TOPOLOGY_REGISTERING = "topology_registering"  # Topology registration in progress
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
    RECOVERY_FAILED = "recovery_failed"  # Recovery failed


class DriverState(str, Enum):
    """SPMD Driver state machine states."""

    INITIALIZING = "initializing"  # Initializing workers
    READY = "ready"  # Ready, waiting for Controller command
    RUNNING = "running"  # Running
    BLOCKED = "blocked"  # Blocked
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


class WorkerHeartbeat(BaseModel):
    """Worker heartbeat data."""

    rank: int
    timestamp: float
    op_steps: Dict[str, int] = Field(default_factory=dict)  # op_type.name -> step
    status: WorkerStatus = WorkerStatus.RUNNING
    error_message: Optional[str] = None
    telemetry: Optional["WorkerTelemetry"] = None
    mailbox_status: Optional[MailboxStatus] = None  # Mailbox status for backpressure


class DriverHeartbeat(BaseModel):
    """Driver heartbeat data reported to Controller."""

    spmd_id: str
    timestamp: float
    status: DriverStatus = DriverStatus.HEALTHY
    state: DriverState = DriverState.INITIALIZING
    error_message: Optional[str] = None
    op_steps: Dict[str, int] = Field(
        default_factory=dict
    )  # op_type.name -> spmd-level step
    worker_count: int = 0
    healthy_worker_count: int = 0
    blocked_ops: List[str] = Field(default_factory=list)
    workers_ready: bool = False  # All workers have reported heartbeat
    workers_running: bool = False  # All workers are running main loop
    is_downstream_busy: bool = (
        False  # Whether downstream SPMDs are busy (for backpressure)
    )
    reader_state: Optional[ReaderState] = None  # Reader state (only for READER nodes)
    busy_worker_ratio: float = 0.0  # Ratio of workers with full mailbox


class HeartbeatConfig(BaseModel):
    """Heartbeat configuration."""

    heartbeat_interval: float = 5.0  # Heartbeat interval (seconds)
    heartbeat_timeout: float = 15.0  # Heartbeat timeout (seconds)
    max_missed_heartbeats: int = 3  # Maximum missed heartbeats before error


class AsyncStepConfig(BaseModel):
    """Configuration for async step difference control."""

    max_forward_step_diff: int = 2  # Maximum forward step difference
    max_backward_step_diff: int = 2  # Maximum backward step difference
    min_lock_duration: float = (
        1.0  # Minimum lock duration (prevent frequent lock/unlock)
    )


class DriverResourceSpec(BaseModel):
    """Driver management resource configuration (system default)."""

    num_cpus: int = 1
    memory_gb: float = 1.0


class SPMDPlacementSpec(BaseModel):
    """Complete SPMD resource configuration."""

    spmd_id: str
    driver_resource: DriverResourceSpec = Field(default_factory=DriverResourceSpec)
    worker_resource: Optional["ResourceSpec"] = None  # Single worker resource
    worker_count: int = 1

    @property
    def total_worker_cpus(self) -> int:
        """Total worker CPUs."""
        if self.worker_resource is None:
            return self.worker_count
        return self.worker_resource.num_cpus * self.worker_count

    @property
    def total_worker_gpus(self) -> int:
        """Total worker GPUs."""
        if self.worker_resource is None:
            return 0
        return self.worker_resource.num_gpus * self.worker_count


class SPMDOpStep(BaseModel):
    """SPMD operation step tracking."""

    spmd_id: str
    forward_step: int = 0
    backward_step: int = 0
    update_step: int = 0


class OpLockState(BaseModel):
    """Operation lock state."""

    spmd_id: str
    op_type: str  # OpType name
    locked_at: float
    reason: str
