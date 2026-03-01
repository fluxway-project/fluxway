"""
Operator Manager for SPMD Worker.

Responsibilities:
1. Store and manage operators (e.g., train_step, rollout, reward)
2. Parse topology from OperatorMeta at initialization (no runtime registration)
3. Compute expected tensor shards based on input_sharding + input_pre_sharding
4. Compute output routing based on output_sharding + output_post_sharding (4 cases)
5. Execute operators
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING
import threading
import logging
import inspect

from fluxway.common.types.configs import ArgumentType
from fluxway.common.types.sharding import ShardingSpec
from fluxway.common.types import EdgeShardingInfo
from fluxway.common.mesh import DeviceMesh

if TYPE_CHECKING:
    from fluxway.common.types import SPMDConfig, OperatorMeta

logger = logging.getLogger(__name__)


# =============================================================================
# Topology Info (parsed from OperatorMeta, not runtime registered)
# =============================================================================


@dataclass
class SPMDTopologyInfo:
    """
    Topology info for a predecessor or successor SPMD.
    Parsed from OperatorMeta at initialization.
    """

    spmd_id: str
    world_size: int = 1

    mesh: Optional[DeviceMesh] = None
    # Tensor sharding: {tensor_name: ShardingSpec}
    tensor_sharding: Dict[str, ShardingSpec] = field(default_factory=dict)
    # Condition function for conditional edges (only for successors)
    condition_fn: Optional[Callable[..., bool]] = None
    # Target op name for cross-op edges (e.g., train_step->update)
    target_op: Optional[str] = None
    # Map of output tensors to target operator input tensors
    tensor_map: Dict[str, str] = field(default_factory=dict)
    # Whether this is a self-loop edge (source_node == target_node)
    is_self_loop: bool = False


class OperatorManager:
    """
    Manages operator execution, sharding computation, and routing.

    Key design:
    - All topology info is parsed from OperatorMeta at initialization
    - No runtime predecessor/successor registration needed
    - Handler management is NOT here (handled by Worker)
    """

    def __init__(self, config: "SPMDConfig", rank: int, spmd_id: str = ""):
        """
        Initialize OperatorManager.

        Args:
            config: SPMD configuration containing user functions and mesh
            rank: This worker's rank
            spmd_id: SPMD identifier (for self-loop detection)
        """
        self._config = config  # Store config for explicit meta access
        self._mesh = config.mesh  # DeviceMesh from config
        self._rank = rank
        self._world_size = config.num_workers
        self._spmd_id = spmd_id

        # Register operators
        # Register operators dynamically from config
        self._operators: Dict[str, Callable] = config.operators

        # Initialize per-operator dictionaries
        self._input_sharding: Dict[str, Dict[str, ShardingSpec]] = {}
        self._output_sharding: Dict[str, Dict[str, ShardingSpec]] = {}
        self._expected_state_names: Dict[str, List[str]] = {}
        self._expected_tensor_names: Dict[str, List[str]] = {}
        self._predecessor_topology: Dict[str, Dict[str, SPMDTopologyInfo]] = {}
        self._successor_topology: Dict[str, Dict[str, SPMDTopologyInfo]] = {}
        self._expected_tensor_shards_cache: Dict[str, Dict[str, int]] = {}
        self._periodic_trigger_cache: Dict[str, bool] = {}
        self._operator_input_types: Dict[str, Optional[Type]] = {}
        self._output_to_input_tensor_mapping: Dict[str, Dict[str, str]] = {}

        for op_name in self._operators:
            self._input_sharding[op_name] = {}
            self._output_sharding[op_name] = {}
            self._expected_state_names[op_name] = []
            self._expected_tensor_names[op_name] = []
            self._predecessor_topology[op_name] = {}
            self._successor_topology[op_name] = {}
            self._operator_input_types[op_name] = None
            self._output_to_input_tensor_mapping[op_name] = {}

        # Static arguments from config
        self._static_args: Dict[str, Any] = {}
        if config.static_args:
            self._static_args.update(config.static_args)

        # Thread lock
        self._lock = threading.Lock()

        # Parse topology from OperatorMeta
        self._parse_operator_metadata()

        # Pre-compute expected tensor shards after routing strategies are ready
        self._precompute_expected_tensor_shards()

        # Pre-compute operator input types from function signatures
        self._precompute_operator_input_types()

    # =========================================================================
    # Metadata Parsing (at initialization)
    # =========================================================================

    def _parse_operator_metadata(self) -> None:
        """
        Parse operator metadata from @operator decorator and config.topology.

        Extracts:
        - input/output sharding specs
        - expected state/tensor names
        - predecessor/successor topology from EdgeShardingInfo and config.topology
        """
        # 0. Parse from config.topology first (explicitly set by Builder)
        if self._config and self._config.topology:
            for op_name, top in self._config.topology.items():
                if op_name not in self._predecessor_topology:
                    self._predecessor_topology[op_name] = {}
                for pred_id, info in top.predecessors.items():
                    self._predecessor_topology[op_name][pred_id] = SPMDTopologyInfo(
                        spmd_id=info.spmd_id,
                        condition_fn=info.condition_fn,
                        tensor_map=info.tensor_map,
                        is_self_loop=(info.spmd_id == self._spmd_id),
                    )
                if op_name not in self._successor_topology:
                    self._successor_topology[op_name] = {}
                for succ_id, info in top.successors.items():
                    self._successor_topology[op_name][succ_id] = SPMDTopologyInfo(
                        spmd_id=info.spmd_id,
                        condition_fn=info.condition_fn,
                        tensor_map=info.tensor_map,
                        is_self_loop=(info.spmd_id == self._spmd_id),
                    )

        for op_name, operator in self._operators.items():
            if operator is None:
                continue

            meta = self._get_operator_metadata(op_name)
            if meta is None:
                continue

            if meta.input_sharding:
                self._input_sharding[op_name] = self._parse_sharding_specs(
                    meta.input_sharding
                )
            if meta.output_sharding:
                self._output_sharding[op_name] = self._parse_sharding_specs(
                    meta.output_sharding
                )

            logger.debug(
                "OperatorManager[%s]: op=%s, input_sharding_keys=%s, output_sharding_keys=%s, "
                "input_pre_sharding_keys=%s, output_post_sharding_keys=%s",
                self._spmd_id,
                op_name,
                list(self._input_sharding[op_name].keys()),
                list(self._output_sharding[op_name].keys()),
                list(meta.input_pre_sharding.keys()),
                list(meta.output_post_sharding.keys()),
            )

            self._expected_state_names[op_name] = list(
                meta.get_arguments(is_output=False, arg_type=ArgumentType.STATE) or []
            )
            self._expected_tensor_names[op_name] = list(
                meta.get_arguments(is_output=False, arg_type=ArgumentType.TENSOR) or []
            )

            # Parse predecessor topology from input_pre_sharding
            # Structure: {spmd_id: List[EdgeShardingInfo]}
            for pre_spmd_id, edge_info_list in meta.input_pre_sharding.items():
                merged_sharding: Dict[str, ShardingSpec] = {}
                condition_fn = None
                # Default to same op_name if not in EdgeShardingInfo.
                # This fallback design balances two scenarios:
                # 1. Graph Mode: Builder usually defines explicit target_op for complex routed edges.
                # 2. Eager Mode: Lacks full topology compilation, so we use duck-typing routing:
                #    if a source sends data here, it's implicitly targeting the current op_name.
                target_op = op_name
                tensor_map = {}
                for edge_info in edge_info_list:
                    # Convert raw sharding dicts to ShardingSpec objects if needed
                    parsed_edge_sharding = self._parse_sharding_specs(
                        edge_info.tensor_sharding
                    )
                    merged_sharding.update(parsed_edge_sharding)
                    if edge_info.condition_fn is not None:
                        condition_fn = edge_info.condition_fn
                    # Get target_op and tensor_map from EdgeShardingInfo
                    if edge_info.target_op:
                        target_op = edge_info.target_op
                    if edge_info.tensor_map:
                        tensor_map.update(edge_info.tensor_map)

                # Infer world_size from sharding spec's mesh if available
                world_size = self._infer_world_size_from_sharding(merged_sharding)

                self._predecessor_topology[op_name][pre_spmd_id] = SPMDTopologyInfo(
                    spmd_id=pre_spmd_id,
                    world_size=world_size,
                    mesh=self._extract_mesh_from_sharding(merged_sharding),
                    tensor_sharding=merged_sharding,
                    condition_fn=condition_fn,
                    target_op=target_op,
                    tensor_map=tensor_map,
                    is_self_loop=(pre_spmd_id == self._spmd_id),
                )

            # Parse successor topology from output_post_sharding
            # Structure: {spmd_id: List[EdgeShardingInfo]}
            for post_spmd_id, edge_info_list in meta.output_post_sharding.items():
                merged_sharding: Dict[str, ShardingSpec] = {}
                condition_fn = None
                # Default to same op_name if not explicitly specified.
                # As the sender, if the downstream target isn't explicitly defined (e.g., via Builder),
                # we assume symmetric execution by triggering the identically-named operator on the successor.
                # This maintains compatibility with both compiled Graph edge mapping and dynamic Eager execution.
                target_op = op_name
                tensor_map = {}
                for edge_info in edge_info_list:
                    # Convert raw sharding dicts to ShardingSpec objects if needed
                    parsed_edge_sharding = self._parse_sharding_specs(
                        edge_info.tensor_sharding
                    )
                    merged_sharding.update(parsed_edge_sharding)
                    if edge_info.condition_fn is not None:
                        condition_fn = edge_info.condition_fn
                    # Get target_op and tensor_map from EdgeShardingInfo
                    if edge_info.target_op:
                        target_op = edge_info.target_op
                    if edge_info.tensor_map:
                        tensor_map.update(edge_info.tensor_map)

                world_size = self._infer_world_size_from_sharding(merged_sharding)

                self._successor_topology[op_name][post_spmd_id] = SPMDTopologyInfo(
                    spmd_id=post_spmd_id,
                    world_size=world_size,
                    mesh=self._extract_mesh_from_sharding(merged_sharding),
                    tensor_sharding=merged_sharding,
                    condition_fn=condition_fn,
                    target_op=target_op,
                    tensor_map=tensor_map,
                    is_self_loop=(post_spmd_id == self._spmd_id),
                )

            # Build output-to-input tensor name mapping (positional correspondence)
            input_tensor_names = list(
                meta.get_arguments(is_output=False, arg_type=ArgumentType.TENSOR) or []
            )
            output_tensor_names = list(
                meta.get_arguments(is_output=True, arg_type=ArgumentType.TENSOR) or []
            )
            mapping: Dict[str, str] = {}
            for idx, out_name in enumerate(output_tensor_names):
                if idx < len(input_tensor_names):
                    mapping[out_name] = input_tensor_names[idx]
            self._output_to_input_tensor_mapping[op_name] = mapping

            logger.debug(
                f"Parsed operator meta for {op_name}: "
                f"state={self._expected_state_names[op_name]}, "
                f"tensor={self._expected_tensor_names[op_name]}, "
                f"predecessors={list(self._predecessor_topology[op_name].keys())}, "
                f"successors={list(self._successor_topology[op_name].keys())}, "
                f"output_to_input_tensor_mapping={mapping}"
            )

    def _parse_sharding_specs(
        self, raw_specs: Dict[str, Any]
    ) -> Dict[str, "ShardingSpec"]:
        """Convert raw sharding dictionaries into ShardingSpec objects."""
        from fluxway.common.types.sharding import ShardingSpec, ShardingConfig

        parsed = {}
        for tensor_name, raw_spec in raw_specs.items():
            if isinstance(raw_spec, dict):
                parsed[tensor_name] = ShardingConfig.model_validate(raw_spec).to_spec(
                    self._mesh
                )
            elif isinstance(raw_spec, ShardingSpec):
                parsed[tensor_name] = raw_spec
        return parsed

    def _get_operator_metadata(self, op_name: str) -> Optional["OperatorMeta"]:
        """
        Get metadata attached to an operator.

        Prefers explicit meta from SPMDConfig (properly serialized) over
        function attributes (lost when pickled by reference across processes).
        """
        if op_name in self._config.operator_metas:
            return self._config.operator_metas[op_name]

        # Fallback to function attribute (may not have edge info in worker processes)
        operator = self._operators.get(op_name)
        if operator is None:
            return None
        return getattr(operator, "_flux_operator_meta", None)

    def _infer_world_size_from_sharding(self, sharding_map: Dict[str, Any]) -> int:
        """Infer total rank from sharding specs."""
        for spec in sharding_map.values():
            if isinstance(spec, ShardingSpec) and spec.mesh is not None:
                return spec.mesh.size
        return 1

    def _extract_mesh_from_sharding(
        self, sharding_map: Dict[str, Any]
    ) -> Optional[DeviceMesh]:
        """Extract mesh from sharding specs."""
        for spec in sharding_map.values():
            if isinstance(spec, ShardingSpec) and spec.mesh is not None:
                return spec.mesh
        return None

    def _precompute_expected_tensor_shards(self) -> None:
        """
        Pre-compute expected tensor shard counts for all operator types.

        This is computed once after _input_sharding, _predecessor_topology, and
        _input_routing_strategies are all initialized.

        The result is cached in _expected_tensor_shards_cache.
        Also caches the pre-computed shard counts in _expected_tensor_shards_cache.

        Special case: if the operator has a periodic barrier, all expected counts
        are set to the barrier's 'n' value (default=1).
        """
        for op_name in self._operators:
            expected: Dict[str, int] = {}

            # Check if periodic_trigger is enabled for this operator
            meta = self._get_operator_metadata(op_name)
            is_periodic = self.is_periodic_trigger(op_name)

            for tensor_name in self._expected_tensor_names.get(op_name, []):
                # If periodic_trigger is enabled, expect N shards to trigger execution (default N=1)
                if is_periodic:
                    from fluxway.common.types.configs import PeriodicBarrierConfig

                    n = 1
                    if meta and isinstance(meta.barrier_config, PeriodicBarrierConfig):
                        n = meta.barrier_config.n
                    expected[tensor_name] = n
                    continue

                total_count = 0
                input_spec = self._input_sharding.get(op_name, {}).get(tensor_name)
                for pre_spmd_id, pre_info in self._predecessor_topology[
                    op_name
                ].items():
                    # Check if this predecessor actually provides this tensor
                    # tensor_sharding uses target-side names (after mapping)
                    if tensor_name not in pre_info.tensor_sharding:
                        continue

                    pre_sharding = pre_info.tensor_sharding.get(tensor_name)
                    if pre_sharding and input_spec:
                        # Use pure Python analytical resolution
                        deps = pre_sharding.resolve_producers(input_spec, self._rank)
                        total_count += len(deps)
                    else:
                        # Full fallback for dynamic/unknown cases
                        total_count += pre_info.world_size

                expected[tensor_name] = total_count

            self._expected_tensor_shards_cache[op_name] = expected

            if expected:
                logger.debug(
                    f"Precomputed expected tensor shards for {op_name}: {expected}"
                    + (" (periodic_trigger=True)" if is_periodic else "")
                )

    def _precompute_operator_input_types(self) -> None:
        """
        Pre-compute operator input types from function signatures.

        Extracts the first parameter type (excluding self) from each operator function
        and caches it for use in _assemble_input.
        """
        for op_name, operator in self._operators.items():
            if operator is None:
                continue

            try:
                sig = inspect.signature(operator)
                params = list(sig.parameters.values())

                is_source = self.is_source_operator(op_name)
                has_self = params and params[0].name == "self"
                param = (
                    params[1]
                    if has_self and len(params) > 1
                    else (params[0] if not has_self and params else None)
                )

                if param is None:
                    if is_source:
                        logger.debug(
                            f"Operator {op_name} is a source, skipping parameter type pre-computation."
                        )
                    else:
                        logger.warning(
                            f"Operator {op_name} has no valid parameters to receive data, "
                            f"but it is not marked as a source operator."
                        )
                    continue

                param_type = param.annotation
                if param_type == inspect.Parameter.empty:
                    logger.warning(
                        f"Operator {op_name} parameter '{param.name}' has no type annotation"
                    )
                    continue

                self._operator_input_types[op_name] = param_type
                logger.debug(f"Precomputed input type for {op_name}: {param_type}")

            except Exception as e:
                logger.warning(f"Failed to extract input type for {op_name}: {e}")
                continue

    # =========================================================================
    # Topology Query
    # =========================================================================

    def get_predecessor_ids(self, op_name: str) -> List[str]:
        """Get all predecessor SPMD IDs for an operator."""
        return list(self._predecessor_topology.get(op_name, {}).keys())

    def get_successor_ids(self, op_name: str) -> List[str]:
        """Get all successor SPMD IDs for an operator."""
        return list(self._successor_topology.get(op_name, {}).keys())

    def get_predecessor_info(
        self, op_name: str, spmd_id: str
    ) -> Optional[SPMDTopologyInfo]:
        """Get predecessor topology info."""
        return self._predecessor_topology.get(op_name, {}).get(spmd_id)

    def get_successor_info(
        self, op_name: str, spmd_id: str
    ) -> Optional[SPMDTopologyInfo]:
        """Get successor topology info."""
        return self._successor_topology.get(op_name, {}).get(spmd_id)

    def get_successor_condition_fn(
        self, op_name: str, spmd_id: str
    ) -> Optional[Callable[..., bool]]:
        """Get condition function for a successor edge."""
        info = self._successor_topology.get(op_name, {}).get(spmd_id)
        return info.condition_fn if info else None

    # =========================================================================
    # Operator Metadata Query
    # =========================================================================

    def get_operator_metadata(self, op_name: str) -> Optional["OperatorMeta"]:
        """Get the OperatorMeta document for a given op name."""
        return self._get_operator_metadata(op_name)

    # =========================================================================

    def is_periodic_trigger(self, op_name: str) -> bool:
        """
        Check if an operator is in periodic_trigger mode.

        In periodic_trigger mode:
        - The operator triggers every N arrivals (typically N=1)
        - No step matching required between sender and receiver

        Args:
            op_name: Operator name to check

        Returns:
            True if periodic_trigger is enabled for this operator
        """
        from fluxway.common.types.configs import BarrierType

        meta = self._get_operator_metadata(op_name)
        if not meta or not getattr(meta, "barrier_config", None):
            return False
        return meta.barrier_config.type == BarrierType.PERIODIC

    def get_expected_state_names(self, op_name: str) -> List[str]:
        """Get expected state variable names."""
        return list(self._expected_state_names.get(op_name, []))

    def get_expected_tensor_names(self, op_name: str) -> List[str]:
        """Get expected tensor variable names."""
        return list(self._expected_tensor_names.get(op_name, []))

    def is_source_operator(self, op_name: str) -> bool:
        """
        Check if an operator is a source operator (data producer).
        A source operator is identified by:
        1. Explicit metadata: @operator(..., is_source=True)
        2. Graph topology: Has no predecessor SPMDs AND no incoming tensor arguments.

        Note: Source operators can still have static/state arguments (e.g., from Driver),
        but they do not depend on other operators for execution triggers.
        """
        meta = self._get_operator_metadata(op_name)
        if meta is None:
            return False

        # 1. Respect explicit user marking if available
        if meta.is_source is True:
            # Strong validation: A source should not have tensor inputs from other nodes
            pred_ids = self.get_predecessor_ids(op_name)
            if pred_ids:
                logger.warning(
                    f"[Fluxway Operator Warning] Operator '{op_name}' is explicitly marked as 'is_source=True', "
                    f"but it has predecessor SPMDs: {pred_ids}. This might indicate a graph configuration error."
                )
            return True

        if meta.is_source is False:
            return False

        # 2. Automated Inference (when meta.is_source is None)
        tensor_names = self.get_expected_tensor_names(op_name)
        pred_ids = self.get_predecessor_ids(op_name)

        has_no_tensors = len(tensor_names) == 0
        has_no_predecessors = len(pred_ids) == 0

        # Heuristic: No upstream dependency implies it's a source (e.g., Reader or standalone op)
        is_source = has_no_tensors and has_no_predecessors

        return is_source

    def get_source_operators(self) -> List[str]:
        """Get names of all source operators."""
        return [op for op in self._operators if self.is_source_operator(op)]

    def get_expected_tensor_shards(self, op_name: str) -> Dict[str, int]:
        """
        Get pre-computed expected tensor shard counts based on topology.

        For each tensor:
        - Dense-to-Dense: compute geometric dependencies
        - Sparse involved: expect from all predecessor ranks

        The result is pre-computed during initialization and cached.

        Returns:
            {tensor_name: expected_shard_count}
        """
        return dict(self._expected_tensor_shards_cache.get(op_name, {}))

    def get_operator_input_type(self, op_name: str) -> Optional[Type]:
        """
        Get the input type (data_type class) expected by an operator.

        Returns:
            The Type of the data_type class, or None if not available
        """
        return self._operator_input_types.get(op_name)

    def get_output_to_input_tensor_mapping(self, op_name: str) -> Dict[str, str]:
        """
        Get the output-to-input tensor name mapping for an operator.

        Built at init from positional correspondence between input_arguments.tensor
        and output_arguments.tensor.  e.g. if input tensors are ["input"] and output
        tensors are ["tensor"], the mapping is {"tensor": "input"}.

        Returns:
            Dict mapping output tensor name → corresponding input tensor name.
        """
        return dict(self._output_to_input_tensor_mapping.get(op_name, {}))

    def get_input_sharding(self, op_name: str) -> Dict[str, ShardingSpec]:
        """Get input sharding specs for an operator."""
        return dict(self._input_sharding.get(op_name, {}))

    def get_predecessor_topology(self, op_name: str) -> Dict[str, SPMDTopologyInfo]:
        """Get predecessor topology info for an operator."""
        return dict(self._predecessor_topology.get(op_name, {}))

    def get_all_input_pre_sharding(
        self, op_name: str
    ) -> Dict[str, List[EdgeShardingInfo]]:
        """
        Get all input pre-sharding strategies for an operator.
        Returns a map from predecessor spmd_id to a list of EdgeShardingInfo.
        """
        meta = self._get_operator_metadata(op_name)
        if not meta:
            return {}
        return dict(meta.input_pre_sharding)

    def get_edge_input_sharding_strategies(
        self, op_name: str, spmd_id: str
    ) -> List[EdgeShardingInfo]:
        """
        Get input sharding strategies for a specific incoming edge.
        """
        meta = self._get_operator_metadata(op_name)
        if not meta:
            return []
        return meta.input_pre_sharding.get(spmd_id, [])

    def get_tensor_input_sharding_spec(
        self, op_name: str, spmd_id: str, tensor_name: str
    ) -> Optional[ShardingSpec]:
        """
        Resolve the ShardingSpec for a specific tensor coming from a specific predecessor.
        Uses the pre-parsed predecessor topology which already contains normalized ShardingSpecs.
        """
        pre_info = self.get_predecessor_info(op_name, spmd_id)
        if not pre_info:
            return None

        return pre_info.tensor_sharding.get(tensor_name)

    # =========================================================================
    # Sharding Query
    # =========================================================================

    def get_output_sharding(self, op_name: str) -> Dict[str, ShardingSpec]:
        """
        Get output sharding for an operator.
        If explicit sharding is not provided, defaults to Replicated on the current mesh
        for all fields identified in output_arguments.
        """
        meta = self._get_operator_metadata(op_name)
        explicit_sharding = {}
        if not meta:
            return explicit_sharding

        from fluxway.common.types.sharding import ShardingConfig, ShardingSpec

        if meta.output_sharding:
            for name, spec_dict in meta.output_sharding.items():
                if isinstance(spec_dict, dict):
                    explicit_sharding[name] = ShardingConfig.model_validate(
                        spec_dict
                    ).to_spec(self._mesh)
                else:
                    explicit_sharding[name] = spec_dict

        # If we have inferred output arguments but no explicit sharding,
        # default them to Replicated on current mesh.
        if meta.output_arguments:
            from fluxway.common.types.configs import ArgumentType

            tensor_names = meta.output_arguments.get(ArgumentType.TENSOR.value, [])
            for name in tensor_names:
                if name not in explicit_sharding:
                    # Default to Replicated on current mesh
                    mesh = self._config.mesh
                    if mesh is None:
                        from fluxway.common.mesh import DeviceMesh

                        mesh = DeviceMesh(device_shape=(1,), axis_names=("data",))
                    from fluxway.common.types.sharding import PartitionSpec

                    explicit_sharding[name] = ShardingSpec.chunk(mesh, PartitionSpec())

        return explicit_sharding

    def get_output_state_names(self, op_name: str) -> List[str]:
        """
        Get output state argument names for an operator.

        These are the state variables that need to be sent to successors
        via the control plane (control_argument transmission path).

        Returns:
            List of state argument names from output_arguments
        """
        meta = self._get_operator_metadata(op_name)
        if meta is None:
            return []
        return meta.state_names(is_output=True)

    # =========================================================================
    # Static Arguments
    # =========================================================================

    def get_static_args(self) -> Dict[str, Any]:
        """Get all static arguments."""
        return dict(self._static_args)

    def set_static_arg(self, name: str, value: Any) -> None:
        """Set a static argument."""
        with self._lock:
            self._static_args[name] = value

    def get_static_arg(self, name: str) -> Optional[Any]:
        """Get a static argument."""
        return self._static_args.get(name)

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(self, op_name: str, *args, **kwargs) -> Any:
        """
        Execute an operator.

        Args:
            op_name: Operator name to execute
            *args, **kwargs: Assembled input data (can be data_type object or kwargs)

        Returns:
            Operator output
        """
        from fluxway.common.decorators import operator_context

        operator = self._operators.get(op_name)
        if operator is None:
            logger.warning(f"No operator registered for {op_name}")
            return None

        try:
            with operator_context(
                static_args=self._static_args,
                rank=self._rank,
                world_size=self._world_size,
            ):
                if self.is_source_operator(op_name):
                    return operator()
                return operator(*args, **kwargs)
        except Exception as e:
            logger.error(f"Execution failed for {op_name}: {e}")
            raise
