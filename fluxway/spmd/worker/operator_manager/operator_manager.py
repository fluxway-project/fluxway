"""
Operator Manager for SPMD Worker.

Responsibilities:
1. Store and manage operators (forward/backward/update)
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

from fluxway.common.types.configs import ArgumentType, OpType
from fluxway.common.types.sharding import ShardingSpec
from fluxway.common.mesh import DeviceMesh

if TYPE_CHECKING:
    from ....common.types import SPMDConfig, OperatorMeta

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
    total_rank: int = 1
    mesh: Optional[DeviceMesh] = None
    # Tensor sharding: {tensor_name: ShardingSpec}
    tensor_sharding: Dict[str, ShardingSpec] = field(default_factory=dict)
    # Condition function for conditional edges (only for successors)
    condition_fn: Optional[Callable[..., bool]] = None
    # Target op type for cross-op-type edges (e.g., forward->backward)
    target_op: OpType = OpType.FORWARD
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
        self._total_rank = config.worker_count
        self._spmd_id = spmd_id

        # Register operators
        self._operators: Dict[OpType, Optional[Callable]] = {
            OpType.FORWARD: config.user_forward_fn,
            OpType.BACKWARD: config.user_backward_fn,
            OpType.UPDATE: config.user_update_fn,
        }

        # Per-operator input sharding: {op_type: {tensor_name: ShardingSpec}}
        self._input_sharding: Dict[OpType, Dict[str, ShardingSpec]] = {
            OpType.FORWARD: {},
            OpType.BACKWARD: {},
            OpType.UPDATE: {},
        }

        # Per-operator output sharding: {op_type: {tensor_name: ShardingSpec}}
        self._output_sharding: Dict[OpType, Dict[str, ShardingSpec]] = {
            OpType.FORWARD: {},
            OpType.BACKWARD: {},
            OpType.UPDATE: {},
        }

        # Expected state/tensor names: {op_type: List[str]}
        self._expected_state_names: Dict[OpType, List[str]] = {
            OpType.FORWARD: [],
            OpType.BACKWARD: [],
            OpType.UPDATE: [],
        }
        self._expected_tensor_names: Dict[OpType, List[str]] = {
            OpType.FORWARD: [],
            OpType.BACKWARD: [],
            OpType.UPDATE: [],
        }

        # Predecessor topology: {op_type: {spmd_id: SPMDTopologyInfo}}
        self._predecessor_topology: Dict[OpType, Dict[str, SPMDTopologyInfo]] = {
            OpType.FORWARD: {},
            OpType.BACKWARD: {},
            OpType.UPDATE: {},
        }

        # Successor topology: {op_type: {spmd_id: SPMDTopologyInfo}}
        self._successor_topology: Dict[OpType, Dict[str, SPMDTopologyInfo]] = {
            OpType.FORWARD: {},
            OpType.BACKWARD: {},
            OpType.UPDATE: {},
        }

        # Static arguments
        self._static_args: Dict[str, Any] = {}

        # Pre-computed expected tensor shards: {op_type: {tensor_name: expected_count}}
        # Computed once after _input_sharding and _predecessor_topology are initialized
        self._expected_tensor_shards_cache: Dict[OpType, Dict[str, int]] = {}

        # Cached eager_trigger status: {op_type: bool}
        # True means the operator triggers immediately on any data, ignoring step ordering
        self._eager_trigger_cache: Dict[OpType, bool] = {}

        # Operator input types (pre-computed from function signatures)
        # {op_type: Type} - the data_type class that the operator expects as input
        self._operator_input_types: Dict[OpType, Optional[Type]] = {
            OpType.FORWARD: None,
            OpType.BACKWARD: None,
            OpType.UPDATE: None,
        }

        # Output-to-input tensor name mapping (positional correspondence)
        # Built from input_arguments.tensor and output_arguments.tensor at init.
        # Used by Worker._send_output to look up the correct input meta when
        # input and output tensor names differ (e.g. input="input", output="tensor").
        self._output_to_input_tensor_mapping: Dict[OpType, Dict[str, str]] = {
            OpType.FORWARD: {},
            OpType.BACKWARD: {},
            OpType.UPDATE: {},
        }

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
        Parse operator metadata from @operator decorator.

        Extracts:
        - input/output sharding specs
        - expected state/tensor names
        - predecessor/successor topology from EdgeShardingInfo
        """
        for op_type, operator in self._operators.items():
            if operator is None:
                continue

            meta = self._get_operator_metadata(op_type)
            if meta is None:
                continue

            # 1. Per-operator input/output sharding
            if meta.input_sharding:
                self._input_sharding[op_type] = self._parse_sharding_specs(
                    meta.input_sharding
                )
            if meta.output_sharding:
                self._output_sharding[op_type] = self._parse_sharding_specs(
                    meta.output_sharding
                )

            logger.debug(
                "OperatorManager[%s]: op=%s, input_sharding_keys=%s, output_sharding_keys=%s, "
                "input_pre_sharding_keys=%s, output_post_sharding_keys=%s",
                self._spmd_id,
                op_type.name,
                list(self._input_sharding[op_type].keys()),
                list(self._output_sharding[op_type].keys()),
                list(meta.input_pre_sharding.keys()),
                list(meta.output_post_sharding.keys()),
            )

            # 2. Expected parameter names
            self._expected_state_names[op_type] = list(
                meta.get_arguments(is_output=False, arg_type=ArgumentType.STATE) or []
            )
            self._expected_tensor_names[op_type] = list(
                meta.get_arguments(is_output=False, arg_type=ArgumentType.TENSOR) or []
            )

            # 3. Parse predecessor topology from input_pre_sharding
            # Structure: {spmd_id: List[EdgeShardingInfo]}
            for pre_spmd_id, edge_info_list in meta.input_pre_sharding.items():
                merged_sharding: Dict[str, ShardingSpec] = {}
                condition_fn = None
                target_op = OpType.FORWARD  # Default, will be updated from edge_info
                for edge_info in edge_info_list:
                    merged_sharding.update(edge_info.tensor_sharding)
                    if edge_info.condition_fn is not None:
                        condition_fn = edge_info.condition_fn
                    # Get target_op from EdgeShardingInfo (if available)
                    if hasattr(edge_info, "target_op"):
                        target_op = edge_info.target_op

                # Infer total_rank from sharding spec's mesh if available
                total_rank = self._infer_total_rank_from_sharding(merged_sharding)

                self._predecessor_topology[op_type][pre_spmd_id] = SPMDTopologyInfo(
                    spmd_id=pre_spmd_id,
                    total_rank=total_rank,
                    mesh=self._extract_mesh_from_sharding(merged_sharding),
                    tensor_sharding=merged_sharding,
                    condition_fn=condition_fn,
                    target_op=target_op,
                    is_self_loop=(pre_spmd_id == self._spmd_id),
                )

            # 4. Parse successor topology from output_post_sharding
            # Structure: {spmd_id: List[EdgeShardingInfo]}
            for post_spmd_id, edge_info_list in meta.output_post_sharding.items():
                merged_sharding: Dict[str, Any] = {}
                condition_fn = None
                target_op = OpType.FORWARD  # Default, will be updated from edge_info
                for edge_info in edge_info_list:
                    merged_sharding.update(edge_info.tensor_sharding)
                    if edge_info.condition_fn is not None:
                        condition_fn = edge_info.condition_fn
                    # Get target_op from EdgeShardingInfo (if available)
                    if hasattr(edge_info, "target_op"):
                        target_op = edge_info.target_op

                total_rank = self._infer_total_rank_from_sharding(merged_sharding)

                self._successor_topology[op_type][post_spmd_id] = SPMDTopologyInfo(
                    spmd_id=post_spmd_id,
                    total_rank=total_rank,
                    mesh=self._extract_mesh_from_sharding(merged_sharding),
                    tensor_sharding=merged_sharding,
                    condition_fn=condition_fn,
                    target_op=target_op,
                    is_self_loop=(post_spmd_id == self._spmd_id),
                )

            # 5. Build output-to-input tensor name mapping (positional correspondence)
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
            self._output_to_input_tensor_mapping[op_type] = mapping

            logger.debug(
                f"Parsed operator meta for {op_type.name}: "
                f"state={self._expected_state_names[op_type]}, "
                f"tensor={self._expected_tensor_names[op_type]}, "
                f"predecessors={list(self._predecessor_topology[op_type].keys())}, "
                f"successors={list(self._successor_topology[op_type].keys())}, "
                f"output_to_input_tensor_mapping={mapping}"
            )

    def _parse_sharding_specs(
        self, raw_specs: Dict[str, Any]
    ) -> Dict[str, "ShardingSpec"]:
        """Convert raw sharding dictionaries into ShardingSpec objects."""
        from fluxway.common.types.sharding import (
            ShardingSpec,
            ChunkPartition,
            PartitionSpec,
        )

        parsed = {}
        for tensor_name, raw_spec in raw_specs.items():
            if isinstance(raw_spec, dict):
                sharding_type = raw_spec.get("type", "dense")
                if sharding_type == "replicated":
                    parsed[tensor_name] = ShardingSpec(
                        strategy=ChunkPartition(),
                        mesh=self._mesh,
                        partition=PartitionSpec(),
                        routing_dim=0,
                    )
                elif sharding_type == "sparse":
                    pass  # will handle shortly
            elif isinstance(raw_spec, ShardingSpec):
                parsed[tensor_name] = raw_spec
        return parsed

    def _get_operator_metadata(self, op_type: OpType) -> Optional["OperatorMeta"]:
        """
        Get metadata attached to an operator.

        Prefers explicit meta from SPMDConfig (properly serialized) over
        function attributes (lost when pickled by reference across processes).
        """
        # First try to get from SPMDConfig's explicit fields (properly serialized)
        if op_type == OpType.FORWARD and self._config.forward_meta is not None:
            return self._config.forward_meta
        elif op_type == OpType.BACKWARD and self._config.backward_meta is not None:
            return self._config.backward_meta
        elif op_type == OpType.UPDATE and self._config.update_meta is not None:
            return self._config.update_meta

        # Fallback to function attribute (may not have edge info in worker processes)
        operator = self._operators.get(op_type)
        if operator is None:
            return None
        return getattr(operator, "_flux_operator_meta", None)

    def _infer_total_rank_from_sharding(self, sharding_map: Dict[str, Any]) -> int:
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
        Also caches eager_trigger status in _eager_trigger_cache.

        Special case: if the operator has eager_trigger=True, all expected counts
        are set to 1, meaning the operator triggers as soon as any data arrives.
        """
        for op_type in [OpType.FORWARD, OpType.BACKWARD, OpType.UPDATE]:
            expected: Dict[str, int] = {}

            # Check if eager_trigger is enabled for this operator
            meta = self._get_operator_metadata(op_type)
            eager_trigger = meta.eager_trigger if meta is not None else False

            # Cache eager_trigger status
            self._eager_trigger_cache[op_type] = eager_trigger

            for tensor_name in self._expected_tensor_names.get(op_type, []):
                # If eager_trigger is enabled, expect only 1 shard to trigger execution
                if eager_trigger:
                    expected[tensor_name] = 1
                    continue

                total_count = 0
                input_spec = self._input_sharding.get(op_type, {}).get(tensor_name)
                input_is_sparse = (
                    input_spec.is_content_addressable if input_spec else False
                )

                for pre_spmd_id, pre_info in self._predecessor_topology[
                    op_type
                ].items():
                    pre_sharding = pre_info.tensor_sharding.get(tensor_name)
                    pre_is_sparse = (
                        pre_sharding.is_content_addressable if pre_sharding else False
                    )

                    if pre_is_sparse or input_is_sparse:
                        # Sparse involved: expect from all ranks
                        total_count += pre_info.total_rank
                    else:
                        # Dense-to-dense topology defaults to expecting shards
                        # based on layout. Since we removed _input_routing_strategies,
                        # AsyncDataDistributor fully encapsulates routing, we relax
                        # expected shards calculation or defer it. For now, fallback to total_rank
                        # or better logic if required (e.g. geometric overlap).
                        total_count += pre_info.total_rank

                expected[tensor_name] = total_count

            self._expected_tensor_shards_cache[op_type] = expected

            if expected:
                logger.debug(
                    f"Precomputed expected tensor shards for {op_type.name}: {expected}"
                    + (" (eager_trigger=True)" if eager_trigger else "")
                )

    def _precompute_operator_input_types(self) -> None:
        """
        Pre-compute operator input types from function signatures.

        Extracts the first parameter type (excluding self) from each operator function
        and caches it for use in _assemble_input.
        """
        for op_type, operator in self._operators.items():
            if operator is None:
                continue

            try:
                sig = inspect.signature(operator)
                params = list(sig.parameters.values())

                if not params:
                    logger.warning(f"Operator {op_type.name} has no parameters")
                    continue

                # Get the first parameter (excluding self)
                param = (
                    params[0]
                    if params[0].name != "self"
                    else (params[1] if len(params) > 1 else None)
                )
                if param is None:
                    logger.warning(f"Operator {op_type.name} has no valid parameters")
                    continue

                param_type = param.annotation
                if param_type == inspect.Parameter.empty:
                    logger.warning(
                        f"Operator {op_type.name} parameter '{param.name}' has no type annotation"
                    )
                    continue

                self._operator_input_types[op_type] = param_type
                logger.debug(f"Precomputed input type for {op_type.name}: {param_type}")

            except Exception as e:
                logger.warning(f"Failed to extract input type for {op_type.name}: {e}")
                continue

    # =========================================================================
    # Topology Query
    # =========================================================================

    def get_predecessor_ids(self, op_type: OpType) -> List[str]:
        """Get all predecessor SPMD IDs for an operator."""
        return list(self._predecessor_topology[op_type].keys())

    def get_successor_ids(self, op_type: OpType) -> List[str]:
        """Get all successor SPMD IDs for an operator."""
        return list(self._successor_topology[op_type].keys())

    def get_predecessor_info(
        self, op_type: OpType, spmd_id: str
    ) -> Optional[SPMDTopologyInfo]:
        """Get predecessor topology info."""
        return self._predecessor_topology[op_type].get(spmd_id)

    def get_successor_info(
        self, op_type: OpType, spmd_id: str
    ) -> Optional[SPMDTopologyInfo]:
        """Get successor topology info."""
        return self._successor_topology[op_type].get(spmd_id)

    def get_successor_condition_fn(
        self, op_type: OpType, spmd_id: str
    ) -> Optional[Callable[..., bool]]:
        """Get condition function for a successor edge."""
        info = self._successor_topology[op_type].get(spmd_id)
        return info.condition_fn if info else None

    # =========================================================================
    # Expected Data Query
    # =========================================================================

    def is_eager_trigger(self, op_type: OpType) -> bool:
        """
        Check if an operator is in eager_trigger mode.

        In eager_trigger mode:
        - The operator triggers immediately on any data, ignoring step ordering
        - No step matching required between sender and receiver
        - Expected tensor shards = 1

        Args:
            op_type: Operator type to check

        Returns:
            True if eager_trigger is enabled for this operator
        """
        return self._eager_trigger_cache.get(op_type, False)

    def get_expected_state_names(self, op_type: OpType) -> List[str]:
        """Get expected state variable names."""
        return list(self._expected_state_names.get(op_type, []))

    def get_expected_tensor_names(self, op_type: OpType) -> List[str]:
        """Get expected tensor variable names."""
        return list(self._expected_tensor_names.get(op_type, []))

    def get_expected_tensor_shards(self, op_type: OpType) -> Dict[str, int]:
        """
        Get pre-computed expected tensor shard counts based on topology.

        For each tensor:
        - Dense-to-Dense: compute geometric dependencies
        - Sparse involved: expect from all predecessor ranks

        The result is pre-computed during initialization and cached.

        Returns:
            {tensor_name: expected_shard_count}
        """
        return dict(self._expected_tensor_shards_cache.get(op_type, {}))

    def get_operator_input_type(self, op_type: OpType) -> Optional[Type]:
        """
        Get the input type (data_type class) expected by an operator.

        Returns:
            The Type of the data_type class, or None if not available
        """
        return self._operator_input_types.get(op_type)

    def get_output_to_input_tensor_mapping(self, op_type: OpType) -> Dict[str, str]:
        """
        Get the output-to-input tensor name mapping for an operator.

        Built at init from positional correspondence between input_arguments.tensor
        and output_arguments.tensor.  e.g. if input tensors are ["input"] and output
        tensors are ["tensor"], the mapping is {"tensor": "input"}.

        Returns:
            Dict mapping output tensor name → corresponding input tensor name.
        """
        return dict(self._output_to_input_tensor_mapping.get(op_type, {}))

    def get_input_sharding(self, op_type: OpType) -> Dict[str, ShardingSpec]:
        """Get input sharding specs for an operator."""
        return dict(self._input_sharding.get(op_type, {}))

    def get_predecessor_topology(self, op_type: OpType) -> Dict[str, SPMDTopologyInfo]:
        """Get predecessor topology info for an operator."""
        return dict(self._predecessor_topology.get(op_type, {}))

    # =========================================================================
    # Sharding Query
    # =========================================================================

    def get_output_sharding(self, op_type: OpType) -> Dict[str, ShardingSpec]:
        """Get output sharding for an operator."""
        return dict(self._output_sharding.get(op_type, {}))

    def get_output_state_names(self, op_type: OpType) -> List[str]:
        """
        Get output state argument names for an operator.

        These are the state variables that need to be sent to successors
        via the control plane (state_argument transmission path).

        Returns:
            List of state argument names from output_arguments
        """
        meta = self._get_operator_metadata(op_type)
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

    def execute(self, op_type: OpType, input_data: Any) -> Any:
        """
        Execute an operator.

        Args:
            op_type: Operator type to execute
            input_data: Assembled input data (already constructed as data_type object)

        Returns:
            Operator output
        """
        operator = self._operators.get(op_type)
        if operator is None:
            logger.warning(f"No operator registered for {op_type.name}")
            return None

        try:
            return operator(input_data)
        except Exception as e:
            logger.error(f"Execution failed for {op_type.name}: {e}")
            raise
