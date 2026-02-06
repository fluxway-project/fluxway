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
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TYPE_CHECKING
import threading
import logging
import inspect

from fluxway.common.types.data_types import ArgumentType, OpType, TensorMeta
from fluxway.common.types.sharding import (
    LayoutMesh,
    ShardingSpec,
    StaticShardingSpec,
    DynamicShardingSpec,
)
from fluxway.common.types.sharding import DenseShardingStrategy

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
    mesh: Optional[LayoutMesh] = None
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
        self._mesh = config.mesh  # LayoutMesh from config
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
        
        # Pre-computed routing strategies: {(op_type, successor_spmd_id, tensor_name): DenseShardingStrategy}
        self._routing_strategies: Dict[Tuple[OpType, str, str], DenseShardingStrategy] = {}
        
        # Pre-computed input routing strategies: {(op_type, predecessor_spmd_id, tensor_name): DenseShardingStrategy}
        # Note: For input, we compute from predecessor's perspective (predecessor -> this worker)
        self._input_routing_strategies: Dict[Tuple[OpType, str, str], DenseShardingStrategy] = {}
        
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
        
        # Pre-compute routing strategies after topology is parsed
        self._precompute_routing_strategies()
        
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
                self._input_sharding[op_type] = dict(meta.input_sharding)
            if meta.output_sharding:
                self._output_sharding[op_type] = dict(meta.output_sharding)
            
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
                    if hasattr(edge_info, 'target_op'):
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
                    if hasattr(edge_info, 'target_op'):
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
            if hasattr(spec, 'layout_mesh') and spec.layout_mesh is not None:
                return spec.layout_mesh.num_devices
        return 1
    
    def _extract_mesh_from_sharding(self, sharding_map: Dict[str, Any]) -> Optional[LayoutMesh]:
        """Extract mesh from sharding specs."""
        for spec in sharding_map.values():
            if hasattr(spec, 'layout_mesh') and spec.layout_mesh is not None:
                return spec.layout_mesh
        return None
    
    def _precompute_routing_strategies(self) -> None:
        """
        预计算所有 dense-to-dense 路由策略。
        
        遍历所有 op_type -> successor -> tensor 的组合，
        对于 dense-to-dense 的情况，创建并缓存 DenseShardingStrategy。
        
        同时预计算输入端的路由策略，用于计算从哪些前置 rank 接收数据。
        """
        for op_type in [OpType.FORWARD, OpType.BACKWARD, OpType.UPDATE]:
            output_sharding = self._output_sharding.get(op_type, {})
            input_sharding = self._input_sharding.get(op_type, {})
            
            # 1. 预计算输出路由策略 (this -> successor)
            for succ_id, succ_info in self._successor_topology[op_type].items():
                for tensor_name, post_spec in succ_info.tensor_sharding.items():
                    output_spec = output_sharding.get(tensor_name)
                    
                    # 只对 dense-to-dense 预计算
                    if (output_spec is not None and 
                        isinstance(output_spec, StaticShardingSpec) and
                        isinstance(post_spec, StaticShardingSpec)):
                        
                        strategy = DenseShardingStrategy.from_specs(
                            src_rank=self._rank,
                            src_spec=output_spec,
                            dst_spec=post_spec,
                        )
                        
                        cache_key = (op_type, succ_id, tensor_name)
                        self._routing_strategies[cache_key] = strategy
                        
                        logger.debug(
                            f"Precomputed output routing strategy for {op_type.name} -> "
                            f"{succ_id}:{tensor_name}, dependent_ranks={strategy.dependent_ranks}"
                        )
            
            # 2. 预计算输入路由策略 (predecessor -> this)
            # 对于每个前置 SPMD 的每个 rank，计算它是否会发送数据给当前 worker
            for pre_id, pre_info in self._predecessor_topology[op_type].items():
                for tensor_name, pre_sharding in pre_info.tensor_sharding.items():
                    input_spec = input_sharding.get(tensor_name)
                    
                    # 只对 dense-to-dense 预计算
                    if (input_spec is not None and 
                        isinstance(input_spec, StaticShardingSpec) and
                        isinstance(pre_sharding, StaticShardingSpec)):
                        
                        # 从前置的视角：前置发送 -> 当前接收
                        # 我们需要知道哪些前置 rank 会发送给当前 rank
                        # 策略：对于每个前置 rank，计算它的 dependent_ranks
                        # 如果当前 rank 在其 dependent_ranks 中，则该前置 rank 是依赖源
                        dependent_pre_ranks: List[int] = []
                        
                        for pre_rank in range(pre_info.total_rank):
                            strategy = DenseShardingStrategy.from_specs(
                                src_rank=pre_rank,
                                src_spec=pre_sharding,
                                dst_spec=input_spec,
                            )
                            # 如果当前 rank 在该前置 rank 的发送目标中
                            if self._rank in strategy.dependent_ranks:
                                dependent_pre_ranks.append(pre_rank)
                        
                        # 创建一个虚拟的策略对象来存储依赖的前置 rank
                        # 这里我们直接存储依赖列表，使用一个简化的 strategy
                        # 实际上我们只需要 dependent_ranks 这个字段
                        input_strategy = DenseShardingStrategy(
                            src_rank=self._rank,  # 当前 rank 作为接收方
                            src_mesh=input_spec.layout_mesh,
                            dst_mesh=pre_sharding.layout_mesh,
                            dim_infos=[],  # 不需要 dim_infos
                            src_normalized_axes=input_spec.normalized_axes,
                            dst_normalized_axes=pre_sharding.normalized_axes,
                            src_coords=input_spec.layout_mesh.get_coordinate(self._rank),
                            dependent_ranks=dependent_pre_ranks,  # 存储依赖的前置 rank
                        )
                        
                        cache_key = (op_type, pre_id, tensor_name)
                        self._input_routing_strategies[cache_key] = input_strategy
                        
                        logger.debug(
                            f"Precomputed input routing strategy for {pre_id}:{tensor_name} -> "
                            f"{op_type.name}, dependent_predecessor_ranks={dependent_pre_ranks}"
                        )
    
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
                input_is_sparse = isinstance(input_spec, DynamicShardingSpec)
                
                for pre_spmd_id, pre_info in self._predecessor_topology[op_type].items():
                    pre_sharding = pre_info.tensor_sharding.get(tensor_name)
                    pre_is_sparse = isinstance(pre_sharding, DynamicShardingSpec)
                    
                    if pre_is_sparse or input_is_sparse:
                        # Sparse involved: expect from all ranks
                        total_count += pre_info.total_rank
                    else:
                        # Dense-to-Dense: use pre-computed input routing strategies
                        cache_key = (op_type, pre_spmd_id, tensor_name)
                        strategy = self._input_routing_strategies.get(cache_key)
                        
                        if strategy is not None:
                            total_count += len(strategy.dependent_ranks)
                        else:
                            # Fallback: depend on all ranks
                            logger.warning(
                                f"No precomputed input strategy for {cache_key}, "
                                f"falling back to all-to-all"
                            )
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
                param = params[0] if params[0].name != 'self' else (params[1] if len(params) > 1 else None)
                if param is None:
                    logger.warning(f"Operator {op_type.name} has no valid parameters")
                    continue
                
                param_type = param.annotation
                if param_type == inspect.Parameter.empty:
                    logger.warning(f"Operator {op_type.name} parameter '{param.name}' has no type annotation")
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
    
    def get_predecessor_info(self, op_type: OpType, spmd_id: str) -> Optional[SPMDTopologyInfo]:
        """Get predecessor topology info."""
        return self._predecessor_topology[op_type].get(spmd_id)
    
    def get_successor_info(self, op_type: OpType, spmd_id: str) -> Optional[SPMDTopologyInfo]:
        """Get successor topology info."""
        return self._successor_topology[op_type].get(spmd_id)
    
    def get_successor_condition_fn(self, op_type: OpType, spmd_id: str) -> Optional[Callable[..., bool]]:
        """Get condition function for a successor edge."""
        info = self._successor_topology[op_type].get(spmd_id)
        return info.condition_fn if info else None
    
    def get_dependent_successor_ranks(
        self,
        op_type: OpType,
        successor_spmd_id: str,
        tensor_name: str,
    ) -> Optional[List[int]]:
        """
        获取当前 rank 依赖的后置 SPMD 的所有目标 rank。
        
        该方法直接返回预计算的结果，不需要 tensor 形状。
        仅对 dense-to-dense 路由有效。
        
        Args:
            op_type: 当前算子类型
            successor_spmd_id: 后置 SPMD ID
            tensor_name: 张量名称
            
        Returns:
            依赖的目标 rank 列表，如果不存在预计算策略则返回 None
        """
        cache_key = (op_type, successor_spmd_id, tensor_name)
        strategy = self._routing_strategies.get(cache_key)
        if strategy is None:
            return None
        return strategy.dependent_ranks
    
    def get_routing_strategy(
        self,
        op_type: OpType,
        successor_spmd_id: str,
        tensor_name: str,
    ) -> Optional[DenseShardingStrategy]:
        """
        获取预计算的路由策略。
        
        Args:
            op_type: 当前算子类型
            successor_spmd_id: 后置 SPMD ID
            tensor_name: 张量名称
            
        Returns:
            DenseShardingStrategy 或 None
        """
        cache_key = (op_type, successor_spmd_id, tensor_name)
        return self._routing_strategies.get(cache_key)
    
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
    
    def get_input_routing_strategy(
        self,
        op_type: OpType,
        pre_spmd_id: str,
        tensor_name: str,
    ) -> Optional[DenseShardingStrategy]:
        """
        Get pre-computed input routing strategy for a tensor.
        
        Args:
            op_type: Operator type
            pre_spmd_id: Predecessor SPMD ID
            tensor_name: Tensor name
            
        Returns:
            DenseShardingStrategy if available, None otherwise
        """
        cache_key = (op_type, pre_spmd_id, tensor_name)
        return self._input_routing_strategies.get(cache_key)
    
    # =========================================================================
    # Output Routing (4 cases)
    # =========================================================================
    
    def get_successor_target_route(
        self,
        current_rank: int,
        target_spmd_id: str,
        op_type: OpType,
        output_tensor_name: str,
        output_tensor: Any,
        output_tensor_meta: TensorMeta,
    ) -> List[Tuple[Tuple[slice, ...], int]]:
        """
        Compute output routing table for a successor.
        
        Four cases based on source/dest sparse/dense:
        1. dense-to-dense: geometric intersection
        2. dense-to-sparse: distribute by rule
        3. sparse-to-dense: use global_indices to compute targets
        4. sparse-to-sparse: distribute by rule
        
        Args:
            target_spmd_id: Target successor SPMD ID
            op_type: Current operator type
            output_tensor_name: Name of the output tensor
            output_tensor: The actual tensor data
            output_tensor_meta: Tensor metadata (contains global_indices for sparse)
        
        Returns:
            List of (tensor_slice, target_rank) tuples
        """
        succ_info = self._successor_topology[op_type].get(target_spmd_id)
        if succ_info is None:
            logger.warning(f"Successor {target_spmd_id} not found for {op_type.name}")
            return []
        
        output_spec = self._output_sharding.get(op_type, {}).get(output_tensor_name)
        post_spec = succ_info.tensor_sharding.get(output_tensor_name)
        
        src_is_sparse = isinstance(output_spec, DynamicShardingSpec)
        dst_is_sparse = isinstance(post_spec, DynamicShardingSpec)
        
        if not src_is_sparse and not dst_is_sparse:
            # 优先使用预计算的策略
            cache_key = (op_type, target_spmd_id, output_tensor_name)
            strategy = self._routing_strategies.get(cache_key)
            if strategy is not None:
                return strategy.compute_route(output_tensor_meta.shape)
            # 回退到原始计算（不应该发生，但保持兼容性）
            return self._route_dense_to_dense(
                current_rank, output_tensor_meta.shape, output_spec, post_spec
            )
        elif not src_is_sparse and dst_is_sparse:
            # Only dst is dynamic, use dst's fn
            return self._route_dense_to_sparse(
                output_tensor, output_tensor_meta, post_spec
            )
        elif src_is_sparse and not dst_is_sparse:
            # Only src is dynamic, use src's fn
            return self._route_sparse_to_dense(
                output_tensor, output_tensor_meta, output_spec, post_spec
            )
        else:
            # Both are dynamic, use src (current SPMD) fn
            return self._route_sparse_to_sparse(
                output_tensor, output_tensor_meta, output_spec, post_spec
            )

    @staticmethod
    def _route_dense_to_dense(
        current_rank: int,
        local_shape: Tuple[int, ...],
        src_spec: StaticShardingSpec,
        dst_spec: StaticShardingSpec,
    ) -> List[Tuple[Tuple[slice, ...], int]]:
        """
        完全基于本地信息推导全局分布并计算发送 Mask。
        
        对于每个目标 rank，计算其在全局坐标系下的区间与当前 worker 区间的交集，
        然后将交集转换为相对于本地 tensor 的切片（即本地索引范围）。
        
        Args:
            current_rank: 当前 worker 的 rank
            local_shape: 本地 tensor 的形状
            src_spec: 源端的 sharding spec
            dst_spec: 目标端的 sharding spec
            
        Returns:
            List[Tuple[Tuple[slice, ...], dst_rank]]
            每个元组包含：本地 tensor 的切片元组（可直接用于索引）和目标 rank
        """
        num_dims = len(local_shape)
        
        # --- 1. 逆向推导 Global Shape ---
        # 根据当前 local_shape 和 sharding 策略还原逻辑上的全局大小
        global_shape = src_spec.infer_global_shape(local_shape, current_rank)

        # --- 2. 计算当前 Worker (Source) 的全局偏移区间 ---
        src_layout = src_spec.lower(global_shape, current_rank)
        src_slices = src_layout.dense_slices

        # --- 3. 计算目标各 Rank 的区间并求交集 ---
        results: List[Tuple[Tuple[slice, ...], int]] = []

        for dst_rank in range(dst_spec.layout_mesh.num_devices):
            dst_layout = dst_spec.lower(global_shape, dst_rank)
            dst_slices = dst_layout.dense_slices
            
            # 计算每个维度的切片
            local_slices: List[slice] = []
            has_intersection = True
            
            for dim_idx in range(num_dims):
                src_slice = src_slices[dim_idx]
                dst_slice = dst_slices[dim_idx]
                
                # 在全局坐标系下计算交集
                global_start = max(src_slice.start, dst_slice.start)
                global_end = min(src_slice.end, dst_slice.end)
                
                if global_start >= global_end:
                    # 该维度没有交集，跳过这个 dst_rank
                    has_intersection = False
                    break
                
                # 转换为本地坐标（相对于 src_slice.start）
                local_start = global_start - src_slice.start
                local_end = global_end - src_slice.start
                
                local_slices.append(slice(local_start, local_end))
            
            if has_intersection:
                results.append((tuple(local_slices), dst_rank))

        return results
    
    def _route_dense_to_sparse(
        self,
        tensor_data: Any,
        tensor_meta: TensorMeta,
        post_spec: DynamicShardingSpec,
    ) -> List[Tuple[Tuple[slice, ...], int]]:
        """
        Dense-to-Sparse routing: use post_spec.fn (only dst is dynamic).
        
        Args:
            tensor_data: The actual tensor data
            tensor_meta: Tensor metadata
            post_spec: Destination DynamicShardingSpec (contains fn)
        
        Returns:
            List of (tensor_slice, target_rank) tuples from fn result
        """
        # Only dst is dynamic, use dst's fn
        # fn(tensor, tensor_meta, post_sharding_spec) -> routing result
        return post_spec.fn(tensor_data, tensor_meta, post_spec)
    
    def _route_sparse_to_dense(
        self,
        tensor_data: Any,
        tensor_meta: TensorMeta,
        output_spec: DynamicShardingSpec,
        post_spec: StaticShardingSpec,
    ) -> List[Tuple[Tuple[slice, ...], int]]:
        """
        Sparse-to-Dense routing: use output_spec.fn (only src is dynamic).
        
        Args:
            tensor_data: The actual tensor data
            tensor_meta: Tensor metadata
            output_spec: Source DynamicShardingSpec (contains fn)
            post_spec: Destination StaticShardingSpec (provides mesh and axis info)
        
        Returns:
            List of (tensor_slice, target_rank) tuples from fn result
        """
        # Only src is dynamic, use src's fn
        # fn(tensor, tensor_meta, post_sharding_spec) -> routing result
        return output_spec.fn(tensor_data, tensor_meta, post_spec)
    
    def _route_sparse_to_sparse(
        self,
        tensor_data: Any,
        tensor_meta: TensorMeta,
        output_spec: DynamicShardingSpec,
        post_spec: DynamicShardingSpec,
    ) -> List[Tuple[Tuple[slice, ...], int]]:
        """
        Sparse-to-Sparse routing: use output_spec.fn (both are dynamic, use src's fn).
        
        Args:
            tensor_data: The actual tensor data
            tensor_meta: Tensor metadata
            output_spec: Source DynamicShardingSpec (contains fn to use)
            post_spec: Destination DynamicShardingSpec (provides mesh and axis info)
        
        Returns:
            List of (tensor_slice, target_rank) tuples from fn result
        """
        # Both are dynamic, use src (current SPMD) fn
        # fn(tensor, tensor_meta, post_sharding_spec) -> routing result
        return output_spec.fn(tensor_data, tensor_meta, post_spec)
    
    def has_sparse_route_fn(
        self,
        op_type: OpType,
        target_spmd_id: str,
        output_tensor_name: str,
    ) -> bool:
        """
        Check whether the sparse routing path for a given successor has a
        user-provided routing function (``DynamicShardingSpec.fn``).

        The check mirrors the ``get_successor_target_route`` routing-case
        selection:

        * dense-to-dense → always ``False`` (no sparse involved).
        * dense-to-sparse → ``post_spec.fn is not None``.
        * sparse-to-dense → ``output_spec.fn is not None``.
        * sparse-to-sparse → ``output_spec.fn is not None``.

        Returns ``False`` when the topology / sharding info cannot be found.
        """
        succ_info = self._successor_topology[op_type].get(target_spmd_id)
        if succ_info is None:
            return False

        output_spec = self._output_sharding.get(op_type, {}).get(output_tensor_name)
        post_spec = succ_info.tensor_sharding.get(output_tensor_name)

        src_is_sparse = isinstance(output_spec, DynamicShardingSpec)
        dst_is_sparse = isinstance(post_spec, DynamicShardingSpec)

        if not src_is_sparse and not dst_is_sparse:
            return False  # dense-to-dense, no route fn concept
        elif not src_is_sparse and dst_is_sparse:
            return post_spec.fn is not None
        elif src_is_sparse and not dst_is_sparse:
            return output_spec.fn is not None
        else:
            return output_spec.fn is not None

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
