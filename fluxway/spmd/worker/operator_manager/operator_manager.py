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
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionDispatcher,
    create_dispatcher,
)

if TYPE_CHECKING:
    from fluxway.common.types import SPMDConfig, OperatorMeta

logger = logging.getLogger(__name__)


# =============================================================================
# Topology Info (parsed from OperatorMeta, not runtime registered)
# =============================================================================


@dataclass
class ResolvedTopologyInfo:
    """
    Runtime-resolved topology info for a predecessor or successor SPMD.

    This is an internal worker-side object produced by
    OperatorManager._parse_operator_metadata().  It enriches the
    serialisable SPMDTopologyInfo DTO (from configs.py) with
    fields that are only meaningful inside a live worker process:

    * mesh / world_size - resolved from parsed ShardingSpec
    * tuuid_sharding - raw dicts converted to ShardingSpec objects
    * target_op, is_interleaved, interleave_unit_size - merged
      from all EdgeShardingInfo entries on the edge
    * is_self_loop - derived from SPMD identity comparison

    Do **not** use this class on the control plane or across Ray object
    boundaries; use ``configs.SPMDTopologyInfo`` for serialisable topology.
    """

    spmd_id: str
    world_size: int = 1

    mesh: Optional[DeviceMesh] = None
    # Sharding by TUUID
    tuuid_sharding: Dict[int, ShardingSpec] = field(default_factory=dict)
    # Map from logical path to TUUID
    tuuid_map: Dict[str, int] = field(default_factory=dict)

    # Condition function for conditional edges (only for successors)
    condition_fn: Optional[Callable[..., bool]] = None
    # Whether this is a step-interleaved data stream. Every interleaved worker unit has its own step and is regarded as a separated sender, a fake SPMD.
    is_interleaved: bool = False
    # Size of the interleaved group
    interleave_unit_size: int = 1
    # Whether this is a self-loop edge (source_node == target_node)
    is_self_loop: bool = False
    # Link to the symmetric edge – carried over from configs.SPMDTopologyInfo
    symmetric_to: Optional[tuple] = None
    # For hierarchical sharding resolution
    input_param_map: Dict[str, List[str]] = field(default_factory=dict)
    output_param_map: Dict[str, List[str]] = field(default_factory=dict)


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
        self._input_sharding: Dict[str, Dict[int, ShardingSpec]] = {}
        self._output_sharding: Dict[str, Dict[int, ShardingSpec]] = {}
        self._expected_state_names: Dict[str, List[str]] = {}
        self._expected_tensor_names: Dict[str, List[str]] = {}
        self._predecessor_topology: Dict[str, Dict[str, ResolvedTopologyInfo]] = {}
        self._successor_topology: Dict[str, Dict[str, ResolvedTopologyInfo]] = {}
        self._expected_tensor_shards_cache: Dict[str, Dict[int, int]] = {}
        self._periodic_trigger_cache: Dict[str, bool] = {}
        self._operator_input_types: Dict[str, Dict[str, Type]] = {}
        self._dispatchers: Dict[str, ExecutionDispatcher] = {}
        # Global dispatch table: {tensor_tuuid: [op_name1, op_name2]}
        self._tuuid_to_op_inputs: Dict[int, List[str]] = {}

        for op_name in self._operators:
            self._input_sharding[op_name] = {}
            self._output_sharding[op_name] = {}
            self._expected_state_names[op_name] = []
            self._expected_tensor_names[op_name] = []
            self._predecessor_topology[op_name] = {}
            self._successor_topology[op_name] = {}
            self._operator_input_types[op_name] = {}

        # Cache for TUUID to arg name mapping (for debugging/errors)
        self._output_tuuid_to_name_cache: Dict[str, Optional[Dict[int, str]]] = {}

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

    def _get_spec_by_tuuid(
        self, sharding_map: Dict[int, ShardingSpec], tuuid: int
    ) -> Optional[ShardingSpec]:
        """Direct lookup for sharding spec by TUUID."""
        return sharding_map.get(tuuid)

    def _get_effective_sharding_spec(
        self, spec: Optional[ShardingSpec], info: ResolvedTopologyInfo
    ) -> Optional[ShardingSpec]:
        """
        Extract the effective sharding spec for routing/assembly.
        If the edge is interleaved, this returns a spec projected onto a sub-mesh
        matching the interleave unit size.
        """
        if spec is None:
            return None

        if not info.is_interleaved:
            return spec

        # INTERLEAVED SOURCE ABSTRACTION:
        # We must override the spec's mesh to the active unit size (typically 1 for source)
        # so that resolve_producers/resolve_consumers correctly maps a single active source
        # rank to all its targets.
        if spec.mesh:
            from fluxway.common.mesh import DeviceMesh

            base_mesh = spec.mesh
            # INTERLEAVED: Override the data dimension to correctly represent the unit size.
            unit_size = info.interleave_unit_size
            other_dims_size = 1
            for s in base_mesh.shape[1:]:
                other_dims_size *= s

            if unit_size % other_dims_size != 0:
                raise ValueError(
                    f"[OperatorManager] Interleave unit size {unit_size} is not compatible with "
                    f"sub-mesh axes {base_mesh.axis_names[1:]} having total size {other_dims_size}. "
                    "The unit size must be a multiple of the size of the non-sharded dimensions."
                )

            first_axis_size = unit_size // other_dims_size
            new_shape = (first_axis_size,) + base_mesh.shape[1:]

            sub_mesh = DeviceMesh(
                device_shape=new_shape, axis_names=base_mesh.axis_names
            )
            return spec.with_mesh(sub_mesh)
        return spec

    def resolve_tensor_producers(
        self,
        op_name: str,
        tensor_tuuid: int,
        pre_spmd_id: str,
        current_rank: int,
        step: int = 0,
    ) -> List[int]:
        """
        Locate the physical source ranks producing a specific tensor shard by its TUUID.
        """
        input_spec = self._get_spec_by_tuuid(
            self._input_sharding.get(op_name, {}), tensor_tuuid
        )

        pre_info = self._predecessor_topology.get(op_name, {}).get(pre_spmd_id)
        if not pre_info:
            return []

        pre_sharding = self._get_spec_by_tuuid(pre_info.tuuid_sharding, tensor_tuuid)

        # Resolve effective sharding (handles interleaved sub-mesh projections)
        effective_pre_spec = self._get_effective_sharding_spec(pre_sharding, pre_info)

        # Handle Interleaved Unit indexing (shifting base rank based on step)
        unit_base_rank = 0
        if pre_info.is_interleaved:
            total_units = pre_info.world_size // pre_info.interleave_unit_size
            active_unit_idx = step % total_units
            unit_base_rank = active_unit_idx * pre_info.interleave_unit_size

        if effective_pre_spec:
            if input_spec is None:
                raise ValueError(
                    f"[OperatorManager] Missing input sharding for TUUID {tensor_tuuid} in '{op_name}'."
                )

            relative_producers = effective_pre_spec.resolve_producers(
                input_spec, current_rank
            )
            return [unit_base_rank + r for r in relative_producers]

        if pre_info.is_interleaved:
            raise ValueError(
                f"[OperatorManager] Missing sharding specs for interleaved source "
                f"TUUID {tensor_tuuid} from {pre_spmd_id} in {op_name}."
            )

        if input_spec is not None:
            raise ValueError(
                f"[OperatorManager] Partial sharding info for TUUID {tensor_tuuid} in {op_name}: target has spec but source doesn't."
            )

        raise ValueError(
            f"[OperatorManager] Missing sharding specifications for TUUID {tensor_tuuid} "
            f"from {pre_spmd_id} in {op_name}. In strict TUUID mode, all tensors must be explicitly sharded."
        )

    # =========================================================================
    # Metadata Parsing (at initialization)
    # =========================================================================

    def _parse_operator_metadata(self) -> None:
        """
        Parse operator metadata from @operator decorator and config.topology.
        """
        if self._config and self._config.topology:
            for op_name, top in self._config.topology.items():
                self._predecessor_topology.setdefault(op_name, {})
                for pred_id, info in top.predecessors.items():
                    # Link build-time UUIDs to runtime ShardingSpecs
                    sharding = {}
                    if info.tuuid_sharding:
                        pred_mesh = info.mesh if info.mesh is not None else self._mesh
                        sharding = self._build_sharding_specs(
                            info.tuuid_sharding, pred_mesh
                        )

                    self._predecessor_topology[op_name][pred_id] = ResolvedTopologyInfo(
                        spmd_id=info.spmd_id,
                        world_size=info.world_size,
                        mesh=info.mesh,
                        tuuid_sharding=sharding,
                        tuuid_map=info.tuuid_map or {},
                        condition_fn=info.condition_fn,
                        symmetric_to=info.symmetric_to,
                        is_interleaved=getattr(info, "is_interleaved", False),
                        interleave_unit_size=getattr(info, "interleave_unit_size", 1),
                        is_self_loop=(info.spmd_id == self._spmd_id),
                    )

                self._successor_topology.setdefault(op_name, {})
                for succ_id, info in top.successors.items():
                    sharding = {}
                    if info.tuuid_sharding:
                        succ_mesh = info.mesh if info.mesh is not None else self._mesh
                        sharding = self._build_sharding_specs(
                            info.tuuid_sharding, succ_mesh
                        )

                    self._successor_topology[op_name][succ_id] = ResolvedTopologyInfo(
                        spmd_id=info.spmd_id,
                        world_size=info.world_size,
                        mesh=info.mesh,
                        tuuid_sharding=sharding,
                        tuuid_map=info.tuuid_map or {},
                        condition_fn=info.condition_fn,
                        symmetric_to=info.symmetric_to,
                        is_interleaved=getattr(info, "is_interleaved", False),
                        interleave_unit_size=getattr(info, "interleave_unit_size", 1),
                        is_self_loop=(info.spmd_id == self._spmd_id),
                    )

        if self._mesh is None:
            raise ValueError(f"OperatorManager[{self._spmd_id}]: mesh is None")

        # Enrich with logical sharding from @operator
        for op_name, operator in self._operators.items():
            if operator is None:
                continue

            meta = self._get_operator_metadata(op_name)
            if meta is None:
                continue

            self._dispatchers[op_name] = create_dispatcher(meta)

            local_input_specs = self._build_sharding_specs(
                meta.input_sharding, self._mesh
            )
            # This ensures that all routing and sharding lookups use TUUIDs.
            for pre_info in self._predecessor_topology.get(op_name, {}).values():
                # Bind Local Spec with TUUID by [Injected Argument Name -> TUUID] dict.
                # The local function arguments name is the pre_info.tuuid_map key, which was binded in _validate_and_link_edge
                for arg_name, tuuid in pre_info.tuuid_map.items():
                    if arg_name in local_input_specs:
                        self._input_sharding[op_name][tuuid] = local_input_specs[
                            arg_name
                        ]

            self._expected_state_names[op_name] = list(
                meta.get_arguments(is_output=False, arg_type=ArgumentType.CONTROL) or []
            )
            self._expected_tensor_names[op_name] = list(
                meta.get_arguments(is_output=False, arg_type=ArgumentType.DATA) or []
            )

            input_types = self._operator_input_types.get(op_name, {})
            pre_topology = self._predecessor_topology.get(op_name, {})

            # Combine all available fields from all predecessors
            available_paths = set()
            for pre_info in pre_topology.values():
                available_paths.update(pre_info.tuuid_map.keys())

            for p_name, p_type in input_types.items():
                if p_name in self._static_args:
                    continue

                # Use treedef from meta if available, otherwise it's a primitive tensor
                tdef = meta.input_treedefs.get(p_name)
                if tdef:
                    # Flatten treedef to get expected leaves/paths
                    # We reuse tree_flatten_type logic to identify expected paths
                    # For @data_type, build paths based on definition order
                    expected_fields = meta.input_param_map.get(p_name, [])
                    for field in expected_fields:
                        if field not in available_paths:
                            raise ValueError(
                                f"[Fluxway Init Error] Operator '{op_name}' requires field '{field}' "
                                f"as part of parameter '{p_name}', but it's not provided by any predecessor.\n"
                                f"Available fields: {sorted(available_paths)}\n"
                                f"Check your graph configuration."
                            )
                else:
                    # Single primitive tensor
                    if (
                        p_name not in available_paths
                        and p_name not in self._expected_state_names[op_name]
                    ):
                        raise ValueError(
                            f"[Fluxway Init Error] Operator '{op_name}' requires parameter '{p_name}', "
                            "but it's not in predecessor outputs, not in control states, "
                            "nor in static arguments.\n"
                            f"Available predecessor outputs: {sorted(available_paths)}\n"
                            f"Registered control states: {self._expected_state_names[op_name]}\n"
                            f"Static arguments: {list(self._static_args.keys())}"
                        )

            # Resolve staleness configuration keys to TUUIDs where applicable
            from fluxway.common.types.configs import TolerantBarrierConfig

            if meta.barrier_config and isinstance(
                meta.barrier_config, TolerantBarrierConfig
            ):
                resolved_staleness = {}
                for name_or_path, value in meta.barrier_config.staleness.items():
                    # Check if it's a tensor field from predecessors
                    tuuid = None
                    for pre_info in pre_topology.values():
                        if name_or_path in pre_info.tuuid_map:
                            tuuid = pre_info.tuuid_map[name_or_path]
                            break

                    if tuuid is not None:
                        resolved_staleness[tuuid] = value
                    elif name_or_path in self._expected_state_names[op_name]:
                        # Valid control state
                        resolved_staleness[name_or_path] = value
                    else:
                        # Neither a tensor field nor a state -> Strict error
                        available_fields = set()
                        for pre_info in pre_topology.values():
                            available_fields.update(pre_info.tuuid_map.keys())

                        raise ValueError(
                            f"[Fluxway Init Error] Operator '{op_name}' has staleness config for '{name_or_path}', "
                            "but it is neither a valid input tensor field provided by any predecessor "
                            "nor a registered control state parameter.\n"
                            f"Available fields: {sorted(available_fields)}\n"
                            f"Expected states: {self._expected_state_names[op_name]}"
                        )

                meta.barrier_config = meta.barrier_config.model_copy(
                    update={"staleness": resolved_staleness}
                )

            # Build global TUUID -> Op dispatch table
            for u in self._input_sharding[op_name]:
                if u not in self._tuuid_to_op_inputs:
                    self._tuuid_to_op_inputs[u] = []
                if op_name not in self._tuuid_to_op_inputs[u]:
                    self._tuuid_to_op_inputs[u].append(op_name)

            logger.debug(
                f"Parsed operator meta for {op_name}: "
                f"state={self._expected_state_names[op_name]}, "
                f"tensor={self._expected_tensor_names[op_name]}, "
                f"predecessors={list(self._predecessor_topology[op_name].keys())}, "
                f"successors={list(self._successor_topology[op_name].keys())}"
            )

    def register_eager_inputs(
        self,
        op_name: str,
        tuuid_sharding: Dict[int, ShardingSpec],
        expected_counts: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Dynamically register TUUIDs and their sharding specs for Eager mode.
        """
        with self._lock:
            if op_name not in self._input_sharding:
                self._input_sharding[op_name] = {}
            if op_name not in self._expected_tensor_shards_cache:
                self._expected_tensor_shards_cache[op_name] = {}

            for u, spec in tuuid_sharding.items():
                self._input_sharding[op_name][u] = spec

                # Update global TUUID -> Op dispatch table
                if u not in self._tuuid_to_op_inputs:
                    self._tuuid_to_op_inputs[u] = []
                if op_name not in self._tuuid_to_op_inputs[u]:
                    self._tuuid_to_op_inputs[u].append(op_name)

                # Update expected shard counts for the mailbox
                if expected_counts and u in expected_counts:
                    self._expected_tensor_shards_cache[op_name][u] = expected_counts[u]
                else:
                    # Default: expect one shard from each rank in the source mesh
                    self._expected_tensor_shards_cache[op_name][u] = (
                        spec.mesh.size if spec.mesh else 1
                    )

        logger.debug(
            f"[OperatorManager] Registered eager inputs for {op_name}: "
            f"{list(tuuid_sharding.keys())}"
        )

    def _build_sharding_specs(
        self, raw_specs: Dict[Any, Any], mesh: "DeviceMesh"
    ) -> Dict[Any, "ShardingSpec"]:
        from fluxway.common.types.sharding import ShardingSpec, ShardingConfig

        parsed = {}
        for key, raw_spec in raw_specs.items():
            if isinstance(raw_spec, dict):
                parsed[key] = ShardingConfig.model_validate(raw_spec).to_spec(mesh)
            elif isinstance(raw_spec, ShardingSpec):
                parsed[key] = raw_spec
        return parsed

    def _get_operator_metadata(self, op_name: str) -> "OperatorMeta":
        """
        Get metadata attached to an operator.
        All metadata must be explicitly provided in SPMDConfig.operator_metas.
        """
        if op_name in self._config.operator_metas:
            meta = self._config.operator_metas[op_name]
            if meta is not None:
                return meta

        raise ValueError(
            f"Operator '{op_name}' has no registered metadata in SPMDConfig. "
            "In strict TUUID mode, all operators must have explicit OperatorMeta."
        )

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
            expected: Dict[int, int] = {}

            # Check if periodic_trigger is enabled for this operator
            is_periodic = self.is_periodic_trigger(op_name)

            # Map input names to UUIDs and sum up expected shards
            pre_topology = self._predecessor_topology.get(op_name, {})

            for pre_id, pre_info in pre_topology.items():
                for u, spec in pre_info.tuuid_sharding.items():
                    if is_periodic:
                        # If periodic_trigger is enabled, expect N shards to trigger execution (default N=1)
                        from fluxway.common.types.configs import PeriodicBarrierConfig

                        meta = self._get_operator_metadata(op_name)
                        n = 1
                        if meta and isinstance(
                            meta.barrier_config, PeriodicBarrierConfig
                        ):
                            n = meta.barrier_config.n
                        expected[u] = n
                    else:
                        # Use the unified topology model to resolve physical producers.
                        producers = self.resolve_tensor_producers(
                            op_name, u, pre_id, self._rank, step=0
                        )
                        expected[u] = expected.get(u, 0) + len(producers)

            self._expected_tensor_shards_cache[op_name] = expected
            self._periodic_trigger_cache[op_name] = is_periodic

            if expected:
                logger.debug(
                    f"Precomputed expected tensor shards for {op_name}: {expected}"
                    + (" (periodic_trigger=True)" if is_periodic else "")
                )

    def _precompute_operator_input_types(self) -> None:
        """
        Pre-compute operator input types from function signatures.

        Extracts all parameter types (excluding self) from each operator function
        and caches them for use in _assemble_input.
        """
        for op_name, operator in self._operators.items():
            if operator is None:
                continue

            try:
                sig = inspect.signature(operator)
                is_source = self.is_source_operator(op_name)

                params_dict = {}
                for p_name, param in sig.parameters.items():
                    if p_name in ("self", "cls"):
                        continue
                    param_type = param.annotation
                    if param_type == inspect.Parameter.empty:
                        logger.warning(
                            f"Operator {op_name} parameter '{p_name}' has no type annotation"
                        )
                        continue
                    params_dict[p_name] = param_type

                if not params_dict:
                    if is_source:
                        logger.debug(
                            f"Operator {op_name} is a source, skipping parameter type pre-computation."
                        )
                    else:
                        logger.warning(
                            f"Operator {op_name} has no valid parameters to receive data, "
                            f"but it is not marked as a source operator."
                        )

                self._operator_input_types[op_name] = params_dict
                logger.debug(f"Precomputed input types for {op_name}: {params_dict}")

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
    ) -> Optional[ResolvedTopologyInfo]:
        """Get predecessor topology info."""
        return self._predecessor_topology.get(op_name, {}).get(spmd_id)

    def get_successor_info(
        self, op_name: str, spmd_id: str
    ) -> Optional[ResolvedTopologyInfo]:
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

    def get_operator_metadata(self, op_name: str) -> "OperatorMeta":
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

    def get_expected_tuuids(self, op_name: str) -> List[int]:
        """Get expected TUUIDs for an operator."""
        return list(self._expected_tensor_shards_cache.get(op_name, {}).keys())

    def is_source_operator(self, op_name: str) -> bool:
        """
        Check if an operator is a source operator (data producer).
        A source operator is identified by:
        1. Explicit metadata: @operator(..., is_source=True).
        2. Graph Topology: 0 predecessors and 0 input tensor arguments.

        Note: Source operators can still have static/state arguments (e.g., from Driver),
        but they do not depend on other operators for execution triggers.
        """
        meta = self._get_operator_metadata(op_name)
        if meta and meta.is_source is True:
            # Strong validation: A source should not have tensor inputs from other nodes
            pred_ids = self.get_predecessor_ids(op_name)
            if pred_ids:
                logger.warning(
                    f"[Fluxway Operator Warning] Operator '{op_name}' is explicitly marked as 'is_source=True', "
                    f"but it has predecessor SPMDs: {pred_ids}. This might indicate a graph configuration error."
                )
            return True

        # Strict Mode: Heuristic detection is disabled. Operators must be explicitly marked as sources.
        return False

    def get_source_operators(self) -> List[str]:
        """Get names of all source operators."""
        return [op for op in self._operators if self.is_source_operator(op)]

    def get_expected_tensor_shards(self, op_name: str) -> Dict[int, int]:
        """
        Get pre-computed expected tensor shard counts based on topology.
        Returns a dictionary mapping TUUIDs (int) to counts.
        """
        return self._expected_tensor_shards_cache.get(op_name, {})

    def get_operator_input_types(self, op_name: str) -> Dict[str, Type]:
        """
        Get the input parameter types expected by an operator.

        Returns:
            A dictionary mapping parameter names to their Types
        """
        return self._operator_input_types.get(op_name, {})

    def get_output_tuuids(self, op_name: str) -> List[int]:
        """Returns ordered list of UUIDs matching tree_flatten(output) order."""
        meta = self._get_operator_metadata(op_name)
        if meta and meta.output_uuids:
            return meta.output_uuids
        return []

    def get_output_tuuid_to_name(self, op_name: str) -> Optional[Dict[int, str]]:
        """Get cached TUUID-to-name mapping for operator outputs."""
        if op_name not in self._output_tuuid_to_name_cache:
            meta = self._get_operator_metadata(op_name)
            output_uuids = self.get_output_tuuids(op_name)
            if meta and output_uuids:
                out_args = meta.get_arguments(
                    is_output=True, arg_type=ArgumentType.DATA
                )
                if out_args and len(out_args) == len(output_uuids):
                    self._output_tuuid_to_name_cache[op_name] = dict(
                        zip(output_uuids, out_args, strict=True)  # type: ignore
                    )
                else:
                    self._output_tuuid_to_name_cache[op_name] = None
            else:
                self._output_tuuid_to_name_cache[op_name] = None
        return self._output_tuuid_to_name_cache[op_name]

    def get_input_sharding(self, op_name: str) -> Dict[int, ShardingSpec]:
        """
        Get sharding specifications for an operator's inputs by TUUID.
        Returns {tuuid: ShardingSpec}.
        """
        return self._input_sharding.get(op_name, {})

    def get_operators_for_tuuid(self, tuuid: int) -> List[str]:
        """Returns all local operator names that consume the given TUUID."""
        return self._tuuid_to_op_inputs.get(tuuid, [])

    def get_predecessor_topology(self, op_name: str) -> Dict[str, ResolvedTopologyInfo]:
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
        self, op_name: str, spmd_id: str, tuuid: int
    ) -> Optional[ShardingSpec]:
        """
        Resolve the ShardingSpec for a specific tensor TUUID coming from a specific predecessor.
        """
        pre_info = self.get_predecessor_info(op_name, spmd_id)
        if not pre_info:
            return None

        return pre_info.tuuid_sharding.get(tuuid)

    def get_sender_world_size(
        self, op_name: str, sender_id: str, tuuid: int
    ) -> Optional[int]:
        """
        Retrieve the world size of the sender for a given TUUID.

        Args:
            op_name: Local operator name consuming the tensor.
            sender_id: ID of the sender SPMD.
            tuuid: Unique identifier of the tensor.

        Returns:
            The world size of the sender, or None if not found.
        """
        # Try resolving from pre-defined topology info
        pre_info = self.get_predecessor_info(op_name, sender_id)
        if pre_info:
            return pre_info.world_size

        # Fallback for Eager mode or dynamic inputs: check input_sharding mesh
        spec = self._input_sharding.get(op_name, {}).get(tuuid)
        if spec and spec.mesh:
            return spec.mesh.size

        raise ValueError(
            f"[OperatorManager] Cannot resolve sender world size: "
            f"op={op_name}, sender={sender_id}, tuuid={tuuid}. "
            f"Ensure topology is correctly configured or sharding is registered."
        )

    # =========================================================================
    # Sharding Query
    # =========================================================================

    def get_output_sharding(self, op_name: str) -> Dict[int, "ShardingSpec"]:
        """
        Get output sharding for an operator, keyed by TUUID.
        """
        meta = self._get_operator_metadata(op_name)
        if not meta:
            return {}

        from fluxway.common.types.sharding import ShardingConfig

        # Build a name-to-TUUID map from local metadata.
        out_args = meta.get_arguments(is_output=True, arg_type=ArgumentType.DATA)
        name_to_tuuid = dict(zip(out_args, meta.output_uuids, strict=True))  # type:ignore

        res = {}
        if meta.output_sharding:
            for key, spec_dict in meta.output_sharding.items():
                if isinstance(spec_dict, dict):
                    spec = ShardingConfig.model_validate(spec_dict).to_spec(self._mesh)
                else:
                    spec = spec_dict

                # Prioritize TUUID keys
                if isinstance(key, int):
                    res[key] = spec
                elif key in name_to_tuuid:
                    res[name_to_tuuid[key]] = spec
                else:
                    raise ValueError(
                        f"Operator '{op_name}' has an output sharding strategy for '{key}', "
                        f"but it is not a valid output tensor recognized by local metadata. "
                        f"Available outputs: {out_args}"
                    )

        return res

    def get_dispatcher(self, op_name: str) -> ExecutionDispatcher:
        """Get the execution dispatcher strategy for an operator."""
        return self._dispatchers[op_name]

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

    def execute(self, op_name: str, step: int, *args, **kwargs) -> Any:
        """
        Execute an operator.

        Args:
            op_name: Operator name to execute
            step: Execution step
            *args, **kwargs: Assembled input data (can be data_type object or kwargs)

        Returns:
            Operator output
        """
        from fluxway.common.decorators import operator_context

        operator = self._operators.get(op_name)
        if operator is None:
            logger.warning(f"No operator registered for {op_name}")
            return None

        # Generate Logical Context
        dispatcher = self.get_dispatcher(op_name)
        logical_ctx = dispatcher.get_logical_context(self._rank, self._world_size, step)

        try:
            with operator_context(
                static_args=self._static_args,
                rank=self._rank,
                world_size=self._world_size,
                operator_ctx=logical_ctx,
            ):
                if self.is_source_operator(op_name):
                    return operator()
                return operator(*args, **kwargs)
        except Exception as e:
            logger.error(f"Execution failed for {op_name}: {e}")
            raise
