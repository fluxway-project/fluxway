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
from typing import Any, Callable, Dict, List, Optional, Set, Type, Tuple, TYPE_CHECKING
from fluxway.common.utils.logging import get_logger
import threading
import inspect as _inspect

from fluxway.common.types.configs import ArgumentType
from fluxway.common.types.sharding import ShardingSpec, PhysicalLayout
from fluxway.common.types import EdgeShardingInfo
from fluxway.common.mesh import DeviceMesh
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionDispatcher,
    create_dispatcher,
)

if TYPE_CHECKING:
    from fluxway.common.types import SPMDConfig, OperatorMeta

logger = get_logger(__name__)


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
    # Sharding by TUUID (Logical)
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

    def __init__(
        self,
        config: "SPMDConfig",
        rank: int,
        spmd_id: str = "",
        *,
        local_rank: Optional[int] = None,
        local_world_size: Optional[int] = None,
        node_rank: int = 0,
        num_nodes: int = 1,
    ):
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
        self._local_rank = local_rank
        self._local_world_size = local_world_size
        self._node_rank = node_rank
        self._num_nodes = num_nodes

        # Register operators
        # Register operators dynamically from config
        self._operators: Dict[str, Callable] = config.operators

        # Initialize per-operator dictionaries
        self._input_sharding: Dict[str, Dict[int, ShardingSpec]] = {}
        self._output_sharding: Dict[str, Dict[int, ShardingSpec]] = {}
        self._expected_control_arg_names: Dict[str, List[str]] = {}
        self._expected_tensor_names: Dict[str, List[str]] = {}
        self._predecessor_topology: Dict[str, Dict[str, ResolvedTopologyInfo]] = {}
        self._successor_topology: Dict[str, Dict[str, ResolvedTopologyInfo]] = {}
        self._expected_tensor_shards_cache: Dict[str, Dict[int, int]] = {}
        self._periodic_trigger_cache: Dict[str, bool] = {}
        self._operator_input_types: Dict[str, Dict[str, Type]] = {}
        self._dispatchers: Dict[str, ExecutionDispatcher] = {}
        # Global dispatch table: {tensor_tuuid: [op_name1, op_name2]}
        self._tuuid_to_op_inputs: Dict[int, List[str]] = {}
        self._tuuid_to_op_outputs: Dict[int, Tuple[str, str]] = {}
        self._logical_path_to_tuuid: Dict[Tuple[str, str], int] = {}

        for op_name in self._operators:
            self._input_sharding[op_name] = {}
            self._output_sharding[op_name] = {}
            self._expected_control_arg_names[op_name] = []
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

        # Central TUUID registry for resolving all routing logic (Source of Truth)
        # Only production-side routing (outputs/predecessors) are registered here.
        self._tuuid_registry: Dict[int, ShardingSpec] = {}

        # Context-isolated resolution caches
        self._resolved_input_cache: Dict[Tuple[str, int], PhysicalLayout] = {}
        self._resolved_producer_cache: Dict[int, ShardingSpec] = {}

        # Performance Caches
        # (op_name, tuuid, pre_spmd_id, rank, step_mod) -> List[int]
        self._producer_cache: Dict[Tuple[str, int, str, int, int], List[int]] = {}
        # (tuuid, pre_spmd_id) -> PhysicalLayout (projected)
        self._effective_layout_cache: Dict[Tuple[int, str], PhysicalLayout] = {}

        # Parse topology from OperatorMeta
        self._parse_operator_metadata()

        # Populate output sharding (TUUID-keyed) for all local operators
        for op_name in self._operators:
            self._output_sharding[op_name] = self.get_output_sharding(op_name)

        # Pre-compute expected tensor shards after routing strategies are ready
        self._precompute_expected_tensor_shards()

        # Pre-compute operator input types from function signatures
        self._precompute_operator_input_types()

    def _resolve_input_spec(self, op_name: str, tuuid: int) -> Optional[PhysicalLayout]:
        """
        Resolve a consumer-side input requirement for a specific operator.
        Returns a PhysicalLayout bound to the current mesh.
        """
        key = (op_name, tuuid)
        if key in self._resolved_input_cache:
            return self._resolved_input_cache[key]

        # Check local requirement map (ShardingSpec)
        routing = self._input_sharding.get(op_name, {}).get(tuuid)

        # Resolution logic
        if routing is None:
            return None

        if routing.is_referential:
            if routing.routing.tuuid is None:
                raise ValueError(
                    f"[Fluxway Topology Resolution Error] Incomplete referential routing for input '{tuuid}' in operator '{op_name}'. "
                    f"The logical path '{routing.routing.path}' failed to resolve to a concrete TUUID. "
                    "This usually happens when the path points to an operator or field that is not defined in the graph as a predecessor."
                )
            # Resolve the producer's logical routing first
            producer_routing = self._resolve_producer_spec(routing.routing.tuuid)
            if producer_routing:
                # bind to PhysicalLayout on current mesh
                layout = producer_routing.bind(self._mesh)
                self._resolved_input_cache[key] = layout
                return layout
            return None

        # Concrete requirement, bind it
        layout = routing.bind(self._mesh)
        self._resolved_input_cache[key] = layout
        return layout

    def _resolve_producer_spec(
        self,
        tuuid: int,
        provider_map: Optional[Dict[int, ShardingSpec]] = None,
        _trace: Optional[Set[int]] = None,
    ) -> Optional[ShardingSpec]:
        """Resolve a producer-side implementation (Logical Truth)."""
        if provider_map is None and tuuid in self._resolved_producer_cache:
            return self._resolved_producer_cache[tuuid]

        spec = provider_map.get(tuuid) if provider_map is not None else None
        if spec is None:
            spec = self._tuuid_registry.get(tuuid)

        if spec is None:
            return None

        if spec.is_referential:
            if spec.routing.tuuid is None:
                raise ValueError(f"Incomplete producer ref for TUUID {tuuid}")

            current_trace = _trace or set()
            if tuuid in current_trace:
                raise RuntimeError(f"Circular sharding reference at TUUID {tuuid}")
            current_trace.add(tuuid)

            resolved = self._resolve_producer_spec(
                spec.routing.tuuid, _trace=current_trace
            )
            if resolved and provider_map is None:
                self._resolved_producer_cache[tuuid] = resolved
            return resolved

        if provider_map is None:
            self._resolved_producer_cache[tuuid] = spec
        return spec

    def _register_sharding_specs(self, specs: Dict[int, ShardingSpec]) -> None:
        """
        Register a batch of ShardingSpec into the global registry.
        """
        for tuuid, routing in specs.items():
            existing = self._tuuid_registry.get(tuuid)
            if existing and not existing.is_referential:
                continue
            self._tuuid_registry[tuuid] = routing

    def _get_effective_layout(
        self, routing: Optional[ShardingSpec], info: ResolvedTopologyInfo
    ) -> Optional[PhysicalLayout]:
        """
        Extract the effective layout for routing/assembly.
        """
        if routing is None:
            return None

        mesh = info.mesh if info.mesh else self._mesh

        if not info.is_interleaved:
            return routing.bind(mesh)

        # INTERLEAVED SOURCE ABSTRACTION:
        # Override the data dimension to correctly represent the unit size.
        unit_size = info.interleave_unit_size
        other_dims_size = 1
        for s in mesh.shape[1:]:
            other_dims_size *= s

        if unit_size % other_dims_size != 0:
            raise ValueError(
                f"[Fluxway Interleave Error] Topology '{info.spmd_id}' specifies interleave_unit_size={unit_size}, "
                f"which is not divisible by the sub-mesh size {other_dims_size} (axes {mesh.axis_names[1:]}). "
                "Ensure that the unit size is a multiple of the size of the non-sharded dimensions."
            )

        first_axis_size = unit_size // other_dims_size
        new_shape = (first_axis_size,) + mesh.shape[1:]

        sub_mesh = DeviceMesh(device_shape=new_shape, axis_names=mesh.axis_names)
        return routing.bind(sub_mesh)

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
        Uses memoization to avoid re-calculating fixed sharding overlaps.
        """
        pre_info = self._predecessor_topology.get(op_name, {}).get(pre_spmd_id)
        if not pre_info:
            return []

        # Determine Cache Key (handles Interleaved source periodicity)
        step_mod = 0
        if pre_info.is_interleaved:
            total_units = pre_info.world_size // pre_info.interleave_unit_size
            step_mod = step % total_units

        cache_key = (op_name, tensor_tuuid, pre_spmd_id, current_rank, step_mod)
        if cache_key in self._producer_cache:
            return self._producer_cache[cache_key]

        # Resolve layouts
        input_layout = self._resolve_input_spec(op_name, tensor_tuuid)
        producer_routing = self._resolve_producer_spec(
            tensor_tuuid, provider_map=pre_info.tuuid_sharding
        )

        eff_key = (tensor_tuuid, pre_spmd_id)
        if eff_key in self._effective_layout_cache:
            effective_pre_layout = self._effective_layout_cache[eff_key]
        else:
            effective_pre_layout = self._get_effective_layout(
                producer_routing, pre_info
            )
            self._effective_layout_cache[eff_key] = effective_pre_layout

        # Calculate Overlap
        unit_base_rank = 0
        if pre_info.is_interleaved:
            unit_base_rank = step_mod * pre_info.interleave_unit_size

        producers: List[int] = []
        if effective_pre_layout:
            if input_layout is None:
                raise ValueError(
                    f"[Fluxway Assembly Error] Missing consumer-side PhysicalLayout for TUUID {tensor_tuuid} in operator '{op_name}'. "
                    "Strict sharding mode requires all data inputs to have a valid routing specification."
                )

            relative_producers = effective_pre_layout.resolve_producers(
                input_layout, current_rank
            )

            if relative_producers:
                unique_producers = {}
                for r in relative_producers:
                    identity = effective_pre_layout.get_logical_identity(r)
                    if identity not in unique_producers:
                        unique_producers[identity] = r + unit_base_rank
                producers = list(unique_producers.values())
        else:
            # Locate the field name for helpful error message
            tuuid_name = pre_info.tuuid_map.get(str(tensor_tuuid), str(tensor_tuuid))
            for path, tid in pre_info.tuuid_map.items():
                if tid == tensor_tuuid:
                    tuuid_name = path
                    break

            raise ValueError(
                f"[Fluxway Topology Error] Missing producer routing for TUUID {tensor_tuuid} (path: '{tuuid_name}') "
                f"coming from predecessor '{pre_spmd_id}' into operator '{op_name}'. "
                "Verify that the predecessor correctly defines its output sharding in @operator or the graph builder."
            )

        self._producer_cache[cache_key] = producers
        return producers

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
                        if info.mesh is None:
                            logger.warning(
                                f"[OperatorManager] Predecessor '{pred_id}' for '{op_name}' has no mesh defined. "
                                f"Falling back to local mesh {self._mesh}, which may cause incorrect shard count precomputation."
                            )
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
                    # Sync edge sharding to global registry
                    self._register_sharding_specs(sharding)

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
                    # Note: We do NOT register successor requirements into the global
                    # Production Truth registry. They are only context for routing.

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

            # Resolve sharding specs for all local input parameters
            local_input_specs = self._build_sharding_specs(
                meta.input_sharding, self._mesh
            )

            # Collect all available TUUIDs from predecessors to create a logical path -> TUUID lookup
            global_path_to_tuuid: Dict[str, int] = {}
            for pre_info in self._predecessor_topology.get(op_name, {}).values():
                # Bind Local Spec with TUUID by [Injected Argument Name -> TUUID] dict.
                # The local function arguments name is the pre_info.tuuid_map key, which was binded in _validate_and_link_edge
                global_path_to_tuuid.update(pre_info.tuuid_map)

            # Structured Expansion: Map top-level specs to physical TUUIDs using input_param_map
            # input_param_map is the 'source of truth' for the parameter tree structure
            for param_name, spec in local_input_specs.items():
                # Get list of logical field paths belonging to this parameter (e.g., 'ctx.val', 'ctx.id')
                nested_fields = meta.input_param_map.get(param_name, [param_name])
                for field_path in nested_fields:
                    if field_path in global_path_to_tuuid:
                        tuuid = global_path_to_tuuid[field_path]
                        self._input_sharding[op_name][tuuid] = spec
                        # Register in global registry ONLY if it doesn't collide with a producer.
                        # Actually, we SHOULD NOT register input requirements into the global registry
                        # at all, as they represent the consumer's perspective, not the truth of the TUUID.
                        # self._register_sharding_specs({tuuid: spec})

            # Strict Validation: Ensure every data tensor required by the operator has a spec
            for field_path in self._expected_tensor_names[op_name]:
                if field_path in global_path_to_tuuid:
                    tuuid = global_path_to_tuuid[field_path]
                    if tuuid not in self._input_sharding[op_name]:
                        raise ValueError(
                            f"[Fluxway Topology Error] Field '{field_path}' in operator '{op_name}' "
                            f"(TUUID {tuuid}) has no sharding specification. "
                            f"Check @operator definition for parameter containing this field."
                        )

            # Resolve local output TUUIDs and register them
            self._output_sharding[op_name] = self.get_output_sharding(op_name)
            self._register_sharding_specs(self._output_sharding[op_name])

            self._expected_control_arg_names[op_name] = list(
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
                        and p_name not in self._expected_control_arg_names[op_name]
                    ):
                        raise ValueError(
                            f"[Fluxway Init Error] Operator '{op_name}' requires parameter '{p_name}', "
                            "but it's not in predecessor outputs, not in control states, "
                            "nor in static arguments.\n"
                            f"Available predecessor outputs: {sorted(available_paths)}\n"
                            f"Registered control args: {self._expected_control_arg_names[op_name]}\n"
                            f"Static arguments: {list(self._static_args.keys())}"
                        )

        # Phase 1: Logical Path & TUUID Discovery
        # Build an exhaustive index of (OpName, FieldName) -> TUUID.
        for op_name in self._operators:
            meta = self._get_operator_metadata(op_name)

            # 1.1 Local Outputs
            out_uuids = self.get_output_tuuids(op_name)
            out_names = meta.get_arguments(is_output=True, arg_type=ArgumentType.DATA)
            if out_names and out_uuids:
                for name, tuuid in zip(out_names, out_uuids, strict=True):  # type: ignore
                    self._logical_path_to_tuuid[(op_name, name)] = tuuid
                    self._tuuid_to_op_outputs[tuuid] = (op_name, name)

            # 1.2 Local Input Dispatch Mapping
            input_map = self._input_sharding.get(op_name, {})
            for u in input_map:
                if u not in self._tuuid_to_op_inputs:
                    self._tuuid_to_op_inputs[u] = []
                if op_name not in self._tuuid_to_op_inputs[u]:
                    self._tuuid_to_op_inputs[u].append(op_name)

            # 1.3 Topology Knowledge (Predecessors & Successors)
            for pre_info in self._predecessor_topology.get(op_name, {}).values():
                for name, u in pre_info.tuuid_map.items():
                    self._logical_path_to_tuuid[(pre_info.spmd_id, name)] = u
                    self._tuuid_to_op_outputs[u] = (pre_info.spmd_id, name)

            for succ_info in self._successor_topology.get(op_name, {}).values():
                for name, u in succ_info.tuuid_map.items():
                    self._logical_path_to_tuuid[(succ_info.spmd_id, name)] = u

        # Phase 2: Strategy Truth Registration
        for op_name in self._operators:
            self._register_all_algebraic_specs(self._input_sharding.get(op_name, {}))
            self._register_all_algebraic_specs(self._output_sharding.get(op_name, {}))
            for pre_info in self._predecessor_topology.get(op_name, {}).values():
                self._register_all_algebraic_specs(pre_info.tuuid_sharding)
            for succ_info in self._successor_topology.get(op_name, {}).values():
                self._register_all_algebraic_specs(succ_info.tuuid_sharding)

        # Phase 3: Path Resolution (Binding)
        for op_name in self._operators:

            def res_fn(path: str, context_op=op_name) -> Optional[int]:
                return self.resolve_path(context_op, path)

            self._resolve_referential_uuids_in_map(
                self._input_sharding.get(op_name, {}), res_fn
            )
            self._resolve_referential_uuids_in_map(
                self._output_sharding.get(op_name, {}), res_fn
            )
            for pre_info in self._predecessor_topology.get(op_name, {}).values():
                self._resolve_referential_uuids_in_map(pre_info.tuuid_sharding, res_fn)
            for succ_info in self._successor_topology.get(op_name, {}).values():
                self._resolve_referential_uuids_in_map(succ_info.tuuid_sharding, res_fn)

            meta = self._get_operator_metadata(op_name)
            if meta:
                meta.resolve_async_strategy(res_fn)

        # Phase 4: Strategy Inheritance
        for op_name in self._operators:
            self._resolve_referential_strategies_in_map(
                self._input_sharding.get(op_name, {})
            )
            self._resolve_referential_strategies_in_map(
                self._output_sharding.get(op_name, {})
            )
            for pre_info in self._predecessor_topology.get(op_name, {}).values():
                self._resolve_referential_strategies_in_map(pre_info.tuuid_sharding)
            for succ_info in self._successor_topology.get(op_name, {}).values():
                self._resolve_referential_strategies_in_map(succ_info.tuuid_sharding)

            meta = self._get_operator_metadata(op_name)
            logger.debug(
                f"Resolved operator {op_name}: "
                f"expected_tensors={self.get_expected_tensor_names(op_name)}"
            )

    def _resolve_referential_uuids_in_map(
        self,
        sharding_map: Dict[int, "ShardingSpec"],
        resolver_fn: Callable[[str], Optional[int]],
    ) -> None:
        """Helper to resolve referential paths to TUUIDs in a specific map."""
        for spec in sharding_map.values():
            if spec.is_referential:
                spec.routing.resolve(resolver_fn)

    def _register_all_algebraic_specs(
        self, sharding_map: Dict[int, "ShardingSpec"]
    ) -> None:
        """Helper to populate the global TUUID registry with concrete sharding information."""
        from fluxway.common.types.sharding import AlgebraicRouting

        for tuuid, spec in sharding_map.items():
            if isinstance(spec.routing, AlgebraicRouting):
                self._tuuid_registry[tuuid] = spec

    def _resolve_referential_strategies_in_map(
        self, sharding_map: Dict[int, "ShardingSpec"]
    ) -> None:
        """Helper to resolve referential sharding in a specific map."""
        from fluxway.common.types.sharding import ShardingSpec

        for _, spec in sharding_map.items():
            if isinstance(spec, ShardingSpec) and spec.is_referential:
                mirror_tuuid = spec.routing.tuuid
                if mirror_tuuid is None:
                    continue

                # Follow the chain to the ultimate producer truth
                truth_spec = self._resolve_producer_spec(mirror_tuuid)
                if truth_spec:
                    if truth_spec.is_referential:
                        continue

                    # Inject the ultimate algebraic truth into our referential spec
                    spec.routing.resolve_sharding(truth_spec.routing)

    def resolve_path(self, op_name: str, path: str) -> int:
        """
        Unified resolver for logical paths to TUUIDs.

        Logic:
        1. If path is 'op.field', looks in Phase 1 index.
        2. If path is just 'field', looks in context op's input/output map.
        """
        if (op_name, path) in self._logical_path_to_tuuid:
            return self._logical_path_to_tuuid[(op_name, path)]

        if "." in path:
            target_op, field = path.split(".", 1)
            if (target_op, field) in self._logical_path_to_tuuid:
                return self._logical_path_to_tuuid[(target_op, field)]
        else:
            target_op, field = op_name, path

        # Fallback to topology search if index missed
        pre_topo = self.get_predecessor_topology(target_op)
        for pre_info in pre_topo.values():
            if field in pre_info.tuuid_map:
                return pre_info.tuuid_map[field]

        # 2. Search in successors (find field produced as output by target_op)
        succ_topo = self.get_successor_topology(target_op)
        for succ_info in succ_topo.values():
            if field in succ_info.tuuid_map:
                return succ_info.tuuid_map[field]

        # Aggregate available fields for helpful error message
        available_pre = set()
        for p_info in pre_topo.values():
            available_pre.update(p_info.tuuid_map.keys())

        available_succ = set()
        for s_info in succ_topo.values():
            available_succ.update(s_info.tuuid_map.keys())

        raise ValueError(
            f"[OperatorManager] Cannot resolve logical path '{path}' to a TUUID. "
            f"Target operator '{target_op}' provides as inputs: {sorted(available_pre)}, "
            f"produced as outputs: {sorted(available_succ)}"
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
                    # Eager mode routing logic would need a mesh to compute counts.
                    # We might need to bind or assume replication.
                    self._expected_tensor_shards_cache[op_name][u] = 1

            # Sync to global registry
            self._register_sharding_specs(tuuid_sharding)

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
                parsed[key] = ShardingConfig.model_validate(raw_spec).to_routing()
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
                sig = _inspect.signature(operator)
                is_source = self.is_source_operator(op_name)

                params_dict = {}
                for p_name, param in sig.parameters.items():
                    if p_name in ("self", "cls"):
                        continue
                    param_type = param.annotation
                    if param_type == _inspect.Parameter.empty:
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

    def get_expected_control_arg_names(self, op_name: str) -> List[str]:
        """Get expected state variable names."""
        return list(self._expected_control_arg_names.get(op_name, []))

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

    def get_successor_topology(self, op_name: str) -> Dict[str, ResolvedTopologyInfo]:
        """Get successor topology info for an operator."""
        return dict(self._successor_topology.get(op_name, {}))

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
        name_to_tuuid = dict(zip(out_args, meta.output_uuids, strict=True))  # type: ignore

        res = {}
        if meta.output_sharding:
            for key, spec_dict in meta.output_sharding.items():
                if isinstance(spec_dict, dict):
                    spec = ShardingConfig.model_validate(spec_dict).to_spec()
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
                local_rank=self._local_rank,
                local_world_size=self._local_world_size,
                node_rank=self._node_rank,
                num_nodes=self._num_nodes,
            ):
                if self.is_source_operator(op_name):
                    return operator()
                return operator(*args, **kwargs)
        except Exception as e:
            logger.error(f"Execution failed for {op_name}: {e}")
            raise
