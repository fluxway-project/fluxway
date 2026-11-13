"""
Operator Manager for SPMD Worker.

Responsibilities:
1. Store and manage operators (e.g., train_step, rollout, reward)
2. Parse compiled topology at initialization
3. Compute expected tensor shards from the compiled topology
4. Compute output routing from the compiled topology
5. Execute operators
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, cast
from collections.abc import Awaitable, Callable, Generator
from fluxway.common.utils.logging import get_logger
import threading
import inspect as _inspect

from fluxway.common.operator_policy import ArgumentType
from fluxway.common.compiled_graph import (
    CompiledDependency,
    CompiledLogicalTensor,
    CompiledOperator,
    CompiledTensorTransfer,
    SPMDGraphView,
)
from fluxway.common.types.sharding import (
    PhysicalLayout,
    ReferentialRouting,
    ShardingSpec,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.profiler.user_api import function_scope
from fluxway.common.operator_instances import OperatorInstanceContainer
from fluxway.common.ingestion import SourceWorkItem
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionDispatcher,
)

if TYPE_CHECKING:
    from fluxway.common.types import SPMDConfig

logger = get_logger(__name__)


def _validated_source_work_items(
    op_name: str, step: int, values: Any | None
) -> tuple[SourceWorkItem, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(
            f"source-control operator {op_name!r} requires canonical source work"
        )
    if any(not isinstance(value, SourceWorkItem) for value in values):
        raise TypeError(
            f"source-control operator {op_name!r} requires SourceWorkItem values"
        )
    items = cast(tuple[SourceWorkItem, ...], values)
    if any(item.key.step != step for item in items):
        raise ValueError(
            f"source-control operator {op_name!r} received work for another step"
        )
    assignment = items[0].assignment
    signature = (
        assignment.source_identity,
        assignment.view_identity,
        assignment.schema_digest,
        assignment.lineage_digest,
        assignment.global_batch_size,
    )
    if any(
        (
            item.assignment.source_identity,
            item.assignment.view_identity,
            item.assignment.schema_digest,
            item.assignment.lineage_digest,
            item.assignment.global_batch_size,
        )
        != signature
        for item in items
    ):
        raise ValueError(
            f"source-control operator {op_name!r} received incompatible work"
        )
    intervals = tuple(
        (item.key, item.logical_start, item.logical_end) for item in items
    )
    if len(set(intervals)) != len(intervals):
        raise ValueError(
            f"source-control operator {op_name!r} received duplicate work intervals"
        )
    return items


# =============================================================================
# Runtime topology
# =============================================================================


@dataclass(frozen=True, slots=True)
class ResolvedDependencyRoute:
    dependency: CompiledDependency
    transfer: CompiledTensorTransfer
    logical_tensor: CompiledLogicalTensor
    peer_is_source: bool

    def __post_init__(self) -> None:
        if self.transfer.dependency_code != self.dependency.code:
            raise ValueError(
                f"Transfer code {self.transfer.transfer_code} belongs to dependency "
                f"code {self.transfer.dependency_code}, not {self.dependency.code}."
            )
        if not any(item is self.transfer for item in self.dependency.transfers):
            raise ValueError(
                f"Transfer code {self.transfer.transfer_code} is not owned by "
                f"dependency code {self.dependency.code}."
            )
        if self.logical_tensor.logical_tuuid != self.transfer.logical_tuuid:
            raise ValueError(
                f"Transfer code {self.transfer.transfer_code} logical TUUID "
                f"{self.transfer.logical_tuuid} does not match canonical logical "
                f"tensor {self.logical_tensor.logical_tuuid}."
            )

    @property
    def spmd_id(self) -> str:
        key = (
            self.dependency.spec.source
            if self.peer_is_source
            else self.dependency.spec.target
        )
        return key.spmd_id

    @property
    def transfer_code(self) -> int:
        return self.transfer.transfer_code

    @property
    def logical_tuuid(self) -> int:
        return self.transfer.logical_tuuid

    @property
    def source_sharding(self) -> ShardingSpec:
        return self.transfer.source_sharding

    @property
    def target_sharding(self) -> ShardingSpec:
        return self.transfer.target_sharding

    @property
    def effective_source_sharding(self) -> ShardingSpec:
        return self.logical_tensor.require_effective_sharding()

    @property
    def effective_target_sharding(self) -> ShardingSpec:
        return self.dependency.target_operator.effective_input_sharding[
            self.target_name
        ]

    @property
    def source_name(self) -> str:
        return self.transfer.spec.source.name

    @property
    def target_name(self) -> str:
        return self.transfer.spec.target.name

    @property
    def source_op(self) -> str:
        return self.dependency.spec.source.op_id

    @property
    def target_op(self) -> str:
        return self.dependency.spec.target.op_id

    @property
    def world_size(self) -> int:
        if self.peer_is_source:
            return self.dependency.source_world_size
        return self.dependency.target_world_size

    @property
    def mesh(self) -> DeviceMesh:
        if self.peer_is_source:
            return self.dependency.source_mesh
        return self.dependency.target_mesh

    @property
    def condition(self) -> Callable[[Any], bool | Awaitable[bool]] | None:
        return self.dependency.spec.condition

    @property
    def local_only(self) -> bool:
        return self.dependency.spec.local_only

    @property
    def symmetric_to(self) -> tuple[str, str, str, str] | None:
        return self.dependency.spec.symmetric_to

    @property
    def is_self_loop(self) -> bool:
        return (
            self.dependency.spec.source.spmd_id == self.dependency.spec.target.spmd_id
        )


class OperatorManager:
    """
    Manages operator execution, sharding computation, and routing.

    Key design:
    - All topology info is parsed from the compiled program at initialization
    - No runtime predecessor/successor registration needed
    - Handler management is NOT here (handled by Worker)
    """

    def __init__(
        self,
        config: "SPMDConfig",
        rank: int,
        spmd_id: str = "",
        *,
        static_args: dict[str, Any],
        instance_container: OperatorInstanceContainer,
        local_rank: int | None = None,
        local_world_size: int | None = None,
        node_rank: int = 0,
        num_nodes: int = 1,
        control_plane_handle: Any | None = None,
        worker_process_runtime: Any | None = None,
    ):
        """
        Initialize OperatorManager.

        Args:
            config: SPMD configuration containing user functions and mesh
            rank: This worker's rank
            spmd_id: SPMD identifier (for self-loop detection)
        """
        self._config = config
        self._plan: SPMDGraphView = config.require_graph_view()
        self._ray_component_limits = self._plan.resource_profile.ray_component_limits
        if spmd_id and spmd_id != config.spmd_id:
            raise ValueError(
                f"OperatorManager SPMD ID {spmd_id!r} does not match config "
                f"{config.spmd_id!r}."
            )
        self._mesh = self._plan.mesh
        self._rank = rank
        self._world_size = self._plan.world_size
        self._spmd_id = self._plan.spmd_id
        self._local_rank = local_rank
        self._local_world_size = local_world_size
        self._node_rank = node_rank
        self._num_nodes = num_nodes
        self._control_plane_handle = control_plane_handle
        self._worker_process_runtime = worker_process_runtime
        if not isinstance(instance_container, OperatorInstanceContainer):
            raise TypeError("OperatorManager requires an OperatorInstanceContainer")
        self._instance_container = instance_container

        self._operators: dict[str, Any] = {}
        self._compiled_operators: dict[str, CompiledOperator] = {}
        self._operator_is_coroutine_function: dict[str, bool] = {}
        for compiled_operator in self._plan.operators:
            op_name = compiled_operator.key.op_id
            spec = compiled_operator.spec
            if config.operator_specs.get(op_name) is not spec:
                raise ValueError(
                    f"Operator {self._spmd_id}.{op_name} declaration does not "
                    "match its compiled graph."
                )
            owner_cls = spec.owner
            if owner_cls is None:
                raise ValueError(
                    f"operator {self._spmd_id}.{op_name} has no owner class"
                )
            op_func = getattr(owner_cls, spec.method_name)
            if not hasattr(op_func, "_fluxway_operator_spec"):
                from fluxway.common.decorators import _OperatorWrapper

                op_func = _OperatorWrapper(op_func, spec).__get__(None, owner_cls)
            self._operators[op_name] = op_func
            self._compiled_operators[op_name] = compiled_operator
            self._operator_is_coroutine_function[op_name] = spec.is_coroutine_function

        # Initialize per-operator dictionaries
        self._input_sharding: dict[str, dict[int, ShardingSpec]] = {}
        self._expected_control_arg_names: dict[str, list[str]] = {}
        self._expected_tensor_names: dict[str, list[str]] = {}
        self._predecessor_topology: dict[str, dict[int, ResolvedDependencyRoute]] = {}
        self._successor_topology: dict[str, dict[int, ResolvedDependencyRoute]] = {}
        self._expected_transfer_shards_cache: dict[str, dict[int, int]] = {}
        self._periodic_trigger_cache: dict[str, bool] = {}
        self._operator_input_types: dict[str, dict[str, type]] = {}
        self._dispatchers: dict[str, ExecutionDispatcher] = {}
        self._input_transfer_codes: dict[str, dict[str, int]] = {}
        self._incoming_routes_by_code: dict[int, ResolvedDependencyRoute] = {}
        self._transfer_target_by_code: dict[int, str] = {}
        self._outgoing_tensor_dependencies_by_source_op: dict[
            str, tuple[CompiledDependency, ...]
        ] = {}
        self._ordering_dependencies_by_source_op: dict[
            str, tuple[CompiledDependency, ...]
        ] = {}
        self._ordering_target_by_code: dict[int, str] = {}
        self._expected_ordering_counts: dict[str, dict[int, int]] = {}
        self._dependency_decision_target_by_code: dict[int, str] = {}
        self._dependency_decision_expected_count_by_code: dict[int, int] = {}
        self._expected_dependency_decisions: dict[str, dict[int, int]] = {}
        self._dependency_transfer_codes: dict[int, tuple[int, ...]] = {}
        self._transfer_dependency_codes: dict[str, dict[int, int]] = {}

        for op_name in self._operators:
            self._input_sharding[op_name] = {}
            self._expected_control_arg_names[op_name] = []
            self._expected_tensor_names[op_name] = []
            self._predecessor_topology[op_name] = {}
            self._successor_topology[op_name] = {}
            self._input_transfer_codes[op_name] = {}
            self._operator_input_types[op_name] = {}
            self._transfer_dependency_codes[op_name] = {}

        self._static_args = static_args

        # Thread lock
        self._lock = threading.Lock()

        self._target_layout_cache: dict[int, PhysicalLayout] = {}

        # Performance Caches
        self._producer_cache: dict[tuple[int, int, int], list[int]] = {}
        self._source_layout_cache: dict[int, PhysicalLayout] = {}
        self._source_dispatchers: dict[Any, ExecutionDispatcher] = {}

        self._compile_dependency_decisions()
        self._compile_ordering_dependencies()

        # Parse the compiled topology
        self._parse_operator_metadata()

        # Pre-compute expected tensor shards after routing strategies are ready
        self._precompute_expected_transfer_shards()

        # Pre-compute operator input types from function signatures
        self._precompute_operator_input_types()

    def _get_target_layout(
        self,
        op_name: str,
        transfer_code: int,
    ) -> PhysicalLayout:
        cached = self._target_layout_cache.get(transfer_code)
        if cached is not None:
            return cached

        route = self.get_predecessor_route(op_name, transfer_code)
        layout = route.effective_target_sharding.bind(self._mesh)
        self._target_layout_cache[transfer_code] = layout
        return layout

    def resolve_tensor_producers(
        self,
        op_name: str,
        transfer_code: int,
        current_rank: int,
        step: int = 0,
    ) -> list[int]:
        route = self.get_predecessor_route(op_name, transfer_code)

        source_key = route.dependency.source_operator.key
        if source_key not in self._source_dispatchers:
            self._source_dispatchers[source_key] = ExecutionDispatcher(
                route.dependency.source_operator.execution_layout
            )
        source_dispatcher = self._source_dispatchers[source_key]
        cycle = route.dependency.source_operator.execution_layout.cycle
        cycle_index = step % len(cycle)
        cache_key = (transfer_code, current_rank, cycle_index)
        if cache_key in self._producer_cache:
            return self._producer_cache[cache_key]
        unit_relation = route.dependency.unit_relation(transfer_code)
        if unit_relation is not None:
            producers = [
                cycle[cycle_index][rank]
                for rank in unit_relation.producers_by_target[current_rank]
            ]
        else:
            input_layout = self._get_target_layout(op_name, transfer_code)
            effective_pre_layout = self.get_transfer_source_layout(
                op_name, transfer_code
            )
            producers = source_dispatcher.resolve_active_producers(
                effective_pre_layout,
                input_layout,
                current_rank,
                cycle_index,
                route.world_size,
            )

        self._producer_cache[cache_key] = producers
        return producers

    # =========================================================================
    # Metadata Parsing (at initialization)
    # =========================================================================

    def _parse_operator_metadata(self) -> None:
        outgoing: dict[str, list[CompiledDependency]] = {}
        for dependency in self._plan.incoming:
            if not dependency.transfers:
                continue
            op_name = dependency.spec.target.op_id
            if op_name not in self._operators:
                raise ValueError(
                    f"Dependency {dependency.spec.key!r} targets inactive operator."
                )
            for transfer in dependency.transfers:
                route = ResolvedDependencyRoute(
                    dependency=dependency,
                    transfer=transfer,
                    logical_tensor=self._plan.logical_tensor(transfer.logical_tuuid),
                    peer_is_source=True,
                )
                code = transfer.transfer_code
                if code in self._incoming_routes_by_code:
                    raise ValueError(f"Duplicate incoming transfer code {code}.")
                target_name = transfer.spec.target.name
                target_path = f"{op_name}.{target_name}"
                if target_path in self._input_transfer_codes[op_name]:
                    existing = self._input_transfer_codes[op_name][target_path]
                    raise ValueError(
                        f"Operator {op_name!r} input {target_path!r} is bound to "
                        f"transfer codes {existing} and {code}."
                    )
                if code in self._transfer_target_by_code:
                    raise ValueError(f"Duplicate transfer target code {code}.")
                self._predecessor_topology[op_name][code] = route
                self._incoming_routes_by_code[code] = route
                self._transfer_target_by_code[code] = op_name
                self._input_transfer_codes[op_name][target_path] = code
                self._input_sharding[op_name][code] = transfer.target_sharding

        for dependency in self._plan.outgoing:
            if not dependency.transfers:
                continue
            op_name = dependency.spec.source.op_id
            if op_name not in self._operators:
                raise ValueError(
                    f"Dependency {dependency.spec.key!r} sources inactive operator."
                )
            outgoing.setdefault(op_name, []).append(dependency)
            for transfer in dependency.transfers:
                route = ResolvedDependencyRoute(
                    dependency=dependency,
                    transfer=transfer,
                    logical_tensor=self._plan.logical_tensor(transfer.logical_tuuid),
                    peer_is_source=False,
                )
                code = transfer.transfer_code
                if code in self._successor_topology[op_name]:
                    raise ValueError(f"Duplicate outgoing transfer code {code}.")
                self._successor_topology[op_name][code] = route
        self._outgoing_tensor_dependencies_by_source_op.update(
            {op_name: tuple(items) for op_name, items in outgoing.items()}
        )

        for op_name in self._operators:
            compiled = self._get_compiled_operator(op_name)
            self._dispatchers[op_name] = ExecutionDispatcher(compiled.execution_layout)
            self._expected_control_arg_names[op_name] = compiled.get_arguments(
                is_output=False,
                arg_type=ArgumentType.CONTROL,
            )
            expected_names = compiled.get_arguments(
                is_output=False,
                arg_type=ArgumentType.DATA,
            )
            if len(set(expected_names)) != len(expected_names):
                raise ValueError(
                    f"Operator {op_name!r} has duplicate data input names."
                )
            self._expected_tensor_names[op_name] = expected_names
            expected_paths = {f"{op_name}.{name}" for name in expected_names}
            actual_paths = set(self._input_transfer_codes[op_name])
            if actual_paths != expected_paths:
                raise ValueError(
                    f"Operator {op_name!r} tensor input bindings differ from its "
                    f"declaration: expected={sorted(expected_paths)}, "
                    f"actual={sorted(actual_paths)}."
                )
        for op_name in self._operators:
            for route in (
                *self.get_predecessor_routes(op_name),
                *self.get_successor_routes(op_name),
            ):
                for spec in (route.source_sharding, route.target_sharding):
                    if (
                        isinstance(spec.routing, ReferentialRouting)
                        and spec.routing.tuuid is None
                    ):
                        raise ValueError(
                            f"operator {op_name!r} contains unresolved referential "
                            f"routing path {spec.routing.path!r}"
                        )
            for transfer_code in self._predecessor_topology[op_name]:
                self._get_target_layout(op_name, transfer_code)
            logger.debug(
                f"Resolved operator {op_name}: "
                f"expected_tensors={self.get_expected_tensor_names(op_name)}"
            )

    @staticmethod
    def _require_sharding_specs(raw_specs: Any) -> dict[Any, ShardingSpec]:
        specs = dict(raw_specs)
        invalid = {
            key: type(value).__name__
            for key, value in specs.items()
            if not isinstance(value, ShardingSpec)
        }
        if invalid:
            raise TypeError(
                f"compiled sharding contains non-ShardingSpec values: {invalid}"
            )
        return specs

    @staticmethod
    def _compiled_output_sharding(
        operator: CompiledOperator,
    ) -> dict[int, ShardingSpec]:
        output_names = operator.get_arguments(
            is_output=True,
            arg_type=ArgumentType.DATA,
        )
        if len(output_names) != len(operator.output_uuids):
            raise ValueError(
                f"Operator {operator.key!r} output names and logical TUUIDs differ "
                "in length."
            )
        name_to_tuuid = dict(zip(output_names, operator.output_uuids, strict=True))
        result: dict[int, ShardingSpec] = {}
        for name, spec in operator.output_sharding.items():
            try:
                logical_tuuid = name_to_tuuid[name]
            except KeyError as exc:
                raise ValueError(
                    f"Operator {operator.key!r} has output sharding for unknown "
                    f"tensor {name!r}."
                ) from exc
            result[logical_tuuid] = spec
        return result

    def _get_operator_spec(self, op_name: str) -> OperatorSpec:
        return self._get_compiled_operator(op_name).spec

    def _get_compiled_operator(self, op_name: str) -> CompiledOperator:
        try:
            return self._compiled_operators[op_name]
        except KeyError as exc:
            raise ValueError(f"operator {op_name!r} is not in SPMDGraphView") from exc

    def _precompute_expected_transfer_shards(self) -> None:
        for op_name in self._operators:
            expected: dict[int, int] = {}
            is_periodic = self.is_periodic_trigger(op_name)
            for route in self.get_predecessor_routes(op_name):
                code = route.transfer_code
                source_spec = route.effective_source_sharding
                if is_periodic:
                    from fluxway.common.operator_policy import PeriodicBarrierConfig

                    meta = self._get_operator_spec(op_name)
                    if not isinstance(meta.barrier_config, PeriodicBarrierConfig):
                        raise TypeError(
                            f"periodic operator {op_name!r} requires "
                            "PeriodicBarrierConfig"
                        )
                    expected[code] = meta.barrier_config.n
                elif (
                    route.transfer.uses_receipt_routing
                    or source_spec.is_content_addressable
                ):
                    if route.world_size <= 0:
                        raise ValueError(
                            f"Transfer code {code} has invalid source world size "
                            f"{route.world_size}."
                        )
                    expected[code] = route.dependency.source_participating_count
                else:
                    expected[code] = len(
                        self.resolve_tensor_producers(
                            op_name,
                            code,
                            self._rank,
                            step=0,
                        )
                    )

            self._expected_transfer_shards_cache[op_name] = expected
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

        For operators that accept **kwargs, the individual tensor parameter names
        are not present in the function signature; they are declared via the
        operator's ``input_sharding`` and recorded in
        ``meta.input_param_map`` by ``_resolve_input_sharding``. We expand those
        synthetic params here so ``_assemble_input`` can collect tuuids for
        each one and forward them as kwargs at runtime.
        """
        from typing import Any as _Any

        for op_name, operator in self._operators.items():
            try:
                sig = _inspect.signature(operator)
                is_source = self.is_source_operator(op_name)

                params_dict = {}
                has_var_keyword = False
                for p_name, param in sig.parameters.items():
                    if p_name in ("self", "cls"):
                        continue
                    if param.kind == _inspect.Parameter.VAR_KEYWORD:
                        has_var_keyword = True
                        continue
                    if param.kind == _inspect.Parameter.VAR_POSITIONAL:
                        continue
                    param_type = param.annotation
                    if param_type == _inspect.Parameter.empty:
                        logger.warning(
                            f"Operator {op_name} parameter '{p_name}' has no type annotation"
                        )
                        continue
                    params_dict[p_name] = param_type

                # Surface **kwargs-injected tensor params declared via
                # input_sharding so _assemble_input doesn't see an empty
                # input_types and silently drop everything.
                if has_var_keyword:
                    meta = self._get_operator_spec(op_name)
                    for k in meta.input_param_map:
                        if k in params_dict:
                            continue
                        if k in self._static_args:
                            continue
                        params_dict[k] = _Any

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

    def get_predecessor_ids(self, op_name: str) -> list[str]:
        return sorted({route.spmd_id for route in self.get_predecessor_routes(op_name)})

    def get_successor_ids(self, op_name: str) -> list[str]:
        return sorted({route.spmd_id for route in self.get_successor_routes(op_name)})

    def get_predecessor_routes(
        self, op_name: str
    ) -> tuple[ResolvedDependencyRoute, ...]:
        self._get_compiled_operator(op_name)
        return tuple(self._predecessor_topology[op_name].values())

    def get_successor_routes(self, op_name: str) -> tuple[ResolvedDependencyRoute, ...]:
        self._get_compiled_operator(op_name)
        return tuple(self._successor_topology[op_name].values())

    def get_predecessor_route(
        self,
        op_name: str,
        transfer_code: int,
    ) -> ResolvedDependencyRoute:
        self._get_compiled_operator(op_name)
        try:
            return self._predecessor_topology[op_name][transfer_code]
        except KeyError as exc:
            raise ValueError(
                f"Operator {op_name!r} has no incoming transfer code {transfer_code}."
            ) from exc

    def get_successor_route(
        self,
        op_name: str,
        transfer_code: int,
    ) -> ResolvedDependencyRoute:
        self._get_compiled_operator(op_name)
        try:
            return self._successor_topology[op_name][transfer_code]
        except KeyError as exc:
            raise ValueError(
                f"Operator {op_name!r} has no outgoing transfer code {transfer_code}."
            ) from exc

    def get_outgoing_tensor_dependencies(
        self,
        op_name: str,
    ) -> tuple[CompiledDependency, ...]:
        self._get_compiled_operator(op_name)
        return self._outgoing_tensor_dependencies_by_source_op.get(op_name, ())

    def get_outgoing_ordering_dependencies(
        self,
        op_name: str,
    ) -> tuple[CompiledDependency, ...]:
        self._get_compiled_operator(op_name)
        return self._ordering_dependencies_by_source_op.get(op_name, ())

    def get_ordering_target_op(self, dependency_code: int) -> str:
        try:
            return self._ordering_target_by_code[dependency_code]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self._spmd_id!r} has no incoming ordering dependency "
                f"code {dependency_code}."
            ) from exc

    def get_expected_ordering_counts(self, op_name: str) -> dict[int, int]:
        return self._expected_ordering_counts.get(op_name, {})

    def get_dependency_decision_target_op(self, dependency_code: int) -> str:
        try:
            return self._dependency_decision_target_by_code[dependency_code]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self._spmd_id!r} has no incoming dependency "
                f"code {dependency_code}."
            ) from exc

    def get_dependency_decision_expected_count(self, dependency_code: int) -> int:
        try:
            return self._dependency_decision_expected_count_by_code[dependency_code]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self._spmd_id!r} has no incoming dependency "
                f"code {dependency_code}."
            ) from exc

    def get_expected_dependency_decisions(self, op_name: str) -> dict[int, int]:
        self._get_compiled_operator(op_name)
        return self._expected_dependency_decisions.get(op_name, {})

    def get_transfer_dependency_codes(self, op_name: str) -> dict[int, int]:
        self._get_compiled_operator(op_name)
        return self._transfer_dependency_codes.get(op_name, {})

    def get_dependency_transfer_codes(self, dependency_code: int) -> tuple[int, ...]:
        try:
            return self._dependency_transfer_codes[dependency_code]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self._spmd_id!r} has no incoming tensor dependency "
                f"code {dependency_code}."
            ) from exc

    def _compile_dependency_decisions(self) -> None:
        for dependency in self._plan.incoming:
            op_name = dependency.spec.target.op_id
            if op_name not in self._operators:
                raise ValueError(
                    f"Dependency {dependency.spec.key!r} targets inactive operator."
                )
            expected_count = dependency.source_participating_count
            dependency_code = dependency.code
            if dependency_code in self._dependency_decision_target_by_code:
                raise ValueError(
                    f"Duplicate incoming dependency code {dependency_code}."
                )
            transfer_codes = tuple(
                transfer.transfer_code for transfer in dependency.transfers
            )
            self._dependency_decision_target_by_code[dependency_code] = op_name
            self._dependency_decision_expected_count_by_code[dependency_code] = (
                expected_count
            )
            self._dependency_transfer_codes[dependency_code] = transfer_codes
            for transfer_code in transfer_codes:
                if transfer_code in self._transfer_dependency_codes[op_name]:
                    raise ValueError(
                        f"Duplicate incoming transfer code {transfer_code}."
                    )
                self._transfer_dependency_codes[op_name][transfer_code] = (
                    dependency_code
                )
            if dependency.spec.condition is not None:
                self._expected_dependency_decisions.setdefault(op_name, {})[
                    dependency_code
                ] = expected_count

    def _compile_ordering_dependencies(self) -> None:
        outgoing: dict[str, list[CompiledDependency]] = {}
        for dependency in self._plan.outgoing:
            if dependency.transfers:
                continue
            op_name = dependency.spec.source.op_id
            if op_name not in self._operators:
                raise ValueError(
                    f"Ordering dependency {dependency.spec.key!r} sources inactive "
                    "operator."
                )
            outgoing.setdefault(op_name, []).append(dependency)
        for dependency in self._plan.incoming:
            if dependency.transfers:
                continue
            if dependency.spec.condition is not None:
                continue
            op_name = dependency.spec.target.op_id
            if op_name not in self._operators:
                raise ValueError(
                    f"Ordering dependency {dependency.spec.key!r} targets inactive "
                    "operator."
                )
            expected_count = dependency.source_participating_count
            self._ordering_target_by_code[dependency.code] = op_name
            self._expected_ordering_counts.setdefault(
                op_name,
                {},
            )[dependency.code] = expected_count
        self._ordering_dependencies_by_source_op.update(
            {op_name: tuple(items) for op_name, items in outgoing.items()}
        )

    # =========================================================================
    # Operator Metadata Query
    # =========================================================================

    def get_operator_spec(self, op_name: str) -> OperatorSpec:
        return self._get_operator_spec(op_name)

    def require_operator_instance(self, op_name: str) -> Any:
        owner = self._get_operator_spec(op_name).owner
        if owner is None:
            raise RuntimeError(f"operator {op_name!r} has no owner class")
        try:
            return self._instance_container.require(owner)
        except RuntimeError as error:
            raise RuntimeError(
                f"operator {op_name!r} has not been instantiated"
            ) from error

    def operator_is_coroutine_function(self, op_name: str) -> bool:
        try:
            return self._operator_is_coroutine_function[op_name]
        except KeyError as exc:
            raise ValueError(f"No operator registered for {op_name}") from exc

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
        from fluxway.common.operator_policy import BarrierType

        meta = self._get_operator_spec(op_name)
        return meta.barrier_config.type == BarrierType.PERIODIC

    def get_expected_control_arg_names(self, op_name: str) -> list[str]:
        """Get expected state variable names."""
        return list(self._expected_control_arg_names.get(op_name, []))

    def get_expected_tensor_names(self, op_name: str) -> list[str]:
        """Get expected tensor variable names."""
        return list(self._expected_tensor_names.get(op_name, []))

    def get_expected_transfer_codes(self, op_name: str) -> list[int]:
        self._get_compiled_operator(op_name)
        input_codes = self._input_transfer_codes[op_name]
        return [
            input_codes[f"{op_name}.{name}"]
            for name in self._expected_tensor_names[op_name]
        ]

    def is_source_operator(self, op_name: str) -> bool:
        """
        Check if an operator is a source operator (data producer).
        A source operator is identified by:
        1. Explicit metadata: @operator(..., is_source=True).
        2. Graph Topology: 0 predecessors and 0 input tensor arguments.

        Note: Source operators can still have static/state arguments (e.g., from Driver),
        but they do not depend on other operators for execution triggers.
        """
        meta = self._get_operator_spec(op_name)
        if meta.is_source is True:
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

    def get_source_operators(self) -> list[str]:
        """Get names of all source operators."""
        return [op for op in self._operators if self.is_source_operator(op)]

    def get_expected_transfer_shards(self, op_name: str) -> dict[int, int]:
        self._get_compiled_operator(op_name)
        return dict(self._expected_transfer_shards_cache[op_name])

    def get_operator_input_types(self, op_name: str) -> dict[str, type]:
        """
        Get the input parameter types expected by an operator.

        Returns:
            A dictionary mapping parameter names to their Types
        """
        return self._operator_input_types.get(op_name, {})

    def get_output_tuuids(self, op_name: str) -> list[int]:
        """Returns ordered list of UUIDs matching tree_flatten(output) order."""
        return list(self._get_compiled_operator(op_name).output_uuids)

    def get_input_sharding(self, op_name: str) -> dict[int, ShardingSpec]:
        self._get_compiled_operator(op_name)
        return dict(self._input_sharding[op_name])

    def get_input_transfer_codes(self, op_name: str) -> dict[str, int]:
        self._get_compiled_operator(op_name)
        return dict(self._input_transfer_codes[op_name])

    def get_incoming_dependencies(self, op_name: str) -> tuple[CompiledDependency, ...]:
        self._get_compiled_operator(op_name)
        return tuple(
            dependency
            for dependency in self._plan.incoming
            if dependency.spec.target.op_id == op_name and dependency.transfers
        )

    def get_transfer_target_op(self, transfer_code: int) -> str:
        try:
            return self._transfer_target_by_code[transfer_code]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self._spmd_id!r} has no target operator for transfer code "
                f"{transfer_code}."
            ) from exc

    def get_predecessor_topology(
        self, op_name: str
    ) -> dict[int, ResolvedDependencyRoute]:
        self._get_compiled_operator(op_name)
        return dict(self._predecessor_topology[op_name])

    def get_successor_topology(
        self, op_name: str
    ) -> dict[int, ResolvedDependencyRoute]:
        self._get_compiled_operator(op_name)
        return dict(self._successor_topology[op_name])

    def get_tensor_input_sharding_spec(
        self,
        transfer_code: int,
    ) -> ShardingSpec:
        try:
            return self._incoming_routes_by_code[transfer_code].target_sharding
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self._spmd_id!r} has no incoming transfer code {transfer_code}."
            ) from exc

    def get_transfer_source_layout(
        self,
        op_name: str,
        transfer_code: int,
    ) -> PhysicalLayout:
        route = self.get_predecessor_route(op_name, transfer_code)
        cached = self._source_layout_cache.get(transfer_code)
        if cached is not None:
            return cached
        source_name = route.dependency.source_operator.key
        if source_name not in self._source_dispatchers:
            self._source_dispatchers[source_name] = ExecutionDispatcher(
                route.dependency.source_operator.execution_layout
            )
        dispatcher = self._source_dispatchers[source_name]
        layout = dispatcher.get_effective_source_layout(
            route.mesh, route.effective_source_sharding
        )
        self._source_layout_cache[transfer_code] = layout
        return layout

    def get_transfer_target_layout(
        self,
        op_name: str,
        transfer_code: int,
    ) -> PhysicalLayout:
        return self._get_target_layout(op_name, transfer_code)

    def get_sender_world_size(
        self,
        transfer_code: int,
    ) -> int:
        try:
            return self._incoming_routes_by_code[transfer_code].world_size
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self._spmd_id!r} has no incoming transfer code {transfer_code}."
            ) from exc

    # =========================================================================
    # Sharding Query
    # =========================================================================

    def get_output_sharding(self, op_name: str) -> dict[int, "ShardingSpec"]:
        return self._compiled_output_sharding(self._get_compiled_operator(op_name))

    def get_dispatcher(self, op_name: str) -> ExecutionDispatcher:
        """Get the execution dispatcher strategy for an operator."""
        return self._dispatchers[op_name]

    def get_output_state_names(self, op_name: str) -> list[str]:
        """
        Get output state argument names for an operator.

        These are the state variables that need to be sent to successors
        via the control plane (control_argument transmission path).

        Returns:
            List of state argument names from output_arguments
        """
        return self._get_compiled_operator(op_name).state_names(is_output=True)

    # =========================================================================
    # Static Arguments
    # =========================================================================

    def get_static_args(self) -> dict[str, Any]:
        """Get all static arguments."""
        return dict(self._static_args)

    def set_static_arg(self, name: str, value: Any) -> None:
        """Set a static argument."""
        with self._lock:
            self._static_args[name] = value

    def get_static_arg(self, name: str) -> Any | None:
        """Get a static argument."""
        return self._static_args.get(name)

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(
        self,
        op_name: str,
        step: int,
        *args: Any,
        source_work_items: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        operator = self._operator_for_execution(op_name, coroutine_function=False)
        call_args, call_kwargs = self._operator_arguments(op_name, args, kwargs)
        with self._operator_execution_context(op_name, step, source_work_items):
            with function_scope():
                result = operator(*call_args, **call_kwargs)
        if _inspect.isawaitable(result):
            raise TypeError(
                f"regular operator {op_name!r} returned an awaitable; "
                "declare it with async def."
            )
        return result

    async def execute_async(
        self,
        op_name: str,
        step: int,
        *args: Any,
        source_work_items: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        operator = self._operator_for_execution(op_name, coroutine_function=True)
        call_args, call_kwargs = self._operator_arguments(op_name, args, kwargs)
        with self._operator_execution_context(op_name, step, source_work_items):
            coro = operator(*call_args, **call_kwargs)
            with function_scope():
                return await coro

    def _operator_for_execution(
        self, op_name: str, *, coroutine_function: bool
    ) -> Callable[..., Any]:
        operator = self._operators.get(op_name)
        if operator is None:
            raise ValueError(f"No operator registered for {op_name}")
        actual = self.operator_is_coroutine_function(op_name)
        if actual != coroutine_function:
            expected = "coroutine" if coroutine_function else "regular"
            actual_name = "coroutine" if actual else "regular"
            raise TypeError(f"operator {op_name!r} is {actual_name}, not {expected}.")
        return cast(Callable[..., Any], operator)

    @contextmanager
    def _operator_execution_context(
        self,
        op_name: str,
        step: int,
        source_work_items: Any | None = None,
    ) -> Generator[None]:
        from fluxway.common.decorators import operator_context

        meta = self.get_operator_spec(op_name)
        controlled_source = getattr(meta, "is_source", False) and getattr(
            meta, "requires_source_control_runtime", False
        )
        if controlled_source:
            source_work_items = _validated_source_work_items(
                op_name, step, source_work_items
            )
        dispatcher = self.get_dispatcher(op_name)
        batch_size = 1
        if isinstance(self._static_args, dict) and "batch_size" in self._static_args:
            raw_batch_size = self._static_args["batch_size"]
            if not isinstance(raw_batch_size, int) or raw_batch_size <= 0:
                raise ValueError(
                    f"batch_size in static_args must be a positive int, got {raw_batch_size!r}"
                )
            batch_size = raw_batch_size

        logical_ctx = dispatcher.get_logical_context(
            self._rank,
            self._world_size,
            step,
            batch_size=batch_size,
            source_work_items=source_work_items,
        )

        lane_runner = None
        worker_process_runtime = getattr(self, "_worker_process_runtime", None)
        if worker_process_runtime is not None:

            async def _run_in_lane(
                fn: Any,
                *args: Any,
                lane: str = "default",
                **kwargs: Any,
            ) -> Any:
                target = (lambda: fn(*args, **kwargs)) if (args or kwargs) else fn
                return await worker_process_runtime.submit(
                    target,
                    callable_id=f"{self._spmd_id}.{op_name}.lane.{lane}",
                    resource_id=0,
                    execution_lane=lane,
                )

            lane_runner = _run_in_lane

        with operator_context(
            static_args=self._static_args,
            rank=self._rank,
            world_size=self._world_size,
            spmd_id=self._spmd_id,
            operator_name=op_name,
            operator_ctx=logical_ctx,
            local_rank=self._local_rank,
            local_world_size=self._local_world_size,
            node_rank=self._node_rank,
            num_nodes=self._num_nodes,
            control_plane=self._control_plane_handle,
            ray_component_limits=self._ray_component_limits,
            instance_container=self._instance_container,
            lane_runner=lane_runner,
        ):
            yield

    def _operator_arguments(
        self,
        op_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.is_source_operator(op_name):
            return (), {}
        return args, kwargs
