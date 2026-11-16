from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Literal, cast
from uuid import uuid4

from fluxway.common.frozen_mapping import freeze_mapping
from fluxway.common.mesh import DeviceMesh
from fluxway.common.execution_layout import ExecutionLayout, compile_execution_layout
from fluxway.common.transfer import UnitRouting, TransferRelation
from fluxway.common.operator_policy import ArgumentType
from fluxway.common.coherence import TensorAccessIntent
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.canonical_lanes import apply_canonical_lane_presets
from fluxway.common.types.sharding import ReferentialRouting, ShardingSpec
from fluxway.resources import ResourceProfile


ArgumentKind = Literal["data", "control", "static"]
DependencyKey = tuple[str, str, str, str]
PortSlot = tuple[str, str, Literal["in", "out"], str]


@dataclass(frozen=True, slots=True, order=True)
class OperatorKey:
    spmd_id: str
    op_id: str

    def __post_init__(self) -> None:
        if not self.spmd_id or not self.op_id:
            raise ValueError("OperatorKey requires non-empty SPMD and operator IDs.")


@dataclass(frozen=True, slots=True)
class OperatorPort:
    name: str
    kind: ArgumentKind = "data"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("OperatorPort requires a non-empty name.")


@dataclass(frozen=True, slots=True)
class TensorTransfer:
    source: OperatorPort
    target: OperatorPort
    unit_routing: UnitRouting | None = None

    def __post_init__(self) -> None:
        if self.source.kind != "data" or self.target.kind != "data":
            raise ValueError("TensorTransfer ports must both use kind='data'.")


def _tensor_transfer_key(
    transfer: TensorTransfer,
) -> tuple[ArgumentKind, str, ArgumentKind, str]:
    return (
        transfer.source.kind,
        transfer.source.name,
        transfer.target.kind,
        transfer.target.name,
    )


@dataclass(frozen=True, slots=True)
class DependencySpec:
    source: OperatorKey
    target: OperatorKey
    transfers: tuple[TensorTransfer, ...] = ()
    local_only: bool = False
    condition: Callable[[Any], bool | Awaitable[bool]] | None = None
    symmetric_to: DependencyKey | None = None

    def __post_init__(self) -> None:
        if self.local_only and self.source.spmd_id != self.target.spmd_id:
            raise ValueError("Local-only dependencies require the same SPMD.")
        if self.condition is not None and not callable(self.condition):
            raise TypeError("Dependency condition must be callable.")
        transfer_keys = {_tensor_transfer_key(transfer) for transfer in self.transfers}
        if len(transfer_keys) != len(self.transfers):
            raise ValueError(f"Dependency {self.key!r} contains duplicate transfers.")

    @property
    def key(self) -> DependencyKey:
        return (
            self.source.spmd_id,
            self.source.op_id,
            self.target.spmd_id,
            self.target.op_id,
        )

    @property
    def source_node(self) -> str:
        return self.source.spmd_id

    @property
    def source_op(self) -> str:
        return self.source.op_id

    @property
    def target_node(self) -> str:
        return self.target.spmd_id

    @property
    def target_op(self) -> str:
        return self.target.op_id


@dataclass(frozen=True, slots=True)
class SPMDGraphSpec:
    spmd_id: str
    spmd_class: type
    mesh: DeviceMesh
    resource_profile: ResourceProfile
    operators: Mapping[str, OperatorSpec]

    @classmethod
    def create(
        cls,
        *,
        spmd_id: str,
        spmd_class: type,
        mesh: DeviceMesh,
        resource_profile: ResourceProfile,
        operators: Mapping[str, OperatorSpec],
    ) -> SPMDGraphSpec:
        return cls(
            spmd_id=spmd_id,
            spmd_class=spmd_class,
            mesh=mesh.frozen_copy(),
            resource_profile=resource_profile,
            operators=freeze_mapping(operators),
        )

    def __post_init__(self) -> None:
        if not self.spmd_id:
            raise ValueError("SPMDGraphSpec requires a non-empty spmd_id.")
        if not isinstance(self.spmd_class, type):
            raise TypeError("SPMDGraphSpec.spmd_class must be a type.")
        if not isinstance(self.mesh, DeviceMesh):
            raise TypeError("SPMDGraphSpec.mesh must be a DeviceMesh.")
        if self.mesh.size <= 0:
            raise ValueError("SPMDGraphSpec.mesh must contain at least one worker.")
        if not isinstance(self.resource_profile, ResourceProfile):
            raise TypeError("SPMDGraphSpec.resource_profile must be a ResourceProfile.")
        if not self.mesh.is_frozen:
            raise TypeError("SPMDGraphSpec.mesh must be frozen")
        if self.operators.__class__.__name__ != "FrozenMap":
            raise TypeError("SPMDGraphSpec.operators must be frozen")


@dataclass(frozen=True, slots=True)
class CompiledSPMDDeclaration:
    spmd_id: str
    spmd_class: type
    mesh: DeviceMesh
    resource_profile: ResourceProfile

    @classmethod
    def from_spec(cls, spec: SPMDGraphSpec) -> CompiledSPMDDeclaration:
        return cls(
            spmd_id=spec.spmd_id,
            spmd_class=spec.spmd_class,
            mesh=spec.mesh,
            resource_profile=spec.resource_profile,
        )

    def __post_init__(self) -> None:
        if not self.spmd_id:
            raise ValueError("CompiledSPMDDeclaration requires a non-empty spmd_id.")
        if not isinstance(self.spmd_class, type):
            raise TypeError("CompiledSPMDDeclaration.spmd_class must be a type.")
        if not isinstance(self.mesh, DeviceMesh):
            raise TypeError("CompiledSPMDDeclaration.mesh must be a DeviceMesh.")
        if self.mesh.size <= 0:
            raise ValueError(
                "CompiledSPMDDeclaration.mesh must contain at least one worker."
            )
        if not isinstance(self.resource_profile, ResourceProfile):
            raise TypeError(
                "CompiledSPMDDeclaration.resource_profile must be a ResourceProfile."
            )
        if not self.mesh.is_frozen:
            raise TypeError("CompiledSPMDDeclaration.mesh must be frozen")

    @property
    def world_size(self) -> int:
        return self.mesh.size


@dataclass(frozen=True, slots=True)
class _CompiledOperatorFacts:
    key: OperatorKey
    spec: OperatorSpec
    input_sharding: Mapping[str, ShardingSpec]
    output_sharding: Mapping[str, ShardingSpec]
    output_uuids: tuple[int, ...]
    output_state_names: tuple[str, ...]
    function_id: str
    input_signature_id: str
    output_signature_id: str

    def __post_init__(self) -> None:
        if self.input_sharding.__class__.__name__ != "FrozenMap":
            raise TypeError("compiled operator input_sharding must be frozen")
        if self.output_sharding.__class__.__name__ != "FrozenMap":
            raise TypeError("compiled operator output_sharding must be frozen")

    def get_arguments(self, is_output: bool, arg_type: ArgumentType) -> list[str]:
        if is_output and arg_type is ArgumentType.CONTROL:
            return list(self.output_state_names)
        return cast(list[str], list(self.spec.get_arguments(is_output, arg_type)))

    def state_names(self, *, is_output: bool) -> list[str]:
        return self.get_arguments(is_output, ArgumentType.CONTROL)


@dataclass(frozen=True, slots=True)
class CompiledOperator(_CompiledOperatorFacts):
    effective_input_sharding: Mapping[str, ShardingSpec]
    execution_layout: ExecutionLayout

    @classmethod
    def create(
        cls,
        *,
        key: OperatorKey,
        spec: OperatorSpec,
        input_sharding: Mapping[str, ShardingSpec],
        output_sharding: Mapping[str, ShardingSpec],
        output_uuids: Sequence[int],
        output_state_names: Sequence[str],
        function_id: str,
        input_signature_id: str,
        output_signature_id: str,
        effective_input_sharding: Mapping[str, ShardingSpec],
        execution_layout: ExecutionLayout,
    ) -> CompiledOperator:
        return cls(
            key=key,
            spec=spec,
            input_sharding=freeze_mapping(input_sharding),
            output_sharding=freeze_mapping(output_sharding),
            output_uuids=tuple(output_uuids),
            output_state_names=tuple(output_state_names),
            function_id=function_id,
            input_signature_id=input_signature_id,
            output_signature_id=output_signature_id,
            effective_input_sharding=freeze_mapping(effective_input_sharding),
            execution_layout=execution_layout,
        )

    def __post_init__(self) -> None:
        _CompiledOperatorFacts.__post_init__(self)
        if self.effective_input_sharding.__class__.__name__ != "FrozenMap":
            raise TypeError("compiled operator effective sharding must be frozen")
        if set(self.effective_input_sharding) != set(self.input_sharding):
            raise ValueError(
                f"Operator {self.key!r} effective input sharding ports differ "
                "from input sharding ports."
            )
        referential = [
            name
            for name, sharding in self.effective_input_sharding.items()
            if sharding.is_referential
        ]
        if referential:
            raise ValueError(
                f"Operator {self.key!r} has referential effective input sharding "
                f"for ports {referential}."
            )


@dataclass(frozen=True, slots=True)
class _LogicalTensorProducer:
    operator: OperatorKey
    port: str
    sharding: ShardingSpec | None

    @property
    def name(self) -> str:
        return f"{self.operator.spmd_id}.{self.operator.op_id}:{self.port}"


@dataclass(frozen=True, slots=True)
class CompiledLogicalTensor:
    logical_tuuid: int
    producer: OperatorKey
    producer_port: str
    producer_sharding: ShardingSpec | None
    effective_sharding: ShardingSpec | None

    def __post_init__(self) -> None:
        if self.logical_tuuid <= 0:
            raise ValueError("CompiledLogicalTensor requires a positive logical TUUID.")
        if not self.producer_port:
            raise ValueError("CompiledLogicalTensor requires a producer port.")
        if self.producer_sharding is None:
            if self.effective_sharding is not None:
                raise ValueError(
                    f"Logical tensor {self.name} has effective sharding without "
                    "producer sharding."
                )
            return
        if self.effective_sharding is None:
            raise ValueError(
                f"Logical tensor {self.name} has producer sharding without "
                "effective sharding."
            )
        if self.effective_sharding.is_referential:
            raise ValueError(
                f"Logical tensor {self.name} requires non-referential effective "
                "sharding."
            )

    @property
    def name(self) -> str:
        return f"{self.producer.spmd_id}.{self.producer.op_id}:{self.producer_port}"

    def require_effective_sharding(self) -> ShardingSpec:
        if self.effective_sharding is None:
            raise ValueError(f"Logical tensor {self.name} has no producer sharding.")
        return self.effective_sharding


class _ShardingReferenceResolver:
    __slots__ = ("_operators", "_producers", "_resolved", "_resolved_targets")

    def __init__(
        self,
        operators: Mapping[OperatorKey, _CompiledOperatorFacts],
    ) -> None:
        producers: dict[int, _LogicalTensorProducer] = {}
        for operator in operators.values():
            output_names = operator.get_arguments(
                is_output=True,
                arg_type=ArgumentType.DATA,
            )
            if len(output_names) != len(operator.output_uuids):
                raise ValueError(
                    f"Operator {operator.key!r} output names and logical TUUIDs "
                    "differ in length."
                )
            for port, tuuid in zip(
                output_names,
                operator.output_uuids,
                strict=True,
            ):
                if tuuid in producers:
                    raise ValueError(
                        f"Operators {producers[tuuid].name} and "
                        f"{operator.key.spmd_id}.{operator.key.op_id}:{port} share "
                        f"logical TUUID {tuuid}."
                    )
                producers[tuuid] = _LogicalTensorProducer(
                    operator=operator.key,
                    port=port,
                    sharding=operator.output_sharding.get(port),
                )
        self._operators = operators
        self._producers = producers
        self._resolved: dict[int, ShardingSpec] = {}
        self._resolved_targets: dict[tuple[OperatorKey, str], ShardingSpec] = {}

    def compile_logical_tensors(self) -> tuple[CompiledLogicalTensor, ...]:
        return tuple(
            CompiledLogicalTensor(
                logical_tuuid=tuuid,
                producer=producer.operator,
                producer_port=producer.port,
                producer_sharding=producer.sharding,
                effective_sharding=(
                    None if producer.sharding is None else self.resolve_producer(tuuid)
                ),
            )
            for tuuid, producer in sorted(self._producers.items())
        )

    def resolve_producer(self, tuuid: int) -> ShardingSpec:
        producer = self._require_producer(tuuid, requested_by=None)
        return self._resolve_tuuid(
            tuuid,
            requested_by=producer.name,
            trace=(),
        )

    def resolve_target(
        self,
        *,
        operator: OperatorKey,
        port: str,
    ) -> ShardingSpec:
        endpoint = (operator, port)
        cached = self._resolved_targets.get(endpoint)
        if cached is not None:
            return cached
        compiled_operator = self._operators[operator]
        try:
            sharding = compiled_operator.input_sharding[port]
        except KeyError as exc:
            raise ValueError(
                f"Target port {operator.spmd_id}.{operator.op_id}:{port} has no "
                "input sharding."
            ) from exc
        if not isinstance(sharding.routing, ReferentialRouting):
            self._resolved_targets[endpoint] = sharding
            return sharding
        resolved = self._resolve_target_reference(
            operator=operator,
            port=port,
            requested_by=f"{operator.spmd_id}.{operator.op_id}:{port}",
            trace=(),
        )
        self._resolved_targets[endpoint] = resolved
        return resolved

    def _resolve_target_reference(
        self,
        *,
        operator: OperatorKey,
        port: str,
        requested_by: str,
        trace: tuple[tuple[OperatorKey, str], ...],
    ) -> ShardingSpec:
        endpoint = (operator, port)
        if endpoint in trace:
            cycle_start = trace.index(endpoint)
            cycle = (*trace[cycle_start:], endpoint)
            names = " -> ".join(
                f"{key.spmd_id}.{key.op_id}:{name}" for key, name in cycle
            )
            raise ValueError(
                f"Referential sharding cycle while resolving {requested_by}: {names}."
            )
        cached = self._resolved_targets.get(endpoint)
        if cached is not None:
            return cached
        compiled_operator = self._operators[operator]
        try:
            sharding = compiled_operator.input_sharding[port]
        except KeyError as exc:
            raise ValueError(
                f"Referential sharding for {requested_by} resolves to target port "
                f"{operator.spmd_id}.{operator.op_id}:{port}, which has no input "
                "sharding."
            ) from exc
        routing = sharding.routing
        if not isinstance(routing, ReferentialRouting):
            self._resolved_targets[endpoint] = sharding
            return sharding
        self._validate_dimension_map(routing, requested_by=requested_by)
        if "." not in routing.path:
            raise ValueError(
                f"Referential target sharding for {requested_by} requires an "
                f"'<operator>.<input>' path, got {routing.path!r}."
            )
        referenced_op, referenced_port = routing.path.split(".", 1)
        if referenced_op == operator.op_id:
            resolved = self._resolve_target_reference(
                operator=operator,
                port=referenced_port,
                requested_by=requested_by,
                trace=(*trace, endpoint),
            )
        else:
            reference_tuuid = routing.tuuid
            if reference_tuuid is None:
                raise ValueError(
                    f"Referential sharding for {requested_by} has unresolved path "
                    f"{routing.path!r}."
                )
            resolved = self._resolve_tuuid(
                reference_tuuid,
                requested_by=requested_by,
                trace=(),
            )
        self._resolved_targets[endpoint] = resolved
        return resolved

    def _resolve_tuuid(
        self,
        tuuid: int,
        *,
        requested_by: str,
        trace: tuple[int, ...],
    ) -> ShardingSpec:
        if tuuid in trace:
            cycle_start = trace.index(tuuid)
            cycle = (*trace[cycle_start:], tuuid)
            names = " -> ".join(
                self._require_producer(item, requested_by=requested_by).name
                for item in cycle
            )
            raise ValueError(
                f"Referential sharding cycle while resolving {requested_by}: {names}."
            )
        cached = self._resolved.get(tuuid)
        if cached is not None:
            return cached
        producer = self._require_producer(tuuid, requested_by=requested_by)
        sharding = producer.sharding
        if sharding is None:
            raise ValueError(
                f"Referential sharding for {requested_by} resolves to "
                f"{producer.name}, which has no output sharding."
            )
        if isinstance(sharding.routing, ReferentialRouting):
            self._validate_dimension_map(
                sharding.routing,
                requested_by=producer.name,
            )
            reference_tuuid = sharding.routing.tuuid
            if reference_tuuid is None:
                raise ValueError(
                    f"Referential sharding for {producer.name} has unresolved path "
                    f"{sharding.routing.path!r}."
                )
            sharding = self._resolve_tuuid(
                reference_tuuid,
                requested_by=(
                    f"{producer.name} through reference {sharding.routing.path!r}"
                ),
                trace=(*trace, tuuid),
            )
        self._resolved[tuuid] = sharding
        return sharding

    @staticmethod
    def _validate_dimension_map(
        routing: ReferentialRouting,
        *,
        requested_by: str,
    ) -> None:
        if routing.dim_map != {0: 0}:
            raise ValueError(
                f"Referential sharding for {requested_by} uses unsupported dimension "
                f"map {routing.dim_map!r}; only identity mapping {{0: 0}} is "
                "currently executable."
            )

    def _require_producer(
        self,
        tuuid: int,
        *,
        requested_by: str | None,
    ) -> _LogicalTensorProducer:
        try:
            return self._producers[tuuid]
        except KeyError as exc:
            context = f" for {requested_by}" if requested_by is not None else ""
            raise ValueError(
                f"Referential sharding{context} resolves to logical TUUID {tuuid}, "
                "which has no producer port."
            ) from exc


@dataclass(frozen=True, slots=True)
class CompiledTensorTransfer:
    spec: TensorTransfer
    transfer_code: int
    dependency_code: int
    logical_tuuid: int
    source_sharding: ShardingSpec
    target_sharding: ShardingSpec

    def __post_init__(self) -> None:
        if self.transfer_code <= 0:
            raise ValueError(
                "CompiledTensorTransfer requires a positive transfer code."
            )
        if self.dependency_code <= 0:
            raise ValueError(
                "CompiledTensorTransfer requires a positive dependency code."
            )
        if self.logical_tuuid <= 0:
            raise ValueError(
                "CompiledTensorTransfer requires a positive logical TUUID."
            )

    @property
    def uses_receipt_routing(self) -> bool:
        return isinstance(self.source_sharding.routing, ReferentialRouting)


@dataclass(frozen=True, slots=True)
class CompiledDependency:
    code: int
    spec: DependencySpec
    source_operator: CompiledOperator
    target_operator: CompiledOperator
    transfers: tuple[CompiledTensorTransfer, ...]
    source_world_size: int
    target_world_size: int
    source_mesh: DeviceMesh
    target_mesh: DeviceMesh

    _unit_relations: Mapping[int, TransferRelation] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        transfer_codes = {transfer.transfer_code for transfer in self.transfers}
        if len(transfer_codes) != len(self.transfers):
            raise ValueError(
                f"Dependency code {self.code} has duplicate transfer codes."
            )
        if any(transfer.dependency_code != self.code for transfer in self.transfers):
            raise ValueError(
                f"Dependency code {self.code} owns a transfer with a different "
                "dependency code."
            )
        if self._unit_relations.__class__.__name__ != "FrozenMap":
            raise TypeError("compiled dependency unit relations must be frozen")

    @classmethod
    def create(
        cls,
        *,
        code: int,
        spec: DependencySpec,
        source_operator: CompiledOperator,
        target_operator: CompiledOperator,
        transfers: tuple[CompiledTensorTransfer, ...]
        | Sequence[CompiledTensorTransfer],
        source_world_size: int,
        target_world_size: int,
        source_mesh: DeviceMesh,
        target_mesh: DeviceMesh,
        _unit_relations: Mapping[int, TransferRelation] | None = None,
    ) -> CompiledDependency:
        canonical_transfers = tuple(transfers)
        if _unit_relations is not None:
            unit_relations = freeze_mapping(_unit_relations)
        else:
            relations: dict[int, TransferRelation] = {}
            for transfer in canonical_transfers:
                routing = transfer.spec.unit_routing
                if routing is None:
                    continue
                if spec.local_only:
                    raise ValueError("whole-unit routing is not a local-only transfer")
                if transfer.uses_receipt_routing:
                    raise ValueError(
                        "whole-unit and receipt routing cannot share a transfer"
                    )
                relation = routing.bind(
                    source_operator.execution_layout.logical_mesh.size,
                    target_world_size,
                )
                if (
                    relation.source_count
                    != source_operator.execution_layout.logical_mesh.size
                    or len(relation.producers_by_target) != target_world_size
                ):
                    raise ValueError("unit routing changed the compiled rank domains")
                if any(not targets for targets in relation.consumers_by_source):
                    raise ValueError("unit routing requires complete producer coverage")
                if any(
                    len(producers) != 1 for producers in relation.producers_by_target
                ):
                    raise ValueError(
                        "whole-unit targets require exactly one producer per execution"
                    )
                relations[transfer.transfer_code] = relation
            unit_relations = freeze_mapping(relations)
        return cls(
            code=code,
            spec=spec,
            source_operator=source_operator,
            target_operator=target_operator,
            transfers=canonical_transfers,
            source_world_size=source_world_size,
            target_world_size=target_world_size,
            source_mesh=source_mesh,
            target_mesh=target_mesh,
            _unit_relations=unit_relations,
        )

    def unit_relation(self, transfer_code: int) -> TransferRelation | None:
        return self._unit_relations.get(transfer_code)

    @property
    def key(self) -> DependencyKey:
        return self.spec.key

    @property
    def source_participating_count(self) -> int:
        if self.spec.local_only:
            return 1
        return self.source_operator.execution_layout.logical_mesh.size


@dataclass(frozen=True, slots=True)
class SPMDGraphView:
    graph_id: str
    generation: int
    name: str
    declaration: CompiledSPMDDeclaration
    operators: tuple[CompiledOperator, ...]
    logical_tensors: tuple[CompiledLogicalTensor, ...]
    incoming: tuple[CompiledDependency, ...]
    outgoing: tuple[CompiledDependency, ...]
    operator_ids: tuple[str, ...]
    _operators_by_id: Mapping[str, CompiledOperator] = field(
        repr=False,
        compare=False,
    )
    _dependencies_by_code: Mapping[int, CompiledDependency] = field(
        repr=False,
        compare=False,
    )
    _transfers_by_code: Mapping[int, CompiledTensorTransfer] = field(
        repr=False,
        compare=False,
    )
    _logical_tensors_by_tuuid: Mapping[int, CompiledLogicalTensor] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.graph_id:
            raise ValueError("SPMDGraphView requires a non-empty graph_id.")
        if self.generation < 0:
            raise ValueError("SPMDGraphView generation must be non-negative.")
        operators = {operator.key.op_id: operator for operator in self.operators}
        if len(operators) != len(self.operators):
            raise ValueError(f"SPMD {self.spmd_id!r} has duplicate operators.")
        if self.operator_ids != tuple(operators):
            raise ValueError("SPMD operator_ids are inconsistent")
        if self._operators_by_id.__class__.__name__ != "FrozenMap":
            raise TypeError("SPMD operator index must be frozen")
        if self._dependencies_by_code.__class__.__name__ != "FrozenMap":
            raise TypeError("SPMD dependency index must be frozen")
        if self._transfers_by_code.__class__.__name__ != "FrozenMap":
            raise TypeError("SPMD transfer index must be frozen")
        if self._logical_tensors_by_tuuid.__class__.__name__ != "FrozenMap":
            raise TypeError("SPMD logical tensor index must be frozen")

    @classmethod
    def create(
        cls,
        *,
        graph_id: str,
        generation: int,
        name: str,
        declaration: CompiledSPMDDeclaration,
        operators: Sequence[CompiledOperator],
        logical_tensors: Sequence[CompiledLogicalTensor],
        incoming: Sequence[CompiledDependency],
        outgoing: Sequence[CompiledDependency],
        operator_ids: Sequence[str] | None = None,
        _operators_by_id: Mapping[str, CompiledOperator] | None = None,
        _dependencies_by_code: Mapping[int, CompiledDependency] | None = None,
        _transfers_by_code: Mapping[int, CompiledTensorTransfer] | None = None,
        _logical_tensors_by_tuuid: Mapping[int, CompiledLogicalTensor] | None = None,
    ) -> SPMDGraphView:
        canonical_operators = tuple(operators)
        canonical_logical_tensors = tuple(logical_tensors)
        canonical_incoming = tuple(incoming)
        canonical_outgoing = tuple(outgoing)

        operators_map = (
            freeze_mapping(_operators_by_id)
            if _operators_by_id is not None
            else freeze_mapping({op.key.op_id: op for op in canonical_operators})
        )
        canonical_operator_ids = (
            tuple(operator_ids)
            if operator_ids is not None
            else tuple(operators_map.keys())
        )
        dependencies_map = (
            freeze_mapping(_dependencies_by_code)
            if _dependencies_by_code is not None
            else freeze_mapping(
                _index_dependencies_by_code(
                    (*canonical_incoming, *canonical_outgoing), owner="SPMD"
                )
            )
        )
        transfers_map = (
            freeze_mapping(_transfers_by_code)
            if _transfers_by_code is not None
            else freeze_mapping(
                _index_transfers_by_code(tuple(dependencies_map.values()), owner="SPMD")
            )
        )
        logical_tensors_map = (
            freeze_mapping(_logical_tensors_by_tuuid)
            if _logical_tensors_by_tuuid is not None
            else freeze_mapping(
                _index_logical_tensors(canonical_logical_tensors, owner="SPMD")
            )
        )
        return cls(
            graph_id=graph_id,
            generation=generation,
            name=name,
            declaration=declaration,
            operators=canonical_operators,
            logical_tensors=canonical_logical_tensors,
            incoming=canonical_incoming,
            outgoing=canonical_outgoing,
            operator_ids=canonical_operator_ids,
            _operators_by_id=operators_map,
            _dependencies_by_code=dependencies_map,
            _transfers_by_code=transfers_map,
            _logical_tensors_by_tuuid=logical_tensors_map,
        )

    @property
    def spmd_id(self) -> str:
        return self.declaration.spmd_id

    @property
    def spmd_class(self) -> type:
        return self.declaration.spmd_class

    @property
    def mesh(self) -> DeviceMesh:
        return self.declaration.mesh

    @property
    def resource_profile(self) -> ResourceProfile:
        return self.declaration.resource_profile

    @property
    def world_size(self) -> int:
        return self.declaration.world_size

    def operator(self, op_id: str) -> CompiledOperator:
        try:
            return self._operators_by_id[op_id]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self.spmd_id!r} has no operator {op_id!r}."
            ) from exc

    def dependency(self, code: int) -> CompiledDependency:
        try:
            return self._dependencies_by_code[code]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self.spmd_id!r} has no dependency code {code}."
            ) from exc

    def transfer(self, code: int) -> CompiledTensorTransfer:
        try:
            return self._transfers_by_code[code]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self.spmd_id!r} has no transfer code {code}."
            ) from exc

    def logical_tensor(self, logical_tuuid: int) -> CompiledLogicalTensor:
        try:
            return self._logical_tensors_by_tuuid[logical_tuuid]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {self.spmd_id!r} has no logical tensor {logical_tuuid}."
            ) from exc


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    graph_id: str
    generation: int
    name: str
    logical_tensors: tuple[CompiledLogicalTensor, ...]
    spmds: tuple[SPMDGraphView, ...]
    dependencies: tuple[CompiledDependency, ...]
    _spmds_by_id: Mapping[str, SPMDGraphView] = field(
        repr=False,
        compare=False,
    )
    _logical_tensors_by_tuuid: Mapping[int, CompiledLogicalTensor] = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.graph_id:
            raise ValueError("CompiledGraph requires a non-empty graph_id.")
        if self.generation < 0:
            raise ValueError("CompiledGraph generation must be non-negative.")
        spmds = {spmd.spmd_id: spmd for spmd in self.spmds}
        if len(spmds) != len(self.spmds):
            raise ValueError("CompiledGraph has duplicate SPMD IDs.")
        logical_tensors = _index_logical_tensors(
            self.logical_tensors,
            owner="CompiledGraph",
        )
        for spmd in self.spmds:
            if spmd.graph_id != self.graph_id:
                raise ValueError(
                    f"SPMD {spmd.spmd_id!r} graph_id {spmd.graph_id!r} does not "
                    f"match CompiledGraph {self.graph_id!r}."
                )
            if spmd.generation != self.generation:
                raise ValueError(
                    f"SPMD {spmd.spmd_id!r} generation {spmd.generation} does not "
                    f"match CompiledGraph generation {self.generation}."
                )
            for logical_tensor in spmd.logical_tensors:
                canonical = logical_tensors.get(logical_tensor.logical_tuuid)
                if canonical is not logical_tensor:
                    raise ValueError(
                        f"SPMD {spmd.spmd_id!r} logical tensor "
                        f"{logical_tensor.logical_tuuid} is not the canonical "
                        "CompiledGraph definition."
                    )
        for dependency in self.dependencies:
            source = spmds[dependency.spec.source.spmd_id].declaration
            target = spmds[dependency.spec.target.spmd_id].declaration
            if dependency.source_mesh is not source.mesh:
                raise ValueError(
                    f"Dependency {dependency.code} source mesh is not the compiled "
                    "SPMD declaration mesh."
                )
            if dependency.target_mesh is not target.mesh:
                raise ValueError(
                    f"Dependency {dependency.code} target mesh is not the compiled "
                    "SPMD declaration mesh."
                )
            if dependency.source_world_size != source.world_size:
                raise ValueError(
                    f"Dependency {dependency.code} source world size differs from "
                    "the compiled SPMD declaration."
                )
            if dependency.target_world_size != target.world_size:
                raise ValueError(
                    f"Dependency {dependency.code} target world size differs from "
                    "the compiled SPMD declaration."
                )
        _index_dependencies_by_code(
            self.dependencies,
            owner="CompiledGraph",
        )
        _index_transfers_by_code(
            self.dependencies,
            owner="CompiledGraph",
        )
        if self._spmds_by_id.__class__.__name__ != "FrozenMap":
            raise TypeError("compiled graph SPMD index must be frozen")
        if self._logical_tensors_by_tuuid.__class__.__name__ != "FrozenMap":
            raise TypeError("compiled graph logical tensor index must be frozen")

    @classmethod
    def create(
        cls,
        *,
        graph_id: str,
        generation: int,
        name: str,
        logical_tensors: Sequence[CompiledLogicalTensor],
        spmds: Sequence[SPMDGraphView],
        dependencies: Sequence[CompiledDependency],
        _spmds_by_id: Mapping[str, SPMDGraphView] | None = None,
        _logical_tensors_by_tuuid: Mapping[int, CompiledLogicalTensor] | None = None,
    ) -> CompiledGraph:
        canonical_logical_tensors = tuple(logical_tensors)
        canonical_spmds = tuple(spmds)
        canonical_dependencies = tuple(dependencies)

        spmds_map = (
            freeze_mapping(_spmds_by_id)
            if _spmds_by_id is not None
            else freeze_mapping({spmd.spmd_id: spmd for spmd in canonical_spmds})
        )
        logical_tensors_map = (
            freeze_mapping(_logical_tensors_by_tuuid)
            if _logical_tensors_by_tuuid is not None
            else freeze_mapping(
                _index_logical_tensors(canonical_logical_tensors, owner="CompiledGraph")
            )
        )
        return cls(
            graph_id=graph_id,
            generation=generation,
            name=name,
            logical_tensors=canonical_logical_tensors,
            spmds=canonical_spmds,
            dependencies=canonical_dependencies,
            _spmds_by_id=spmds_map,
            _logical_tensors_by_tuuid=logical_tensors_map,
        )

    def spmd(self, spmd_id: str) -> SPMDGraphView:
        try:
            return self._spmds_by_id[spmd_id]
        except KeyError as exc:
            raise ValueError(f"CompiledGraph has no SPMD {spmd_id!r}.") from exc

    def logical_tensor(self, logical_tuuid: int) -> CompiledLogicalTensor:
        try:
            return self._logical_tensors_by_tuuid[logical_tuuid]
        except KeyError as exc:
            raise ValueError(
                f"CompiledGraph has no logical tensor {logical_tuuid}."
            ) from exc

    @property
    def spmd_ids(self) -> tuple[str, ...]:
        return tuple(self._spmds_by_id)

    @property
    def declarations(self) -> tuple[CompiledSPMDDeclaration, ...]:
        return tuple(spmd.declaration for spmd in self.spmds)

    @property
    def source_spmd_ids(self) -> tuple[str, ...]:
        targets = {dependency.spec.target.spmd_id for dependency in self.dependencies}
        return tuple(spmd_id for spmd_id in self._spmds_by_id if spmd_id not in targets)

    @property
    def source_control_spmd_ids(self) -> tuple[str, ...]:
        """SPMDs whose source descriptor owns ordered publication windows."""
        return tuple(
            spmd_id
            for spmd_id in self.spmd_ids
            if any(
                operator.spec.requires_source_control_runtime
                for operator in self.spmd(spmd_id).operators
            )
        )

    @property
    def sink_spmd_ids(self) -> tuple[str, ...]:
        sources = {dependency.spec.source.spmd_id for dependency in self.dependencies}
        return tuple(spmd_id for spmd_id in self._spmds_by_id if spmd_id not in sources)

    @property
    def dependency_specs(self) -> tuple[DependencySpec, ...]:
        return tuple(dependency.spec for dependency in self.dependencies)

    @property
    def requires_source_control_runtime(self) -> bool:
        """Whether a descriptor-backed source needs global source control.

        This is true for a source operator that publishes windows through a
        source descriptor. Such operators require ordered control records,
        admission leases, and drain-boundary retention; it is not a generic
        test for every source operator or for data-plane ordering.
        """
        return bool(self.source_control_spmd_ids)


def _index_dependencies_by_code(
    dependencies: tuple[CompiledDependency, ...],
    *,
    owner: str,
) -> dict[int, CompiledDependency]:
    indexed: dict[int, CompiledDependency] = {}
    for dependency in dependencies:
        existing = indexed.get(dependency.code)
        if existing is not None and existing is not dependency:
            raise ValueError(
                f"{owner} has duplicate dependency code {dependency.code}."
            )
        indexed[dependency.code] = dependency
    return indexed


def _index_transfers_by_code(
    dependencies: tuple[CompiledDependency, ...],
    *,
    owner: str,
) -> dict[int, CompiledTensorTransfer]:
    indexed: dict[int, CompiledTensorTransfer] = {}
    for dependency in dependencies:
        for transfer in dependency.transfers:
            existing = indexed.get(transfer.transfer_code)
            if existing is not None and existing is not transfer:
                raise ValueError(
                    f"{owner} has duplicate transfer code {transfer.transfer_code}."
                )
            indexed[transfer.transfer_code] = transfer
    return indexed


def _index_logical_tensors(
    logical_tensors: tuple[CompiledLogicalTensor, ...],
    *,
    owner: str,
) -> dict[int, CompiledLogicalTensor]:
    indexed: dict[int, CompiledLogicalTensor] = {}
    for logical_tensor in logical_tensors:
        if logical_tensor.logical_tuuid in indexed:
            raise ValueError(
                f"{owner} has duplicate logical tensor {logical_tensor.logical_tuuid}."
            )
        indexed[logical_tensor.logical_tuuid] = logical_tensor
    return indexed


def _project_logical_tensors(
    *,
    spmd_id: str,
    operators: Mapping[OperatorKey, CompiledOperator],
    dependencies: tuple[CompiledDependency, ...],
    logical_tensors: Mapping[int, CompiledLogicalTensor],
) -> tuple[CompiledLogicalTensor, ...]:
    local_operators = tuple(
        operator for key, operator in operators.items() if key.spmd_id == spmd_id
    )
    pending = {
        logical_tuuid
        for operator in local_operators
        for logical_tuuid in operator.output_uuids
    }
    for operator in local_operators:
        for sharding in (
            *operator.input_sharding.values(),
            *operator.output_sharding.values(),
        ):
            routing = sharding.routing
            if isinstance(routing, ReferentialRouting):
                if routing.tuuid is None:
                    raise ValueError(
                        f"Operator {operator.key!r} has unresolved referential "
                        f"sharding path {routing.path!r}."
                    )
                pending.add(routing.tuuid)
    for dependency in dependencies:
        if spmd_id not in {
            dependency.spec.source.spmd_id,
            dependency.spec.target.spmd_id,
        }:
            continue
        pending.update(transfer.logical_tuuid for transfer in dependency.transfers)

    projected: dict[int, CompiledLogicalTensor] = {}
    while pending:
        logical_tuuid = pending.pop()
        if logical_tuuid in projected:
            continue
        try:
            logical_tensor = logical_tensors[logical_tuuid]
        except KeyError as exc:
            raise ValueError(
                f"SPMD {spmd_id!r} references logical tensor {logical_tuuid}, "
                "which has no canonical CompiledGraph definition."
            ) from exc
        projected[logical_tuuid] = logical_tensor
        producer_sharding = logical_tensor.producer_sharding
        if producer_sharding is None:
            continue
        routing = producer_sharding.routing
        if not isinstance(routing, ReferentialRouting):
            continue
        if routing.tuuid is None:
            raise ValueError(
                f"Logical tensor {logical_tensor.name} has unresolved referential "
                f"sharding path {routing.path!r}."
            )
        pending.add(routing.tuuid)
    return tuple(projected[tuuid] for tuuid in sorted(projected))


class GraphCompiler:
    """Compile an initial graph and append-only graph generations.

    A compiler instance owns wire identities for one worker lifetime. Existing
    dependency, transfer, and logical tensor identities are never reused.
    """

    def __init__(self, *, graph_id: str | None = None) -> None:
        self._graph_id = graph_id or uuid4().hex
        self._generation = -1
        self._known_name: str | None = None
        self._known_declarations: dict[str, CompiledSPMDDeclaration] = {}
        self._known_operators: dict[OperatorKey, OperatorSpec] = {}
        self._known_dependencies: dict[DependencyKey, DependencySpec] = {}
        self._port_tuuids: dict[PortSlot, int] = {}
        self._dependency_codes: dict[DependencyKey, int] = {}
        self._transfer_codes: dict[
            tuple[DependencyKey, tuple[str, str, str, str]], int
        ] = {}

    @property
    def graph_id(self) -> str:
        return self._graph_id

    @property
    def generation(self) -> int:
        if self._generation < 0:
            raise RuntimeError("GraphCompiler has not compiled a graph.")
        return self._generation

    def compile(
        self,
        *,
        name: str,
        spmds: tuple[SPMDGraphSpec, ...],
        dependencies: tuple[DependencySpec, ...],
        canonical_lane_presets: bool = False,
    ) -> CompiledGraph:
        if canonical_lane_presets:
            spmds = apply_canonical_lane_presets(spmds)
        spmd_by_id = _index_spmds(spmds)
        declarations = _compile_spmd_declarations(spmd_by_id)
        specs_by_key = _index_operator_specs(spmds)
        self._validate_generation(name, declarations, specs_by_key, dependencies)
        port_tuuids, output_uuids = _allocate_tuuids(
            specs_by_key,
            existing=self._port_tuuids,
        )
        ordered_dependencies = _validate_dependencies(
            dependencies,
            specs_by_key,
            port_tuuids,
        )
        output_state_names = _compile_output_state_names(
            specs_by_key,
            ordered_dependencies,
        )
        operator_facts = _compile_operator_facts(
            specs_by_key,
            port_tuuids,
            output_uuids,
            output_state_names,
        )
        sharding_resolver = _ShardingReferenceResolver(operator_facts)
        compiled_operators = _compile_operators(
            operator_facts,
            sharding_resolver,
            declarations,
        )
        dependency_codes = _allocate_dependency_codes(
            ordered_dependencies,
            existing=self._dependency_codes,
        )
        transfer_codes = _allocate_transfer_codes(
            ordered_dependencies,
            existing=self._transfer_codes,
        )
        compiled_dependencies = _compile_dependencies(
            ordered_dependencies,
            declarations,
            compiled_operators,
            port_tuuids,
            dependency_codes=dependency_codes,
            transfer_codes=transfer_codes,
        )
        logical_tensors = sharding_resolver.compile_logical_tensors()
        logical_tensors_by_tuuid = _index_logical_tensors(
            logical_tensors,
            owner="GraphCompiler",
        )
        self._known_name = name
        self._known_declarations = dict(declarations)
        self._known_operators = dict(specs_by_key)
        self._known_dependencies = {
            dependency.key: dependency for dependency in ordered_dependencies
        }
        self._port_tuuids = dict(port_tuuids)
        self._dependency_codes = dict(dependency_codes)
        self._transfer_codes = dict(transfer_codes)
        self._generation += 1
        graph_id = self._graph_id
        generation = self._generation
        spmd_views = tuple(
            SPMDGraphView.create(
                graph_id=graph_id,
                generation=generation,
                name=name,
                declaration=declarations[spmd.spmd_id],
                operators=tuple(
                    compiled_operators[OperatorKey(spmd.spmd_id, op_id)]
                    for op_id in sorted(spmd.operators)
                ),
                logical_tensors=_project_logical_tensors(
                    spmd_id=spmd.spmd_id,
                    operators=compiled_operators,
                    dependencies=compiled_dependencies,
                    logical_tensors=logical_tensors_by_tuuid,
                ),
                incoming=tuple(
                    dependency
                    for dependency in compiled_dependencies
                    if dependency.spec.target.spmd_id == spmd.spmd_id
                ),
                outgoing=tuple(
                    dependency
                    for dependency in compiled_dependencies
                    if dependency.spec.source.spmd_id == spmd.spmd_id
                ),
            )
            for spmd in spmds
        )
        return CompiledGraph.create(
            graph_id=graph_id,
            generation=generation,
            name=name,
            logical_tensors=logical_tensors,
            spmds=spmd_views,
            dependencies=compiled_dependencies,
        )

    def _validate_generation(
        self,
        name: str,
        declarations: Mapping[str, CompiledSPMDDeclaration],
        specs_by_key: Mapping[OperatorKey, OperatorSpec],
        dependencies: tuple[DependencySpec, ...],
    ) -> None:
        if self._generation < 0:
            return
        if name != self._known_name:
            raise ValueError(
                f"GraphCompiler graph name changed from {self._known_name!r} to {name!r}."
            )
        for spmd_id, previous in self._known_declarations.items():
            current = declarations.get(spmd_id)
            if current is None:
                raise ValueError(f"GraphCompiler cannot remove SPMD {spmd_id!r}.")
            if not _same_spmd_declaration(current, previous):
                raise ValueError(
                    f"GraphCompiler cannot replace declaration for SPMD {spmd_id!r}."
                )
        for key, previous in self._known_operators.items():
            current = specs_by_key.get(key)
            if current is None or current != previous:
                raise ValueError(
                    f"GraphCompiler cannot replace operator {key!r} after install."
                )
        current_by_key = {dependency.key: dependency for dependency in dependencies}
        for key, previous in self._known_dependencies.items():
            current = current_by_key.get(key)
            if current is None:
                raise ValueError(
                    f"GraphCompiler cannot remove dependency {key!r} before drain."
                )
            if replace(current, transfers=()) != replace(previous, transfers=()):
                raise ValueError(
                    f"GraphCompiler cannot replace dependency metadata {key!r}."
                )
            previous_transfers = {
                _transfer_identity(key, transfer) for transfer in previous.transfers
            }
            current_transfers = {
                _transfer_identity(key, transfer) for transfer in current.transfers
            }
            if not previous_transfers.issubset(current_transfers):
                raise ValueError(
                    f"GraphCompiler cannot remove transfers from dependency {key!r}."
                )


def compile_graph(
    *,
    name: str,
    spmds: tuple[SPMDGraphSpec, ...],
    dependencies: tuple[DependencySpec, ...],
    canonical_lane_presets: bool = False,
) -> CompiledGraph:
    return GraphCompiler().compile(
        name=name,
        spmds=spmds,
        dependencies=dependencies,
        canonical_lane_presets=canonical_lane_presets,
    )


def _index_spmds(
    spmds: tuple[SPMDGraphSpec, ...],
) -> dict[str, SPMDGraphSpec]:
    indexed: dict[str, SPMDGraphSpec] = {}
    for spmd in spmds:
        if spmd.spmd_id in indexed:
            raise ValueError(f"Duplicate SPMD ID {spmd.spmd_id!r}.")
        indexed[spmd.spmd_id] = spmd
    return indexed


def _compile_spmd_declarations(
    spmds: Mapping[str, SPMDGraphSpec],
) -> dict[str, CompiledSPMDDeclaration]:
    return {
        spmd_id: CompiledSPMDDeclaration.from_spec(spmd)
        for spmd_id, spmd in spmds.items()
    }


def _same_spmd_declaration(
    left: CompiledSPMDDeclaration,
    right: CompiledSPMDDeclaration,
) -> bool:
    return (
        left.spmd_id == right.spmd_id
        and left.spmd_class is right.spmd_class
        and left.resource_profile == right.resource_profile
        and left.mesh.axis_names == right.mesh.axis_names
        and left.mesh.shape == right.mesh.shape
        and left.mesh.device_ids.tobytes() == right.mesh.device_ids.tobytes()
    )


def _index_operator_specs(
    spmds: tuple[SPMDGraphSpec, ...],
) -> dict[OperatorKey, OperatorSpec]:
    indexed: dict[OperatorKey, OperatorSpec] = {}
    for spmd in spmds:
        for op_id, spec in spmd.operators.items():
            if op_id != spec.name:
                raise ValueError(
                    f"SPMD {spmd.spmd_id!r} operator key {op_id!r} does not match "
                    f"OperatorSpec.name {spec.name!r}."
                )
            indexed[OperatorKey(spmd.spmd_id, op_id)] = spec
    return indexed


def _allocate_tuuids(
    specs_by_key: Mapping[OperatorKey, OperatorSpec],
    *,
    existing: Mapping[PortSlot, int] | None = None,
) -> tuple[dict[PortSlot, int], dict[OperatorKey, tuple[int, ...]]]:
    port_tuuids: dict[PortSlot, int] = dict(existing or {})
    output_uuids: dict[OperatorKey, tuple[int, ...]] = {}
    next_tuuid = max(port_tuuids.values(), default=0) + 1
    for key in sorted(specs_by_key):
        values = []
        for name in specs_by_key[key].get_arguments(True, ArgumentType.DATA):
            slot = (key.spmd_id, key.op_id, "out", name)
            if slot not in port_tuuids:
                port_tuuids[slot] = next_tuuid
                next_tuuid += 1
            values.append(port_tuuids[slot])
        output_uuids[key] = tuple(values)
    return port_tuuids, output_uuids


def _transfer_identity(
    dependency_key: DependencyKey,
    transfer: TensorTransfer,
) -> tuple[DependencyKey, tuple[str, str, str, str]]:
    return (
        dependency_key,
        _tensor_transfer_key(transfer),
    )


def _allocate_dependency_codes(
    dependencies: tuple[DependencySpec, ...],
    *,
    existing: Mapping[DependencyKey, int] | None = None,
) -> dict[DependencyKey, int]:
    codes = dict(existing or {})
    next_code = max(codes.values(), default=0) + 1
    for dependency in dependencies:
        if dependency.key not in codes:
            codes[dependency.key] = next_code
            next_code += 1
    return codes


def _allocate_transfer_codes(
    dependencies: tuple[DependencySpec, ...],
    *,
    existing: Mapping[tuple[DependencyKey, tuple[str, str, str, str]], int]
    | None = None,
) -> dict[tuple[DependencyKey, tuple[str, str, str, str]], int]:
    codes = dict(existing or {})
    next_code = max(codes.values(), default=0) + 1
    for dependency in dependencies:
        for transfer in sorted(dependency.transfers, key=_tensor_transfer_key):
            identity = _transfer_identity(dependency.key, transfer)
            if identity not in codes:
                codes[identity] = next_code
                next_code += 1
    return codes


def _validate_dependencies(
    dependencies: tuple[DependencySpec, ...],
    specs_by_key: Mapping[OperatorKey, OperatorSpec],
    port_tuuids: dict[PortSlot, int],
) -> tuple[DependencySpec, ...]:
    dependencies_by_key: dict[DependencyKey, DependencySpec] = {}
    for dependency in dependencies:
        if dependency.key in dependencies_by_key:
            raise ValueError(f"Duplicate dependency {dependency.key!r}.")
        source = specs_by_key.get(dependency.source)
        if source is None:
            raise ValueError(
                f"Dependency references unknown source operator {dependency.source!r}."
            )
        target = specs_by_key.get(dependency.target)
        if target is None:
            raise ValueError(
                f"Dependency references unknown target operator {dependency.target!r}."
            )
        for transfer in dependency.transfers:
            _validate_port(dependency, transfer.source, source, output=True)
            _validate_port(dependency, transfer.target, target, output=False)
            intent = target.tensor_access[transfer.target.name]
            if intent is TensorAccessIntent.DONATE and not dependency.local_only:
                raise ValueError(
                    f"DONATE input port {dependency.target.spmd_id}."
                    f"{dependency.target.op_id}.{transfer.target.name} requires "
                    "a local_only dependency."
                )
            source_slot: PortSlot = (
                dependency.source.spmd_id,
                dependency.source.op_id,
                "out",
                transfer.source.name,
            )
            tuuid = port_tuuids[source_slot]
            target_slot: PortSlot = (
                dependency.target.spmd_id,
                dependency.target.op_id,
                "in",
                transfer.target.name,
            )
            existing = port_tuuids.get(target_slot)
            if existing is not None and existing != tuuid:
                raise ValueError(
                    f"Input port {target_slot!r} is bound to TUUID {existing}, "
                    f"not producer TUUID {tuuid}."
                )
            port_tuuids[target_slot] = tuuid
        dependencies_by_key[dependency.key] = dependency
    for dependency in dependencies_by_key.values():
        symmetric_key = dependency.symmetric_to
        if symmetric_key is None:
            continue
        symmetric = dependencies_by_key.get(symmetric_key)
        if symmetric is None:
            raise ValueError(
                f"Dependency {dependency.key!r} references unknown symmetric "
                f"dependency {symmetric_key!r}."
            )
        if symmetric.symmetric_to != dependency.key:
            raise ValueError(
                f"Symmetric dependencies {dependency.key!r} and "
                f"{symmetric_key!r} must reference each other."
            )
    return tuple(dependencies_by_key[key] for key in sorted(dependencies_by_key))


def _validate_port(
    dependency: DependencySpec,
    port: OperatorPort,
    operator: OperatorSpec,
    *,
    output: bool,
) -> None:
    arguments = operator.get_arguments(output, ArgumentType(port.kind))
    if port.name not in arguments:
        direction = "source" if output else "target"
        raise ValueError(
            f"Dependency {dependency.key!r} references unknown {direction} "
            f"{port.kind} port {port.name!r}."
        )


def _compile_output_state_names(
    specs_by_key: Mapping[OperatorKey, OperatorSpec],
    dependencies: tuple[DependencySpec, ...],
) -> dict[OperatorKey, tuple[str, ...]]:
    output_states = {
        key: tuple(spec.get_arguments(True, ArgumentType.CONTROL))
        for key, spec in specs_by_key.items()
    }
    for dependency in dependencies:
        source = specs_by_key[dependency.source]
        target = specs_by_key[dependency.target]
        required = tuple(target.get_arguments(False, ArgumentType.CONTROL))
        current = output_states[dependency.source]
        if current:
            if set(current) != set(required):
                raise ValueError(
                    f"State names mismatch in dependency {dependency.key!r}: "
                    f"source={sorted(current)}, target={sorted(required)}."
                )
            continue
        if not required:
            continue
        if source.owner is None:
            raise ValueError(
                f"Operator {dependency.source!r} requires an owner class to infer "
                "state outputs."
            )
        from fluxway.common.utils import infer_fluxway_output_fields

        output_fields = set(
            infer_fluxway_output_fields(source.owner, source.method_name)
        )
        missing = set(required) - output_fields
        if missing:
            raise ValueError(
                f"State coverage mismatch in dependency {dependency.key!r}: "
                f"missing output fields {sorted(missing)}."
            )
        output_states[dependency.source] = required
    return output_states


def _compile_operator_facts(
    specs_by_key: Mapping[OperatorKey, OperatorSpec],
    port_tuuids: Mapping[PortSlot, int],
    output_uuids: Mapping[OperatorKey, tuple[int, ...]],
    output_state_names: Mapping[OperatorKey, tuple[str, ...]],
) -> dict[OperatorKey, _CompiledOperatorFacts]:
    compiled = {}
    known_operators = frozenset(specs_by_key)
    for key, spec in specs_by_key.items():

        def input_resolver(path: str, operator: OperatorKey = key) -> int | None:
            return _resolve_port_tuuid(
                path,
                operator,
                "in",
                port_tuuids,
                known_operators,
            )

        def output_resolver(path: str, operator: OperatorKey = key) -> int | None:
            return _resolve_port_tuuid(
                path,
                operator,
                "out",
                port_tuuids,
                known_operators,
            )

        input_sharding = {
            name: sharding.resolved(input_resolver)
            for name, sharding in spec.input_sharding.items()
        }
        output_sharding = {
            name: sharding.resolved(output_resolver)
            for name, sharding in spec.output_sharding.items()
        }
        output_arguments = dict(spec.output_arguments)
        if output_state_names[key]:
            output_arguments[ArgumentType.CONTROL.value] = output_state_names[key]
        compiled[key] = _CompiledOperatorFacts(
            key=key,
            spec=spec,
            input_sharding=freeze_mapping(input_sharding),
            output_sharding=freeze_mapping(output_sharding),
            output_uuids=output_uuids[key],
            output_state_names=output_state_names[key],
            function_id=_function_id(spec),
            input_signature_id=_signature_id(spec.input_arguments),
            output_signature_id=_signature_id(output_arguments),
        )
    return compiled


def _compile_operators(
    operator_facts: Mapping[OperatorKey, _CompiledOperatorFacts],
    sharding_resolver: _ShardingReferenceResolver,
    declarations: Mapping[str, CompiledSPMDDeclaration],
) -> dict[OperatorKey, CompiledOperator]:
    return {
        key: CompiledOperator.create(
            key=facts.key,
            spec=facts.spec,
            input_sharding=facts.input_sharding,
            output_sharding=facts.output_sharding,
            output_uuids=facts.output_uuids,
            output_state_names=facts.output_state_names,
            function_id=facts.function_id,
            input_signature_id=facts.input_signature_id,
            output_signature_id=facts.output_signature_id,
            effective_input_sharding={
                port: sharding_resolver.resolve_target(operator=key, port=port)
                for port in facts.input_sharding
            },
            execution_layout=compile_execution_layout(
                declarations[key.spmd_id].mesh,
                unit_size=facts.spec.interleave_unit_size
                if facts.spec.is_interleaved
                else None,
                groups=facts.spec.execution_groups,
            ),
        )
        for key, facts in operator_facts.items()
    }


def _resolve_port_tuuid(
    path: str,
    operator: OperatorKey,
    primary_direction: Literal["in", "out"],
    port_tuuids: Mapping[PortSlot, int],
    known_operators: frozenset[OperatorKey],
) -> int | None:
    target_operator = operator
    field_name = path
    if "." in path:
        candidate_op, candidate_field = path.split(".", 1)
        candidate_key = OperatorKey(operator.spmd_id, candidate_op)
        if candidate_key in known_operators:
            target_operator = candidate_key
            field_name = candidate_field
    other_direction: Literal["in", "out"] = "out" if primary_direction == "in" else "in"
    candidate_slots: tuple[PortSlot, PortSlot] = (
        (
            target_operator.spmd_id,
            target_operator.op_id,
            primary_direction,
            field_name,
        ),
        (
            target_operator.spmd_id,
            target_operator.op_id,
            other_direction,
            field_name,
        ),
    )
    matches = {port_tuuids[slot] for slot in candidate_slots if slot in port_tuuids}
    if len(matches) > 1:
        raise ValueError(
            f"Referential path {path!r} is ambiguous for operator {operator!r}."
        )
    return next(iter(matches), None)


def _compile_dependencies(
    dependencies: tuple[DependencySpec, ...],
    declarations: Mapping[str, CompiledSPMDDeclaration],
    operators: Mapping[OperatorKey, CompiledOperator],
    port_tuuids: Mapping[PortSlot, int],
    *,
    dependency_codes: Mapping[DependencyKey, int],
    transfer_codes: Mapping[tuple[DependencyKey, tuple[str, str, str, str]], int],
) -> tuple[CompiledDependency, ...]:
    compiled = []
    for dependency in dependencies:
        code = dependency_codes[dependency.key]
        source = operators[dependency.source]
        target = operators[dependency.target]
        transfers = []
        for transfer in sorted(dependency.transfers, key=_tensor_transfer_key):
            transfers.append(
                _compile_transfer(
                    dependency,
                    transfer,
                    source,
                    target,
                    port_tuuids,
                    dependency_code=code,
                    transfer_code=transfer_codes[
                        _transfer_identity(dependency.key, transfer)
                    ],
                )
            )
        source_spmd = declarations[dependency.source.spmd_id]
        target_spmd = declarations[dependency.target.spmd_id]
        compiled.append(
            CompiledDependency.create(
                code=code,
                spec=dependency,
                source_operator=source,
                target_operator=target,
                transfers=tuple(transfers),
                source_world_size=source_spmd.world_size,
                target_world_size=target_spmd.world_size,
                source_mesh=source_spmd.mesh,
                target_mesh=target_spmd.mesh,
            )
        )
    return tuple(compiled)


def _compile_transfer(
    dependency: DependencySpec,
    transfer: TensorTransfer,
    source: CompiledOperator,
    target: CompiledOperator,
    port_tuuids: Mapping[PortSlot, int],
    *,
    dependency_code: int,
    transfer_code: int,
) -> CompiledTensorTransfer:
    logical_tuuid = port_tuuids[
        (
            dependency.source.spmd_id,
            dependency.source.op_id,
            "out",
            transfer.source.name,
        )
    ]
    try:
        source_sharding = source.output_sharding[transfer.source.name]
    except KeyError as exc:
        raise ValueError(
            f"Dependency {dependency.key!r} source port {transfer.source.name!r} "
            "has no output sharding."
        ) from exc
    try:
        target_sharding = target.input_sharding[transfer.target.name]
    except KeyError as exc:
        raise ValueError(
            f"Dependency {dependency.key!r} target port {transfer.target.name!r} "
            "has no input sharding."
        ) from exc
    return CompiledTensorTransfer(
        spec=transfer,
        transfer_code=transfer_code,
        dependency_code=dependency_code,
        logical_tuuid=logical_tuuid,
        source_sharding=source_sharding,
        target_sharding=target_sharding,
    )


def _function_id(spec: OperatorSpec) -> str:
    owner = spec.owner
    if owner is None:
        return spec.method_name
    return f"{owner.__module__}.{owner.__qualname__}.{spec.method_name}"


def _signature_id(arguments: Mapping[str, tuple[str, ...]]) -> str:
    canonical = tuple((kind, tuple(names)) for kind, names in sorted(arguments.items()))
    return sha256(repr(canonical).encode("utf-8")).hexdigest()
