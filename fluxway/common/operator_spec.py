from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
import inspect
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    field_validator,
    model_validator,
)

from fluxway.common.frozen_mapping import freeze_mapping
from fluxway.common.coherence import TensorAccessIntent
from fluxway.common.operator_policy import (
    ArgumentType,
    AsyncPolicy,
    CompiledAsyncPolicy,
    PeriodicBarrierConfig,
    StandardBarrierConfig,
)
from fluxway.common.tree_spec import TreeSpec
from fluxway.common.types.sharding import ShardingConfig, ShardingSpec


type StringTupleMap = Annotated[
    Mapping[str, tuple[str, ...]],
    AfterValidator(freeze_mapping),
    PlainSerializer(lambda value: dict(value), return_type=dict),
]
type ShardingMap = Annotated[
    Mapping[str, ShardingSpec],
    AfterValidator(freeze_mapping),
    PlainSerializer(lambda value: dict(value), return_type=dict),
]
type TreeMap = Annotated[
    Mapping[str, TreeSpec],
    AfterValidator(freeze_mapping),
    PlainSerializer(lambda value: dict(value), return_type=dict),
]
type TensorAccessMap = Annotated[
    Mapping[str, TensorAccessIntent],
    AfterValidator(freeze_mapping),
    PlainSerializer(lambda value: dict(value), return_type=dict),
]


class OperatorExecutionTrait(StrEnum):
    SYNC_COMPUTE = "sync_compute"
    ASYNC_COROUTINE = "async_coroutine"

    @classmethod
    def from_callable(cls, fn: Any) -> OperatorExecutionTrait:
        """Resolve the execution trait from any callable, safely unwrapping wrappers."""
        if fn is None:
            return cls.SYNC_COMPUTE
        unwrapped = inspect.unwrap(fn)
        if hasattr(unwrapped, "func"):
            unwrapped = inspect.unwrap(unwrapped.func)
        if inspect.iscoroutinefunction(unwrapped):
            return cls.ASYNC_COROUTINE
        return cls.SYNC_COMPUTE

    @classmethod
    def from_owner(
        cls, owner: type | None, method_name: str | None
    ) -> OperatorExecutionTrait:
        """Resolve the execution trait from an owner class and method name."""
        if owner is not None and isinstance(owner, type) and method_name:
            fn = getattr(owner, method_name, None)
            if fn is not None:
                return cls.from_callable(fn)
        return cls.SYNC_COMPUTE


class OperatorSpec(BaseModel):
    _runtime_policy: CompiledAsyncPolicy = PrivateAttr()

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    name: str = Field(min_length=1)
    owner: type | None = None
    method_name: str = Field(default="", min_length=1)
    execution_trait: OperatorExecutionTrait | None = None
    requires_worker_process: bool = False
    is_source: bool | None = None
    source: Any | None = None
    is_interleaved: bool = False
    interleave_unit_size: int = Field(default=1, gt=0)
    execution_groups: tuple[tuple[int, ...], ...] | None = None
    async_policy: AsyncPolicy = Field(default_factory=AsyncPolicy.sync)
    tensor_access: TensorAccessMap = Field(default_factory=dict)
    input_arguments: StringTupleMap = Field(default_factory=dict)
    output_arguments: StringTupleMap = Field(default_factory=dict)
    input_sharding: ShardingMap = Field(default_factory=dict)
    output_sharding: ShardingMap = Field(default_factory=dict)
    input_param_map: StringTupleMap = Field(default_factory=dict)
    output_param_map: StringTupleMap = Field(default_factory=dict)
    input_treedefs: TreeMap = Field(default_factory=dict)
    output_treedef: TreeSpec | None = None
    barrier_config: StandardBarrierConfig | PeriodicBarrierConfig = Field(
        default_factory=StandardBarrierConfig
    )

    @model_validator(mode="before")
    @classmethod
    def default_execution_trait_and_method(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            val_dict = dict(value)
            if not val_dict.get("method_name") and val_dict.get("name"):
                val_dict["method_name"] = val_dict["name"]
            if val_dict.get("execution_trait") is None:
                val_dict["execution_trait"] = OperatorExecutionTrait.from_owner(
                    val_dict.get("owner"), val_dict.get("method_name")
                )
            return val_dict
        return value

    @field_validator("owner")
    @classmethod
    def validate_owner(cls, value: type | None) -> type | None:
        if value is not None and not isinstance(value, type):
            raise TypeError("operator owner must be a class")
        return value

    @field_validator("input_arguments", "output_arguments", mode="before")
    @classmethod
    def validate_argument_map(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("tensor_access", mode="before")
    @classmethod
    def validate_tensor_access(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            return value
        return {
            name: intent
            if isinstance(intent, TensorAccessIntent)
            else TensorAccessIntent(intent)
            for name, intent in value.items()
        }

    @model_validator(mode="before")
    @classmethod
    def normalize_tensor_access(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        input_arguments = value.get("input_arguments") or {}
        data_ports = tuple(input_arguments.get(ArgumentType.DATA.value, ()))
        if not data_ports:
            data_ports = tuple(value.get("input_sharding", {}))
        declared = dict(value.get("tensor_access") or {})
        unknown = set(declared).difference(data_ports)
        if unknown:
            raise ValueError(
                f"operator {value.get('name')!r} tensor_access references unknown data "
                f"ports: {sorted(unknown)}."
            )
        normalized = {
            port: declared.get(port, TensorAccessIntent.READONLY) for port in data_ports
        }
        return {**value, "tensor_access": normalized}

    @model_validator(mode="after")
    def validate_tensor_access_contract(self) -> OperatorSpec:
        data_ports = self.get_arguments(is_output=False, arg_type=ArgumentType.DATA)
        if set(self.tensor_access) != set(data_ports):
            raise ValueError("operator tensor_access must cover every data port")
        return self

    @field_validator("input_sharding", "output_sharding", mode="before")
    @classmethod
    def validate_sharding_map(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            name: (
                spec
                if isinstance(spec, ShardingSpec)
                else (
                    ShardingSpec.model_validate(spec)
                    if isinstance(spec, Mapping) and "routing" in spec
                    else ShardingConfig.model_validate(spec).to_spec()
                )
            )
            for name, spec in value.items()
        }

    @field_validator("input_treedefs", mode="before")
    @classmethod
    def validate_input_trees(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            name: (
                tree
                if isinstance(tree, TreeSpec)
                else (
                    TreeSpec.model_validate(tree)
                    if isinstance(tree, Mapping) and "type" in tree
                    else TreeSpec.from_runtime(tree)
                )
            )
            for name, tree in value.items()
        }

    @field_validator("output_treedef", mode="before")
    @classmethod
    def validate_output_tree(cls, value: Any) -> Any:
        if value is None or isinstance(value, TreeSpec):
            return value
        if isinstance(value, Mapping) and "type" in value:
            return TreeSpec.model_validate(value)
        return TreeSpec.from_runtime(value)

    def model_post_init(self, __context: Any) -> None:
        self._runtime_policy = self.async_policy.compile()

    @model_validator(mode="after")
    def validate_source_descriptor(self) -> OperatorSpec:
        if self.source is None:
            return self
        if self.is_source is not True:
            raise ValueError("operator source descriptor requires is_source=True")
        if not callable(getattr(self.source, "build_component", None)):
            raise TypeError(
                "operator source descriptor must implement build_component(context)"
            )
        capability = getattr(self.source, "requires_source_control_runtime", None)
        if not isinstance(capability, bool):
            raise TypeError(
                "operator source descriptor must declare boolean "
                "requires_source_control_runtime"
            )
        if capability:
            required_scan_result_capabilities = ("build_source_scan_binding",)
            missing = tuple(
                name
                for name in required_scan_result_capabilities
                if not callable(getattr(self.source, name, None))
            )
            if missing:
                raise TypeError(
                    "operator source descriptor with "
                    "requires_source_control_runtime=True must implement bounded "
                    f"scan capabilities: {missing!r}"
                )
        return self

    @property
    def has_source_descriptor(self) -> bool:
        return self.source is not None

    @property
    def is_coroutine_function(self) -> bool:
        return self.execution_trait == OperatorExecutionTrait.ASYNC_COROUTINE

    @property
    def requires_source_control_runtime(self) -> bool:
        if self.source is None:
            return False
        return self.source.requires_source_control_runtime

    @property
    def runtime_policy(self) -> CompiledAsyncPolicy:
        return self._runtime_policy

    def get_arguments(self, is_output: bool, arg_type: ArgumentType) -> list[str]:
        arguments = self.output_arguments if is_output else self.input_arguments
        if arg_type.value in arguments:
            return list(arguments[arg_type.value])
        if arg_type == ArgumentType.DATA:
            sharding = self.output_sharding if is_output else self.input_sharding
            return list(sharding)
        return []

    def state_names(self, *, is_output: bool) -> list[str]:
        return self.get_arguments(
            is_output=is_output,
            arg_type=ArgumentType.CONTROL,
        )

    @property
    def owner_class_module(self) -> str:
        return "" if self.owner is None else self.owner.__module__

    @property
    def owner_class_qualname(self) -> str:
        return "" if self.owner is None else self.owner.__qualname__

    def updated(self, **updates: Any) -> OperatorSpec:
        values = {name: getattr(self, name) for name in self.__class__.model_fields}
        values.update(updates)
        return self.__class__.model_validate(values)

    def resolve_sharding_references(
        self,
        *,
        input_resolver: Any,
        output_resolver: Any,
        required: bool = True,
    ) -> OperatorSpec:
        input_sharding = {
            name: spec.resolved(input_resolver, required=required)
            for name, spec in self.input_sharding.items()
        }
        output_sharding = {
            name: spec.resolved(output_resolver, required=required)
            for name, spec in self.output_sharding.items()
        }
        if all(
            input_sharding[name] is spec for name, spec in self.input_sharding.items()
        ) and all(
            output_sharding[name] is spec for name, spec in self.output_sharding.items()
        ):
            return self
        return self.updated(
            input_sharding=input_sharding,
            output_sharding=output_sharding,
        )
