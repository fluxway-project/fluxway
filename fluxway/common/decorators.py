"""
Decorators for Fluxway framework.

Provides @operator and @data_type decorators to manage metadata and transport.
"""

import contextvars
import functools
import inspect
import logging
from contextlib import contextmanager
from decimal import Decimal
from dataclasses import dataclass, field
from typing import Annotated, Any, TYPE_CHECKING, TypeVar, Union, get_type_hints

if TYPE_CHECKING:
    from fluxway.common.types import SPMDConfig
from collections.abc import Callable, Mapping
from fluxway.common.operator_initialization import operator_initialization_arguments
from fluxway.common.utils.tree import tree_flatten_type, _is_fluxway_data_type
from fluxway.resources import FrozenResourceVector, normalize_resource_vector
from fluxway.common.operator_instances import OperatorInstanceContainer

logger = logging.getLogger(__name__)


@dataclass
class _OperatorContextState:
    static_args: dict[str, Any] = field(default_factory=dict)
    spmd_id: str | None = None
    operator_name: str | None = None
    rank: int = 0
    world_size: int = 1
    ctx: Any = None
    control_plane: Any = None
    local_rank: int | None = None
    local_world_size: int | None = None
    node_rank: int = 0
    num_nodes: int = 1
    rank0_address: str | None = None
    ray_component_limits: Mapping[str, Decimal] = field(
        default_factory=lambda: FrozenResourceVector({})
    )
    instance_container: OperatorInstanceContainer = field(
        default_factory=OperatorInstanceContainer
    )
    lane_runner: Any = None


_DEFAULT_CONTEXT_STATE = _OperatorContextState()

_operator_context_var: contextvars.ContextVar[_OperatorContextState] = (
    contextvars.ContextVar("operator_context", default=_DEFAULT_CONTEXT_STATE)
)


@contextmanager
def operator_context(
    static_args: dict[str, Any],
    rank: int,
    world_size: int,
    spmd_id: str | None = None,
    operator_name: str | None = None,
    operator_ctx: Any | None = None,
    control_plane: Any | None = None,
    local_rank: int | None = None,
    local_world_size: int | None = None,
    node_rank: int = 0,
    num_nodes: int = 1,
    rank0_address: str | None = None,
    ray_component_limits: Mapping[str, Decimal] | None = None,
    instance_container: OperatorInstanceContainer | None = None,
    lane_runner: Any = None,
):
    """Context manager to inject operator runtime context into current coroutine/thread.

    Uses contextvars.ContextVar so each asyncio Task and each to_thread worker
    gets an isolated snapshot — no cross-coroutine state pollution.
    """
    normalized_ray_component_limits = (
        FrozenResourceVector({})
        if ray_component_limits is None
        else ray_component_limits
        if isinstance(ray_component_limits, FrozenResourceVector)
        else FrozenResourceVector(normalize_resource_vector(ray_component_limits))
    )
    state = _OperatorContextState(
        static_args=static_args,
        spmd_id=spmd_id,
        operator_name=operator_name,
        rank=rank,
        world_size=world_size,
        ctx=operator_ctx,
        control_plane=control_plane,
        local_rank=local_rank,
        local_world_size=local_world_size,
        node_rank=node_rank,
        num_nodes=num_nodes,
        rank0_address=rank0_address,
        ray_component_limits=normalized_ray_component_limits,
        instance_container=(
            OperatorInstanceContainer()
            if instance_container is None
            else instance_container
        ),
        lane_runner=lane_runner,
    )
    token = _operator_context_var.set(state)
    try:
        yield state
    finally:
        _operator_context_var.reset(token)


def get_current_context() -> _OperatorContextState:
    """Return the operator context state for the current coroutine/thread."""
    return _operator_context_var.get()


def _unwrap_union_type(tp: Any) -> Any:
    import types as _types

    origin = getattr(tp, "__origin__", None)
    if origin is Union or isinstance(tp, _types.UnionType):
        for arg in getattr(tp, "__args__", []):
            if arg is not type(None) and _is_fluxway_data_type(arg):
                return arg
    return tp


def _get_fluxway_streams(tp: Any, prefix: str = "") -> tuple[list[str], list[str]]:
    from fluxway.common.types.configs import ControlFlow

    if not _is_fluxway_data_type(tp):
        return [], []
    try:
        hints = get_type_hints(_unwrap_union_type(tp), include_extras=True)
    except Exception:
        return [], []

    data_fields: list[str] = []
    control_fields: list[str] = []
    for name, hint in hints.items():
        if name.startswith("_"):
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        is_control = hasattr(hint, "__metadata__") and any(
            m is ControlFlow for m in hint.__metadata__
        )
        if _is_fluxway_data_type(hint):
            d, c = _get_fluxway_streams(hint, prefix=full_name)
            data_fields.extend(d)
            control_fields.extend(c)
        elif is_control:
            control_fields.append(full_name)
        else:
            data_fields.append(full_name)
    return data_fields, control_fields


def _build_hierarchical_map(
    tp: Any, prefix: str = "", include_root: bool = False
) -> dict[str, list[str]]:
    from fluxway.common.types.configs import ControlFlow

    if not _is_fluxway_data_type(tp):
        return {}
    try:
        hints = get_type_hints(_unwrap_union_type(tp), include_extras=True)
    except Exception:
        return {}

    mapping: dict[str, list[str]] = {}
    all_primitive_fields: list[str] = []
    for name, hint in hints.items():
        if name.startswith("_"):
            continue
        full_name = f"{prefix}.{name}" if prefix else name
        is_control = hasattr(hint, "__metadata__") and any(
            m is ControlFlow for m in hint.__metadata__
        )
        if _is_fluxway_data_type(hint):
            sub_map = _build_hierarchical_map(hint, prefix=full_name, include_root=True)
            mapping.update(sub_map)
            all_primitive_fields.extend(sub_map.get(full_name, []))
        elif not is_control:
            mapping[full_name] = [full_name]
            all_primitive_fields.append(full_name)

    if include_root and prefix:
        mapping[prefix] = all_primitive_fields
    return mapping


def _validate_unique(lst: list, context: str) -> list:
    seen: set = set()
    for x in lst:
        if x in seen:
            raise ValueError(
                f"Fluxway Operator Error: Duplicate argument '{x}' in {context}.\n"
                f"Fix: Remove the duplicate entry from your @operator configuration."
            )
        seen.add(x)
    return lst


def _classify_sig_params(
    sig: inspect.Signature,
    static_argument: list[str] | None,
    func_qualname: str,
    func_name: str,
) -> tuple[list[str], bool]:
    data_params: list[str] = []
    has_var_keyword = False
    for p_name, p in sig.parameters.items():
        if p_name in ("self", "cls"):
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if p.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        tp = p.annotation
        if tp is inspect.Parameter.empty:
            if static_argument and p_name in static_argument:
                continue
            raise TypeError(
                f"Fluxway Operator Error: Argument '{p_name}' in '{func_qualname}' missing type annotation.\n"
                f"Fix: Add a type hint, e.g., 'def {func_name}(self, batch: MyDataType):'"
            )
        if _is_fluxway_data_type(tp):
            data_params.append(p_name)
    return data_params, has_var_keyword


def _infer_input_streams(
    sig: inspect.Signature,
    data_params: list[str],
    use_prefix: bool,
) -> tuple[list[str], list[str], dict, dict]:
    all_data_fields: list[str] = []
    all_control_fields: list[str] = []
    input_param_map: dict = {}
    input_treedefs: dict = {}

    for p_name, p in sig.parameters.items():
        if p_name in ("self", "cls"):
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        tp = p.annotation
        if _is_fluxway_data_type(tp):
            data_fields, control_fields = _get_fluxway_streams(tp)
            d_f_final = (
                [f"{p_name}.{df}" for df in data_fields] if use_prefix else data_fields
            )
            c_f_final = (
                [f"{p_name}.{cf}" for cf in control_fields]
                if use_prefix
                else control_fields
            )
            param_map = _build_hierarchical_map(tp, prefix=p_name if use_prefix else "")
            param_map[p_name] = d_f_final + c_f_final
            input_param_map.update(param_map)
            input_treedefs[p_name] = tree_flatten_type(tp)
        else:
            d_f_final, c_f_final = [p_name], []
            input_param_map[p_name] = [p_name]
            input_treedefs[p_name] = None

        for df in d_f_final:
            if df not in all_data_fields:
                all_data_fields.append(df)
        for cf in c_f_final:
            if cf not in all_control_fields:
                all_control_fields.append(cf)

    return all_data_fields, all_control_fields, input_param_map, input_treedefs


def _resolve_sharding_spec(
    sharding_val: Any,
    data_fields: list[str],
    control_fields: list[str],
    prefix: str,
    operator_name: str,
) -> dict:
    if not isinstance(sharding_val, dict):
        return dict.fromkeys(data_fields, sharding_val)
    if "type" in sharding_val and "axis" in sharding_val:
        return dict.fromkeys(data_fields, sharding_val)

    resolved: dict = {}
    for k, v in sharding_val.items():
        full_k = f"{prefix}.{k}" if prefix else k
        sub_data = [f for f in data_fields if f == full_k or f.startswith(full_k + ".")]
        sub_control = [
            f for f in control_fields if f == full_k or f.startswith(full_k + ".")
        ]
        if not sub_data and not sub_control:
            if k in data_fields:
                resolved[k] = v
                continue
            raise ValueError(
                f"Fluxway Operator Error: Unknown field '{k}' at path '{prefix}' for operator '{operator_name}'.\n"
                f"Available Data fields here: {[f[len(prefix) + 1 :] if prefix else f for f in data_fields]}\n"
                f"Fix: Check for typos in your sharding dictionary or ensure the field is part of the @data_type."
            )
        if sub_control and not sub_data:
            raise ValueError(
                f"Fluxway Operator Error: Control field '{full_k}' does not support sharding.\n"
                f"Fix: Control fields (marked with ControlFlow) are broadcasted; remove them from your sharding configuration."
            )
        resolved.update(
            _resolve_sharding_spec(v, sub_data, sub_control, full_k, operator_name)
        )
    return resolved


def _resolve_input_sharding(
    input_sharding: dict | None,
    data_params: list[str],
    sig: inspect.Signature,
    use_prefix: bool,
    final_tensors: list[str],
    final_states: list[str],
    has_var_keyword: bool,
    input_param_map: dict,
    input_treedefs: dict,
    operator_name: str,
) -> tuple[dict, list[str]]:
    if not input_sharding:
        return {}, final_tensors

    param_to_fields: dict = {}
    for p_name in data_params:
        tp = sig.parameters[p_name].annotation
        d_f, c_f = _get_fluxway_streams(tp)
        if use_prefix:
            d_f = [f"{p_name}.{f}" for f in d_f]
            c_f = [f"{p_name}.{f}" for f in c_f]
        param_to_fields[p_name] = (d_f, c_f)

    resolved: dict = {}
    for k, v in input_sharding.items():
        if k in param_to_fields:
            d_f, c_f = param_to_fields[k]
            resolved.update(
                _resolve_sharding_spec(
                    v, d_f, c_f, k if use_prefix else "", operator_name
                )
            )
        elif k in final_tensors:
            resolved[k] = v
        elif k in final_states:
            raise ValueError(
                f"Fluxway Operator Error: Control field '{k}' does not support sharding."
            )
        elif has_var_keyword:
            final_tensors.append(k)
            resolved[k] = v
            input_param_map[k] = [k]
            input_treedefs[k] = None
        else:
            raise ValueError(
                f"Fluxway Operator Error: Unknown sharding key '{k}' for operator '{operator_name}'."
            )
    return resolved, final_tensors


def _resolve_output_sharding(
    output_sharding: dict | None,
    return_tp: Any,
    operator_name: str,
) -> tuple[dict, list[str], list[str], dict, Any]:
    out_data_fields: list[str] = []
    out_control_fields: list[str] = []
    output_param_map: dict = {}
    meta_output_treedef = None

    # Unwrap Optional / X | None so all downstream code sees the concrete type.
    return_tp = _unwrap_union_type(return_tp)

    if return_tp is not None and return_tp is not inspect.Signature.empty:
        if _is_fluxway_data_type(return_tp):
            out_data_fields, out_control_fields = _get_fluxway_streams(return_tp)
            output_param_map = _build_hierarchical_map(return_tp, include_root=False)
            output_param_map["return"] = out_data_fields + out_control_fields
        elif output_sharding:
            out_data_fields = list(output_sharding.keys())
            output_param_map["return"] = out_data_fields

    if not output_sharding:
        return (
            {},
            out_data_fields,
            out_control_fields,
            output_param_map,
            meta_output_treedef,
        )

    resolved: dict = {}
    if _is_fluxway_data_type(return_tp):
        if "type" in output_sharding and "axis" in output_sharding:
            resolved = dict.fromkeys(out_data_fields, output_sharding)
        else:
            return_type_name = return_tp.__name__.lower()  # type: ignore
            remaining: dict = {}
            for k, v in output_sharding.items():
                if k.lower() in ("return", return_type_name):
                    resolved.update(
                        _resolve_sharding_spec(
                            v, out_data_fields, out_control_fields, "", operator_name
                        )
                    )
                else:
                    remaining[k] = v
            if remaining:
                resolved.update(
                    _resolve_sharding_spec(
                        remaining,
                        out_data_fields,
                        out_control_fields,
                        "",
                        operator_name,
                    )
                )
    else:
        resolved = output_sharding
        if isinstance(output_sharding, dict):
            sharding_keys = list(output_sharding)
            meta_output_treedef = ("dict", sharding_keys, [None] * len(sharding_keys))

    return (
        resolved,
        out_data_fields,
        out_control_fields,
        output_param_map,
        meta_output_treedef,
    )


_STATIC_CONTEXT_ATTR_NAMES: frozenset[str] = frozenset(
    {
        "rank",
        "world_size",
        "local_rank",
        "local_world_size",
        "node_rank",
        "num_nodes",
        "static_args",
        "rank0_address",
        "ray_component_limits",
    }
)
_DYNAMIC_CONTEXT_ATTR_NAMES: frozenset[str] = frozenset({"ctx"})
_CONTEXT_ATTR_NAMES: frozenset[str] = (
    _STATIC_CONTEXT_ATTR_NAMES | _DYNAMIC_CONTEXT_ATTR_NAMES
)
_WORKER_PROCESS_PROTOCOL_ATTR = "__fluxway_process_affine__"


def requires_worker_process(cls: type) -> type:
    """Run a stateful operator class inside its owning SPMD worker process.

    Use this for operators that initialize process-local systems such as
    ``jax.distributed.initialize``. The marker declares that construction
    depends on worker-process topology being available before execution starts.
    """
    setattr(cls, _WORKER_PROCESS_PROTOCOL_ATTR, True)
    return cls


def _is_worker_process_required(cls: type) -> bool:
    return bool(getattr(cls, _WORKER_PROCESS_PROTOCOL_ATTR, False))


def _source_mentions_jax_distributed_initialize(cls: type) -> bool:
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        source = ""
    if "jax.distributed.initialize" in source:
        return True
    for value in cls.__dict__.values():
        fn = value
        if isinstance(value, (staticmethod, classmethod)):
            fn = value.__func__
        code = getattr(fn, "__code__", None)
        if code is None:
            continue
        names = set(code.co_names)
        if {"jax", "distributed", "initialize"}.issubset(names):
            return True
    return False


def _validate_worker_process_marker(cls: type) -> None:
    if _is_worker_process_required(cls):
        return
    if not _source_mentions_jax_distributed_initialize(cls):
        return
    raise RuntimeError(
        "Fluxway Worker Process Error: "
        f"{cls.__module__}.{cls.__qualname__} calls jax.distributed.initialize "
        "but is not marked with @fluxway.requires_worker_process.\n"
        "Fix: add @fluxway.requires_worker_process above the class definition. "
        "This declares that operator construction depends on worker-process "
        "topology initialization."
    )


def _inject_context_properties(cls: type) -> None:
    """Inject ContextVar-backed property only for per-invocation dynamic attributes.

    Static attributes (rank, world_size, local_rank, …) are fixed for the
    lifetime of a worker actor and are stored as plain instance attributes in
    ``_get_or_create_instance``.  Only ``ctx`` (which carries per-step state
    like ``step``) varies between concurrent coroutine invocations and
    therefore needs a ContextVar property for isolation.
    """
    sentinel = f"_fluxway_ctx_props_{cls.__qualname__}"
    if getattr(cls, sentinel, False):
        return
    for attr in _DYNAMIC_CONTEXT_ATTR_NAMES:
        prop = property(lambda self, a=attr: getattr(get_current_context(), a))
        setattr(cls, attr, prop)
    setattr(cls, sentinel, True)


class _OperatorWrapper:
    def __init__(self, func: Callable, meta: Any):
        self.func = func
        self._fluxway_operator_spec = meta
        self._cls = None
        functools.update_wrapper(self, func)

    def __get__(self, instance, owner):
        if self._cls is None:
            self._cls = owner
        if not hasattr(owner, "_fluxway_ctx_injected"):
            _inject_context_properties(owner)
            owner._fluxway_ctx_injected = True

        def _get_or_create_instance():
            ctx = get_current_context()

            def factory() -> Any:
                # Write static attributes BEFORE __init__ so lifecycle hooks
                # can use them without a context.
                inst = owner.__new__(owner)
                for attr in _STATIC_CONTEXT_ATTR_NAMES:
                    setattr(inst, attr, getattr(ctx, attr))
                init_kwargs = operator_initialization_arguments(
                    owner, ctx.static_args or {}
                )
                inst.__init__(**init_kwargs)
                return inst

            return ctx.instance_container.get_or_create(owner, factory)

        if inspect.iscoroutinefunction(self.func):

            @functools.wraps(self.func)
            async def async_bound_method(*args, **kwargs):
                return await self.func(_get_or_create_instance(), *args, **kwargs)

            async_bound_method._fluxway_operator_spec = self._fluxway_operator_spec  # type: ignore
            return async_bound_method

        @functools.wraps(self.func)
        def bound_method(*args, **kwargs):
            return self.func(_get_or_create_instance(), *args, **kwargs)

        bound_method._fluxway_operator_spec = self._fluxway_operator_spec  # type: ignore
        return bound_method

    def __call__(self, *args, **kwargs):
        # Fallback if called directly without binding (should not happen in fluxway)
        raise RuntimeError(
            "Fluxway operators must be called as class/instance methods."
        )


def operator(
    name: str | Callable | None = None,
    input_sharding: dict[str, Any] | None = None,
    output_sharding: dict[str, Any] | None = None,
    control_argument: list[str] | None = None,
    data_argument: list[str] | None = None,
    static_argument: list[str] | None = None,
    periodic_barrier: int | None = None,
    is_source: bool | None = None,
    source: Any | None = None,
    barrier_config: Any | None = None,
    async_policy: Any | None = None,
    tensor_access: Mapping[str, Any] | None = None,
) -> "Any":
    """
    Mark a function as a Fluxway operator with automatic metadata inference.

    The recommended async interface is ``async_policy``.  Use
    ``AsyncPolicy.sync()`` for dense/default synchronous training,
    ``AsyncPolicy.pipelined(...)`` for exact-input local pipelining, and
    ``AsyncPolicy.bounded_stale(max_staleness=N)`` for bounded-lag async
    execution.  Tensors and state remain exact-step inputs; bounded stale never
    substitutes data from another step.
    """
    if callable(name):
        func = name
        return _create_operator_wrapper(
            func,
            func.__name__,
            input_sharding,
            output_sharding,
            control_argument,
            data_argument,
            static_argument,
            periodic_barrier,
            is_source,
            source,
            barrier_config,
            async_policy,
            tensor_access,
        )

    name_str: str | None = name

    def decorator(func: Callable):
        return _create_operator_wrapper(
            func,
            name_str or func.__name__,
            input_sharding,
            output_sharding,
            control_argument,
            data_argument,
            static_argument,
            periodic_barrier,
            is_source,
            source,
            barrier_config,
            async_policy,
            tensor_access,
        )

    return decorator


def _create_operator_wrapper(
    func,
    name: str,
    input_sharding,
    output_sharding,
    control_argument,
    data_argument,
    static_argument,
    periodic_barrier,
    is_source,
    source,
    barrier_config=None,
    async_policy=None,
    tensor_access=None,
):
    from fluxway.common.operator_policy import (
        ArgumentType,
        AsyncPolicy,
        StandardBarrierConfig,
        PeriodicBarrierConfig,
    )
    from fluxway.common.operator_spec import OperatorExecutionTrait, OperatorSpec

    sig = inspect.signature(func)
    data_params, has_var_keyword = _classify_sig_params(
        sig, static_argument, func.__qualname__, func.__name__
    )
    use_prefix = len(data_params) > 1

    all_data_fields, all_control_fields, input_param_map, input_treedefs = (
        _infer_input_streams(sig, data_params, use_prefix)
    )

    final_tensors = (
        _validate_unique(data_argument, "data_argument")
        if data_argument is not None
        else all_data_fields
    )
    final_states = (
        _validate_unique(control_argument, "control_argument")
        if control_argument is not None
        else all_control_fields
    )
    final_statics = (
        _validate_unique(static_argument, "static_argument")
        if static_argument is not None
        else []
    )

    if has_var_keyword:
        for k in final_tensors:
            input_param_map.setdefault(k, [k])
            input_treedefs.setdefault(k, None)

    resolved_input, final_tensors = _resolve_input_sharding(
        input_sharding,
        data_params,
        sig,
        use_prefix,
        final_tensors,
        final_states,
        has_var_keyword,
        input_param_map,
        input_treedefs,
        name,
    )

    return_tp = sig.return_annotation
    (
        resolved_output,
        out_data_fields,
        out_control_fields,
        output_param_map,
        meta_output_treedef,
    ) = _resolve_output_sharding(output_sharding, return_tp, name)

    if periodic_barrier is not None and len(final_tensors) + len(final_states) != 1:
        raise ValueError(
            f"Fluxway Operator Error: Periodic operator '{name}' must have exactly one input field "
            f"(Data or Control), but found {len(final_tensors)} Data fields and {len(final_states)} "
            f"Control fields.\nFix: Use a @data_type with exactly one field for streaming/periodic processing."
        )
    normalized_async_policy = (
        AsyncPolicy.sync() if async_policy is None else async_policy
    )
    if not isinstance(normalized_async_policy, AsyncPolicy):
        raise ValueError(
            f"Fluxway Operator Error: async_policy for operator '{name}' must be an AsyncPolicy, "
            f"got {type(normalized_async_policy).__name__}."
        )
    runtime_policy = normalized_async_policy.compile()
    _is_class_method = "." in func.__qualname__
    if runtime_policy.state_access == "thread_safe" and not _is_class_method:
        raise ValueError(
            f"Fluxway Operator Error: state_access='thread_safe' for operator "
            f"'{name}' is only valid on class methods with an explicit state owner."
        )

    if _is_fluxway_data_type(return_tp):
        output_args: dict | None = {
            ArgumentType.DATA.value: out_data_fields,
            ArgumentType.CONTROL.value: out_control_fields,
        }
    elif resolved_output:
        output_args = {ArgumentType.DATA.value: tuple(resolved_output.keys())}
    else:
        output_args = None

    if is_source or source is not None:
        execution_trait = OperatorExecutionTrait.IO_SOURCE
    else:
        func_unwrap = func
        while hasattr(func_unwrap, "__wrapped__"):
            func_unwrap = func_unwrap.__wrapped__
        if hasattr(func_unwrap, "func"):
            func_unwrap = func_unwrap.func
        if inspect.iscoroutinefunction(func_unwrap):
            execution_trait = OperatorExecutionTrait.ASYNC_COROUTINE
        else:
            execution_trait = OperatorExecutionTrait.SYNC_COMPUTE

    spec = OperatorSpec(
        name=name,
        owner=None,
        method_name=func.__name__,
        execution_trait=execution_trait,
        is_source=is_source,
        source=source,
        async_policy=normalized_async_policy,
        tensor_access={} if tensor_access is None else tensor_access,
        input_arguments={
            ArgumentType.CONTROL.value: tuple(final_states),
            ArgumentType.DATA.value: tuple(final_tensors),
            ArgumentType.STATIC.value: tuple(final_statics),
        },
        output_arguments={} if output_args is None else output_args,
        input_sharding=resolved_input,
        output_sharding=resolved_output,
        input_param_map=input_param_map,
        output_param_map=output_param_map,
        input_treedefs=input_treedefs,
        output_treedef=(
            meta_output_treedef
            if meta_output_treedef is not None
            else (
                tree_flatten_type(return_tp)
                if _is_fluxway_data_type(return_tp)
                else None
            )
        ),
        barrier_config=(
            barrier_config
            if barrier_config
            else (
                PeriodicBarrierConfig(n=periodic_barrier)
                if periodic_barrier is not None
                else StandardBarrierConfig()
            )
        ),
    )

    wrapper = _OperatorWrapper(func, spec)

    # Expose meta on the wrapper for framework inspection
    wrapper._fluxway_operator_spec = spec

    # We return the descriptor directly. It will naturally act like a staticmethod
    # while gaining access to the owner class via the __get__ protocol.
    return wrapper


def data_type():
    """
    Mark a class as a Fluxway transportable data type.
    Enforces flattening and adds as_dict() for standard access.
    """
    T = TypeVar("T")

    def decorator(cls: type[T]) -> type[T]:
        try:
            hints = get_type_hints(cls, include_extras=True)
        except Exception as e:
            raise TypeError(
                f"Fluxway Data Error: Cannot inspect type hints for '{cls.__name__}': {e}"
            ) from e

        for name, tp in hints.items():
            if name.startswith("_"):
                continue
            base_tp = tp
            if hasattr(tp, "__origin__") and tp.__origin__ is Annotated:
                base_tp = tp.__args__[0]
            origin = getattr(base_tp, "__origin__", base_tp)
            if origin is dict:
                raise TypeError(
                    f"Fluxway Data Error: Class '{cls.__name__}' field '{name}' uses 'dict', which is not supported.\n"
                    f"Fix: Use a @data_type decorated dataclass instead of a dictionary."
                )

        cls.is_fluxway_data = classmethod(lambda c: True)  # type: ignore

        if not hasattr(cls, "as_dict"):

            def as_dict(self):
                return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

            cls.as_dict = as_dict  # type: ignore

        return cls

    return decorator


def spmd(
    resource_profile: Any | None = None,
    runtime_env: dict[str, Any] | None = None,
    driver_runtime_env: dict[str, Any] | None = None,
    mesh_axis_names: tuple[str, ...] | None = None,
    mesh_axis_shapes: tuple[int, ...] | None = None,
    checkpoint_components: tuple[Any, ...] = (),
    heartbeat_interval_s: float = 5.0,
    execution_progress_warning_s: float = 90.0,
    heartbeat_timeout_s: float = 90.0,
):
    """
    Configure a class for SPMD distributed execution.

    Args:
        resource_profile: Per-worker Fluxway actor resources plus optional Ray
                          component soft limits.
        runtime_env: Ray runtime_env dict for Driver/Worker actors.
                     Example: {"env_vars": {"JAX_PLATFORMS": "gpu"}}
        driver_runtime_env: Dedicated Ray runtime_env for the SPMDDriver actor.
                            When None, the framework applies a safe default that
                            hides GPUs from the driver process.  Override this
                            only if you intentionally want the driver to see
                            specific resources (rare).
        mesh_axis_names: Axis names for the execution mesh.
        mesh_axis_shapes: Shape of the execution mesh.
    """
    from fluxway.resources import ResourceProfile
    from fluxway.common.mesh import DeviceMesh

    T = TypeVar("T")

    if resource_profile is not None and not isinstance(
        resource_profile, ResourceProfile
    ):
        raise TypeError("resource_profile must be a ResourceProfile")

    if not isinstance(checkpoint_components, tuple):
        raise TypeError("checkpoint_components must be a tuple")
    from fluxway.control_plane.checkpointer.component import (
        CheckpointComponentRegistration,
    )

    invalid_components = [
        component
        for component in checkpoint_components
        if not isinstance(component, CheckpointComponentRegistration)
    ]
    if invalid_components:
        raise TypeError(
            "checkpoint_components must contain CheckpointComponentRegistration instances"
        )
    component_ids = [
        component.descriptor.component_id for component in checkpoint_components
    ]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("checkpoint_components must have unique component IDs")

    mesh = None
    if mesh_axis_names and mesh_axis_shapes:
        mesh = DeviceMesh(axis_names=mesh_axis_names, device_shape=mesh_axis_shapes)

    def decorator(cls: type[T]) -> type[T]:
        cls._resource_profile = resource_profile  # type: ignore
        cls._mesh = mesh  # type: ignore
        cls._runtime_env = runtime_env  # type: ignore
        cls._driver_runtime_env = driver_runtime_env  # type: ignore
        cls._checkpoint_components = checkpoint_components  # type: ignore

        cls.get_resource_profile = classmethod(lambda c: c._resource_profile)  # type: ignore
        cls.get_mesh = classmethod(lambda c: c._mesh)  # type: ignore
        cls.get_runtime_env = classmethod(lambda c: c._runtime_env)  # type: ignore
        cls.get_driver_runtime_env = classmethod(lambda c: c._driver_runtime_env)  # type: ignore
        cls.get_num_workers = classmethod(lambda c: c._mesh.size if c._mesh else 1)  # type: ignore
        cls.is_fluxway_spmd = classmethod(lambda c: True)  # type: ignore
        _inject_context_properties(cls)

        original_new = cls.__new__

        def __new__(cls_obj, *args, **kwargs):
            from fluxway.control_plane.graph.functional import FunctionalGraph

            current_graph = FunctionalGraph.current()
            if current_graph is not None:
                name = kwargs.pop("name", cls_obj.__name__)
                return current_graph.node(name, cls_obj, *args, **kwargs)

            if original_new is object.__new__:
                return original_new(cls_obj)
            return original_new(cls_obj, *args, **kwargs)

        cls.__new__ = __new__  # type: ignore

        def get_spmd_config(c) -> "SPMDConfig":
            from fluxway.common.types import SPMDConfig
            from fluxway.common.operator_spec import (
                OperatorExecutionTrait,
                OperatorSpec,
            )

            # Pass 1: walk the full MRO to collect operator metadata from _OperatorWrapper.
            # A subclass may override a method with a plain function, losing the wrapper.
            # We scan from the bottom of MRO up so that the most-derived _OperatorWrapper
            # (if any) wins, but we never discard metadata just because the attr was already
            # seen as a plain function in a more-derived class.
            wrapper_specs: dict[str, tuple[str, OperatorSpec, type]] = {}
            for klass in reversed(c.__mro__):
                for attr_name, attr in klass.__dict__.items():
                    if not isinstance(attr, _OperatorWrapper):
                        continue
                    spec = attr._fluxway_operator_spec
                    if not isinstance(spec, OperatorSpec):
                        raise TypeError(
                            f"operator {c.__qualname__}.{attr_name} does not carry "
                            "an OperatorSpec"
                        )
                    wrapper_specs[attr_name] = (spec.name, spec, klass)

            # Pass 2: for each discovered operator, resolve the attr_name that the final
            # class actually exposes (may be a plain override — that is fine, the framework
            # only needs to know the correct attr_name for dispatch and the metadata).
            from fluxway.reader.base_reader import BaseReader

            is_reader = issubclass(c, BaseReader)
            operator_specs = {}
            for attr_name, (op_name, spec, owner_cls) in wrapper_specs.items():
                updates: dict[str, Any] = {
                    "owner": c,
                    "method_name": attr_name,
                    "requires_worker_process": _is_worker_process_required(owner_cls),
                }
                if is_reader or spec.is_source or spec.source is not None:
                    updates["execution_trait"] = OperatorExecutionTrait.IO_SOURCE
                else:
                    actual_fn = getattr(c, attr_name)
                    func_unwrap = actual_fn
                    while hasattr(func_unwrap, "__wrapped__"):
                        func_unwrap = func_unwrap.__wrapped__
                    if hasattr(func_unwrap, "func"):
                        func_unwrap = func_unwrap.func
                    if inspect.iscoroutinefunction(func_unwrap):
                        updates["execution_trait"] = (
                            OperatorExecutionTrait.ASYNC_COROUTINE
                        )
                    else:
                        updates["execution_trait"] = OperatorExecutionTrait.SYNC_COMPUTE
                operator_specs[op_name] = spec.model_copy(update=updates)

            return SPMDConfig(
                mesh=c._mesh,
                operator_specs=operator_specs,
                runtime_env=c._runtime_env,
                driver_runtime_env=c._driver_runtime_env,
                checkpoint_components=c._checkpoint_components,
                heartbeat_interval_s=heartbeat_interval_s,
                execution_progress_warning_s=execution_progress_warning_s,
                heartbeat_timeout_s=heartbeat_timeout_s,
            )

        cls.get_spmd_config = classmethod(get_spmd_config)  # type: ignore
        return cls

    return decorator


def spmd_source(
    resource_profile: Any | None = None,
    mesh_axis_names: tuple[str, ...] | None = None,
    mesh_axis_shapes: tuple[int, ...] | None = None,
    interleaved: bool = False,
    interleave_unit_size: int = 1,
    execution_groups: tuple[tuple[int, ...], ...] | None = None,
    source: Any | None = None,
    runtime_env: dict[str, Any] | None = None,
    driver_runtime_env: dict[str, Any] | None = None,
    checkpoint_components: tuple[Any, ...] = (),
    heartbeat_interval_s: float = 5.0,
    execution_progress_warning_s: float = 90.0,
    heartbeat_timeout_s: float = 90.0,
):
    """
    Configure a class as an SPMD source.
    Operators defined within will automatically inherit source and interleaving traits.
    """
    base_decorator = spmd(
        resource_profile=resource_profile,
        mesh_axis_names=mesh_axis_names,
        mesh_axis_shapes=mesh_axis_shapes,
        runtime_env=runtime_env,
        driver_runtime_env=driver_runtime_env,
        checkpoint_components=checkpoint_components,
        heartbeat_interval_s=heartbeat_interval_s,
        execution_progress_warning_s=execution_progress_warning_s,
        heartbeat_timeout_s=heartbeat_timeout_s,
    )

    def decorator(cls):
        cls = base_decorator(cls)
        orig_get_config = cls.get_spmd_config

        def new_get_config(c):
            from fluxway.common.operator_spec import OperatorExecutionTrait

            config = orig_get_config()
            config.operator_specs = {
                name: spec.model_copy(
                    update={
                        "is_source": True,
                        "execution_trait": OperatorExecutionTrait.IO_SOURCE,
                        "is_interleaved": interleaved,
                        "interleave_unit_size": interleave_unit_size,
                        "execution_groups": execution_groups,
                        "source": source,
                    }
                )
                for name, spec in config.operator_specs.items()
            }
            return config

        cls.get_spmd_config = classmethod(new_get_config)
        return cls

    return decorator


def get_spmd_config(target: type) -> "SPMDConfig":
    """Retrieve the compiled SPMDConfig for an SPMD-decorated class."""
    config_getter = getattr(target, "get_spmd_config", None)
    if config_getter is None:
        raise TypeError(f"Class '{target.__name__}' is not decorated with @spmd")
    return config_getter()
