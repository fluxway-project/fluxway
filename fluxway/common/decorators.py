"""
Decorators for Fluxway framework.

Provides @operator and @data_type decorators to manage metadata and transport.
"""

import contextvars
import copy
import functools
import importlib
import inspect
import logging
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, TypeVar, Union, get_type_hints

from fluxway.common.utils.tree import tree_flatten_type, _is_fluxway_data_type

logger = logging.getLogger(__name__)


@dataclass
class _OperatorContextState:
    static_args: dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    world_size: int = 1
    ctx: Any = None
    local_rank: int | None = None
    local_world_size: int | None = None
    node_rank: int = 0
    num_nodes: int = 1
    rank0_address: str | None = None


class _PicklableLock:
    """A threading.Lock wrapper that survives pickle (Ray serialization).

    On deserialization a fresh lock is created in the new process.
    This is correct because _instance_storage is per-process, so each
    process needs its own lock — not a shared one.
    """

    def __init__(self):
        self._lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *args):
        self._lock.release()

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._lock = threading.Lock()


_DEFAULT_CONTEXT_STATE = _OperatorContextState()

_operator_context_var: contextvars.ContextVar[_OperatorContextState] = (
    contextvars.ContextVar("operator_context", default=_DEFAULT_CONTEXT_STATE)
)

_instance_storage_lock = _PicklableLock()
_instance_storage: dict[int, Any] = {}


def get_instance_storage() -> tuple[dict[int, Any], Any]:
    import fluxway.common.decorators as fd

    return fd._instance_storage, fd._instance_storage_lock


def _resolve_owner_class(module_name: str, class_qualname: str) -> type:
    module = importlib.import_module(module_name)
    cls = module
    for part in class_qualname.split("."):
        cls = getattr(cls, part)
    return cls


@contextmanager
def operator_context(
    static_args: dict[str, Any],
    rank: int,
    world_size: int,
    operator_ctx: Any | None = None,
    local_rank: int | None = None,
    local_world_size: int | None = None,
    node_rank: int = 0,
    num_nodes: int = 1,
    rank0_address: str | None = None,
):
    """Context manager to inject operator runtime context into current coroutine/thread.

    Uses contextvars.ContextVar so each asyncio Task and each to_thread worker
    gets an isolated snapshot — no cross-coroutine state pollution.
    """
    state = _OperatorContextState(
        static_args=static_args,
        rank=rank,
        world_size=world_size,
        ctx=operator_ctx,
        local_rank=local_rank,
        local_world_size=local_world_size,
        node_rank=node_rank,
        num_nodes=num_nodes,
        rank0_address=rank0_address,
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
            return_type_name = return_tp.__name__.lower()
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


def _normalize_sharding_values(sharding: dict) -> dict:
    result: dict = {}
    for k, v in sharding.items():
        if isinstance(v, dict):
            result[k] = v
        elif hasattr(v, "as_dict"):
            result[k] = v.as_dict()
        else:
            result[k] = v.__dict__
    return result


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
    }
)
_DYNAMIC_CONTEXT_ATTR_NAMES: frozenset[str] = frozenset({"ctx"})
_CONTEXT_ATTR_NAMES: frozenset[str] = (
    _STATIC_CONTEXT_ATTR_NAMES | _DYNAMIC_CONTEXT_ATTR_NAMES
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


def _build_init_kwargs(cls: type, static_args: dict) -> dict:
    try:
        init_sig = inspect.signature(cls.__init__)
        init_params = set(init_sig.parameters.keys()) - {"self"}
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in init_sig.parameters.values()
        )
        return (
            dict(static_args)
            if has_var_kw
            else {k: v for k, v in static_args.items() if k in init_params}
        )
    except (ValueError, TypeError):
        return {}


class _OperatorWrapper:
    def __init__(self, func: Callable, meta: Any):
        self.func = func
        self._fluxway_operator_meta = meta
        self._cls = None
        functools.update_wrapper(self, func)

    def __get__(self, instance, owner):
        if self._cls is None:
            self._cls = owner
        if not hasattr(owner, "_fluxway_ctx_injected"):
            _inject_context_properties(owner)
            owner._fluxway_ctx_injected = True

        def _get_or_create_instance():
            instance_key = id(owner)
            storage, lock = get_instance_storage()
            if instance_key not in storage:
                with lock:
                    if instance_key not in storage:
                        ctx = get_current_context()
                        # Write static attributes directly into inst.__dict__
                        # BEFORE __init__ so that self.rank / self.world_size
                        # etc. are available as plain instance attributes
                        # throughout the object's lifetime, including in
                        # lifecycle hooks like save_state / restore_state that
                        # run outside any active operator context.
                        inst = owner.__new__(owner)
                        for attr in _STATIC_CONTEXT_ATTR_NAMES:
                            inst.__dict__[attr] = getattr(ctx, attr)
                        init_kwargs = _build_init_kwargs(owner, ctx.static_args or {})
                        inst.__init__(**init_kwargs)
                        storage[instance_key] = inst
            return storage[id(owner)]

        if inspect.iscoroutinefunction(self.func):

            @functools.wraps(self.func)
            async def async_bound_method(*args, **kwargs):
                return await self.func(_get_or_create_instance(), *args, **kwargs)

            async_bound_method._fluxway_operator_meta = self._fluxway_operator_meta
            return async_bound_method

        @functools.wraps(self.func)
        def bound_method(*args, **kwargs):
            return self.func(_get_or_create_instance(), *args, **kwargs)

        bound_method._fluxway_operator_meta = self._fluxway_operator_meta
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
    max_inflight_steps: int = 0,
    is_source: bool | None = None,
    barrier_config: Any | None = None,
):
    """
    Mark a function as a Fluxway operator with automatic metadata inference.
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
            max_inflight_steps,
            is_source,
            barrier_config,
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
            max_inflight_steps,
            is_source,
            barrier_config,
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
    max_inflight_steps,
    is_source,
    barrier_config=None,
):
    from fluxway.common.types.configs import (
        ArgumentType,
        OperatorMeta,
        StandardBarrierConfig,
        PeriodicBarrierConfig,
    )

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

    if _is_fluxway_data_type(return_tp):
        output_args: dict | None = {
            ArgumentType.DATA.value: out_data_fields,
            ArgumentType.CONTROL.value: out_control_fields,
        }
    elif resolved_output:
        output_args = {ArgumentType.DATA.value: list(resolved_output.keys())}
    else:
        output_args = None

    _is_class_method = "." in func.__qualname__
    meta = OperatorMeta(
        name=name,
        owner_class_qualname=func.__qualname__.rsplit(".", 1)[0]
        if _is_class_method
        else "",
        owner_class_module=func.__module__ if _is_class_method else "",
        is_source=is_source,
        max_inflight_steps=max_inflight_steps,
        input_arguments={
            ArgumentType.CONTROL.value: final_states,
            ArgumentType.DATA.value: final_tensors,
            ArgumentType.STATIC.value: list(final_statics),
        },
        output_arguments=output_args,
        input_sharding=_normalize_sharding_values(resolved_input),
        output_sharding=_normalize_sharding_values(resolved_output),
        input_param_map=input_param_map,
        output_param_map=output_param_map,
        input_treedefs=input_treedefs,
        output_treedef=(
            meta_output_treedef
            if meta_output_treedef
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

    wrapper = _OperatorWrapper(func, meta)

    # Expose meta on the wrapper for framework inspection
    wrapper._fluxway_operator_meta = meta

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

        cls.is_fluxway_data = classmethod(lambda c: True)

        if not hasattr(cls, "as_dict"):

            def as_dict(self):
                return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

            cls.as_dict = as_dict

        return cls

    return decorator


def spmd(
    resource: dict[str, Any] | None = None,
    runtime_env: dict[str, Any] | None = None,
    mesh_axis_names: tuple[str, ...] | None = None,
    mesh_axis_shapes: tuple[int, ...] | None = None,
):
    """
    Configure a class for SPMD distributed execution.

    Args:
        resource: Resource requirements (num_cpus, num_gpus, memory_gb).
        runtime_env: Ray runtime_env dict for Driver/Worker actors.
                     Example: {"env_vars": {"JAX_PLATFORMS": "gpu"}}
        mesh_axis_names: Axis names for the execution mesh.
        mesh_axis_shapes: Shape of the execution mesh.
    """
    from fluxway.common.types import ResourceSpec, SPMDConfig
    from fluxway.common.mesh import DeviceMesh

    T = TypeVar("T")

    res_spec = None
    if resource:
        res_spec = ResourceSpec(
            num_cpus=resource.get("num_cpus", 1),
            num_gpus=resource.get("num_gpus", 0),
            memory_gb=resource.get("memory_gb", 0.0),
        )

    mesh = None
    if mesh_axis_names and mesh_axis_shapes:
        mesh = DeviceMesh(axis_names=mesh_axis_names, device_shape=mesh_axis_shapes)

    def decorator(cls: type[T]) -> type[T]:
        cls._resource = res_spec
        cls._mesh = mesh
        cls._runtime_env = runtime_env

        cls.get_resource = classmethod(lambda c: c._resource)
        cls.get_mesh = classmethod(lambda c: c._mesh)
        cls.get_runtime_env = classmethod(lambda c: c._runtime_env)
        cls.get_num_workers = classmethod(lambda c: c._mesh.size if c._mesh else 1)
        cls.is_fluxway_spmd = classmethod(lambda c: True)

        original_new = cls.__new__

        @functools.wraps(original_new)
        def __new__(cls_obj, *args, **kwargs):
            from fluxway.control_plane.graph.functional import FunctionalGraph

            current_graph = FunctionalGraph.current()
            if current_graph is not None:
                name = kwargs.pop("name", cls_obj.__name__)
                return current_graph.node(name, cls_obj, *args, **kwargs)

            if original_new is object.__new__:
                return original_new(cls_obj)
            return original_new(cls_obj, *args, **kwargs)

        cls.__new__ = __new__

        def get_spmd_config(c) -> "SPMDConfig":
            from fluxway.common.types import SPMDConfig, OperatorMeta

            # Pass 1: walk the full MRO to collect operator metadata from _OperatorWrapper.
            # A subclass may override a method with a plain function, losing the wrapper.
            # We scan from the bottom of MRO up so that the most-derived _OperatorWrapper
            # (if any) wins, but we never discard metadata just because the attr was already
            # seen as a plain function in a more-derived class.
            wrapper_meta: dict[
                str, tuple[str, OperatorMeta]
            ] = {}  # attr_name -> (op_name, meta)
            for klass in reversed(c.__mro__):
                for attr_name, attr in klass.__dict__.items():
                    if not isinstance(attr, _OperatorWrapper):
                        continue
                    meta = attr._fluxway_operator_meta
                    if not isinstance(meta, OperatorMeta):
                        continue
                    wrapper_meta[attr_name] = (meta.name, copy.deepcopy(meta))

            # Pass 2: for each discovered operator, resolve the attr_name that the final
            # class actually exposes (may be a plain override — that is fine, the framework
            # only needs to know the correct attr_name for dispatch and the metadata).
            operators, metas = {}, {}
            for attr_name, (op_name, meta) in wrapper_meta.items():
                operators[op_name] = attr_name
                metas[op_name] = meta

            return SPMDConfig(
                mesh=c._mesh,
                operators=operators,
                operator_metas=metas,
                runtime_env=c._runtime_env,
            )

        cls.get_spmd_config = classmethod(get_spmd_config)
        return cls

    return decorator


def spmd_source(
    resource: dict[str, Any] | None = None,
    mesh_axis_names: tuple[str, ...] | None = None,
    mesh_axis_shapes: tuple[int, ...] | None = None,
    interleaved: bool = False,
    interleave_unit_size: int = 1,
    runtime_env: dict[str, Any] | None = None,
):
    """
    Configure a class as an SPMD source.
    Operators defined within will automatically inherit source and interleaving traits.
    """
    base_decorator = spmd(
        resource=resource,
        mesh_axis_names=mesh_axis_names,
        mesh_axis_shapes=mesh_axis_shapes,
        runtime_env=runtime_env,
    )

    def decorator(cls):
        cls = base_decorator(cls)
        orig_get_config = cls.get_spmd_config

        def new_get_config(c):
            config = orig_get_config()
            for meta in config.operator_metas.values():
                meta.is_source = True
                meta.is_interleaved = interleaved
                meta.interleave_unit_size = interleave_unit_size
            return config

        cls.get_spmd_config = classmethod(new_get_config)
        return cls

    return decorator
