"""
Decorators for Fluxway framework.

Provides @operator and @data_type decorators to manage metadata and transport.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Tuple,
    Union,
    get_type_hints,
    Annotated,
)
import functools
import inspect
from contextlib import contextmanager
import threading
import logging

from fluxway.common.utils.tree import tree_flatten_type, _is_fluxway_data_type

logger = logging.getLogger(__name__)


class PicklableThreadLocal(threading.local):
    """A threading.local that can be pickled (state is lost)."""

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


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


# Thread-local storage for operator execution context (per-call, must be thread-isolated)
_operator_context = PicklableThreadLocal()

# Process-wide storage for operator class instances.
# Instances are created once (in SPMDWorker._pre_instantiate_operators, before the
# async thread pool starts) and then read-only shared across threads. Using a plain
# dict here is intentional: threading.local would hide the pre-instantiated entries
# from the asyncio.to_thread() worker threads.
_instance_storage_lock = _PicklableLock()
_instance_storage: Dict[int, Any] = {}


@contextmanager
def operator_context(
    static_args: Dict[str, Any],
    rank: int,
    world_size: int,
    operator_ctx: Optional[Any] = None,
):
    """Context manager to inject operator runtime context into current thread."""
    prev_static = getattr(_operator_context, "static_args", {})
    prev_rank = getattr(_operator_context, "rank", 0)
    prev_world = getattr(_operator_context, "world_size", 1)
    prev_operator_ctx = getattr(_operator_context, "operator_ctx", None)

    _operator_context.static_args = static_args
    _operator_context.rank = rank
    _operator_context.world_size = world_size
    _operator_context.operator_ctx = operator_ctx
    try:
        yield _operator_context
    finally:
        _operator_context.static_args = prev_static
        _operator_context.rank = prev_rank
        _operator_context.world_size = prev_world
        _operator_context.operator_ctx = prev_operator_ctx


def get_current_context():
    """Access current operator thread context."""
    return _operator_context


def _get_fluxway_streams(tp: Any, prefix: str = "") -> Tuple[List[str], List[str]]:
    """
    Recursively extract Data (Worker-to-Worker) and Control (Driver-to-Driver) fields from a @data_type.
    Returns: (data_fields, control_fields) as flattened paths (e.g. 'active.user_id').
    """
    from fluxway.common.types.configs import ControlFlow

    if not _is_fluxway_data_type(tp):
        return [], []

    # Use include_extras=True to see Annotated markers
    try:
        # Important: tp might be a Union (e.g. Optional[T])
        cls = tp
        origin = getattr(tp, "__origin__", None)
        if origin is Union:
            args = getattr(tp, "__args__", [])
            for arg in args:
                if arg is not type(None) and _is_fluxway_data_type(arg):
                    cls = arg
                    break

        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        return [], []

    data_fields = []
    control_fields = []

    for name, hint in hints.items():
        if name.startswith("_"):
            continue

        full_name = f"{prefix}.{name}" if prefix else name

        # Check for Annotated markers
        is_control = False
        if hasattr(hint, "__metadata__"):
            for meta in hint.__metadata__:
                if meta is ControlFlow:
                    is_control = True
                    break

        if _is_fluxway_data_type(hint):
            # Recurse for nested @data_type
            d, c = _get_fluxway_streams(hint, prefix=full_name)
            data_fields.extend(d)
            control_fields.extend(c)
        elif is_control:
            control_fields.append(full_name)
        else:
            # Default to Data plane for all other public fields (assumed Tensors or primitives)
            data_fields.append(full_name)

    return data_fields, control_fields


def _build_hierarchical_map(
    tp: Any, prefix: str = "", include_root: bool = False
) -> Dict[str, List[str]]:
    """
    Recursively builds a map from every valid field path prefix to the list of
    flattened physical stream names it contains.
    Example: 'dp_data' -> ['dp_data.labels.is_click', 'dp_data.dense_stats']
    """
    from fluxway.common.types.configs import ControlFlow

    if not _is_fluxway_data_type(tp):
        return {}

    mapping = {}
    try:
        cls = tp
        origin = getattr(tp, "__origin__", None)
        if origin is Union:
            args = getattr(tp, "__args__", [])
            for arg in args:
                if arg is not type(None) and _is_fluxway_data_type(arg):
                    cls = arg
                    break
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        return {}

    all_primitive_fields = []

    for name, hint in hints.items():
        if name.startswith("_"):
            continue

        full_name = f"{prefix}.{name}" if prefix else name

        is_control = False
        if hasattr(hint, "__metadata__"):
            for meta in hint.__metadata__:
                if meta is ControlFlow:
                    is_control = True
                    break

        if _is_fluxway_data_type(hint):
            sub_map = _build_hierarchical_map(hint, prefix=full_name, include_root=True)
            mapping.update(sub_map)
            all_primitive_fields.extend(sub_map.get(full_name, []))
        else:
            if not is_control:
                mapping[full_name] = [full_name]
                all_primitive_fields.append(full_name)

    if include_root and prefix:
        mapping[prefix] = all_primitive_fields

    return mapping


def operator(
    name: Optional[Union[str, Callable]] = None,
    input_sharding: Optional[Dict[str, Any]] = None,
    output_sharding: Optional[Dict[str, Any]] = None,
    control_argument: Optional[List[str]] = None,
    data_argument: Optional[List[str]] = None,
    static_argument: Optional[List[str]] = None,
    periodic_barrier: Optional[int] = None,
    max_inflight_steps: int = 0,
    is_source: Optional[bool] = None,
    barrier_config: Optional[Any] = None,
):
    """
    Mark a function as a Fluxway operator with automatic metadata inference.
    """

    # Handle both @operator and @operator(name="name")
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

    # At this point, name is either a str or None
    name_str: Optional[str] = name

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

    # --- Automated Inference from Type Annotations ---
    all_data_fields = []
    all_control_fields = []
    input_param_map = {}
    input_treedefs = {}

    # Identify data-plane parameters to determine prefixing mode
    data_params = []
    primitive_tensors = []
    for p_name, p in sig.parameters.items():
        if p_name in ("self", "cls"):
            continue

        tp = p.annotation
        if tp is inspect.Parameter.empty:
            if p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            raise TypeError(
                f"Fluxway Operator Error: Argument '{p_name}' in '{func.__qualname__}' missing type annotation.\n"
                f"Fix: Add a type hint, e.g., 'def {func.__name__}(self, batch: MyDataType):'"
            )

        if _is_fluxway_data_type(tp):
            data_params.append(p_name)
        else:
            primitive_tensors.append(p_name)

    # Use prefixing if there are multiple @data_type parameters
    use_prefix = len(data_params) > 1

    # Check arguments (for Input streams)
    for p_name, p in sig.parameters.items():
        if p_name in ("self", "cls"):
            continue

        tp = p.annotation
        if _is_fluxway_data_type(tp):
            data_fields, control_fields = _get_fluxway_streams(tp)

            # Apply prefix if needed for the global argument list
            if use_prefix:
                d_f_final = [f"{p_name}.{df}" for df in data_fields]
                c_f_final = [f"{p_name}.{cf}" for cf in control_fields]
            else:
                d_f_final = data_fields
                c_f_final = control_fields

            # Also build the hierarchical mapping for this parameter
            param_hierarchical_map = _build_hierarchical_map(
                tp, prefix=p_name if use_prefix else ""
            )

            # Always alias the python parameter name to all its resolved streams
            param_hierarchical_map[p_name] = d_f_final + c_f_final

            input_param_map.update(param_hierarchical_map)
        else:
            d_f_final = [p_name]
            c_f_final = []
            # primitive tensors map to themselves
            input_param_map[p_name] = [p_name]
            input_treedefs[p_name] = None  # Primitive is None or scalar treedef

        if _is_fluxway_data_type(tp):
            input_treedefs[p_name] = tree_flatten_type(tp)

        for df in d_f_final:
            if df not in all_data_fields:
                all_data_fields.append(df)
        for cf in c_f_final:
            if cf not in all_control_fields:
                all_control_fields.append(cf)

    # Merge explicit overrides
    def _dedup_preserve_order(lst):
        res = []
        for x in lst:
            if x not in res:
                res.append(x)
        return res

    final_tensors = _dedup_preserve_order(
        data_argument if data_argument is not None else all_data_fields
    )
    final_states = _dedup_preserve_order(
        control_argument if control_argument is not None else all_control_fields
    )
    final_statics = _dedup_preserve_order(
        static_argument if static_argument is not None else []
    )

    # --- Sharding Resolution (Pytree-like recursive mapping) ---
    meta_output_treedef = None

    def _resolve_recursive_sharding(
        sharding_val, data_fields, control_fields, prefix=""
    ):
        resolved = {}
        if isinstance(sharding_val, dict):
            if "type" in sharding_val and "axis" in sharding_val:
                for df in data_fields:
                    resolved[df] = sharding_val
            else:
                # Nested mapping: {"subfield": spec}
                for k, v in sharding_val.items():
                    full_k = f"{prefix}.{k}" if prefix else k
                    # Sub-fields belonging to this branch
                    sub_data = [
                        f
                        for f in data_fields
                        if f == full_k or f.startswith(full_k + ".")
                    ]
                    sub_control = [
                        f
                        for f in control_fields
                        if f == full_k or f.startswith(full_k + ".")
                    ]

                    if not sub_data and not sub_control:
                        # Fallback for flat field names provided directly
                        if k in data_fields:
                            resolved[k] = v
                            continue
                        raise ValueError(
                            f"Fluxway Operator Error: Unknown field '{k}' at path '{prefix}' for operator '{name}'.\n"
                            f"Available Data fields here: {[f[len(prefix) + 1 :] if prefix else f for f in data_fields]}\n"
                            f"Fix: Check for typos in your sharding dictionary or ensure the field is part of the @data_type."
                        )

                    if sub_control and not sub_data:
                        raise ValueError(
                            f"Fluxway Operator Error: Control field '{full_k}' does not support sharding.\n"
                            f"Fix: Control fields (marked with ControlFlow) are broadcasted; remove them from your sharding configuration."
                        )

                    resolved.update(
                        _resolve_recursive_sharding(
                            v, sub_data, sub_control, prefix=full_k
                        )
                    )
        else:
            # Uniform sharding: Apply this object/spec to all data fields in this branch
            for df in data_fields:
                resolved[df] = sharding_val
        return resolved

    resolved_input_sharding = {}
    if input_sharding:
        param_to_fields = {}
        for p_name in data_params:
            tp = sig.parameters[p_name].annotation
            d_f, c_f = _get_fluxway_streams(tp)
            if use_prefix:
                d_f = [f"{p_name}.{f}" for f in d_f]
                c_f = [f"{p_name}.{f}" for f in c_f]
            param_to_fields[p_name] = (d_f, c_f)

        for k, v in input_sharding.items():
            if k in param_to_fields:
                d_f, c_f = param_to_fields[k]
                resolved_input_sharding.update(
                    _resolve_recursive_sharding(
                        v, d_f, c_f, prefix=k if use_prefix else ""
                    )
                )
            elif k in final_tensors:
                resolved_input_sharding[k] = v
            elif k in final_states:
                raise ValueError(
                    f"Fluxway Operator Error: Control field '{k}' does not support sharding."
                )
            else:
                raise ValueError(
                    f"Fluxway Operator Error: Unknown sharding key '{k}' for operator '{name}'."
                )
    input_sharding = resolved_input_sharding

    resolved_output_sharding = {}
    return_tp = sig.return_annotation
    out_data_fields, out_control_fields = [], []
    output_param_map = {}

    if return_tp is not None and return_tp is not inspect.Signature.empty:
        if _is_fluxway_data_type(return_tp):
            out_data_fields, out_control_fields = _get_fluxway_streams(return_tp)
            # Add hierarchical mappings
            output_param_map = _build_hierarchical_map(return_tp, include_root=False)
            output_param_map["return"] = out_data_fields + out_control_fields
        elif output_sharding:
            out_data_fields = list(output_sharding.keys())
            output_param_map["return"] = out_data_fields

    if output_sharding:
        if _is_fluxway_data_type(return_tp):
            # Handle single spec applied to all fields
            if "type" in output_sharding and "axis" in output_sharding:
                for df in out_data_fields:
                    resolved_output_sharding[df] = output_sharding
            else:
                # Canonicalize aliased keys ("return" or class name)
                return_type_name = return_tp.__name__.lower()
                remaining_sharding = {}

                for k, v in output_sharding.items():
                    if k.lower() in ("return", return_type_name):
                        # Root alias: apply its value recursively to the entire return object
                        resolved_output_sharding.update(
                            _resolve_recursive_sharding(
                                v, out_data_fields, out_control_fields
                            )
                        )
                    else:
                        remaining_sharding[k] = v

                # 3. Resolve remaining explicit field mappings
                if remaining_sharding:
                    resolved_output_sharding.update(
                        _resolve_recursive_sharding(
                            remaining_sharding, out_data_fields, out_control_fields
                        )
                    )
        else:
            # For non-data_type returns (Dict or single Tensor), the output_sharding
            # serves as the ground-truth contract for TUUID ordering.
            resolved_output_sharding = output_sharding

            if isinstance(output_sharding, dict):
                # If a sharding map is provided, the return value MUST be a dictionary
                # matching these keys. We generate a template to enforce this order.
                sharding_keys = list(output_sharding)
                meta_output_treedef = (
                    "dict",
                    sharding_keys,
                    [None] * len(sharding_keys),
                )
    output_sharding = resolved_output_sharding

    # Periodic barrier constraints
    if periodic_barrier is not None:
        if len(final_tensors) + len(final_states) != 1:
            raise ValueError(
                f"Fluxway Operator Error: Periodic operator '{name}' must have exactly one input field (Data or Control), "
                f"but found {len(final_tensors)} Data fields and {len(final_states)} Control fields.\n"
                f"Fix: Use a @data_type with exactly one field for streaming/periodic processing."
            )

    # --- Build Metadata ---
    input_sharding_processed = {}
    if input_sharding:
        for k, v in input_sharding.items():
            if isinstance(v, dict):
                input_sharding_processed[k] = v
            elif hasattr(v, "as_dict"):
                input_sharding_processed[k] = v.as_dict()
            else:
                input_sharding_processed[k] = v.__dict__

    output_sharding_processed = {}
    if output_sharding:
        for k, v in output_sharding.items():
            if isinstance(v, dict):
                output_sharding_processed[k] = v
            elif hasattr(v, "as_dict"):
                output_sharding_processed[k] = v.as_dict()
            else:
                output_sharding_processed[k] = v.__dict__

    # Auto-infer output arguments from return type
    output_args = None
    return_tp = sig.return_annotation
    if (
        return_tp is not None
        and return_tp is not inspect.Signature.empty
        and _is_fluxway_data_type(return_tp)
    ):
        out_data, out_control = _get_fluxway_streams(return_tp)
        output_args = {
            ArgumentType.DATA.value: out_data,
            ArgumentType.CONTROL.value: out_control,
        }
    elif output_sharding:
        output_args = {ArgumentType.DATA.value: list(output_sharding.keys())}

    meta = OperatorMeta(
        name=name,
        is_source=is_source,
        max_inflight_steps=max_inflight_steps,
        input_arguments={
            ArgumentType.CONTROL.value: final_states,
            ArgumentType.DATA.value: final_tensors,
            ArgumentType.STATIC.value: list(final_statics),
        },
        output_arguments=output_args,
        input_sharding=input_sharding_processed,
        output_sharding=output_sharding_processed,
        input_param_map=input_param_map,
        output_param_map=output_param_map,
        input_treedefs=input_treedefs,
        output_treedef=meta_output_treedef
        if meta_output_treedef
        else (
            tree_flatten_type(return_tp) if _is_fluxway_data_type(return_tp) else None
        ),
        barrier_config=barrier_config
        if barrier_config
        else (
            PeriodicBarrierConfig(n=periodic_barrier)
            if periodic_barrier is not None
            else StandardBarrierConfig()
        ),
    )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # We need the class to instantiate it
        cls = getattr(wrapper, "_flux_spmd_class", None)
        if not cls:
            # Fallback: try to find it from qualname
            qualname_parts = func.__qualname__.split(".")
            if len(qualname_parts) >= 2:
                module = inspect.getmodule(func)
                cls = getattr(module, qualname_parts[-2], None)

        if not cls:
            raise RuntimeError(
                f"Fluxway Execution Error: Cannot instantiate class for operator '{name}'."
            )

        instance_key = id(cls)  # different operators in one class share one instance.
        if instance_key not in _instance_storage:
            with _instance_storage_lock:
                if instance_key not in _instance_storage:
                    ctx = get_current_context()
                    rank = getattr(ctx, "rank", 0)
                    world_size = getattr(ctx, "world_size", 1)
                    static_args = getattr(ctx, "static_args", {}) or {}

                    try:
                        init_sig = inspect.signature(cls.__init__)
                        init_params = set(init_sig.parameters.keys()) - {"self"}
                        has_var_keyword = any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in init_sig.parameters.values()
                        )
                        init_kwargs = (
                            dict(static_args)
                            if has_var_keyword
                            else {
                                k: v for k, v in static_args.items() if k in init_params
                            }
                        )
                    except (ValueError, TypeError):
                        init_kwargs = {}

                    instance = cls.__new__(cls)
                    instance.rank = rank
                    instance.world_size = world_size
                    instance.static_args = static_args
                    instance.ctx = getattr(ctx, "operator_ctx", None)
                    instance.__init__(**init_kwargs)
                    _instance_storage[instance_key] = instance

        instance = _instance_storage[instance_key]
        ctx = get_current_context()
        instance.static_args = getattr(ctx, "static_args", {})
        instance.rank = getattr(ctx, "rank", 0)
        instance.world_size = getattr(ctx, "world_size", 1)
        instance.ctx = getattr(ctx, "operator_ctx", None)
        return func(instance, *args, **kwargs)

    def set_spmd_class(cls):
        wrapper._flux_spmd_class = cls

    # Attach metadata to both original and wrapper
    for target in (func, wrapper):
        target._fluxway_operator_meta = meta
        target._flux_set_spmd_class = set_spmd_class

    # Only wrap in staticmethod if it's actually inside a class
    if "." in func.__qualname__:
        return staticmethod(wrapper)
    return wrapper


def data_type():
    """
    Mark a class as a Fluxway transportable data type.
    Enforces flattening and adds as_dict() for standard access.
    """
    T = TypeVar("T")

    def decorator(cls: Type[T]) -> Type[T]:
        try:
            hints = get_type_hints(cls, include_extras=True)
        except Exception as e:
            raise TypeError(
                f"Fluxway Data Error: Cannot inspect type hints for '{cls.__name__}': {e}"
            ) from e

        for name, tp in hints.items():
            if name.startswith("_"):
                continue

            # Extract base type from Annotated if present
            base_tp = tp
            if hasattr(tp, "__origin__") and tp.__origin__ is Annotated:
                base_tp = tp.__args__[0]

            origin = getattr(base_tp, "__origin__", base_tp)
            if origin is dict or origin is Dict:
                raise TypeError(
                    f"Fluxway Data Error: Class '{cls.__name__}' field '{name}' uses 'dict', which is not supported.\n"
                    f"Fix: Use a @data_type decorated dataclass instead of a dictionary."
                )

        cls.is_fluxway_data = classmethod(lambda c: True)

        if not hasattr(cls, "as_dict"):

            def as_dict(self):
                """Generic, library-agnostic dictionary conversion."""
                return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

            cls.as_dict = as_dict

        return cls

    return decorator


def spmd(
    resource: Optional[Dict[str, Any]] = None,
    mesh_axis_names: Optional[Tuple[str, ...]] = None,
    mesh_axis_shapes: Optional[Tuple[int, ...]] = None,
):
    """
    Configure a class for SPMD distributed execution.
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

    def decorator(cls: Type[T]) -> Type[T]:
        cls._resource = res_spec
        cls._mesh = mesh

        # Link operators
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name, None)
            getattr(attr, "_flux_set_spmd_class", lambda c: None)(cls)

        # Standard API
        cls.get_resource = classmethod(lambda c: c._resource)
        cls.get_mesh = classmethod(lambda c: c._mesh)
        cls.get_num_workers = classmethod(lambda c: c._mesh.size if c._mesh else 1)
        cls.is_fluxway_spmd = classmethod(lambda c: True)

        def get_spmd_config(c) -> "SPMDConfig":
            from fluxway.common.types import SPMDConfig, OperatorMeta

            operators, metas = {}, {}

            import copy

            for name in dir(c):
                attr = getattr(c, name, None)
                meta = getattr(attr, "_fluxway_operator_meta", None)
                if isinstance(meta, OperatorMeta):
                    operators[meta.name] = attr
                    metas[meta.name] = copy.deepcopy(meta)

            return SPMDConfig(mesh=c._mesh, operators=operators, operator_metas=metas)

        cls.get_spmd_config = classmethod(get_spmd_config)
        return cls

    return decorator


def spmd_source(
    resource: Optional[Dict[str, Any]] = None,
    mesh_axis_names: Optional[Tuple[str, ...]] = None,
    mesh_axis_shapes: Optional[Tuple[int, ...]] = None,
    interleaved: bool = False,
    interleave_unit_size: int = 1,
):
    """
    Configure a class as an SPMD source.
    Operators defined within will automatically inherit source and interleaving traits.
    """
    # Reuse the core logic of @spmd
    base_decorator = spmd(
        resource=resource,
        mesh_axis_names=mesh_axis_names,
        mesh_axis_shapes=mesh_axis_shapes,
    )

    def decorator(cls):
        # Apply the base @spmd to setup mesh and resource
        cls = base_decorator(cls)

        # Override the get_spmd_config to stamp Source properties
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
