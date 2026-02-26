"""
Decorators for Fluxway framework.

Provides @operator and @data_type decorators to manage metadata and transport.
"""

from typing import (
    Any,
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

logger = logging.getLogger(__name__)

# Thread-local storage for operator context
_operator_context = threading.local()


@contextmanager
def operator_context(static_args: Dict[str, Any], rank: int, world_size: int):
    """Context manager to inject operator runtime context into current thread."""
    prev_static = getattr(_operator_context, "static_args", {})
    prev_rank = getattr(_operator_context, "rank", 0)
    prev_world = getattr(_operator_context, "world_size", 1)

    _operator_context.static_args = static_args
    _operator_context.rank = rank
    _operator_context.world_size = world_size
    try:
        yield _operator_context
    finally:
        _operator_context.static_args = prev_static
        _operator_context.rank = prev_rank
        _operator_context.world_size = prev_world


def get_current_context():
    """Access current operator thread context."""
    return _operator_context


def _is_fluxway_data_type(tp: Any) -> bool:
    """Check if a type is marked with @data_type."""
    if tp is None:
        return False
    # Handle Optional[T] or Union
    origin = getattr(tp, "__origin__", None)
    if origin is Union:
        args = getattr(tp, "__args__", [])
        return any(_is_fluxway_data_type(arg) for arg in args if arg is not type(None))

    return hasattr(tp, "is_fluxway_data") and tp.is_fluxway_data()


def _get_fluxway_streams(tp: Any) -> Tuple[List[str], List[str]]:
    """
    Extract Data (Worker-to-Worker) and Control (Driver-to-Driver) fields from a @data_type.
    Returns: (data_fields, control_fields)
    """
    from fluxway.common.types.configs import ControlStream

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

        # Check for Annotated markers
        is_control = False
        if hasattr(hint, "__metadata__"):
            for meta in hint.__metadata__:
                if meta is ControlStream:
                    is_control = True
                    break

        if is_control:
            control_fields.append(name)
        else:
            # Default to Data plane for all other public fields
            data_fields.append(name)

    return data_fields, control_fields


def operator(
    name: str,
    input_sharding: Optional[Dict[str, Any]] = None,
    output_sharding: Optional[Dict[str, Any]] = None,
    control_argument: Optional[List[str]] = None,
    data_argument: Optional[List[str]] = None,
    static_argument: Optional[List[str]] = None,
    periodic_barrier: Optional[int] = None,
):
    """
    Mark a function as a Fluxway operator with automatic metadata inference.
    """
    from fluxway.common.types.configs import (
        ArgumentType,
        OperatorMeta,
        StandardBarrierConfig,
        PeriodicBarrierConfig,
    )

    def decorator(func):
        sig = inspect.signature(func)

        # --- 1. Automated Inference from Type Annotations ---
        all_data_fields = set()
        all_control_fields = set()

        # Check arguments (for Input streams)
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            tp = param.annotation

            if tp is inspect.Parameter.empty:
                raise TypeError(
                    f"Fluxway Operator Error: Argument '{param_name}' in '{func.__qualname__}' missing type annotation.\n"
                    f"Fix: Add a type hint, e.g., 'def {func.__name__}(self, batch: MyDataType):'"
                )

            if not _is_fluxway_data_type(tp):
                continue

            data_fields, control_fields = _get_fluxway_streams(tp)
            all_data_fields.update(data_fields)
            all_control_fields.update(control_fields)

        # Merge explicit overrides
        final_tensors = list(
            set(data_argument) if data_argument is not None else all_data_fields
        )
        final_states = list(
            set(control_argument)
            if control_argument is not None
            else all_control_fields
        )
        final_statics = list(
            set(static_argument) if static_argument is not None else set()
        )

        # --- 2. Deep Validation with Clear Messages ---
        # Sharding vs Streams consistency
        if input_sharding:
            illegal_shards = set(input_sharding.keys()) & set(final_states)
            if illegal_shards:
                raise ValueError(
                    f"Fluxway Operator Error: Field(s) {illegal_shards} in operator '{name}' are marked as Control (ControlStream), "
                    f"but have sharding configured. Control fields are broadcast via Driver and do not support sharding.\n"
                    f"Fix: Remove input_sharding for these fields OR change them to Data (DataStream) in @data_type definition."
                )

            unknown_shards = set(input_sharding.keys()) - set(final_tensors)
            if unknown_shards:
                raise ValueError(
                    f"Fluxway Operator Error: input_sharding contains unknown keys {unknown_shards} for operator '{name}'.\n"
                    f"Fix: Ensure sharding keys match Data fields defined in the input @data_type class."
                )

        # Periodic barrier constraints
        if periodic_barrier is not None:
            if len(final_tensors) + len(final_states) != 1:
                raise ValueError(
                    f"Fluxway Operator Error: Periodic operator '{name}' must have exactly one input field (Data or Control), "
                    f"but found {len(final_tensors)} Data fields and {len(final_states)} Control fields.\n"
                    f"Fix: Use a @data_type with exactly one field for streaming/periodic processing."
                )

        # --- 3. Build Metadata ---
        input_sharding_processed = {}
        if input_sharding:
            for k, v in input_sharding.items():
                input_sharding_processed[k] = v if isinstance(v, dict) else v.__dict__

        output_sharding_processed = {}
        if output_sharding:
            for k, v in output_sharding.items():
                output_sharding_processed[k] = v if isinstance(v, dict) else v.__dict__

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
                ArgumentType.TENSOR.value: out_data,
                ArgumentType.STATE.value: out_control,
            }
        elif output_sharding:
            output_args = {ArgumentType.TENSOR.value: list(output_sharding.keys())}

        barrier_config = (
            PeriodicBarrierConfig(n=periodic_barrier)
            if periodic_barrier is not None
            else StandardBarrierConfig()
        )

        meta = OperatorMeta(
            name=name,
            input_arguments={
                ArgumentType.STATE.value: final_states,
                ArgumentType.TENSOR.value: final_tensors,
                ArgumentType.STATIC.value: list(final_statics),
            },
            output_arguments=output_args,
            input_sharding=input_sharding_processed,
            output_sharding=output_sharding_processed,
            barrier_config=barrier_config,
        )

        # --- 4. Wrapper Generation ---
        func._flux_operator_meta = meta

        import threading

        _local = threading.local()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create thread-local instance
            if not hasattr(_local, "instance"):
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
                _local.instance = cls()

            # Inject runtime context into the class instance
            ctx = get_current_context()
            if hasattr(ctx, "static_args"):
                _local.instance.static_args = ctx.static_args
            if hasattr(ctx, "rank"):
                _local.instance.rank = ctx.rank
            if hasattr(ctx, "world_size"):
                _local.instance.world_size = ctx.world_size

            return func(_local.instance, *args, **kwargs)

        def set_spmd_class(cls):
            wrapper._flux_spmd_class = cls

        wrapper._flux_set_spmd_class = set_spmd_class
        wrapper._flux_operator_meta = meta

        return staticmethod(wrapper)

    return decorator


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
    use_global_step_indexing: bool = False,
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
        cls._use_global_step_indexing = use_global_step_indexing

        # Link operators
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name, None)
            if hasattr(attr, "_flux_set_spmd_class"):
                attr._flux_set_spmd_class(cls)

        # Standard API
        cls.get_resource = classmethod(lambda c: c._resource)
        cls.get_mesh = classmethod(lambda c: c._mesh)
        cls.get_num_workers = classmethod(lambda c: c._mesh.size if c._mesh else 1)
        cls.is_fluxway_spmd = classmethod(lambda c: True)

        def get_spmd_config(c) -> "SPMDConfig":
            from fluxway.common.types import SPMDConfig, OperatorMeta

            operators, metas = {}, {}

            for name in dir(c):
                attr = getattr(c, name, None)
                meta = getattr(attr, "_flux_operator_meta", None)
                if isinstance(meta, OperatorMeta):
                    operators[meta.name] = attr
                    metas[meta.name] = meta

            return SPMDConfig(
                mesh=c._mesh,
                use_global_step_indexing=c._use_global_step_indexing,
                operators=operators,
                operator_metas=metas,
            )

        cls.get_spmd_config = classmethod(get_spmd_config)
        return cls

    return decorator
