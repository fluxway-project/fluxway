"""
Decorators for Fluxway framework.

Provides @operator decorator to attach metadata (sharding, argument types) to user functions.
"""

from typing import Any, Dict, List, Optional, Type, TypeVar, Tuple
import functools
import inspect


def operator(
    name: str,
    input_sharding: Optional[Dict[str, Any]] = None,
    output_sharding: Optional[Dict[str, Any]] = None,
    state_argument: Optional[List[str]] = None,
    tensor_argument: Optional[List[str]] = None,
    static_argument: Optional[List[str]] = None,
    eager_trigger: bool = False,
):
    """
    Decorator to mark a function as a Fluxway operator and attach metadata.

    Args:
        name: Operator name (unique within SPMD)
        input_sharding: Input parameter sharding spec. Dict[arg_name, raw_spec_dict]
            Raw spec dict should contain basic types only:
            - For dense: {"type": "dense", "axis": ["dp", None, "tp"]}
            - For sparse: {"type": "sparse", "axis": [0], "fn": my_routing_fn}
              where fn(tensor, tensor_meta, post_spec) -> routing result
            - For replicate: {"type": "dense", "axis": [None]} or omit sharding
        output_sharding: Output return value sharding spec. Dict[field_name, raw_spec_dict]
            Same format as input_sharding
        state_argument: List of argument names to treat as state (control flow)
        tensor_argument: List of argument names to treat as tensor (data flow)
        static_argument: List of argument names to treat as static args
        eager_trigger: If True, the operator triggers as soon as any data arrives
            (expected tensor shards = 1 for all tensors, ignoring sharding topology).
            Useful for operators that should process data immediately without waiting
            for all expected shards.

    Note:
        Raw sharding specs are validated and stored as basic dicts.
        They are dynamically resolved by the AsyncDataDistributor when
        routing tensors between SPMD pipeline stages.
    """
    from fluxway.common.types.data_types import ArgumentType

    def decorator(func):
        # Import here to avoid circular imports at module import time
        from fluxway.common.types import OperatorMeta

        # 1. Validate arguments
        # 1.1. Validate state_argument, tensor_argument, and static_argument are mutually exclusive (no overlap)
        state_args = set(state_argument or [])
        tensor_args = set(tensor_argument or [])
        static_args = set(static_argument or [])
        intersect = (
            (state_args & tensor_args)
            | (state_args & static_args)
            | (tensor_args & static_args)
        )
        if intersect:
            raise ValueError(
                f"Arguments {intersect} are present in more than one of state_argument, tensor_argument, static_argument for operator '{func.__qualname__}'"
            )

        # 1.2. Validate that tensor_argument and input_sharding keys, if both present, match exactly
        tensor_args = set(tensor_argument or [])
        input_sharding_keys = set(input_sharding.keys()) if input_sharding else set()
        if tensor_args != input_sharding_keys:
            raise ValueError(
                f"In operator '{func.__qualname__}', tensor_argument ({tensor_args}) and input_sharding keys ({input_sharding_keys}) must match exactly"
            )

        # 1.3. Validate eager_trigger constraint: state_arg + tensor_arg must have exactly one element
        if eager_trigger:
            total_transferable_args = len(state_args) + len(tensor_args)
            if total_transferable_args != 1:
                raise ValueError(
                    f"Operator '{func.__qualname__}' has eager_trigger=True, but state_argument + tensor_argument "
                    f"count is {total_transferable_args} (state={len(state_args)}, tensor={len(tensor_args)}). "
                    f"eager_trigger operators must have exactly one transferable argument."
                )

        # 2. Validate that both input and output are of fluxway_data type

        # Helper to check fluxway_data: class has is_fluxway_data method (set by @data_type decorator)
        def _is_fluxway_data_type(tp):
            # Check if the class has the is_fluxway_data classmethod set by @data_type decorator
            if hasattr(tp, "is_fluxway_data"):
                # Try to call it as a classmethod to verify it's the right one
                try:
                    result = tp.is_fluxway_data()
                    return result is True
                except (TypeError, AttributeError):
                    # If it's not callable or doesn't work as expected, check if it's the classmethod
                    return isinstance(getattr(tp, "is_fluxway_data", None), classmethod)
            return False

        sig = inspect.signature(func)
        input_annots = []
        for param in sig.parameters.values():
            # Only check parameters that are not 'self' or 'cls'
            if param.name in ("self", "cls"):
                continue
            if param.annotation is inspect.Parameter.empty:
                raise TypeError(
                    f"Argument '{param.name}' of operator '{func.__qualname__}' missing type annotation"
                )
            input_annots.append((param.name, param.annotation))

        # Input arguments may be empty; if so, input_sharding, state_argument, tensor_argument, and static_argument must also be empty
        if len(input_annots) == 0:
            if (
                input_sharding
                or (state_argument and len(state_argument) > 0)
                or (tensor_argument and len(tensor_argument) > 0)
                or (static_argument and len(static_argument) > 0)
            ):
                raise ValueError(
                    f"Operator '{func.__qualname__}' has no input arguments, so input_sharding, state_argument, tensor_argument, and static_argument must all be empty."
                )
        else:
            # Check that all input types are fluxway_data
            for arg_name, tp in input_annots:
                if not _is_fluxway_data_type(tp):
                    raise TypeError(
                        f"Argument '{arg_name}' of operator '{func.__qualname__}' "
                        f"must be a fluxway_data type (use @data_type on input class), got {tp}"
                    )

        # Check output type; allow None (no output)
        return_annot = sig.return_annotation
        if (
            return_annot is inspect.Signature.empty
            or return_annot is type(None)
            or return_annot is None
        ):
            # If there is no output, then output_sharding must be empty
            if output_sharding:
                raise ValueError(
                    f"Operator '{func.__qualname__}' has no output (return type is None), so output_sharding must be empty."
                )
        else:
            if not _is_fluxway_data_type(return_annot):
                raise TypeError(
                    f"Output of operator '{func.__qualname__}' must be a fluxway_data type (use @data_type), got {return_annot}"
                )

        # 3. Validate that the set of custom fields of all input classes contains the union of state_argument, tensor_argument, and static_argument
        # Only perform this check if there are input arguments
        if len(input_annots) > 0:
            all_input_fields = set()
            for _, tp in input_annots:
                # Assume all fields except those starting with "_" are custom fields
                try:
                    custom_fields = set()
                    # Try dataclass_fields
                    if hasattr(tp, "__dataclass_fields__"):
                        custom_fields = {
                            name
                            for name in tp.__dataclass_fields__
                            if not name.startswith("_")
                        }
                    # Try pydantic BaseModel
                    elif hasattr(tp, "__fields__"):
                        custom_fields = {
                            name for name in tp.__fields__ if not name.startswith("_")
                        }
                    # fallback: try __annotations__
                    elif hasattr(tp, "__annotations__"):
                        custom_fields = {
                            name
                            for name in tp.__annotations__
                            if not name.startswith("_")
                        }
                    else:
                        raise Exception()
                    all_input_fields.update(custom_fields)
                except Exception as err:
                    raise TypeError(
                        f"Cannot extract fields from argument class {tp} for operator '{func.__qualname__}'"
                    ) from err

            # The union set
            all_specified = state_args | tensor_args | static_args
            extra_in_specified = all_specified - all_input_fields
            if extra_in_specified:
                raise ValueError(
                    f"Arguments {extra_in_specified} specified in state_argument, tensor_argument, or static_argument "
                    f"do not exist in input data classes of operator '{func.__qualname__}'"
                )

        # 4. Validate that all keys in output_sharding correspond to output data fields
        if output_sharding:
            # Extract all output data fields (ignore fields starting with an underscore)
            output_fields = set()
            try:
                if hasattr(return_annot, "__dataclass_fields__"):
                    output_fields = {
                        name
                        for name in return_annot.__dataclass_fields__
                        if not name.startswith("_")
                    }
                elif hasattr(return_annot, "__fields__"):
                    output_fields = {
                        name
                        for name in return_annot.__fields__
                        if not name.startswith("_")
                    }
                elif hasattr(return_annot, "__annotations__"):
                    output_fields = {
                        name
                        for name in return_annot.__annotations__
                        if not name.startswith("_")
                    }
                else:
                    raise Exception()
            except Exception as err:
                raise TypeError(
                    f"Cannot extract fields from output class {return_annot} for operator '{func.__qualname__}'"
                ) from err
            output_sharding_keys = set(output_sharding.keys())
            extra_in_sharding = output_sharding_keys - output_fields
            if extra_in_sharding:
                raise ValueError(
                    f"output_sharding specifies keys {extra_in_sharding} which do not exist in the output data fields "
                    f"for operator '{func.__qualname__}'"
                )
        # Store raw sharding specs as plain dicts (serializable, mesh-independent)
        input_sharding_raw: Dict[str, Any] = {}
        if input_sharding:
            for tensor_name, raw_spec in input_sharding.items():
                spec_dict = (
                    raw_spec if isinstance(raw_spec, dict) else raw_spec.__dict__
                )
                if spec_dict.get("type") not in ("dense", "sparse"):
                    raise ValueError(
                        f"input_sharding['{tensor_name}'] type must be 'dense' or 'sparse'"
                    )
                input_sharding_raw[tensor_name] = spec_dict

        output_sharding_raw: Dict[str, Any] = {}
        if output_sharding:
            for tensor_name, raw_spec in output_sharding.items():
                spec_dict = (
                    raw_spec if isinstance(raw_spec, dict) else raw_spec.__dict__
                )
                if spec_dict.get("type") not in ("dense", "sparse"):
                    raise ValueError(
                        f"output_sharding['{tensor_name}'] type must be 'dense' or 'sparse'"
                    )
                output_sharding_raw[tensor_name] = spec_dict

        # Prepare output_arguments
        output_args = None
        if output_sharding:
            output_args = {ArgumentType.TENSOR.value: list(output_sharding.keys())}

        meta = OperatorMeta(
            name=name,
            input_sharding_raw=input_sharding_raw,
            output_sharding_raw=output_sharding_raw,
            input_arguments={
                ArgumentType.STATE.value: state_argument or [],
                ArgumentType.TENSOR.value: tensor_argument or [],
                ArgumentType.STATIC.value: static_argument or [],
            },
            output_arguments=output_args,
            eager_trigger=eager_trigger,
        )

        # Attach metadata to the underlying function
        # We attach it to both the wrapper and the original function if possible
        func._flux_operator_meta = meta

        # Check if function has 'self' parameter and convert to staticmethod
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        has_self = params and params[0].name == "self"

        if has_self:
            # Remove 'self' parameter from signature
            new_params = params[1:]  # Skip 'self'
            new_sig = sig.replace(parameters=new_params)

            # Get the class that contains this method for instance creation
            qualname_parts = func.__qualname__.split(".")
            module = inspect.getmodule(func)
            spmd_class = None
            if len(qualname_parts) >= 2 and module:
                class_name = qualname_parts[-2]
                spmd_class = getattr(module, class_name, None)

            # Create a wrapper function without 'self' parameter
            # Use thread-local storage to ensure each thread gets its own instance
            import threading

            _local = threading.local()

            def wrapper(*args, **kwargs):
                # Get or create thread-local instance
                if not hasattr(_local, "instance"):
                    if spmd_class:
                        try:
                            _local.instance = spmd_class()
                        except Exception:
                            # If instance creation fails, create a minimal object
                            class DummySelf:
                                pass

                            _local.instance = DummySelf()
                    else:

                        class DummySelf:
                            pass

                        _local.instance = DummySelf()

                # Call original function with instance as self
                return func(_local.instance, *args, **kwargs)

            # Update function metadata
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            wrapper.__module__ = func.__module__
            wrapper.__qualname__ = func.__qualname__
            wrapper.__signature__ = new_sig
            # Update annotations to exclude 'self'
            if hasattr(func, "__annotations__"):
                annotations = func.__annotations__.copy()
                annotations.pop("self", None)
                wrapper.__annotations__ = annotations

            # Preserve metadata
            wrapper._flux_operator_meta = meta

            # Convert to staticmethod
            return staticmethod(wrapper)
        else:
            # No 'self' parameter, return as-is but ensure it's a staticmethod
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            wrapper._flux_operator_meta = meta
            wrapper = staticmethod(wrapper)
            wrapper.__func__._flux_operator_meta = meta
            return wrapper

    return decorator


def data_type():
    """
    Decorator to mark a class as a fluxway transfer data type.

    Usage:
        @data_type()
        class ActionState:
            id: int
            tensor: torch.Tensor
            config: string

    Returns:
        Decorated class with metadata of fluxway transfer data.
    """
    T = TypeVar("T")

    def decorator(cls: Type[T]) -> Type[T]:
        # Validate all custom class variables (excluding python built-in variables), and ensure none of them are of dict type.
        # "Built-in variables" refer to those named like __xxx__.
        # Custom variables include class variables and annotation fields (excluding functions/methods).
        for var_name, var_value in cls.__dict__.items():
            if var_name.startswith("__") and var_name.endswith("__"):
                continue
            # Exclude methods/functions
            if callable(var_value):
                continue
            # Check if variable or its type hint is a dict
            # Type hint: check __annotations__ if present
            var_type = None
            if "__annotations__" in cls.__dict__ and var_name in cls.__annotations__:
                var_type = cls.__annotations__[var_name]
            if isinstance(var_value, dict):
                raise TypeError(
                    f"Class variable '{var_name}' is of dict type, which is not allowed for fluxway data class"
                )
            if var_type is not None:
                # allow typing.Dict or dict and similar
                origin = getattr(var_type, "__origin__", None)
                if (
                    origin is dict
                    or origin is Dict
                    or var_type is dict
                    or var_type is Dict
                ):
                    raise TypeError(
                        f"Class annotation '{var_name}' is of dict type, which is not allowed for fluxway data class"
                    )
        cls.is_fluxway_data = classmethod(lambda c: True)

        return cls

    return decorator


# =============================================================================
# SPMD Decorator (formerly in common/api/spmd.py)
# =============================================================================


def spmd(
    resource: Optional[Dict[str, Any]] = None,
    mesh_axis_names: Optional[Tuple[str, ...]] = None,
    mesh_axis_shapes: Optional[Tuple[int, ...]] = None,
    max_async_steps: int = 1,
    params_update_steps: int = 1,
    enable_output_step_remap: bool = False,
):
    """
    Decorator to mark a class with SPMD execution configuration.

    Usage:
        @spmd(
            resource={"nums_cpu": 4, "nums_gpu": 2},
            mesh_axis_names=("data", "model"),
            mesh_axis_shapes=(4, 2),
            max_async_steps=10
        )
        class MySPMD:
            def forward(self, state):
                return process(state)

    Args:
        resource: Resource requirements dict
        mesh_axis_names: Ordered tuple of mesh axis names (e.g., ("data", "model"))
        mesh_axis_shapes: Ordered tuple of mesh axis sizes (e.g., (4, 2))
        max_async_steps: Maximum async steps between workers
        params_update_steps: Sync barrier every N steps
        enable_output_step_remap: Whether to remap output step numbers (see SPMDConfig.enable_output_step_remap)

    Returns:
        Decorated class with SPMD configuration
    """
    from fluxway.common.types import (
        ResourceSpec,
        SPMDConfig,
        DeviceMesh,
    )  # Internal import

    T = TypeVar("T")

    # Parse resource specification
    resource_spec = None
    if resource:
        resource_spec = ResourceSpec(
            num_cpus=resource.get("nums_cpu", resource.get("num_cpus", 1)),
            num_gpus=resource.get("nums_gpu", resource.get("num_gpus", 0)),
            memory_gb=resource.get("memory", resource.get("memory_gb", 0.0)),
        )

    # Create DeviceMesh from axis tuples
    mesh = None
    if mesh_axis_names is not None and mesh_axis_shapes is not None:
        mesh = DeviceMesh(axis_names=mesh_axis_names, device_shape=mesh_axis_shapes)

    def decorator(cls: Type[T]) -> Type[T]:
        # Add attributes directly to the class
        cls._resource = resource_spec
        cls._mesh = mesh
        cls._max_async_steps = max_async_steps
        cls._params_update_steps = params_update_steps
        cls._enable_output_step_remap = enable_output_step_remap

        cls._enable_output_step_remap = enable_output_step_remap

        # Add methods directly to the class
        cls.get_resource = classmethod(lambda c: c._resource)
        cls.get_mesh = classmethod(lambda c: c._mesh)
        cls.get_max_async_steps = classmethod(lambda c: c._max_async_steps)
        cls.get_params_update_steps = classmethod(lambda c: c._params_update_steps)
        cls.get_enable_output_step_remap = classmethod(
            lambda c: getattr(c, "_enable_output_step_remap", False)
        )
        cls.is_fluxway_spmd = classmethod(lambda c: True)

        def get_operator_meta(c, op_type):
            """Get OperatorMeta for a given OpType (e.g. OpType.FORWARD)."""
            from fluxway.common.types import (
                OperatorMeta,
            )  # runtime import to avoid circular refs
            from fluxway.common.types.data_types import OpType

            if not isinstance(op_type, OpType):
                raise TypeError(f"op_type must be OpType, got {type(op_type)}")
            method_name = op_type.name.lower()
            method = getattr(c, method_name, None)
            if method is None:
                return None
            meta = getattr(method, "_flux_operator_meta", None)
            if isinstance(meta, OperatorMeta):
                return meta
            return None

        cls.get_operator_meta = classmethod(get_operator_meta)

        def get_spmd_config(c) -> SPMDConfig:
            from fluxway.common.types.data_types import OpType

            forward_fn = getattr(c, OpType.FORWARD.name.lower(), None)
            backward_fn = getattr(c, OpType.BACKWARD.name.lower(), None)
            update_fn = getattr(c, OpType.UPDATE.name.lower(), None)

            # Extract OperatorMeta explicitly - ensures it survives Ray serialization
            # (function attributes are lost when pickled by reference across processes)
            forward_meta = (
                getattr(forward_fn, "_flux_operator_meta", None) if forward_fn else None
            )
            backward_meta = (
                getattr(backward_fn, "_flux_operator_meta", None)
                if backward_fn
                else None
            )
            update_meta = (
                getattr(update_fn, "_flux_operator_meta", None) if update_fn else None
            )

            return SPMDConfig(
                mesh=c._mesh,
                user_forward_fn=forward_fn,
                user_backward_fn=backward_fn,
                user_update_fn=update_fn,
                forward_meta=forward_meta,
                backward_meta=backward_meta,
                update_meta=update_meta,
                max_async_steps=c._max_async_steps,
                params_update_steps=c._params_update_steps,
                enable_output_step_remap=c._enable_output_step_remap,
            )

        cls.get_spmd_config = classmethod(get_spmd_config)

        return cls

    return decorator
