from typing import Any

from fluxway.common.decorators import operator, spmd, spmd_source


def _apply_operator_configs(
    spmd_config: Any,
    operator_configs: dict[str, dict[str, Any]],
    spmd_class: type,
) -> None:
    from fluxway.common.operator_policy import ArgumentType

    for op_name, op_cfg in operator_configs.items():
        existing_spec = spmd_config.operator_specs.get(op_name)

        if existing_spec is not None:
            updates: dict[str, Any] = {}
            if "input_sharding" in op_cfg:
                updates["input_sharding"] = op_cfg["input_sharding"]
            if "output_sharding" in op_cfg:
                updates["output_sharding"] = op_cfg["output_sharding"]
            if "tensor_access" in op_cfg:
                updates["tensor_access"] = op_cfg["tensor_access"]

            if "input_sharding" in op_cfg and existing_spec.input_arguments:
                data_keys = list(op_cfg["input_sharding"].keys())
                input_arguments = dict(existing_spec.input_arguments)
                existing_data = list(input_arguments.get(ArgumentType.DATA.value, ()))
                for k in data_keys:
                    if k not in existing_data:
                        existing_data.append(k)
                input_arguments[ArgumentType.DATA.value] = tuple(existing_data)
                updates["input_arguments"] = input_arguments

                input_param_map = dict(existing_spec.input_param_map)
                input_treedefs = dict(existing_spec.input_treedefs)
                for k in data_keys:
                    input_param_map.setdefault(k, (k,))
                    input_treedefs.setdefault(k, None)
                updates["input_param_map"] = input_param_map
                updates["input_treedefs"] = input_treedefs

            if "output_sharding" in op_cfg and existing_spec.output_arguments:
                data_keys = list(op_cfg["output_sharding"].keys())
                output_arguments = dict(existing_spec.output_arguments)
                existing_data = list(output_arguments.get(ArgumentType.DATA.value, ()))
                for k in data_keys:
                    if k not in existing_data:
                        existing_data.append(k)
                output_arguments[ArgumentType.DATA.value] = tuple(existing_data)
                updates["output_arguments"] = output_arguments

                output_param_map = dict(existing_spec.output_param_map)
                for k in data_keys:
                    output_param_map.setdefault(k, (k,))
                updates["output_param_map"] = output_param_map
            spmd_config.operator_specs[op_name] = existing_spec.updated(**updates)
        else:
            method = getattr(spmd_class, op_name, None)
            if method is None:
                raise ValueError(
                    f"Cannot register operator '{op_name}': "
                    f"method not found on class '{spmd_class.__name__}'"
                )

            raw_func = method
            if hasattr(method, "func"):
                raw_func = method.func
            elif hasattr(method, "__func__"):
                raw_func = method.__func__

            decorated = operator(name=op_name, **op_cfg)(raw_func)
            spec = decorated._fluxway_operator_spec

            setattr(spmd_class, op_name, decorated)
            spmd_config.operator_specs[op_name] = spec.updated(owner=spmd_class)


def bind_spmd(
    cls: type,
    *,
    resource_profile: Any | None = None,
    mesh_axis_names: tuple[str, ...] | None = None,
    mesh_axis_shapes: tuple[int, ...] | None = None,
    runtime_env: dict[str, Any] | None = None,
    driver_runtime_env: dict[str, Any] | None = None,
    is_source: bool = False,
    interleaved: bool = False,
    interleave_unit_size: int = 1,
    checkpoint_components: tuple[Any, ...] = (),
    heartbeat_interval_s: float = 5.0,
    execution_progress_warning_s: float = 90.0,
    heartbeat_timeout_s: float = 90.0,
    max_rank_step_skew: int = 0,
    **operator_configs: dict[str, Any],
) -> type:
    dynamic_cls = type(
        cls.__name__,
        (cls,),
        {"__module__": cls.__module__, "__qualname__": cls.__qualname__},
    )
    source_descriptor: Any | None = None

    for op_name, op_cfg in operator_configs.items():
        original_method = getattr(cls, op_name, None)
        if original_method is None:
            raise ValueError(
                f"Cannot bind operator '{op_name}': "
                f"method not found on class '{cls.__name__}'"
            )

        raw_func = original_method
        if hasattr(original_method, "func"):
            raw_func = original_method.func
        elif hasattr(original_method, "__func__"):
            raw_func = original_method.__func__

        effective_op_config = dict(op_cfg)
        if is_source:
            if effective_op_config.get("is_source") is False:
                raise ValueError(
                    f"Source SPMD operator {op_name!r} cannot set is_source=False"
                )
            effective_op_config["is_source"] = True
            configured_source = effective_op_config.get("source")
            if configured_source is not None:
                if source_descriptor is None:
                    source_descriptor = configured_source
                elif configured_source is not source_descriptor:
                    raise ValueError(
                        "source SPMD operators must share one source descriptor"
                    )
        decorated = operator(name=op_name, **effective_op_config)(raw_func)
        setattr(dynamic_cls, op_name, decorated)

    if is_source:
        spmd_dec = spmd_source(
            resource_profile=resource_profile,
            mesh_axis_names=mesh_axis_names,
            mesh_axis_shapes=mesh_axis_shapes,
            max_rank_step_skew=max_rank_step_skew,
            interleaved=interleaved,
            interleave_unit_size=interleave_unit_size,
            source=source_descriptor,
            runtime_env=runtime_env,
            driver_runtime_env=driver_runtime_env,
            checkpoint_components=checkpoint_components,
            heartbeat_interval_s=heartbeat_interval_s,
            execution_progress_warning_s=execution_progress_warning_s,
            heartbeat_timeout_s=heartbeat_timeout_s,
        )
    else:
        spmd_dec = spmd(
            resource_profile=resource_profile,
            mesh_axis_names=mesh_axis_names,
            mesh_axis_shapes=mesh_axis_shapes,
            max_rank_step_skew=max_rank_step_skew,
            runtime_env=runtime_env,
            driver_runtime_env=driver_runtime_env,
            checkpoint_components=checkpoint_components,
            heartbeat_interval_s=heartbeat_interval_s,
            execution_progress_warning_s=execution_progress_warning_s,
            heartbeat_timeout_s=heartbeat_timeout_s,
        )

    return spmd_dec(dynamic_cls)
