import copy
from typing import Any

from fluxway.common.decorators import operator, spmd, spmd_source


def _apply_operator_configs(
    spmd_config: Any,
    operator_configs: dict[str, dict[str, Any]],
    spmd_class: type,
) -> None:
    from fluxway.common.types.configs import ArgumentType

    for op_name, op_cfg in operator_configs.items():
        existing_meta = spmd_config.operator_metas.get(op_name)

        if existing_meta is not None:
            if "input_sharding" in op_cfg:
                existing_meta.input_sharding = op_cfg["input_sharding"]
            if "output_sharding" in op_cfg:
                existing_meta.output_sharding = op_cfg["output_sharding"]

            if "input_sharding" in op_cfg and existing_meta.input_arguments:
                data_keys = list(op_cfg["input_sharding"].keys())
                existing_data = existing_meta.input_arguments.get(
                    ArgumentType.DATA.value, []
                )
                for k in data_keys:
                    if k not in existing_data:
                        existing_data.append(k)
                existing_meta.input_arguments[ArgumentType.DATA.value] = existing_data

                for k in data_keys:
                    existing_meta.input_param_map.setdefault(k, [k])
                    existing_meta.input_treedefs.setdefault(k, None)

            if "output_sharding" in op_cfg and existing_meta.output_arguments:
                data_keys = list(op_cfg["output_sharding"].keys())
                existing_data = existing_meta.output_arguments.get(
                    ArgumentType.DATA.value, []
                )
                for k in data_keys:
                    if k not in existing_data:
                        existing_data.append(k)
                existing_meta.output_arguments[ArgumentType.DATA.value] = existing_data

                for k in data_keys:
                    existing_meta.output_param_map.setdefault(k, [k])
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
            meta = decorated._fluxway_operator_meta

            setattr(spmd_class, op_name, decorated)
            spmd_config.operators[op_name] = op_name
            spmd_config.operator_metas[op_name] = copy.deepcopy(meta)


def bind_spmd(
    cls: type,
    *,
    resource: dict[str, Any] | None = None,
    mesh_axis_names: tuple[str, ...] | None = None,
    mesh_axis_shapes: tuple[int, ...] | None = None,
    runtime_env: dict[str, Any] | None = None,
    is_source: bool = False,
    interleaved: bool = False,
    interleave_unit_size: int = 1,
    **operator_configs: dict[str, Any],
) -> type:
    dynamic_cls = type(cls.__name__, (cls,), {})

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

        decorated = operator(name=op_name, **op_cfg)(raw_func)
        setattr(dynamic_cls, op_name, decorated)

    if is_source:
        spmd_dec = spmd_source(
            resource=resource,
            mesh_axis_names=mesh_axis_names,
            mesh_axis_shapes=mesh_axis_shapes,
            interleaved=interleaved,
            interleave_unit_size=interleave_unit_size,
            runtime_env=runtime_env,
        )
    else:
        spmd_dec = spmd(
            resource=resource,
            mesh_axis_names=mesh_axis_names,
            mesh_axis_shapes=mesh_axis_shapes,
            runtime_env=runtime_env,
        )

    return spmd_dec(dynamic_cls)
