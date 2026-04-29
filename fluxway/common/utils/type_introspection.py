"""
Type introspection helpers.

These utilities are intentionally framework-agnostic and can be reused by both
control-plane and worker components.
"""

from __future__ import annotations

import typing


def infer_fluxway_output_fields(spmd_class: type, method_name: str) -> list[str]:
    """
    Infer output data fields from an operator return annotation.

    Assumes return type is a @data_type decorated class (already validated upstream).
    Supports Optional[T] / Union[T, None] by unwrapping to T.
    """
    method = getattr(spmd_class, method_name, None)
    if method is None:
        return []

    ret = getattr(method, "__annotations__", {}).get("return")
    if ret is None:
        return []

    # unwrap Optional/Union[T, None]
    origin = getattr(ret, "__origin__", None)
    if origin is typing.Union:
        args = [a for a in getattr(ret, "__args__", []) if a is not type(None)]
        if len(args) == 1:
            ret = args[0]

    ann = getattr(ret, "__annotations__", None)
    if isinstance(ann, dict):
        return list(ann.keys())

    # fallback: custom class vars
    fields: list[str] = []
    for var_name, var_value in getattr(ret, "__dict__", {}).items():
        if var_name.startswith("__") and var_name.endswith("__"):
            continue
        if callable(var_value):
            continue
        fields.append(var_name)
    return fields
