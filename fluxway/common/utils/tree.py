import types as _types
from typing import Any, Union
from collections.abc import Callable


def _is_fluxway_data_type(tp: Any) -> bool:
    """Check if a type is marked with @data_type."""
    if tp is None:
        return False
    # Handle both typing.Union/Optional[T] and Python 3.10+ X | Y (types.UnionType)
    origin = getattr(tp, "__origin__", None)
    if origin is Union or isinstance(tp, _types.UnionType):
        args = getattr(tp, "__args__", [])
        return any(_is_fluxway_data_type(arg) for arg in args if arg is not type(None))

    return getattr(tp, "is_fluxway_data", lambda: False)()


def tree_flatten(
    tree: Any,
    is_leaf: Callable[[Any], bool] | None = None,
    treedef_template: Any | None = None,
) -> tuple[list[Any], Any]:
    """
    Flattens a nested structure (dict, list, tuple) into a list of leaves and a treedef.

    Args:
        tree: The nested structure to flatten.
        is_leaf: An optional predicate to determine if a node should be treated as a leaf.
        treedef_template: Optional structure template to guide flattening order.

    Returns:
        A tuple of (leaves, treedef).
    """
    leaves = []

    def _flatten(v, template=None):
        if is_leaf and is_leaf(v):
            leaves.append(v)
            return None

        # Template-guided flattening for data_type
        if _is_fluxway_data_type(type(v)):
            if template and template[0] == "dict":
                # Convert the data_type to a dict and let it fall through to the dict branch below
                if not hasattr(v, "as_dict"):
                    raise TypeError(
                        f"Fluxway Data Error: data_type {type(v).__name__} lacks 'as_dict'."
                    )
                v = v.as_dict()
            else:
                if template and template[0] != "data_type":
                    raise ValueError(
                        f"Type mismatch at runtime: expected {template[0]}, but "
                        f"operator returned data_type {type(v).__name__}."
                    )
                from typing import get_type_hints

                # If template matches this node, use its predefined key order
                if template and template[1] is type(v):
                    public_keys = template[2]
                    child_templates = template[3]
                else:
                    try:
                        hints = get_type_hints(type(v))
                        public_keys = [k for k in hints.keys() if not k.startswith("_")]
                    except Exception as err:
                        if not hasattr(v, "as_dict"):
                            raise TypeError(
                                f"Fluxway Data Error: data_type {type(v).__name__} lacks 'as_dict'."
                            ) from err
                        data_map = v.as_dict()
                        public_keys = sorted(
                            [k for k in data_map.keys() if not k.startswith("_")]
                        )
                    child_templates = [None] * len(public_keys)

                if not hasattr(v, "as_dict"):
                    raise TypeError(
                        f"Fluxway Data Error: data_type {type(v).__name__} lacks 'as_dict'."
                    )
                data_map = v.as_dict()

                children = []
                for k, child_template in zip(public_keys, child_templates, strict=True):
                    children.append(_flatten(data_map[k], child_template))
                return ("data_type", type(v), public_keys, children)

        # Template-guided flattening for dict
        if isinstance(v, dict):
            if template and template[0] != "dict":
                raise ValueError(
                    f"Type mismatch at runtime: expected {template[0]}, but "
                    f"operator returned a dictionary."
                )
            if template:
                keys = template[1]
                # Strict key set matching
                if set(keys) != set(v.keys()):
                    raise ValueError(
                        f"Operator output keys {sorted(v.keys())} do not match "
                        f"declared sharding keys {sorted(keys)}."
                    )
                child_templates = template[2]
            else:
                keys = sorted(v.keys())
                child_templates = [None] * len(keys)

            children = []
            for k, child_template in zip(keys, child_templates, strict=True):
                children.append(_flatten(v[k], child_template))
            return ("dict", keys, children)
        elif isinstance(v, list):
            if template and template[0] != "list":
                raise ValueError(
                    f"Type mismatch at runtime: expected {template[0]}, but "
                    f"operator returned a list."
                )
            child_templates = (
                template[2]
                if template and len(template[2]) == len(v)
                else [None] * len(v)
            )
            children = []
            for child, child_template in zip(v, child_templates, strict=True):
                children.append(_flatten(child, child_template))
            return ("list", len(v), children)
        elif isinstance(v, tuple):
            if template and template[0] != "tuple":
                raise ValueError(
                    f"Type mismatch at runtime: expected {template[0]}, but "
                    f"operator returned a tuple."
                )
            child_templates = (
                template[2]
                if template and len(template[2]) == len(v)
                else [None] * len(v)
            )
            children = []
            for child, child_template in zip(v, child_templates, strict=True):
                children.append(_flatten(child, child_template))
            return ("tuple", len(v), children)
        else:
            if template:
                raise ValueError(
                    f"Type mismatch at runtime: expected {template[0]}, but "
                    f"operator returned a leaf node (non-container)."
                )
            leaves.append(v)
            return None

    treedef = _flatten(tree, treedef_template)
    return leaves, treedef


def tree_flatten_type(cls: Any) -> Any:
    """
    Generate a treedef for a Type (specifically @data_type) without an instance.
    This is used for static contract generation.
    """

    if not _is_fluxway_data_type(cls):
        return None

    from typing import get_type_hints

    try:
        hints = get_type_hints(cls, include_extras=True)
        public_keys = [k for k in hints.keys() if not k.startswith("_")]
    except Exception:
        return None

    children = []
    for k in public_keys:
        hint = hints[k]
        if _is_fluxway_data_type(hint):
            children.append(tree_flatten_type(hint))
        else:
            children.append(None)  # Leaf

    return ("data_type", cls, public_keys, children)


def tree_unflatten(treedef: Any, leaves: list[Any]) -> Any:
    """
    Reconstructs a nested structure from its treedef and leaves.

    Args:
        treedef: The treedef returned by tree_flatten.
        leaves: The list of leaves returned by tree_flatten.

    Returns:
        The reconstructed nested structure.
    """
    it = iter(leaves)

    def _unflatten(d):
        if d is None:
            return next(it)

        type_name = d[0]
        if type_name == "data_type":
            cls, keys, children = d[1], d[2], d[3]
            field_values = {
                k: _unflatten(child)
                for k, child in zip(keys, children, strict=True)  # type: ignore
            }
            return cls(**field_values)
        elif type_name == "dict":
            keys, children = d[1], d[2]
            return {
                k: _unflatten(child)  # type: ignore
                for k, child in zip(keys, children, strict=True)  # type: ignore
            }
        elif type_name == "list":
            children = d[2]
            return [_unflatten(child) for child in children]
        elif type_name == "tuple":
            children = d[2]
            return tuple(_unflatten(child) for child in children)
        return None

    return _unflatten(treedef)
