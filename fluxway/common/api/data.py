"""
Fluxway Data Type decorator.

Provides the @data_type decorator to mark user-defined classes
with dual-stream transmission metadata.
"""

from typing import List, Type, TypeVar

T = TypeVar('T')


def data_type(
    state_argument: List[str] = None,
    tensor_argument: List[str] = None,
    static_argument: List[str] = None
):
    """
    Decorator to mark a class with dual-stream transmission metadata.
    
    Usage:
        @data_type(
            state_argument=["id"], 
            tensor_argument=["tensor"],
            static_argument=["config"]
        )
        class ActionState:
            id: int
            tensor: torch.Tensor
            config: string
    
    Args:
        state_argument: List of field names to transmit via control plane
        tensor_argument: List of field names to transmit via data plane
        static_argument: List of field names that are static and not transmitted
        
    Returns:
        Decorated class with transmission metadata
    """
    state_args = state_argument or []
    tensor_args = tensor_argument or []
    static_args = static_argument or []
    
    def decorator(cls: Type[T]) -> Type[T]:
        # Validate that specified arguments exist in the class
        original_annotations = getattr(cls, '__annotations__', {})
        all_specified = set(state_args + tensor_args + static_args)
        for arg_name in all_specified:
            if arg_name not in original_annotations:
                raise ValueError(
                    f"Field '{arg_name}' specified in decorator but not found in class '{cls.__name__}'"
                )
            # Check if field type is dict, which is not allowed
            field_type = original_annotations[arg_name]
            if field_type is dict or (hasattr(field_type, '__origin__') and field_type.__origin__ is dict):
                raise TypeError(
                    f"Field '{arg_name}' in class '{cls.__name__}' has type 'dict' which is not allowed. "
                    f"Use a typed dataclass or Pydantic model instead."
                )
        
        # Add attributes and methods directly to the class
        cls._state_arguments = state_args
        cls._tensor_arguments = tensor_args
        cls._static_arguments = static_args
        cls.get_state_arguments = classmethod(lambda c: c._state_arguments)
        cls.get_tensor_arguments = classmethod(lambda c: c._tensor_arguments)
        cls.get_static_arguments = classmethod(lambda c: c._static_arguments)
        cls.is_fluxway_data = classmethod(lambda c: True)
        
        return cls
    
    return decorator
