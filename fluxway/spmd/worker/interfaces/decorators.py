"""
Decorators for user module function type annotations.

Based on ARCHITECTURE.md Section 3.2.3

Input type annotations:
- data_inputs: Worker-independent/async data, goes to Data Plane (ULP)
- control_inputs: Worker-shared replicated or sync data, goes to Control Plane
- sync_inputs: Data needing aggregation, AllReduce within SPMD first

Output type annotations:
- data_outputs: Data to send to downstream SPMD
- sync_outputs: Data needing aggregation (e.g., Loss) before sending
"""

from functools import wraps
from typing import Callable, List, Optional


def input_type(
    data_inputs: Optional[List[str]] = None,
    control_inputs: Optional[List[str]] = None,
    sync_inputs: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator to declare input parameter transmission types.

    Args:
        data_inputs: Parameters going through Data Plane (ULP protocol)
                    e.g., Sharded Tensors from predecessor SPMD
        control_inputs: Parameters going through Control Plane
                       e.g., learning_rate, step, shared configs
        sync_inputs: Parameters needing AllReduce before use

    Example:
        @input_type(data_inputs=['input0', 'input1'],
                   control_inputs=['learning_rate', 'step'])
        def forward(input0, input1, learning_rate, step):
            ...

    TODO: Implement decorator that stores metadata on function
    TODO: Support framework-specific handling (JAX static_argnums, etc.)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Store metadata on function
        wrapper._input_type_metadata = {
            "data_inputs": data_inputs or [],
            "control_inputs": control_inputs or [],
            "sync_inputs": sync_inputs or [],
        }
        return wrapper

    return decorator


def output_type(
    data_outputs: Optional[List[str]] = None, sync_outputs: Optional[List[str]] = None
) -> Callable:
    """
    Decorator to declare output transmission types.

    Args:
        data_outputs: Outputs going through Data Plane
        sync_outputs: Outputs needing AllReduce (e.g., loss)

    Example:
        @output_type(data_outputs=['grad_input'], sync_outputs=['loss'])
        def backward(grad, clip_norm):
            ...

    TODO: Implement decorator that stores metadata on function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Store metadata on function
        wrapper._output_type_metadata = {
            "data_outputs": data_outputs or [],
            "sync_outputs": sync_outputs or [],
        }
        return wrapper

    return decorator
