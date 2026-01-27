"""
Base interface for user-defined modules.

Based on ARCHITECTURE.md Section 3.2.3

User interface design follows "logically unified, physically separated" principle.
Users define pure computation logic without worrying about data transmission.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from .decorators import input_type, output_type


class UserModuleInterface(ABC):
    """
    Abstract base class for user-defined modules.

    Users should inherit from this class and implement forward/backward/apply_updates.

    Design principles:
    - Users only define pure computation logic
    - Users can customize any function inputs
    - For compilation optimization, forward/backward forbid Dict, *args, **kwargs
    - Use decorators to distinguish Data Plane and Control Plane params
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tuple[Any, ...]:
        """
        User-defined forward logic.

        Use @input_type decorator to specify param types:
        - data_inputs: Sharded Tensors from predecessor SPMD
        - control_inputs: Shared variables like learning_rate, step

        Returns:
            Output data tuple (will be sent to successor SPMD)

        Example:
            @input_type(data_inputs=['input0', 'input1'],
                       control_inputs=['learning_rate', 'step'])
            def forward(self, input0, input1, learning_rate, step):
                ...
                return (output0, output1)
        """
        pass

    @abstractmethod
    def backward(self, *args, **kwargs) -> Tuple[Any, ...]:
        """
        User-defined backward logic.

        Typically computes two gradients from one gradient:
        - One for continued backpropagation
        - One for parameter update

        Use @output_type decorator to specify output types:
        - data_outputs: Gradients for predecessor SPMD
        - sync_outputs: Values needing AllReduce (e.g., loss)

        Returns:
            Tuple of gradients for predecessor SPMD

        Example:
            @input_type(data_inputs=['grad'], control_inputs=['clip_norm'])
            @output_type(data_outputs=['grad_input'], sync_outputs=['loss'])
            def backward(self, grad, clip_norm):
                ...
                return (grad_input, loss)
        """
        pass

    def apply_updates(self, *args, **kwargs) -> Dict[str, Any]:
        """
        User-defined parameter update logic.

        May return some update metrics.
        Generally triggered by Sub FSM via Stateful Executor,
        or can be triggered by backward.

        Note: If triggered by backward, user needs to maintain
        Step state internally in UserCustomModule.

        Returns:
            Update metrics dict

        Example:
            @input_type(control_inputs=['learning_rate'])
            def apply_updates(self, learning_rate):
                ...
                return {'grad_norm': grad_norm, 'param_norm': param_norm}
        """
        return {}

    def save_states(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Save state for checkpointing.

        TODO: Implement state serialization
        TODO: Include model params, optimizer state, RNG seeds
        """
        pass

    def load_states(self, state: Dict[str, Any], *args, **kwargs) -> None:
        """
        Load state from checkpoint.

        TODO: Implement state deserialization
        """
        pass
