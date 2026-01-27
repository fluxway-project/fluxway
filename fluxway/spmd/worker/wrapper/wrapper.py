"""
User Module Wrapper for forward/backward execution.

Based on ARCHITECTURE.md Section 3.2.4

Responsibilities:
- Wrap user's forward/backward functions
- Implement dual-plane propagation protocol routing logic
- Separate Data Plane and Control Plane parameters
- Manage async execution and sync points
- Handle data transmission state management
- Auto-aggregate Sync Output (Loss, Gradient Norm)
"""

from typing import Any, Callable, Dict, Optional

from .type_router import TypeRouter


class ForwardWrapper:
    """
    Wrapper for user forward function.

    Implements dual-plane routing for forward pass.

    Key design:
    - Unified interface: User function signature stays unified
    - Auto routing: Wrapper routes params to different planes based on decorators
    - Async execution: Data Plane params received and sent async
    - Auto aggregation: Sync Output auto AllReduce in SPMD
    - Framework agnostic: Adapts to JAX/PyTorch/NumPy via abstract interface
    """

    def __init__(self, user_fn: Callable, type_router: TypeRouter):
        self._user_fn = user_fn
        self._type_router = type_router

    async def __call__(self, step: int) -> Any:
        """
        Execute wrapped forward function.

        1. Async receive Data Plane params (ULP protocol)
        2. Get Control Plane params from Orchestrator
        3. Wait for Data param reception complete
        4. Merge params, call user forward function
        5. Route output: separate data_outputs and sync_outputs
        6. Async send Data Output to successor SPMD
        7. Aggregate Sync Output (AllReduce) then send to main control
        8. Non-blocking send completion signal to FSM

        TODO: Implement all steps above
        """
        pass

    def set_data_receiver(self, receiver: Any) -> None:
        """
        Set receiver for Data Plane parameters.

        TODO: Store ULP receiver reference
        """
        pass

    def set_control_provider(self, provider: Any) -> None:
        """
        Set provider for Control Plane parameters.

        TODO: Store Control Plane provider reference
        """
        pass

    def set_output_sender(self, sender: Any) -> None:
        """
        Set sender for output data.

        TODO: Store ULP sender reference
        """
        pass


class BackwardWrapper:
    """
    Wrapper for user backward function.

    Implements dual-plane routing for backward pass.
    """

    def __init__(self, user_fn: Callable, type_router: TypeRouter):
        self._user_fn = user_fn
        self._type_router = type_router

    async def __call__(self, step: int) -> Any:
        """
        Execute wrapped backward function.

        1. Async receive gradient data (Data Plane)
        2. Get Control Plane params
        3. Wait for gradient reception complete
        4. Call user backward function
        5. Route output: separate gradients (Data) and aggregates (Sync)
        6. Async send gradients to predecessor SPMD
        7. Aggregate Sync Output (e.g., AllReduce Loss)
        8. Non-blocking send completion signal to FSM

        TODO: Implement all steps above
        """
        pass

    def set_grad_receiver(self, receiver: Any) -> None:
        """
        Set receiver for gradient data.

        TODO: Store gradient receiver reference
        """
        pass

    def set_control_provider(self, provider: Any) -> None:
        """
        Set provider for Control Plane parameters.

        TODO: Store Control Plane provider reference
        """
        pass

    def set_grad_sender(self, sender: Any) -> None:
        """
        Set sender for gradient output.

        TODO: Store gradient sender reference
        """
        pass
