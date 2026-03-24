"""Control Plane: Core coordination layer for the training framework."""

# Ensure INFO logs are visible even when importing this subpackage directly.
from fluxway.common.utils.logging import ensure_logging_configured


from .graph import GraphParser, Builder
from .scheduler import Scheduler
from .spmd_manager import SPMDManager, SPMDHandle
from .state_manager import StateManager
from .monitor import Monitor
from .heartbeat import HeartbeatManager
from .state_machine import ControllerStateMachine
from .async_manager import AsyncStepManager
from .controller import Controller, create_controller
from .checkpointer import FluxwayCheckpointer, OrbaxCheckpointer, SafeTensorCheckpointer

try:
    from .controller import ControllerActor
except ImportError:
    # Ray not available
    ControllerActor = None  # type: ignore

__all__ = [
    "GraphParser",
    "Builder",
    "Scheduler",
    "SPMDManager",
    "SPMDHandle",
    "StateManager",
    "Monitor",
    "HeartbeatManager",
    "ControllerStateMachine",
    "AsyncStepManager",
    "Controller",
    "ControllerActor",
    "create_controller",
    "FluxwayCheckpointer",
    "OrbaxCheckpointer",
    "SafeTensorCheckpointer",
]

ensure_logging_configured()
