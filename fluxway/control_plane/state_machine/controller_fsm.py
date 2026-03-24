"""
ControllerStateMachine for managing Controller lifecycle.

State transitions (event-driven initialization):
- INITIALIZING -> PENDING (on INIT_COMPLETE: resources allocated, SPMDs started)
- PENDING -> TOPOLOGY_REGISTERING (on ALL_DRIVERS_READY: all drivers report workers_ready)
    -> triggers topology registration
- TOPOLOGY_REGISTERING -> RUNNING (on ALL_SPMDS_TOPOLOGY_READY: all workers running)
    -> starts heartbeat timeout monitoring
- RUNNING -> BLOCKED (on DRIVER_ERROR, blocks all SPMDs, triggers restart)
- BLOCKED -> RUNNING (on DRIVER_RECOVERED)
- BLOCKED -> FAILED (on RECOVERY_FAILED)
- RUNNING -> COMPLETED (on ALL_DRIVERS_COMPLETED, reclaims resources)

Supports async callbacks for Ray Actor Controller.
"""

import asyncio
from typing import TYPE_CHECKING, Dict

from fluxway.common.types import ControllerState, ControllerEvent
from fluxway.common.utils.logging import get_logger

if TYPE_CHECKING:
    from fluxway.control_plane.controller import ControllerActor

logger = get_logger(__name__)


class ControllerStateMachine:
    """
    State machine for Controller lifecycle management.

    Handles state transitions based on events from heartbeat manager
    and driver reports. Supports async callbacks for Ray Actor.
    """

    # State transition table: (current_state, event) -> next_state
    _TRANSITIONS = {
        # --- Event-driven initialization flow ---
        (
            ControllerState.INITIALIZING,
            ControllerEvent.INIT_COMPLETE,
        ): ControllerState.PENDING,
        (
            ControllerState.PENDING,
            ControllerEvent.ALL_DRIVERS_READY,
        ): ControllerState.TOPOLOGY_REGISTERING,
        (
            ControllerState.TOPOLOGY_REGISTERING,
            ControllerEvent.ALL_SPMDS_TOPOLOGY_READY,
        ): ControllerState.RUNNING,
        # --- Normal operation ---
        (
            ControllerState.RUNNING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.BLOCKED,
        (
            ControllerState.BLOCKED,
            ControllerEvent.DRIVER_RECOVERED,
        ): ControllerState.RUNNING,
        (
            ControllerState.BLOCKED,
            ControllerEvent.RECOVERY_FAILED,
        ): ControllerState.FAILED,
        (
            ControllerState.RUNNING,
            ControllerEvent.ALL_DRIVERS_COMPLETED,
        ): ControllerState.COMPLETED,
        # # If task run super fast, RUNNING may not be reached, and ALL_DRIVERS_COMPLETED may be received.
        # (
        #     ControllerState.PENDING,
        #     ControllerEvent.ALL_DRIVERS_COMPLETED,
        # ): ControllerState.COMPLETED,
        (
            ControllerState.TOPOLOGY_REGISTERING,
            ControllerEvent.ALL_DRIVERS_COMPLETED,
        ): ControllerState.COMPLETED,
        # Recovery from BLOCKED
        (
            ControllerState.BLOCKED,
            ControllerEvent.ALL_DRIVERS_READY,
        ): ControllerState.RUNNING,
        # Errors during init phases
        (
            ControllerState.PENDING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.BLOCKED,
        (
            ControllerState.TOPOLOGY_REGISTERING,
            ControllerEvent.DRIVER_ERROR,
        ): ControllerState.BLOCKED,
    }

    def __init__(self, controller: "ControllerActor"):
        """
        Initialize ControllerStateMachine.

        Args:
            controller: Reference to the Controller instance
        """
        self._controller = controller
        self._state = ControllerState.INITIALIZING
        self._lock = asyncio.Lock()

        # Track error information
        self._error_spmds: Dict[str, str] = {}  # spmd_id -> error_message

    @property
    def state(self) -> ControllerState:
        """Get current state (sync access for simple reads)."""
        return self._state

    async def trigger_event(self, event: ControllerEvent, **kwargs) -> bool:
        """
        Trigger a state transition event.

        Args:
            event: The event to trigger
            **kwargs: Additional event data
                - spmd_id: SPMD ID (for DRIVER_ERROR, DRIVER_RECOVERED)
                - error: Error message (for DRIVER_ERROR)

        Returns:
            True if transition occurred, False otherwise
        """
        async with self._lock:
            current = self._state
            key = (current, event)

            if key not in self._TRANSITIONS:
                logger.warning(
                    f"Invalid state transition: {current} + {event}. Ignoring event."
                )
                return False

            next_state = self._TRANSITIONS[key]

            # Handle event-specific data
            if event == ControllerEvent.DRIVER_ERROR:
                spmd_id = kwargs.get("spmd_id", "unknown")
                error = kwargs.get("error", "Unknown error")
                self._error_spmds[spmd_id] = error
                logger.error(f"Driver error: {spmd_id} - {error}")

            elif event == ControllerEvent.DRIVER_RECOVERED:
                spmd_id = kwargs.get("spmd_id")
                if spmd_id and spmd_id in self._error_spmds:
                    del self._error_spmds[spmd_id]
                logger.debug(f"Driver recovered: {spmd_id}")

            # Perform state transition
            old_state = self._state
            self._state = next_state
            logger.debug(
                f"State transition: {old_state} -> {next_state} (event: {event})"
            )

        # Execute entry callback (outside lock to prevent deadlock)
        await self._execute_entry_callback(next_state)

        return True

    async def _execute_entry_callback(self, state: ControllerState) -> None:
        """Execute the entry callback for a state."""
        callbacks = {
            ControllerState.PENDING: self._on_enter_pending,
            ControllerState.TOPOLOGY_REGISTERING: self._on_enter_topology_registering,
            ControllerState.RUNNING: self._on_enter_running,
            ControllerState.BLOCKED: self._on_enter_blocked,
            ControllerState.COMPLETED: self._on_enter_completed,
            ControllerState.FAILED: self._on_enter_failed,
        }

        callback = callbacks.get(state)
        if callback:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Error in entry callback for {state}: {e}")

    def get_state(self) -> ControllerState:
        """Get current state (alias for property)."""
        return self.state

    def is_running(self) -> bool:
        """Check if controller is in RUNNING state."""
        return self.state == ControllerState.RUNNING

    def is_completed(self) -> bool:
        """Check if controller has completed."""
        return self.state in (ControllerState.COMPLETED, ControllerState.FAILED)

    def get_error_spmds(self) -> Dict[str, str]:
        """Get SPMDs currently in error state."""
        return dict(self._error_spmds)

    # =========================================================================
    # State entry callbacks (async)
    # =========================================================================

    async def _on_enter_pending(self) -> None:
        """Called when entering PENDING state.

        Resources are allocated and SPMDs are started (fire-and-forget).
        Waiting for all drivers to report workers_ready via heartbeat.
        """
        logger.debug(
            "Controller entering PENDING state, waiting for all drivers to report workers_ready"
        )

    async def _on_enter_topology_registering(self) -> None:
        """Called when entering TOPOLOGY_REGISTERING state.

        All drivers have reported workers_ready.
        Initiates topology registration on all SPMDs (fire-and-forget).
        Workers will auto-start after handler registration.
        """
        logger.debug(
            "Controller entering TOPOLOGY_REGISTERING state, initiating topology registration"
        )

        # Initiate topology registration (fire-and-forget, no await needed)
        try:
            await self._controller._initiate_topology_registration()
        except Exception as e:
            logger.error(f"Failed to initiate topology registration: {e}")

    async def _on_enter_running(self) -> None:
        """Called when entering RUNNING state.

        All workers are running after topology registration.
        Starts heartbeat timeout monitoring.
        """
        logger.debug("Controller entering RUNNING state")

        # Start heartbeat timeout monitoring now that system is fully initialized
        try:
            await self._controller._heartbeat_manager.start_monitoring_async()
            logger.debug("Heartbeat timeout monitoring started")
        except Exception as e:
            logger.error(f"Failed to start heartbeat monitoring: {e}")

    async def _on_enter_blocked(self) -> None:
        """
        Called when entering BLOCKED state.

        Blocks all SPMDs and initiates recovery for error SPMDs.
        """
        logger.warning("Controller entering BLOCKED state")

        # Block all SPMDs
        try:
            await self._controller.block_all_spmds()
        except Exception as e:
            logger.error(f"Failed to block all SPMDs: {e}")

        # Initiate recovery for error SPMDs
        error_spmds = list(self._error_spmds.keys())

        for spmd_id in error_spmds:
            try:
                await self._controller.restart_spmd(spmd_id)
            except Exception as e:
                logger.error(f"Failed to restart SPMD {spmd_id}: {e}")

    async def _on_enter_completed(self) -> None:
        """
        Called when entering COMPLETED state.

        Reclaims all SPMD resources and performs cleanup.
        """
        logger.debug("Controller entering COMPLETED state")

        try:
            await self._controller._cleanup_all_spmds()
        except Exception as e:
            logger.error(f"Failed to cleanup SPMDs: {e}")

    async def _on_enter_failed(self) -> None:
        """
        Called when entering FAILED state.

        Handles unrecoverable failures.
        """
        logger.error("Controller entering FAILED state")

        errors = dict(self._error_spmds)
        logger.error(f"Failed SPMDs: {errors}")

        # Attempt to cleanup resources even on failure
        try:
            await self._controller._cleanup_all_spmds()
        except Exception as e:
            logger.error(f"Failed to cleanup SPMDs during failure handling: {e}")

    async def reset(self) -> None:
        """Reset state machine to initial state."""
        async with self._lock:
            self._state = ControllerState.INITIALIZING
            self._error_spmds.clear()
        logger.debug("Controller state machine reset")
