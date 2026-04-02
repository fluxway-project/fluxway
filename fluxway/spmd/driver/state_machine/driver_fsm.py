"""
DriverStateMachine for managing SPMD Driver lifecycle.

State transitions:
- INITIALIZING -> READY (on INIT_COMPLETE)
- INITIALIZING -> ERROR (on INIT_FAILED)
- READY -> RUNNING (on START)
- RUNNING -> BLOCKED (on BLOCK_REQUESTED)
- RUNNING -> ERROR (on WORKER_ERROR that can't be recovered)
- BLOCKED -> RUNNING (on UNBLOCK_REQUESTED)
- RUNNING -> RECOVERING (on WORKER_ERROR that triggers recovery)
- RECOVERING -> RUNNING (on WORKER_RECOVERED)
- RECOVERING -> ERROR (on recovery failed)
- RUNNING -> COMPLETED (on ALL_WORKERS_COMPLETED or COMPLETE)
"""

import logging
import threading
from typing import TYPE_CHECKING, Callable, Dict, Optional

from fluxway.common.types import DriverState, DriverEvent

if TYPE_CHECKING:
    from fluxway.spmd.driver import SPMDDriver

logger = logging.getLogger(__name__)


class DriverStateMachine:
    """
    State machine for SPMD Driver lifecycle management.

    Handles state transitions based on events from workers and controller.
    """

    # State transition table: (current_state, event) -> next_state
    _TRANSITIONS = {
        (DriverState.INITIALIZING, DriverEvent.INIT_COMPLETE): DriverState.READY,
        (DriverState.INITIALIZING, DriverEvent.INIT_FAILED): DriverState.ERROR,
        (DriverState.READY, DriverEvent.START): DriverState.RUNNING,
        (DriverState.RUNNING, DriverEvent.BLOCK_REQUESTED): DriverState.BLOCKED,
        (DriverState.RUNNING, DriverEvent.WORKER_ERROR): DriverState.RECOVERING,
        (DriverState.BLOCKED, DriverEvent.UNBLOCK_REQUESTED): DriverState.RUNNING,
        (DriverState.BLOCKED, DriverEvent.WORKER_ERROR): DriverState.RECOVERING,
        (DriverState.RECOVERING, DriverEvent.WORKER_RECOVERED): DriverState.RUNNING,
        (DriverState.RECOVERING, DriverEvent.INIT_FAILED): DriverState.ERROR,
        (DriverState.RUNNING, DriverEvent.ALL_WORKERS_COMPLETED): DriverState.COMPLETED,
        (DriverState.RUNNING, DriverEvent.COMPLETE): DriverState.COMPLETED,
        # Drain → Drained → Snapshot → Running
        (DriverState.RUNNING, DriverEvent.DRAIN_REQUESTED): DriverState.DRAINING,
        (DriverState.READY, DriverEvent.DRAIN_REQUESTED): DriverState.DRAINING,
        (DriverState.DRAINING, DriverEvent.DRAIN_COMPLETE): DriverState.DRAINED,
        (DriverState.DRAINING, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        (DriverState.DRAINED, DriverEvent.SNAPSHOT_REQUESTED): DriverState.SNAPSHOTTING,
        (DriverState.DRAINED, DriverEvent.WORKER_ERROR): DriverState.ERROR,
        # Allow resuming even if we didn't snapshot
        (DriverState.DRAINED, DriverEvent.SNAPSHOT_COMPLETE): DriverState.RUNNING,
        (DriverState.SNAPSHOTTING, DriverEvent.SNAPSHOT_COMPLETE): DriverState.RUNNING,
        (DriverState.SNAPSHOTTING, DriverEvent.SNAPSHOT_FAILED): DriverState.ERROR,
        (DriverState.SNAPSHOTTING, DriverEvent.WORKER_ERROR): DriverState.ERROR,
    }

    def __init__(self, driver: "SPMDDriver"):
        """
        Initialize DriverStateMachine.

        Args:
            driver: Reference to the SPMDDriver instance
        """
        self._driver = driver
        self._state = DriverState.INITIALIZING
        self._lock = threading.Lock()

        # Track error information
        self._error_workers: Dict[int, str] = {}  # rank -> error_message
        self._error_message: Optional[str] = None

        # Callbacks for state entry
        self._entry_callbacks: Dict[DriverState, Callable[[], None]] = {
            DriverState.READY: self._on_enter_ready,
            DriverState.RUNNING: self._on_enter_running,
            DriverState.BLOCKED: self._on_enter_blocked,
            DriverState.DRAINING: self._on_enter_draining,
            DriverState.DRAINED: self._on_enter_drained,
            DriverState.SNAPSHOTTING: self._on_enter_snapshotting,
            DriverState.RECOVERING: self._on_enter_recovering,
            DriverState.COMPLETED: self._on_enter_completed,
            DriverState.ERROR: self._on_enter_error,
        }

    @property
    def state(self) -> DriverState:
        """Get current state."""
        with self._lock:
            return self._state

    def trigger(self, event: DriverEvent, **kwargs) -> bool:
        """
        Trigger a state transition event.

        Args:
            event: The event to trigger
            **kwargs: Additional event data
                - rank: Worker rank (for WORKER_ERROR, WORKER_RECOVERED)
                - error: Error message (for WORKER_ERROR, INIT_FAILED)

        Returns:
            True if transition occurred, False otherwise
        """
        with self._lock:
            current = self._state
            key = (current, event)

            # Guard: BLOCK/UNBLOCK must not interrupt drain or snapshot
            if event in (
                DriverEvent.BLOCK_REQUESTED,
                DriverEvent.UNBLOCK_REQUESTED,
            ) and current in (
                DriverState.DRAINING,
                DriverState.DRAINED,
                DriverState.SNAPSHOTTING,
            ):
                logger.warning(
                    f"Ignoring {event} in {current} state spmd_id={self._driver.spmd_id}"
                )
                return False

            if key not in self._TRANSITIONS:
                logger.debug(f"No transition for: {current} + {event}. Ignoring event.")
                return False

            next_state = self._TRANSITIONS[key]

            # Handle event-specific data
            if event == DriverEvent.WORKER_ERROR:
                rank = kwargs.get("rank", -1)
                error = kwargs.get("error", "Unknown error")
                self._error_workers[rank] = error
                self._error_message = f"Worker {rank}: {error}"
                logger.error(
                    f"Worker error in SPMD {self._driver.spmd_id}: {self._error_message}"
                )

            elif event == DriverEvent.WORKER_RECOVERED:
                rank = kwargs.get("rank")
                if rank is not None and rank in self._error_workers:
                    del self._error_workers[rank]
                logger.debug(f"Worker {rank} recovered in SPMD {self._driver.spmd_id}")

            elif event == DriverEvent.INIT_FAILED:
                error = kwargs.get("error", "Initialization failed")
                self._error_message = error
                logger.error(f"Init failed for SPMD {self._driver.spmd_id}: {error}")

            # Perform state transition
            old_state = self._state
            self._state = next_state
            logger.debug(
                f"SPMD {self._driver.spmd_id} state transition: "
                f"{old_state} -> {next_state} (event: {event})"
            )

        # Execute entry callback (outside lock to prevent deadlock)
        entry_callback = self._entry_callbacks.get(next_state)
        if entry_callback:
            try:
                entry_callback()
            except Exception as e:
                logger.error(f"Error in entry callback for {next_state}: {e}")

        return True

    def get_state(self) -> DriverState:
        """Get current state (alias for property)."""
        return self.state

    def is_running(self) -> bool:
        """Check if driver is in RUNNING state."""
        return self.state == DriverState.RUNNING

    def is_ready(self) -> bool:
        """Check if driver is in READY state."""
        return self.state == DriverState.READY

    def is_error(self) -> bool:
        """Check if driver is in ERROR state."""
        return self.state == DriverState.ERROR

    def is_completed(self) -> bool:
        """Check if driver has completed."""
        return self.state == DriverState.COMPLETED

    def get_error_workers(self) -> Dict[int, str]:
        """Get workers currently in error state."""
        with self._lock:
            return dict(self._error_workers)

    def get_error_message(self) -> Optional[str]:
        """Get the latest error message."""
        with self._lock:
            return self._error_message

    # =========================================================================
    # State entry callbacks
    # =========================================================================

    def _on_enter_ready(self) -> None:
        """Called when entering READY state.

        Note: Heartbeat is already started in initialize() (before READY),
        so we do NOT call _start_heartbeat() here.
        """
        logger.debug(f"SPMD {self._driver.spmd_id} is ready")

    def _on_enter_running(self) -> None:
        """Called when entering RUNNING state."""
        logger.debug(f"SPMD {self._driver.spmd_id} is running")

    def _on_enter_blocked(self) -> None:
        """Called when entering BLOCKED state."""
        logger.warning(f"SPMD {self._driver.spmd_id} is blocked")
        # Block all workers
        self._driver._block_all_workers()

    def _on_enter_draining(self) -> None:
        """Called when entering DRAINING state."""
        logger.debug(f"SPMD {self._driver.spmd_id} is draining to target step")

    def _on_enter_drained(self) -> None:
        """Called when entering DRAINED state."""
        logger.debug(f"SPMD {self._driver.spmd_id} has reached its drain target")

    def _on_enter_snapshotting(self) -> None:
        """Called when entering SNAPSHOTTING state."""
        logger.debug(f"SPMD {self._driver.spmd_id} is snapshotting operator states")

    def _on_enter_recovering(self) -> None:
        """Called when entering RECOVERING state."""
        logger.warning(f"SPMD {self._driver.spmd_id} is recovering")

        # Try to recover error workers
        with self._lock:
            error_workers = list(self._error_workers.keys())

        for rank in error_workers:
            try:
                success = self._driver._recover_worker(rank)
                if success:
                    self.trigger(DriverEvent.WORKER_RECOVERED, rank=rank)
            except Exception as e:
                logger.error(f"Failed to recover worker {rank}: {e}")

    def _on_enter_completed(self) -> None:
        """Called when entering COMPLETED state."""
        logger.debug(f"SPMD {self._driver.spmd_id} completed")
        # Report completion to controller
        self._driver._report_completion()

    def _on_enter_error(self) -> None:
        """Called when entering ERROR state."""
        logger.error(f"SPMD {self._driver.spmd_id} entered ERROR state")
        # Report error to controller
        self._driver._report_error(self._error_message or "Unknown error")

    def reset(self) -> None:
        """Reset state machine to initial state."""
        with self._lock:
            self._state = DriverState.INITIALIZING
            self._error_workers.clear()
            self._error_message = None
        logger.debug(f"SPMD {self._driver.spmd_id} state machine reset")
