"""
HeartbeatManager for managing SPMD driver heartbeats.

Responsibilities:
- Maintain heartbeat state for all SPMD drivers
- Provide heartbeat reporting interface (called by drivers)
- Periodically detect heartbeat timeouts
- Notify Controller state machine on heartbeat anomalies

Uses asyncio for non-blocking monitoring loop.
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Dict, List, Optional, Set, Union

from ...common.types import DriverHeartbeat, DriverStatus, DriverState, HeartbeatConfig

logger = logging.getLogger(__name__)

# Type aliases for callbacks
SyncCallback = Callable[[], None]
AsyncCallback = Callable[[], Awaitable[None]]
SyncErrorCallback = Callable[[str, str], None]
AsyncErrorCallback = Callable[[str, str], Awaitable[None]]
SyncRecoverCallback = Callable[[str], None]
AsyncRecoverCallback = Callable[[str], Awaitable[None]]


class HeartbeatManager:
    """
    Manages heartbeats from all SPMD drivers.

    Tracks driver health status and triggers callbacks on state changes.
    Uses asyncio for the monitoring loop.
    """

    def __init__(
        self,
        config: Optional[HeartbeatConfig] = None,
        on_all_ready: Optional[Union[SyncCallback, AsyncCallback]] = None,
        on_all_workers_running: Optional[Union[SyncCallback, AsyncCallback]] = None,
        on_driver_error: Optional[Union[SyncErrorCallback, AsyncErrorCallback]] = None,
        on_driver_recovered: Optional[
            Union[SyncRecoverCallback, AsyncRecoverCallback]
        ] = None,
    ):
        """
        Initialize HeartbeatManager.

        Args:
            config: Heartbeat configuration
            on_all_ready: Callback when all registered drivers are ready (workers_ready=True)
            on_all_workers_running: Callback when all drivers report workers_running=True
                (after topology registration completes)
            on_driver_error: Callback when a driver has error (spmd_id, error_msg) (sync or async)
            on_driver_recovered: Callback when a driver recovers (spmd_id) (sync or async)
        """
        self._config = config or HeartbeatConfig()
        self._on_all_ready = on_all_ready
        self._on_all_workers_running = on_all_workers_running
        self._on_driver_error = on_driver_error
        self._on_driver_recovered = on_driver_recovered

        # Registered SPMD IDs that we expect heartbeats from
        self._registered_spmds: Set[str] = set()

        # Latest heartbeat from each driver
        self._heartbeats: Dict[str, DriverHeartbeat] = {}

        # Missed heartbeat counts
        self._missed_counts: Dict[str, int] = {}

        # Drivers currently in error state
        self._error_drivers: Set[str] = set()

        # Phase trigger flags (one-shot)
        self._all_ready_triggered = False
        self._all_workers_running_triggered = False

        # Lock for thread safety (asyncio.Lock for async context)
        self._lock = asyncio.Lock()

        # Monitoring task
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    def register_spmd(self, spmd_id: str) -> None:
        """
        Register an SPMD to expect heartbeats from.

        Note: This is a sync method for initialization phase.

        Args:
            spmd_id: SPMD identifier
        """
        self._registered_spmds.add(spmd_id)
        self._missed_counts[spmd_id] = 0
        logger.info(f"Registered SPMD {spmd_id} for heartbeat monitoring")

    def unregister_spmd(self, spmd_id: str) -> None:
        """
        Unregister an SPMD from heartbeat monitoring.

        Note: This is a sync method.

        Args:
            spmd_id: SPMD identifier
        """
        self._registered_spmds.discard(spmd_id)
        self._heartbeats.pop(spmd_id, None)
        self._missed_counts.pop(spmd_id, None)
        self._error_drivers.discard(spmd_id)
        logger.info(f"Unregistered SPMD {spmd_id} from heartbeat monitoring")

    async def receive_heartbeat(self, heartbeat: DriverHeartbeat) -> None:
        """
        Receive heartbeat from a driver.

        Args:
            heartbeat: Driver heartbeat data
        """
        spmd_id = heartbeat.spmd_id

        async with self._lock:
            if spmd_id not in self._registered_spmds:
                logger.warning(f"Received heartbeat from unregistered SPMD: {spmd_id}")
                return

            self._heartbeats[spmd_id] = heartbeat
            self._missed_counts[spmd_id] = 0

            # Check if driver recovered from error
            was_in_error = spmd_id in self._error_drivers
            if was_in_error and heartbeat.status == DriverStatus.HEALTHY:
                self._error_drivers.discard(spmd_id)
                logger.info(f"Driver {spmd_id} recovered from error")
                # Will call callback outside lock

            # Check if driver reported error
            is_new_error = False
            if heartbeat.status == DriverStatus.ERROR:
                if spmd_id not in self._error_drivers:
                    self._error_drivers.add(spmd_id)
                    is_new_error = True
                    logger.error(
                        f"Driver {spmd_id} reported error: {heartbeat.error_message}"
                    )

            # Check if all drivers are ready (first time) — Phase 1
            should_trigger_ready = False
            if not self._all_ready_triggered:
                if self._check_all_ready_internal():
                    self._all_ready_triggered = True
                    should_trigger_ready = True
                    logger.info("All drivers are ready (workers_ready=True)")

            # Check if all workers are running (first time) — Phase 2
            should_trigger_running = False
            if self._all_ready_triggered and not self._all_workers_running_triggered:
                if self._check_all_workers_running_internal():
                    self._all_workers_running_triggered = True
                    should_trigger_running = True
                    logger.info("All workers are running (workers_running=True)")

        # Call callbacks outside lock to prevent deadlock
        if was_in_error and heartbeat.status == DriverStatus.HEALTHY:
            await self._call_recovered_callback(spmd_id)

        if is_new_error:
            error_msg = heartbeat.error_message or "Unknown error"
            await self._call_error_callback(spmd_id, error_msg)

        if should_trigger_ready:
            await self._call_ready_callback()

        if should_trigger_running:
            await self._call_workers_running_callback()

        logger.debug(
            f"Received heartbeat from {spmd_id}: status={heartbeat.status.name}, "
            f"state={heartbeat.state.name}, timestamp={heartbeat.timestamp:.3f}, "
            f"healthy_workers={heartbeat.healthy_num_workers}/{heartbeat.num_workers}, "
            f"op_steps={heartbeat.op_steps}, workers_ready={heartbeat.workers_ready}"
        )

    async def _call_ready_callback(self) -> None:
        """Call the on_all_ready callback (handles sync and async)."""
        if self._on_all_ready is None:
            return
        try:
            result = self._on_all_ready()
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in on_all_ready callback: {e}")

    async def _call_error_callback(self, spmd_id: str, error: str) -> None:
        """Call the on_driver_error callback (handles sync and async)."""
        if self._on_driver_error is None:
            return
        try:
            result = self._on_driver_error(spmd_id, error)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in on_driver_error callback: {e}")

    async def _call_recovered_callback(self, spmd_id: str) -> None:
        """Call the on_driver_recovered callback (handles sync and async)."""
        if self._on_driver_recovered is None:
            return
        try:
            result = self._on_driver_recovered(spmd_id)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in on_driver_recovered callback: {e}")

    async def _call_workers_running_callback(self) -> None:
        """Call the on_all_workers_running callback (handles sync and async)."""
        if self._on_all_workers_running is None:
            return
        try:
            result = self._on_all_workers_running()
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Error in on_all_workers_running callback: {e}")

    def _check_all_ready_internal(self) -> bool:
        """Check if all registered drivers are ready (must hold lock).

        Phase 1 check: All drivers have workers_ready=True and are in
        READY or RUNNING state.
        """
        if not self._registered_spmds:
            return False

        for spmd_id in self._registered_spmds:
            heartbeat = self._heartbeats.get(spmd_id)
            if heartbeat is None:
                return False
            if heartbeat.status != DriverStatus.HEALTHY:
                return False
            if heartbeat.state not in (DriverState.READY, DriverState.RUNNING):
                return False
            # Check if all workers have reported heartbeat
            if not heartbeat.workers_ready:
                return False

        return True

    def _check_all_workers_running_internal(self) -> bool:
        """Check if all registered drivers report workers_running=True (must hold lock).

        Phase 2 check: After topology registration, all workers should be running.
        """
        if not self._registered_spmds:
            return False

        for spmd_id in self._registered_spmds:
            heartbeat = self._heartbeats.get(spmd_id)
            if heartbeat is None:
                return False
            if not heartbeat.workers_running:
                return False

        return True

    async def all_drivers_healthy(self) -> bool:
        """Check if all drivers are currently healthy."""
        async with self._lock:
            if not self._registered_spmds:
                return False

            for spmd_id in self._registered_spmds:
                if not self._is_driver_healthy_internal(spmd_id):
                    return False

            return True

    def _is_driver_healthy_internal(self, spmd_id: str) -> bool:
        """Check if a driver is healthy (must hold lock)."""
        heartbeat = self._heartbeats.get(spmd_id)
        if heartbeat is None:
            return False

        if heartbeat.status != DriverStatus.HEALTHY:
            return False

        # Check if heartbeat is recent
        if time.time() - heartbeat.timestamp > self._config.heartbeat_timeout:
            return False

        return True

    async def is_driver_healthy(self, spmd_id: str) -> bool:
        """Check if a specific driver is healthy."""
        async with self._lock:
            return self._is_driver_healthy_internal(spmd_id)

    async def get_driver_status(self, spmd_id: str) -> Optional[DriverStatus]:
        """Get the status of a specific driver."""
        async with self._lock:
            heartbeat = self._heartbeats.get(spmd_id)
            if heartbeat is None:
                return None
            return heartbeat.status

    async def get_all_driver_status(self) -> Dict[str, DriverStatus]:
        """Get status of all registered drivers."""
        async with self._lock:
            result = {}
            for spmd_id in self._registered_spmds:
                heartbeat = self._heartbeats.get(spmd_id)
                if heartbeat:
                    result[spmd_id] = heartbeat.status
                else:
                    result[spmd_id] = DriverStatus.TIMEOUT
            return result

    async def get_unhealthy_drivers(self) -> List[str]:
        """Get list of unhealthy driver SPMD IDs."""
        async with self._lock:
            unhealthy = []
            for spmd_id in self._registered_spmds:
                if not self._is_driver_healthy_internal(spmd_id):
                    unhealthy.append(spmd_id)
            return unhealthy

    async def get_driver_heartbeat(self, spmd_id: str) -> Optional[DriverHeartbeat]:
        """Get the latest heartbeat from a driver."""
        async with self._lock:
            return self._heartbeats.get(spmd_id)

    async def get_all_heartbeats(self) -> Dict[str, DriverHeartbeat]:
        """Get all latest heartbeats."""
        async with self._lock:
            return dict(self._heartbeats)

    def start_monitoring(self) -> None:
        """
        Start the heartbeat monitoring loop.

        Creates an asyncio task for monitoring.
        Should be called from an async context or event loop.
        """
        if self._monitoring:
            return

        self._monitoring = True

        # Get or create event loop
        try:
            loop = asyncio.get_running_loop()
            self._monitor_task = loop.create_task(self._monitor_loop())
        except RuntimeError:
            # No running loop, will be started when loop runs
            logger.warning(
                "No running event loop, monitoring will start when loop runs"
            )

        logger.info("Heartbeat monitoring started")

    async def start_monitoring_async(self) -> None:
        """Start the heartbeat monitoring loop (async version)."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Heartbeat monitoring started (async)")

    def stop_monitoring(self) -> None:
        """Stop the heartbeat monitoring loop."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        logger.info("Heartbeat monitoring stopped")

    async def stop_monitoring_async(self) -> None:
        """Stop the heartbeat monitoring loop (async version)."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Heartbeat monitoring stopped (async)")

    async def _monitor_loop(self) -> None:
        """Async monitoring loop that checks for heartbeat timeouts."""
        logger.info("Heartbeat monitor loop started")

        while self._monitoring:
            try:
                await self._check_heartbeat_timeouts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitoring loop: {e}")

            await asyncio.sleep(self._config.heartbeat_interval)

        logger.info("Heartbeat monitor loop ended")

    async def _check_heartbeat_timeouts(self) -> None:
        """Check for heartbeat timeouts and trigger callbacks."""
        current_time = time.time()
        timeout_drivers: List[tuple] = []

        async with self._lock:
            for spmd_id in self._registered_spmds:
                heartbeat = self._heartbeats.get(spmd_id)

                if heartbeat is None:
                    # Never received a heartbeat
                    self._missed_counts[spmd_id] = (
                        self._missed_counts.get(spmd_id, 0) + 1
                    )
                    missed = self._missed_counts[spmd_id]
                    logger.debug(
                        f"Heartbeat check: {spmd_id} never received heartbeat, "
                        f"missed_count={missed}/{self._config.max_missed_heartbeats}"
                    )
                elif (
                    current_time - heartbeat.timestamp > self._config.heartbeat_timeout
                ):
                    # Heartbeat timeout
                    self._missed_counts[spmd_id] = (
                        self._missed_counts.get(spmd_id, 0) + 1
                    )
                    missed = self._missed_counts[spmd_id]
                    age = current_time - heartbeat.timestamp
                    logger.debug(
                        f"Heartbeat check: {spmd_id} timeout (age={age:.2f}s > "
                        f"timeout={self._config.heartbeat_timeout}s), "
                        f"missed_count={missed}/{self._config.max_missed_heartbeats}"
                    )
                else:
                    # Heartbeat is healthy
                    age = current_time - heartbeat.timestamp
                    logger.debug(
                        f"Heartbeat check: {spmd_id} healthy (age={age:.2f}s < "
                        f"timeout={self._config.heartbeat_timeout}s)"
                    )
                    continue

                missed = self._missed_counts[spmd_id]

                if missed >= self._config.max_missed_heartbeats:
                    if spmd_id not in self._error_drivers:
                        self._error_drivers.add(spmd_id)
                        error_msg = f"Heartbeat timeout (missed {missed})"
                        timeout_drivers.append((spmd_id, error_msg))
                        logger.error(
                            f"Driver {spmd_id} heartbeat timeout (missed {missed})"
                        )

        # Call callbacks outside lock
        for spmd_id, error_msg in timeout_drivers:
            await self._call_error_callback(spmd_id, error_msg)

    async def reset(self) -> None:
        """Reset all heartbeat state."""
        async with self._lock:
            self._heartbeats.clear()
            self._missed_counts.clear()
            self._error_drivers.clear()
            self._all_ready_triggered = False
            self._all_workers_running_triggered = False
            for spmd_id in self._registered_spmds:
                self._missed_counts[spmd_id] = 0
