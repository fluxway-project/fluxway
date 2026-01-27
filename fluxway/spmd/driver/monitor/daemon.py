"""
SPMD Monitor Daemon - Background monitoring process for SPMD.

Runs continuously to:
- Check Worker heartbeat timeouts
- Detect anomalies
- Aggregate and report metrics
- Handle health checks
"""

import asyncio
import time
import threading
from typing import Optional, Callable, Dict
import logging

from .monitor import SPMDMonitor

logger = logging.getLogger(__name__)


class SPMDMonitorDaemon:
    """
    SPMD Monitor Daemon.

    Background process that continuously monitors workers within an SPMD.
    Runs asynchronously and does not block the main execution flow.
    """

    def __init__(
        self,
        spmd_monitor: SPMDMonitor,
        check_interval: float = 10.0,
        report_interval: float = 30.0,
        anomaly_callback: Optional[Callable[[str, Dict[str, str]], None]] = None,
    ):
        """
        Initialize SPMD Monitor Daemon.

        Args:
            spmd_monitor: SPMDMonitor instance to monitor
            check_interval: Interval between health checks (seconds, default: 10s)
            report_interval: Interval between metric reports (seconds, default: 30s)
            anomaly_callback: Callback function for anomaly handling
                Signature: callback(spmd_id, anomalies_dict)
        """
        self.spmd_monitor = spmd_monitor
        self.spmd_id = spmd_monitor.spmd_id
        self.check_interval = check_interval
        self.report_interval = report_interval
        self.anomaly_callback = anomaly_callback
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_report_time = 0.0

    async def run(self) -> None:
        """
        Run the daemon loop.

        Continuously:
        1. Check Worker heartbeat timeouts
        2. Detect anomalies
        3. Aggregate and report metrics (periodically)
        """
        self._running = True
        logger.info(f"SPMD Monitor Daemon started for {self.spmd_id}")

        while self._running:
            try:
                # 1. Check heartbeat timeouts
                health_status = self.spmd_monitor.check_heartbeat_timeout()
                unhealthy_workers = [
                    worker_id for worker_id, is_healthy in health_status.items() if not is_healthy
                ]

                if unhealthy_workers:
                    logger.warning(
                        f"SPMD {self.spmd_id}: Unhealthy workers detected: {unhealthy_workers}"
                    )

                    # Send alerts for unhealthy workers
                    if self.anomaly_callback:
                        try:
                            self.anomaly_callback(
                                self.spmd_id,
                                {w: "Unhealthy (heartbeat timeout)" for w in unhealthy_workers},
                            )
                        except Exception as e:
                            logger.error(f"Error in anomaly callback: {e}")

                # 2. Detect anomalies
                anomalies = self.spmd_monitor.detect_anomalies()
                if anomalies:
                    logger.warning(f"SPMD {self.spmd_id}: Anomalies detected: {anomalies}")

                    # Call anomaly callback if provided
                    if self.anomaly_callback:
                        try:
                            self.anomaly_callback(self.spmd_id, anomalies)
                        except Exception as e:
                            logger.error(f"Error in anomaly callback: {e}")

                # 3. Aggregate and report metrics (periodically)
                now = time.time()
                if now - self._last_report_time >= self.report_interval:
                    aggregated = self.spmd_monitor.aggregate_telemetry()

                    # Report to main monitor
                    self.spmd_monitor.report_worker_telemetry(
                        self.spmd_monitor.get_all_worker_telemetry()
                    )

                    logger.debug(
                        f"SPMD {self.spmd_id}: Reported metrics "
                        f"(workers: {aggregated.get('worker_count', 0)}, "
                        f"CPU: {aggregated.get('cpu_percent_mean', 0):.2f}%)"
                    )

                    self._last_report_time = now

                # Sleep until next check
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in SPMD Monitor Daemon for {self.spmd_id}: {e}")
                await asyncio.sleep(self.check_interval)

    def start(self) -> None:
        """
        Start the daemon in background thread.

        Creates a background thread with its own event loop to run the daemon.
        """
        if self._thread is not None and self._thread.is_alive():
            logger.warning(f"SPMD Monitor Daemon for {self.spmd_id} is already running")
            return

        def run_in_thread():
            """Run the async daemon in a new event loop."""
            try:
                # Create new event loop for this thread
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)

                # Run the daemon
                self._loop.run_until_complete(self.run())
            except Exception as e:
                logger.error(f"Error in SPMD Monitor Daemon thread for {self.spmd_id}: {e}")
            finally:
                if self._loop:
                    self._loop.close()

        # Start daemon in background thread
        self._thread = threading.Thread(
            target=run_in_thread, name=f"SPMDMonitorDaemon-{self.spmd_id}", daemon=True
        )
        self._thread.start()
        logger.info(f"SPMD Monitor Daemon started for {self.spmd_id} (thread: {self._thread.name})")

    def stop(self) -> None:
        """
        Stop the daemon.

        Sets running flag to False and waits for thread to complete.
        """
        self._running = False

        # Cancel task if exists
        if self._task is not None:
            self._task.cancel()

        # Signal loop to stop
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for thread to finish (with timeout)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    f"SPMD Monitor Daemon thread for {self.spmd_id} did not stop within timeout"
                )

        logger.info(f"SPMD Monitor Daemon stopped for {self.spmd_id}")

    async def wait(self) -> None:
        """Wait for the daemon task to complete."""
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
