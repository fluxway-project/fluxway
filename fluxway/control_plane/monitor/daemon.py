"""
Monitor Daemon - Background monitoring process for Controller.

Runs continuously to:
- Probe SPMD sub-controllers for health
- Check cluster health status
- Aggregate task-level metrics
- Handle failures and alerts
"""

import asyncio
import time
from typing import Optional, Callable, Dict
import logging

from .monitor import Monitor

logger = logging.getLogger(__name__)


class MonitorDaemon:
    """
    Monitor Daemon for Controller.

    Background process that continuously monitors all SPMDs in the cluster.
    Runs asynchronously and does not block the main execution flow.
    """

    def __init__(
        self,
        monitor: Monitor,
        probe_interval: float = 30.0,
        health_check_interval: float = 10.0,
        failure_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize Monitor Daemon.

        Args:
            monitor: Monitor instance to use
            probe_interval: Interval between SPMD probes (seconds, default: 30s)
            health_check_interval: Interval between health checks (seconds, default: 10s)
            failure_callback: Callback function for failure handling
                Signature: callback(spmd_id, reason)
        """
        self.monitor = monitor
        self.probe_interval = probe_interval
        self.health_check_interval = health_check_interval
        self.failure_callback = failure_callback
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_probe_time: Dict[str, float] = {}

    async def run(self) -> None:
        """
        Run the daemon loop.

        Continuously:
        1. Probe SPMD sub-controllers for health
        2. Check cluster health status
        3. Handle failures and alerts
        """
        self._running = True
        logger.info("Monitor Daemon started")

        while self._running:
            try:
                # 1. Probe SPMD sub-controllers (periodically)
                now = time.time()
                spmd_ids = self.monitor.get_all_spmd_ids()

                for spmd_id in spmd_ids:
                    last_probe = self._last_probe_time.get(spmd_id, 0)
                    if now - last_probe >= self.probe_interval:
                        is_healthy = self.monitor.probe_sub_controller(spmd_id)
                        self._last_probe_time[spmd_id] = now

                        if not is_healthy:
                            logger.error(f"SPMD {spmd_id} probe failed")

                            # Send alert via Monitor's alert backend
                            if hasattr(self.monitor, "alert") and self.monitor.alert:
                                try:
                                    from fluxway.common.monitoring.backends import AlertLevel

                                    self.monitor.alert.send_spmd_alert(
                                        spmd_id=spmd_id,
                                        level=AlertLevel.ERROR,
                                        title=f"SPMD {spmd_id} Probe Failed",
                                        message=f"SPMD {spmd_id} health probe failed",
                                        metadata={"reason": "probe_failed"},
                                    )
                                except Exception as e:
                                    logger.error(f"Failed to send alert: {e}")

                            # Call failure callback if provided
                            if self.failure_callback:
                                try:
                                    self.failure_callback(spmd_id, "probe_failed")
                                except Exception as e:
                                    logger.error(f"Error in failure callback: {e}")

                # 2. Check cluster health status
                health_status = self.monitor.get_cluster_health()

                # Log health summary
                healthy_count = sum(1 for status in health_status.values() if status == "running")
                total_count = len(health_status)

                if healthy_count < total_count:
                    logger.warning(
                        f"Cluster health: {healthy_count}/{total_count} SPMDs healthy. "
                        f"Status: {health_status}"
                    )

                # 3. Check for failed components
                failed_components = self.monitor.get_failed_components()
                if failed_components:
                    logger.error(f"Failed components detected: {failed_components}")

                    # Call failure callback for each failed component
                    if self.failure_callback:
                        for component in failed_components:
                            try:
                                self.failure_callback(
                                    component["component_id"], component["status"]
                                )
                            except Exception as e:
                                logger.error(f"Error in failure callback: {e}")

                # Sleep until next check
                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Error in Monitor Daemon: {e}")
                await asyncio.sleep(self.health_check_interval)

    def start(self) -> None:
        """
        Start the daemon in background.

        Creates an async task that runs the daemon loop.
        """
        if self._task is not None and not self._task.done():
            logger.warning("Monitor Daemon is already running")
            return

        self._task = asyncio.create_task(self.run())
        logger.info("Monitor Daemon started")

    def stop(self) -> None:
        """
        Stop the daemon.

        Sets running flag to False and waits for task to complete.
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()
            logger.info("Monitor Daemon stopped")

    async def wait(self) -> None:
        """Wait for the daemon task to complete."""
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
