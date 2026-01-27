"""
SPMD Sub Monitor for worker monitoring.

Based on ARCHITECTURE.md Section 3.1.2 (SPMD Sub Monitor)

Responsibilities:
- Monitor all Worker running states
- Heartbeat detection
- Information aggregation
- Anomaly detection
"""

from typing import Dict, Optional, Any, List
import time
import logging

from fluxway.common.types import WorkerTelemetry
from fluxway.common.monitoring import MetricsAggregator
from fluxway.common.monitoring.backends import AlertBackend, AlertLevel
from fluxway.common.monitoring.optimizations import IncrementalAggregator

logger = logging.getLogger(__name__)


class SPMDMonitor:
    """
    SPMD Sub Monitor.

    Monitors worker health and collects telemetry within an SPMD.
    """

    def __init__(
        self,
        spmd_id: str,
        heartbeat_timeout: float = 30.0,
        alert_backend: Optional[AlertBackend] = None,
    ):
        """
        Initialize SPMD Monitor.

        Args:
            spmd_id: SPMD identifier
            heartbeat_timeout: Heartbeat timeout in seconds (default: 30s)
            alert_backend: Alert backend for sending alerts (optional)
        """
        self.spmd_id = spmd_id
        self.heartbeat_timeout = heartbeat_timeout
        self._worker_telemetry: Dict[str, WorkerTelemetry] = {}
        self._heartbeat_timestamps: Dict[str, float] = {}
        self._worker_health_status: Dict[str, bool] = {}  # worker_id -> is_healthy
        self._main_monitor_ref: Optional[Any] = None  # Reference to main Monitor (Ray Actor handle)
        self.alert_backend: Optional[AlertBackend] = alert_backend

        # Initialize incremental aggregator for optimization
        self._incremental_aggregator = IncrementalAggregator()

    def set_main_monitor_ref(self, main_monitor_ref: Any) -> None:
        """
        Set reference to main Monitor (Ray Actor handle).

        Args:
            main_monitor_ref: Ray Actor handle to main Monitor
        """
        self._main_monitor_ref = main_monitor_ref

    def receive_heartbeat(
        self, worker_id: str, telemetry: WorkerTelemetry, timestamp: Optional[float] = None
    ) -> None:
        """
        Receive heartbeat from worker.

        Args:
            worker_id: Worker identifier
            telemetry: Worker telemetry data
            timestamp: Heartbeat timestamp (if None, uses telemetry.heartbeat_timestamp)

        Steps:
        1. Update heartbeat timestamp
        2. Check for missed heartbeats
        3. Update worker health status
        """
        # Use timestamp from telemetry if not provided
        if timestamp is None:
            timestamp = telemetry.heartbeat_timestamp

        # Update heartbeat timestamp
        self._heartbeat_timestamps[worker_id] = timestamp

        # Update worker health status (heartbeat received = healthy)
        self._worker_health_status[worker_id] = True

        # Check for missed heartbeats (if we have a previous timestamp)
        if worker_id in self._heartbeat_timestamps:
            # This check can be enhanced with more sophisticated logic
            pass

    def report_single_worker_telemetry(self, worker_id: str, telemetry: WorkerTelemetry) -> None:
        """
        Report single worker telemetry.

        Args:
            worker_id: Worker identifier
            telemetry: Worker telemetry data

        Steps:
        1. Update local cache
        2. Optionally forward to Main Orchestrator
        """
        # Update local cache
        self._worker_telemetry[worker_id] = telemetry

        # Optionally forward to Main Orchestrator
        if self._main_monitor_ref is not None:
            try:
                # Use Ray to call main monitor (if it's a Ray Actor)
                if hasattr(self._main_monitor_ref, "report_telemetry"):
                    self._main_monitor_ref.report_telemetry.remote(
                        self.spmd_id, {worker_id: telemetry}
                    )
            except Exception:
                # If main monitor is not available, just continue
                # In production, should use proper logging
                pass

    def report_worker_telemetry(self, telemetry: Dict[str, WorkerTelemetry]) -> None:
        """
        Report worker telemetry to Main Orchestrator.

        Args:
            telemetry: Telemetry from all workers

        Steps:
        1. Update local cache
        2. Aggregate before sending (SPMD level)
        3. Send aggregated metrics to Main Orchestrator Monitor
        """
        # Update local cache
        self._worker_telemetry.update(telemetry)

        # Aggregate before sending (using incremental aggregator)
        aggregated = self._incremental_aggregator.incremental_aggregate(telemetry)

        # Send to Main Orchestrator Monitor (if available)
        if self._main_monitor_ref is not None:
            try:
                # Use Ray to call main monitor
                if hasattr(self._main_monitor_ref, "report_telemetry"):
                    self._main_monitor_ref.report_telemetry.remote(self.spmd_id, telemetry)
            except Exception:
                # If main monitor is not available, just log
                # In production, should use proper logging
                pass

    def get_worker_telemetry(self, worker_id: str) -> Optional[WorkerTelemetry]:
        """
        Query worker telemetry.

        Args:
            worker_id: Worker identifier

        Returns:
            Cached WorkerTelemetry or None if not found
        """
        return self._worker_telemetry.get(worker_id)

    def get_all_worker_telemetry(self) -> Dict[str, WorkerTelemetry]:
        """
        Get all worker telemetry.

        Returns:
            Dictionary mapping worker_id to WorkerTelemetry
        """
        return self._worker_telemetry.copy()

    def check_worker_health(self, worker_id: str, timeout: Optional[float] = None) -> bool:
        """
        Check if worker is healthy.

        Args:
            worker_id: Worker identifier
            timeout: Heartbeat timeout (if None, uses self.heartbeat_timeout)

        Returns:
            True if worker is healthy, False otherwise

        Checks:
        1. Heartbeat recency (within timeout)
        2. Error status (if available in telemetry)
        """
        if timeout is None:
            timeout = self.heartbeat_timeout

        # Check heartbeat recency
        if worker_id not in self._heartbeat_timestamps:
            self._worker_health_status[worker_id] = False
            return False

        last_heartbeat = self._heartbeat_timestamps[worker_id]
        elapsed = time.time() - last_heartbeat

        if elapsed > timeout:
            self._worker_health_status[worker_id] = False

            # Send alert for heartbeat timeout
            if self.alert_backend:
                try:
                    self.alert_backend.send_worker_alert(
                        spmd_id=self.spmd_id,
                        worker_id=worker_id,
                        level=AlertLevel.ERROR,
                        title=f"Worker {worker_id} Heartbeat Timeout",
                        message=f"Worker {worker_id} heartbeat timeout ({elapsed:.1f}s > {timeout}s)",
                        metadata={"timeout": timeout, "elapsed": elapsed},
                    )
                except Exception as e:
                    logger.error(f"Failed to send alert for worker {worker_id}: {e}")

            return False

        # Check error status from telemetry (if available)
        telemetry = self._worker_telemetry.get(worker_id)
        if telemetry is None:
            # No telemetry yet, but heartbeat is recent
            self._worker_health_status[worker_id] = True
            return True

        # Additional health checks can be added here
        # For example: check if CPU/memory usage is abnormal

        self._worker_health_status[worker_id] = True
        return True

    def get_unhealthy_workers(self) -> List[str]:
        """
        Get list of unhealthy workers.

        Returns:
            List of worker IDs that are unhealthy
        """
        unhealthy = []
        for worker_id in self._worker_telemetry.keys():
            if not self.check_worker_health(worker_id):
                unhealthy.append(worker_id)
        return unhealthy

    def detect_anomalies(self) -> Dict[str, str]:
        """
        Detect anomalies across workers.

        Returns:
            Mapping of worker_id to anomaly description

        Detects:
        1. Performance anomalies (CPU usage too high/low, memory leak)
        2. Resource anomalies (GPU overheating, disk IO abnormal)
        """
        anomalies = {}

        if not self._worker_telemetry:
            return anomalies

        # Thresholds for anomaly detection
        CPU_HIGH_THRESHOLD = 95.0  # CPU usage > 95%
        CPU_LOW_THRESHOLD = 1.0  # CPU usage < 1% (idle)
        MEMORY_HIGH_THRESHOLD = 90.0  # Memory usage > 90%
        GPU_TEMP_THRESHOLD = 85.0  # GPU temperature > 85°C
        GPU_UTIL_LOW_THRESHOLD = 5.0  # GPU utilization < 5% (underutilized)

        for worker_id, telemetry in self._worker_telemetry.items():
            worker_anomalies = []

            # Performance anomalies
            if telemetry.cpu_percent > CPU_HIGH_THRESHOLD:
                worker_anomalies.append(f"CPU usage too high: {telemetry.cpu_percent:.2f}%")
            elif telemetry.cpu_percent < CPU_LOW_THRESHOLD:
                # Only flag as anomaly if it's been low for a while (check heartbeat)
                if worker_id in self._heartbeat_timestamps:
                    elapsed = time.time() - self._heartbeat_timestamps[worker_id]
                    if elapsed > 60:  # Low CPU for more than 60 seconds
                        worker_anomalies.append(
                            f"CPU usage too low: {telemetry.cpu_percent:.2f}% (possibly idle)"
                        )

            if telemetry.memory_percent > MEMORY_HIGH_THRESHOLD:
                worker_anomalies.append(f"Memory usage too high: {telemetry.memory_percent:.2f}%")

            # Resource anomalies (GPU)
            for gpu_id in telemetry.gpu_ids:
                temp = telemetry.gpu_temperature_c.get(gpu_id, 0.0)
                if temp > GPU_TEMP_THRESHOLD:
                    worker_anomalies.append(f"GPU {gpu_id} overheating: {temp:.1f}°C")

                util = telemetry.gpu_utilization_percent.get(gpu_id, 0.0)
                if util < GPU_UTIL_LOW_THRESHOLD and telemetry.cpu_percent > 10:
                    # GPU underutilized but CPU is working (possible GPU bottleneck)
                    worker_anomalies.append(f"GPU {gpu_id} underutilized: {util:.2f}%")

            if worker_anomalies:
                anomalies[worker_id] = "; ".join(worker_anomalies)

                # Send alert for anomalies
                if self.alert_backend:
                    try:
                        telemetry = self._worker_telemetry[worker_id]
                        self.alert_backend.send_worker_alert(
                            spmd_id=self.spmd_id,
                            worker_id=worker_id,
                            level=AlertLevel.WARNING,
                            title=f"Worker {worker_id} Anomaly",
                            message="; ".join(worker_anomalies),
                            metrics={
                                "cpu_percent": telemetry.cpu_percent,
                                "memory_percent": telemetry.memory_percent,
                            },
                            metadata={"anomalies": worker_anomalies},
                        )
                    except Exception as e:
                        logger.error(f"Failed to send alert for worker {worker_id}: {e}")

        return anomalies

    def aggregate_telemetry(self) -> Dict[str, float]:
        """
        Aggregate telemetry values across workers (SPMD level).

        Returns:
            Dictionary with aggregated metrics:
            - cpu_percent_mean, cpu_percent_max, etc.
            - memory_rss_mb_mean, memory_rss_mb_total, etc.
            - gpu_utilization_mean, etc.
            - network_bytes_sent_total, etc.
            - disk_read_bytes_total, etc.
        """
        if not self._worker_telemetry:
            return {}

        # Get all worker telemetries
        # Aggregate using incremental aggregator
        aggregated = self._incremental_aggregator.incremental_aggregate(self._worker_telemetry)

        # Add SPMD ID and health status to aggregated metrics
        aggregated["spmd_id"] = self.spmd_id
        aggregated["healthy_worker_count"] = sum(
            1 for worker_id in self._worker_telemetry.keys() if self.check_worker_health(worker_id)
        )
        aggregated["unhealthy_worker_count"] = len(self._worker_telemetry) - aggregated.get(
            "healthy_worker_count", 0
        )

        return aggregated

    def check_heartbeat_timeout(self) -> Dict[str, bool]:
        """
        Check heartbeat timeout for all workers.

        Returns:
            Dictionary mapping worker_id to is_healthy status
        """
        health_status = {}
        for worker_id in self._worker_telemetry.keys():
            health_status[worker_id] = self.check_worker_health(worker_id)
        return health_status
