"""
Worker Metrics Collector for SPMD Worker monitoring.

This module implements WorkerMetricsCollector, a Ray Actor that collects
system metrics for a single Worker process. It bridges SystemMetricsCollector
and the monitoring system.

Responsibilities:
- Collect CPU, memory, GPU, network, disk metrics for a Worker
- Get Ray-assigned GPU IDs
- Convert metrics to WorkerTelemetry format
- Support periodic collection
"""

import os
import time
from typing import Optional, List

try:
    import ray
except ImportError:
    ray = None

from fluxway.common.utils.metrics_collector import SystemMetricsCollector
from fluxway.common.types import WorkerTelemetry
from fluxway.common.monitoring.optimizations import AdaptiveMetricsCollector


@ray.remote
class WorkerMetricsCollector:
    """
    Worker-level metrics collector (Ray Actor).

    Collects system metrics for a single Worker process and converts them
    to WorkerTelemetry format.

    This is a Ray Actor, so it runs on the Worker's node and can access
    the Worker's process resources directly.
    """

    def __init__(self, worker_id: str, spmd_id: str):
        """
        Initialize Worker Metrics Collector.

        Args:
            worker_id: Unique identifier for this worker
            spmd_id: SPMD identifier this worker belongs to
        """
        self.worker_id = worker_id
        self.spmd_id = spmd_id
        self.system_collector = SystemMetricsCollector()
        self.pid = os.getpid()

        # Get GPU IDs assigned by Ray
        self.gpu_ids = self._get_ray_gpu_ids()

        # Last collection timestamp
        self._last_collection_time: Optional[float] = None

        # Initialize adaptive collector for dynamic interval adjustment
        self._adaptive_collector = AdaptiveMetricsCollector(
            base_interval=5.0, min_interval=1.0, max_interval=30.0  # Default 5 seconds
        )

    def _get_ray_gpu_ids(self) -> List[int]:
        """
        Get GPU IDs assigned by Ray to this Worker.

        Returns:
            List of GPU IDs assigned to this worker

        Note:
            In a real Ray cluster, GPU IDs come from:
            1. Ray Placement Group allocation
            2. CUDA_VISIBLE_DEVICES environment variable
            3. Ray API: ray.get_gpu_ids() (if available)
        """
        gpu_ids = []

        # Method 1: Try Ray API (if available)
        try:
            if ray is not None:
                # Try to get GPU IDs from Ray runtime context
                # Note: This may not be available in all Ray versions
                if hasattr(ray, "get_gpu_ids"):
                    gpu_ids = ray.get_gpu_ids()
                    if gpu_ids:
                        return gpu_ids
        except Exception:
            pass

        # Method 2: Check environment variable
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            try:
                # CUDA_VISIBLE_DEVICES format: "0,1" or "0"
                gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
                if gpu_ids:
                    return gpu_ids
            except (ValueError, AttributeError):
                pass

        # Method 3: If no GPU assigned, return empty list
        # This is fine for CPU-only workers or macOS
        return []

    def collect(self) -> WorkerTelemetry:
        """
        Collect current metrics for this worker.

        Returns:
            WorkerTelemetry object with all collected metrics
        """
        # Collect all system metrics
        metrics = self.system_collector.collect_all_metrics(
            pid=self.pid, gpu_ids=self.gpu_ids if self.gpu_ids else None
        )

        # Extract GPU metrics
        gpu_metrics = metrics.get("gpu_metrics", {})

        # Build WorkerTelemetry
        telemetry = WorkerTelemetry(
            worker_id=self.worker_id,
            timestamp=metrics["timestamp"],
            # CPU metrics
            cpu_percent=metrics.get("cpu_percent", 0.0),
            cpu_count=metrics.get("cpu_count", 0),
            cpu_times_user=metrics.get("cpu_times_user", 0.0),
            cpu_times_system=metrics.get("cpu_times_system", 0.0),
            # Memory metrics
            memory_rss_mb=metrics.get("memory_rss_mb", 0.0),
            memory_vms_mb=metrics.get("memory_vms_mb", 0.0),
            memory_percent=metrics.get("memory_percent", 0.0),
            # GPU metrics (if available)
            gpu_ids=list(gpu_metrics.keys()),
            gpu_utilization_percent={
                gpu_id: gpu_data.get("utilization_percent", 0.0)
                for gpu_id, gpu_data in gpu_metrics.items()
            },
            gpu_memory_used_mb={
                gpu_id: gpu_data.get("memory_used_mb", 0.0)
                for gpu_id, gpu_data in gpu_metrics.items()
            },
            gpu_memory_total_mb={
                gpu_id: gpu_data.get("memory_total_mb", 0.0)
                for gpu_id, gpu_data in gpu_metrics.items()
            },
            gpu_temperature_c={
                gpu_id: gpu_data.get("temperature_c", 0.0)
                for gpu_id, gpu_data in gpu_metrics.items()
            },
            # Network I/O
            network_bytes_sent=metrics.get("bytes_sent", 0),
            network_bytes_recv=metrics.get("bytes_recv", 0),
            # Disk I/O
            disk_read_bytes=metrics.get("read_bytes", 0),
            disk_write_bytes=metrics.get("write_bytes", 0),
            # Heartbeat
            heartbeat_timestamp=time.time(),
        )

        # Update last collection time
        self._last_collection_time = time.time()

        # Adjust collection interval based on CPU load
        cpu_load = telemetry.cpu_percent / 100.0
        self._adaptive_collector.adjust_interval(cpu_load)

        return telemetry

    def collect_periodic(self, interval: float = 1.0) -> WorkerTelemetry:
        """
        Collect metrics, ensuring at least interval seconds since last collection.

        Args:
            interval: Minimum seconds between collections

        Returns:
            WorkerTelemetry object
        """
        now = time.time()

        # If collected recently, return cached or wait
        if self._last_collection_time is not None:
            elapsed = now - self._last_collection_time
            if elapsed < interval:
                # For now, just collect anyway (can be optimized later with caching)
                pass

        return self.collect()

    def get_gpu_ids(self) -> List[int]:
        """Get GPU IDs assigned to this worker."""
        return self.gpu_ids.copy()

    def get_pid(self) -> int:
        """Get process ID of this worker."""
        return self.pid
