"""
Prometheus backend for metrics export.

Exports metrics in Prometheus format for scraping.
"""

from typing import Dict, Optional
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.warning("Prometheus client not available. Install with: pip install prometheus-client")


class PrometheusBackend:
    """
    Prometheus backend for metrics export.

    Exports metrics in Prometheus format for scraping.
    """

    def __init__(self, port: int = 8000, enable_server: bool = True):
        """
        Initialize Prometheus backend.

        Args:
            port: Port for Prometheus HTTP server (default: 8000)
            enable_server: Whether to start HTTP server for scraping
        """
        self.port = port
        self._metrics: Dict[str, Gauge] = {}
        self._counters: Dict[str, Counter] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._server_started = False

        if HAS_PROMETHEUS:
            if enable_server:
                try:
                    start_http_server(port)
                    self._server_started = True
                    logger.info(f"Prometheus HTTP server started on port {port}")
                except Exception as e:
                    logger.error(f"Failed to start Prometheus HTTP server: {e}")
        else:
            logger.warning("Prometheus client not available, metrics will not be exported")

    def _get_or_create_gauge(self, name: str, description: str, labels: Optional[list] = None):
        """Get or create a Prometheus Gauge metric."""
        if not HAS_PROMETHEUS:
            return None

        key = f"{name}:{labels}" if labels else name
        if key not in self._metrics:
            try:
                self._metrics[key] = Gauge(name, description, labels or [])
            except Exception as e:
                logger.error(f"Failed to create Prometheus gauge {name}: {e}")
                return None

        return self._metrics[key]

    def record_metrics(
        self, metrics: Dict[str, float], labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record metrics to Prometheus.

        Args:
            metrics: Dictionary of metric name to value
            labels: Optional labels for metrics (e.g., {"spmd_id": "spmd_1"})
        """
        if not HAS_PROMETHEUS:
            return

        label_list = list(labels.keys()) if labels else []
        label_values = list(labels.values()) if labels else []

        for metric_name, value in metrics.items():
            # Sanitize metric name (Prometheus format)
            sanitized_name = metric_name.replace(".", "_").replace("-", "_")

            gauge = self._get_or_create_gauge(
                f"fluxway_{sanitized_name}", f"Fluxway metric: {metric_name}", label_list
            )

            if gauge is not None:
                try:
                    if label_values:
                        gauge.labels(*label_values).set(value)
                    else:
                        gauge.set(value)
                except Exception as e:
                    logger.error(f"Failed to record metric {metric_name}: {e}")
    
    def write_metrics(
        self, step: int, metrics: Dict[str, float], prefix: str = "", **kwargs
    ) -> None:
        """
        Write metrics to Prometheus (compatible with AsyncBackendWriter).

        Args:
            step: Step number (ignored for Prometheus)
            metrics: Dictionary of metric name to value
            prefix: Prefix for metric names (ignored for Prometheus)
            **kwargs: Additional arguments (e.g., labels)
        """
        labels = kwargs.get("labels")
        self.record_metrics(metrics, labels)

    def record_worker_metrics(
        self, spmd_id: str, worker_id: str, metrics: Dict[str, float]
    ) -> None:
        """
        Record worker-level metrics to Prometheus.

        Args:
            spmd_id: SPMD identifier
            worker_id: Worker identifier
            metrics: Worker metrics dictionary
        """
        labels = {"spmd_id": spmd_id, "worker_id": worker_id, "level": "worker"}
        self.record_metrics(metrics, labels)

    def record_spmd_metrics(self, spmd_id: str, metrics: Dict[str, float]) -> None:
        """
        Record SPMD-level metrics to Prometheus.

        Args:
            spmd_id: SPMD identifier
            metrics: SPMD aggregated metrics dictionary
        """
        labels = {"spmd_id": spmd_id, "level": "spmd"}
        self.record_metrics(metrics, labels)

    def record_task_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Record task-level metrics to Prometheus.

        Args:
            metrics: Task aggregated metrics dictionary
        """
        labels = {"level": "task"}
        self.record_metrics(metrics, labels)

    def get_metrics_endpoint(self) -> str:
        """
        Get Prometheus metrics endpoint URL.

        Returns:
            URL for Prometheus scraping
        """
        return f"http://localhost:{self.port}/metrics"
