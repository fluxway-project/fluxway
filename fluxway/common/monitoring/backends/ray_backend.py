"""
Ray monitoring backend integration.

Integrates with Ray's built-in monitoring system.
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    logger.warning("Ray not available")


class RayBackend:
    """
    Ray monitoring backend.

    Integrates with Ray's built-in monitoring system.
    """

    def __init__(self):
        """Initialize Ray monitoring backend."""
        self._initialized = False

        if HAS_RAY:
            try:
                # Ray monitoring is automatically available when Ray is initialized
                self._initialized = True
                logger.info("Ray monitoring backend initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Ray monitoring: {e}")
        else:
            logger.warning("Ray not available, monitoring will not be integrated")

    def record_metrics(
        self, metrics: Dict[str, float], tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record metrics to Ray monitoring.

        Args:
            metrics: Dictionary of metric name to value
            tags: Optional tags for metrics (e.g., {"spmd_id": "spmd_1"})

        Note:
            Ray's metrics are automatically collected and can be viewed
            in Ray Dashboard. This method provides a programmatic interface.
        """
        if not self._initialized:
            return

        try:
            # Ray metrics are typically recorded via ray.util.metrics
            # For now, we log them (Ray Dashboard will collect them automatically)
            if tags:
                tag_str = ", ".join(f"{k}={v}" for k, v in tags.items())
                logger.debug(f"Ray metrics [{tag_str}]: {metrics}")
            else:
                logger.debug(f"Ray metrics: {metrics}")

            # TODO: Use ray.util.metrics when available
            # Example:
            # from ray.util.metrics import Counter, Gauge
            # for name, value in metrics.items():
            #     gauge = Gauge(name, description=f"{name} metric")
            #     gauge.set(value)
        except Exception as e:
            logger.error(f"Failed to record metrics to Ray: {e}")

    def record_worker_metrics(
        self, spmd_id: str, worker_id: str, metrics: Dict[str, float]
    ) -> None:
        """
        Record worker-level metrics to Ray monitoring.

        Args:
            spmd_id: SPMD identifier
            worker_id: Worker identifier
            metrics: Worker metrics dictionary
        """
        tags = {"spmd_id": spmd_id, "worker_id": worker_id, "level": "worker"}
        self.record_metrics(metrics, tags)

    def record_spmd_metrics(self, spmd_id: str, metrics: Dict[str, float]) -> None:
        """
        Record SPMD-level metrics to Ray monitoring.

        Args:
            spmd_id: SPMD identifier
            metrics: SPMD aggregated metrics dictionary
        """
        tags = {"spmd_id": spmd_id, "level": "spmd"}
        self.record_metrics(metrics, tags)

    def record_task_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Record task-level metrics to Ray monitoring.

        Args:
            metrics: Task aggregated metrics dictionary
        """
        tags = {"level": "task"}
        self.record_metrics(metrics, tags)
    
    def record_training_metrics(
        self, step: int, spmd_id: Optional[str], metrics: Dict[str, float]
    ) -> None:
        """
        Record training metrics to Ray monitoring.

        Args:
            step: Step number
            spmd_id: SPMD identifier
            metrics: Training metrics dictionary
        """
        tags = {"spmd_id": spmd_id, "level": "training"} if spmd_id else {"level": "training"}
        self.record_metrics(metrics, tags)
    
    def write_metrics(
        self, step: int, metrics: Dict[str, float], prefix: str = "", **kwargs
    ) -> None:
        """
        Write metrics to Ray monitoring (compatible with AsyncBackendWriter).

        Args:
            step: Step number
            metrics: Dictionary of metric name to value
            prefix: Prefix for metric names (ignored for Ray)
            **kwargs: Additional arguments (e.g., spmd_id)
        """
        spmd_id = kwargs.get("spmd_id")
        if spmd_id:
            self.record_training_metrics(step, spmd_id, metrics)
        else:
            self.record_task_metrics(metrics)
