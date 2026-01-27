"""
Training metrics collector for non-blocking, batched metric collection.

Collects training performance, quality, and stability metrics asynchronously.
"""

import time
import logging
from typing import Dict, Optional, Any, Deque
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """
    Training metrics data structure.

    Contains all training-related metrics for a single step.
    """

    step: int
    timestamp: float = field(default_factory=time.time)

    # Basic metrics
    epoch: Optional[int] = None
    loss: Optional[float] = None

    # Performance metrics
    forward_time_ms: Optional[float] = None
    backward_time_ms: Optional[float] = None
    update_time_ms: Optional[float] = None
    total_step_time_ms: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None
    throughput_steps_per_sec: Optional[float] = None

    # Quality metrics
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    gradient_mean: Optional[float] = None
    gradient_std: Optional[float] = None
    gradient_max: Optional[float] = None
    gradient_min: Optional[float] = None
    param_update_magnitude: Optional[float] = None
    param_norm: Optional[float] = None

    # Communication metrics
    allreduce_time_ms: Optional[float] = None
    sync_wait_time_ms: Optional[float] = None
    staleness: Optional[int] = None
    ulp_transfer_time_ms: Optional[float] = None
    ulp_assembly_time_ms: Optional[float] = None

    # Stability metrics
    nan_inf_detected: bool = False
    gradient_explosion: bool = False
    gradient_vanishing: bool = False
    loss_variance: Optional[float] = None

    # Load balancing metrics
    load_balance_score: Optional[float] = None
    slow_worker_ratio: Optional[float] = None

    # Data metrics
    batch_size: Optional[int] = None
    data_loading_time_ms: Optional[float] = None

    # Additional metrics
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for reporting."""
        metrics = {}

        if self.epoch is not None:
            metrics["epoch"] = float(self.epoch)
        if self.loss is not None:
            metrics["loss"] = self.loss

        # Performance
        if self.forward_time_ms is not None:
            metrics["forward_time_ms"] = self.forward_time_ms
        if self.backward_time_ms is not None:
            metrics["backward_time_ms"] = self.backward_time_ms
        if self.update_time_ms is not None:
            metrics["update_time_ms"] = self.update_time_ms
        if self.total_step_time_ms is not None:
            metrics["total_step_time_ms"] = self.total_step_time_ms
        if self.throughput_samples_per_sec is not None:
            metrics["throughput_samples_per_sec"] = self.throughput_samples_per_sec
        if self.throughput_steps_per_sec is not None:
            metrics["throughput_steps_per_sec"] = self.throughput_steps_per_sec

        # Quality
        if self.learning_rate is not None:
            metrics["learning_rate"] = self.learning_rate
        if self.gradient_norm is not None:
            metrics["gradient_norm"] = self.gradient_norm
        if self.gradient_mean is not None:
            metrics["gradient_mean"] = self.gradient_mean
        if self.gradient_std is not None:
            metrics["gradient_std"] = self.gradient_std
        if self.gradient_max is not None:
            metrics["gradient_max"] = self.gradient_max
        if self.gradient_min is not None:
            metrics["gradient_min"] = self.gradient_min
        if self.param_update_magnitude is not None:
            metrics["param_update_magnitude"] = self.param_update_magnitude
        if self.param_norm is not None:
            metrics["param_norm"] = self.param_norm

        # Communication
        if self.allreduce_time_ms is not None:
            metrics["allreduce_time_ms"] = self.allreduce_time_ms
        if self.sync_wait_time_ms is not None:
            metrics["sync_wait_time_ms"] = self.sync_wait_time_ms
        if self.staleness is not None:
            metrics["staleness"] = float(self.staleness)
        if self.ulp_transfer_time_ms is not None:
            metrics["ulp_transfer_time_ms"] = self.ulp_transfer_time_ms
        if self.ulp_assembly_time_ms is not None:
            metrics["ulp_assembly_time_ms"] = self.ulp_assembly_time_ms

        # Stability
        if self.nan_inf_detected:
            metrics["nan_inf_detected"] = 1.0
        if self.gradient_explosion:
            metrics["gradient_explosion"] = 1.0
        if self.gradient_vanishing:
            metrics["gradient_vanishing"] = 1.0
        if self.loss_variance is not None:
            metrics["loss_variance"] = self.loss_variance

        # Load balancing
        if self.load_balance_score is not None:
            metrics["load_balance_score"] = self.load_balance_score
        if self.slow_worker_ratio is not None:
            metrics["slow_worker_ratio"] = self.slow_worker_ratio

        # Data
        if self.batch_size is not None:
            metrics["batch_size"] = float(self.batch_size)
        if self.data_loading_time_ms is not None:
            metrics["data_loading_time_ms"] = self.data_loading_time_ms

        # Additional
        metrics.update(self.additional_metrics)

        return metrics


class TrainingMetricsCollector:
    """
    Training metrics collector with batching and async reporting.

    Collects metrics from multiple sources and batches them for efficient reporting.
    Non-blocking and async to avoid impacting training performance.
    """

    def __init__(
        self, spmd_id: Optional[str] = None, batch_size: int = 10, batch_timeout: float = 5.0
    ):
        """
        Initialize training metrics collector.

        Args:
            spmd_id: SPMD identifier (if None, collects at task level)
            batch_size: Number of metrics to batch before reporting
            batch_timeout: Maximum time to wait before reporting batch (seconds)
        """
        self.spmd_id = spmd_id
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout

        self._metrics_buffer: Deque[TrainingMetrics] = deque(maxlen=batch_size * 2)
        self._last_report_time: float = time.time()
        self._step_timings: Dict[int, Dict[str, float]] = {}  # step -> {event: timestamp}
        self._loss_history: Deque[float] = deque(maxlen=100)  # For variance calculation

    def start_step(self, step: int) -> None:
        """
        Mark start of a training step.

        Args:
            step: Step number
        """
        self._step_timings[step] = {"step_start": time.time()}

    def record_forward_start(self, step: int) -> None:
        """Record forward pass start."""
        if step in self._step_timings:
            self._step_timings[step]["forward_start"] = time.time()

    def record_forward_end(self, step: int) -> None:
        """Record forward pass end."""
        if step in self._step_timings:
            self._step_timings[step]["forward_end"] = time.time()

    def record_backward_start(self, step: int) -> None:
        """Record backward pass start."""
        if step in self._step_timings:
            self._step_timings[step]["backward_start"] = time.time()

    def record_backward_end(self, step: int) -> None:
        """Record backward pass end."""
        if step in self._step_timings:
            self._step_timings[step]["backward_end"] = time.time()

    def record_update_start(self, step: int) -> None:
        """Record parameter update start."""
        if step in self._step_timings:
            self._step_timings[step]["update_start"] = time.time()

    def record_update_end(self, step: int) -> None:
        """Record parameter update end."""
        if step in self._step_timings:
            self._step_timings[step]["update_end"] = time.time()

    def record_sync_wait(self, step: int, wait_time_ms: float) -> None:
        """Record synchronization wait time."""
        if step in self._step_timings:
            self._step_timings[step]["sync_wait_ms"] = wait_time_ms

    def add_metrics(self, step: int, metrics: TrainingMetrics) -> Optional[Dict[str, float]]:
        """
        Add training metrics (non-blocking).

        Args:
            step: Step number
            metrics: TrainingMetrics instance

        Returns:
            Batched metrics dict if batch is ready, None otherwise
        """
        # Calculate timings if available
        if step in self._step_timings:
            timings = self._step_timings[step]

            if "forward_start" in timings and "forward_end" in timings:
                metrics.forward_time_ms = (timings["forward_end"] - timings["forward_start"]) * 1000

            if "backward_start" in timings and "backward_end" in timings:
                metrics.backward_time_ms = (
                    timings["backward_end"] - timings["backward_start"]
                ) * 1000

            if "update_start" in timings and "update_end" in timings:
                metrics.update_time_ms = (timings["update_end"] - timings["update_start"]) * 1000

            if "step_start" in timings:
                total_time = time.time() - timings["step_start"]
                metrics.total_step_time_ms = total_time * 1000
                if total_time > 0:
                    metrics.throughput_steps_per_sec = 1.0 / total_time

                    # Calculate throughput_samples_per_sec if batch_size is available
                    if metrics.batch_size is not None and metrics.batch_size > 0:
                        metrics.throughput_samples_per_sec = metrics.batch_size / total_time

        # If total_step_time_ms is already set but throughput not calculated, calculate it
        if metrics.total_step_time_ms is not None and metrics.total_step_time_ms > 0:
            if metrics.throughput_steps_per_sec is None:
                # Calculate from total_step_time_ms
                metrics.throughput_steps_per_sec = 1000.0 / metrics.total_step_time_ms

            # Calculate throughput_samples_per_sec if batch_size is available
            if (
                metrics.throughput_samples_per_sec is None
                and metrics.batch_size is not None
                and metrics.batch_size > 0
            ):
                step_time_sec = metrics.total_step_time_ms / 1000.0
                if step_time_sec > 0:
                    metrics.throughput_samples_per_sec = metrics.batch_size / step_time_sec

        # Calculate loss variance if loss is available
        if metrics.loss is not None:
            self._loss_history.append(metrics.loss)
            if len(self._loss_history) >= 10:
                try:
                    import numpy as np

                    metrics.loss_variance = float(np.var(list(self._loss_history)))
                except ImportError:
                    # Fallback to manual calculation if numpy not available
                    loss_list = list(self._loss_history)
                    mean_loss = sum(loss_list) / len(loss_list)
                    variance = sum((x - mean_loss) ** 2 for x in loss_list) / len(loss_list)
                    metrics.loss_variance = float(variance)

        # Add to buffer
        self._metrics_buffer.append(metrics)

        # Check if should report
        if self._should_report():
            return self._flush_batch()

        return None

    def _should_report(self) -> bool:
        """Check if should report batch."""
        # Check batch size
        if len(self._metrics_buffer) >= self.batch_size:
            return True

        # Check timeout
        elapsed = time.time() - self._last_report_time
        if elapsed >= self.batch_timeout:
            return True

        return False

    def _flush_batch(self) -> Dict[str, float]:
        """
        Flush metrics buffer and return aggregated metrics.

        Returns:
            Aggregated metrics dictionary
        """
        if not self._metrics_buffer:
            return {}

        # Aggregate metrics (mean for most, sum for counts)
        aggregated = {}
        metrics_list = list(self._metrics_buffer)

        # Aggregate each metric
        metric_dicts = [m.to_dict() for m in metrics_list]

        # Get all metric keys
        all_keys = set()
        for md in metric_dicts:
            all_keys.update(md.keys())

        # Aggregate (mean for most metrics, sum for counts)
        count_metrics = {"nan_inf_detected", "gradient_explosion", "gradient_vanishing"}

        for key in all_keys:
            values = [md.get(key) for md in metric_dicts if key in md]
            if not values:
                continue

            if key in count_metrics:
                aggregated[key] = sum(values)
            else:
                # Mean
                aggregated[key] = sum(values) / len(values)

        # Clear buffer
        self._metrics_buffer.clear()
        self._last_report_time = time.time()

        # Clean up old step timings (keep last 100 steps)
        if len(self._step_timings) > 100:
            oldest_step = min(self._step_timings.keys())
            del self._step_timings[oldest_step]

        return aggregated

    def flush(self) -> Dict[str, float]:
        """Force flush current batch."""
        return self._flush_batch()
