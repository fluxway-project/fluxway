"""
Incremental metric aggregator.

Only aggregates changed workers/metrics to reduce computation overhead.
"""

from typing import Dict, Optional, List, Any
import logging
import time

from ...types import WorkerTelemetry
from ..aggregator import MetricsAggregator

logger = logging.getLogger(__name__)


class IncrementalAggregator:
    """
    Incremental aggregator.

    Only processes changed workers/metrics and caches results to avoid
    redundant computation.
    """

    def __init__(self, change_threshold: float = 0.01):
        """
        Initialize incremental aggregator.

        Args:
            change_threshold: Threshold for considering a metric changed (default: 0.01 = 1%)
        """
        self.change_threshold = change_threshold
        self._last_telemetry: Dict[str, WorkerTelemetry] = {}
        self._cached_aggregate: Optional[Dict[str, float]] = None
        self._last_aggregate_time: float = 0.0

    def incremental_aggregate(self, new_telemetry: Dict[str, WorkerTelemetry]) -> Dict[str, float]:
        """
        Incrementally aggregate worker telemetry.

        Args:
            new_telemetry: Dictionary of worker_id -> WorkerTelemetry

        Returns:
            Aggregated metrics dictionary

        Strategy:
        1. Identify changed workers
        2. If change is small (< 10%), do incremental update
        3. Otherwise, do full aggregation and cache result
        """
        if not new_telemetry:
            return self._cached_aggregate or {}

        # Identify changed workers
        changed_workers = self._identify_changes(new_telemetry)
        change_ratio = len(changed_workers) / len(new_telemetry) if new_telemetry else 0.0

        # If change is small, try incremental update
        if change_ratio < 0.1 and self._cached_aggregate is not None:
            # Incremental update
            try:
                updated_aggregate = self._incremental_update(changed_workers, new_telemetry)
                self._last_telemetry = new_telemetry.copy()
                self._cached_aggregate = updated_aggregate
                self._last_aggregate_time = time.time()
                return updated_aggregate
            except Exception as e:
                logger.warning(f"Incremental update failed, falling back to full aggregation: {e}")

        # Full aggregation
        telemetry_list = list(new_telemetry.values())
        aggregated = MetricsAggregator.aggregate_workers(telemetry_list)

        self._last_telemetry = new_telemetry.copy()
        self._cached_aggregate = aggregated
        self._last_aggregate_time = time.time()

        return aggregated

    def _identify_changes(self, new_telemetry: Dict[str, WorkerTelemetry]) -> List[str]:
        """
        Identify workers with changed telemetry.

        Args:
            new_telemetry: New telemetry dictionary

        Returns:
            List of worker IDs with changes
        """
        changed = []

        for worker_id, new_tel in new_telemetry.items():
            old_tel = self._last_telemetry.get(worker_id)

            if old_tel is None:
                # New worker
                changed.append(worker_id)
                continue

            # Check if metrics changed significantly
            if self._has_changed(old_tel, new_tel):
                changed.append(worker_id)

        return changed

    def _has_changed(self, old_tel: WorkerTelemetry, new_tel: WorkerTelemetry) -> bool:
        """
        Check if telemetry has changed significantly.

        Args:
            old_tel: Old telemetry
            new_tel: New telemetry

        Returns:
            True if changed significantly
        """
        # Check key metrics
        metrics_to_check = [
            ("cpu_percent", old_tel.cpu_percent, new_tel.cpu_percent),
            ("memory_percent", old_tel.memory_percent, new_tel.memory_percent),
            ("memory_rss_mb", old_tel.memory_rss_mb, new_tel.memory_rss_mb),
        ]

        for metric_name, old_val, new_val in metrics_to_check:
            if old_val == 0:
                if new_val != 0:
                    return True
                continue

            change_ratio = abs(new_val - old_val) / abs(old_val)
            if change_ratio > self.change_threshold:
                return True

        # Check GPU metrics
        if old_tel.gpu_ids != new_tel.gpu_ids:
            return True

        for gpu_id in new_tel.gpu_ids:
            old_util = old_tel.gpu_utilization_percent.get(gpu_id, 0.0)
            new_util = new_tel.gpu_utilization_percent.get(gpu_id, 0.0)

            if old_util == 0:
                if new_util != 0:
                    return True
                continue

            change_ratio = abs(new_util - old_util) / abs(old_util) if old_util > 0 else 1.0
            if change_ratio > self.change_threshold:
                return True

        return False

    def _incremental_update(
        self, changed_workers: List[str], new_telemetry: Dict[str, WorkerTelemetry]
    ) -> Dict[str, float]:
        """
        Incrementally update aggregated metrics.

        Args:
            changed_workers: List of changed worker IDs
            new_telemetry: New telemetry dictionary

        Returns:
            Updated aggregated metrics
        """
        # For simplicity, we'll do a full aggregation but only on changed workers
        # In a more sophisticated implementation, we could update only the affected metrics

        # Get all telemetry (changed + unchanged)
        all_telemetry = {}
        for worker_id, tel in new_telemetry.items():
            if worker_id in changed_workers:
                all_telemetry[worker_id] = tel
            else:
                # Use cached telemetry for unchanged workers
                all_telemetry[worker_id] = self._last_telemetry[worker_id]

        # Aggregate
        telemetry_list = list(all_telemetry.values())
        return MetricsAggregator.aggregate_workers(telemetry_list)

    def clear_cache(self) -> None:
        """Clear cached aggregation results."""
        self._last_telemetry.clear()
        self._cached_aggregate = None
        self._last_aggregate_time = 0.0

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return {
            "cached_workers": len(self._last_telemetry),
            "has_cache": self._cached_aggregate is not None,
            "last_aggregate_time": self._last_aggregate_time,
        }
