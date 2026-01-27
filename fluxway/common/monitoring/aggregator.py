"""
Metrics Aggregator for multi-level aggregation.

Aggregates metrics across:
- Worker level: Direct metrics (no aggregation needed)
- SPMD level: Aggregate all workers in an SPMD
- Task level: Aggregate all SPMDs in a job
"""

from typing import Dict, List
import statistics

from ..types import WorkerTelemetry


class MetricsAggregator:
    """
    Metrics aggregator for multi-level aggregation.

    Supports aggregation across:
    - Worker level: Direct metrics (no aggregation needed)
    - SPMD level: Aggregate workers within an SPMD
    - Task level: Aggregate SPMDs within a job
    """

    @staticmethod
    def aggregate_workers(telemetries: List[WorkerTelemetry]) -> Dict[str, float]:
        """
        Aggregate metrics from multiple workers (SPMD level).

        Args:
            telemetries: List of WorkerTelemetry from workers in the same SPMD

        Returns:
            Dictionary with aggregated metrics:
            - cpu_percent_mean, cpu_percent_max, cpu_percent_min
            - memory_rss_mb_mean, memory_rss_mb_max, memory_rss_mb_total
            - gpu_utilization_mean, gpu_memory_used_total, etc.
            - network_bytes_sent_total, network_bytes_recv_total
            - disk_read_bytes_total, disk_write_bytes_total
        """
        if not telemetries:
            return {}

        # Extract metrics
        cpu_percents = [t.cpu_percent for t in telemetries]
        memory_rss = [t.memory_rss_mb for t in telemetries]
        memory_vms = [t.memory_vms_mb for t in telemetries]
        memory_percents = [t.memory_percent for t in telemetries]

        # Network I/O (sum across workers)
        network_sent_total = sum(t.network_bytes_sent for t in telemetries)
        network_recv_total = sum(t.network_bytes_recv for t in telemetries)

        # Disk I/O (sum across workers)
        disk_read_total = sum(t.disk_read_bytes for t in telemetries)
        disk_write_total = sum(t.disk_write_bytes for t in telemetries)

        # GPU metrics aggregation
        gpu_metrics = MetricsAggregator._aggregate_gpu_metrics(telemetries)

        # Build aggregated result
        aggregated = {
            # CPU metrics
            "cpu_percent_mean": statistics.mean(cpu_percents) if cpu_percents else 0.0,
            "cpu_percent_max": max(cpu_percents) if cpu_percents else 0.0,
            "cpu_percent_min": min(cpu_percents) if cpu_percents else 0.0,
            "cpu_count": telemetries[0].cpu_count if telemetries else 0,
            # Memory metrics
            "memory_rss_mb_mean": statistics.mean(memory_rss) if memory_rss else 0.0,
            "memory_rss_mb_max": max(memory_rss) if memory_rss else 0.0,
            "memory_rss_mb_total": sum(memory_rss),
            "memory_vms_mb_mean": statistics.mean(memory_vms) if memory_vms else 0.0,
            "memory_vms_mb_total": sum(memory_vms),
            "memory_percent_mean": statistics.mean(memory_percents) if memory_percents else 0.0,
            "memory_percent_max": max(memory_percents) if memory_percents else 0.0,
            # Network I/O (total across workers)
            "network_bytes_sent_total": network_sent_total,
            "network_bytes_recv_total": network_recv_total,
            # Disk I/O (total across workers)
            "disk_read_bytes_total": disk_read_total,
            "disk_write_bytes_total": disk_write_total,
            # Worker count
            "worker_count": len(telemetries),
        }

        # Add GPU metrics
        aggregated.update(gpu_metrics)

        return aggregated

    @staticmethod
    def _aggregate_gpu_metrics(telemetries: List[WorkerTelemetry]) -> Dict[str, float]:
        """
        Aggregate GPU metrics across workers.

        Args:
            telemetries: List of WorkerTelemetry

        Returns:
            Dictionary with aggregated GPU metrics
        """
        # Collect all GPU IDs
        all_gpu_ids = set()
        for t in telemetries:
            all_gpu_ids.update(t.gpu_ids)

        if not all_gpu_ids:
            return {}

        aggregated = {}

        # Aggregate per GPU
        for gpu_id in sorted(all_gpu_ids):
            # Collect metrics for this GPU from all workers
            utilizations = []
            memory_used = []
            memory_totals = []
            temperatures = []

            for t in telemetries:
                if gpu_id in t.gpu_ids:
                    util = t.gpu_utilization_percent.get(gpu_id, 0.0)
                    if util > 0:  # Only include valid values
                        utilizations.append(util)

                    mem_used = t.gpu_memory_used_mb.get(gpu_id, 0.0)
                    if mem_used > 0:
                        memory_used.append(mem_used)

                    mem_total = t.gpu_memory_total_mb.get(gpu_id, 0.0)
                    if mem_total > 0:
                        memory_totals.append(mem_total)

                    temp = t.gpu_temperature_c.get(gpu_id, 0.0)
                    if temp > 0:  # Only include valid temperatures
                        temperatures.append(temp)

            if utilizations:
                aggregated[f"gpu_{gpu_id}_utilization_mean"] = statistics.mean(utilizations)
                aggregated[f"gpu_{gpu_id}_utilization_max"] = max(utilizations)

            if memory_used:
                aggregated[f"gpu_{gpu_id}_memory_used_total"] = sum(memory_used)

            if memory_totals:
                aggregated[f"gpu_{gpu_id}_memory_total"] = memory_totals[
                    0
                ]  # Should be same for all

            if temperatures:
                aggregated[f"gpu_{gpu_id}_temperature_mean"] = statistics.mean(temperatures)
                aggregated[f"gpu_{gpu_id}_temperature_max"] = max(temperatures)

        # Overall GPU metrics
        all_utilizations = []
        all_memory_used = []
        for t in telemetries:
            for gpu_id in t.gpu_ids:
                util = t.gpu_utilization_percent.get(gpu_id, 0.0)
                if util > 0:
                    all_utilizations.append(util)
                mem_used = t.gpu_memory_used_mb.get(gpu_id, 0.0)
                if mem_used > 0:
                    all_memory_used.append(mem_used)

        if all_utilizations:
            aggregated["gpu_utilization_mean"] = statistics.mean(all_utilizations)
            aggregated["gpu_utilization_max"] = max(all_utilizations)

        if all_memory_used:
            aggregated["gpu_memory_used_total"] = sum(all_memory_used)

        return aggregated

    @staticmethod
    def aggregate_spmds(spmd_aggregates: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics from multiple SPMDs (Task level).

        Args:
            spmd_aggregates: List of aggregated metrics from each SPMD

        Returns:
            Dictionary with task-level aggregated metrics
        """
        if not spmd_aggregates:
            return {}

        # Collect all metric keys
        all_keys = set()
        for agg in spmd_aggregates:
            all_keys.update(agg.keys())

        task_aggregated = {}

        # Aggregate each metric
        for key in all_keys:
            values = [agg.get(key, 0.0) for agg in spmd_aggregates if key in agg]

            if not values:
                continue

            # Determine aggregation strategy based on metric name
            if "total" in key or "sum" in key or "count" in key:
                # Sum metrics (e.g., network_bytes_sent_total, worker_count)
                task_aggregated[f"task_{key}"] = sum(values)
            else:
                # Mean/Max/Min metrics
                task_aggregated[f"task_{key}_mean"] = statistics.mean(values)
                task_aggregated[f"task_{key}_max"] = max(values)
                task_aggregated[f"task_{key}_min"] = min(values)

        # Add SPMD count
        task_aggregated["spmd_count"] = len(spmd_aggregates)

        return task_aggregated
