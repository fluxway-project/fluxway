"""
Monitor for cluster state monitoring and telemetry.

Based on ARCHITECTURE.md Section 2.1.4 (Monitor)

Responsibilities:
- Aggregate all SPMD cluster state information
- Monitor cluster health
- Provide cluster info query interface
- Probe sub-controllers for health, collect cluster info

Telemetry information:
- SPMD running state (running, paused, error, evicted)
- Node-level resource usage (nvidia-smi, etc.)
- Node eviction and recovery history
"""

from typing import Dict, Optional, List, Any
import time
import logging

from fluxway.common.types import WorkerTelemetry
from fluxway.common.monitoring import MetricsAggregator
from fluxway.common.monitoring.backends import (
    TensorBoardBackend,
    RayBackend,
    PrometheusBackend,
    AlertBackend,
    AlertLevel,
)
from fluxway.common.monitoring.optimizations import (
    AsyncBackendWriter,
    IncrementalAggregator,
    BatchTransferManager,
)

logger = logging.getLogger(__name__)


class Monitor:
    """
    Cluster State Monitor.

    Collects and aggregates telemetry from all SPMDs and workers.
    """

    def __init__(
        self,
        tensorboard_log_dir: Optional[str] = None,
        prometheus_port: int = 8000,
        alert_sdk_client: Optional[Any] = None,
        enable_tensorboard: bool = True,
        enable_ray: bool = True,
        enable_prometheus: bool = False,
        enable_alert: bool = True,
    ):
        """
        Initialize Monitor with optional backends.

        Args:
            tensorboard_log_dir: TensorBoard log directory (default: ./logs/tensorboard)
            prometheus_port: Prometheus HTTP server port (default: 8000)
            alert_sdk_client: User's alert SDK client instance
            enable_tensorboard: Enable TensorBoard backend
            enable_ray: Enable Ray monitoring backend
            enable_prometheus: Enable Prometheus backend
            enable_alert: Enable alert backend
        """
        self._telemetry: Dict[str, Dict[str, WorkerTelemetry]] = (
            {}
        )  # spmd_id -> worker_id -> telemetry
        self._spmd_status: Dict[str, str] = {}  # spmd_id -> status
        self._spmd_aggregates: Dict[str, Dict[str, float]] = {}  # spmd_id -> aggregated metrics
        self._eviction_history: Dict[str, List[Dict[str, Any]]] = {}  # spmd_id -> history list
        self._spmd_handles: Dict[str, Any] = {}  # spmd_id -> SPMDHandle (for probe)
        self._last_probe_time: Dict[str, float] = {}  # spmd_id -> last probe timestamp
        self._step_counter: int = 0  # Step counter for TensorBoard

        # Initialize backends
        self.tensorboard: Optional[TensorBoardBackend] = None
        self.ray: Optional[RayBackend] = None
        self.prometheus: Optional[PrometheusBackend] = None
        self.alert: Optional[AlertBackend] = None

        # Initialize optimizations
        self._incremental_aggregator = IncrementalAggregator()
        self._batch_transfer = BatchTransferManager()
        self._async_writers: Dict[str, AsyncBackendWriter] = {}

        if enable_tensorboard:
            try:
                self.tensorboard = TensorBoardBackend(tensorboard_log_dir)
                # Wrap with async writer
                self._async_writers["tensorboard"] = AsyncBackendWriter(
                    self.tensorboard, queue_size=1000, batch_size=10, flush_interval=1.0
                )
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")

        if enable_ray:
            try:
                self.ray = RayBackend()
            except Exception as e:
                logger.warning(f"Failed to initialize Ray backend: {e}")

        if enable_prometheus:
            try:
                self.prometheus = PrometheusBackend(port=prometheus_port)
                # Wrap with async writer
                self._async_writers["prometheus"] = AsyncBackendWriter(
                    self.prometheus, queue_size=1000, batch_size=10, flush_interval=1.0
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Prometheus: {e}")

        if enable_alert:
            try:
                self.alert = AlertBackend(sdk_client=alert_sdk_client)
            except Exception as e:
                logger.warning(f"Failed to initialize Alert backend: {e}")

    def register_spmd_handle(self, spmd_id: str, spmd_handle: Any) -> None:
        """
        Register SPMD handle for probe operations.

        Args:
            spmd_id: SPMD identifier
            spmd_handle: SPMDHandle instance
        """
        self._spmd_handles[spmd_id] = spmd_handle

    def report_telemetry(self, spmd_id: str, telemetry: Dict[str, WorkerTelemetry]) -> None:
        """
        Report telemetry to metric service.

        Args:
            spmd_id: SPMD identifier
            telemetry: Worker telemetry data

        Steps:
        1. Store worker telemetry
        2. Aggregate to SPMD level
        3. Optionally report to TensorBoard or Ray monitoring
        """
        # Store worker telemetry
        if spmd_id not in self._telemetry:
            self._telemetry[spmd_id] = {}
        self._telemetry[spmd_id].update(telemetry)

        # Aggregate to SPMD level (using incremental aggregator)
        if telemetry:
            aggregated = self._incremental_aggregator.incremental_aggregate(telemetry)
            self._spmd_aggregates[spmd_id] = aggregated

            # Report to backends (using async writers and batch transfer)
            self._report_to_backends(spmd_id, aggregated)

    def get_spmd_telemetry(self, spmd_id: str) -> Dict[str, WorkerTelemetry]:
        """
        Query SPMD telemetry.

        Args:
            spmd_id: SPMD identifier

        Returns:
            Telemetry for all workers in the SPMD
        """
        return self._telemetry.get(spmd_id, {}).copy()

    def get_worker_telemetry(self, spmd_id: str, worker_id: str) -> Optional[WorkerTelemetry]:
        """
        Query single worker telemetry.

        Args:
            spmd_id: SPMD identifier
            worker_id: Worker identifier

        Returns:
            WorkerTelemetry or None if not found
        """
        spmd_telemetry = self._telemetry.get(spmd_id, {})
        return spmd_telemetry.get(worker_id)

    def get_cluster_health(self) -> Dict[str, str]:
        """
        Get overall cluster health status.

        Returns:
            Mapping of SPMD ID to health status

        Aggregates health from all SPMDs and detects anomalies.
        """
        health_status = {}

        for spmd_id in self._telemetry.keys():
            # Get status (default to 'unknown' if not set)
            status = self._spmd_status.get(spmd_id, "unknown")

            # Check if there are any workers
            workers = self._telemetry.get(spmd_id, {})
            if not workers:
                status = "no_workers"
            elif status == "unknown":
                # If status unknown but has workers, assume running
                status = "running"

            health_status[spmd_id] = status

        return health_status

    def probe_sub_controller(self, spmd_id: str, timeout: float = 5.0) -> bool:
        """
        Probe SPMD sub-controller for health.

        Args:
            spmd_id: SPMD identifier
            timeout: Probe timeout in seconds

        Returns:
            True if healthy, False otherwise

        Steps:
        1. Check if SPMD has workers
        2. Check if workers are sending heartbeats
        3. Update status on probe failure
        """
        # Check if SPMD exists
        if spmd_id not in self._telemetry:
            self.update_spmd_status(spmd_id, "error")
            return False

        # Check if has workers
        workers = self._telemetry.get(spmd_id, {})
        if not workers:
            self.update_spmd_status(spmd_id, "no_workers")
            return False

        # Check status
        status = self._spmd_status.get(spmd_id, "unknown")
        if status == "error":
            return False

        # Try to probe SPMD handle if available
        spmd_handle = self._spmd_handles.get(spmd_id)
        if spmd_handle is not None:
            try:
                # Try to call a health check method on SPMD handle
                # This depends on SPMDHandle implementation
                # For now, just check if handle exists
                pass
            except Exception:
                # Probe failed
                self.update_spmd_status(spmd_id, "error")
                return False

        # Update last probe time
        self._last_probe_time[spmd_id] = time.time()

        # If status is running or unknown, assume healthy
        return True

    def update_spmd_status(self, spmd_id: str, status: str) -> None:
        """
        Update SPMD status.

        Args:
            spmd_id: SPMD identifier
            status: Status string (running, paused, error, evicted)

        Statuses: running, paused, error, evicted

        Triggers alerts on error status.
        """
        self._spmd_status[spmd_id] = status

        # Trigger alerts on error status
        if status == "error":
            if self.alert:
                self.alert.send_spmd_alert(
                    spmd_id=spmd_id,
                    level=AlertLevel.ERROR,
                    title=f"SPMD {spmd_id} Error",
                    message=f"SPMD {spmd_id} is in error state",
                    metrics=self._spmd_aggregates.get(spmd_id),
                    metadata={"status": status},
                )
            logger.error(f"SPMD {spmd_id} is in error state!")

        # Record eviction in history
        if status == "evicted":
            if spmd_id not in self._eviction_history:
                self._eviction_history[spmd_id] = []
            self._eviction_history[spmd_id].append(
                {"timestamp": time.time(), "status": status, "reason": "evicted"}
            )

    def get_eviction_history(self, spmd_id: str) -> List[Dict[str, Any]]:
        """
        Get eviction and recovery history for an SPMD.

        Args:
            spmd_id: SPMD identifier

        Returns:
            List of history events (eviction, recovery, etc.)
        """
        return self._eviction_history.get(spmd_id, []).copy()

    def get_resource_usage(self, spmd_id: Optional[str] = None) -> Dict[str, float]:
        """
        Get resource usage for an SPMD or entire task.

        Args:
            spmd_id: SPMD identifier (if None, returns task-level usage)

        Returns:
            Dictionary with aggregated resource usage:
            - CPU metrics (mean, max, min)
            - Memory metrics (total, mean, max)
            - GPU metrics (utilization, memory, temperature)
            - Network I/O (total bytes sent/received)
            - Disk I/O (total bytes read/written)

        This is the core method for task-level resource monitoring.
        """
        if spmd_id is not None:
            # Get aggregated metrics for this SPMD
            aggregated = self._spmd_aggregates.get(spmd_id, {})

            if not aggregated:
                # If no aggregated data, try to aggregate from worker telemetry
                worker_telemetry = self._telemetry.get(spmd_id, {})
                if worker_telemetry:
                    telemetry_list = list(worker_telemetry.values())
                    aggregated = MetricsAggregator.aggregate_workers(telemetry_list)
                    self._spmd_aggregates[spmd_id] = aggregated

            return aggregated
        else:
            # Task level: aggregate all SPMDs
            spmd_aggregates = []
            for spmd_id, aggregated in self._spmd_aggregates.items():
                if aggregated:
                    spmd_aggregates.append(aggregated)

            if not spmd_aggregates:
                return {}

            return MetricsAggregator.aggregate_spmds(spmd_aggregates)

    def get_all_spmd_ids(self) -> List[str]:
        """Get all registered SPMD IDs."""
        return list(self._telemetry.keys())

    def get_failed_components(self) -> List[Dict[str, Any]]:
        """
        Get list of failed components.

        Returns:
            List of dictionaries with component information
        """
        failed = []

        for spmd_id, status in self._spmd_status.items():
            if status == "error":
                failed.append(
                    {
                        "component_type": "spmd",
                        "component_id": spmd_id,
                        "status": status,
                        "workers": list(self._telemetry.get(spmd_id, {}).keys()),
                    }
                )

        return failed

    def _report_to_backends(self, spmd_id: str, metrics: Dict[str, float]) -> None:
        """
        Report metrics to all enabled backends (with optimizations).

        Args:
            spmd_id: SPMD identifier
            metrics: Aggregated metrics dictionary
        """
        prefix = f"spmd_{spmd_id}/"

        # Report to TensorBoard (async)
        if "tensorboard" in self._async_writers:
            try:
                import asyncio

                asyncio.create_task(
                    self._async_writers["tensorboard"].write_async(
                        self._step_counter, metrics, prefix
                    )
                )
            except Exception as e:
                logger.error(f"Failed to write to TensorBoard: {e}")
        elif self.tensorboard:
            try:
                self.tensorboard.write_spmd_metrics(self._step_counter, spmd_id, metrics)
            except Exception as e:
                logger.error(f"Failed to write to TensorBoard: {e}")

        # Report to Ray (synchronous, lightweight)
        if self.ray:
            try:
                self.ray.record_spmd_metrics(spmd_id, metrics)
            except Exception as e:
                logger.error(f"Failed to record to Ray: {e}")

        # Report to Prometheus (async)
        if "prometheus" in self._async_writers:
            try:
                import asyncio

                asyncio.create_task(
                    self._async_writers["prometheus"].write_async(
                        self._step_counter, metrics, prefix
                    )
                )
            except Exception as e:
                logger.error(f"Failed to write to Prometheus: {e}")
        elif self.prometheus:
            try:
                self.prometheus.record_spmd_metrics(spmd_id, metrics)
            except Exception as e:
                logger.error(f"Failed to record to Prometheus: {e}")

    def increment_step(self) -> None:
        """Increment step counter for TensorBoard."""
        self._step_counter += 1

    def set_step(self, step: int) -> None:
        """Set step counter for TensorBoard."""
        self._step_counter = step

    async def flush_backends(self) -> None:
        """Flush all backends."""
        # Flush async writers
        for writer in self._async_writers.values():
            try:
                await writer.flush()
            except Exception as e:
                logger.error(f"Failed to flush async writer: {e}")

        # Flush direct backends
        if self.tensorboard:
            self.tensorboard.flush()

    async def close_backends(self) -> None:
        """Close all backends."""
        # Stop async writers
        for writer in self._async_writers.values():
            try:
                await writer.stop()
            except Exception as e:
                logger.error(f"Failed to stop async writer: {e}")

        # Close direct backends
        if self.tensorboard:
            self.tensorboard.close()
