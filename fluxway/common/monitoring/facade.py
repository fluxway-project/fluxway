"""
Monitoring Facade - Unified entry point for monitoring functionality.

This module provides a facade interface to minimize code intrusion.
Other modules only need to call simple APIs without knowing internal implementation.
"""

from typing import Optional, Dict, Any, Callable
import time
import os
import threading
import asyncio
import logging

try:
    import ray
except ImportError:
    ray = None

from ..types import WorkerTelemetry
from .aggregator import MetricsAggregator
from .training_metrics import TrainingMetrics, TrainingMetricsCollector

logger = logging.getLogger(__name__)


class MonitoringFacade:
    """
    Monitoring Facade - Unified entry point for monitoring.

    Provides simple APIs for other modules to use monitoring functionality
    without knowing internal implementation details.
    """

    # Class-level storage for monitor instances
    _spmd_monitors: Dict[str, Any] = {}  # spmd_id -> SPMDMonitor instance
    _main_monitor: Optional[Any] = None  # Main Monitor instance
    _training_collectors: Dict[str, TrainingMetricsCollector] = (
        {}
    )  # spmd_id -> TrainingMetricsCollector
    
    # Async writers for backends (lazy initialization)
    _async_writers: Dict[str, Any] = {}  # backend_name -> AsyncBackendWriter
    _async_writers_lock = threading.Lock()
    _async_loop: Optional[asyncio.AbstractEventLoop] = None
    _async_loop_thread: Optional[threading.Thread] = None

    @staticmethod
    def create_worker_collector(worker_id: str, spmd_id: str):
        """
        Create a Worker metrics collector (Ray Actor).

        Args:
            worker_id: Worker identifier
            spmd_id: SPMD identifier

        Returns:
            Ray Actor handle to WorkerMetricsCollector

        Example:
            collector = MonitoringFacade.create_worker_collector("worker_0", "spmd_1")
            telemetry = ray.get(collector.collect.remote())
        """
        if ray is None:
            raise RuntimeError("Ray is not installed. Install it with: pip install ray")

        from ...spmd.driver.monitor.worker_collector import WorkerMetricsCollector

        return WorkerMetricsCollector.remote(worker_id, spmd_id)

    @staticmethod
    def send_heartbeat(
        worker_id: str, spmd_id: str, telemetry: WorkerTelemetry, spmd_monitor: Optional[Any] = None
    ) -> None:
        """
        Send heartbeat from Worker to SPMDMonitor.

        Args:
            worker_id: Worker identifier
            spmd_id: SPMD identifier
            telemetry: Worker telemetry data
            spmd_monitor: SPMDMonitor instance (if None, will try to get from cache)

        Example:
            telemetry = ray.get(collector.collect.remote())
            MonitoringFacade.send_heartbeat("worker_0", "spmd_1", telemetry)
        """
        # Get SPMDMonitor instance
        if spmd_monitor is None:
            spmd_monitor = MonitoringFacade._spmd_monitors.get(spmd_id)
            if spmd_monitor is None:
                # Try to import and create if not exists
                from ...spmd.driver.monitor.monitor import SPMDMonitor

                spmd_monitor = SPMDMonitor(spmd_id)
                MonitoringFacade._spmd_monitors[spmd_id] = spmd_monitor

        # Send heartbeat
        spmd_monitor.receive_heartbeat(
            worker_id=worker_id, telemetry=telemetry, timestamp=telemetry.heartbeat_timestamp
        )

        # Report telemetry
        spmd_monitor.report_single_worker_telemetry(worker_id, telemetry)

    @staticmethod
    def register_spmd_monitor(spmd_id: str, spmd_monitor: Any) -> None:
        """
        Register SPMDMonitor instance.

        Args:
            spmd_id: SPMD identifier
            spmd_monitor: SPMDMonitor instance

        This is typically called by SPMDDriver during initialization.
        """
        MonitoringFacade._spmd_monitors[spmd_id] = spmd_monitor

    @staticmethod
    def register_main_monitor(main_monitor: Any) -> None:
        """
        Register main Monitor instance.

        Args:
            main_monitor: Monitor instance from Controller

        This is typically called by Controller during initialization.
        """
        MonitoringFacade._main_monitor = main_monitor

        # Link SPMDMonitors to main monitor
        for spmd_monitor in MonitoringFacade._spmd_monitors.values():
            if hasattr(spmd_monitor, "set_main_monitor_ref"):
                spmd_monitor.set_main_monitor_ref(main_monitor)

    @staticmethod
    def get_resource_usage(spmd_id: Optional[str] = None) -> Dict[str, float]:
        """
        Get resource usage (SPMD or task level).

        Args:
            spmd_id: SPMD identifier (if None, returns task-level usage)

        Returns:
            Dictionary with aggregated resource usage metrics

        Example:
            # SPMD level
            spmd_usage = MonitoringFacade.get_resource_usage("spmd_1")

            # Task level
            task_usage = MonitoringFacade.get_resource_usage()
        """
        if spmd_id is not None:
            # SPMD level
            spmd_monitor = MonitoringFacade._spmd_monitors.get(spmd_id)
            if spmd_monitor is None:
                return {}
            return spmd_monitor.aggregate_telemetry()
        else:
            # Task level
            if MonitoringFacade._main_monitor is None:
                return {}

            # Get all SPMD aggregates
            spmd_aggregates = []
            for spmd_id, spmd_monitor in MonitoringFacade._spmd_monitors.items():
                aggregated = spmd_monitor.aggregate_telemetry()
                if aggregated:
                    spmd_aggregates.append(aggregated)

            # Aggregate to task level
            return MetricsAggregator.aggregate_spmds(spmd_aggregates)

    @staticmethod
    def check_health(component_id: str, component_type: str = "worker") -> bool:
        """
        Check health status of a component.

        Args:
            component_id: Component identifier (worker_id or spmd_id)
            component_type: Component type ("worker" or "spmd")

        Returns:
            True if healthy, False otherwise

        Example:
            is_healthy = MonitoringFacade.check_health("worker_0", "worker")
            is_healthy = MonitoringFacade.check_health("spmd_1", "spmd")
        """
        if component_type == "worker":
            # Extract spmd_id and worker_id from component_id
            # Format: "spmd_id:worker_id" or just "worker_id"
            parts = component_id.split(":")
            if len(parts) == 2:
                spmd_id, worker_id = parts
            else:
                # Try to find worker in any SPMD
                worker_id = component_id
                spmd_id = None

            if spmd_id:
                spmd_monitor = MonitoringFacade._spmd_monitors.get(spmd_id)
                if spmd_monitor:
                    return spmd_monitor.check_worker_health(worker_id)
            else:
                # Search all SPMDs
                for spmd_id, spmd_monitor in MonitoringFacade._spmd_monitors.items():
                    if spmd_monitor.check_worker_health(component_id):
                        return True
            return False

        elif component_type == "spmd":
            if MonitoringFacade._main_monitor is None:
                return False
            return MonitoringFacade._main_monitor.probe_sub_controller(component_id)

        else:
            raise ValueError(f"Unknown component_type: {component_type}")

    @staticmethod
    def get_anomalies(spmd_id: Optional[str] = None) -> Dict[str, str]:
        """
        Get detected anomalies.

        Args:
            spmd_id: SPMD identifier (if None, returns all anomalies)

        Returns:
            Dictionary mapping component_id to anomaly description

        Example:
            anomalies = MonitoringFacade.get_anomalies("spmd_1")
            all_anomalies = MonitoringFacade.get_anomalies()
        """
        if spmd_id is not None:
            spmd_monitor = MonitoringFacade._spmd_monitors.get(spmd_id)
            if spmd_monitor is None:
                return {}
            return spmd_monitor.detect_anomalies()
        else:
            # Get anomalies from all SPMDs
            all_anomalies = {}
            for spmd_id, spmd_monitor in MonitoringFacade._spmd_monitors.items():
                anomalies = spmd_monitor.detect_anomalies()
                for worker_id, description in anomalies.items():
                    all_anomalies[f"{spmd_id}:{worker_id}"] = description
            return all_anomalies

    @staticmethod
    def start_spmd_monitoring(
        spmd_id: str,
        spmd_monitor: Any,
        check_interval: float = 10.0,
        report_interval: float = 30.0,
        anomaly_callback: Optional[Callable[[str, Dict[str, str]], None]] = None,
    ) -> Any:
        """
        Start SPMD monitoring daemon.

        Args:
            spmd_id: SPMD identifier
            spmd_monitor: SPMDMonitor instance
            check_interval: Health check interval (seconds)
            report_interval: Metric report interval (seconds)
            anomaly_callback: Callback for anomaly handling

        Returns:
            SPMDMonitorDaemon instance

        Example:
            daemon = MonitoringFacade.start_spmd_monitoring("spmd_1", spmd_monitor)
        """
        from ...spmd.driver.monitor.daemon import SPMDMonitorDaemon

        daemon = SPMDMonitorDaemon(
            spmd_monitor=spmd_monitor,
            check_interval=check_interval,
            report_interval=report_interval,
            anomaly_callback=anomaly_callback,
        )
        daemon.start()
        return daemon

    @staticmethod
    def start_task_monitoring(
        monitor: Any,
        probe_interval: float = 30.0,
        health_check_interval: float = 10.0,
        failure_callback: Optional[Callable[[str, str], None]] = None,
    ) -> Any:
        """
        Start task monitoring daemon.

        Args:
            monitor: Monitor instance
            probe_interval: SPMD probe interval (seconds)
            health_check_interval: Health check interval (seconds)
            failure_callback: Callback for failure handling

        Returns:
            MonitorDaemon instance

        Example:
            daemon = MonitoringFacade.start_task_monitoring(monitor)
        """
        from ...control_plane.monitor.daemon import MonitorDaemon

        daemon = MonitorDaemon(
            monitor=monitor,
            probe_interval=probe_interval,
            health_check_interval=health_check_interval,
            failure_callback=failure_callback,
        )
        daemon.start()
        return daemon

    @staticmethod
    def _get_training_collector(spmd_id: Optional[str] = None) -> TrainingMetricsCollector:
        """
        Get or create training metrics collector for an SPMD.

        Args:
            spmd_id: SPMD identifier (if None, uses "task" as key)

        Returns:
            TrainingMetricsCollector instance
        """
        key = spmd_id or "task"
        if key not in MonitoringFacade._training_collectors:
            MonitoringFacade._training_collectors[key] = TrainingMetricsCollector(
                spmd_id=spmd_id, batch_size=10, batch_timeout=5.0
            )
        return MonitoringFacade._training_collectors[key]

    @staticmethod
    def record_training_progress(
        step: int,
        spmd_id: Optional[str] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        component_type: str = "spmd",
        component_id: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Record training progress metrics (non-blocking, batched).

        Args:
            step: Current training step
            spmd_id: SPMD identifier (if None, records at task level)
            epoch: Current epoch number
            loss: Loss value
            metrics: Dictionary of additional metrics (e.g., accuracy, throughput)
            component_type: Type of component reporting ("spmd" or "task")
            component_id: Component identifier (defaults to spmd_id)
            batch_size: Batch size for this step (used to calculate throughput_samples_per_sec)

        This method is non-blocking and batches metrics for efficient reporting.

        Throughput calculation:
        - throughput_steps_per_sec: Automatically calculated from step timing if start_step() was called
        - throughput_samples_per_sec: Calculated from batch_size / step_time if batch_size is provided

        Example:
            MonitoringFacade.record_training_progress(
                step=100,
                epoch=5,
                loss=0.1234,
                metrics={"accuracy": 0.95},
                batch_size=32,
                spmd_id="spmd_1"
            )
        """
        try:
            import logging

            logger = logging.getLogger(__name__)

            # Create training metrics
            training_metrics = TrainingMetrics(
                step=step, epoch=epoch, loss=loss, batch_size=batch_size
            )

            # Add additional metrics
            if metrics:
                # Extract batch_size from metrics if not provided directly
                if batch_size is None and "batch_size" in metrics:
                    training_metrics.batch_size = int(metrics["batch_size"])

                # Extract total_step_time_ms from metrics if available (for throughput calculation)
                if "total_step_time_ms" in metrics:
                    training_metrics.total_step_time_ms = metrics["total_step_time_ms"]

                training_metrics.additional_metrics.update(metrics)

            # Get collector
            collector = MonitoringFacade._get_training_collector(spmd_id)

            # Add metrics (non-blocking, may return batched metrics)
            batched_metrics = collector.add_metrics(step, training_metrics)

            # If batch is ready, report asynchronously
            if batched_metrics:
                MonitoringFacade._report_training_metrics_async(
                    step=step, spmd_id=spmd_id, metrics=batched_metrics
                )

            # Log basic progress (non-blocking)
            if step % 100 == 0:  # Log every 100 steps
                logger.info(
                    f"Training progress: step={step}, epoch={epoch}, loss={loss}, spmd_id={spmd_id}"
                )

        except Exception as e:
            # Non-intrusive: continue if monitoring fails
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to record training progress: {e}")

    @staticmethod
    def _get_async_loop() -> asyncio.AbstractEventLoop:
        """Get or create the async event loop for background writes."""
        if MonitoringFacade._async_loop is None or MonitoringFacade._async_loop.is_closed():
            # Create new event loop in a background thread
            def run_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                MonitoringFacade._async_loop = loop
                loop.run_forever()
            
            MonitoringFacade._async_loop_thread = threading.Thread(
                target=run_loop, daemon=True, name="MonitoringAsyncLoop"
            )
            MonitoringFacade._async_loop_thread.start()
            
            # Wait for loop to be ready
            import time
            for _ in range(10):  # Wait up to 1 second
                if MonitoringFacade._async_loop is not None:
                    break
                time.sleep(0.1)
        
        return MonitoringFacade._async_loop
    
    @staticmethod
    def _get_async_writer(backend_name: str, backend: Any) -> Any:
        """Get or create async writer for a backend."""
        with MonitoringFacade._async_writers_lock:
            if backend_name not in MonitoringFacade._async_writers:
                from .optimizations.async_writer import AsyncBackendWriter
                
                async_writer = AsyncBackendWriter(
                    backend=backend,
                    queue_size=1000,
                    batch_size=10,
                    flush_interval=1.0
                )
                
                # Start async writer in background
                loop = MonitoringFacade._get_async_loop()
                if loop and not loop.is_closed():
                    asyncio.run_coroutine_threadsafe(async_writer.start(), loop)
                
                MonitoringFacade._async_writers[backend_name] = async_writer
            
            return MonitoringFacade._async_writers[backend_name]
    
    @staticmethod
    def _report_training_metrics_async(
        step: int, spmd_id: Optional[str], metrics: Dict[str, float]
    ) -> None:
        """
        Report training metrics to backends asynchronously (non-blocking, truly async).

        Args:
            step: Step number
            spmd_id: SPMD identifier
            metrics: Metrics dictionary
        """
        try:
            from .backends.tensorboard_backend import TensorBoardBackend
            from .backends.ray_backend import RayBackend
            from .backends.prometheus_backend import PrometheusBackend

            # Get log directory from environment variable (supports remote paths)
            # Examples:
            #   - Local: "./logs/tensorboard"
            #   - NFS: "/mnt/shared_logs/tensorboard"
            #   - S3 (via s3fs): "/mnt/s3_logs/tensorboard"
            log_dir = os.getenv("FLUXWAY_TENSORBOARD_LOG_DIR", "./logs/tensorboard")
            
            # Determine prefix
            prefix = f"spmd_{spmd_id}/training" if spmd_id else "task/training"

            # Create backends (lazy initialization)
            tensorboard = TensorBoardBackend(log_dir=log_dir)
            ray_backend = RayBackend()
            prometheus = PrometheusBackend(enable_server=False)

            # Get async writers and write asynchronously (non-blocking)
            loop = MonitoringFacade._get_async_loop()
            if loop and not loop.is_closed():
                # TensorBoard: truly async write (non-blocking)
                try:
                    async_writer = MonitoringFacade._get_async_writer("tensorboard", tensorboard)
                    # Submit async write task to background event loop (non-blocking)
                    asyncio.run_coroutine_threadsafe(
                        async_writer.write_async(step=step, metrics=metrics, prefix=prefix),
                        loop
                    )
                except Exception as e:
                    logger.debug(f"Failed to async write to TensorBoard: {e}")

                # Ray backend: async write (non-blocking)
                try:
                    async_writer = MonitoringFacade._get_async_writer("ray", ray_backend)
                    asyncio.run_coroutine_threadsafe(
                        async_writer.write_async(
                            step=step,
                            metrics=metrics,
                            prefix=prefix,
                            spmd_id=spmd_id
                        ),
                        loop
                    )
                except Exception:
                    pass

                # Prometheus: async write (non-blocking)
                try:
                    async_writer = MonitoringFacade._get_async_writer("prometheus", prometheus)
                    asyncio.run_coroutine_threadsafe(
                        async_writer.write_async(
                            step=step,
                            metrics=metrics,
                            prefix=prefix,
                            labels={"spmd_id": spmd_id} if spmd_id else None
                        ),
                        loop
                    )
                except Exception:
                    pass
            else:
                # Fallback to synchronous write if async loop unavailable
                logger.warning("Async loop unavailable, falling back to sync write")
                try:
                    tensorboard.write_metrics(step=step, metrics=metrics, prefix=prefix)
                    tensorboard.flush()
                except Exception as e:
                    logger.debug(f"Failed to write to TensorBoard: {e}")

        except Exception as e:
            # Non-intrusive: continue if reporting fails
            logger.debug(f"Failed to report training metrics: {e}")

    @staticmethod
    def record_checkpoint_event(
        event_type: str,
        checkpoint_id: str,
        component_type: str,
        component_id: str,
        status: str,
        message: str = "",
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a checkpoint-related event (non-blocking).

        Args:
            event_type: Type of checkpoint event (e.g., "save_start", "save_complete", "load_fail")
            checkpoint_id: Identifier of the checkpoint
            component_type: Type of component involved ("task", "spmd", "worker")
            component_id: Identifier of the component
            status: Status of the event ("success", "failure", "not_implemented")
            message: Optional descriptive message
            duration_seconds: Optional duration of the event
            metadata: Optional additional metadata

        This method is non-blocking and handles logging, metrics, and alerts internally.

        Example:
            MonitoringFacade.record_checkpoint_event(
                event_type="save_complete",
                checkpoint_id="checkpoint_100",
                component_type="spmd",
                component_id="spmd_1",
                status="success",
                duration_seconds=5.2
            )
        """
        try:
            import logging

            logger = logging.getLogger(__name__)

            # Log based on status
            log_msg = (
                f"Checkpoint {event_type}: checkpoint_id={checkpoint_id}, "
                f"component={component_type}:{component_id}, status={status}"
            )
            if message:
                log_msg += f", message={message}"
            if duration_seconds is not None:
                log_msg += f", duration={duration_seconds:.2f}s"

            if status == "failure":
                logger.error(log_msg)
            elif status == "not_implemented":
                logger.warning(log_msg)
            else:
                logger.info(log_msg)

            # Send to backends (non-blocking)
            try:
                from .backends.tensorboard_backend import TensorBoardBackend
                from .backends.ray_backend import RayBackend
                from .backends.prometheus_backend import PrometheusBackend
                from .backends.alert_backend import AlertBackend, AlertLevel

                tensorboard = TensorBoardBackend()
                ray_backend = RayBackend()
                prometheus = PrometheusBackend(enable_server=False)
                alert_backend = AlertBackend()

                # Record metrics
                checkpoint_metrics = {f"checkpoint/{event_type}_count": 1.0}
                if duration_seconds is not None:
                    checkpoint_metrics[f"checkpoint/{event_type}_duration_seconds"] = (
                        duration_seconds
                    )

                prefix = (
                    f"{component_type}_{component_id}/checkpoint"
                    if component_id
                    else "task/checkpoint"
                )

                try:
                    tensorboard.write_metrics(step=0, metrics=checkpoint_metrics, prefix=prefix)
                except Exception:
                    pass

                try:
                    ray_backend.record_checkpoint_event(
                        event_type=event_type,
                        checkpoint_id=checkpoint_id,
                        component_type=component_type,
                        component_id=component_id,
                        status=status,
                    )
                except Exception:
                    pass

                try:
                    prometheus.record_metrics(
                        checkpoint_metrics,
                        labels={
                            "event_type": event_type,
                            "component_type": component_type,
                            "component_id": component_id,
                            "status": status,
                        },
                    )
                except Exception:
                    pass

                # Send alert if failure
                if status == "failure":
                    try:
                        alert_backend.send_alert(
                            level=AlertLevel.ERROR,
                            title=f"Checkpoint {event_type} Failed",
                            message=message or log_msg,
                            metadata={
                                "checkpoint_id": checkpoint_id,
                                "component_type": component_type,
                                "component_id": component_id,
                                "event_type": event_type,
                                **(metadata or {}),
                            },
                        )
                    except Exception:
                        pass

            except Exception:
                # Non-intrusive: continue if backend reporting fails
                pass

        except Exception:
            # Non-intrusive: continue if monitoring fails
            pass

    @staticmethod
    def record_training_timing(step: int, event: str, spmd_id: Optional[str] = None) -> None:
        """
        Record a training timing event (non-blocking).

        Args:
            step: Step number
            event: Event name ("step_start", "forward_start", "forward_end",
                   "backward_start", "backward_end", "update_start", "update_end")
            spmd_id: SPMD identifier

        This is a lightweight method for recording timing events.
        Actual metrics are calculated and reported when record_training_progress is called.

        Example:
            MonitoringFacade.record_training_timing(step=100, event="forward_start", spmd_id="spmd_1")
            # ... forward pass ...
            MonitoringFacade.record_training_timing(step=100, event="forward_end", spmd_id="spmd_1")
        """
        try:
            collector = MonitoringFacade._get_training_collector(spmd_id)

            if event == "step_start":
                collector.start_step(step)
            elif event == "forward_start":
                collector.record_forward_start(step)
            elif event == "forward_end":
                collector.record_forward_end(step)
            elif event == "backward_start":
                collector.record_backward_start(step)
            elif event == "backward_end":
                collector.record_backward_end(step)
            elif event == "update_start":
                collector.record_update_start(step)
            elif event == "update_end":
                collector.record_update_end(step)

        except Exception:
            # Non-intrusive: continue if timing fails
            pass

    @staticmethod
    def record_gradient_metrics(
        step: int,
        gradient_norm: Optional[float] = None,
        gradient_mean: Optional[float] = None,
        gradient_std: Optional[float] = None,
        gradient_max: Optional[float] = None,
        gradient_min: Optional[float] = None,
        spmd_id: Optional[str] = None,
        check_anomalies: bool = True,
    ) -> None:
        """
        Record gradient metrics (non-blocking).

        Args:
            step: Step number
            gradient_norm: Gradient L2 norm
            gradient_mean: Gradient mean
            gradient_std: Gradient standard deviation
            gradient_max: Gradient maximum value
            gradient_min: Gradient minimum value
            spmd_id: SPMD identifier
            check_anomalies: Whether to check for gradient anomalies

        Example:
            MonitoringFacade.record_gradient_metrics(
                step=100,
                gradient_norm=1.5,
                gradient_mean=0.01,
                spmd_id="spmd_1"
            )
        """
        try:
            collector = MonitoringFacade._get_training_collector(spmd_id)

            # Create metrics
            metrics = TrainingMetrics(step=step)
            metrics.gradient_norm = gradient_norm
            metrics.gradient_mean = gradient_mean
            metrics.gradient_std = gradient_std
            metrics.gradient_max = gradient_max
            metrics.gradient_min = gradient_min

            # Check for anomalies
            if check_anomalies:
                if gradient_norm is not None:
                    # Gradient explosion threshold (configurable)
                    if gradient_norm > 100.0:
                        metrics.gradient_explosion = True
                    # Gradient vanishing threshold
                    elif gradient_norm < 1e-6:
                        metrics.gradient_vanishing = True

            # Add to collector
            collector.add_metrics(step, metrics)

        except Exception:
            # Non-intrusive: continue if gradient recording fails
            pass

    @staticmethod
    def record_communication_metrics(
        step: int,
        allreduce_time_ms: Optional[float] = None,
        sync_wait_time_ms: Optional[float] = None,
        staleness: Optional[int] = None,
        ulp_transfer_time_ms: Optional[float] = None,
        ulp_assembly_time_ms: Optional[float] = None,
        spmd_id: Optional[str] = None,
    ) -> None:
        """
        Record communication and synchronization metrics (non-blocking).

        Args:
            step: Step number
            allreduce_time_ms: AllReduce operation time in milliseconds
            sync_wait_time_ms: Synchronization wait time in milliseconds
            staleness: Data staleness (for async training)
            ulp_transfer_time_ms: ULP transfer time in milliseconds
            ulp_assembly_time_ms: ULP assembly time in milliseconds
            spmd_id: SPMD identifier

        Example:
            MonitoringFacade.record_communication_metrics(
                step=100,
                allreduce_time_ms=50.0,
                sync_wait_time_ms=10.0,
                spmd_id="spmd_1"
                )
        """
        try:
            collector = MonitoringFacade._get_training_collector(spmd_id)

            # Create metrics
            metrics = TrainingMetrics(step=step)
            metrics.allreduce_time_ms = allreduce_time_ms
            metrics.sync_wait_time_ms = sync_wait_time_ms
            metrics.staleness = staleness
            metrics.ulp_transfer_time_ms = ulp_transfer_time_ms
            metrics.ulp_assembly_time_ms = ulp_assembly_time_ms

            # Add to collector
            collector.add_metrics(step, metrics)

        except Exception:
            # Non-intrusive: continue if communication recording fails
            pass
