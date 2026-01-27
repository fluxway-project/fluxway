"""
System Metrics Collector for cross-platform resource monitoring.

Based on MONITOR_IMPLEMENTATION.md

Provides unified interface for collecting:
- CPU metrics (per-process)
- Memory metrics (per-process)
- GPU metrics (optional, Linux only)
- Network I/O (per-process)
- Disk I/O (per-process)

Design principles:
- Cross-platform compatible (macOS and Linux)
- Gracefully handles missing GPU support
- Minimal dependencies (psutil, optional GPU libraries)
"""

import platform
import time
from typing import Dict, List, Optional
import psutil

# Conditional GPU import
try:
    import pynvml

    HAS_GPU_LIB = True
except ImportError:
    HAS_GPU_LIB = False

try:
    import gpustat

    HAS_GPUSTAT = True
except ImportError:
    HAS_GPUSTAT = False


class SystemMetricsCollector:
    """
    System metrics collector with cross-platform support.

    Handles:
    - macOS: CPU, memory, network, disk (no GPU)
    - Linux: All metrics including GPU (if available)
    """

    def __init__(self):
        self.system = platform.system()
        self._gpu_initialized = False

        # Initialize GPU on Linux if available
        if self._is_linux_with_gpu():
            self._init_gpu()

    def _is_linux_with_gpu(self) -> bool:
        """Check if running on Linux with GPU support."""
        if self.system != "Linux":
            return False
        if not HAS_GPU_LIB:
            return False
        return True

    def _init_gpu(self) -> bool:
        """Initialize GPU monitoring (Linux only)."""
        if not self._is_linux_with_gpu():
            return False

        try:
            pynvml.nvmlInit()
            self._gpu_initialized = True
            return True
        except Exception:
            # GPU driver not available
            return False

    def collect_cpu_metrics(self, pid: int) -> Dict[str, float]:
        """
        Collect CPU metrics for a process.

        Args:
            pid: Process ID

        Returns:
            Dictionary with CPU metrics:
            - cpu_percent: CPU usage percentage
            - cpu_count: Number of CPU cores
            - cpu_times_user: User CPU time
            - cpu_times_system: System CPU time
        """
        try:
            process = psutil.Process(pid)
            cpu_percent = process.cpu_percent(interval=0.1)
            cpu_times = process.cpu_times()

            return {
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "cpu_times_user": cpu_times.user,
                "cpu_times_system": cpu_times.system,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {
                "cpu_percent": 0.0,
                "cpu_count": psutil.cpu_count(),
                "cpu_times_user": 0.0,
                "cpu_times_system": 0.0,
            }

    def collect_memory_metrics(self, pid: int) -> Dict[str, float]:
        """
        Collect memory metrics for a process.

        Args:
            pid: Process ID

        Returns:
            Dictionary with memory metrics:
            - memory_rss_mb: Physical memory (RSS) in MB
            - memory_vms_mb: Virtual memory (VMS) in MB
            - memory_percent: Memory usage percentage
        """
        try:
            process = psutil.Process(pid)
            mem_info = process.memory_info()
            mem_percent = process.memory_percent()

            return {
                "memory_rss_mb": mem_info.rss / 1024 / 1024,  # Bytes to MB
                "memory_vms_mb": mem_info.vms / 1024 / 1024,
                "memory_percent": mem_percent,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {
                "memory_rss_mb": 0.0,
                "memory_vms_mb": 0.0,
                "memory_percent": 0.0,
            }

    def collect_gpu_metrics(self, gpu_ids: List[int]) -> Dict[int, Dict[str, float]]:
        """
        Collect GPU metrics for specified GPU IDs.

        Args:
            gpu_ids: List of GPU IDs to monitor

        Returns:
            Dictionary mapping GPU ID to metrics:
            - utilization_percent: GPU utilization
            - memory_used_mb: Used GPU memory in MB
            - memory_total_mb: Total GPU memory in MB
            - temperature_c: GPU temperature in Celsius

        Note:
            Returns empty dict on macOS or when GPU not available.
        """
        if not self._is_linux_with_gpu() or not self._gpu_initialized:
            return {}

        gpu_metrics = {}

        try:
            for gpu_id in gpu_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temp = 0.0

                gpu_metrics[gpu_id] = {
                    "utilization_percent": util.gpu,
                    "memory_used_mb": mem_info.used / 1024 / 1024,
                    "memory_total_mb": mem_info.total / 1024 / 1024,
                    "temperature_c": temp,
                }
        except Exception:
            # GPU monitoring failed, return empty
            return {}

        return gpu_metrics

    def collect_network_io(self, pid: int) -> Dict[str, int]:
        """
        Collect network I/O metrics for a process.

        Args:
            pid: Process ID

        Returns:
            Dictionary with network I/O:
            - bytes_sent: Bytes sent
            - bytes_recv: Bytes received
        """
        try:
            process = psutil.Process(pid)
            io_counters = process.io_counters()

            return {
                "bytes_sent": io_counters.write_bytes if hasattr(io_counters, "write_bytes") else 0,
                "bytes_recv": io_counters.read_bytes if hasattr(io_counters, "read_bytes") else 0,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            # Some systems don't support per-process network IO
            return {
                "bytes_sent": 0,
                "bytes_recv": 0,
            }

    def collect_disk_io(self, pid: int) -> Dict[str, int]:
        """
        Collect disk I/O metrics for a process.

        Args:
            pid: Process ID

        Returns:
            Dictionary with disk I/O:
            - read_bytes: Bytes read
            - write_bytes: Bytes written
        """
        try:
            process = psutil.Process(pid)
            io_counters = process.io_counters()

            return {
                "read_bytes": io_counters.read_bytes,
                "write_bytes": io_counters.write_bytes,
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            return {
                "read_bytes": 0,
                "write_bytes": 0,
            }

    def collect_all_metrics(self, pid: int, gpu_ids: Optional[List[int]] = None) -> Dict:
        """
        Collect all available metrics for a process.

        Args:
            pid: Process ID
            gpu_ids: Optional list of GPU IDs to monitor

        Returns:
            Dictionary with all collected metrics
        """
        metrics = {
            "pid": pid,
            "timestamp": time.time(),
        }

        # Collect all metrics
        metrics.update(self.collect_cpu_metrics(pid))
        metrics.update(self.collect_memory_metrics(pid))
        metrics.update(self.collect_network_io(pid))
        metrics.update(self.collect_disk_io(pid))

        # GPU metrics (optional)
        if gpu_ids is not None:
            gpu_metrics = self.collect_gpu_metrics(gpu_ids)
            metrics["gpu_metrics"] = gpu_metrics

        return metrics
