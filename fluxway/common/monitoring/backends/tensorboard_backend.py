"""
TensorBoard backend for metrics visualization.

Writes metrics to TensorBoard format for visualization.
"""

from typing import Dict, Optional
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    try:
        # Fallback to tensorboardX if torch is not available
        from tensorboardX import SummaryWriter

        HAS_TENSORBOARD = True
    except ImportError:
        HAS_TENSORBOARD = False
        logger.warning(
            "TensorBoard not available. Install with: pip install tensorboard tensorboardX (or torch)"
        )


class TensorBoardBackend:
    """
    TensorBoard backend for metrics visualization.

    Writes metrics to TensorBoard log directory for visualization.
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize TensorBoard backend.

        Args:
            log_dir: Directory to write TensorBoard logs.
                    If None, uses environment variable FLUXWAY_TENSORBOARD_LOG_DIR
                    or defaults to "./logs/tensorboard"
                    
        Examples:
            # Local directory
            TensorBoardBackend(log_dir="./logs/tensorboard")
            
            # Remote directory (NFS)
            TensorBoardBackend(log_dir="/mnt/shared_logs/tensorboard")
            
            # Remote directory (S3 via s3fs)
            TensorBoardBackend(log_dir="/mnt/s3_logs/tensorboard")
            
            # Via environment variable
            import os
            os.environ["FLUXWAY_TENSORBOARD_LOG_DIR"] = "/mnt/shared_logs/tensorboard"
            TensorBoardBackend()  # Uses environment variable
        """
        if log_dir is None:
            log_dir = os.getenv("FLUXWAY_TENSORBOARD_LOG_DIR", "./logs/tensorboard")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Optional[SummaryWriter] = None

        if HAS_TENSORBOARD:
            try:
                self._writer = SummaryWriter(str(self.log_dir))
                logger.info(f"TensorBoard backend initialized: {self.log_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize TensorBoard: {e}")
                self._writer = None
        else:
            logger.warning("TensorBoard not available, metrics will not be written")

    def write_metrics(self, step: int, metrics: Dict[str, float], prefix: str = "") -> None:
        """
        Write metrics to TensorBoard.

        Args:
            step: Step number (e.g., training step)
            metrics: Dictionary of metric name to value
            prefix: Prefix for metric names (e.g., "spmd_1/", "task/")
        """
        if self._writer is None:
            return

        try:
            for metric_name, value in metrics.items():
                full_name = f"{prefix}{metric_name}" if prefix else metric_name
                self._writer.add_scalar(full_name, value, step)
        except Exception as e:
            logger.error(f"Failed to write metrics to TensorBoard: {e}")

    def write_worker_metrics(
        self, step: int, spmd_id: str, worker_id: str, metrics: Dict[str, float]
    ) -> None:
        """
        Write worker-level metrics to TensorBoard.

        Args:
            step: Step number
            spmd_id: SPMD identifier
            worker_id: Worker identifier
            metrics: Worker metrics dictionary
        """
        prefix = f"spmd_{spmd_id}/worker_{worker_id}/"
        self.write_metrics(step, metrics, prefix)

    def write_spmd_metrics(self, step: int, spmd_id: str, metrics: Dict[str, float]) -> None:
        """
        Write SPMD-level metrics to TensorBoard.

        Args:
            step: Step number
            spmd_id: SPMD identifier
            metrics: SPMD aggregated metrics dictionary
        """
        prefix = f"spmd_{spmd_id}/"
        self.write_metrics(step, metrics, prefix)

    def write_task_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Write task-level metrics to TensorBoard.

        Args:
            step: Step number
            metrics: Task aggregated metrics dictionary
        """
        prefix = "task/"
        self.write_metrics(step, metrics, prefix)

    def flush(self) -> None:
        """Flush TensorBoard writer."""
        if self._writer is not None:
            try:
                self._writer.flush()
            except Exception:
                pass

    def close(self) -> None:
        """Close TensorBoard writer."""
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
            self._writer = None
