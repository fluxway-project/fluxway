"""
Batch transfer manager for metrics.

Batches multiple metrics together and optionally compresses them
to reduce network overhead.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from collections import deque
import json

logger = logging.getLogger(__name__)

try:
    import gzip

    HAS_GZIP = True
except ImportError:
    HAS_GZIP = False


class BatchTransferManager:
    """
    Batch transfer manager.

    Batches multiple metrics together and optionally compresses them
    for efficient network transfer.
    """

    def __init__(
        self,
        batch_size: int = 10,
        batch_timeout: float = 5.0,
        enable_compression: bool = True,
        compression_threshold: int = 1024,  # Compress if > 1KB
    ):
        """
        Initialize batch transfer manager.

        Args:
            batch_size: Number of metrics to batch together (default: 10)
            batch_timeout: Maximum time to wait before sending batch (seconds, default: 5.0)
            enable_compression: Whether to compress batches (default: True)
            compression_threshold: Minimum size to compress (bytes, default: 1024)
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.enable_compression = enable_compression and HAS_GZIP
        self.compression_threshold = compression_threshold

        self._batch_buffer: deque = deque(maxlen=batch_size)
        self._batch_start_time: Optional[float] = None

    def add_metric(
        self, metric_type: str, metric_data: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Add a metric to the batch.

        Args:
            metric_type: Type of metric (e.g., "worker", "spmd", "task")
            metric_data: Metric data dictionary

        Returns:
            Batch of metrics if batch is ready, None otherwise
        """
        # Initialize batch start time
        if self._batch_start_time is None:
            self._batch_start_time = time.time()

        # Add to batch
        metric_item = {"type": metric_type, "data": metric_data, "timestamp": time.time()}
        self._batch_buffer.append(metric_item)

        # Check if batch is ready
        if self._should_send_batch():
            return self._flush_batch()

        return None

    def _should_send_batch(self) -> bool:
        """Check if batch should be sent."""
        # Check batch size
        if len(self._batch_buffer) >= self.batch_size:
            return True

        # Check timeout
        if self._batch_start_time is not None:
            elapsed = time.time() - self._batch_start_time
            if elapsed >= self.batch_timeout:
                return True

        return False

    def _flush_batch(self) -> List[Dict[str, Any]]:
        """
        Flush batch buffer.

        Returns:
            List of batched metrics
        """
        if not self._batch_buffer:
            return []

        batch = list(self._batch_buffer)
        self._batch_buffer.clear()
        self._batch_start_time = None

        return batch

    def flush(self) -> List[Dict[str, Any]]:
        """
        Force flush current batch.

        Returns:
            List of batched metrics
        """
        return self._flush_batch()

    def serialize_batch(
        self, batch: List[Dict[str, Any]], compress: Optional[bool] = None
    ) -> bytes:
        """
        Serialize batch to bytes, optionally with compression.

        Args:
            batch: List of metric dictionaries
            compress: Whether to compress (if None, uses enable_compression)

        Returns:
            Serialized batch as bytes
        """
        # Serialize to JSON
        json_str = json.dumps(batch)
        json_bytes = json_str.encode("utf-8")

        # Compress if needed
        should_compress = compress if compress is not None else self.enable_compression
        if should_compress and len(json_bytes) > self.compression_threshold:
            if HAS_GZIP:
                try:
                    compressed = gzip.compress(json_bytes)
                    logger.debug(
                        f"Compressed batch: {len(json_bytes)} -> {len(compressed)} bytes "
                        f"({len(compressed) / len(json_bytes) * 100:.1f}%)"
                    )
                    return compressed
                except Exception as e:
                    logger.warning(f"Compression failed: {e}, using uncompressed")
            else:
                logger.warning("gzip not available, using uncompressed")

        return json_bytes

    def deserialize_batch(self, data: bytes) -> List[Dict[str, Any]]:
        """
        Deserialize batch from bytes.

        Args:
            data: Serialized batch data

        Returns:
            List of metric dictionaries
        """
        # Try decompression
        try:
            if HAS_GZIP:
                decompressed = gzip.decompress(data)
                json_str = decompressed.decode("utf-8")
            else:
                json_str = data.decode("utf-8")
        except Exception:
            # Not compressed, try direct decode
            try:
                json_str = data.decode("utf-8")
            except Exception as e:
                logger.error(f"Failed to deserialize batch: {e}")
                return []

        # Parse JSON
        try:
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to parse batch JSON: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get transfer statistics."""
        return {
            "batch_size": self.batch_size,
            "current_batch_size": len(self._batch_buffer),
            "batch_timeout": self.batch_timeout,
            "enable_compression": self.enable_compression,
            "compression_threshold": self.compression_threshold,
        }
