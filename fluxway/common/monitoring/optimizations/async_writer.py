"""
Asynchronous backend writer for non-blocking metric writes.

Uses async queues to buffer writes and prevent blocking the main execution flow.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, Callable
from collections import deque
import time

logger = logging.getLogger(__name__)


class AsyncBackendWriter:
    """
    Asynchronous backend writer.

    Buffers writes in a queue and processes them asynchronously in the background.
    Prevents blocking the main execution flow.
    """

    def __init__(
        self,
        backend: Any,
        queue_size: int = 1000,
        batch_size: int = 10,
        flush_interval: float = 1.0,
    ):
        """
        Initialize async backend writer.

        Args:
            backend: Backend instance (TensorBoardBackend, PrometheusBackend, etc.)
            queue_size: Maximum queue size (default: 1000)
            batch_size: Number of metrics to batch together (default: 10)
            flush_interval: Interval between flushes in seconds (default: 1.0)
        """
        self.backend = backend
        self.queue_size = queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._writer_task: Optional[asyncio.Task] = None
        self._running = False
        self._batch_buffer: deque = deque(maxlen=batch_size)
        self._last_flush_time = time.time()

    async def write_async(
        self, step: int, metrics: Dict[str, float], prefix: str = "", **kwargs
    ) -> bool:
        """
        Asynchronously write metrics (non-blocking).

        Args:
            step: Step number
            metrics: Dictionary of metric name to value
            prefix: Prefix for metric names
            **kwargs: Additional arguments for backend write methods

        Returns:
            True if queued successfully, False if queue is full
        """
        if not self._running:
            await self.start()

        try:
            write_item = {
                "step": step,
                "metrics": metrics,
                "prefix": prefix,
                "kwargs": kwargs,
                "timestamp": time.time(),
            }
            self._queue.put_nowait(write_item)
            return True
        except asyncio.QueueFull:
            logger.warning("Async writer queue is full, dropping oldest item")
            try:
                # Drop oldest item
                self._queue.get_nowait()
                self._queue.put_nowait(write_item)
                return True
            except asyncio.QueueEmpty:
                return False

    async def start(self) -> None:
        """Start the async writer task."""
        if self._running:
            return

        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop())
        logger.info("Async backend writer started")

    async def stop(self) -> None:
        """Stop the async writer task and flush remaining items."""
        if not self._running:
            return

        self._running = False

        # Flush remaining items
        await self.flush()

        # Wait for writer task to finish
        if self._writer_task:
            try:
                await asyncio.wait_for(self._writer_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Async writer task did not finish in time")
            self._writer_task = None

        logger.info("Async backend writer stopped")

    async def flush(self) -> None:
        """Flush all buffered metrics."""
        # Process all items in queue
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                await self._write_item(item)
            except asyncio.QueueEmpty:
                break

        # Flush batch buffer
        if self._batch_buffer:
            await self._flush_batch()

        # Flush backend
        if hasattr(self.backend, "flush"):
            try:
                self.backend.flush()
            except Exception as e:
                logger.error(f"Failed to flush backend: {e}")

    async def _writer_loop(self) -> None:
        """Background writer loop."""
        while self._running:
            try:
                # Wait for item with timeout
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=self.flush_interval)
                    self._batch_buffer.append(item)
                except asyncio.TimeoutError:
                    # Timeout - flush batch if needed
                    pass

                # Check if should flush batch
                current_time = time.time()
                should_flush = (
                    len(self._batch_buffer) >= self.batch_size
                    or (current_time - self._last_flush_time) >= self.flush_interval
                )

                if should_flush and self._batch_buffer:
                    await self._flush_batch()
                    self._last_flush_time = current_time

            except Exception as e:
                logger.error(f"Error in async writer loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _flush_batch(self) -> None:
        """Flush batch buffer to backend."""
        if not self._batch_buffer:
            return

        try:
            # Write all items in batch
            for item in self._batch_buffer:
                await self._write_item(item)

            self._batch_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")

    async def _write_item(self, item: Dict[str, Any]) -> None:
        """Write a single item to backend."""
        try:
            step = item["step"]
            metrics = item["metrics"]
            prefix = item["prefix"]
            kwargs = item.get("kwargs", {})

            # Run backend write in thread pool to avoid blocking async loop
            # (SummaryWriter.add_scalar is synchronous I/O)
            import concurrent.futures
            loop = asyncio.get_event_loop()
            
            def sync_write():
                """Synchronous write function to run in thread pool."""
                try:
                    # Determine write method based on prefix
                    if prefix.startswith("spmd_"):
                        # SPMD-level metrics
                        spmd_id = prefix.replace("spmd_", "").split("/")[0]
                        if hasattr(self.backend, "write_spmd_metrics"):
                            self.backend.write_spmd_metrics(step, spmd_id, metrics)
                        elif hasattr(self.backend, "record_spmd_metrics"):
                            self.backend.record_spmd_metrics(spmd_id, metrics)
                        elif hasattr(self.backend, "record_training_metrics"):
                            self.backend.record_training_metrics(step, spmd_id, metrics)
                    elif prefix.startswith("task/"):
                        # Task-level metrics
                        if hasattr(self.backend, "write_task_metrics"):
                            self.backend.write_task_metrics(step, metrics)
                        elif hasattr(self.backend, "record_task_metrics"):
                            self.backend.record_task_metrics(metrics)
                    else:
                        # Generic write
                        if hasattr(self.backend, "write_metrics"):
                            self.backend.write_metrics(step, metrics, prefix, **kwargs)
                        elif hasattr(self.backend, "record_metrics"):
                            self.backend.record_metrics(metrics, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to write item to backend: {e}")
            
            # Run in thread pool to avoid blocking
            await loop.run_in_executor(None, sync_write)

        except Exception as e:
            logger.error(f"Failed to write item to backend: {e}")
