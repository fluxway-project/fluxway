"""
Base datasource abstract class.

All datasources must inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class ReadTask:
    """
    Read task that can be assigned to a worker.

    Attributes:
        task_id: Unique identifier for the task
        metadata: Additional metadata for the task
    """

    task_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSource(ABC):
    """
    Base class for all datasources.

    Defines the interface for reading data from various sources.
    Subclasses must implement all abstract methods.

    Example:
        class MyDatasource(BaseSource):
            def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
                # Return list of tasks for workers
                pass

            def read_batch(self, task: ReadTask, batch_size: int) -> Iterator[Any]:
                # Yield batches of data
                pass
    """

    @abstractmethod
    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        """
        Get a list of read tasks that can be distributed to workers.

        This method should partition the data source into multiple tasks
        that can be processed in parallel by different workers.

        Args:
            parallelism: Number of parallel workers (determines number of tasks)

        Returns:
            List of ReadTask objects, one for each worker
        """
        pass

    @abstractmethod
    def read_batch(self, task: ReadTask, batch_size: int) -> Iterator[Any]:
        """
        Read data in batches from a task.

        This is a generator that yields batches of data from the given task.
        Each batch should contain approximately batch_size records.

        Args:
            task: The read task to process
            batch_size: Target number of records per batch

        Yields:
            Batches of data (format depends on the datasource implementation)
        """
        pass

    @abstractmethod
    def estimate_inmemory_size(self, task: ReadTask) -> int:
        """
        Estimate the in-memory size of a task's data in bytes.

        This is used for memory planning and backpressure decisions.

        Args:
            task: The read task to estimate

        Returns:
            Estimated size in bytes
        """
        pass

    def get_schema(self) -> Optional[Dict[str, Any]]:
        """
        Get the schema of the data source (if available).

        Returns:
            Schema dict or None if schema is not available
        """
        return None

    def supports_pushdown(self) -> bool:
        """
        Check if the datasource supports filter/projection pushdown.

        Returns:
            True if pushdown is supported, False otherwise
        """
        return False
