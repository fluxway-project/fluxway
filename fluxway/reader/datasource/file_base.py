"""
File-based datasource base class.

Handles common file operations like discovery, partitioning, and size estimation.
"""

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .base import BaseSource, ReadTask

logger = logging.getLogger(__name__)


@dataclass
class FileReadTask(ReadTask):
    """
    Read task for file-based datasources.
    
    Attributes:
        task_id: Unique identifier for the task
        metadata: Additional metadata for the task
        files: List of file paths assigned to this task
        total_size: Total size of all files in bytes
    """
    files: List[str] = field(default_factory=list)
    total_size: int = 0


class FileBaseSource(BaseSource):
    """
    Base class for file-based datasources.
    
    Handles common operations for reading from files:
    - File discovery using glob patterns
    - File partitioning for parallel processing
    - Size estimation based on file sizes
    
    Subclasses should implement the actual file reading logic in read_batch().
    
    Example:
        class MyFileSource(FileBaseSource):
            def __init__(self, src_dir: str):
                super().__init__(src_dir, file_extensions=[".myext"])
            
            def read_batch(self, task: ReadTask, batch_size: int) -> Iterator[Any]:
                for file_path in task.files:
                    # Read and yield batches from file
                    pass
    """
    
    def __init__(
        self,
        src_dir: str,
        file_pattern: str = "*",
        file_extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ):
        """
        Initialize file-based datasource.
        
        Args:
            src_dir: Root directory to search for files
            file_pattern: Glob pattern for matching files (e.g., "*.json")
            file_extensions: List of allowed file extensions (e.g., [".json", ".jsonl"])
            recursive: Whether to search recursively in subdirectories
        """
        self.src_dir = src_dir
        self.file_pattern = file_pattern
        self.file_extensions = file_extensions or []
        self.recursive = recursive
        
        # Cache for discovered files
        self._discovered_files: Optional[List[str]] = None
    
    def _discover_files(self) -> List[str]:
        """
        Discover all files matching the pattern and extensions.
        
        Returns:
            Sorted list of file paths
        """
        if self._discovered_files is not None:
            return self._discovered_files
        
        # Build glob pattern
        if self.recursive:
            pattern = os.path.join(self.src_dir, "**", self.file_pattern)
        else:
            pattern = os.path.join(self.src_dir, self.file_pattern)
        
        # Find all matching files
        all_files = glob.glob(pattern, recursive=self.recursive)
        
        # Filter by extensions if specified
        if self.file_extensions:
            all_files = [
                f for f in all_files
                if any(f.endswith(ext) for ext in self.file_extensions)
            ]
        
        # Only keep actual files (not directories)
        all_files = [f for f in all_files if os.path.isfile(f)]
        
        # Sort for deterministic ordering
        all_files = sorted(all_files)
        
        self._discovered_files = all_files
        logger.info(f"Discovered {len(all_files)} files in {self.src_dir}")
        
        return all_files
    
    def _partition_files(
        self,
        files: List[str],
        parallelism: int,
    ) -> List[List[str]]:
        """
        Partition files across workers using load-balanced distribution.
        
        Uses a greedy algorithm based on file sizes to balance the load:
        - Sort files by size (descending)
        - Assign each file to the partition with the smallest current total size
        
        Args:
            files: List of file paths to partition
            parallelism: Number of partitions (workers)
            
        Returns:
            List of file lists, one per partition
        """
        if not files:
            return [[] for _ in range(parallelism)]
        
        # Get file sizes
        file_sizes = []
        for f in files:
            try:
                size = os.path.getsize(f)
                file_sizes.append((f, size))
            except OSError as e:
                logger.warning(f"Could not get size of {f}: {e}")
                file_sizes.append((f, 0))
        
        # Sort by size (descending) for better load balancing
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize partitions
        partitions: List[List[str]] = [[] for _ in range(parallelism)]
        partition_sizes = [0] * parallelism
        
        # Greedy assignment: assign each file to the smallest partition
        for file_path, size in file_sizes:
            # Find partition with minimum current size
            min_idx = partition_sizes.index(min(partition_sizes))
            partitions[min_idx].append(file_path)
            partition_sizes[min_idx] += size
        
        return partitions
    
    def get_read_tasks(self, parallelism: int) -> List[FileReadTask]:
        """
        Get file read tasks distributed across workers.
        
        Args:
            parallelism: Number of parallel workers
            
        Returns:
            List of FileReadTask objects
        """
        files = self._discover_files()
        partitions = self._partition_files(files, parallelism)
        
        tasks = []
        for i, file_list in enumerate(partitions):
            # Calculate total size for this partition
            total_size = 0
            for f in file_list:
                try:
                    total_size += os.path.getsize(f)
                except OSError:
                    pass
            
            task = FileReadTask(
                task_id=f"file_task_{i}",
                metadata={"partition_id": i, "file_count": len(file_list)},
                files=file_list,
                total_size=total_size,
            )
            tasks.append(task)
        
        logger.info(
            f"Created {len(tasks)} file read tasks "
            f"(total {len(files)} files, {sum(t.total_size for t in tasks)} bytes)"
        )
        
        return tasks
    
    def estimate_inmemory_size(self, task: ReadTask) -> int:
        """
        Estimate in-memory size based on file size.
        
        Uses a simple heuristic: file_size * 2 (assuming decompression/parsing overhead).
        
        Args:
            task: The read task
            
        Returns:
            Estimated size in bytes
        """
        if isinstance(task, FileReadTask):
            # Heuristic: in-memory size is typically 2x file size
            return task.total_size * 2
        return 0
    
    def read_batch(self, task: ReadTask, batch_size: int) -> Iterator[Any]:
        """
        Read batches from files.
        
        This method should be overridden by subclasses to implement
        format-specific reading logic.
        
        Args:
            task: The read task containing file list
            batch_size: Target batch size
            
        Yields:
            Batches of data
        """
        raise NotImplementedError("Subclasses must implement read_batch()")
    
    def get_file_count(self) -> int:
        """Get the total number of discovered files."""
        return len(self._discover_files())
    
    def get_total_size(self) -> int:
        """Get the total size of all discovered files in bytes."""
        files = self._discover_files()
        total = 0
        for f in files:
            try:
                total += os.path.getsize(f)
            except OSError:
                pass
        return total
