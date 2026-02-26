"""
JSON datasource implementation.

Supports reading JSON files in various formats:
- JSON Lines (JSONL): One JSON object per line
- JSON Array: File contains a JSON array of objects
- Single JSON Object: File contains a single JSON object

Reference: Ray Data's json_datasource.py implementation.
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional

from .base import ReadTask
from .file_base import FileBaseSource, FileReadTask

logger = logging.getLogger(__name__)


class JsonDatasource(FileBaseSource):
    """
    Datasource for reading JSON files.

    Supports multiple JSON formats:
    - JSON Lines (JSONL): Each line is a valid JSON object
    - JSON Array: The entire file is a JSON array
    - Single Object: The file contains a single JSON object

    Example:
        # Create datasource
        source = JsonDatasource(src_dir="/data/input")

        # Get tasks for parallel reading
        tasks = source.get_read_tasks(parallelism=4)

        # Read data in batches
        for task in tasks:
            for batch in source.read_batch(task, batch_size=100):
                process(batch)
    """

    def __init__(
        self,
        src_dir: str,
        file_pattern: str = "*.json",
        encoding: str = "utf-8",
        recursive: bool = True,
        tensor_type: Optional[Any] = None,
    ):
        """
        Initialize JSON datasource.

        Args:
            src_dir: Directory containing JSON files
            file_pattern: Glob pattern for matching files
            encoding: File encoding (default: utf-8)
            recursive: Whether to search subdirectories recursively
            tensor_type: Target type for converting the entire batch (e.g., np.ndarray)
        """
        super().__init__(
            src_dir=src_dir,
            file_pattern=file_pattern,
            file_extensions=[".json", ".jsonl"],
            recursive=recursive,
        )
        self.encoding = encoding
        # tensor_type is used to convert the entire batch (List[Dict]) to target type
        self.tensor_type = tensor_type
        logger.debug(f"Initialized JsonDatasource with tensor_type: {tensor_type}")

    def _read_json_lines(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Read JSON Lines format file.

        Each line should be a valid JSON object.
        Empty lines and lines that fail to parse are skipped with a warning.

        Args:
            file_path: Path to the JSON Lines file

        Yields:
            Parsed JSON objects (dicts)
        """
        with open(file_path, "r", encoding=self.encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    yield obj
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse JSON at {file_path}:{line_num}: {e}"
                    )

    def _read_json_array(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Read JSON array format file.

        The entire file should be a valid JSON array.

        Args:
            file_path: Path to the JSON array file

        Yields:
            Items from the JSON array
        """
        with open(file_path, "r", encoding=self.encoding) as f:
            try:
                data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        yield item
                else:
                    # Single object, yield as-is
                    yield data

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON file {file_path}: {e}")

    def _detect_format_and_read(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Auto-detect JSON format and read accordingly.

        Detection logic:
        - If file starts with '[', treat as JSON array
        - Otherwise, treat as JSON Lines

        Args:
            file_path: Path to the JSON file

        Yields:
            Parsed JSON objects
        """
        try:
            with open(file_path, "r", encoding=self.encoding) as f:
                # Read first non-whitespace character
                first_char = None
                while True:
                    char = f.read(1)
                    if not char:
                        return  # Empty file
                    if not char.isspace():
                        first_char = char
                        break

            if first_char == "[":
                # JSON Array format
                yield from self._read_json_array(file_path)
            else:
                # JSON Lines format (also handles single objects on one line)
                yield from self._read_json_lines(file_path)

        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")

    def read_batch(self, task: ReadTask, batch_size: int) -> Iterator[Any]:
        """
        Read JSON data in batches.

        Reads from all files in the task and yields batches of records.
        If tensor_type is configured, converts the entire batch (List[Dict]) to the target type.
        Otherwise, yields batches as list of dicts.

        Args:
            task: File read task containing list of files
            batch_size: Maximum number of records per batch

        Yields:
            If tensor_type is set: converted tensor data (e.g., np.ndarray)
            Otherwise: List of dicts (JSON objects), one batch at a time
        """
        if not isinstance(task, FileReadTask):
            logger.error(f"Expected FileReadTask, got {type(task)}")
            return

        batch: List[Dict[str, Any]] = []
        total_records = 0

        for file_path in task.files:
            logger.debug(f"Reading JSON file: {file_path}")

            try:
                for record in self._detect_format_and_read(file_path):
                    batch.append(record)
                    total_records += 1

                    if len(batch) >= batch_size:
                        yield self._convert_batch(batch)
                        batch = []

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue

        # Yield remaining records
        if batch:
            yield self._convert_batch(batch)

        logger.debug(
            f"Task {task.task_id} completed: "
            f"read {total_records} records from {len(task.files)} files"
        )

    def _convert_batch(self, batch: List[Dict[str, Any]]) -> Any:
        """
        Convert a batch of records to the target tensor type.

        The entire batch (List[Dict]) is converted to tensor_type.
        tensor_name is not used in conversion, it's only for worker to send data.

        Args:
            batch: List of dicts (JSON records)

        Returns:
            Converted batch data (e.g., np.ndarray) or original batch if no conversion needed
        """
        # If no tensor_type configured, return batch as-is
        if not self.tensor_type:
            return batch

        # If tensor_type is list, return batch as-is (no conversion)
        if self.tensor_type is list:
            return batch

        # Convert based on tensor_type
        import numpy as np

        if self.tensor_type is np.ndarray:
            # Convert entire List[Dict] to np.ndarray
            # This converts the list of dicts directly to a numpy array
            return np.array(batch)
        
        # Add JAX support
        try:
            import jax
            import jax.numpy as jnp
            if self.tensor_type is jax.Array or self.tensor_type is getattr(jax, "Array", None):
                # Convert to jax array (via numpy for safety)
                return jnp.array(np.array(batch))
        except ImportError:
            pass

        raise NotImplementedError(
            f"Tensor type conversion for {self.tensor_type} is not implemented. "
            f"Only np.ndarray, jax.Array and list are currently supported."
        )

    def read_all(self, task: ReadTask) -> List[Dict[str, Any]]:
        """
        Read all records from a task into a single list.

        WARNING: This loads all data into memory. Only use for small datasets.

        Args:
            task: File read task

        Returns:
            List of all records
        """
        all_records = []
        for batch in self.read_batch(task, batch_size=10000):
            all_records.extend(batch)
        return all_records


def create_datasource(
    data_type: str,
    src_dir: str,
    file_pattern: Optional[str] = None,
    encoding: str = "utf-8",
    recursive: bool = True,
    tensor_type: Optional[Any] = None,
    **kwargs: Any,
) -> FileBaseSource:
    """
    Factory function to create a datasource based on data type.

    Args:
        data_type: Type of data source ("json", "parquet", "csv", etc.)
        src_dir: Source directory containing data files
        file_pattern: Optional file pattern (defaults based on data_type)
        encoding: File encoding for text formats
        recursive: Whether to search recursively
        **kwargs: Additional arguments passed to the datasource constructor

    Returns:
        Appropriate datasource instance

    Raises:
        ValueError: If data_type is not supported

    Example:
        source = create_datasource("json", "/data/input")
        tasks = source.get_read_tasks(parallelism=4)
    """
    data_type = data_type.lower()

    if data_type == "json":
        pattern = file_pattern or "*.json"
        return JsonDatasource(
            src_dir=src_dir,
            file_pattern=pattern,
            encoding=encoding,
            recursive=recursive,
            tensor_type=tensor_type,
        )

    # TODO: Add support for other formats
    # elif data_type == "parquet":
    #     pattern = file_pattern or "*.parquet"
    #     return ParquetDatasource(src_dir=src_dir, file_pattern=pattern, **kwargs)
    # elif data_type == "csv":
    #     pattern = file_pattern or "*.csv"
    #     return CsvDatasource(src_dir=src_dir, file_pattern=pattern, **kwargs)

    else:
        raise ValueError(f"Unsupported data type: {data_type}. Supported types: json")
