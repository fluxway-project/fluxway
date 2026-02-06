"""
Fluxway Reader module for data reading.

This module provides data reading capabilities for Fluxway framework,
including various datasource implementations.
"""

# Ensure INFO logs are visible even when importing this subpackage directly.
from ..common.utils.logging import ensure_logging_configured  # noqa: E402

ensure_logging_configured()

from .datasource import (
    BaseSource,
    FileBaseSource,
    JsonDatasource,
    ReadTask,
    FileReadTask,
    create_datasource,
)

__all__ = [
    "BaseSource",
    "FileBaseSource",
    "JsonDatasource",
    "ReadTask",
    "FileReadTask",
    "create_datasource",
]
