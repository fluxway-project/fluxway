"""Common utilities and types for Fluxway framework."""

# Ensure INFO logs are visible even when importing subpackages directly.
from .utils.logging import ensure_logging_configured

__all__ = [ensure_logging_configured]

ensure_logging_configured()
