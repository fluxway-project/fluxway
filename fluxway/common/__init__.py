"""Common utilities and types for Fluxway framework."""

# Ensure INFO logs are visible even when importing subpackages directly.
from .utils.logging import ensure_logging_configured  # noqa: E402

ensure_logging_configured()
