"""
Logging utilities.

Why this exists:
- Ray actors run in separate worker processes.
- Those processes often don't inherit the driver's logging configuration
  (e.g. logging.basicConfig in an example script), so INFO logs can be dropped.
"""

from __future__ import annotations

import logging
import os


DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

ENV_LOG_LEVEL = "FLUXWAY_LOG_LEVEL"
ENV_LOG_FORCE = "FLUXWAY_LOG_FORCE"

_logger = logging.getLogger(__name__)


def ensure_logging_configured(
    *,
    level: int | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    force: bool | None = None,
) -> None:
    """
    Ensure the current process has at least one logging handler configured.

    This is intentionally conservative: if handlers already exist, it does nothing
    (unless force=True). This avoids clobbering the application's logging setup.
    """

    root_logger = logging.getLogger()

    HEARTBEAT = 5  # DEBUG=10, INFO=20, WARNING=30
    logging.addLevelName(HEARTBEAT, "HEARTBEAT")
    _logger.heartbeat = lambda msg: _logger.log(HEARTBEAT, msg)

    env_level = _parse_log_level(os.getenv(ENV_LOG_LEVEL))
    if level is not None:
        resolved_level = level
    else:
        resolved_level = env_level

    env_force = _parse_bool(os.getenv(ENV_LOG_FORCE))
    if force is not None:
        resolved_force = force
    elif env_force is not None:
        resolved_force = env_force
    else:
        resolved_force = True

    logging.basicConfig(
        level=resolved_level, format=log_format, datefmt=datefmt, force=resolved_force
    )

    root_logger.setLevel(resolved_level)
    for handler in root_logger.handlers:
        handler.setLevel(resolved_level)

    _logger.debug(
        "Logging configured in PID=%s: level=%s, force=%s, env_level=%r, env_force=%r",
        os.getpid(),
        logging.getLevelName(resolved_level),
        resolved_force,
        os.getenv(ENV_LOG_LEVEL),
        os.getenv(ENV_LOG_FORCE),
    )


def ensure_logger_info_visible(
    logger: logging.Logger, *, level: int = logging.INFO
) -> None:
    """
    Ensure the given logger will emit INFO logs in this process.

    If the logger has an explicit level already, we don't override it.
    """

    if logger.level == logging.NOTSET:
        logger.setLevel(level)


def _parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _parse_log_level(value: str | None) -> int:
    """
    Parse a log level from env.

    Supported:
    - Names: DEBUG/INFO/WARNING/ERROR/CRITICAL (case-insensitive)
    - Integers: "10", "20", ...
    """
    if not value:
        return logging.INFO

    raw = value.strip()
    if raw.isdigit():
        return int(raw)

    name = raw.upper()
    return getattr(logging, name, logging.INFO)
