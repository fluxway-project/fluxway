from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine
from typing import Any


def use_uvloop() -> bool:
    value = os.environ.get("FLUXWAY_USE_UVLOOP", "1").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "FLUXWAY_USE_UVLOOP must be one of 1/0/true/false/yes/no/on/off, "
        f"got {value!r}."
    )


def run(coro: Coroutine[Any, Any, Any]) -> Any:
    if use_uvloop():
        import uvloop  # pyrefly: ignore [missing-import]

        return uvloop.run(coro)
    return asyncio.run(coro)
