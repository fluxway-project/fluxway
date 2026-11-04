from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Awaitable, Callable, Coroutine
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


async def invoke_callable_async(
    callable_obj: Callable[..., Any],
    *args: Any,
    lane: str = "default",
    lane_runner: Callable[..., Awaitable[Any]] | None = None,
    **kwargs: Any,
) -> Any:
    """Invoke a synchronous or asynchronous callable without blocking the event loop."""
    if inspect.iscoroutinefunction(callable_obj) or (
        callable(callable_obj) and inspect.iscoroutinefunction(callable_obj.__call__)
    ):
        result = callable_obj(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    if lane_runner is not None:
        return await lane_runner(callable_obj, *args, lane=lane, **kwargs)

    from fluxway.common.decorators import get_current_context

    ctx_state = get_current_context()
    if ctx_state.lane_runner is not None:
        return await ctx_state.lane_runner(callable_obj, *args, lane=lane, **kwargs)

    raise RuntimeError(
        f"Cannot invoke synchronous callable {callable_obj!r} asynchronously: "
        "no lane runner is available in the current operator context."
    )
