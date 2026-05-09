import asyncio
import os
import sys
from typing import Any
from collections.abc import Callable, Awaitable
import logging

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None  # type: ignore

logger = logging.getLogger(__name__)


class _EnvRestorer:
    """Context manager to restore environment variables."""

    def __init__(self, keys):
        self.keys = keys
        self.saved = {}

    def __enter__(self):
        for k in self.keys:
            self.saved[k] = os.environ.get(k)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self.saved.items():
            if v is None and k in os.environ:
                del os.environ[k]
            elif v is not None:
                os.environ[k] = v


def launch(
    entry_fn: Callable[..., Awaitable[Any]],
    *args: Any,
    ray_init_kwargs: dict[str, Any] | None = None,
    num_cpus: int = 1,
    runtime_env: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Fluxway Standard Launcher — a pure execution sandbox.

    Runs ``entry_fn`` inside an isolated Ray Task with **zero GPUs** so that
    the main / Jupyter process never triggers eager GPU memory allocation when
    JAX or PyTorch is imported.

    All stateful orchestration (scheduling, heartbeat, checkpoint, drain,
    backpressure, dynamic graph reconfiguration, …) is the exclusive
    responsibility of the **Controller** created inside ``entry_fn``.
    This function intentionally uses a Ray Task (not an Actor) to make
    that boundary unambiguous.

    Args:
        entry_fn: An async coroutine containing the user's training logic
                  (graph construction → Controller creation → execution).
        *args:    Positional arguments forwarded to ``entry_fn``.
        ray_init_kwargs:
                  Optional kwargs passed to ``ray.init()`` when Ray has not
                  been initialised yet.  Defaults to
                  ``ignore_reinit_error=True`` for Jupyter compatibility.
        num_cpus: CPUs allocated to the sandbox task (default 1).
        runtime_env:
                  Optional Ray runtime_env dict forwarded to the sandbox task
                  (e.g. ``{"env_vars": {"MY_VAR": "1"}}``).
        **kwargs: Keyword arguments forwarded to ``entry_fn``.

    Returns:
        Whatever ``entry_fn`` returns.
    """
    if not RAY_AVAILABLE:
        raise RuntimeError(
            "Ray is required by fluxway.launch.  "
            "Install it with: pip install 'ray[default]'"
        )

    if not ray.is_initialized():
        init_kw = (ray_init_kwargs or {}).copy()
        init_kw.setdefault("ignore_reinit_error", True)
        if "address" not in init_kw and os.environ.get("RAY_ADDRESS"):
            init_kw["address"] = os.environ["RAY_ADDRESS"]
        ray.init(**init_kw)

    sandbox_env = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "CUDA_VISIBLE_DEVICES": "",
    }
    if runtime_env is not None and "env_vars" in runtime_env:
        sandbox_env.update(runtime_env["env_vars"])

    task_options: dict[str, Any] = {
        "num_cpus": num_cpus,
        "num_gpus": 0,
        "runtime_env": runtime_env or {},
    }

    @ray.remote(**task_options)
    def _sandbox(fn, a, kw):
        # Apply sandbox environment variables locally to this task's process.
        # We do this here instead of in `runtime_env` to avoid polluting the
        # environment of child actors (which inherit runtime_env by default).
        with _EnvRestorer(sandbox_env.keys()):
            for k, v in sandbox_env.items():
                os.environ[k] = v
            return asyncio.run(fn(*a, **kw))

    ref = _sandbox.remote(entry_fn, args, kwargs)
    try:
        return ray.get(ref)
    except KeyboardInterrupt:
        logger.warning("Interrupted — cancelling sandbox task …")
        ray.cancel(ref, force=True)
        sys.exit(1)
