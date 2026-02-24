"""Execution utilities for controlling JIT behavior."""

from typing import Any, Callable, Optional
import functools

from fluxway.common.backend import get_backend


def shield(func: Optional[Callable] = None, *, result_shape: Any = None):
    """
    Decorator/Context Manager to shield a function from JIT compilation.

    When running under JAX JIT, this executes the function on the host using
    `jax.debug.callback` (if no result_shape) or `jax.pure_callback` (if result_shape).

    When running under PyTorch Compile, this uses `torch.compiler.disable`.

    Args:
        func: The function to shield.
        result_shape: (JAX only) The expected shape/dtype of the result if the function returns a value.
                      If None, the function is assumed to have no return value (side-effect only)
                      and will return None.
    """
    if func is None:
        return functools.partial(shield, result_shape=result_shape)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        backend = get_backend()
        backend_name = backend.name()

        if backend_name == "jax":
            import jax

            # Safe Strategy: Always use callback mechanism.
            # In eager execution, jax.debug.callback / pure_callback executes immediately
            # with minimal overhead, which is acceptable.
            # In JIT execution, this safely breaks out to host, ensuring no Tracer leaks.
            # Trying to manually detect 'is_jitted' via tracers is error-prone.

            if result_shape is None:

                def _host_runner(packed):
                    a, k = packed
                    return func(*a, **k)

                # Use debug.callback for side-effects (returns None)
                jax.debug.callback(_host_runner, (args, kwargs))
                return None
            else:

                def _host_runner(packed):
                    a, k = packed
                    return func(*a, **k)

                # Use pure_callback for return values
                return jax.pure_callback(_host_runner, result_shape, (args, kwargs))

        elif backend_name == "torch":
            import torch

            # Optimization: Only use disable if we are actually compiling.
            # Avoids overhead in eager mode.
            if torch.compiler.is_compiling():
                return torch.compiler.disable(func)(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        else:
            return func(*args, **kwargs)

    return wrapper


def compile(func: Optional[Callable] = None, **kwargs):
    """
    Unified compilation decorator (JAX JIT / PyTorch Compile).

    Args:
        func: Function to compile.
        **kwargs: Backend-specific arguments.
            - JAX: static_argnums, inline, etc.
            - PyTorch: mode, fullgraph, dynamic, etc.
              (Defaults to mode="reduce-overhead" if not specified)
    """
    if func is None:
        return functools.partial(compile, **kwargs)

    backend = get_backend()
    backend_name = backend.name()

    if backend_name == "jax":
        import jax

        # Identify JIT-specific args.
        # jax.jit(fun, static_argnums=None, device=None, backend=None, donate_argnums=None, inline=False, keep_unused=False, abstracted_axes=None)
        # We pass everything in kwargs to jax.jit
        return jax.jit(func, **kwargs)

    elif backend_name == "torch":
        import torch

        # Set default mode to reduce-overhead if not provided, per user request
        if "mode" not in kwargs:
            kwargs["mode"] = "reduce-overhead"

        return torch.compile(func, **kwargs)

    else:
        # No-op for other backends (e.g. numpy?)
        return func
