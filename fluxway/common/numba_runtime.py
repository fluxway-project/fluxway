from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Generator
import hashlib
import os
from pathlib import Path
import tempfile
import threading
from typing import Any


# `numba-cuda` 0.30 mutates this process-global registry when `numba.cuda` is
# imported: the normal CPU `int` and `bool` literal constructors are replaced by
# CUDA-only classes. A CPU dispatcher compiled afterwards then receives CUDA
# literal types for ordinary constants such as the `0` in `range(x.shape[0])`.
# CPU's data-model registry cannot lower those types. This lock protects the
# short interval in which a CPU cold compile temporarily restores the core map.
# It is separate from the file lock below because the map is shared by threads in
# this process, while the file lock coordinates Numba's on-disk cache across
# processes.
_numba_literal_lock = threading.RLock()
_threading_layer_verified = False
_threading_layer_lock = threading.Lock()


def verify_numba_threading_layer() -> str:
    global _threading_layer_verified
    if not _threading_layer_verified:
        with _threading_layer_lock:
            if not _threading_layer_verified:
                # pyrefly: ignore[missing-import]
                import numba.np.ufunc.parallel as parallel

                parallel._launch_threads()
                layer = str(parallel.threading_layer())
                if layer == "workqueue":
                    raise RuntimeError(
                        f"Numba threading layer is '{layer}'. "
                        "The workqueue threading layer does not support concurrent multithreaded execution "
                        "and crashes with SIGABRT upon concurrent access. "
                        "Please install 'tbb' ('pip install tbb') or configure a thread-safe threading layer."
                    )
                _threading_layer_verified = True
    # pyrefly: ignore[missing-import]
    import numba.np.ufunc.parallel as parallel

    return str(parallel.threading_layer())


def _lock_path() -> Path:
    configured = os.environ.get("FLUXWAY_NUMBA_CACHE_LOCK")
    if configured:
        return Path(configured)
    cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    if cache_dir:
        return Path(cache_dir) / "fluxway-compile.lock"
    root = Path(__file__).resolve().parents[2]
    token = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"fluxway-numba-{token}.lock"


@contextmanager
def numba_compile_lock() -> Generator[None, None, None]:
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            import fcntl
        except ImportError as exc:
            raise RuntimeError("Numba cache coordination requires fcntl.") from exc
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def call_numba_cached(dispatcher: Any, *args: Any) -> Any:
    if getattr(dispatcher, "targetoptions", {}).get("parallel", False):
        verify_numba_threading_layer()
    argument_types = tuple(dispatcher.typeof_pyval(argument) for argument in args)
    if argument_types not in dispatcher.overloads:
        with _numba_literal_lock, numba_compile_lock():
            if argument_types not in dispatcher.overloads:
                # pyrefly: ignore[missing-import]
                from numba.core.types import Literal

                # pyrefly: ignore[missing-import]
                from numba.core.types.scalars import BooleanLiteral, IntegerLiteral

                constructors = Literal.ctor_map
                previous_int = constructors.get(int)
                previous_bool = constructors.get(bool)

                # Compile through the public dispatcher using the CPU literal
                # constructors. `dispatcher(*args)` performs normal typing,
                # TargetContext selection, lowering, caching, and execution; this
                # does not catch a compiler error or select an alternate NumPy
                # path. The `finally` block restores the exact registry entries
                # observed on entry so CUDA code keeps its own literal mapping
                # outside this CPU compilation critical section.
                constructors[int] = IntegerLiteral
                constructors[bool] = BooleanLiteral
                try:
                    return dispatcher(*args)
                finally:
                    if previous_int is None:
                        del constructors[int]
                    else:
                        constructors[int] = previous_int
                    if previous_bool is None:
                        del constructors[bool]
                    else:
                        constructors[bool] = previous_bool
    return dispatcher(*args)
