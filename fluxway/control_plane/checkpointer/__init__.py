from .checkpoint import FluxwayCheckpointer
from .backend import WorkerCheckpointBackend
from .safetensors import SafeTensorsBackend

try:
    from .orbax import OrbaxBackend
except ModuleNotFoundError as exc:
    _ORBAX_IMPORT_ERROR = exc

    class OrbaxBackend(WorkerCheckpointBackend, backend_name="orbax"):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "OrbaxBackend requires optional jax/orbax dependencies."
            ) from _ORBAX_IMPORT_ERROR

__all__ = [
    "FluxwayCheckpointer",
    "WorkerCheckpointBackend",
    "SafeTensorsBackend",
    "OrbaxBackend",
]
