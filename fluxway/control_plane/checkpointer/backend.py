from abc import ABC, abstractmethod
from typing import Any, Dict, Type

_BACKEND_REGISTRY: Dict[str, Type["WorkerCheckpointBackend"]] = {}


class WorkerCheckpointBackend(ABC):
    """
    Per-worker checkpoint serialization interface.

    Each implementation persists a worker's in-memory state dict to a shared
    filesystem and reconstructs it on demand.  Implementations must be
    picklable (Ray actor constraint).

    Subclasses register themselves automatically via ``backend_name``:

        class MyBackend(WorkerCheckpointBackend, backend_name="myformat"):
            ...

    After which ``WorkerCheckpointBackend.create("myformat")`` works without
    any changes to the call site (open for extension, closed for modification).
    """

    def __init_subclass__(cls, backend_name: str = "", **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if backend_name:
            _BACKEND_REGISTRY[backend_name] = cls

    @classmethod
    def create(cls, name: str, force_write: bool = False) -> "WorkerCheckpointBackend":
        """Instantiate a registered backend by name."""
        backend_cls = _BACKEND_REGISTRY.get(name)
        if backend_cls is None:
            raise ValueError(
                f"Unknown checkpoint backend: {name!r}. "
                f"Registered backends: {sorted(_BACKEND_REGISTRY)}"
            )
        return backend_cls(force_write=force_write)

    @classmethod
    def registered_names(cls) -> list:
        return sorted(_BACKEND_REGISTRY)

    @abstractmethod
    async def save(
        self, state: Dict[str, Any], ckpt_dir: str, spmd_id: str, rank: int, step: int
    ) -> Dict[str, Any]:
        """Persist ``state`` and return opaque metadata needed by :meth:`load`."""

    @abstractmethod
    async def load(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct and return the state dict described by ``metadata``."""

    @abstractmethod
    def finalize(self) -> None:
        """
        Wait for any outstanding background I/O operations to complete.
        Default implementation is a no-op for synchronous backends.

        Called by workers after each checkpoint to ensure durability before the
        Controller writes the manifest.  The checkpointer may be reused afterwards.
        """

    def close(self) -> None:
        """Flush outstanding I/O and release all resources (called on actor teardown).

        Default implementation delegates to finalize(); backends that manage
        persistent resources (e.g. file handles, thread pools) should override.
        """
        self.finalize()
