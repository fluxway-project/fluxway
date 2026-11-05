"""Capability-composed checkpoint component registrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from fluxway.common.types.checkpoint import (
    CheckpointComponentCapability,
    CheckpointComponentPublication,
    CheckpointComponentSpec,
    CheckpointCommitPermit,
    CheckpointHold,
    CheckpointObjectRef,
    CheckpointRuntime,
    CleanupReservation,
    CleanupReservationRequest,
    CommitSaveSession,
    RestoreSession,
    SaveSession,
    WorkerSaveResult,
)


class CheckpointComponent(ABC):
    @property
    @abstractmethod
    def descriptor(self) -> CheckpointComponentSpec: ...


class CheckpointComponentBinding(CheckpointComponent):
    @property
    @abstractmethod
    def bound_spmd_id(self) -> str | None: ...

    @abstractmethod
    def bind_to_spmd(self, spmd_id: str) -> None: ...

    def require_spmd_binding(self, spmd_id: str) -> None:
        if self.bound_spmd_id != spmd_id:
            raise ValueError(
                f"Checkpoint component {self.descriptor.component_id!r} requires "
                f"binding to SPMD {spmd_id!r}; currently bound to "
                f"{self.bound_spmd_id!r}. Instantiate the decorated SPMD in a "
                "FunctionalGraph for automatic binding. Low-level callers must "
                f"call component.bind_to_spmd({spmd_id!r}) on a fresh component "
                "before creating a checkpointer, worker coordinator, or "
                "opening its registry."
            )


class CheckpointComponentLifecycle(ABC):
    @abstractmethod
    async def open(self, runtime: CheckpointRuntime) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class CheckpointComponentFinalize(ABC):
    @abstractmethod
    async def finalize_save(self, session: SaveSession) -> None: ...


class CheckpointComponentSave(ABC):
    @abstractmethod
    async def prepare_save(self, session: SaveSession) -> None: ...

    @abstractmethod
    async def save(self, session: SaveSession) -> WorkerSaveResult: ...

    @abstractmethod
    async def finish_save(self, session: SaveSession) -> None: ...

    @abstractmethod
    async def commit_save(
        self, session: CommitSaveSession
    ) -> CheckpointComponentPublication: ...

    @abstractmethod
    async def rollback_save(self, session: SaveSession) -> None: ...

    @abstractmethod
    async def publish_save(self, session: SaveSession) -> None: ...

    @abstractmethod
    async def abort_save(self, permit: CheckpointCommitPermit) -> None:
        """Idempotently discard an incomplete attempt from the control runtime."""
        ...

    @abstractmethod
    async def acknowledge_save(self, permit: CheckpointCommitPermit) -> None:
        """Idempotently acknowledge a globally committed attempt on control."""
        ...


class CheckpointComponentRestore(ABC):
    @abstractmethod
    async def restore(self, session: RestoreSession) -> None: ...


class CheckpointObjectLifecycle(ABC):
    @abstractmethod
    async def hold_checkpoint_object(
        self,
        checkpoint_object: CheckpointObjectRef,
        lease_holder_id: str,
        ttl_seconds: float,
        request_id: str,
    ) -> CheckpointHold: ...

    @abstractmethod
    async def renew_checkpoint_hold(
        self, lease: CheckpointHold, ttl_seconds: float
    ) -> CheckpointHold: ...

    @abstractmethod
    async def release_checkpoint_hold(self, lease: CheckpointHold) -> None: ...

    @abstractmethod
    async def reserve_cleanup(
        self, request: CleanupReservationRequest
    ) -> CleanupReservation: ...

    @abstractmethod
    async def commit_cleanup(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None: ...

    @abstractmethod
    async def rollback_cleanup(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None: ...


class CheckpointObjectCleanup(ABC):
    @abstractmethod
    async def cleanup(self, reservation: CleanupReservation) -> None: ...


_CAPABILITY_IMPLEMENTATIONS = (
    (
        CheckpointComponentCapability.COMPONENT_LIFECYCLE,
        "component_lifecycle",
        CheckpointComponentLifecycle,
    ),
    (
        CheckpointComponentCapability.FINALIZE,
        "finalize",
        CheckpointComponentFinalize,
    ),
    (CheckpointComponentCapability.SAVE, "save", CheckpointComponentSave),
    (CheckpointComponentCapability.RESTORE, "restore", CheckpointComponentRestore),
    (
        CheckpointComponentCapability.OBJECT_LIFECYCLE,
        "object_lifecycle",
        CheckpointObjectLifecycle,
    ),
    (
        CheckpointComponentCapability.OBJECT_CLEANUP,
        "object_cleanup",
        CheckpointObjectCleanup,
    ),
)


@dataclass(frozen=True, slots=True)
class CheckpointComponentRegistration:
    component: CheckpointComponent
    component_lifecycle: CheckpointComponentLifecycle | None = None
    finalize: CheckpointComponentFinalize | None = None
    save: CheckpointComponentSave | None = None
    restore: CheckpointComponentRestore | None = None
    object_lifecycle: CheckpointObjectLifecycle | None = None
    object_cleanup: CheckpointObjectCleanup | None = None

    @property
    def descriptor(self) -> CheckpointComponentSpec:
        return self.component.descriptor

    @property
    def capabilities(self) -> frozenset[CheckpointComponentCapability]:
        return frozenset(
            capability
            for capability, field_name, _ in _CAPABILITY_IMPLEMENTATIONS
            if getattr(self, field_name) is not None
        )

    def __post_init__(self) -> None:
        for capability, field_name, interface in _CAPABILITY_IMPLEMENTATIONS:
            implementation = getattr(self, field_name)
            if implementation is not None and not isinstance(implementation, interface):
                raise TypeError(
                    f"{capability.value} capability must implement {interface.__name__}"
                )
        if (self.save is None) != (self.descriptor.save is None):
            raise ValueError(
                "checkpoint save implementation must match save specification"
            )
        if (self.restore is None) != (self.descriptor.restore is None):
            raise ValueError(
                "checkpoint restore implementation must match restore specification"
            )
        if self.object_cleanup is not None and self.object_lifecycle is None:
            raise ValueError(
                "object cleanup capability requires object lifecycle capability"
            )


class CheckpointComponentRegistry:
    """Immutable registrations with startup-validated, direct capability dispatch."""

    def __init__(
        self, registrations: tuple[CheckpointComponentRegistration, ...]
    ) -> None:
        component_ids = [
            registration.descriptor.component_id for registration in registrations
        ]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("checkpoint component IDs must be unique; duplicate found")
        self._registrations = registrations
        self._by_id = {
            registration.descriptor.component_id: registration
            for registration in registrations
        }

    def __iter__(self):
        return iter(self._registrations)

    def __len__(self) -> int:
        return len(self._registrations)

    def require(self, component_id: str) -> CheckpointComponentRegistration:
        registration = self._by_id.get(component_id)
        if registration is None:
            raise KeyError(f"checkpoint component is not registered: {component_id!r}")
        return registration

    def get(self, component_id: str) -> CheckpointComponentRegistration | None:
        return self._by_id.get(component_id)

    def __contains__(self, component_id: str) -> bool:
        return component_id in self._by_id

    def validate_spmd_binding(self, spmd_id: str) -> None:
        for registration in self:
            if isinstance(registration.component, CheckpointComponentBinding):
                registration.component.require_spmd_binding(spmd_id)

    async def open_all(self, runtime: CheckpointRuntime) -> None:
        self.validate_spmd_binding(runtime.spmd_id)
        opened: list[CheckpointComponentLifecycle] = []
        try:
            for registration in self._registrations:
                lifecycle = registration.component_lifecycle
                if lifecycle is None:
                    continue
                await lifecycle.open(runtime)
                opened.append(lifecycle)
        except BaseException as primary_error:
            cleanup_errors: list[BaseException] = []
            for lifecycle in reversed(opened):
                try:
                    await lifecycle.close()
                except BaseException as close_error:
                    cleanup_errors.append(close_error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "checkpoint component open and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise

    async def close_all(self) -> None:
        errors: list[BaseException] = []
        for registration in reversed(self._registrations):
            lifecycle = registration.component_lifecycle
            if lifecycle is None:
                continue
            try:
                await lifecycle.close()
            except BaseException as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("checkpoint component close failed", errors)
