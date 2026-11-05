from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fluxway.common.types.checkpoint import (
    CheckpointHold,
    CheckpointIdReference,
    CheckpointManifest,
    CheckpointObjectRef,
    CheckpointObjectSpec,
    CheckpointRuntime,
    CommitAuthority,
    CommitSaveSession,
    CheckpointComponentPublication,
    CheckpointComponentSpec,
    CheckpointRetentionPolicy,
    NameReference,
    RestoreSession,
    SaveCheckpointRequest,
    SaveSession,
    StepReference,
    WorkerSaveResult,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponent,
    CheckpointComponentBinding,
    CheckpointComponentLifecycle,
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
    CheckpointObjectCleanup,
    CheckpointObjectLifecycle,
    CheckpointComponentRestore,
    CheckpointComponentSave,
)
from fluxway.control_plane.checkpointer.inline import StateCheckpointComponent

StateSave = Callable[[Any, SaveSession], Any]
StateRestore = Callable[[Any, Any, RestoreSession], Any]


def state_checkpoint_component(
    component_id: str,
    save: StateSave,
    restore: StateRestore,
) -> StateCheckpointComponent:
    return StateCheckpointComponent(component_id, save, restore)


def register_checkpoint_component(
    component: CheckpointComponent,
    *,
    component_lifecycle: CheckpointComponentLifecycle | None = None,
    save: CheckpointComponentSave | None = None,
    restore: CheckpointComponentRestore | None = None,
    object_lifecycle: CheckpointObjectLifecycle | None = None,
    object_cleanup: CheckpointObjectCleanup | None = None,
) -> CheckpointComponentRegistration:
    return CheckpointComponentRegistration(
        component=component,
        component_lifecycle=component_lifecycle,
        save=save,
        restore=restore,
        object_lifecycle=object_lifecycle,
        object_cleanup=object_cleanup,
    )


__all__ = [
    "CheckpointComponent",
    "CheckpointComponentBinding",
    "CheckpointComponentLifecycle",
    "CheckpointComponentRegistration",
    "CheckpointComponentRegistry",
    "CheckpointHold",
    "CheckpointObjectCleanup",
    "CheckpointObjectLifecycle",
    "CheckpointComponentRestore",
    "CheckpointComponentSave",
    "CheckpointIdReference",
    "CheckpointManifest",
    "CheckpointObjectRef",
    "CheckpointObjectSpec",
    "CheckpointRuntime",
    "CommitAuthority",
    "CommitSaveSession",
    "CheckpointComponentPublication",
    "CheckpointComponentSpec",
    "CheckpointRetentionPolicy",
    "NameReference",
    "RestoreSession",
    "SaveCheckpointRequest",
    "SaveSession",
    "StateCheckpointComponent",
    "StepReference",
    "WorkerSaveResult",
    "state_checkpoint_component",
    "register_checkpoint_component",
]
