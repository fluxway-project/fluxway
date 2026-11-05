"""Checkpoint domain identities shared across control and worker processes."""

from __future__ import annotations

import asyncio
import copy
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import prod
from typing import Any


type RestoreOption = bytes
type RestoreOptionsByComponent = Mapping[str, RestoreOption]
type RestoreOptionsByRank = Mapping[int, RestoreOptionsByComponent]
type RestoreOptions = Mapping[str, RestoreOptionsByRank]


def validate_restore_options(
    options: RestoreOptions | None,
) -> None:
    if options is None:
        return
    if not isinstance(options, Mapping):
        raise TypeError("restore_options must be an SPMD mapping")
    for spmd_id, rank_options in options.items():
        if not isinstance(spmd_id, str) or not spmd_id:
            raise TypeError("restore_options SPMD IDs must be non-empty strings")
        if not isinstance(rank_options, Mapping):
            raise TypeError(f"restore_options for {spmd_id!r} must be a rank mapping")
        for rank, component_options in rank_options.items():
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise TypeError(
                    "restore_options rank keys must be non-negative integers"
                )
            if not isinstance(component_options, Mapping):
                raise TypeError(
                    f"restore_options for {spmd_id!r} rank {rank} "
                    "must be a component mapping"
                )
            for component_id, option in component_options.items():
                if not isinstance(component_id, str) or not component_id:
                    raise TypeError(
                        "restore_options component IDs must be non-empty strings"
                    )
                if not isinstance(option, bytes) or not option:
                    raise TypeError("restore_options values must be non-empty bytes")


class CheckpointError(RuntimeError):
    pass


class CheckpointBusyError(CheckpointError):
    pass


class CheckpointObjectCleanedUpError(CheckpointError):
    pass


class LeaseConflictError(CheckpointError):
    pass


class CheckpointCleanupBlocked(CheckpointError):
    pass


class TopologyMismatchError(CheckpointError):
    pass


class CheckpointId(str):
    def __new__(cls, value: str) -> CheckpointId:
        if not value:
            raise ValueError("checkpoint ID must be non-empty")
        return str.__new__(cls, value)

    @classmethod
    def generate(cls) -> CheckpointId:
        return cls(uuid.uuid4().hex)


@dataclass(frozen=True, slots=True)
class CheckpointDeadline:
    """Canonical deadline manager for checkpoint transaction lifecycle."""

    deadline_monotonic: float
    total_timeout_s: float

    @classmethod
    def start(cls, timeout_s: float) -> CheckpointDeadline:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TypeError("checkpoint timeout must be a float or int")
        if timeout_s <= 0:
            raise ValueError(f"checkpoint timeout must be positive, got {timeout_s}")
        return cls(
            deadline_monotonic=time.monotonic() + float(timeout_s),
            total_timeout_s=float(timeout_s),
        )

    def remaining_s(self) -> float:
        remaining = self.deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Checkpoint transaction exceeded deadline of {self.total_timeout_s:.1f}s"
            )
        return remaining

    def is_expired(self) -> bool:
        return (self.deadline_monotonic - time.monotonic()) <= 0

    async def run_phase(self, phase_or_coro: Any, coro: Any = None) -> Any:
        if coro is None:
            awaitable = phase_or_coro
        else:
            awaitable = coro
        remaining = self.remaining_s()
        return await asyncio.wait_for(awaitable, timeout=remaining)

    def run_transition[T](self, transition: Callable[..., T], *args: Any) -> T:
        self.remaining_s()
        result = transition(*args)
        self.remaining_s()
        return result


class CheckpointRuntimeRole(StrEnum):
    CONTROL = "control"
    WORKER = "worker"


def checkpoint_position(runtime_state: Mapping[str, Any]) -> int:
    if not isinstance(runtime_state, Mapping):
        raise TypeError("checkpoint runtime state must be a mapping")
    if set(runtime_state) != {"drain_boundary"}:
        raise ValueError("checkpoint runtime state must contain only drain_boundary")
    boundary = runtime_state["drain_boundary"]
    if not isinstance(boundary, Mapping):
        raise TypeError("checkpoint runtime drain_boundary must be a mapping")
    position = boundary.get("committed_cut")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError(
            "checkpoint runtime drain_boundary requires a non-negative committed_cut"
        )
    return position


class CheckpointComponentCapability(StrEnum):
    COMPONENT_LIFECYCLE = "component_lifecycle"
    FINALIZE = "finalize"
    SAVE = "save"
    RESTORE = "restore"
    OBJECT_LIFECYCLE = "object_lifecycle"
    OBJECT_CLEANUP = "object_cleanup"


class CheckpointSaveState(StrEnum):
    FINALIZING = "finalizing"
    PREPARED = "prepared"
    SAVING = "saving"
    SAVED = "saved"
    FINISHED = "finished"
    COMMITTED = "committed"


class CheckpointTransactionState(StrEnum):
    STAGED = "staged"
    COMMIT_DECIDED = "commit_decided"
    COMMITTED = "committed"
    ABANDONED = "abandoned"


class CheckpointSaveScope(StrEnum):
    LEADER = "leader"
    ALL_RANKS = "all_ranks"


class RestoreTopologyMode(StrEnum):
    EXACT_TOPOLOGY = "exact_topology"
    LOGICAL_PARTITION_REMAP = "logical_partition_remap"
    GLOBAL_RESHARD = "global_reshard"


@dataclass(frozen=True, slots=True)
class CheckpointCut:
    cut_id: str
    position: int | None
    stream_cuts: Mapping[str, Any] = field(default_factory=dict)
    sealed_ends: Mapping[str, Any] = field(default_factory=dict)
    control_epoch: int = 0

    def __post_init__(self) -> None:
        if not self.cut_id:
            raise ValueError("checkpoint cut requires cut_id")
        if self.position is not None and (
            isinstance(self.position, bool) or self.position < 0
        ):
            raise ValueError("position must be a non-negative integer or None")
        if self.control_epoch < 0:
            raise ValueError("control_epoch must be non-negative")

    @classmethod
    def for_position(cls, cut_id: str, position: int | None) -> CheckpointCut:
        return cls(cut_id=cut_id, position=position)

    @classmethod
    def for_step(cls, cut_id: str, step: int) -> CheckpointCut:
        return cls.for_position(cut_id, step)

    def frozen_copy(self) -> CheckpointCut:
        return CheckpointCut(
            cut_id=self.cut_id,
            position=self.position,
            stream_cuts=copy.deepcopy(dict(self.stream_cuts)),
            sealed_ends=copy.deepcopy(dict(self.sealed_ends)),
            control_epoch=self.control_epoch,
        )


@dataclass(frozen=True, slots=True)
class CheckpointCommitPermit:
    attempt_id: str
    coordinator_generation: int
    token: bytes

    def __post_init__(self) -> None:
        if not self.attempt_id:
            raise ValueError("checkpoint commit permit requires attempt_id")
        if self.coordinator_generation < 0:
            raise ValueError("checkpoint commit permit generation must be non-negative")
        if not isinstance(self.token, bytes) or not self.token:
            raise ValueError("checkpoint commit permit requires an opaque token")


@dataclass(frozen=True, slots=True)
class CommitAuthority:
    scope: str
    generation: int
    token: bytes

    def __post_init__(self) -> None:
        if not self.scope:
            raise ValueError("commit authority requires scope")
        if self.generation < 0:
            raise ValueError("commit authority generation must be non-negative")
        if not self.token:
            raise ValueError("commit authority requires token")


@dataclass(frozen=True, slots=True)
class TopologySnapshot:
    world_size: int
    mesh_axis_names: tuple[str, ...]
    mesh_axis_shapes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("topology world_size must be positive")
        if len(self.mesh_axis_names) != len(self.mesh_axis_shapes):
            raise ValueError("mesh axis names and shapes must have equal length")
        if len(set(self.mesh_axis_names)) != len(self.mesh_axis_names):
            raise ValueError("mesh axis names must be unique")
        if any(size <= 0 for size in self.mesh_axis_shapes):
            raise ValueError("mesh axis shapes must be positive")
        if self.mesh_axis_shapes and prod(self.mesh_axis_shapes) != self.world_size:
            raise ValueError("mesh shape product must equal topology world_size")


@dataclass(frozen=True, slots=True)
class CheckpointObjectKey:
    component_id: str
    checkpoint_object_id: str
    checkpoint_id: str
    position: int | None
    spmd_id: str
    partition_key: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("component_id", self.component_id),
            ("checkpoint_object_id", self.checkpoint_object_id),
            ("checkpoint_id", self.checkpoint_id),
            ("spmd_id", self.spmd_id),
        ):
            if not value:
                raise ValueError(f"checkpoint_object key requires {name}")
        if self.position is not None and (
            isinstance(self.position, bool) or self.position < 0
        ):
            raise ValueError("checkpoint_object position must be non-negative or None")


@dataclass(frozen=True, slots=True)
class CheckpointObjectRef:
    component_id: str
    checkpoint_object_id: str
    version_id: str | bytes
    storage_locator: bytes
    partition_key: str | None = None
    cleanup_group: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.version_id, (str, bytes)):
            raise TypeError("checkpoint_object version_id must be str or bytes")
        if (
            not self.component_id
            or not self.checkpoint_object_id
            or not self.version_id
        ):
            raise ValueError(
                "checkpoint_object reference requires component, checkpoint_object, and version"
            )
        if not isinstance(self.storage_locator, bytes) or not self.storage_locator:
            raise ValueError(
                "checkpoint_object reference requires an opaque storage_locator"
            )


@dataclass(frozen=True, slots=True)
class CheckpointRetentionPolicy:
    retain_last: int | None = None
    retain_forever: bool = False

    def __post_init__(self) -> None:
        if self.retain_last is not None and self.retain_last <= 0:
            raise ValueError("retain_last must be positive or None")
        if self.retain_forever and self.retain_last is not None:
            raise ValueError("retain_forever cannot define retain_last")


@dataclass(frozen=True, slots=True)
class CheckpointSaveSpec:
    prepare_scope: CheckpointSaveScope = CheckpointSaveScope.LEADER
    commit_scope: CheckpointSaveScope = CheckpointSaveScope.LEADER


@dataclass(frozen=True, slots=True)
class CheckpointRestoreSpec:
    topology: RestoreTopologyMode = RestoreTopologyMode.EXACT_TOPOLOGY


@dataclass(frozen=True, slots=True)
class CheckpointObjectSpec:
    checkpoint_object_id: str = "state"
    required_for_restore: bool = True
    cleanup_group: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_object_id:
            raise ValueError(
                "checkpoint_object checkpoint_object requires checkpoint_object_id"
            )


@dataclass(frozen=True, slots=True)
class CheckpointComponentSpec:
    component_id: str
    retention_domain: str | None = None
    save: CheckpointSaveSpec | None = None
    restore: CheckpointRestoreSpec | None = None
    checkpoint_objects: tuple[CheckpointObjectSpec, ...] = ()
    output_path: str | None = None

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("checkpoint component requires component_id")
        if self.retention_domain is not None and not self.retention_domain:
            raise ValueError("retention_domain must be non-empty when provided")
        if self.output_path is not None:
            if not isinstance(self.output_path, str):
                raise TypeError(
                    f"Checkpoint component {self.component_id!r} output_path "
                    f"must be a string or None; got {self.output_path!r}. "
                    "Use a step-relative path such as 'sparse/UserTable'."
                )
            path = self.output_path
            if (
                not path
                or path.startswith("/")
                or "\\" in path
                or "\x00" in path
                or any(part in {"", ".", ".."} for part in path.split("/"))
            ):
                raise ValueError(
                    f"Checkpoint component {self.component_id!r} has invalid "
                    f"output_path={path!r}. Use a non-empty relative POSIX path "
                    "under steps/<step>/, for example 'sparse/UserTable', "
                    "without backslashes, NUL, empty segments, '.' or '..'."
                )
        checkpoint_object_ids = [
            checkpoint_object.checkpoint_object_id
            for checkpoint_object in self.checkpoint_objects
        ]
        if len(checkpoint_object_ids) != len(set(checkpoint_object_ids)):
            raise ValueError(
                "checkpoint component checkpoint_object IDs must be unique"
            )
        if self.checkpoint_objects and self.save is None and self.restore is None:
            raise ValueError("checkpoint objects require save or restore capability")
        if (
            any(
                checkpoint_object.required_for_restore
                for checkpoint_object in self.checkpoint_objects
            )
            and self.restore is None
        ):
            raise ValueError(
                "checkpoint objects required for restore require restore capability"
            )


@dataclass(frozen=True, slots=True)
class SaveSession:
    instance: Any
    attempt_id: str
    checkpoint_id: str
    commit_permit: CheckpointCommitPermit
    cut: CheckpointCut
    spmd_id: str
    rank: int
    source_topology: TopologySnapshot
    component_id: str
    repository_root: str = ""
    staging_output_path: str | None = None
    published_output_path: str | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id or not self.checkpoint_id:
            raise ValueError("save session requires attempt and checkpoint IDs")
        if self.commit_permit.attempt_id != self.attempt_id:
            raise ValueError("save session permit belongs to another attempt")
        if not self.spmd_id or not self.component_id:
            raise ValueError("save session requires SPMD and component IDs")
        if self.rank < 0 or self.rank >= self.source_topology.world_size:
            raise ValueError("save session rank is outside source topology")

    def checkpoint_object_key(
        self, checkpoint_object_id: str, partition_key: str | None = None
    ) -> CheckpointObjectKey:
        return CheckpointObjectKey(
            component_id=self.component_id,
            checkpoint_object_id=checkpoint_object_id,
            checkpoint_id=self.checkpoint_id,
            position=self.cut.position,
            spmd_id=self.spmd_id,
            partition_key=partition_key,
        )

    @property
    def current_topology(self) -> None:
        return None

    def worker_save_result(self, payload: bytes = b"") -> WorkerSaveResult:
        return WorkerSaveResult(commit_metadata=payload)

    def checkpoint_object_ref(
        self,
        checkpoint_object_id: str,
        version_id: str,
        storage_locator: bytes,
        *,
        partition_key: str | None = None,
        cleanup_group: str | None = None,
    ) -> CheckpointObjectRef:
        return CheckpointObjectRef(
            component_id=self.component_id,
            checkpoint_object_id=checkpoint_object_id,
            version_id=version_id,
            storage_locator=storage_locator,
            partition_key=partition_key,
            cleanup_group=cleanup_group,
        )

    def component_checkpoint(
        self, objects: tuple[CheckpointObjectRef, ...]
    ) -> CheckpointComponentPublication:
        return CheckpointComponentPublication(
            component_id=self.component_id,
            shared_records=(CheckpointObjectRecord(checkpoint_objects=objects),),
        )

    def commit_authority(self, scope: str) -> CommitAuthority:
        return CommitAuthority(
            scope=scope,
            generation=self.commit_permit.coordinator_generation,
            token=self.commit_permit.token,
        )


@dataclass(frozen=True, slots=True)
class CommitSaveSession(SaveSession):
    worker_results: Mapping[int, WorkerSaveResult] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RestoreSession:
    instance: Any
    checkpoint_id: str
    cut: CheckpointCut
    spmd_id: str
    rank: int
    saved_topology: TopologySnapshot
    restore_topology: TopologySnapshot
    component_id: str
    component_checkpoint: CheckpointComponentPublication
    restore_options: RestoreOption | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_id or not self.spmd_id or not self.component_id:
            raise ValueError(
                "restore session requires checkpoint, SPMD, and component IDs"
            )
        if self.rank < 0 or self.rank >= self.restore_topology.world_size:
            raise ValueError("restore session rank is outside restore topology")
        if self.restore_options is not None and (
            not isinstance(self.restore_options, bytes) or not self.restore_options
        ):
            raise TypeError("restore_options must be non-empty opaque bytes")

    @property
    def source_topology(self) -> TopologySnapshot:
        return self.saved_topology

    @property
    def current_topology(self) -> TopologySnapshot:
        return self.restore_topology

    @property
    def commit_permit(self) -> None:
        return None

    @property
    def attempt_id(self) -> str:
        return f"restore:{self.checkpoint_id}"


@dataclass(frozen=True, slots=True)
class CheckpointRuntime:
    role: CheckpointRuntimeRole
    repository_root: str
    spmd_id: str
    rank: int | None = None

    def __post_init__(self) -> None:
        if not self.repository_root or not self.spmd_id:
            raise ValueError("component runtime requires repository root and SPMD ID")
        if self.role is CheckpointRuntimeRole.WORKER and self.rank is None:
            raise ValueError("worker component runtime requires rank")


@dataclass(frozen=True, slots=True)
class CheckpointObjectRecord:
    payload: bytes = b""
    checkpoint_objects: tuple[CheckpointObjectRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            raise TypeError("component record payload must be opaque bytes")


@dataclass(frozen=True, slots=True)
class WorkerSaveResult:
    record: CheckpointObjectRecord | None = None
    commit_metadata: bytes = b""

    def __post_init__(self) -> None:
        if not isinstance(self.commit_metadata, bytes):
            raise TypeError("prepared checkpoint metadata must be opaque bytes")


@dataclass(frozen=True, slots=True)
class PartitionRecord:
    partition_key: str
    source_rank: int | None
    record: CheckpointObjectRecord

    def __post_init__(self) -> None:
        if not self.partition_key:
            raise ValueError("partition record requires partition_key")
        if self.source_rank is not None and self.source_rank < 0:
            raise ValueError("partition source_rank must be non-negative")


@dataclass(frozen=True, slots=True)
class CheckpointComponentPublication:
    component_id: str
    shared_records: tuple[CheckpointObjectRecord, ...] = ()
    partitions: tuple[PartitionRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component publication requires component_id")
        keys = [partition.partition_key for partition in self.partitions]
        if len(keys) != len(set(keys)):
            raise ValueError("component publication partition keys must be unique")
        records = (
            *self.shared_records,
            *(partition.record for partition in self.partitions),
        )
        if any(
            checkpoint_object.component_id != self.component_id
            for record in records
            for checkpoint_object in record.checkpoint_objects
        ):
            raise ValueError(
                "checkpoint_object component does not match its publication"
            )
        versions_by_checkpoint_object: dict[str | bytes, set[str | bytes]] = {}
        for record in records:
            for checkpoint_object in record.checkpoint_objects:
                versions_by_checkpoint_object.setdefault(
                    checkpoint_object.checkpoint_object_id, set()
                ).add(checkpoint_object.version_id)
        if any(
            len(versions) != 1 for versions in versions_by_checkpoint_object.values()
        ):
            raise ValueError(
                "one checkpoint_object publication must use one logical checkpoint_object version"
            )


@dataclass(frozen=True, slots=True)
class SpmdCheckpoint:
    source_topology: TopologySnapshot
    runtime_state: Mapping[str, Any]
    components: Mapping[str, CheckpointComponentPublication]
    driver_components: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            component_id != publication.component_id
            for component_id, publication in self.components.items()
        ):
            raise ValueError("component publication does not match its mapping key")
        if not isinstance(self.driver_components, Mapping):
            raise TypeError("driver_components must be a mapping")


@dataclass(frozen=True, slots=True)
class StepReference:
    step: int

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or self.step < 0:
            raise ValueError("step reference must be non-negative")


@dataclass(frozen=True, slots=True)
class NameReference:
    name: str
    retain_forever: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name reference requires a name")


@dataclass(frozen=True, slots=True)
class CheckpointIdReference:
    checkpoint_id: str

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint reference requires checkpoint_id")


CheckpointReference = StepReference | NameReference | CheckpointIdReference


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    checkpoint_id: str
    attempt_id: str
    coordinator_generation: int
    cut: CheckpointCut
    checkpoint_sequence: int
    references: tuple[CheckpointReference, ...]
    framework_state: Mapping[str, Any]
    spmds: Mapping[str, SpmdCheckpoint]

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint manifest requires checkpoint_id")
        if self.checkpoint_sequence < 0:
            raise ValueError("checkpoint_sequence must be non-negative")
        if not self.attempt_id:
            raise ValueError("checkpoint manifest requires attempt_id")
        if self.coordinator_generation < 0:
            raise ValueError("checkpoint manifest coordinator generation is invalid")


@dataclass(frozen=True, slots=True)
class CheckpointAssembly:
    cut: CheckpointCut
    framework_state: Mapping[str, Any]
    spmds: Mapping[str, SpmdCheckpoint]


@dataclass(frozen=True, slots=True)
class RestoreRequest:
    checkpoint_id: str
    source_topology: TopologySnapshot
    current_topology: TopologySnapshot


@dataclass(frozen=True, slots=True)
class SaveCheckpointRequest:
    attempt_id: str
    cut: CheckpointCut
    references: tuple[CheckpointReference, ...]

    def __post_init__(self) -> None:
        if not self.attempt_id:
            raise ValueError("save checkpoint request requires attempt_id")
        if not self.references:
            raise ValueError("save checkpoint request requires at least one reference")
        step_references = tuple(
            reference.step
            for reference in self.references
            if isinstance(reference, StepReference)
        )
        if step_references and self.cut.position is None:
            raise ValueError("step references require a durable checkpoint position")
        if any(step != self.cut.position for step in step_references):
            raise ValueError("step reference must match the checkpoint cut")
        names = tuple(
            reference.name
            for reference in self.references
            if isinstance(reference, NameReference)
        )
        if len(names) != len(set(names)):
            raise ValueError(
                "save checkpoint request contains duplicate named references"
            )
        if any(
            isinstance(reference, CheckpointIdReference)
            for reference in self.references
        ):
            raise ValueError(
                "checkpoint references select existing manifests and cannot publish new ones"
            )

    @classmethod
    def for_step(
        cls,
        step: int,
        *,
        names: tuple[str, ...] = (),
    ) -> SaveCheckpointRequest:
        attempt_id = uuid.uuid4().hex
        references: tuple[CheckpointReference, ...] = (
            StepReference(step),
            *(NameReference(name) for name in names),
        )
        return cls(
            attempt_id=attempt_id,
            cut=CheckpointCut.for_step(uuid.uuid4().hex, step),
            references=references,
        )


@dataclass(frozen=True, slots=True)
class CheckpointHold:
    lease_id: str
    checkpoint_object: CheckpointObjectRef
    lease_holder_id: str
    expires_at: float
    generation: int
    request_id: str

    @property
    def expired(self) -> bool:
        return self.expires_at <= time.time()


@dataclass(frozen=True, slots=True)
class CleanupReservationRequest:
    permit: CheckpointCommitPermit
    checkpoint_object_id: str
    ordered_candidates: tuple[CheckpointObjectRef, ...]
    required_count: int
    protected_versions: frozenset[str | bytes] = frozenset()

    def __post_init__(self) -> None:
        if not self.checkpoint_object_id:
            raise ValueError("cleanup reservation requires checkpoint_object")
        if self.required_count < 0:
            raise ValueError("retention required_count must be non-negative")


@dataclass(frozen=True, slots=True)
class CleanupReservation:
    attempt_id: str
    checkpoint_object_id: str
    claimed: tuple[CheckpointObjectRef, ...]
    generation: int
    owner_handle: bytes = b""

    def __post_init__(self) -> None:
        if not isinstance(self.owner_handle, bytes):
            raise TypeError("cleanup reservation owner_handle must be bytes")


@dataclass(frozen=True, slots=True)
class CleanupReservationBinding:
    artifact_id: str
    spmd_id: str
    component_id: str
    reservation: CleanupReservation

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.spmd_id or not self.component_id:
            raise ValueError(
                "cleanup reservation binding requires artifact, SPMD, and component IDs"
            )
        if any(
            checkpoint_object.component_id != self.component_id
            for checkpoint_object in self.reservation.claimed
        ):
            raise ValueError("cleanup reservation object belongs to another component")


@dataclass(frozen=True, slots=True)
class CheckpointCleanupIntent:
    artifact_id: str
    reservations: tuple[CleanupReservationBinding, ...]

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("checkpoint cleanup intent requires artifact ID")
        if not self.reservations:
            raise ValueError("checkpoint cleanup intent requires reservations")
        keys = [
            (
                binding.artifact_id,
                binding.spmd_id,
                binding.component_id,
                binding.reservation.checkpoint_object_id,
            )
            for binding in self.reservations
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "checkpoint cleanup intent contains duplicate reservations"
            )
        if any(
            binding.artifact_id != self.artifact_id for binding in self.reservations
        ):
            raise ValueError(
                "checkpoint cleanup intent reservation belongs to another artifact"
            )
