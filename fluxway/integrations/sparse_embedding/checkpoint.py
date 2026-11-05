"""Checkpoint integration for the optional ``sparse_embedding`` package."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, UTC
from pathlib import Path
from typing import Protocol


from sparse_embedding.checkpoint import (  # type: ignore
    CheckpointSaveMode,
    GravitinoLanceStorage,
    LocalLanceStorage,
    PartitionSpec,
    RestorePlan,
    resolve_effective_save_mode,
)
from sparse_embedding.checkpoint.gravitino_lance import SdkConfig  # type: ignore
from sparse_embedding.checkpoint.base import (  # type: ignore
    CheckpointCommitAuthority,
    CheckpointParticipant,
    CheckpointSaveIdentity,
    CheckpointConflictError,
    CheckpointLeaseRequest as SparseLeaseRequest,
    CleanupReservation as SparseCleanupReservation,
    CheckpointLease as SparseCheckpointLease,
    SparseCheckpointStorage,
)

from fluxway.common.io_lane import IOLane

from fluxway.common.types.checkpoint import (
    CheckpointObjectRef,
    CheckpointObjectSpec,
    CheckpointComponentPublication,
    CheckpointObjectRecord,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CheckpointComponentSpec,
    CheckpointCommitPermit,
    SaveSession,
    CommitSaveSession,
    RestoreSession,
    CheckpointRestoreSpec,
    CheckpointSaveScope,
    CheckpointSaveSpec,
    WorkerSaveResult,
    LeaseConflictError,
    CheckpointCleanupBlocked,
    CheckpointHold,
    CleanupReservation,
    CleanupReservationRequest,
    RestoreTopologyMode,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponentRegistration,
    CheckpointComponentBinding,
    CheckpointComponentLifecycle,
    CheckpointObjectCleanup,
    CheckpointObjectLifecycle,
    CheckpointComponentRestore,
    CheckpointComponentSave,
)


class SparseCheckpointStorageFactory(Protocol):
    def __call__(
        self,
        runtime: CheckpointRuntime,
        component_id: str,
    ) -> SparseCheckpointStorage: ...


@dataclass(frozen=True, slots=True)
class LocalLanceStorageFactory:
    max_storage_bytes: int | None = None

    def __call__(
        self,
        runtime: CheckpointRuntime,
        component_id: str,
    ) -> SparseCheckpointStorage:
        root = (
            Path(runtime.repository_root)
            / "application-checkpoint_objects"
            / runtime.spmd_id
            / component_id
        )
        return LocalLanceStorage(root, max_storage_bytes=self.max_storage_bytes)


@dataclass(frozen=True, slots=True)
class GravitinoLanceStorageFactory:
    config: SdkConfig
    namespace: tuple[str, str]
    location: str
    max_storage_bytes: int | None = None
    lock_timeout_seconds: float = 60.0
    lock_ttl_seconds: float = 300.0

    def __call__(
        self,
        runtime: CheckpointRuntime,
        component_id: str,
    ) -> SparseCheckpointStorage:
        storage_component = f"{runtime.spmd_id}.{component_id}"
        storage_location = (
            f"{self.location.rstrip('/')}/{runtime.spmd_id}/{component_id}"
        )
        return GravitinoLanceStorage(
            self.config,
            self.namespace,
            component_id=storage_component,
            location=storage_location,
            max_storage_bytes=self.max_storage_bytes,
            lock_timeout_seconds=self.lock_timeout_seconds,
            lock_ttl_seconds=self.lock_ttl_seconds,
            provision=runtime.role is CheckpointRuntimeRole.CONTROL,
        )


class SparseEmbeddingCheckpointComponent(
    CheckpointComponentBinding,
    CheckpointComponentLifecycle,
    CheckpointComponentSave,
    CheckpointComponentRestore,
    CheckpointObjectLifecycle,
    CheckpointObjectCleanup,
):
    def __init__(
        self,
        storage_factory: SparseCheckpointStorageFactory,
        *,
        component_id: str = "embedding",
        output_path: str | None = None,
        load_optimizer_state: bool = True,
        retention_domain: str | None = None,
        mode: CheckpointSaveMode = CheckpointSaveMode.INCREMENTAL,
        initial_mode: CheckpointSaveMode = CheckpointSaveMode.FULL,
        include_page_images: bool = False,
        lineage_name: str | None = None,
        partition_spec: PartitionSpec | None = None,
    ) -> None:
        if not callable(storage_factory):
            raise TypeError("storage_factory must be callable")
        if not isinstance(load_optimizer_state, bool):
            raise TypeError("load_optimizer_state must be bool")
        if not isinstance(mode, CheckpointSaveMode):
            raise TypeError("mode must be a CheckpointSaveMode")
        if initial_mode is not CheckpointSaveMode.FULL:
            raise ValueError("initial_mode must be CheckpointSaveMode.FULL")
        if not isinstance(include_page_images, bool):
            raise TypeError("include_page_images must be bool")
        if lineage_name is None:
            lineage_name = component_id
        if not isinstance(lineage_name, str) or not lineage_name.strip():
            raise ValueError("lineage_name cannot be empty")
        if partition_spec is None:
            partition_spec = PartitionSpec.modulo()
        if not isinstance(partition_spec, PartitionSpec):
            raise TypeError("partition_spec must be a PartitionSpec")
        self._descriptor = CheckpointComponentSpec(
            component_id=component_id,
            retention_domain=retention_domain,
            checkpoint_objects=(CheckpointObjectSpec(component_id),),
            save=CheckpointSaveSpec(
                prepare_scope=CheckpointSaveScope.LEADER,
                commit_scope=CheckpointSaveScope.LEADER,
            ),
            restore=CheckpointRestoreSpec(
                topology=RestoreTopologyMode.LOGICAL_PARTITION_REMAP
            ),
            output_path=output_path,
        )
        self._bound_spmd_id: str | None = None
        self._storage_factory = storage_factory
        self._load_optimizer_state = load_optimizer_state
        self._mode = mode
        self._initial_mode = initial_mode
        self._include_page_images = include_page_images
        self._lineage_name = lineage_name
        self._partition_spec = partition_spec
        self._active_parent_handle: bytes | None = None
        self._runtime: CheckpointRuntime | None = None
        self._storage: SparseCheckpointStorage | None = None
        self._io_lane = IOLane("sparse-checkpoint", capacity=1)

    @property
    def descriptor(self) -> CheckpointComponentSpec:
        return self._descriptor

    @property
    def bound_spmd_id(self) -> str | None:
        return self._bound_spmd_id

    def bind_to_spmd(self, spmd_id: str) -> None:
        if self._bound_spmd_id is not None:
            raise ValueError(
                f"Sparse checkpoint component {self.descriptor.component_id!r} "
                f"is already bound to SPMD {self._bound_spmd_id!r}; "
                f"cannot bind it again to {spmd_id!r}. Create a fresh component "
                "for each node. FunctionalGraph does this automatically."
            )
        if (
            not isinstance(spmd_id, str)
            or not spmd_id
            or spmd_id in {".", ".."}
            or any(char in spmd_id for char in ("/", "\\", "\x00"))
        ):
            raise ValueError(
                f"Cannot bind sparse checkpoint component "
                f"{self.descriptor.component_id!r} to SPMD {spmd_id!r}: "
                "the node name must be a non-empty single path segment without "
                "slashes, backslashes, NUL, or '.'/'..'. Use a name such as "
                "'UserTable'; it also identifies the checkpoint storage namespace."
            )
        if self._descriptor.output_path is None:
            self._descriptor = replace(
                self._descriptor, output_path=f"sparse/{spmd_id}"
            )
        self._bound_spmd_id = spmd_id

    def registration(self) -> CheckpointComponentRegistration:
        return CheckpointComponentRegistration(
            component=self,
            component_lifecycle=self,
            save=self,
            restore=self,
            object_lifecycle=self,
            object_cleanup=self,
        )

    async def open(self, runtime: CheckpointRuntime) -> None:
        if self._bound_spmd_id is None:
            self.bind_to_spmd(runtime.spmd_id)
        else:
            self.require_spmd_binding(runtime.spmd_id)
        if self._runtime is not None:
            raise RuntimeError(
                f"sparse checkpoint component {self.descriptor.component_id!r} is open"
            )
        try:
            storage = await self._io(
                self._storage_factory,
                runtime,
                self.descriptor.component_id,
            )
        except BaseException as primary:
            try:
                await self._io_lane.close()
            except BaseException as cleanup:
                primary.add_note(
                    f"sparse checkpoint lane cleanup failed: {type(cleanup).__name__}: {cleanup}"
                )
            else:
                self._io_lane = IOLane("sparse-checkpoint", capacity=1)
            raise
        self._storage = storage
        self._runtime = runtime

    async def prepare_save(self, session: SaveSession) -> None:
        self._require_worker(session)
        identity = self._identity(session.commit_permit)
        await self._io(self._require_storage().prepare_checkpoint, identity)

    async def save(self, session: SaveSession) -> WorkerSaveResult:
        self._require_worker(session)
        identity = self._identity(session.commit_permit)
        participant = CheckpointParticipant(
            session.rank, session.source_topology.world_size
        )
        store = session.instance.store
        parent_handle = (
            self._active_parent_handle
            if self._active_parent_handle is not None
            else await self._io(
                self._require_storage().resolve_lineage_head_handle,
                self._lineage_name,
            )
        )
        requires_full = await self._io(
            self._require_storage().requires_full_checkpoint,
            store,
        )
        configured_mode = self._initial_mode if parent_handle is None else self._mode
        save_mode, effective_parent_handle = resolve_effective_save_mode(
            configured_mode,
            requires_full,
            parent_handle,
        )

        if save_mode is CheckpointSaveMode.INCREMENTAL:
            baseline = await self._io(store.checkpoint_baseline)
            if not isinstance(baseline, str) or not baseline:
                raise RuntimeError(
                    "incremental checkpoint requires a non-empty store baseline"
                )

        fragment = await self._io(
            self._require_storage().write_checkpoint_fragment_bytes,
            store,
            identity,
            participant,
            include_page_images=self._include_page_images,
            mode=save_mode,
            parent_handle=effective_parent_handle,
        )
        return WorkerSaveResult(commit_metadata=fragment)

    async def finish_save(self, session: SaveSession) -> None:
        self._require_worker(session)
        identity = self._identity(session.commit_permit)
        await self._io(
            self._require_storage().finalize_checkpoint,
            identity,
        )

    async def commit_save(
        self, session: CommitSaveSession
    ) -> CheckpointComponentPublication:
        self._require_worker(session)
        identity = self._identity(session.commit_permit)
        fragments = tuple(
            record.commit_metadata
            for _, record in sorted(session.worker_results.items())
        )
        committed = await self._io(
            self._require_storage().commit_checkpoint_bytes,
            identity,
            fragments,
        )
        storage_locator = committed.handle
        # Publish the owner-private inference reference before the control-plane
        # acknowledgement.  The inference reader only receives this path and
        # never interprets the checkpoint locator above.
        if session.staging_output_path is None:
            raise RuntimeError("Fluxway did not bind sparse component output path")
        weight_root = Path(session.staging_output_path)
        await self._io(weight_root.mkdir, parents=True, exist_ok=True)
        await self._io(
            self._require_storage().publish_weight_handle,
            str(weight_root),
            storage_locator,
        )
        version_id = committed.version_id
        checkpoint_object = CheckpointObjectRef(
            component_id=self.descriptor.component_id,
            checkpoint_object_id=self.descriptor.component_id,
            version_id=version_id,
            storage_locator=storage_locator,
        )
        return CheckpointComponentPublication(
            component_id=self.descriptor.component_id,
            shared_records=(
                CheckpointObjectRecord(checkpoint_objects=(checkpoint_object,)),
            ),
        )

    async def rollback_save(self, session: SaveSession) -> None:
        self._require_worker(session)
        identity = self._identity(session.commit_permit)
        if session.rank == 0:
            await self._io(self._require_storage().abort_checkpoint, identity)

    async def abort_save(self, permit: CheckpointCommitPermit) -> None:
        self._require_control()
        await self._io(self._require_storage().abort_checkpoint, self._identity(permit))

    async def publish_save(self, session: SaveSession) -> None:
        self._require_worker(session)
        if session.staging_output_path is None:
            raise RuntimeError("Fluxway did not bind sparse component output path")
        staging = Path(session.staging_output_path)
        if not await self._io(staging.is_dir):
            raise FileNotFoundError(
                f"sparse weight staging directory is missing: {staging}"
            )
        handle = await self._io(
            self._require_storage().read_published_weight_handle,
            staging,
        )
        await self._io(
            self._require_storage().acknowledge_store_mutations,
            session.instance.store,
            handle,
            session.rank,
        )
        self._active_parent_handle = handle

    async def acknowledge_save(self, permit: CheckpointCommitPermit) -> None:
        self._require_control()
        identity = self._identity(permit)
        await self._io(
            self._require_storage().acknowledge_checkpoint,
            identity,
        )

    async def restore(self, session: RestoreSession) -> None:
        self._require_worker(session)
        publication = session.component_checkpoint
        if publication.component_id != self.descriptor.component_id:
            raise ValueError("sparse publication belongs to another component")
        if len(publication.shared_records) != 1:
            raise ValueError("sparse publication requires one shared record")
        record = publication.shared_records[0]
        if len(record.checkpoint_objects) != 1:
            raise ValueError("sparse publication requires one checkpoint_object")
        checkpoint_object = record.checkpoint_objects[0]
        if checkpoint_object.component_id != self.descriptor.component_id:
            raise ValueError("sparse checkpoint_object belongs to another component")
        if checkpoint_object.checkpoint_object_id != self.descriptor.component_id:
            raise ValueError(
                "sparse checkpoint_object belongs to another checkpoint_object"
            )
        self._validate_checkpoint_object(checkpoint_object)
        restore_options = session.restore_options
        if restore_options is None:
            plan = RestorePlan.from_partition(
                self._partition_spec,
                session.restore_topology.world_size,
                session.rank,
                key_dtype=session.instance.store.key_dtype(),
            )
            restore_payload = plan.encode()
        elif isinstance(restore_options, bytes):
            restore_payload = restore_options
        elif isinstance(restore_options, RestorePlan):
            restore_payload = restore_options.encode()
        else:
            raise TypeError(
                f"restore_options must be bytes, RestorePlan, or None; got {type(restore_options).__name__}"
            )
        await self._io(
            self._require_storage().load_component_checkpoint,
            session.instance.store,
            checkpoint_object.storage_locator,
            restore_payload,
            load_optimizer_state=self._load_optimizer_state,
        )
        self._active_parent_handle = checkpoint_object.storage_locator

    async def hold_checkpoint_object(
        self,
        checkpoint_object: CheckpointObjectRef,
        lease_holder_id: str,
        ttl_seconds: float,
        request_id: str,
    ) -> CheckpointHold:
        self._require_control()
        self._validate_checkpoint_object(checkpoint_object)
        try:
            lease = await self._io(
                self._require_storage().acquire_checkpoint_lease,
                checkpoint_object.storage_locator,
                SparseLeaseRequest(lease_holder_id, ttl_seconds, request_id),
            )
        except CheckpointConflictError as error:
            raise LeaseConflictError(str(error)) from error
        return self._from_sparse_lease(checkpoint_object, lease)

    async def renew_checkpoint_hold(
        self, lease: CheckpointHold, ttl_seconds: float
    ) -> CheckpointHold:
        self._require_control()
        try:
            renewed = await self._io(
                self._require_storage().renew_checkpoint_lease,
                self._to_sparse_lease(lease),
                ttl_seconds,
            )
        except CheckpointConflictError as error:
            raise LeaseConflictError(str(error)) from error
        return self._from_sparse_lease(lease.checkpoint_object, renewed)

    async def release_checkpoint_hold(self, lease: CheckpointHold) -> None:
        self._require_control()
        try:
            await self._io(
                self._require_storage().release_checkpoint_lease,
                self._to_sparse_lease(lease),
            )
        except CheckpointConflictError as error:
            raise LeaseConflictError(str(error)) from error

    async def reserve_cleanup(
        self, request: CleanupReservationRequest
    ) -> CleanupReservation:
        self._require_control()
        candidates = tuple(
            checkpoint_object
            for checkpoint_object in request.ordered_candidates
            if checkpoint_object.version_id not in request.protected_versions
        )
        for checkpoint_object in candidates:
            self._validate_checkpoint_object(checkpoint_object)
        storage_locators = tuple(item.storage_locator for item in candidates)
        reservation_id = self._reservation_id(
            request.permit.attempt_id,
            request.permit.coordinator_generation,
            request.checkpoint_object_id,
        )
        try:
            sparse_reservation = await self._io(
                self._require_storage().reserve_cleanup,
                reservation_id,
                storage_locators,
                request.required_count,
            )
        except CheckpointConflictError as error:
            raise CheckpointCleanupBlocked(str(error)) from error
        by_storage_locator = dict(zip(storage_locators, candidates, strict=True))
        return CleanupReservation(
            attempt_id=request.permit.attempt_id,
            checkpoint_object_id=request.checkpoint_object_id,
            claimed=tuple(
                by_storage_locator[item] for item in sparse_reservation.handles
            ),
            generation=request.permit.coordinator_generation,
            owner_handle=sparse_reservation.owner_handle,
        )

    async def commit_cleanup(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None:
        self._require_control()
        sparse = self._sparse_reservation(reservation, permit)
        await self._io(self._require_storage().commit_cleanup, sparse)

    async def rollback_cleanup(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None:
        self._require_control()
        sparse = self._sparse_reservation(reservation, permit)
        await self._io(self._require_storage().rollback_cleanup, sparse)

    async def cleanup(
        self,
        reservation: CleanupReservation,
    ) -> None:
        self._require_control()
        sparse = self._sparse_reservation_from_reservation(reservation)
        for _, storage_locator in zip(reservation.claimed, sparse.handles, strict=True):
            await self._io(
                self._require_storage().cleanup_checkpoint,
                sparse,
                storage_locator,
            )

    async def compact_checkpoints(self) -> None:
        self._require_control()
        handle = await self._io(
            self._require_storage().resolve_lineage_head_handle,
            self._lineage_name,
        )
        if handle is None:
            return
        await self._io(self._require_storage().compact_checkpoint, handle)

    async def compact_checkpoint(
        self,
        checkpoint_object: CheckpointObjectRef,
    ) -> None:
        self._require_control()
        self._validate_checkpoint_object(checkpoint_object)
        await self._io(
            self._require_storage().compact_checkpoint,
            checkpoint_object.storage_locator,
        )

    async def close(self) -> None:
        if self._runtime is None or self._storage is None:
            raise RuntimeError(
                f"sparse checkpoint component {self.descriptor.component_id!r} is not open"
            )
        storage = self._storage
        primary: BaseException | None = None
        try:
            await self._io(storage.close)
        except BaseException as error:
            primary = error
        self._storage = None
        self._runtime = None
        try:
            await self._io_lane.close()
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note(
                    f"sparse checkpoint lane close failed: {type(error).__name__}: {error}"
                )
        else:
            self._io_lane = IOLane("sparse-checkpoint", capacity=1)
        if primary is not None:
            raise primary

    async def _io(self, operation, /, *args, **kwargs):
        return await self._io_lane.call(operation, *args, **kwargs)

    def _require_open(
        self, context: SaveSession | RestoreSession | None = None
    ) -> None:
        runtime = self._runtime
        if runtime is None or self._storage is None:
            raise RuntimeError(
                f"sparse checkpoint component {self.descriptor.component_id!r} is not open"
            )
        if context is not None and context.component_id != self.descriptor.component_id:
            raise ValueError("checkpoint context belongs to another component")

    def _require_worker(self, context: SaveSession | RestoreSession) -> None:
        self._require_open(context)
        assert self._runtime is not None
        if self._runtime.role is not CheckpointRuntimeRole.WORKER:
            raise RuntimeError(
                "sparse checkpoint save and restore require a worker runtime"
            )

    def _require_control(self) -> None:
        runtime = self._runtime
        if runtime is None or self._storage is None:
            raise RuntimeError(
                f"sparse checkpoint component {self.descriptor.component_id!r} is not open"
            )
        if runtime.role is not CheckpointRuntimeRole.CONTROL:
            raise RuntimeError("sparse cleanup requires a control runtime")

    def _require_storage(self) -> SparseCheckpointStorage:
        if self._storage is None:
            raise RuntimeError(
                f"sparse checkpoint component {self.descriptor.component_id!r} is not open"
            )
        return self._storage

    def _validate_checkpoint_object(
        self, checkpoint_object: CheckpointObjectRef
    ) -> None:
        if checkpoint_object.component_id != self.descriptor.component_id:
            raise ValueError("sparse checkpoint_object belongs to another component")
        if checkpoint_object.checkpoint_object_id != self.descriptor.component_id:
            raise ValueError(
                "sparse checkpoint_object belongs to another checkpoint_object"
            )
        if not checkpoint_object.storage_locator:
            raise ValueError("sparse checkpoint_object requires an owner handle")

    @staticmethod
    def _from_sparse_lease(
        checkpoint_object: CheckpointObjectRef, lease: SparseCheckpointLease
    ) -> CheckpointHold:
        return CheckpointHold(
            lease_id=lease.lease_id,
            checkpoint_object=checkpoint_object,
            lease_holder_id=lease.lease_holder_id,
            expires_at=lease.expires_at.timestamp(),
            generation=lease.generation,
            request_id=lease.request_id,
        )

    def _to_sparse_lease(self, lease: CheckpointHold) -> SparseCheckpointLease:
        return SparseCheckpointLease(
            lease_id=lease.lease_id,
            lease_holder_id=lease.lease_holder_id,
            expires_at=datetime.fromtimestamp(lease.expires_at, UTC),
            generation=lease.generation,
            request_id=lease.request_id,
        )

    def _sparse_reservation(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> SparseCleanupReservation:
        if (
            reservation.attempt_id != permit.attempt_id
            or reservation.generation != permit.coordinator_generation
        ):
            raise PermissionError(
                "cleanup reservation does not match commit_save permit"
            )
        return self._sparse_reservation_from_reservation(reservation)

    def _sparse_reservation_from_reservation(
        self,
        reservation: CleanupReservation,
    ) -> SparseCleanupReservation:
        return SparseCleanupReservation(
            self._reservation_id(
                reservation.attempt_id,
                reservation.generation,
                reservation.checkpoint_object_id,
            ),
            reservation.owner_handle,
            tuple(
                checkpoint_object.storage_locator
                for checkpoint_object in reservation.claimed
            ),
        )

    def _reservation_id(
        self, attempt_id: str, generation: int, checkpoint_object_id: str
    ) -> str:
        return ":".join(
            (
                "fluxway",
                self.descriptor.component_id,
                checkpoint_object_id,
                str(generation),
                attempt_id,
            )
        )

    def _identity(self, permit: CheckpointCommitPermit) -> CheckpointSaveIdentity:
        self._require_open()
        assert self._runtime is not None
        return CheckpointSaveIdentity(
            checkpoint_name=self._lineage_name,
            save_attempt=permit.attempt_id,
            commit_authority=CheckpointCommitAuthority(
                scope=f"fluxway:{self._runtime.spmd_id}:{self.descriptor.component_id}",
                generation=permit.coordinator_generation,
                authority_id=hashlib.sha256(permit.token).hexdigest(),
            ),
        )


async def compact_sparse_checkpoints(
    repository_root: str | Path,
    spmd_id: str,
    *,
    lineage_name: str,
    component_id: str = "embedding",
    storage_factory: SparseCheckpointStorageFactory | None = None,
) -> None:
    runtime = CheckpointRuntime(
        role=CheckpointRuntimeRole.CONTROL,
        repository_root=str(repository_root),
        spmd_id=spmd_id,
    )
    actual_factory = storage_factory or LocalLanceStorageFactory()
    component = SparseEmbeddingCheckpointComponent(
        actual_factory,
        component_id=component_id,
        mode=CheckpointSaveMode.INCREMENTAL,
        initial_mode=CheckpointSaveMode.FULL,
        lineage_name=lineage_name,
    )
    component.bind_to_spmd(spmd_id)
    await component.open(runtime)
    try:
        await component.compact_checkpoints()
    finally:
        await component.close()


__all__ = [
    "GravitinoLanceStorageFactory",
    "LocalLanceStorageFactory",
    "SparseCheckpointStorageFactory",
    "SparseEmbeddingCheckpointComponent",
    "compact_sparse_checkpoints",
]
