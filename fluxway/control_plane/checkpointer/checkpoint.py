"""Controller-side global checkpoint orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from fluxway.common.types.checkpoint import (
    CheckpointCommitPermit,
    CheckpointDeadline,
    CheckpointObjectRef,
    CheckpointObjectCleanedUpError,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CheckpointCut,
    CheckpointManifest,
    SaveCheckpointRequest,
    CheckpointIdReference,
    CheckpointId,
    NameReference,
    StepReference,
    CheckpointHold,
    CheckpointTransactionState,
    CheckpointRetentionPolicy,
    RestoreOptions,
    validate_restore_options,
)
from fluxway.common.types.configs import (
    DEFAULT_SNAPSHOT_TIMEOUT_SECS,
    DEFAULT_INACTIVITY_TIMEOUT_SECS,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
)
from fluxway.control_plane.checkpointer.repository import (
    CheckpointTransactionRecord,
)
from fluxway.control_plane.checkpointer.repository_session import (
    CheckpointRepositorySession,
)
from fluxway.control_plane.checkpointer.retention import (
    CheckpointRetentionCoordinator,
)

logger = logging.getLogger(__name__)


class FluxwayCheckpointer:
    @classmethod
    async def create(
        cls, controller: Any, repository_dir: str, **configuration: Any
    ) -> FluxwayCheckpointer:
        session = await CheckpointRepositorySession.open(repository_dir)
        try:
            return cls(controller, session, **configuration)
        except BaseException:
            await session.close()
            raise

    def __init__(
        self,
        controller: Any,
        repository_session: CheckpointRepositorySession,
        *,
        interval_steps: int,
        snapshot_timeout_s: float | None = None,
        inactivity_timeout_s: float | None = None,
        retention_policy: CheckpointRetentionPolicy = CheckpointRetentionPolicy(),
        retention_domains: Mapping[str, CheckpointRetentionPolicy] | None = None,
        component_registries: Mapping[str, CheckpointComponentRegistry] | None = None,
        restore_scope: tuple[str, ...] | None = None,
        restore_spmd_bindings: Mapping[str, str] | None = None,
    ) -> None:
        if interval_steps < 1:
            raise ValueError("interval_steps must be positive")
        if snapshot_timeout_s is not None:
            if snapshot_timeout_s <= 0:
                raise ValueError("snapshot_timeout_s must be positive")
            self._snapshot_timeout_s = snapshot_timeout_s
        else:
            self._snapshot_timeout_s = float(
                os.environ.get(
                    "FLUXWAY_SNAPSHOT_TIMEOUT_SECS",
                    str(DEFAULT_SNAPSHOT_TIMEOUT_SECS),
                )
            )
        if inactivity_timeout_s is not None:
            if inactivity_timeout_s <= 0:
                raise ValueError("inactivity_timeout_s must be positive")
            self._inactivity_timeout_s = inactivity_timeout_s
        else:
            self._inactivity_timeout_s = DEFAULT_INACTIVITY_TIMEOUT_SECS
        if not isinstance(retention_policy, CheckpointRetentionPolicy):
            raise TypeError("retention_policy must be CheckpointRetentionPolicy")
        self._retention_policy = retention_policy
        self._retention_domains = dict(retention_domains or {})
        if any(not domain for domain in self._retention_domains):
            raise ValueError("retention_domains keys must be non-empty")
        if any(
            not isinstance(policy, CheckpointRetentionPolicy)
            for policy in self._retention_domains.values()
        ):
            raise TypeError(
                "retention_domains values must be CheckpointRetentionPolicy"
            )
        self._controller = controller
        self._repository = repository_session.repository
        self._coordinator_generation = repository_session.generation
        self._repository_dir = str(self._repository.root)
        self._interval_steps = interval_steps
        self._registries = dict(component_registries or {})
        self._validate_output_paths()
        self._validate_retention_capabilities()
        self._restore_scope = restore_scope
        self._restore_spmd_bindings = dict(restore_spmd_bindings or {})
        self._validate_restore_contract()
        self._repository_lane = repository_session.lane
        self._retention = CheckpointRetentionCoordinator(
            self._registries,
            self._repository,
            self._retention_policy,
            self._retention_domains,
            io_lane=self._repository_lane,
        )
        self._save_lock = asyncio.Lock()
        self._cleanup_lane_lock = asyncio.Lock()
        self._cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cleanup_failures: list[BaseException] = []
        self._cleanup_artifacts_inflight: set[str] = set()
        self._control_open = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def interval_steps(self) -> int:
        return self._interval_steps

    def _validate_output_paths(self) -> None:
        outputs: list[tuple[tuple[str, ...], str, str]] = []
        for spmd_id, registry in self._registries.items():
            registry.validate_spmd_binding(spmd_id)
            for registration in registry:
                descriptor = registration.descriptor
                if descriptor.output_path is not None:
                    outputs.append(
                        (
                            tuple(descriptor.output_path.split("/")),
                            spmd_id,
                            descriptor.component_id,
                        )
                    )
        outputs.sort()
        for previous, current in zip(outputs, outputs[1:], strict=False):
            previous_parts, previous_spmd, previous_component = previous
            current_parts, current_spmd, current_component = current
            if current_parts[: len(previous_parts)] == previous_parts:
                raise ValueError(
                    "Conflicting checkpoint output paths: "
                    f"SPMD {previous_spmd!r}, component {previous_component!r}, "
                    f"output_path={'/'.join(previous_parts)!r} overlaps "
                    f"SPMD {current_spmd!r}, component {current_component!r}, "
                    f"output_path={'/'.join(current_parts)!r}. "
                    "Each component exclusively owns its output directory and "
                    "all descendants under steps/<step>/. Remove explicit sparse "
                    "output_path overrides to use node-derived defaults, or "
                    "set distinct output_path values that do not overlap. "
                    "For multiple publishers in one SPMD, configure separate "
                    "output_path values for each component."
                )

    @property
    def snapshot_timeout_s(self) -> float:
        return self._snapshot_timeout_s

    @property
    def inactivity_timeout_s(self) -> float:
        return self._inactivity_timeout_s

    @property
    def retention_policy(self) -> CheckpointRetentionPolicy:
        return self._retention_policy

    @property
    def retention_domains(self) -> dict[str, CheckpointRetentionPolicy]:
        return dict(self._retention_domains)

    @property
    def restore_scope(self) -> tuple[str, ...] | None:
        return self._restore_scope

    @property
    def restore_spmd_bindings(self) -> dict[str, str]:
        return dict(self._restore_spmd_bindings)

    async def save(self, request: SaveCheckpointRequest) -> CheckpointManifest:
        if self._closed:
            raise RuntimeError("FluxwayCheckpointer is closed")
        if not isinstance(request, SaveCheckpointRequest):
            raise TypeError("FluxwayCheckpointer.save requires SaveCheckpointRequest")
        async with self._save_lock:
            self._require_open()
            return await self._save(request)

    async def _save(self, request: SaveCheckpointRequest) -> CheckpointManifest:
        deadline = CheckpointDeadline.start(self._snapshot_timeout_s)
        await deadline.run_phase(
            "fence_incomplete_transactions",
            self._fence_incomplete_transactions(),
        )
        await deadline.run_phase(
            "ensure_control_components",
            self._ensure_control_components(),
        )
        await deadline.run_phase(
            "repair_cleanup_intents",
            self._repair_cleanup_intents(),
        )
        checkpoint_id = CheckpointId.generate()
        permit = await deadline.run_phase(
            "issue_permit",
            self._repository_lane.call(
                self._repository.issue_commit_permit,
                request.attempt_id,
                self._coordinator_generation,
            ),
        )
        prepared_state: Any = None
        snapshot_assembled = False
        reservations: tuple[Any, ...] = ()
        cleanup_intents: tuple[Any, ...] = ()

        try:
            assembly = await deadline.run_phase(
                "request_snapshot",
                self._controller.request_snapshot(
                    str(checkpoint_id),
                    request.attempt_id,
                    permit,
                    request.cut,
                    self._repository_dir,
                    deadline=deadline,
                ),
            )
            snapshot_assembled = True
            self._validate_assembly_cut(request, assembly.cut)
            sequence = await deadline.run_phase(
                "next_sequence",
                self._repository_lane.call(self._repository.next_checkpoint_sequence),
            )
            manifest = CheckpointManifest(
                checkpoint_id=str(checkpoint_id),
                attempt_id=request.attempt_id,
                coordinator_generation=permit.coordinator_generation,
                cut=assembly.cut,
                checkpoint_sequence=sequence,
                references=request.references,
                framework_state=assembly.framework_state,
                spmds=assembly.spmds,
            )
            self._validate_manifest_progress(manifest)
            prepared_state = deadline.run_transition(
                self._controller.prepare_checkpoint_transaction,
                str(checkpoint_id),
                permit,
                manifest.cut,
                manifest.framework_state,
            )
            existing_manifests = await deadline.run_phase(
                "list_repository",
                self._repository_lane.call(self._repository.list),
            )
            reservations = await deadline.run_phase(
                "reserve_retention",
                self._retention.reserve(permit, manifest, existing_manifests),
            )
            await deadline.run_phase(
                "require_permit",
                self._repository_lane.call(
                    self._repository.require_commit_permit, permit
                ),
            )
            await deadline.run_phase(
                "publish_retention",
                self._retention.publish(reservations, permit),
            )
            cleanup_intents = deadline.run_transition(
                CheckpointRetentionCoordinator.cleanup_intents, reservations
            )

            # Stage transaction durably in repository BEFORE publishing manifest
            await deadline.run_phase(
                "stage_transaction",
                self._repository_lane.call(
                    self._repository.stage_transaction,
                    str(checkpoint_id),
                    manifest,
                    permit,
                ),
            )
            # Publish immutable manifest to repository
            await deadline.run_phase(
                "publish_manifest",
                self._repository_lane.call(
                    self._repository.publish,
                    manifest,
                    permit,
                    cleanup_intents,
                ),
            )

            # Durable commit decision: physical publication has started
            await deadline.run_phase(
                "decide_transaction",
                self._repository_lane.call(
                    self._repository.decide_transaction,
                    str(checkpoint_id),
                    permit,
                ),
            )
            # Controller & Drivers commit snapshot
            await deadline.run_phase(
                "publish_snapshot",
                self._controller.publish_snapshot(
                    str(checkpoint_id),
                    permit,
                    deadline=deadline,
                ),
            )
            deadline.run_transition(
                self._controller.apply_prepared_checkpoint_transaction,
                str(checkpoint_id),
                permit,
                manifest.cut,
                manifest.framework_state,
                prepared_state,
            )
            await deadline.run_phase(
                "mark_committed",
                self._repository_lane.call(
                    self._repository.mark_committed,
                    str(checkpoint_id),
                    permit,
                ),
            )
            await deadline.run_phase(
                "acknowledge_transaction",
                self._acknowledge_committed_transaction(manifest, permit),
            )
            await deadline.run_phase(
                "complete_transaction",
                self._repository_lane.call(
                    self._repository.complete_transaction,
                    str(checkpoint_id),
                    permit,
                ),
            )
            await deadline.run_phase(
                "evict_cleanup_artifacts",
                self._repository_lane.call(
                    self._repository.evict_manifests_for_cleanup_intents,
                    cleanup_intents,
                ),
            )
            self._schedule_cleanup_intents(cleanup_intents)
            logger.info(
                "checkpoint published snapshot=%s position=%s",
                checkpoint_id,
                manifest.cut.position,
            )
            return manifest
        except BaseException as primary_error:
            cleanup_errors: list[BaseException] = []
            try:
                durable_tx = await self._repository_lane.call(
                    self._repository.transaction, str(checkpoint_id)
                )
            except BaseException as read_error:
                await self._repository_lane.call(
                    self._repository.revoke_commit_permit, permit
                )
                raise BaseExceptionGroup(
                    "checkpoint save failed and transaction state is unreadable",
                    [primary_error, read_error],
                ) from None

            is_post_decision = durable_tx is not None and durable_tx.status in {
                CheckpointTransactionState.COMMIT_DECIDED,
                CheckpointTransactionState.COMMITTED,
            }

            if is_post_decision:
                assert durable_tx is not None
                logger.error(
                    "checkpoint transaction decided but post-decision step failed checkpoint=%s status=%s",
                    checkpoint_id,
                    durable_tx.status,
                    exc_info=primary_error,
                )
            else:
                if prepared_state is not None:
                    try:
                        self._controller.abort_checkpoint_transaction(
                            str(checkpoint_id), permit
                        )
                    except BaseException as error:
                        cleanup_errors.append(error)
                if reservations:
                    try:
                        await self._retention.rollback_save(reservations, permit)
                    except BaseException as error:
                        cleanup_errors.append(error)
                if snapshot_assembled:
                    try:
                        await self._controller.rollback_snapshot(
                            str(checkpoint_id), permit
                        )
                    except BaseException as error:
                        cleanup_errors.append(error)
                if (
                    durable_tx is not None
                    and durable_tx.status == CheckpointTransactionState.STAGED
                ):
                    try:
                        await self._repository_lane.call(
                            self._repository.abandon_transaction,
                            str(checkpoint_id),
                            permit,
                        )
                    except BaseException as error:
                        cleanup_errors.append(error)
                else:
                    try:
                        await self._repository_lane.call(
                            self._repository.revoke_commit_permit, permit
                        )
                    except BaseException as error:
                        cleanup_errors.append(error)

            if cleanup_errors:
                raise BaseExceptionGroup(
                    "checkpoint save and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise

    async def restore(
        self,
        reference: StepReference
        | NameReference
        | CheckpointIdReference
        | str = "latest",
        *,
        restore_options: RestoreOptions | None = None,
    ) -> int | None:
        if self._closed:
            raise RuntimeError("FluxwayCheckpointer is closed")
        validate_restore_options(restore_options)
        await self._fence_incomplete_transactions()
        await self._ensure_control_components()
        if reference == "latest":
            manifest = await self._repository_lane.call(self._repository.latest)
            if manifest is None:
                return None
        else:
            if isinstance(reference, str):
                raise TypeError(
                    "restore reference must be latest, StepReference, or CheckpointIdReference"
                )
            manifest = await self._repository_lane.call(
                self._repository.resolve, reference
            )
        self._validate_manifest_progress(manifest)
        manifest = self._select_restore_manifest(manifest)
        await self._assert_restorable(manifest)
        await self._controller.restore_snapshot(
            manifest.checkpoint_id,
            manifest,
            self._repository_dir,
            restore_options,
        )
        return manifest.cut.position

    async def restore_latest(
        self,
        *,
        restore_options: RestoreOptions | None = None,
    ) -> int | None:
        return await self.restore(restore_options=restore_options)

    async def restore_step(
        self,
        step: int,
        *,
        restore_options: RestoreOptions | None = None,
    ) -> int | None:
        return await self.restore(
            StepReference(step),
            restore_options=restore_options,
        )

    async def restore_named(
        self,
        name: str,
        *,
        restore_options: RestoreOptions | None = None,
    ) -> int | None:
        return await self.restore(
            NameReference(name),
            restore_options=restore_options,
        )

    async def restore_checkpoint(
        self,
        checkpoint_id: str,
        *,
        restore_options: RestoreOptions | None = None,
    ) -> int | None:
        return await self.restore(
            CheckpointIdReference(checkpoint_id),
            restore_options=restore_options,
        )

    async def list_steps(self) -> tuple[int, ...]:
        return await self._repository_lane.call(self._repository.list_steps)

    async def latest(self) -> CheckpointManifest | None:
        return await self._repository_lane.call(self._repository.latest)

    async def hold_checkpoint_object(
        self,
        spmd_id: str,
        checkpoint_object: CheckpointObjectRef,
        lease_holder_id: str,
        ttl_seconds: float,
        request_id: str,
    ) -> CheckpointHold:
        self._require_open()
        if not await self._repository_lane.call(
            self._repository.contains_checkpoint_object, spmd_id, checkpoint_object
        ):
            raise KeyError(
                "cannot lease an checkpoint_object that is not a live published reference"
            )
        await self._ensure_control_components()
        registration = self._require_component(spmd_id, checkpoint_object.component_id)
        if registration.object_lifecycle is None:
            raise RuntimeError(
                "component does not expose checkpoint object lifecycle capability"
            )
        return await registration.object_lifecycle.hold_checkpoint_object(
            checkpoint_object, lease_holder_id, ttl_seconds, request_id
        )

    async def renew_checkpoint_hold(
        self,
        spmd_id: str,
        lease: CheckpointHold,
        ttl_seconds: float,
    ) -> CheckpointHold:
        self._require_open()
        await self._ensure_control_components()
        registration = self._require_component(
            spmd_id, lease.checkpoint_object.component_id
        )
        if registration.object_lifecycle is None:
            raise RuntimeError(
                "component does not expose checkpoint object lifecycle capability"
            )
        return await registration.object_lifecycle.renew_checkpoint_hold(
            lease, ttl_seconds
        )

    async def release_checkpoint_hold(
        self, spmd_id: str, lease: CheckpointHold
    ) -> None:
        self._require_open()
        await self._ensure_control_components()
        registration = self._require_component(
            spmd_id, lease.checkpoint_object.component_id
        )
        if registration.object_lifecycle is None:
            raise RuntimeError(
                "component does not expose checkpoint object lifecycle capability"
            )
        await registration.object_lifecycle.release_checkpoint_hold(lease)

    async def repair_cleanup_intents(self) -> int:
        self._require_open()
        await self._ensure_control_components()
        return await self._repair_cleanup_intents()

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(self._observe_close)
        await asyncio.shield(self._close_task)

    @staticmethod
    def _observe_close(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _close(self) -> None:
        async with self._save_lock:
            await self._close_resources()

    async def _close_resources(self) -> None:
        errors: list[BaseException] = []
        if self._control_open:
            if self._cleanup_tasks:
                await asyncio.gather(
                    *tuple(self._cleanup_tasks), return_exceptions=True
                )
            errors.extend(self._cleanup_failures)
            for registry in reversed(tuple(self._registries.values())):
                try:
                    await registry.close_all()
                except BaseException as error:
                    errors.append(error)
        try:
            await self._repository_lane.close()
        except BaseException as error:
            errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("checkpoint close failed", errors)

    async def _ensure_control_components(self) -> None:
        if self._control_open:
            return
        opened: list[CheckpointComponentRegistry] = []
        try:
            for spmd_id, registry in self._registries.items():
                await registry.open_all(
                    CheckpointRuntime(
                        role=CheckpointRuntimeRole.CONTROL,
                        repository_root=self._repository_dir,
                        spmd_id=spmd_id,
                    )
                )
                opened.append(registry)
        except BaseException as primary_error:
            cleanup_errors: list[BaseException] = []
            for registry in reversed(opened):
                try:
                    await registry.close_all()
                except BaseException as error:
                    cleanup_errors.append(error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "control component open and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise
        self._control_open = True

    async def _acknowledge_committed_transaction(
        self, manifest: CheckpointManifest, permit: CheckpointCommitPermit
    ) -> None:
        await self._ensure_control_components()
        for spmd_id, spmd in manifest.spmds.items():
            for component_id in spmd.components:
                registration = self._require_component(spmd_id, component_id)
                if registration.save is None:
                    raise RuntimeError(
                        "committed checkpoint component has no save capability: "
                        f"spmd={spmd_id!r} component={component_id!r}"
                    )
                await registration.save.acknowledge_save(permit)

    async def _abort_incomplete_transaction(
        self, record: CheckpointTransactionRecord
    ) -> None:
        await self._ensure_control_components()
        for registry in self._registries.values():
            for registration in registry:
                if registration.save is not None:
                    await registration.save.abort_save(record.permit)

    async def _fence_incomplete_transactions(self) -> None:
        """Reconcile all incomplete transactions on startup.

        Transactions that have already been marked committed with a durable global
        marker must replay owner acknowledgement using their persisted permit and
        advance to COMMITTED. Pre-marker incomplete transactions are rolled back across
        components and marked ABANDONED.
        """
        incomplete = await self._repository_lane.call(
            self._repository.incomplete_transactions
        )
        for record in incomplete:
            is_committed = await self._repository_lane.call(
                self._repository.is_committed, record.checkpoint_id
            )
            if is_committed:
                manifest = await self._repository_lane.call(
                    self._repository.read, record.checkpoint_id
                )
                await self._acknowledge_committed_transaction(manifest, record.permit)
                await self._repository_lane.call(
                    self._repository.complete_transaction,
                    record.checkpoint_id,
                    record.permit,
                )
                logger.info(
                    "replayed committed checkpoint acknowledgement checkpoint=%s",
                    record.checkpoint_id,
                )
            else:
                await self._abort_incomplete_transaction(record)
                await self._repository_lane.call(
                    self._repository.abandon_transaction,
                    record.checkpoint_id,
                    record.permit,
                )
                logger.warning(
                    "fenced incomplete checkpoint attempt checkpoint=%s status=%s",
                    record.checkpoint_id,
                    record.status,
                )

    async def _repair_cleanup_intents(self) -> int:
        intents = await self._repository_lane.call(
            self._repository.list_cleanup_intents
        )
        if not intents:
            return 0
        await self._repository_lane.call(
            self._repository.evict_manifests_for_cleanup_intents,
            intents,
        )
        self._schedule_cleanup_intents(intents)
        return len(intents)

    def _schedule_cleanup_intents(self, intents: tuple[Any, ...]) -> None:
        selected: list[Any] = []
        seen: set[str] = set()
        for intent in intents:
            if (
                intent.artifact_id in self._cleanup_artifacts_inflight
                or intent.artifact_id in seen
            ):
                continue
            seen.add(intent.artifact_id)
            selected.append(intent)
        if not selected:
            return
        self._cleanup_artifacts_inflight.update(
            intent.artifact_id for intent in selected
        )
        task = asyncio.create_task(self._run_cleanup_intents(tuple(selected)))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_tasks.discard)
        task.add_done_callback(self._record_cleanup_task_failure)

    async def _run_cleanup_intents(self, intents: tuple[Any, ...]) -> None:
        try:
            async with self._cleanup_lane_lock:
                await self._retention.cleanup(intents)
        finally:
            self._cleanup_artifacts_inflight.difference_update(
                intent.artifact_id for intent in intents
            )

    def _record_cleanup_task_failure(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            self._cleanup_failures.append(asyncio.CancelledError())
            return
        error = task.exception()
        if error is not None:
            self._cleanup_failures.append(error)
            logger.error(
                "checkpoint cleanup intent failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _require_component(
        self, spmd_id: str, component_id: str
    ) -> CheckpointComponentRegistration:
        registry = self._registries.get(spmd_id)
        if registry is None:
            raise KeyError(
                f"checkpoint component registry not found for SPMD {spmd_id!r}"
            )
        return registry.require(component_id)

    def _validate_restore_contract(self) -> None:
        if self._restore_scope is not None:
            if not self._restore_scope:
                raise ValueError("restore_scope must not be empty when provided")
            if any(not value for value in self._restore_scope):
                raise ValueError("restore_scope entries must be non-empty")
            if len(set(self._restore_scope)) != len(self._restore_scope):
                raise ValueError("restore_scope entries must be unique")
        if any(
            not source or not target
            for source, target in self._restore_spmd_bindings.items()
        ):
            raise ValueError("restore_spmd_bindings requires non-empty IDs")
        if len(set(self._restore_spmd_bindings.values())) != len(
            self._restore_spmd_bindings
        ):
            raise ValueError("restore_spmd_bindings targets must be unique")
        if self._restore_scope is not None and not set(
            self._restore_spmd_bindings
        ).issubset(self._restore_scope):
            raise ValueError(
                "restore_spmd_bindings sources must belong to restore_scope"
            )

    def _validate_retention_capabilities(self) -> None:
        for spmd_id, registry in self._registries.items():
            for registration in registry:
                if not registration.descriptor.checkpoint_objects:
                    continue
                policy = self._retention_policy
                domain = registration.descriptor.retention_domain
                if domain is not None:
                    try:
                        policy = self._retention_domains[domain]
                    except KeyError as error:
                        raise ValueError(
                            f"retention domain is not configured: {domain!r}"
                        ) from error
                if policy.retain_last is None or policy.retain_forever:
                    continue
                if (
                    registration.object_lifecycle is None
                    or registration.object_cleanup is None
                ):
                    raise ValueError(
                        "finite artifact retention requires object lifecycle and cleanup capabilities: "
                        f"spmd={spmd_id!r} component={registration.descriptor.component_id!r}"
                    )

    def _select_restore_manifest(
        self, manifest: CheckpointManifest
    ) -> CheckpointManifest:
        source_ids = tuple(manifest.spmds)
        selected = source_ids if self._restore_scope is None else self._restore_scope
        missing = tuple(
            spmd_id for spmd_id in selected if spmd_id not in manifest.spmds
        )
        if missing:
            raise KeyError(
                "restore_scope references SPMDs absent from checkpoint manifest: "
                f"{missing}"
            )
        remapped: dict[str, Any] = {}
        for source_id in selected:
            target_id = self._restore_spmd_bindings.get(source_id, source_id)
            if target_id not in self._registries:
                raise KeyError(
                    f"restore target SPMD {target_id!r} has no current component registry"
                )
            if target_id in remapped:
                raise ValueError(
                    f"restore target SPMD {target_id!r} is bound more than once"
                )
            remapped[target_id] = manifest.spmds[source_id]
        return replace(manifest, spmds=remapped)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("FluxwayCheckpointer is closed")

    async def _assert_restorable(self, manifest: CheckpointManifest) -> None:
        for spmd_id, snapshot in manifest.spmds.items():
            registry = self._registries.get(spmd_id)
            if registry is None:
                raise KeyError(
                    f"checkpoint component registry not found for SPMD {spmd_id!r}"
                )
            for component_id, publication in snapshot.components.items():
                component = registry.require(component_id)
                if component.restore is None:
                    raise RuntimeError(
                        f"checkpoint component does not expose restore capability: {component_id!r}"
                    )
                declared_objects = {
                    checkpoint_object.checkpoint_object_id: checkpoint_object
                    for checkpoint_object in component.descriptor.checkpoint_objects
                }
                records = (
                    *publication.shared_records,
                    *(partition.record for partition in publication.partitions),
                )
                for record in records:
                    for object_ref in record.checkpoint_objects:
                        declared_object = declared_objects.get(
                            object_ref.checkpoint_object_id
                        )
                        if declared_object is None:
                            raise ValueError(
                                f"manifest checkpoint object {object_ref.checkpoint_object_id!r} "
                                f"is not declared by component {component_id!r}"
                            )
                        if declared_object.required_for_restore and (
                            await self._repository_lane.call(
                                self._repository.is_checkpoint_object_cleaned_up,
                                spmd_id,
                                object_ref,
                            )
                        ):
                            raise CheckpointObjectCleanedUpError(
                                "checkpoint object was cleaned up: "
                                f"spmd={spmd_id!r} component={component_id!r} "
                                f"checkpoint_object={object_ref.checkpoint_object_id!r} "
                                f"version={object_ref.version_id!r}"
                            )

    @staticmethod
    def _validate_assembly_cut(
        request: SaveCheckpointRequest, resolved: CheckpointCut
    ) -> None:
        if not isinstance(resolved, CheckpointCut):
            raise TypeError("checkpoint assembly returned an invalid checkpoint cut")
        if resolved.cut_id != request.cut.cut_id:
            raise ValueError("checkpoint assembly changed the checkpoint cut ID")
        if resolved.position != request.cut.position:
            raise ValueError("checkpoint assembly changed the durable position")
        if resolved.control_epoch != request.cut.control_epoch:
            raise ValueError("checkpoint assembly changed the control epoch")

    @staticmethod
    def _validate_manifest_progress(manifest: CheckpointManifest) -> None:
        if not isinstance(manifest.framework_state, Mapping):
            raise TypeError("manifest framework state must be a mapping")
        # Source progress validation belongs to the registered Controller
        # components.  The generic checkpointer only validates the global cut
        # envelope and must not parse a source-specific state projection.
