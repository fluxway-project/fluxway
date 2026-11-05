import asyncio
import hashlib
import inspect
from dataclasses import replace
from typing import Any

from fluxway.common.decorators import data_type, operator, spmd
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.resources import ResourceProfile

import lance
import numpy as np
import pytest
import sparse_embedding as se
from sparse_embedding.checkpoint import (
    CheckpointSaveMode,
    PartitionSpec,
    RestorePlan,
)
from sparse_embedding.checkpoint.base import CheckpointConflictError
from sparse_embedding.checkpoint.schema import SNAPSHOT_MANIFEST_DATASET

from fluxway.common.types.checkpoint import (
    CheckpointComponentCapability,
    CheckpointCommitPermit,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CheckpointCut,
    CheckpointAssembly,
    SaveCheckpointRequest,
    SpmdCheckpoint,
    StepReference,
    CheckpointCleanupBlocked,
    TopologySnapshot,
)
from fluxway.common.types import DrainBoundary
from fluxway.control_plane.checkpointer.component import CheckpointComponentRegistry
from fluxway.control_plane.checkpointer.checkpoint import FluxwayCheckpointer
from fluxway.control_plane.checkpointer.repository import CheckpointRepository
from fluxway.common.types.checkpoint import CheckpointRetentionPolicy
from fluxway.integrations.sparse_embedding.checkpoint import (
    LocalLanceStorageFactory,
    SparseEmbeddingCheckpointComponent,
    compact_sparse_checkpoints,
)
from fluxway.spmd.worker.checkpoint import WorkerCheckpointCoordinator


def _config() -> se.EmbeddingConfig:
    return se.EmbeddingConfig(
        dims=[4], thread_budget=1, value_dtype="float32", key_dtype="uint64"
    )


def _store(keys: np.ndarray) -> se.EmbeddingStore:
    store = se.EmbeddingStore(_config())
    values = np.repeat(keys[:, None].astype(np.float32), 4, axis=1)
    store.assign(keys, values)
    return store


def _lookup(store: se.EmbeddingStore, keys: np.ndarray) -> np.ndarray:
    values, _ = store.lookup(keys, dim=4)
    return np.from_dlpack(values).reshape(-1, 4)


def _modulo_restore_options(
    logical_partition_count: int, logical_partition_id: int
) -> bytes:
    return RestorePlan.from_partition(
        PartitionSpec.modulo(),
        logical_partition_count,
        logical_partition_id,
        key_dtype="uint64",
    ).encode()


class _Owner:
    def __init__(self, store: se.EmbeddingStore) -> None:
        self.store = store


class _LocalLanceController:
    def __init__(self, owner, topology) -> None:
        self.owner = owner
        self.topology = topology
        self.restore_owner = None
        self._active = {}

    async def request_snapshot(
        self,
        checkpoint_id,
        attempt_id,
        permit,
        cut,
        repository_dir,
        deadline=None,
    ):
        coordinator = _coordinator(repository_dir, 0, self.topology, self.owner)
        await coordinator.prepare_save(checkpoint_id, attempt_id, permit, cut)
        record = (await coordinator.save())["embedding"]
        await coordinator.finish_save()
        publications = await coordinator.commit_save({"embedding": {0: record}})
        self._active[checkpoint_id] = coordinator
        return CheckpointAssembly(
            cut=cut,
            framework_state={},
            spmds={
                "embedding": SpmdCheckpoint(
                    source_topology=self.topology,
                    runtime_state={
                        "drain_boundary": DrainBoundary(
                            committed_cut=cut.position or 0
                        ).to_dict()
                    },
                    components=publications,
                )
            },
        )

    async def publish_snapshot(self, checkpoint_id, permit, deadline=None):
        coordinator = self._active.pop(checkpoint_id)
        await coordinator.publish_save()
        await coordinator.close()

    async def rollback_snapshot(self, checkpoint_id, permit):
        coordinator = self._active.pop(checkpoint_id, None)
        if coordinator is not None:
            await coordinator.rollback_active()
            await coordinator.close()

    def prepare_checkpoint_transaction(self, checkpoint_id, permit, cut, control_state):
        return {}

    def apply_prepared_checkpoint_transaction(
        self, checkpoint_id, permit, cut, control_state, prepared
    ):
        return None

    def abort_checkpoint_transaction(self, checkpoint_id, permit):
        return None

    async def restore_snapshot(
        self,
        checkpoint_id,
        manifest,
        repository_dir,
        restore_options=None,
    ):
        if self.restore_owner is None:
            raise RuntimeError("restore owner is not configured")
        coordinator = _coordinator(repository_dir, 0, self.topology, self.restore_owner)
        snapshot = manifest.spmds["embedding"]
        await coordinator.restore(
            checkpoint_id,
            manifest.cut,
            snapshot.components,
            snapshot.source_topology,
            self.topology,
            None
            if restore_options is None
            else restore_options.get("embedding", {}).get(0),
        )
        await coordinator.close()


def _coordinator(root, rank, topology, owner):
    component = SparseEmbeddingCheckpointComponent(
        LocalLanceStorageFactory(),
        lineage_name="embedding",
        mode=CheckpointSaveMode.INCREMENTAL,
        initial_mode=CheckpointSaveMode.FULL,
    )
    return WorkerCheckpointCoordinator(
        (component.registration(),),
        owner,
        str(root),
        "embedding",
        rank,
        topology,
    )


async def _acknowledge_owner_save(root, permit):
    component = SparseEmbeddingCheckpointComponent(
        LocalLanceStorageFactory(),
        lineage_name="embedding",
        mode=CheckpointSaveMode.INCREMENTAL,
        initial_mode=CheckpointSaveMode.FULL,
    )
    await component.open(
        CheckpointRuntime(CheckpointRuntimeRole.CONTROL, str(root), "embedding")
    )
    try:
        await component.acknowledge_save(permit)
    finally:
        await component.close()


@pytest.mark.asyncio
async def test_fluxway_restores_selected_step_from_exact_lance_version(tmp_path):
    topology = TopologySnapshot(1, (), ())
    owner = _Owner(_store(np.asarray([1, 2], dtype=np.uint64)))
    controller = _LocalLanceController(owner, topology)
    control_component = SparseEmbeddingCheckpointComponent(
        LocalLanceStorageFactory(),
        lineage_name="embedding",
        mode=CheckpointSaveMode.INCREMENTAL,
        initial_mode=CheckpointSaveMode.FULL,
    )
    checkpointer = await FluxwayCheckpointer.create(
        controller,
        str(tmp_path / "repository"),
        interval_steps=1,
        retention_policy=CheckpointRetentionPolicy(retain_last=3),
        component_registries={
            "embedding": CheckpointComponentRegistry(
                (control_component.registration(),)
            )
        },
    )

    first = await checkpointer.save(SaveCheckpointRequest.for_step(10))
    owner.store.assign(
        np.asarray([3, 4], dtype=np.uint64),
        np.repeat(np.asarray([3, 4], dtype=np.float32)[:, None], 4, axis=1),
    )
    second = await checkpointer.save(SaveCheckpointRequest.for_step(20))
    assert first.checkpoint_id != second.checkpoint_id
    await checkpointer.close()

    restarted_component = SparseEmbeddingCheckpointComponent(
        LocalLanceStorageFactory(),
        lineage_name="embedding",
        mode=CheckpointSaveMode.INCREMENTAL,
        initial_mode=CheckpointSaveMode.FULL,
    )
    checkpointer = await FluxwayCheckpointer.create(
        controller,
        str(tmp_path / "repository"),
        interval_steps=1,
        retention_policy=CheckpointRetentionPolicy(retain_last=3),
        component_registries={
            "embedding": CheckpointComponentRegistry(
                (restarted_component.registration(),)
            )
        },
    )
    third = await checkpointer.save(SaveCheckpointRequest.for_step(30))
    assert third.checkpoint_id not in {first.checkpoint_id, second.checkpoint_id}

    sparse_root = (
        tmp_path
        / "repository"
        / "application-checkpoint_objects"
        / "embedding"
        / "embedding"
    )
    manifests = lance.dataset(str(sparse_root / SNAPSHOT_MANIFEST_DATASET)).to_table()
    sparse_rows = sorted(
        (row for row in manifests.to_pylist() if row["checkpoint_name"] == "embedding"),
        key=lambda row: row["checkpoint_version"],
    )
    assert len(sparse_rows) == 3
    assert sparse_rows[-1]["checkpoint_version"] > sparse_rows[-2]["checkpoint_version"]
    assert sparse_rows[-1]["parent_checkpoint_name"] == "embedding"

    controller.restore_owner = _Owner(se.EmbeddingStore(_config()))
    restored_step = await checkpointer.restore(
        StepReference(10),
        restore_options={
            "embedding": {0: {"embedding": _modulo_restore_options(1, 0)}}
        },
    )
    assert restored_step == 10
    restored = controller.restore_owner.store
    assert restored.total_size() == 2
    np.testing.assert_allclose(
        _lookup(restored, np.asarray([1, 2], dtype=np.uint64)),
        np.asarray([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32),
    )
    await checkpointer.close()


@pytest.mark.asyncio
async def test_fluxway_automatically_cleanups_sparse_versions(tmp_path):
    topology = TopologySnapshot(1, (), ())
    owner = _Owner(_store(np.asarray([1], dtype=np.uint64)))
    controller = _LocalLanceController(owner, topology)
    control_component = SparseEmbeddingCheckpointComponent(
        LocalLanceStorageFactory(),
        lineage_name="embedding",
        mode=CheckpointSaveMode.INCREMENTAL,
        initial_mode=CheckpointSaveMode.FULL,
    )
    repository = tmp_path / "repository"
    checkpointer = await FluxwayCheckpointer.create(
        controller,
        str(repository),
        interval_steps=1,
        retention_policy=CheckpointRetentionPolicy(retain_last=1),
        component_registries={
            "embedding": CheckpointComponentRegistry(
                (control_component.registration(),)
            )
        },
    )

    first = await checkpointer.save(SaveCheckpointRequest.for_step(1))
    owner.store.assign(
        np.asarray([2], dtype=np.uint64),
        np.full((1, 4), 2, dtype=np.float32),
    )
    await checkpointer.save(SaveCheckpointRequest.for_step(2))
    await checkpointer.close()

    checkpoint_object = (
        first.spmds["embedding"]
        .components["embedding"]
        .shared_records[0]
        .checkpoint_objects[0]
    )
    assert CheckpointRepository(
        tmp_path / "repository"
    ).is_checkpoint_object_cleaned_up("embedding", checkpoint_object)
    next_checkpointer = await FluxwayCheckpointer.create(
        controller,
        str(repository),
        interval_steps=1,
        retention_policy=CheckpointRetentionPolicy(retain_last=1),
        component_registries={
            "embedding": CheckpointComponentRegistry(
                (control_component.registration(),)
            )
        },
    )
    with pytest.raises(FileNotFoundError, match="checkpoint reference not found"):
        await next_checkpointer.restore(
            StepReference(1),
            restore_options={
                "embedding": {0: {"embedding": _modulo_restore_options(1, 0)}}
            },
        )
    await next_checkpointer.close()

    storage_root = (
        repository / "application-checkpoint_objects" / "embedding" / "embedding"
    )
    rows = (
        lance.dataset(str(storage_root / SNAPSHOT_MANIFEST_DATASET))
        .to_table()
        .to_pylist()
    )
    assert all(row["checkpoint_name"] != first.checkpoint_id for row in rows)


@pytest.mark.asyncio
async def test_sparse_lease_blocks_automatic_cleanup_until_release(tmp_path):
    topology = TopologySnapshot(1, (), ())
    owner = _Owner(_store(np.asarray([1], dtype=np.uint64)))
    controller = _LocalLanceController(owner, topology)
    control_component = SparseEmbeddingCheckpointComponent(
        LocalLanceStorageFactory(),
        lineage_name="embedding",
        mode=CheckpointSaveMode.INCREMENTAL,
        initial_mode=CheckpointSaveMode.FULL,
    )
    checkpointer = await FluxwayCheckpointer.create(
        controller,
        str(tmp_path / "repository"),
        interval_steps=1,
        retention_policy=CheckpointRetentionPolicy(retain_last=1),
        component_registries={
            "embedding": CheckpointComponentRegistry(
                (control_component.registration(),)
            )
        },
    )
    first = await checkpointer.save(SaveCheckpointRequest.for_step(1))
    checkpoint_object = (
        first.spmds["embedding"]
        .components["embedding"]
        .shared_records[0]
        .checkpoint_objects[0]
    )
    lease = await checkpointer.hold_checkpoint_object(
        "embedding",
        checkpoint_object,
        "evaluator",
        60,
        "evaluation-1",
    )
    owner.store.assign(
        np.asarray([2], dtype=np.uint64),
        np.full((1, 4), 2, dtype=np.float32),
    )

    with pytest.raises(CheckpointCleanupBlocked):
        await checkpointer.save(SaveCheckpointRequest.for_step(2))
    await checkpointer.release_checkpoint_hold("embedding", lease)
    await checkpointer.save(SaveCheckpointRequest.for_step(2))
    await checkpointer.close()
    assert CheckpointRepository(
        tmp_path / "repository"
    ).is_checkpoint_object_cleaned_up("embedding", checkpoint_object)


@pytest.mark.asyncio
async def test_local_lance_component_roundtrip_with_elastic_restore(tmp_path):
    source_topology = TopologySnapshot(2, ("shard",), (2,))
    permit = CheckpointCommitPermit("attempt-1", 7, b"coordinator-token")
    cut = CheckpointCut.for_step("cut-10", 10)
    source_keys = (
        np.asarray([0, 2, 4, 6], dtype=np.uint64),
        np.asarray([1, 3, 5, 7], dtype=np.uint64),
    )
    source = [
        _coordinator(tmp_path, rank, source_topology, _Owner(_store(keys)))
        for rank, keys in enumerate(source_keys)
    ]

    for coordinator in source:
        await coordinator.prepare_save(
            "snapshot-10",
            "attempt-1",
            permit,
            cut,
        )
    save_results = await asyncio.gather(*(coordinator.save() for coordinator in source))
    for coordinator in source:
        await coordinator.finish_save()
    gathered = {
        "embedding": {
            rank: records["embedding"] for rank, records in enumerate(save_results)
        }
    }
    publication = (await source[0].commit_save(gathered))["embedding"]
    await source[1].commit_save(gathered)
    for coordinator in source:
        await coordinator.publish_save()
        await coordinator.close()
    await _acknowledge_owner_save(tmp_path, permit)

    checkpoint_object = publication.shared_records[0].checkpoint_objects[0]
    assert checkpoint_object.storage_locator
    assert isinstance(checkpoint_object.storage_locator, bytes)
    assert checkpoint_object.storage_locator
    manifest_path = (
        tmp_path
        / "application-checkpoint_objects"
        / "embedding"
        / "embedding"
        / SNAPSHOT_MANIFEST_DATASET
    )
    manifest = lance.dataset(str(manifest_path)).to_table().to_pylist()[0]
    assert manifest["checkpoint_name"] == "embedding"
    assert manifest["save_attempt"] == "attempt-1"
    assert manifest["authority_scope"] == "fluxway:embedding:embedding"
    assert manifest["authority_generation"] == permit.coordinator_generation
    assert manifest["commit_authority"] == hashlib.sha256(permit.token).hexdigest()

    with pytest.raises(ValueError, match="checkpoint_object component"):
        replace(
            publication,
            shared_records=(
                replace(
                    publication.shared_records[0],
                    checkpoint_objects=(
                        replace(checkpoint_object, component_id="other"),
                    ),
                ),
            ),
        )

    invalid_checkpoint_objects = (
        replace(checkpoint_object, checkpoint_object_id="other"),
    )

    owner = _Owner(se.EmbeddingStore(_config()))
    natural_coordinator = _coordinator(
        tmp_path,
        0,
        source_topology,
        owner,
    )
    await natural_coordinator.restore(
        "snapshot-10",
        cut,
        {"embedding": publication},
        source_topology,
        source_topology,
    )
    assert owner.store.total_size() == 4
    np.testing.assert_allclose(
        _lookup(owner.store, np.asarray([0, 2, 4, 6], dtype=np.uint64)),
        np.repeat(np.asarray([0, 2, 4, 6], dtype=np.float32)[:, None], 4, axis=1),
    )
    await natural_coordinator.close()

    for invalid in invalid_checkpoint_objects:
        invalid_publication = replace(
            publication,
            shared_records=(
                replace(publication.shared_records[0], checkpoint_objects=(invalid,)),
            ),
        )
        invalid_coordinator = _coordinator(
            tmp_path,
            0,
            source_topology,
            _Owner(se.EmbeddingStore(_config())),
        )
        with pytest.raises(ValueError):
            await invalid_coordinator.restore(
                "snapshot-10",
                cut,
                {"embedding": invalid_publication},
                source_topology,
                source_topology,
                {"embedding": _modulo_restore_options(1, 0)},
            )
        await invalid_coordinator.close()

    checkpoint_object_topology = TopologySnapshot(3, ("shard",), (3,))
    checkpoint_object_owners = [_Owner(se.EmbeddingStore(_config())) for _ in range(3)]
    checkpoint_object = [
        _coordinator(tmp_path, rank, checkpoint_object_topology, owner)
        for rank, owner in enumerate(checkpoint_object_owners)
    ]
    for rank, coordinator in enumerate(checkpoint_object):
        await coordinator.restore(
            "snapshot-10",
            cut,
            {"embedding": publication},
            source_topology,
            checkpoint_object_topology,
            {
                "embedding": _modulo_restore_options(
                    checkpoint_object_topology.world_size, rank
                )
            },
        )

    all_keys = np.arange(8, dtype=np.uint64)
    for rank, owner in enumerate(checkpoint_object_owners):
        expected = all_keys[all_keys % checkpoint_object_topology.world_size == rank]
        assert owner.store.total_size() == expected.size
        np.testing.assert_allclose(
            _lookup(owner.store, expected),
            np.repeat(expected[:, None].astype(np.float32), 4, axis=1),
        )

    permit2 = CheckpointCommitPermit("attempt-2", 8, b"coordinator-token-2")
    cut2 = CheckpointCut.for_step("cut-20", 20)
    for coordinator in checkpoint_object:
        await coordinator.prepare_save(
            "snapshot-20",
            "attempt-2",
            permit2,
            cut2,
        )
    save_results2 = await asyncio.gather(*(c.save() for c in checkpoint_object))
    for coordinator in checkpoint_object:
        await coordinator.finish_save()
    gathered2 = {
        "embedding": {
            rank: records["embedding"] for rank, records in enumerate(save_results2)
        }
    }
    for coordinator in checkpoint_object:
        await coordinator.commit_save(gathered2)
        await coordinator.publish_save()
    await _acknowledge_owner_save(tmp_path, permit2)

    rows = lance.dataset(str(manifest_path)).to_table().to_pylist()
    assert len(rows) == 2
    assert rows[1]["checkpoint_name"] == "embedding"
    assert rows[1]["checkpoint_version"] == 2
    assert rows[1]["checkpoint_mode"] == "full"
    assert rows[1]["participant_count"] == 3
    assert rows[1]["parent_checkpoint_version"] == 0

    checkpoint_object_owners[1].store.assign(
        np.asarray([1], dtype=np.uint64),
        np.full((1, 4), 99, dtype=np.float32),
    )
    permit3 = CheckpointCommitPermit("attempt-3", 9, b"coordinator-token-3")
    cut3 = CheckpointCut.for_step("cut-30", 30)
    for coordinator in checkpoint_object:
        await coordinator.prepare_save(
            "snapshot-30",
            "attempt-3",
            permit3,
            cut3,
        )
    save_results3 = await asyncio.gather(*(c.save() for c in checkpoint_object))
    for coordinator in checkpoint_object:
        await coordinator.finish_save()
    gathered3 = {
        "embedding": {
            rank: records["embedding"] for rank, records in enumerate(save_results3)
        }
    }
    for coordinator in checkpoint_object:
        await coordinator.commit_save(gathered3)
        await coordinator.publish_save()
        await coordinator.close()
    await _acknowledge_owner_save(tmp_path, permit3)

    rows = lance.dataset(str(manifest_path)).to_table().to_pylist()
    assert len(rows) == 3
    assert rows[2]["checkpoint_name"] == "embedding"
    assert rows[2]["checkpoint_version"] == 3
    assert rows[2]["checkpoint_mode"] == "incremental"
    assert rows[2]["participant_count"] == 3
    assert rows[2]["parent_checkpoint_version"] == 2


def test_sparse_component_declares_only_implemented_capabilities():
    registration = SparseEmbeddingCheckpointComponent(
        LocalLanceStorageFactory(),
        lineage_name="embedding",
        mode=CheckpointSaveMode.INCREMENTAL,
        initial_mode=CheckpointSaveMode.FULL,
    ).registration()
    assert registration.capabilities == frozenset(
        {
            CheckpointComponentCapability.COMPONENT_LIFECYCLE,
            CheckpointComponentCapability.SAVE,
            CheckpointComponentCapability.RESTORE,
            CheckpointComponentCapability.OBJECT_LIFECYCLE,
            CheckpointComponentCapability.OBJECT_CLEANUP,
        }
    )


def test_sparse_save_apis_do_not_accept_partition_strategy():
    assert (
        "partition_strategy"
        not in inspect.signature(SparseEmbeddingCheckpointComponent).parameters
    )
    assert (
        "partition_strategy"
        not in inspect.signature(compact_sparse_checkpoints).parameters
    )
    lineage_parameter = inspect.signature(compact_sparse_checkpoints).parameters[
        "lineage_name"
    ]
    assert lineage_parameter.default is inspect.Parameter.empty


@pytest.mark.asyncio
async def test_local_lance_component_rejects_stale_commit_authority(tmp_path):
    topology = TopologySnapshot(1, (), ())
    cut = CheckpointCut.for_step("cut", 1)
    owner = _Owner(_store(np.asarray([1], dtype=np.uint64)))
    stale = _coordinator(tmp_path, 0, topology, owner)
    current = _coordinator(tmp_path, 0, topology, owner)
    stale_permit = CheckpointCommitPermit("attempt-old", 1, b"old")
    current_permit = CheckpointCommitPermit("attempt-new", 2, b"new")

    await current.prepare_save(
        "shared-snapshot",
        "attempt-new",
        current_permit,
        cut,
    )
    current_record = (await current.save())["embedding"]
    await current.finish_save()

    with pytest.raises(CheckpointConflictError, match="authority is stale"):
        await stale.prepare_save(
            "shared-snapshot",
            "attempt-old",
            stale_permit,
            cut,
        )
    await stale.close()

    publication = await current.commit_save({"embedding": {0: current_record}})
    assert publication["embedding"].shared_records
    await current.publish_save()
    await current.close()


@data_type()
class _ProbePayload:
    value: Any | None = None


@pytest.mark.asyncio
async def test_sparse_checkpoint_graph_node_isolation(tmp_path):
    template_owner = SparseEmbeddingCheckpointComponent(
        LocalLanceStorageFactory(),
        partition_spec=PartitionSpec.hash(seed=42),
    )

    @spmd(
        resource_profile=ResourceProfile(base={"CPU": 1}),
        mesh_axis_names=("shard",),
        mesh_axis_shapes=(1,),
        checkpoint_components=(template_owner.registration(),),
    )
    class Probe:
        @operator(
            input_sharding={"value": {"type": "replicated", "axis": []}},
            output_sharding={"value": {"type": "replicated", "axis": []}},
        )
        def execute(self, payload: _ProbePayload) -> _ProbePayload:
            return payload

    with FunctionalGraph("partition-copy-isolation") as graph:
        left = Probe(name="left")
        right = Probe(name="right")
        right.execute(left.execute())
        compiled = graph.build()

    left_registration = compiled.nodes["left"].spmd_config.checkpoint_components[0]
    right_registration = compiled.nodes["right"].spmd_config.checkpoint_components[0]
    assert left_registration.descriptor.output_path == "sparse/left"
    assert right_registration.descriptor.output_path == "sparse/right"
    assert template_owner.descriptor.output_path is None

    for registration in (left_registration, right_registration):
        for capability in (
            "component_lifecycle",
            "save",
            "restore",
            "object_lifecycle",
            "object_cleanup",
        ):
            assert getattr(registration, capability) is registration.component

    assert left_registration.component is not right_registration.component
    assert left_registration.component is not template_owner
    assert right_registration.component is not template_owner
    assert (
        left_registration.component._io_lane
        is not right_registration.component._io_lane
    )
    assert left_registration.component._io_lane is not template_owner._io_lane

    left_registry = CheckpointComponentRegistry((left_registration,))
    right_registry = CheckpointComponentRegistry((right_registration,))

    await asyncio.gather(
        left_registry.open_all(
            CheckpointRuntime(
                CheckpointRuntimeRole.CONTROL,
                str(tmp_path / "left"),
                "left",
            )
        ),
        right_registry.open_all(
            CheckpointRuntime(
                CheckpointRuntimeRole.CONTROL,
                str(tmp_path / "right"),
                "right",
            )
        ),
    )
    try:
        assert left_registration.component._runtime.spmd_id == "left"
        assert right_registration.component._runtime.spmd_id == "right"
        assert (
            left_registration.component._storage
            is not right_registration.component._storage
        )
        assert template_owner._runtime is None
        assert template_owner._storage is None
    finally:
        await asyncio.gather(left_registry.close_all(), right_registry.close_all())
