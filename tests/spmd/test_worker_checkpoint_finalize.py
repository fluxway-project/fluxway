import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from fluxway.common.types.checkpoint import (
    CheckpointCommitPermit,
    CheckpointComponentSpec,
    CheckpointComponentPublication,
    CheckpointCut,
    CheckpointSaveScope,
    CheckpointSaveSpec,
    TopologySnapshot,
    WorkerSaveResult,
    CommitSaveSession,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponent,
    CheckpointComponentFinalize,
    CheckpointComponentRegistration,
    CheckpointComponentSave,
)
from fluxway.control_plane.controller import Controller, _publish_checkpoint_output_step
from fluxway.spmd.worker.checkpoint import WorkerCheckpointCoordinator


class _Finalizer(CheckpointComponent, CheckpointComponentFinalize):
    def __init__(self, events, fail_rank=None):
        self._events = events
        self._fail_rank = fail_rank
        self._descriptor = CheckpointComponentSpec("algorithm-finalize")

    @property
    def descriptor(self):
        return self._descriptor

    async def finalize_save(self, session):
        self._events.append(("finalize", session.rank))
        if session.rank == self._fail_rank:
            raise RuntimeError("finalizer failed")

    def registration(self):
        return CheckpointComponentRegistration(component=self, finalize=self)


class _LeaderStorage(CheckpointComponent, CheckpointComponentSave):
    def __init__(self, events, *, output_path=None):
        self._events = events
        self.sessions = []
        self._descriptor = CheckpointComponentSpec(
            "storage",
            save=CheckpointSaveSpec(
                prepare_scope=CheckpointSaveScope.LEADER,
                commit_scope=CheckpointSaveScope.LEADER,
            ),
            output_path=output_path,
        )

    @property
    def descriptor(self):
        return self._descriptor

    async def prepare_save(self, session):
        self._events.append(("prepare", session.rank))
        self.sessions.append(session)

    async def save(self, session):
        self._events.append(("save", session.rank))
        return WorkerSaveResult()

    async def finish_save(self, session):
        self._events.append(("finish", session.rank))

    async def commit_save(self, session):
        raise AssertionError("not exercised")

    async def rollback_save(self, session):
        self._events.append(("rollback", session.rank))

    async def abort_save(self, permit):
        self._events.append(("abort", permit.attempt_id))

    async def acknowledge_save(self, permit):
        self._events.append(("acknowledge", permit.attempt_id))

    async def publish_save(self, session):
        self._events.append(("publish", session.rank))

    def registration(self):
        return CheckpointComponentRegistration(component=self, save=self)


class _FailingStorage(_LeaderStorage):
    def __init__(self, events, component_id):
        super().__init__(events)
        self._descriptor = CheckpointComponentSpec(
            component_id,
            save=CheckpointSaveSpec(
                prepare_scope=CheckpointSaveScope.ALL_RANKS,
                commit_scope=CheckpointSaveScope.ALL_RANKS,
            ),
        )

    async def prepare_save(self, session):
        self._events.append(("prepare", session.rank, self.descriptor.component_id))
        raise RuntimeError("storage prepare failed")


def _coordinator(rank, registrations):
    return WorkerCheckpointCoordinator(
        components=registrations,
        operator_instance=object(),
        repository_root="/tmp/checkpoint-finalize-test",
        spmd_id="test",
        rank=rank,
        source_topology=TopologySnapshot(2, ("dp",), (2,)),
    )


async def _prepare(coordinator):
    await coordinator.prepare_save(
        "checkpoint",
        "attempt",
        CheckpointCommitPermit("attempt", 0, b"permit"),
        CheckpointCut.for_step("cut", 4),
    )


class _OutputComponent(CheckpointComponent, CheckpointComponentSave):
    def __init__(self, output_path: str, *, fail_publish: bool = False):
        self._descriptor = CheckpointComponentSpec(
            output_path=output_path,
            save=CheckpointSaveSpec(
                prepare_scope=CheckpointSaveScope.LEADER,
                commit_scope=CheckpointSaveScope.LEADER,
            ),
            component_id=output_path.replace("/", "_"),
        )
        self._fail_publish = fail_publish

    @property
    def descriptor(self):
        return self._descriptor

    async def prepare_save(self, session):
        return None

    async def save(self, session):
        return WorkerSaveResult()

    async def finish_save(self, session):
        return None

    async def commit_save(self, session: CommitSaveSession):
        if session.staging_output_path is None:
            raise AssertionError("output component needs a staging path")
        Path(session.staging_output_path).mkdir(parents=True, exist_ok=False)
        return CheckpointComponentPublication(
            component_id=self.descriptor.component_id,
        )

    async def rollback_save(self, session):
        return None

    async def abort_save(self, permit):
        return None

    async def acknowledge_save(self, permit):
        return None

    async def publish_save(self, session):
        if self._fail_publish:
            raise RuntimeError("output publish failed")

    def registration(self):
        return CheckpointComponentRegistration(component=self, save=self)


@pytest.mark.asyncio
async def test_output_components_publish_as_one_step_root(tmp_path):
    components = (
        _OutputComponent("model"),
        _OutputComponent("sparse/boss"),
    )
    coordinator = WorkerCheckpointCoordinator(
        tuple(component.registration() for component in components),
        object(),
        str(tmp_path),
        "spmd",
        0,
        TopologySnapshot(1, (), ()),
    )
    await coordinator.prepare_save(
        "checkpoint",
        "attempt",
        CheckpointCommitPermit("attempt", 0, b"permit"),
        CheckpointCut.for_step("cut", 17),
    )
    records = await coordinator.save()
    await coordinator.finish_save()
    await coordinator.commit_save(
        {
            component.descriptor.component_id: {
                0: records[component.descriptor.component_id]
            }
            for component in components
        }
    )
    assert await coordinator.publish_save()
    await _publish_checkpoint_output_step(str(tmp_path), "checkpoint", "attempt", 17)
    assert (tmp_path / "steps" / "17" / "model").is_dir()
    assert (tmp_path / "steps" / "17" / "sparse" / "boss").is_dir()
    assert not (tmp_path / "steps" / ".checkpoint.attempt").exists()


@pytest.mark.asyncio
async def test_output_component_failure_keeps_step_invisible(tmp_path):
    components = (
        _OutputComponent("model"),
        _OutputComponent("sparse/boss", fail_publish=True),
    )
    coordinator = WorkerCheckpointCoordinator(
        tuple(component.registration() for component in components),
        object(),
        str(tmp_path),
        "spmd",
        0,
        TopologySnapshot(1, (), ()),
    )
    await coordinator.prepare_save(
        "checkpoint",
        "attempt",
        CheckpointCommitPermit("attempt", 0, b"permit"),
        CheckpointCut.for_step("cut", 18),
    )
    records = await coordinator.save()
    await coordinator.finish_save()
    await coordinator.commit_save(
        {
            component.descriptor.component_id: {
                0: records[component.descriptor.component_id]
            }
            for component in components
        }
    )
    with pytest.raises(RuntimeError, match="output publish failed"):
        await coordinator.publish_save()
    assert not (tmp_path / "steps" / "18").exists()


@pytest.mark.asyncio
async def test_controller_waits_for_all_spmds_before_publishing_step(tmp_path):
    class _Driver:
        def __init__(self):
            async def acknowledge():
                return True

            class _RemoteCall:
                def remote(self, *args):
                    return acknowledge()

            self.publish_snapshot = _RemoteCall()

    class _Manager:
        def get_all_spmd_ids(self):
            return ("model", "sparse")

        def get_driver_handle(self, spmd_id):
            return _Driver()

    staging = tmp_path / "steps" / ".checkpoint.attempt"
    (staging / "model").mkdir(parents=True)
    (staging / "sparse" / "boss").mkdir(parents=True)
    controller = object.__new__(cast(Any, Controller).__ray_actor_class__)
    controller._spmd_manager = _Manager()
    controller._snapshot_publication_context = {
        "checkpoint": (str(tmp_path), "attempt", 73)
    }
    await controller.publish_snapshot(
        "checkpoint", CheckpointCommitPermit("attempt", 0, b"permit")
    )
    assert (tmp_path / "steps" / "73" / "model").is_dir()
    assert (tmp_path / "steps" / "73" / "sparse" / "boss").is_dir()


def test_published_step_conflict_is_rejected(tmp_path):
    existing = tmp_path / "steps" / "73" / "model"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="step already exists"):
        asyncio.run(
            _publish_checkpoint_output_step(
                str(tmp_path), "different-checkpoint", "attempt", 73
            )
        )


@pytest.mark.asyncio
async def test_finalizers_run_on_every_rank_before_leader_storage_prepare():
    events = []
    finalizer = _Finalizer(events)
    storage = _LeaderStorage(events)
    workers = tuple(
        _coordinator(rank, (finalizer.registration(), storage.registration()))
        for rank in range(2)
    )

    await asyncio.gather(*(_prepare(worker) for worker in workers))

    assert sorted(event for event in events if event[0] == "finalize") == [
        ("finalize", 0),
        ("finalize", 1),
    ]
    assert [event for event in events if event[0] == "prepare"] == [("prepare", 0)]
    assert not [event for event in events if event[0] == "save"]

    await asyncio.gather(*(worker.save() for worker in workers))
    assert [event for event in events if event[0] == "save"] == [
        ("save", 0),
        ("save", 1),
    ]


@pytest.mark.asyncio
async def test_finalizer_failure_skips_storage_prepare_and_clears_transaction():
    events = []
    finalizer = _Finalizer(events, fail_rank=0)
    storage = _LeaderStorage(events)
    worker = _coordinator(0, (finalizer.registration(), storage.registration()))

    with pytest.raises(RuntimeError, match="finalizer failed"):
        await _prepare(worker)

    assert events == [("finalize", 0)]
    assert worker.state is None


@pytest.mark.asyncio
async def test_prepare_failure_rolls_back_only_entered_save_capabilities():
    events = []
    first = _LeaderStorage(events)
    failing = _FailingStorage(events, "failing")
    worker = _coordinator(1, (first.registration(), failing.registration()))

    with pytest.raises(RuntimeError, match="storage prepare failed"):
        await _prepare(worker)

    assert ("rollback", 1) in events
    assert events.count(("rollback", 1)) == 1
    assert worker.state is None


@pytest.mark.asyncio
async def test_component_output_is_published_under_durable_step_position(tmp_path):
    storage = _LeaderStorage([], output_path="sparse/embedding")
    worker = WorkerCheckpointCoordinator(
        components=(storage.registration(),),
        operator_instance=object(),
        repository_root=str(tmp_path),
        spmd_id="test",
        rank=0,
        source_topology=TopologySnapshot(1, (), ()),
    )

    await worker.prepare_save(
        "opaque-checkpoint-id",
        "attempt",
        CheckpointCommitPermit("attempt", 0, b"permit"),
        CheckpointCut.for_step("cut", 73),
    )

    session = storage.sessions[0]
    assert session.staging_output_path == str(
        tmp_path / "steps" / ".opaque-checkpoint-id.attempt" / "sparse" / "embedding"
    )
    assert session.published_output_path == str(
        tmp_path / "steps" / "73" / "sparse" / "embedding"
    )
    await worker.rollback_save()


@pytest.mark.asyncio
async def test_component_output_requires_a_durable_step_position(tmp_path):
    storage = _LeaderStorage([], output_path="sparse/embedding")
    worker = WorkerCheckpointCoordinator(
        components=(storage.registration(),),
        operator_instance=object(),
        repository_root=str(tmp_path),
        spmd_id="test",
        rank=0,
        source_topology=TopologySnapshot(1, (), ()),
    )

    with pytest.raises(ValueError, match="durable step position"):
        await worker.prepare_save(
            "opaque-checkpoint-id",
            "attempt",
            CheckpointCommitPermit("attempt", 0, b"permit"),
            CheckpointCut.for_position("cut", None),
        )

    assert worker.state is None


@pytest.mark.asyncio
async def test_worker_requires_save_specification_when_save_capability_is_present(
    tmp_path,
):
    storage = _LeaderStorage([])
    component_without_save_spec = _Finalizer([])
    broken_registration = object.__new__(CheckpointComponentRegistration)
    object.__setattr__(broken_registration, "component", component_without_save_spec)
    object.__setattr__(broken_registration, "save", storage)
    object.__setattr__(broken_registration, "finalize", None)
    object.__setattr__(broken_registration, "restore", None)
    object.__setattr__(broken_registration, "component_lifecycle", None)
    object.__setattr__(broken_registration, "object_lifecycle", None)
    object.__setattr__(broken_registration, "object_cleanup", None)

    worker = WorkerCheckpointCoordinator(
        components=(broken_registration,),
        operator_instance=object(),
        repository_root=str(tmp_path),
        spmd_id="test",
        rank=0,
        source_topology=TopologySnapshot(1, (), ()),
    )

    with pytest.raises(
        RuntimeError, match="defines save capability without save specification"
    ):
        await worker.prepare_save(
            "opaque-checkpoint-id",
            "attempt",
            CheckpointCommitPermit("attempt", 0, b"permit"),
            CheckpointCut.for_step("cut", 1),
        )
