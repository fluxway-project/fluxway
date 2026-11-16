import asyncio
import types
from typing import Any

import pytest
import numpy as np
from fluxway.common.allocation import AllocationRegistry

from fluxway.common.coherence import (
    CoherenceAction,
    CoherenceDirectory,
    CoherenceState,
    CoherenceStateTransitionError,
    CoherenceSubject,
    ExecutionToken,
    GrantKind,
    GrantReady,
    TensorAccessIntent,
    CoherenceInstance,
)
from fluxway.common.tensor_backend import set_backend
from fluxway.common.types.tensor import TransportTensor
from fluxway.spmd.worker.worker import LocalSendToken, SPMDWorker, _InputCoherenceGrant
from fluxway.spmd.worker.coherence_runtime import (
    CoherenceAdmissionRuntime,
    CoherenceBatchRetireRuntime,
    ParentGrantReturnRuntime,
    _ChildDeliveryReceipts,
    _delegated_reads,
)
from fluxway.spmd.worker.data_plane import InstancePublication, PublicationBatch
from fluxway.rpc.protocol import ReceiverIngressCompletion
from fluxway.spmd.worker.coherence_sequencer import CoherenceSequencer
from fluxway.spmd.worker.data_plane import build_coherence_frame
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionResult,
    ExecutionStatus,
)
from fluxway.transport.communication import (
    TransportDelegationDescriptor,
    TransportFrameBatch,
)


def _entry(directory: CoherenceDirectory, key: CoherenceSubject, epoch: int):
    return directory.admit(CoherenceInstance(key, epoch))


raw_cls = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]


class _Handler:
    def __init__(self, callback):
        self._callback = callback

    async def receive_transport_frames(self, frames):
        return await self._callback(frames)


def _bind(worker, *names):
    if hasattr(worker, "_coherence_directory") and not hasattr(
        worker, "_coherence_admission_runtime"
    ):
        worker._coherence_admission_runtime = CoherenceAdmissionRuntime(
            worker._coherence_directory
        )
    if hasattr(worker, "_coherence_directory") and not hasattr(
        worker, "_coherence_sequencer"
    ):
        worker._coherence_sequencer = CoherenceSequencer(
            lambda _instances, _payload: asyncio.sleep(0),
            lambda _instance: asyncio.sleep(0),
        )
    for name in names:
        value = getattr(raw_cls, name)
        if name == "_bitmap_indices":
            setattr(worker, name, value)
        else:
            setattr(worker, name, types.MethodType(value, worker))


def _grant_map(*grants):
    return {grant.identity: grant for grant in grants}


def _attach_parent_return_runtime(worker):
    _bind(
        worker,
        "_send_parent_return_group",
        "_send_coherence_frame",
        "_require_local_spmd_code",
    )
    worker._parent_grant_return_runtime = ParentGrantReturnRuntime(
        worker._coherence_directory,
        worker._send_parent_return_group,
    )


@pytest.mark.asyncio
async def test_rfo_invalidates_sibling_and_delegates_ownership() -> None:
    route_code = 11
    epoch = 7
    parent_code = 3
    child_code = 4
    parent = types.SimpleNamespace(
        _coherence_directory=CoherenceDirectory(),
        _local_spmd_code=parent_code,
        rank=0,
        _world_size=1,
        _spmd_id_by_code={parent_code: "parent", child_code: "child"},
        _successor_handlers={},
        _predecessor_handlers={},
    )
    _bind(
        parent,
        "_receive_coherence_frame",
        "_grant_child_ownership",
        "_send_coherence_to_child",
        "_send_coherence_frame",
        "_bitmap_indices",
        "_require_local_spmd_code",
    )
    transfer = types.SimpleNamespace(dependency_code=21, logical_tuuid=7)
    dependency = types.SimpleNamespace(
        spec=types.SimpleNamespace(target=types.SimpleNamespace(spmd_id="child"))
    )
    parent._graph_view = types.SimpleNamespace(
        transfer=lambda code: transfer,
        dependency=lambda code: dependency,
    )

    subject = CoherenceSubject(parent_code, 0, 7)
    parent._coherence_directory.publish_local_payload(subject, epoch)
    parent._coherence_directory.delegate_read(subject, epoch, 0)
    parent._coherence_directory.delegate_read(subject, epoch, 1)
    coherence_plan = types.SimpleNamespace(
        coherence_child_slot=lambda _route_code, rank: rank - 1,
        coherence_child=lambda _logical_tuuid, slot: types.SimpleNamespace(
            transfer_code=route_code,
            target_rank=slot + 1,
        ),
    )
    parent._coherence_send_plan = lambda _route_code: coherence_plan
    actions = {1: [], 2: []}

    async def receive(child_rank, frames):
        from fluxway.transport.coherence import entry_coherence_action
        from fluxway.transport.manifest import ENTRY_STATE_FLAGS

        action = entry_coherence_action(
            int(frames[0].header.entry_i32[0, ENTRY_STATE_FLAGS])
        )
        actions[child_rank].append(action)
        if action is CoherenceAction.SNOOP_INVALIDATE:
            parent._coherence_directory.return_child(
                subject,
                epoch,
                child_rank - 1,
            )
        return (0,)

    parent._successor_handlers = {
        "child": {
            rank: _Handler(lambda frames, rank=rank: receive(rank, frames))
            for rank in (1, 2)
        }
    }

    await parent._grant_child_ownership(
        subject=subject,
        epoch=epoch,
        requester_spmd_code=child_code,
        requester_rank=1,
        route_code=route_code,
    )

    assert actions[1] == [CoherenceAction.OWNERSHIP_GRANT]
    assert actions[2] == [CoherenceAction.SNOOP_INVALIDATE]
    entry = _entry(parent._coherence_directory, subject, epoch)
    assert entry.ownership_child_id == 0
    assert entry.state is CoherenceState.INVALID


@pytest.mark.asyncio
async def test_rfo_waits_for_inflight_sibling_delivery_before_invalidate() -> None:
    route_code = 11
    epoch = 7
    parent_code = 3
    child_code = 4
    directory = CoherenceDirectory()
    parent = types.SimpleNamespace(
        _coherence_directory=directory,
        _local_spmd_code=parent_code,
        rank=0,
        _world_size=1,
        _spmd_id_by_code={parent_code: "parent", child_code: "child"},
        _successor_handlers={},
    )
    _bind(
        parent,
        "_grant_child_ownership",
        "_send_coherence_to_child",
        "_send_coherence_frame",
        "_bitmap_indices",
        "_require_local_spmd_code",
    )
    transfer = types.SimpleNamespace(dependency_code=21, logical_tuuid=7)
    dependency = types.SimpleNamespace(
        spec=types.SimpleNamespace(target=types.SimpleNamespace(spmd_id="child"))
    )
    parent._graph_view = types.SimpleNamespace(
        transfer=lambda _code: transfer,
        dependency=lambda _code: dependency,
    )
    coherence_plan = types.SimpleNamespace(
        coherence_child_slot=lambda _route_code, rank: rank - 1,
        coherence_child=lambda _logical_tuuid, slot: types.SimpleNamespace(
            transfer_code=route_code,
            target_rank=slot + 1,
        ),
    )
    parent._coherence_send_plan = lambda _route_code: coherence_plan
    subject = CoherenceSubject(parent_code, 0, 7)
    directory.publish_local_payload(subject, epoch)
    directory.delegate_read(subject, epoch, child_id=0)
    sibling = directory.delegate_read(subject, epoch, child_id=1)
    assert isinstance(sibling, GrantReady)
    sibling_receipt = directory.register_child_delivery(sibling.token)
    actions = []

    async def receive(child_rank, frames):
        from fluxway.transport.coherence import entry_coherence_action
        from fluxway.transport.manifest import ENTRY_STATE_FLAGS

        action = entry_coherence_action(
            int(frames[0].header.entry_i32[0, ENTRY_STATE_FLAGS])
        )
        actions.append((child_rank, action))
        if action is CoherenceAction.SNOOP_INVALIDATE:
            directory.return_child(subject, epoch, child_id=child_rank - 1)
        return (0,)

    parent._successor_handlers = {
        "child": {
            rank: _Handler(lambda frames, rank=rank: receive(rank, frames))
            for rank in (1, 2)
        }
    }
    ownership = asyncio.create_task(
        parent._grant_child_ownership(
            subject=subject,
            epoch=epoch,
            requester_spmd_code=child_code,
            requester_rank=1,
            route_code=route_code,
        )
    )
    await asyncio.sleep(0)

    assert not actions
    assert not ownership.done()
    directory.complete_child_delivery(sibling_receipt, delivered=True)
    await asyncio.wait_for(ownership, timeout=0.1)

    assert actions == [
        (2, CoherenceAction.SNOOP_INVALIDATE),
        (1, CoherenceAction.OWNERSHIP_GRANT),
    ]


@pytest.mark.asyncio
async def test_operator_tensor_access_defaults_to_readonly_and_rejects_missing_grant():
    set_backend("numba_cpu")
    tensor = TransportTensor(np.arange(4, dtype=np.int64))
    operator = types.SimpleNamespace(
        spec=types.SimpleNamespace(
            tensor_access={"input": TensorAccessIntent.READONLY},
        )
    )
    worker = types.SimpleNamespace(
        _coherence_input_grants={},
        _coherence_input_tensors={},
        _coherence_input_intents={},
        _coherence_input_bindings={},
        _local_input_origins={},
        _coherence_sequencer=CoherenceSequencer(
            lambda _instances, _payload: asyncio.sleep(0),
            lambda _instance: asyncio.sleep(0),
        ),
        config=types.SimpleNamespace(
            require_graph_view=lambda: types.SimpleNamespace(
                operator=lambda _op_name: operator
            )
        ),
    )
    worker._apply_operator_tensor_access = types.MethodType(
        raw_cls._apply_operator_tensor_access,
        worker,
    )
    worker._acquire_owned_inputs = types.MethodType(
        raw_cls._acquire_owned_inputs,
        worker,
    )
    prepared = types.SimpleNamespace(
        tensor_bindings=(
            types.SimpleNamespace(
                port="input",
                tensor=tensor,
                grants=(),
            ),
        )
    )
    await worker._apply_operator_tensor_access("op", 7, prepared)

    assert not tensor.flat_values.flags.writeable

    operator.spec.tensor_access["input"] = TensorAccessIntent.DONATE
    with pytest.raises(CoherenceStateTransitionError, match="no grants"):
        await worker._apply_operator_tensor_access("op", 7, prepared)


@pytest.mark.asyncio
async def test_port_access_rejects_conflicting_alias_intents():
    set_backend("numba_cpu")
    tensor = TransportTensor(np.arange(4, dtype=np.int64))
    operator = types.SimpleNamespace(
        spec=types.SimpleNamespace(
            tensor_access={
                "readonly_alias": TensorAccessIntent.READONLY,
                "writable_alias": TensorAccessIntent.WRITABLE,
            }
        )
    )
    worker = types.SimpleNamespace(
        _coherence_input_grants={},
        _coherence_input_tensors={},
        _coherence_input_intents={},
        _local_input_origins={},
        _coherence_input_bindings={},
        config=types.SimpleNamespace(
            require_graph_view=lambda: types.SimpleNamespace(
                operator=lambda _op_name: operator
            )
        ),
    )
    _bind(worker, "_apply_operator_tensor_access")
    prepared = types.SimpleNamespace(
        tensor_bindings=(
            types.SimpleNamespace(port="readonly_alias", tensor=tensor, grants=()),
            types.SimpleNamespace(port="writable_alias", tensor=tensor, grants=()),
        )
    )
    with pytest.raises(
        CoherenceStateTransitionError, match="conflicting access intents"
    ):
        await worker._apply_operator_tensor_access("op", 7, prepared)


@pytest.mark.asyncio
async def test_share_return_resolves_route_to_logical_subject_and_child_slot():
    parent_code = 3
    child_code = 4
    route_code = 11
    epoch = 7
    subject = CoherenceSubject(parent_code, 0, 19)
    worker = types.SimpleNamespace(
        _coherence_directory=CoherenceDirectory(),
        _coherence_revocations=set(),
        _local_spmd_code=parent_code,
        rank=0,
        _world_size=1,
    )
    _bind(worker, "_receive_coherence_frame", "_require_local_spmd_code")
    worker._coherence_subject = lambda owner_code, owner_rank, transfer_code: (
        CoherenceSubject(owner_code, owner_rank, 19)
    )
    worker._coherence_child_slot = lambda transfer_code, target_rank: 5
    worker._coherence_directory.publish_local_payload(subject, epoch)
    worker._coherence_directory.delegate_read(subject, epoch, child_id=5)
    header = build_coherence_frame(
        step=epoch,
        sender_spmd_code=child_code,
        sender_rank=2,
        sender_world_size=3,
        epoch=epoch,
        action=CoherenceAction.GRANT_RETURN,
        route_codes=(route_code,),
    )

    await worker._receive_coherence_frame(
        sender_spmd_code=child_code,
        frame=header.to_frame(()),
    )

    assert _entry(worker._coherence_directory, subject, epoch).child_share_bitmap == 0


@pytest.mark.asyncio
async def test_worker_coalesces_invalidation_during_parent_return() -> None:
    subject = CoherenceSubject(3, 0, 19)
    epoch = 7
    directory = CoherenceDirectory()
    directory.accept_parent_read(subject, epoch)
    local = directory.request_read(subject, epoch, ExecutionToken(epoch))
    assert isinstance(local, GrantReady)
    directory.release(local.token)
    returning = directory.begin_parent_return(subject, epoch)
    worker = types.SimpleNamespace(
        _coherence_directory=directory,
        _coherence_revocations=set(),
    )
    _bind(worker, "_invalidate_parent_grant")

    await worker._invalidate_parent_grant(
        subject,
        route_code=11,
        epoch=epoch,
        sender_rank=0,
    )

    directory.complete_parent_return(returning)


def test_reserved_flush_code_is_rejected_before_dispatch() -> None:
    from fluxway.transport.manifest import ENTRY_STATE_FLAGS

    header = build_coherence_frame(
        step=7,
        sender_spmd_code=4,
        sender_rank=0,
        sender_world_size=1,
        epoch=7,
        action=CoherenceAction.GRANT_RETURN,
        route_codes=(11,),
    )
    header.entry_i32[0, ENTRY_STATE_FLAGS] &= ~(15 << 20)
    header.entry_i32[0, ENTRY_STATE_FLAGS] |= 6 << 20

    with pytest.raises(ValueError):
        header.validate()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("access", "intent"),
    (
        ("writable", "WRITABLE"),
        ("donate", "DONATE"),
    ),
)
async def test_write_intent_suspends_for_rfo_and_retires_dirty_parent_copy(
    access,
    intent,
):
    set_backend("numba_cpu")
    parent_code = 3
    child_code = 4
    route_code = 11
    epoch = 7
    subject = CoherenceSubject(parent_code, 0, 19)
    parent_directory = CoherenceDirectory()
    child_directory = CoherenceDirectory()
    parent_directory.publish_local_payload(subject, epoch)
    parent_directory.delegate_read(subject, epoch, child_id=0)
    child_directory.accept_parent_read(subject, epoch)
    local_read = child_directory.request_read(subject, epoch, ExecutionToken(7))
    assert isinstance(local_read, GrantReady)
    input_grant = _InputCoherenceGrant(
        parent_spmd_id="parent",
        parent_rank=0,
        subject=subject,
        route_code=route_code,
        epoch=epoch,
        local_access_token=local_read.token,
    )
    child = types.SimpleNamespace(
        _coherence_directory=child_directory,
        _coherence_input_grants={("op", epoch): _grant_map(input_grant)},
        _coherence_revocations=set(),
        _coherence_input_tensors={},
        _coherence_input_intents={},
        _local_input_origins={},
        _coherence_alias_dependencies={},
        _allocation_registry=AllocationRegistry(),
        _predecessor_handlers={},
        _local_spmd_code=child_code,
        _coherence_sequencer=CoherenceSequencer(
            lambda _instances, _payload: asyncio.sleep(0),
            lambda _instance: asyncio.sleep(0),
        ),
        rank=0,
        _world_size=1,
        config=types.SimpleNamespace(
            require_graph_view=lambda: types.SimpleNamespace(
                operator=lambda _op_name: types.SimpleNamespace(
                    spec=types.SimpleNamespace(
                        tensor_access={"input": TensorAccessIntent(access)}
                    )
                )
            )
        ),
    )
    _bind(
        child,
        "_acquire_owned_inputs",
        "_apply_operator_tensor_access",
        "_rollback_ownership_reservations",
        "_return_input_coherence",
        "_require_local_spmd_code",
    )

    async def request_parent_ownership(
        incoming_subject,
        incoming_route_codes,
        incoming_epoch,
        action,
    ):
        assert incoming_subject == subject
        assert incoming_route_codes == (route_code,)
        assert incoming_epoch == epoch
        assert action is CoherenceAction.RFO
        parent_directory.return_child(subject, epoch, child_id=0)
        parent_write = parent_directory.request_write(
            subject,
            epoch,
            ExecutionToken(0),
        )
        assert isinstance(parent_write, GrantReady)
        parent_directory.delegate_ownership(parent_write.token, child_id=0)
        child_directory.accept_parent_ownership(subject, epoch)

    child._send_coherence_batch_to_parent = request_parent_ownership
    tensor = TransportTensor(np.arange(4, dtype=np.int64))
    child.config.require_graph_view = lambda: types.SimpleNamespace(
        operator=lambda _name: types.SimpleNamespace(
            spec=types.SimpleNamespace(
                tensor_access={"input": TensorAccessIntent(access)}
            )
        )
    )
    prepared = types.SimpleNamespace(
        tensor_bindings=(
            types.SimpleNamespace(port="input", tensor=tensor, grants=(input_grant,)),
        )
    )
    child._coherence_input_bindings = {("op", epoch): prepared.tensor_bindings}
    child._coherence_input_intents = {}
    await child._apply_operator_tensor_access("op", epoch, prepared)

    donated_grant = next(iter(child._coherence_input_grants[("op", epoch)].values()))
    assert donated_grant.local_access_token.kind is GrantKind.WRITE
    assert tensor._runtime_access_intent.name == intent
    assert _entry(child_directory, subject, epoch).state is CoherenceState.MODIFIED

    async def receive_return(_frames):
        parent_directory.return_ownership_child(
            subject,
            epoch,
            child_id=0,
            data_valid=False,
        )
        return (0,)

    child._predecessor_handlers = {"parent": {0: _Handler(receive_return)}}
    _attach_parent_return_runtime(child)
    await child._return_input_coherence("op", epoch)

    assert _entry(child_directory, subject, epoch).state is CoherenceState.INVALID
    assert _entry(parent_directory, subject, epoch).state is CoherenceState.INVALID


@pytest.mark.asyncio
async def test_failed_rfo_fences_without_speculative_remote_rollback() -> None:
    subject = CoherenceSubject(3, 0, 19)
    epoch = 7
    parent_directory = CoherenceDirectory()
    child_directory = CoherenceDirectory()
    parent_directory.publish_local_payload(subject, epoch)
    parent_directory.delegate_read(subject, epoch, child_id=0)
    child_directory.accept_parent_read(subject, epoch)
    local_read = child_directory.request_read(subject, epoch, ExecutionToken(epoch))
    assert isinstance(local_read, GrantReady)
    input_grant = _InputCoherenceGrant(
        parent_spmd_id="parent",
        parent_rank=0,
        subject=subject,
        route_code=11,
        epoch=epoch,
        local_access_token=local_read.token,
    )
    grants = _grant_map(input_grant)
    child = types.SimpleNamespace(
        _coherence_directory=child_directory,
        _coherence_input_grants={("op", epoch): grants},
        _coherence_input_intents={},
        _local_input_origins={},
    )
    _bind(
        child,
        "_acquire_owned_inputs",
        "_rollback_ownership_reservations",
    )

    async def fail_rfo(subject_arg, route_code, epoch_arg, action):
        if action is CoherenceAction.RFO:
            raise RuntimeError("RFO failed")
        assert action is CoherenceAction.GRANT_RETURN
        parent_directory.return_child(subject_arg, epoch_arg, child_id=0)

    async def fail_rfo_batch(subject_arg, route_codes, epoch_arg, action):
        assert route_codes == (11,)
        await fail_rfo(subject_arg, route_codes[0], epoch_arg, action)

    child._send_coherence_batch_to_parent = fail_rfo_batch
    child._send_coherence_to_parent = fail_rfo
    failures = []
    child._record_failure = lambda error, **context: failures.append((error, context))

    with pytest.raises(RuntimeError, match="RFO failed"):
        await child._acquire_owned_inputs("op", epoch, grants)

    assert grants
    assert len(failures) == 1
    assert failures[0][1]["phase"] == "COHERENCE_RFO"
    assert _entry(parent_directory, subject, epoch).state is CoherenceState.SHARED


@pytest.mark.asyncio
async def test_multi_subject_reservation_failure_rolls_back_earlier_writer() -> None:
    class FailSecondWrite:
        def __init__(self, inner):
            self.inner = inner
            self.count = 0

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def request_write(self, *args):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("second reservation failed")
            return self.inner.request_write(*args)

    inner = CoherenceDirectory()
    epoch = 7
    subjects = (
        CoherenceSubject(3, 0, 19),
        CoherenceSubject(3, 0, 20),
    )
    grants = {}
    for route_code, subject in enumerate(subjects, start=11):
        inner.accept_parent_read(subject, epoch)
        read = inner.request_read(subject, epoch, ExecutionToken(epoch))
        assert isinstance(read, GrantReady)
        grant = _InputCoherenceGrant(
            parent_spmd_id="parent",
            parent_rank=0,
            subject=subject,
            route_code=route_code,
            epoch=epoch,
            local_access_token=read.token,
        )
        grants[grant.identity] = grant
    worker = types.SimpleNamespace(
        _coherence_directory=FailSecondWrite(inner),
        _coherence_input_grants={("op", epoch): grants},
        _coherence_input_intents={},
        _local_input_origins={},
    )
    actions = []

    async def send_to_parent(_subject, _route_code, _epoch, action):
        actions.append(action)

    worker._send_coherence_to_parent = send_to_parent

    async def send_parent_group(routes):
        for route in routes:
            await send_to_parent(
                route.subject,
                route.route_code,
                route.epoch,
                CoherenceAction.GRANT_RETURN,
            )

    worker._parent_grant_return_runtime = ParentGrantReturnRuntime(
        inner,
        send_parent_group,
    )
    _bind(worker, "_acquire_owned_inputs", "_rollback_ownership_reservations")

    with pytest.raises(RuntimeError, match="second reservation failed"):
        await worker._acquire_owned_inputs("op", epoch, grants)

    assert not grants
    assert actions == [CoherenceAction.GRANT_RETURN, CoherenceAction.GRANT_RETURN]
    assert all(
        _entry(inner, subject, epoch).state is CoherenceState.INVALID
        for subject in subjects
    )


@pytest.mark.asyncio
async def test_alias_lineage_delays_parent_return_until_downstream_quiescence() -> None:
    set_backend("numba_cpu")
    epoch = 7
    input_subject = CoherenceSubject(3, 0, 19)
    output_subject = CoherenceSubject(4, 0, 23)
    directory = CoherenceDirectory()
    directory.accept_parent_read(input_subject, epoch)
    local_read = directory.request_read(
        input_subject,
        epoch,
        ExecutionToken(epoch),
    )
    assert isinstance(local_read, GrantReady)
    directory.publish_local_payload(output_subject, epoch)
    directory.delegate_read(output_subject, epoch, child_id=0)
    tensor = TransportTensor(np.arange(4, dtype=np.int64))
    grant = _InputCoherenceGrant(
        parent_spmd_id="parent",
        parent_rank=0,
        subject=input_subject,
        route_code=11,
        epoch=epoch,
        local_access_token=local_read.token,
    )

    async def receive_return(_frames):
        return (0,)

    worker = types.SimpleNamespace(
        _coherence_directory=directory,
        _coherence_input_grants={("op", epoch): _grant_map(grant)},
        _coherence_input_bindings={},
        _coherence_input_intents={},
        _coherence_alias_dependencies={
            ("op", epoch): {CoherenceInstance(output_subject, epoch)}
        },
        _allocation_registry=AllocationRegistry(),
        _coherence_input_tensors={("op", epoch): {tensor}},
        _local_input_origins={},
        _coherence_revocations=set(),
        _predecessor_handlers={"parent": {0: _Handler(receive_return)}},
        _local_spmd_code=4,
        rank=0,
        _world_size=1,
    )
    _bind(worker, "_return_input_coherence", "_require_local_spmd_code")
    _attach_parent_return_runtime(worker)

    returning = asyncio.create_task(worker._return_input_coherence("op", epoch))
    await asyncio.sleep(0)

    assert not returning.done()
    assert tensor.flat_values is not None

    directory.return_child(output_subject, epoch, child_id=0)
    await asyncio.wait_for(returning, timeout=0.1)

    with pytest.raises(CoherenceStateTransitionError, match="released"):
        _ = tensor.flat_values


@pytest.mark.asyncio
async def test_input_retirement_accepts_reentrant_next_epoch_after_return() -> None:
    set_backend("numba_cpu")
    epoch = 7
    subject = CoherenceSubject(3, 0, 19)
    instance = CoherenceInstance(subject, epoch)
    directory = CoherenceDirectory()
    directory.accept_parent_read(subject, epoch)
    local_read = directory.request_read(subject, epoch, ExecutionToken(epoch))
    assert isinstance(local_read, GrantReady)
    tensor = TransportTensor(np.arange(4, dtype=np.int64))
    tensor._bind_coherence_instances((instance,))
    registry = AllocationRegistry()
    registry.bind(instance, tensor._runtime_allocation_descriptor())
    grant = _InputCoherenceGrant(
        parent_spmd_id="parent",
        parent_rank=0,
        subject=subject,
        route_code=11,
        epoch=epoch,
        local_access_token=local_read.token,
    )

    async def receive_return(_frames):
        directory.accept_parent_read(subject, epoch + 1)
        return (0,)

    worker = types.SimpleNamespace(
        _coherence_directory=directory,
        _coherence_input_grants={("op", epoch): _grant_map(grant)},
        _coherence_input_bindings={},
        _coherence_input_intents={},
        _coherence_alias_dependencies={},
        _allocation_registry=registry,
        _coherence_input_tensors={("op", epoch): {tensor}},
        _local_input_origins={},
        _coherence_revocations=set(),
        _predecessor_handlers={"parent": {0: _Handler(receive_return)}},
        _local_spmd_code=4,
        rank=0,
        _world_size=1,
    )
    _bind(worker, "_return_input_coherence", "_require_local_spmd_code")
    _attach_parent_return_runtime(worker)

    await worker._return_input_coherence("op", epoch)

    assert not registry.is_bound(instance)
    assert _entry(directory, subject, epoch + 1).epoch == epoch + 1
    with pytest.raises(CoherenceStateTransitionError, match="released"):
        _ = tensor.flat_values


def test_alias_dependency_is_derived_from_runtime_allocation_identity() -> None:
    set_backend("numba_cpu")
    tensor = TransportTensor(np.arange(4, dtype=np.int64))
    route = types.SimpleNamespace(output_index=0, logical_tuuid=23)
    plan = types.SimpleNamespace(
        successor_routes=(types.SimpleNamespace(tensor_routes=(route,)),)
    )
    worker = types.SimpleNamespace(
        _coherence_input_tensors={("op", 7): {tensor}},
        _coherence_input_bindings={},
        _coherence_input_intents={},
        _coherence_alias_dependencies={},
        _local_spmd_code=4,
        rank=0,
        op_manager=types.SimpleNamespace(
            get_operator_spec=lambda _op_name: types.SimpleNamespace(
                output_treedef=None
            )
        ),
        config=types.SimpleNamespace(
            require_graph_view=lambda: types.SimpleNamespace(
                operator=lambda _op_name: types.SimpleNamespace(
                    spec=types.SimpleNamespace(
                        tensor_access={"input": TensorAccessIntent.READONLY}
                    )
                )
            )
        ),
    )
    _bind(
        worker,
        "_register_coherence_alias_dependencies",
        "_normalize_local_plan_result",
        "_result_requires_send",
        "_require_local_spmd_code",
        "_commit_allocation_handoffs",
    )
    worker._operator_send_context = lambda _op_name: types.SimpleNamespace(
        compiled_plan=plan
    )

    worker._register_coherence_alias_dependencies("op", 7, tensor)

    assert worker._coherence_alias_dependencies[("op", 7)] == {
        CoherenceInstance(CoherenceSubject(4, 0, 23), 7)
    }


def test_strict_donate_handoffs_exact_allocation_to_new_ssa_instance() -> None:
    set_backend("numba_cpu")
    step = 7
    old = CoherenceInstance(CoherenceSubject(3, 0, 19), step)
    new = CoherenceInstance(CoherenceSubject(4, 0, 23), step)
    tensor = TransportTensor(np.arange(4, dtype=np.int64))
    tensor._bind_coherence_instances((old,))
    tensor._bind_coherence_ticket_id(31)
    output = tensor.zero_copy_alias()
    assert output.coherence_ticket_id is None
    directory = CoherenceDirectory()
    directory.publish_local_payload(old.subject, old.epoch)
    write = directory.request_write(old.subject, old.epoch, ExecutionToken(step))
    assert isinstance(write, GrantReady)
    registry = AllocationRegistry()
    registry.bind(old, tensor._runtime_allocation_descriptor())
    route = types.SimpleNamespace(output_index=0, logical_tuuid=23)
    plan = types.SimpleNamespace(
        successor_routes=(types.SimpleNamespace(tensor_routes=(route,)),)
    )
    worker = types.SimpleNamespace(
        _coherence_directory=directory,
        _allocation_registry=registry,
        _coherence_input_tensors={("op", step): {tensor}},
        _coherence_input_bindings={},
        _coherence_input_intents={},
        _coherence_alias_dependencies={},
        _local_spmd_code=4,
        rank=0,
        op_manager=types.SimpleNamespace(
            get_operator_spec=lambda _op_name: types.SimpleNamespace(
                output_treedef=None
            )
        ),
        config=types.SimpleNamespace(
            require_graph_view=lambda: types.SimpleNamespace(
                operator=lambda _op_name: types.SimpleNamespace(
                    spec=types.SimpleNamespace(
                        tensor_access={"input": TensorAccessIntent.DONATE}
                    )
                )
            )
        ),
    )
    _bind(
        worker,
        "_register_coherence_alias_dependencies",
        "_normalize_local_plan_result",
        "_result_requires_send",
        "_require_local_spmd_code",
        "_commit_allocation_handoffs",
    )
    worker._operator_send_context = lambda _op_name: types.SimpleNamespace(
        compiled_plan=plan
    )
    worker._coherence_input_bindings[("op", step)] = (
        types.SimpleNamespace(
            port="input",
            tensor=tensor,
            grants=(),
        ),
    )

    handoffs = worker._register_coherence_alias_dependencies("op", step, output)

    assert registry.is_bound(old)
    assert not registry.is_bound(new)
    worker._commit_allocation_handoffs(handoffs, step)

    assert registry.binding(new).descriptor.owner is tensor.flat_values
    assert tensor._coherence_instances == (old,)
    assert output._coherence_instances == (new,)
    assert tensor.coherence_ticket_id == 31
    assert output.coherence_ticket_id is None


def test_alias_dependency_rejects_noncanonical_output_epoch() -> None:
    set_backend("numba_cpu")
    input_tensor = TransportTensor(np.arange(4, dtype=np.int64))
    output_tensor = TransportTensor(
        input_tensor.flat_values,
        source_step=99,
    )
    route = types.SimpleNamespace(output_index=0, logical_tuuid=23)
    plan = types.SimpleNamespace(
        successor_routes=(types.SimpleNamespace(tensor_routes=(route,)),)
    )
    worker = types.SimpleNamespace(
        _coherence_input_tensors={("op", 7): {input_tensor}},
        _coherence_alias_dependencies={},
        _local_spmd_code=4,
        rank=0,
        op_manager=types.SimpleNamespace(
            get_operator_spec=lambda _op_name: types.SimpleNamespace(
                output_treedef=None
            )
        ),
        config=types.SimpleNamespace(
            require_graph_view=lambda: types.SimpleNamespace(
                operator=lambda _op_name: types.SimpleNamespace(
                    spec=types.SimpleNamespace(
                        tensor_access={"input": TensorAccessIntent.READONLY}
                    )
                )
            )
        ),
    )
    _bind(
        worker,
        "_register_coherence_alias_dependencies",
        "_normalize_local_plan_result",
        "_result_requires_send",
        "_require_local_spmd_code",
    )
    worker._operator_send_context = lambda _op_name: types.SimpleNamespace(
        compiled_plan=plan
    )

    with pytest.raises(ValueError, match="source_step=99"):
        worker._register_coherence_alias_dependencies("op", 7, output_tensor)

    assert not worker._coherence_alias_dependencies


@pytest.mark.asyncio
async def test_send_barrier_waits_for_registration_not_delivery() -> None:
    delivery_gate = asyncio.Event()
    delivery = asyncio.create_task(delivery_gate.wait())
    result = ExecutionResult(ExecutionStatus.SUCCESS, data=object())
    worker = types.SimpleNamespace(
        _coherence_alias_dependencies={("op", 7): {object()}},
    )

    async def send_output(*_args, **_kwargs):
        return (delivery,)

    worker._send_output = send_output
    _bind(worker, "_send_local_plan_result", "_await_send_registration")
    registration = asyncio.create_task(
        worker._send_local_plan_result("op", 7, result, {})
    )
    token = LocalSendToken("op", 7, result, registration)

    await worker._await_send_registration(token)

    assert registration.done()
    assert not delivery.done()
    delivery.cancel()
    await asyncio.gather(delivery, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_batch_delegation_rolls_back_prior_child_grants() -> None:
    directory = CoherenceDirectory()
    first = CoherenceSubject(1, 0, 10)
    blocked = CoherenceSubject(1, 0, 11)
    directory.publish_local_payload(blocked, 7)
    blocker = directory.request_write(blocked, 7, ExecutionToken(9))
    assert isinstance(blocker, GrantReady)
    sequencer: Any = types.SimpleNamespace(
        commit_payload=lambda _instances, payload: payload()
    )
    runtime = CoherenceBatchRetireRuntime(directory, sequencer)
    plan: Any = types.SimpleNamespace(
        sender_spmd_code=1,
        rank=0,
        coherence_logical_tuuid=lambda code: 10 if code == 1 else 11,
        coherence_child_slot=lambda _code, _rank: 0,
        describe_publication=lambda batches: PublicationBatch(
            tuple(batches),
            (InstancePublication(CoherenceInstance(first, 7), (0,)),),
        ),
    )
    batch = TransportFrameBatch(
        context_index=0,
        target_rank=0,
        batch_seq=0,
        frames=(),
        delegations=(
            TransportDelegationDescriptor(1, 7),
            TransportDelegationDescriptor(2, 7),
        ),
    )
    receipts = _ChildDeliveryReceipts(directory)
    task = asyncio.create_task(
        runtime._delegate_publication_reads(plan, (batch,), receipts)
    )
    await asyncio.sleep(0)

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert _entry(directory, first, 7).child_share_bitmap == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery_fails", (False, True))
async def test_delivery_failure_rolls_back_but_success_retentions_child_grant(
    delivery_fails,
) -> None:
    directory = CoherenceDirectory()
    subject = CoherenceSubject(1, 0, 12)
    directory.publish_local_payload(subject, 7)
    sequencer: Any = types.SimpleNamespace(
        commit_payload=lambda _instances, payload: payload()
    )
    runtime = CoherenceBatchRetireRuntime(directory, sequencer)
    plan: Any = types.SimpleNamespace(
        sender_spmd_code=1,
        rank=0,
        coherence_instance=lambda transfer_code, epoch: CoherenceInstance(
            subject, epoch
        ),
        coherence_child_slot=lambda transfer_code, target_rank: 0,
        describe_publication=lambda batches: PublicationBatch(
            tuple(batches),
            (InstancePublication(CoherenceInstance(subject, 7), (0,)),),
        ),
    )
    batch = TransportFrameBatch(
        context_index=0,
        target_rank=0,
        batch_seq=0,
        frames=(),
        delegations=(TransportDelegationDescriptor(1, 7),),
    )

    async def delivery():
        if delivery_fails:
            raise RuntimeError("delivery failed")
        return ReceiverIngressCompletion(None)

    task = asyncio.create_task(
        runtime.retire_publication(plan, (batch,), lambda _batch: delivery())
    )
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)

    expected = 0 if delivery_fails else 1
    assert _entry(directory, subject, 7).child_share_bitmap == expected


def test_delegated_reads_preserve_batch_order_and_drop_duplicates() -> None:
    plan: Any = types.SimpleNamespace(
        sender_spmd_code=7,
        rank=1,
        coherence_logical_tuuid=lambda transfer_code: transfer_code + 100,
        coherence_instance=lambda transfer_code, epoch: CoherenceInstance(
            CoherenceSubject(7, 1, transfer_code + 100), epoch
        ),
        coherence_child_slot=lambda transfer_code, target_rank: (
            target_rank + transfer_code
        ),
    )
    batch = TransportFrameBatch(
        context_index=0,
        target_rank=2,
        batch_seq=0,
        frames=(),
        delegations=(
            TransportDelegationDescriptor(1, 1),
            TransportDelegationDescriptor(1, 0),
            TransportDelegationDescriptor(1, 1),
        ),
    )

    delegated_reads = _delegated_reads(plan, batch)

    assert [(epoch, slot) for _subject, epoch, slot in delegated_reads] == [
        (1, 3),
        (0, 3),
    ]
    assert len(delegated_reads) == 2


@pytest.mark.asyncio
async def test_late_delivery_failure_cannot_revoke_overlapping_success() -> None:
    directory = CoherenceDirectory()
    subject = CoherenceSubject(1, 0, 13)
    directory.publish_local_payload(subject, 7)
    sequencer: Any = types.SimpleNamespace(
        commit_payload=lambda _instances, payload: payload()
    )
    runtime = CoherenceBatchRetireRuntime(directory, sequencer)
    plan: Any = types.SimpleNamespace(
        sender_spmd_code=1,
        rank=0,
        coherence_instance=lambda transfer_code, epoch: CoherenceInstance(
            subject, epoch
        ),
        coherence_child_slot=lambda transfer_code, target_rank: 0,
        describe_publication=lambda batches: PublicationBatch(
            tuple(batches),
            (InstancePublication(CoherenceInstance(subject, 7), (0,)),),
        ),
    )
    batch = TransportFrameBatch(
        context_index=0,
        target_rank=0,
        batch_seq=0,
        frames=(),
        delegations=(TransportDelegationDescriptor(1, 7),),
    )
    fail_first = asyncio.Event()

    async def first_delivery():
        await fail_first.wait()
        raise RuntimeError("first delivery failed")

    async def second_delivery():
        return ReceiverIngressCompletion(None)

    first_task = asyncio.create_task(
        runtime.retire_publication(plan, (batch,), lambda _batch: first_delivery())
    )
    second_task = asyncio.create_task(
        runtime.retire_publication(plan, (batch,), lambda _batch: second_delivery())
    )
    await second_task
    await asyncio.sleep(0)
    fail_first.set()
    await asyncio.gather(first_task, return_exceptions=True)
    await asyncio.sleep(0)

    assert _entry(directory, subject, 7).child_share_bitmap == 1
    directory.return_child(subject, 7, child_id=0)
