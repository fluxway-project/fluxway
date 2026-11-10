import asyncio
import threading

import pytest

from fluxway.common.coherence import (
    CoherenceDirectory,
    CoherenceSubject,
    StaleEpochError,
    CoherenceStateTransitionError,
    CoherenceInstance,
)
from fluxway.spmd.worker.coherence_sequencer import CoherenceSequencer


def _instance(subject: CoherenceSubject, epoch: int) -> CoherenceInstance:
    return CoherenceInstance(subject, epoch)


@pytest.mark.asyncio
async def test_frame_ready_out_of_order_retires_in_epoch_order() -> None:
    directory = CoherenceDirectory()
    subject = CoherenceSubject(1, 0, 1)
    retired: list[int] = []

    async def retire(instances, payload):
        del instances
        retired.append(payload)
        return payload

    async def admit(instance):
        directory.admit(instance)

    sequencer = CoherenceSequencer(retire, admit)
    first = _instance(subject, 0)
    second = _instance(subject, 1)
    sequencer.issue_intent(first)
    sequencer.issue_intent(second)

    second_task = asyncio.create_task(sequencer.commit_payload((second,), 1))
    await asyncio.sleep(0)
    assert retired == []
    first_task = asyncio.create_task(sequencer.commit_payload((first,), 0))
    await asyncio.sleep(0)
    sequencer.close_intent(first)
    assert await first_task == 0
    sequencer.close_intent(second)
    assert await second_task == 1
    assert retired == [0, 1]


@pytest.mark.asyncio
async def test_subjects_progress_independently_and_multi_subject_payload_is_atomic() -> (
    None
):
    directory = CoherenceDirectory()
    subject_a = CoherenceSubject(1, 0, 1)
    subject_b = CoherenceSubject(1, 0, 2)
    retired: list[str] = []

    async def retire(instances, payload):
        retired.append(payload)
        assert tuple(sorted(instances)) == instances
        return payload

    async def admit(instance):
        directory.admit(instance)

    sequencer = CoherenceSequencer(retire, admit)
    a0 = _instance(subject_a, 0)
    a1 = _instance(subject_a, 1)
    b0 = _instance(subject_b, 0)
    for instance in (a0, a1, b0):
        sequencer.issue_intent(instance)

    a1_task = asyncio.create_task(sequencer.commit_payload((a1,), "a1"))
    b0_task = asyncio.create_task(sequencer.commit_payload((b0,), "b0"))
    await asyncio.sleep(0)
    sequencer.close_intent(b0)
    assert await b0_task == "b0"
    assert retired == ["b0"]
    a0_task = asyncio.create_task(sequencer.commit_payload((a0,), "a0"))
    await asyncio.sleep(0)
    sequencer.close_intent(a0)
    sequencer.close_intent(a1)
    assert await a0_task == "a0"
    assert await a1_task == "a1"
    assert retired == ["b0", "a0", "a1"]


@pytest.mark.asyncio
async def test_same_epoch_multiple_deliveries_complete_before_next_epoch() -> None:
    directory = CoherenceDirectory()
    subject = CoherenceSubject(1, 0, 3)
    retired: list[str] = []
    release = asyncio.Event()

    async def retire(instances, payload):
        del instances
        if payload == "first":
            await release.wait()
        retired.append(payload)
        return payload

    async def admit(instance):
        directory.admit(instance)

    sequencer = CoherenceSequencer(retire, admit)
    epoch0 = _instance(subject, 0)
    epoch1 = _instance(subject, 1)
    sequencer.issue_intent(epoch0)
    sequencer.issue_intent(epoch1)
    first_task = asyncio.create_task(sequencer.commit_payload((epoch0,), "first"))
    second_delivery_task = asyncio.create_task(
        sequencer.commit_payload((epoch0,), "second")
    )
    await asyncio.sleep(0)
    sequencer.close_intent(epoch0)
    next_task = asyncio.create_task(sequencer.commit_payload((epoch1,), "next"))
    await asyncio.sleep(0)
    assert retired == []
    sequencer.close_intent(epoch1)
    release.set()
    assert await first_task == "first"
    assert await second_delivery_task == "second"
    assert await next_task == "next"
    assert retired == ["first", "second", "next"]


@pytest.mark.asyncio
async def test_stale_directory_admission_is_not_retried() -> None:
    directory = CoherenceDirectory()
    subject = CoherenceSubject(1, 0, 4)

    async def admit(instance):
        if instance.epoch < 2:
            raise StaleEpochError("stale epoch error")
        directory.admit(instance)

    sequencer = CoherenceSequencer(lambda _instances, _payload: asyncio.sleep(0), admit)
    stale = _instance(subject, 1)
    sequencer.issue_intent(stale)
    stale_task = asyncio.create_task(sequencer.commit_payload((stale,), "stale"))
    sequencer.close_intent(stale)
    with pytest.raises(StaleEpochError):
        await stale_task


@pytest.mark.asyncio
async def test_cross_subject_dependency_cycle_is_rejected_without_pending_tasks() -> (
    None
):
    subject_a = CoherenceSubject(1, 0, 5)
    subject_b = CoherenceSubject(1, 0, 6)
    a0 = _instance(subject_a, 0)
    a1 = _instance(subject_a, 1)
    b0 = _instance(subject_b, 0)
    b1 = _instance(subject_b, 1)
    retired: list[str] = []

    async def retire(_instances, payload):
        retired.append(payload)
        return payload

    async def admit(_instance):
        return None

    sequencer = CoherenceSequencer(retire, admit)
    for instance in (a0, a1, b0, b1):
        sequencer.issue_intent(instance)
    first = asyncio.create_task(sequencer.commit_payload((a0, b1), "a0+b1"))
    second = asyncio.create_task(sequencer.commit_payload((a1, b0), "a1+b0"))
    for instance in (a0, a1, b0, b1):
        sequencer.close_intent(instance)
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert all(isinstance(result, CoherenceStateTransitionError) for result in results)
    assert retired == []


@pytest.mark.asyncio
async def test_in_flight_predecessor_is_not_reported_as_dependency_cycle() -> None:
    subject_a = CoherenceSubject(1, 0, 7)
    subject_b = CoherenceSubject(1, 0, 8)
    a0 = _instance(subject_a, 0)
    a1 = _instance(subject_a, 1)
    b0 = _instance(subject_b, 0)
    b1 = _instance(subject_b, 1)
    release = asyncio.Event()
    retired: list[str] = []

    async def retire(_instances, payload):
        if payload == "first":
            await release.wait()
        retired.append(payload)
        return payload

    async def admit(_instance):
        return None

    sequencer = CoherenceSequencer(retire, admit)
    for instance in (a0, a1, b0, b1):
        sequencer.issue_intent(instance)
    first = asyncio.create_task(sequencer.commit_payload((a0, b0), "first"))
    await asyncio.sleep(0)
    sequencer.close_intent(a0)
    sequencer.close_intent(b0)
    second = asyncio.create_task(sequencer.commit_payload((a1, b1), "second"))
    await asyncio.sleep(0)
    assert not second.done()
    sequencer.close_intent(a1)
    sequencer.close_intent(b1)
    release.set()
    assert await first == "first"
    assert await second == "second"
    assert retired == ["first", "second"]


@pytest.mark.asyncio
async def test_commit_payload_deferred_await_after_drain_retirement() -> None:
    subject = CoherenceSubject(1, 0, 9)
    retired: list[str] = []

    async def retire(_instances, payload):
        retired.append(payload)
        return f"OK_{payload}"

    async def admit(_instance):
        return None

    sequencer = CoherenceSequencer(retire, admit)
    inst = _instance(subject, 0)
    sequencer.issue_intent(inst)

    # 1. commit_payload creates ticket with future=None while unsealed
    coro = sequencer.commit_payload((inst,), "payload_0")

    # 2. close_intent triggers drain
    sequencer.close_intent(inst)

    # 3. Allow drain to complete retirement before coro is awaited
    await asyncio.sleep(0.01)
    assert retired == ["payload_0"]

    # 4. Await coro after retirement is already finished
    result = await asyncio.wait_for(coro, timeout=1.0)
    assert result == "OK_payload_0"


@pytest.mark.asyncio
async def test_consumer_lease_reservation_transfers_ownership_before_retirement() -> (
    None
):
    subject = CoherenceSubject(1, 0, 10)
    instance = _instance(subject, 0)
    retired: list[str] = []
    reservation_ready = asyncio.Event()
    release = asyncio.Event()
    reservations = []

    async def retire(_instances, payload):
        reservations.append(sequencer.reserve_consumer_lease((instance,)))
        reservation_ready.set()
        await release.wait()
        retired.append(payload)

    async def admit(_instance):
        return None

    sequencer = CoherenceSequencer(retire, admit)
    sequencer.issue_intent(instance)
    delivery = asyncio.create_task(sequencer.commit_payload((instance,), "payload"))
    sequencer.close_intent(instance)
    await reservation_ready.wait()
    assert not delivery.done()
    assert retired == []

    lease = reservations[0].claim()
    lease.release()
    release.set()
    await delivery
    assert retired == ["payload"]


@pytest.mark.asyncio
async def test_consumer_lease_release_from_worker_thread_is_non_blocking() -> None:
    subject = CoherenceSubject(1, 0, 12)
    instance = _instance(subject, 0)
    reservation_ready = asyncio.Event()
    release_callback = asyncio.Event()
    reservations = []

    async def retire(instances, _payload):
        reservations.append(sequencer.reserve_consumer_lease(instances))
        reservation_ready.set()
        await release_callback.wait()

    async def admit(_instance):
        return None

    sequencer = CoherenceSequencer(retire, admit)
    sequencer.issue_intent(instance)
    delivery = asyncio.create_task(sequencer.commit_payload((instance,), "payload"))
    sequencer.close_intent(instance)
    await reservation_ready.wait()
    lease = reservations[0].claim()
    thread = threading.Thread(target=lease.release)
    thread.start()
    thread.join()
    release_callback.set()
    await delivery


@pytest.mark.asyncio
async def test_publication_frontier_decoupled_from_reclamation_frontier() -> None:
    subject = CoherenceSubject(1, 0, 100)
    inst0 = _instance(subject, 0)
    inst1 = _instance(subject, 1)

    retired: list[str] = []

    async def retire(_instances, payload):
        retired.append(payload)
        return f"result_{payload}"

    async def admit(_instance):
        return None

    sequencer = CoherenceSequencer(retire, admit)
    sequencer.issue_intent(inst0)
    task0 = asyncio.create_task(sequencer.commit_payload((inst0,), "p0"))
    await asyncio.sleep(0.01)

    lease0 = sequencer.acquire_consumer_lease((inst0,))
    sequencer.close_intent(inst0)
    assert await task0 == "result_p0"

    # Epoch 1 must deliver payload even though Epoch 0 consumer lease is active
    sequencer.issue_intent(inst1)
    task1 = asyncio.create_task(sequencer.commit_payload((inst1,), "p1"))
    sequencer.close_intent(inst1)
    await asyncio.sleep(0.01)
    assert task1.done()
    assert await task1 == "result_p1"
    assert retired == ["p0", "p1"]

    # Epoch 0 is retained in intent index while lease0 is active
    assert inst0 in sequencer._intent_index
    assert sequencer._intent_index[inst0].active_consumer_leases == 1

    # Releasing lease0 reclaims Epoch 0
    lease0.release()
    assert inst0 not in sequencer._intent_index


@pytest.mark.asyncio
async def test_precise_ticket_level_lease_binding() -> None:
    subject = CoherenceSubject(1, 0, 200)
    inst = _instance(subject, 0)

    async def retire(_instances, payload):
        return f"result_{payload}"

    async def admit(_instance):
        return None

    sequencer = CoherenceSequencer(retire, admit)
    sequencer.issue_intent(inst)

    task_a = asyncio.create_task(
        sequencer.commit_payload((inst,), "payload_A", ticket_id=11)
    )
    task_b = asyncio.create_task(
        sequencer.commit_payload((inst,), "payload_B", ticket_id=22)
    )
    await asyncio.sleep(0.01)

    # Bind specifically to ticket 22
    lease_b = sequencer.acquire_consumer_lease((inst,), ticket_id=22)
    assert lease_b._bindings[0][0].ticket_id == 22
    assert lease_b._bindings[0][0].active_consumer_leases == 1

    sequencer.close_intent(inst)
    assert await task_a == "result_payload_A"
    assert await task_b == "result_payload_B"

    # Ticket 11 was retired and removed from sequencer
    assert 11 not in sequencer._tickets
    # Ticket 22 is retained because of active consumer lease
    assert 22 in sequencer._tickets
    assert inst in sequencer._intent_index

    lease_b.release()
    assert 22 not in sequencer._tickets
    assert inst not in sequencer._intent_index


@pytest.mark.asyncio
async def test_ambiguous_tickets_without_ticket_id_raises_error():
    async def deliver(instances, value):
        return f"result_{value}"

    async def _admit(_instance):
        pass

    sequencer = CoherenceSequencer(deliver, _admit)
    subject = CoherenceSubject(1, 0, 88)
    inst = CoherenceInstance(subject, 0)
    sequencer.issue_intent(inst)

    t1 = asyncio.create_task(
        sequencer.commit_payload((inst,), "payload_1", ticket_id=101)
    )
    t2 = asyncio.create_task(
        sequencer.commit_payload((inst,), "payload_2", ticket_id=102)
    )

    with pytest.raises(CoherenceStateTransitionError, match="ambiguous tickets"):
        sequencer.acquire_consumer_lease((inst,))

    lease1 = sequencer.acquire_consumer_lease((inst,), ticket_id=101)
    assert lease1._bindings[0][0].ticket_id == 101
    lease1.release()

    sequencer.close_intent(inst)
    await t1
    await t2


def test_fsm_state_setter_enforces_transition_rules():
    from fluxway.spmd.worker.coherence_sequencer import _Ticket, _TicketPhase

    subject = CoherenceSubject(1, 0, 99)
    inst = CoherenceInstance(subject, 0)
    ticket = _Ticket(instances=(inst,), payload="test", future=None)

    ticket.transition_to(_TicketPhase.ADMITTING)
    ticket.transition_to(_TicketPhase.DELIVERING)
    ticket.transition_to(_TicketPhase.DELIVERED)

    with pytest.raises(
        CoherenceStateTransitionError, match="illegal ticket state transition"
    ):
        ticket.state = _TicketPhase.READY


@pytest.mark.asyncio
async def test_transport_tensor_ticket_id_lease_propagation():
    from fluxway.common.types.tensor import TransportTensor
    import numpy as np

    deliver_gate = asyncio.Event()

    async def deliver(instances, value):
        tensor._reserve_consumer_lease()
        await deliver_gate.wait()
        return value

    async def _admit(_instance):
        pass

    sequencer = CoherenceSequencer(deliver, _admit)
    subject = CoherenceSubject(1, 0, 77)
    inst = CoherenceInstance(subject, 0)
    sequencer.issue_intent(inst)

    ticket_id = sequencer.allocate_ticket_id()
    tensor = TransportTensor(np.array([1, 2, 3]))
    tensor._bind_coherence_instances((inst,))
    tensor._bind_coherence_ticket_id(ticket_id)
    tensor._bind_consumer_lease_owner(sequencer)

    task = asyncio.create_task(
        sequencer.commit_payload((inst,), "data", ticket_id=ticket_id)
    )
    sequencer.close_intent(inst)
    await asyncio.sleep(0.01)

    lease = tensor.acquire_consumer_lease()
    assert tensor.coherence_ticket_id == ticket_id

    deliver_gate.set()
    assert await task == "data"
    assert ticket_id in sequencer._tickets

    lease.release()
    assert ticket_id not in sequencer._tickets
