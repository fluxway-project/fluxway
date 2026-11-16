"""Ordered coherence payload retirement for asynchronous frame construction."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from dataclasses import dataclass, field
import logging
from enum import StrEnum
import threading
from typing import Any, TypeVar

from fluxway.common.coherence import CoherenceStateTransitionError, CoherenceInstance

PayloadT = TypeVar("PayloadT")
_MISSING = object()
logger = logging.getLogger(__name__)
RetirePayload = Callable[[tuple[CoherenceInstance, ...], PayloadT], Awaitable[Any]]
AdmitInstance = Callable[[CoherenceInstance], Awaitable[None]]


class IntentPhase(StrEnum):
    OPEN = "open"
    SEALED = "sealed"
    DELIVERED = "delivered"
    RECLAIMED = "reclaimed"


@dataclass(slots=True)
class _Intent:
    instance: CoherenceInstance
    phase: IntentPhase = IntentPhase.OPEN
    payloads: deque[_Ticket[Any]] = field(default_factory=deque)
    active_payloads: int = 0
    active_consumer_leases: int = 0
    admitted: bool = False
    admission_task: asyncio.Task[None] | None = None

    @property
    def sealed(self) -> bool:
        return self.phase in (
            IntentPhase.SEALED,
            IntentPhase.DELIVERED,
            IntentPhase.RECLAIMED,
        )

    @sealed.setter
    def sealed(self, value: bool) -> None:
        if value and self.phase is IntentPhase.OPEN:
            self.phase = IntentPhase.SEALED
            self._refresh_phase()
        elif not value and self.phase is not IntentPhase.OPEN:
            raise CoherenceStateTransitionError(
                f"cannot unseal coherence intent: {self.instance}."
            )

    @property
    def is_delivery_completed(self) -> bool:
        return self.sealed and all(t.delivery_complete for t in self.payloads)

    @property
    def is_completed(self) -> bool:
        return (
            self.sealed
            and self.active_payloads == 0
            and self.active_consumer_leases == 0
        )

    def seal(self) -> None:
        if self.phase is not IntentPhase.OPEN:
            raise CoherenceStateTransitionError(
                f"coherence intent is already sealed: {self.instance}."
            )
        self.phase = IntentPhase.SEALED
        self._refresh_phase()

    def mark_admitted(self) -> None:
        self.admitted = True

    def add_ticket(self, ticket: _Ticket[Any]) -> None:
        if self.phase is not IntentPhase.OPEN:
            raise CoherenceStateTransitionError(
                f"cannot add ticket to sealed intent: {self.instance}."
            )
        self.payloads.append(ticket)
        self.active_payloads += 1

    def remove_ticket(self, ticket: _Ticket[Any]) -> None:
        if ticket not in self.payloads:
            raise CoherenceStateTransitionError(
                f"ticket not found in intent payloads for {self.instance}."
            )
        self.payloads.remove(ticket)
        self.active_payloads -= 1
        if self.active_payloads < 0:
            raise CoherenceStateTransitionError(
                f"coherence payload count underflow for {self.instance}."
            )
        self._refresh_phase()

    def acquire_lease(self) -> None:
        self.active_consumer_leases += 1

    def release_lease(self) -> None:
        if self.active_consumer_leases <= 0:
            raise CoherenceStateTransitionError(
                f"coherence intent lease count underflow for {self.instance}."
            )
        self.active_consumer_leases -= 1
        self._refresh_phase()

    def _refresh_phase(self) -> None:
        if self.is_completed:
            self.phase = IntentPhase.RECLAIMED
        elif self.is_delivery_completed:
            self.phase = IntentPhase.DELIVERED

    def next_ticket(self) -> _Ticket[Any] | None:
        return next(iter(self.payloads), None)

    def owns_next_ticket(self, ticket: _Ticket[Any]) -> bool:
        return self.next_ticket() is ticket

    def remove_next_ticket(self, ticket: _Ticket[Any]) -> None:
        if not self.owns_next_ticket(ticket):
            raise CoherenceStateTransitionError(
                "coherence intent next ticket changed during retirement."
            )
        self.remove_ticket(ticket)


class _TicketPhase(StrEnum):
    READY = "ready"
    ADMITTING = "admitting"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    RETIRED = "retired"
    ABORTED = "aborted"


class ConsumerLease:
    __slots__ = (
        "_bindings",
        "_lock",
        "_released",
        "_reservation_id",
        "_sequencer",
    )

    def __init__(
        self,
        sequencer: CoherenceSequencer[Any],
        bindings: tuple[tuple[_Ticket[Any], _Intent], ...],
        reservation_id: int | None = None,
    ) -> None:
        self._sequencer = sequencer
        self._bindings = bindings
        self._lock = threading.Lock()
        self._released = False
        self._reservation_id = reservation_id

    def _release(self, *, strict: bool) -> None:
        with self._lock:
            if self._released:
                if strict:
                    raise CoherenceStateTransitionError(
                        "coherence consumer lease was already released."
                    )
                return
            self._released = True
        self._sequencer._release_consumer_lease(
            self._bindings,
            reservation_id=self._reservation_id,
        )

    def release(self) -> None:
        self._release(strict=True)

    def release_for_cleanup(self) -> None:
        self._release(strict=False)

    def cancel(self) -> None:
        self.release()


class _ReservationPhase(StrEnum):
    RESERVED = "reserved"
    CLAIMED = "claimed"
    RELEASED = "released"


class _ConsumerLeaseReservationRecord:
    __slots__ = ("bindings", "claimed_lease", "lock", "state")

    def __init__(
        self,
        bindings: tuple[tuple[_Ticket[Any], _Intent], ...],
    ) -> None:
        self.bindings = bindings
        self.claimed_lease: ConsumerLease | None = None
        self.lock = threading.Lock()
        self.state = _ReservationPhase.RESERVED


class ConsumerLeaseReservationRef:
    """Non-owning tensor reference to one sequencer-owned reservation."""

    __slots__ = ("_sequencer", "_reservation_id")

    def __init__(self, sequencer: CoherenceSequencer[Any], reservation_id: int) -> None:
        self._sequencer = sequencer
        self._reservation_id = reservation_id

    def claim(self) -> ConsumerLease:
        return self._sequencer._claim_consumer_lease_reservation(self._reservation_id)

    @property
    def claimed(self) -> bool:
        return self._sequencer._consumer_lease_reservation_claimed(self._reservation_id)

    def release_for_cleanup(self) -> None:
        self._sequencer._cleanup_consumer_lease_reservation(self._reservation_id)

    def cancel(self) -> None:
        self._sequencer._cancel_consumer_lease_reservation(self._reservation_id)


@dataclass(slots=True)
class _Ticket[PayloadT]:
    instances: tuple[CoherenceInstance, ...]
    payload: PayloadT
    future: asyncio.Future[Any] | None
    ticket_id: int = 0
    intents: tuple[_Intent, ...] = field(default_factory=tuple)
    phase: _TicketPhase = _TicketPhase.READY
    inline_claimed: bool = False
    direct_executing: bool = False
    active_consumer_leases: int = 0
    result: Any = _MISSING
    error: BaseException | None = None

    @property
    def state(self) -> _TicketPhase:
        return self.phase

    @state.setter
    def state(self, value: _TicketPhase) -> None:
        self.transition_to(value)

    @property
    def delivery_complete(self) -> bool:
        return self.phase in (_TicketPhase.DELIVERED, _TicketPhase.RETIRED)

    def transition_to(self, new_phase: _TicketPhase) -> None:
        if self.phase is new_phase:
            return
        valid_transitions = {
            _TicketPhase.READY: (_TicketPhase.ADMITTING, _TicketPhase.ABORTED),
            _TicketPhase.ADMITTING: (_TicketPhase.DELIVERING, _TicketPhase.ABORTED),
            _TicketPhase.DELIVERING: (
                _TicketPhase.DELIVERED,
                _TicketPhase.RETIRED,
                _TicketPhase.ABORTED,
            ),
            _TicketPhase.DELIVERED: (_TicketPhase.RETIRED, _TicketPhase.ABORTED),
            _TicketPhase.RETIRED: (),
            _TicketPhase.ABORTED: (),
        }
        allowed = valid_transitions.get(self.phase, ())
        if new_phase not in allowed:
            raise CoherenceStateTransitionError(
                f"illegal ticket state transition from {self.phase} to {new_phase}."
            )
        self.phase = new_phase


class _SubjectLane:
    __slots__ = ("_intents",)

    def __init__(self) -> None:
        self._intents: deque[_Intent] = deque()

    def append(self, intent: _Intent) -> None:
        self._intents.append(intent)

    def __bool__(self) -> bool:
        return bool(self._intents)

    def active_intents(self) -> tuple[_Intent, ...]:
        return tuple(self._intents)

    def next_intent_to_publish(self) -> _Intent | None:
        for intent in self._intents:
            if not intent.is_delivery_completed:
                return intent
        return None

    def next_intent_to_admit(self) -> _Intent | None:
        return self.next_intent_to_publish()

    def head_intent(self) -> _Intent | None:
        return next(iter(self._intents), None)

    def owns_next_intent(self, intent: _Intent) -> bool:
        return self.head_intent() is intent

    def next_ticket_to_retire(self) -> _Ticket[Any] | None:
        intent = self.head_intent()
        return intent.next_ticket() if intent is not None else None

    def discard_completed_prefix(self) -> tuple[_Intent, ...]:
        discarded: list[_Intent] = []
        while self._intents and self._intents[0].is_completed:
            discarded.append(self._intents.popleft())
        return tuple(discarded)

    def retire_next_intent(self, intent: _Intent) -> None:
        if not self.owns_next_intent(intent):
            raise CoherenceStateTransitionError(
                "coherence ROB next intent changed during retirement."
            )
        if not intent.is_completed:
            raise CoherenceStateTransitionError(
                "coherence ROB head intent is not complete."
            )
        self._intents.popleft()


class CoherenceSequencer[PayloadT]:
    """Event-loop-owned per-subject reorder buffer."""

    def __init__(
        self,
        retire_payload: RetirePayload[PayloadT],
        admit_instance: AdmitInstance,
    ) -> None:
        self._retire_payload = retire_payload
        self._admit_instance = admit_instance
        self._lanes: dict[object, _SubjectLane] = {}
        self._intent_index: dict[CoherenceInstance, _Intent] = {}
        self._last_issued_instance: dict[object, CoherenceInstance] = {}
        self._last_retired_instance: dict[object, CoherenceInstance] = {}
        self._next_ticket_id = 0
        self._tickets: dict[int, _Ticket[Any]] = {}
        self._drain_handle: asyncio.Handle | None = None
        self._retire_tasks: dict[int, asyncio.Task[None]] = {}
        self._consumer_lease_reservations: dict[
            int, _ConsumerLeaseReservationRecord
        ] = {}
        self._consumer_lease_reservations_lock = threading.Lock()
        self._next_consumer_lease_reservation_id = 1
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._fenced_error: BaseException | None = None

    def _register_consumer_lease_reservation(
        self, control: _ConsumerLeaseReservationRecord
    ) -> int:
        with self._consumer_lease_reservations_lock:
            reservation_id = self._next_consumer_lease_reservation_id
            self._next_consumer_lease_reservation_id += 1
            self._consumer_lease_reservations[reservation_id] = control
            return reservation_id

    def _find_consumer_lease_reservation(
        self, reservation_id: int
    ) -> _ConsumerLeaseReservationRecord:
        with self._consumer_lease_reservations_lock:
            try:
                return self._consumer_lease_reservations[reservation_id]
            except KeyError as exc:
                raise CoherenceStateTransitionError(
                    "coherence consumer lease reservation was already released."
                ) from exc

    def _retire_consumer_lease_reservation(self, reservation_id: int) -> None:
        with self._consumer_lease_reservations_lock:
            self._consumer_lease_reservations.pop(reservation_id, None)

    def _take_consumer_lease_reservations(
        self,
    ) -> tuple[tuple[int, _ConsumerLeaseReservationRecord], ...]:
        with self._consumer_lease_reservations_lock:
            return tuple(self._consumer_lease_reservations.items())

    def issue_intent(self, instance: CoherenceInstance) -> None:
        self._ensure_open()
        self._capture_owner_loop()
        lane = self._lanes.setdefault(instance.subject, _SubjectLane())
        last_instance = self._last_issued_instance.get(instance.subject)
        if last_instance is not None and instance.epoch <= last_instance.epoch:
            raise CoherenceStateTransitionError(
                "coherence intents must be issued in strictly increasing order: "
                f"subject={instance.subject} epoch={instance.epoch} "
                f"last={last_instance.epoch}."
            )
        intent = _Intent(instance)
        lane.append(intent)
        self._intent_index[instance] = intent
        self._last_issued_instance[instance.subject] = instance

    def allocate_ticket_id(self) -> int:
        self._ensure_open()
        ticket_id = self._next_ticket_id
        self._next_ticket_id += 1
        return ticket_id

    def _resolve_target_ticket(
        self,
        intent: _Intent,
        ticket_id: int | None = None,
    ) -> _Ticket[Any]:
        if ticket_id is not None:
            for cand in intent.payloads:
                if cand.ticket_id == ticket_id:
                    return cand
            raise CoherenceStateTransitionError(
                f"ticket_id {ticket_id} not found in intent {intent.instance}."
            )
        if len(intent.payloads) == 1:
            return intent.payloads[0]
        if not intent.payloads:
            raise CoherenceStateTransitionError(
                f"cannot bind consumer lease for an inactive coherence instance: {intent.instance}."
            )
        delivering = [c for c in intent.payloads if c.phase is _TicketPhase.DELIVERING]
        delivered = [c for c in intent.payloads if c.phase is _TicketPhase.DELIVERED]
        active_candidates = delivering + delivered
        if len(active_candidates) == 1:
            return active_candidates[0]
        raise CoherenceStateTransitionError(
            f"ambiguous tickets ({len(intent.payloads)} tickets present) for intent {intent.instance}; "
            "ticket_id must be specified."
        )

    def acquire_consumer_lease(
        self,
        instances: Iterable[CoherenceInstance],
        ticket_id: int | None = None,
    ) -> ConsumerLease:
        self._ensure_open()
        self._capture_owner_loop()
        ordered = tuple(dict.fromkeys(instances))
        if not ordered:
            raise CoherenceStateTransitionError(
                "consumer lease requires at least one coherence instance."
            )
        bindings: list[tuple[_Ticket[Any], _Intent]] = []
        for instance in ordered:
            intent = self._intent_index.get(instance)
            if intent is None:
                raise CoherenceStateTransitionError(
                    "cannot acquire consumer lease for an inactive coherence "
                    f"instance: {instance}."
                )
            ticket = self._resolve_target_ticket(intent, ticket_id)
            bindings.append((ticket, intent))
        for ticket, intent in bindings:
            ticket.active_consumer_leases += 1
            intent.acquire_lease()
        return ConsumerLease(self, tuple(bindings))

    def reserve_consumer_lease(
        self,
        instances: Iterable[CoherenceInstance],
        ticket_id: int | None = None,
    ) -> ConsumerLeaseReservationRef:
        self._ensure_open()
        self._capture_owner_loop()
        ordered = tuple(dict.fromkeys(instances))
        if not ordered:
            raise CoherenceStateTransitionError(
                "consumer lease reservation requires at least one coherence instance."
            )
        bindings: list[tuple[_Ticket[Any], _Intent]] = []
        for instance in ordered:
            intent = self._intent_index.get(instance)
            if intent is None:
                raise CoherenceStateTransitionError(
                    "cannot reserve consumer lease for an inactive coherence "
                    f"instance: {instance}."
                )
            ticket = self._resolve_target_ticket(intent, ticket_id)
            if ticket.phase not in (_TicketPhase.DELIVERING, _TicketPhase.DELIVERED):
                raise CoherenceStateTransitionError(
                    "consumer lease reservation requires an active payload: "
                    f"{instance}."
                )
            bindings.append((ticket, intent))
        for ticket, _intent in bindings:
            ticket.active_consumer_leases += 1
            _intent.acquire_lease()
        control = _ConsumerLeaseReservationRecord(tuple(bindings))
        reservation_id = self._register_consumer_lease_reservation(control)
        return ConsumerLeaseReservationRef(self, reservation_id)

    def _claim_consumer_lease_reservation(self, reservation_id: int) -> ConsumerLease:
        control = self._find_consumer_lease_reservation(reservation_id)
        with control.lock:
            if control.state is _ReservationPhase.CLAIMED:
                raise CoherenceStateTransitionError(
                    "coherence consumer lease reservation was already claimed."
                )
            if control.state is _ReservationPhase.RELEASED:
                raise CoherenceStateTransitionError(
                    "coherence consumer lease reservation was already released."
                )
            lease = ConsumerLease(self, control.bindings, reservation_id=reservation_id)
            control.state = _ReservationPhase.CLAIMED
            control.claimed_lease = lease
            return lease

    def _consumer_lease_reservation_claimed(self, reservation_id: int) -> bool:
        try:
            control = self._find_consumer_lease_reservation(reservation_id)
        except CoherenceStateTransitionError:
            return False
        with control.lock:
            return control.state is _ReservationPhase.CLAIMED

    def _cancel_consumer_lease_reservation(self, reservation_id: int) -> None:
        control = self._find_consumer_lease_reservation(reservation_id)
        with control.lock:
            if control.state is _ReservationPhase.CLAIMED:
                raise CoherenceStateTransitionError(
                    "cannot cancel a claimed coherence consumer lease reservation."
                )
            if control.state is _ReservationPhase.RELEASED:
                raise CoherenceStateTransitionError(
                    "cannot cancel a released coherence consumer lease reservation."
                )
            control.state = _ReservationPhase.RELEASED
            bindings = control.bindings
        self._release_consumer_lease(bindings, reservation_id=reservation_id)

    def _cleanup_consumer_lease_reservation(self, reservation_id: int) -> None:
        try:
            control = self._find_consumer_lease_reservation(reservation_id)
        except CoherenceStateTransitionError:
            return
        with control.lock:
            if control.state is _ReservationPhase.RELEASED:
                return
            control.state = _ReservationPhase.RELEASED
            lease = control.claimed_lease
            bindings = control.bindings
        if lease is None:
            self._release_consumer_lease(bindings, reservation_id=reservation_id)
        else:
            lease.release_for_cleanup()

    def _release_consumer_lease(
        self,
        bindings: tuple[tuple[_Ticket[Any], _Intent], ...],
        *,
        reservation_id: int | None = None,
    ) -> None:
        owner_loop = self._owner_loop
        if owner_loop is None:
            raise CoherenceStateTransitionError(
                "coherence sequencer has no owner event loop."
            )
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not owner_loop:
            owner_loop.call_soon_threadsafe(
                self._release_consumer_lease_on_owner_loop,
                bindings,
                reservation_id,
            )
            return
        self._release_consumer_lease_on_owner_loop(bindings, reservation_id)

    def _release_consumer_lease_on_owner_loop(
        self,
        bindings: tuple[tuple[_Ticket[Any], _Intent], ...],
        reservation_id: int | None = None,
    ) -> None:
        if self._fenced_error is not None:
            for ticket, intent in bindings:
                if ticket.active_consumer_leases > 0:
                    ticket.active_consumer_leases -= 1
                if intent.active_consumer_leases > 0:
                    intent.release_lease()
            if reservation_id is not None:
                self._retire_consumer_lease_reservation(reservation_id)
            return
        for ticket, _intent in bindings:
            if ticket.active_consumer_leases <= 0:
                raise CoherenceStateTransitionError(
                    "coherence consumer lease count underflow."
                )
        for ticket, intent in bindings:
            ticket.active_consumer_leases -= 1
            intent.release_lease()
            if (
                ticket.phase is _TicketPhase.DELIVERED
                and ticket.active_consumer_leases == 0
            ):
                self._finish_retired_ticket(ticket, ticket.result)
        if reservation_id is not None:
            self._retire_consumer_lease_reservation(reservation_id)
        self._schedule_drain()

    def _capture_owner_loop(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._owner_loop is None:
            self._owner_loop = loop
        elif self._owner_loop is not loop:
            raise CoherenceStateTransitionError(
                "coherence sequencer is bound to another event loop."
            )

    def commit_payload(
        self,
        instances: Iterable[CoherenceInstance],
        payload: PayloadT,
        ticket_id: int | None = None,
    ) -> Coroutine[Any, Any, Any]:
        self._ensure_open()
        self._capture_owner_loop()
        ordered_values = tuple(instances)
        if not ordered_values:
            raise ValueError("coherence payload must reference at least one instance.")
        seen_instances: set[CoherenceInstance] = set()
        for instance in ordered_values:
            if instance in seen_instances:
                raise CoherenceStateTransitionError(
                    f"coherence payload references an instance more than once: {instance}."
                )
            seen_instances.add(instance)
        ordered = ordered_values
        subjects: dict[object, CoherenceInstance] = {}
        for instance in ordered:
            previous = subjects.setdefault(instance.subject, instance)
            if previous != instance:
                raise CoherenceStateTransitionError(
                    "one coherence payload cannot span multiple epochs of one subject: "
                    f"subject={instance.subject} epochs={previous.epoch},{instance.epoch}."
                )
        if ticket_id is None:
            ticket_id = self._next_ticket_id
            self._next_ticket_id += 1
        else:
            if ticket_id in self._tickets:
                raise CoherenceStateTransitionError(
                    f"ticket_id {ticket_id} is already in use."
                )
            if ticket_id >= self._next_ticket_id:
                self._next_ticket_id = ticket_id + 1
        ticket = _Ticket(ordered, payload, None, ticket_id=ticket_id)
        intents: list[_Intent] = []
        try:
            for instance in ordered:
                intent = self._intent_index.get(instance)
                if intent is None:
                    state = (
                        "retired"
                        if self._last_retired_instance.get(instance.subject) == instance
                        else "never-issued"
                    )
                    active = tuple(self._intent_index)
                    last_issued = self._last_issued_instance.get(instance.subject)
                    raise CoherenceStateTransitionError(
                        "coherence intent is not declared: "
                        f"instance={instance} state={state}; "
                        f"last_issued={last_issued!r} active_intents={active!r}."
                    )
                if intent.sealed:
                    raise CoherenceStateTransitionError(
                        f"coherence intent is already sealed: {instance}."
                    )
                intents.append(intent)
            ticket.intents = tuple(intents)
            for intent in intents:
                intent.add_ticket(ticket)
            self._tickets[ticket.ticket_id] = ticket
            if self._claim_inline_retirement(ticket):
                ticket.inline_claimed = True
        except BaseException:
            self._abort_ticket(ticket)
            raise
        return self._await_ticket(ticket)

    async def _await_ticket(self, ticket: _Ticket[Any]) -> Any:
        if (
            self._fenced_error is not None
            and ticket.future is None
            and ticket.phase is not _TicketPhase.RETIRED
        ):
            raise self._fenced_error
        if ticket.phase in (_TicketPhase.DELIVERED, _TicketPhase.RETIRED):
            return ticket.result if ticket.result is not _MISSING else None
        if ticket.phase is _TicketPhase.ABORTED:
            if ticket.error is not None:
                raise ticket.error
            if self._fenced_error is not None:
                raise self._fenced_error
            raise RuntimeError("coherence payload aborted")

        if ticket.future is None and not ticket.inline_claimed:
            if self._claim_inline_retirement(ticket):
                ticket.inline_claimed = True
            else:
                ticket.future = asyncio.get_running_loop().create_future()

        if ticket.inline_claimed:
            ticket.inline_claimed = False
            ticket.direct_executing = True
            current_task = asyncio.current_task()
            if current_task is None:
                raise RuntimeError("coherence retirement requires an active task.")
            self._retire_tasks[ticket.ticket_id] = current_task
            try:
                result = await self._admit_and_retire(ticket, direct=True)
                self._schedule_drain()
                return result
            except asyncio.CancelledError:
                self._abort_ticket(ticket, asyncio.CancelledError())
                if ticket.phase is not _TicketPhase.DELIVERING:
                    self._schedule_drain()
                raise
            except BaseException as error:
                self._fence_after_failure(error)
                raise
            finally:
                ticket.direct_executing = False
                self._retire_tasks.pop(ticket.ticket_id, None)

        try:
            future = ticket.future
            if future is None:
                raise RuntimeError("blocked coherence ticket has no completion future.")
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            self._abort_ticket(ticket, asyncio.CancelledError())
            if ticket.phase is not _TicketPhase.DELIVERING:
                self._schedule_drain()
            raise

    def _is_ticket_ready(self, ticket: _Ticket[Any]) -> bool:
        if ticket.phase not in (
            _TicketPhase.READY,
            _TicketPhase.DELIVERING,
            _TicketPhase.DELIVERED,
        ):
            return False
        for intent in ticket.intents:
            lane = self._lanes.get(intent.instance.subject)
            if lane is None:
                return False
            if ticket.phase is _TicketPhase.DELIVERED:
                if ticket.active_consumer_leases != 0:
                    return False
                continue
            if lane.next_intent_to_publish() is not intent or not intent.sealed:
                return False
            for prior in intent.payloads:
                if prior is ticket:
                    break
                if not prior.delivery_complete:
                    return False
        return True

    def _claim_inline_retirement(self, ticket: _Ticket[Any]) -> bool:
        if not self._is_ticket_ready(ticket):
            return False
        ticket.transition_to(_TicketPhase.ADMITTING)
        return True

    def close_intent(self, instance: CoherenceInstance) -> None:
        self._ensure_open()
        intent = self._intent_index.get(instance)
        if intent is None:
            raise CoherenceStateTransitionError(
                f"coherence intent is not declared: {instance}."
            )
        if intent.sealed:
            raise CoherenceStateTransitionError(
                f"coherence intent is already sealed: {instance}."
            )
        intent.sealed = True
        self._schedule_drain()

    def inspect_lane_status(self, instance: CoherenceInstance) -> tuple[str, int]:
        lane = self._lanes.get(instance.subject)
        if lane is None:
            return "ready", 0
        head = lane.head_intent()
        if head is None or head.instance == instance:
            return "ready", 0
        if head.active_consumer_leases > 0:
            return "consumer_lease_wait", head.active_consumer_leases
        return "lane_order_wait", 0

    def abort_intent(self, instance: CoherenceInstance, error: BaseException) -> None:
        if self._fenced_error is not None:
            return
        if instance not in self._intent_index:
            raise CoherenceStateTransitionError(
                f"coherence intent is not declared: {instance}."
            )
        self.fence(error)

    def fence(self, error: BaseException | None = None) -> None:
        if self._fenced_error is not None:
            return
        for reservation_id, _control in self._take_consumer_lease_reservations():
            self._cleanup_consumer_lease_reservation(reservation_id)
        self._fenced_error = error or RuntimeError("coherence sequencer fenced")
        if self._drain_handle is not None:
            self._drain_handle.cancel()
            self._drain_handle = None
        current_task = asyncio.current_task()
        for task in tuple(self._retire_tasks.values()):
            if task is not current_task and not task.done():
                task.cancel()
        for lane in self._lanes.values():
            for intent in lane.active_intents():
                if (
                    intent.admission_task is not None
                    and not intent.admission_task.done()
                ):
                    intent.admission_task.cancel()
        self._abort_all(self._fenced_error)
        self._lanes.clear()
        self._intent_index.clear()

    def _ensure_open(self) -> None:
        if self._fenced_error is not None:
            raise self._fenced_error

    def _schedule_drain(self) -> None:
        if self._fenced_error is not None:
            return
        self._capture_owner_loop()
        if self._drain_handle is None:
            self._drain_handle = asyncio.get_running_loop().call_soon(self._run_drain)

    def _run_drain(self) -> None:
        self._drain_handle = None
        try:
            self._drain()
        except Exception as error:
            self._fence_after_failure(error)
            raise

    def _drain(self) -> None:
        while self._lanes and self._fenced_error is None:
            self._discard_completed_intents()
            if not self._lanes:
                return
            ready_tickets = self._next_ready_tickets()
            if not ready_tickets:
                if self._all_active_intents_sealed():
                    cycle = self._dependency_cycle()
                    if cycle:
                        error = CoherenceStateTransitionError(
                            "coherence payload dependency cycle detected: "
                            f"tickets={cycle}."
                        )
                        self._fence_after_failure(error)
                    elif any(
                        intent.active_consumer_leases
                        for lane in self._lanes.values()
                        for intent in lane.active_intents()
                    ):
                        return
                    elif not any(
                        not task.done() for task in self._retire_tasks.values()
                    ):
                        error = CoherenceStateTransitionError(
                            "coherence reorder buffer is sealed but has no ready ticket: "
                            f"pending={self._pending_snapshot()}."
                        )
                        self._fence_after_failure(error)
                return
            for ticket in ready_tickets:
                self._start_retirement(ticket)

    def _start_retirement(self, ticket: _Ticket[Any]) -> None:
        if (
            ticket.phase
            not in (
                _TicketPhase.READY,
                _TicketPhase.DELIVERING,
                _TicketPhase.DELIVERED,
            )
            or ticket.ticket_id not in self._tickets
            or ticket.ticket_id in self._retire_tasks
        ):
            return
        if ticket.phase is _TicketPhase.DELIVERED:
            self._finish_retired_ticket(ticket, ticket.result)
            return
        if ticket.phase is _TicketPhase.DELIVERING:
            return
        ticket.transition_to(_TicketPhase.ADMITTING)
        task = asyncio.create_task(self._admit_and_retire(ticket))
        self._retire_tasks[ticket.ticket_id] = task

        def done(done_task: asyncio.Task[None]) -> None:
            self._retire_tasks.pop(ticket.ticket_id, None)
            if done_task.cancelled():
                if ticket.phase is not _TicketPhase.ABORTED:
                    self._abort_all(asyncio.CancelledError())
                return
            error = done_task.exception()
            if error is not None:
                self._fence_after_failure(error)
                return
            self._schedule_drain()

        task.add_done_callback(done)

    async def _admit_and_retire(
        self, ticket: _Ticket[Any], *, direct: bool = False
    ) -> Any:
        admission_tasks: list[asyncio.Future[None]] = []
        for intent in ticket.intents:
            if intent.admitted:
                continue
            if direct and intent.admission_task is None:
                await self._admit_intent(intent)
                continue
            if intent.admission_task is None:
                intent.admission_task = asyncio.create_task(self._admit_intent(intent))
            admission_tasks.append(asyncio.shield(intent.admission_task))
        if admission_tasks:
            await asyncio.gather(*admission_tasks)
        return await self._retire_ticket(ticket)

    async def _admit_intent(self, intent: _Intent) -> None:
        await self._admit_instance(intent.instance)
        intent.mark_admitted()

    async def _retire_ticket(self, ticket: _Ticket[Any]) -> Any:
        ticket.transition_to(_TicketPhase.DELIVERING)
        try:
            result = await self._retire_payload(ticket.instances, ticket.payload)
        except BaseException as error:
            try:
                self._finish_retired_ticket(ticket)
            except BaseException as cleanup_error:
                self._fence_after_failure(cleanup_error)
                raise cleanup_error from error
            future = ticket.future
            if future is not None and not future.cancelled() and not future.done():
                future.set_exception(error)
            raise
        try:
            ticket.transition_to(_TicketPhase.DELIVERED)
            for intent in ticket.intents:
                intent._refresh_phase()
            self._publish_ticket_result(ticket, result)
            self._finish_retired_ticket(ticket, result)
        except BaseException as cleanup_error:
            self._fence_after_failure(cleanup_error)
            raise
        return result

    def _finish_retired_ticket(
        self, ticket: _Ticket[Any], result: Any = _MISSING
    ) -> None:
        if ticket.phase is _TicketPhase.RETIRED:
            return
        if result is not _MISSING:
            ticket.result = result
        if ticket.active_consumer_leases > 0:
            return
        if ticket.phase is not _TicketPhase.DELIVERED:
            return
        if ticket.phase is not _TicketPhase.ABORTED:
            ticket.transition_to(_TicketPhase.RETIRED)
        for intent in ticket.intents:
            lane = self._lanes.get(intent.instance.subject)
            if lane is None:
                raise CoherenceStateTransitionError(
                    "coherence ROB lane disappeared during retirement."
                )
            intent.remove_ticket(ticket)
            if intent.is_completed:
                self._intent_index.pop(intent.instance, None)
                self._last_retired_instance[intent.instance.subject] = intent.instance
                lane.discard_completed_prefix()
                if not lane:
                    del self._lanes[intent.instance.subject]
        self._tickets.pop(ticket.ticket_id, None)
        self._publish_ticket_result(ticket, result)

    @staticmethod
    def _publish_ticket_result(ticket: _Ticket[Any], result: Any) -> None:
        if result is _MISSING:
            return
        ticket.result = result
        future = ticket.future
        if future is not None and not future.done():
            future.set_result(result)

    def _discard_completed_intents(self) -> None:
        for subject, lane in tuple(self._lanes.items()):
            for intent in lane.discard_completed_prefix():
                self._intent_index.pop(intent.instance, None)
                self._last_retired_instance[intent.instance.subject] = intent.instance
            if not lane:
                del self._lanes[subject]

    def _next_ready_tickets(self) -> tuple[_Ticket[PayloadT], ...]:
        candidates: dict[int, _Ticket[PayloadT]] = {}
        for lane in self._lanes.values():
            intent = lane.next_intent_to_publish()
            if intent is None:
                continue
            for ticket in intent.payloads:
                if ticket.ticket_id in self._retire_tasks:
                    if not ticket.delivery_complete:
                        break
                    continue
                if self._is_ticket_ready(ticket):
                    candidates[ticket.ticket_id] = ticket
                if not ticket.delivery_complete:
                    break
        return tuple(candidates.values())

    def _dependency_cycle(self) -> tuple[int, ...]:
        active_tickets = {
            ticket.ticket_id: ticket
            for lane in self._lanes.values()
            for intent in lane.active_intents()
            for ticket in intent.payloads
            if ticket.phase not in (_TicketPhase.ABORTED, _TicketPhase.RETIRED)
        }
        if not active_tickets:
            return ()

        edges: dict[int, set[int]] = {ticket_id: set() for ticket_id in active_tickets}
        for ticket_id, ticket in active_tickets.items():
            for instance in ticket.instances:
                lane = self._lanes.get(instance.subject)
                if lane is None:
                    continue
                current_intent = lane.next_intent_to_publish()
                if current_intent is None:
                    continue
                intent = self._intent_index[instance]
                if current_intent is intent:
                    continue
                for blocker in current_intent.payloads:
                    blocker_id = blocker.ticket_id
                    if (
                        blocker_id != ticket_id
                        and blocker.phase
                        not in (_TicketPhase.ABORTED, _TicketPhase.RETIRED)
                        and blocker_id in self._tickets
                    ):
                        edges[ticket_id].add(blocker_id)

        visiting: set[int] = set()
        visited: set[int] = set()
        path: list[int] = []

        def visit(ticket_id: int) -> tuple[int, ...] | None:
            if ticket_id in visiting:
                start = path.index(ticket_id)
                return tuple(path[start:] + [ticket_id])
            if ticket_id in visited:
                return None
            visiting.add(ticket_id)
            path.append(ticket_id)
            for dependency in edges[ticket_id]:
                cycle = visit(dependency)
                if cycle:
                    return cycle
            path.pop()
            visiting.remove(ticket_id)
            visited.add(ticket_id)
            return None

        for ticket_id in edges:
            cycle = visit(ticket_id)
            if cycle:
                return cycle
        return ()

    def _all_active_intents_sealed(self) -> bool:
        return all(
            intent.sealed
            for lane in self._lanes.values()
            for intent in lane.active_intents()
        )

    def _pending_snapshot(self) -> tuple[dict[str, Any], ...]:
        pending: list[dict[str, Any]] = []
        for subject, lane in self._lanes.items():
            for intent in lane.active_intents():
                ticket = next(iter(intent.payloads), None)
                pending.append(
                    {
                        "subject": repr(subject),
                        "instance": repr(intent.instance),
                        "sealed": intent.sealed,
                        "active_payloads": intent.active_payloads,
                        "active_consumer_leases": intent.active_consumer_leases,
                        "head_ticket_id": (
                            ticket.ticket_id if ticket is not None else None
                        ),
                        "head_ticket_state": (
                            ticket.phase if ticket is not None else None
                        ),
                    }
                )
        return tuple(pending)

    def _abort_ticket(
        self,
        ticket: _Ticket[Any],
        error: BaseException | None = None,
    ) -> None:
        if ticket.phase is _TicketPhase.RETIRED:
            if ticket.future is not None and not ticket.future.done():
                ticket.future.set_exception(
                    error or RuntimeError("retired coherence payload has no result")
                )
            return
        if ticket.phase is _TicketPhase.ABORTED:
            return
        retire_task = self._retire_tasks.get(ticket.ticket_id)
        ticket.transition_to(_TicketPhase.ABORTED)
        abort_error = error or RuntimeError("coherence payload aborted")
        ticket.error = abort_error
        if (
            retire_task is not None
            and retire_task is not asyncio.current_task()
            and not retire_task.done()
        ):
            retire_task.cancel()
        for intent in ticket.intents:
            if ticket in intent.payloads:
                intent.remove_ticket(ticket)
            if (
                intent.active_payloads == 0
                and intent.admission_task is not None
                and not intent.admission_task.done()
            ):
                intent.admission_task.cancel()
        future = ticket.future
        if future is not None and not future.done():
            if ticket.direct_executing or isinstance(
                abort_error, asyncio.CancelledError
            ):
                future.cancel()
            else:
                future.set_exception(abort_error)
        self._tickets.pop(ticket.ticket_id, None)

    def _abort_all(self, error: BaseException) -> None:
        for ticket in tuple(self._tickets.values()):
            self._abort_ticket(ticket, error)
        self._tickets.clear()
        self._lanes.clear()
        self._intent_index.clear()
        self._retire_tasks.clear()

    def _fence_after_failure(self, error: BaseException) -> None:
        self.fence(error)
