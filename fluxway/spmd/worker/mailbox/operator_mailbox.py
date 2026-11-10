# fluxway/spmd/worker/mailbox/operator_mailbox.py
"""
Operator Mailboxes for Worker Direct Local Fast-Path and Distributed Transports.

Provides a unified, high-performance Mailbox abstraction for operators:
- OperatorMailbox: Abstract base class defining the operator mailbox lifecycle.
- DirectLocalMailbox: Zero-lock, zero-await fast-path for intra-worker local transfers.
- DelegatedWorkerMailbox: Delegated operator mailbox backed by the shared WorkerMailbox.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fluxway.common.types.tensor import TransportTensor


@dataclass(slots=True)
class DirectLocalInputSlot:
    tensor: TransportTensor
    source_step: int
    sender_id: str
    rank: int


class OperatorMailbox(ABC):
    """Abstract base class defining the contract and lifecycle of an operator mailbox."""

    def __init__(self, op_name: str) -> None:
        self.op_name = op_name

    @property
    @abstractmethod
    def is_direct_fast_path(self) -> bool:
        """Whether this mailbox uses the zero-lock direct local input fast-path."""
        ...

    @abstractmethod
    def is_ready(self, step: int) -> bool:
        """Check if inputs for the specified step are fully assembled/ready."""
        ...

    @abstractmethod
    async def clear_step(self, step: int) -> None:
        """Release and clean up input slots and state for the completed step."""
        ...

    @abstractmethod
    async def deposit_local_tensor(
        self,
        *,
        transfer_code: int,
        step: int,
        sender_id: str,
        rank: int,
        tensor: TransportTensor,
    ) -> None:
        """Deposit a local tensor into this operator mailbox."""
        ...


class DirectLocalMailbox(OperatorMailbox):
    """Zero-lock, zero-await direct local mailbox for an operator."""

    __slots__ = (
        "op_name",
        "expected_transfer_codes",
        "_expected_count",
        "_single_code",
        "_slots",
        "_ready_steps",
        "_on_step_ready_cb",
    )

    def __init__(
        self,
        op_name: str,
        expected_transfer_codes: set[int] | frozenset[int],
        on_step_ready: Callable[[str, int], None],
        *,
        slots: dict[int, dict[int, DirectLocalInputSlot]] | None = None,
        ready_steps: set[int] | None = None,
    ) -> None:
        super().__init__(op_name)
        if not expected_transfer_codes:
            raise ValueError(
                f"DirectLocalMailbox requires non-empty expected_transfer_codes for op={op_name!r}"
            )
        self.expected_transfer_codes: frozenset[int] = frozenset(
            expected_transfer_codes
        )
        self._expected_count: int = len(self.expected_transfer_codes)
        self._single_code: int | None = (
            next(iter(self.expected_transfer_codes))
            if self._expected_count == 1
            else None
        )
        self._slots: dict[int, dict[int, DirectLocalInputSlot]] = (
            slots if slots is not None else {}
        )
        self._ready_steps: set[int] = ready_steps if ready_steps is not None else set()
        self._on_step_ready_cb = on_step_ready

    @property
    def is_direct_fast_path(self) -> bool:
        return True

    @property
    def slots(self) -> dict[int, dict[int, DirectLocalInputSlot]]:
        return self._slots

    @property
    def ready_steps(self) -> set[int]:
        return self._ready_steps

    def deposit(
        self,
        transfer_code: int,
        step: int,
        sender_id: str,
        rank: int,
        tensor: TransportTensor,
    ) -> None:
        """Deposit a local tensor directly into the step slot without locking."""
        single = self._single_code
        if single is not None:
            if transfer_code != single:
                raise ValueError(
                    "Direct local input fast path received a tensor outside the "
                    f"compiled local-only contract: op={self.op_name} "
                    f"transfer_code={transfer_code} expected={sorted(self.expected_transfer_codes)}."
                )
        elif transfer_code not in self.expected_transfer_codes:
            raise ValueError(
                "Direct local input fast path received a tensor outside the "
                f"compiled local-only contract: op={self.op_name} "
                f"transfer_code={transfer_code} expected={sorted(self.expected_transfer_codes)}."
            )

        slots = self._slots
        step_slots = slots.get(step)
        if step_slots is None:
            step_slots = {}
            slots[step] = step_slots
        elif transfer_code in step_slots:
            existing = step_slots[transfer_code]
            raise ValueError(
                "Duplicate direct local tensor deposit: "
                f"op={self.op_name} transfer_code={transfer_code} step={step} "
                f"existing={existing.sender_id}:{existing.rank} "
                f"new={sender_id}:{rank}."
            )

        step_slots[transfer_code] = DirectLocalInputSlot(
            tensor,
            step,
            sender_id,
            rank,
        )
        if len(step_slots) == self._expected_count:
            self._ready_steps.add(step)
            self._on_step_ready_cb(self.op_name, step)

    async def deposit_local_tensor(
        self,
        *,
        transfer_code: int,
        step: int,
        sender_id: str,
        rank: int,
        tensor: TransportTensor,
    ) -> None:
        """Asynchronous adapter for polymorphic deposit into direct local input slots."""
        self.deposit(
            transfer_code,
            step,
            sender_id,
            rank,
            tensor,
        )

    def is_ready(self, step: int) -> bool:
        """Check if all required tensors have arrived for this step."""
        step_slots = self._slots.get(step)
        if step_slots is None:
            return False
        return len(step_slots) == self._expected_count

    def get_step_slots(self, step: int) -> dict[int, DirectLocalInputSlot]:
        """Get the deposited tensor slots for the given step."""
        step_slots = self._slots.get(step)
        if step_slots is None or len(step_slots) != self._expected_count:
            raise ValueError(
                f"Direct local input slots not ready for op={self.op_name!r} step={step}"
            )
        return step_slots

    def clear_step_sync(self, step: int) -> None:
        """Clear the step slots and ready status synchronously."""
        self._slots.pop(step, None)
        self._ready_steps.discard(step)

    async def clear_step(self, step: int) -> None:
        """Clear the step slots and ready status."""
        self.clear_step_sync(step)


class DelegatedWorkerMailbox(OperatorMailbox):
    """Delegated operator mailbox backed by the shared WorkerMailbox."""

    __slots__ = ("worker_mailbox",)

    def __init__(self, op_name: str, worker_mailbox: Any) -> None:
        super().__init__(op_name)
        self.worker_mailbox = worker_mailbox

    @property
    def is_direct_fast_path(self) -> bool:
        return False

    def is_ready(self, step: int) -> bool:
        if self.worker_mailbox is None:
            return False
        return self.worker_mailbox.is_step_ready(self.op_name, step)

    async def clear_step(self, step: int) -> None:
        """Release all data for the step in the delegated worker mailbox."""
        if self.worker_mailbox is not None:
            await self.worker_mailbox.clear_step(self.op_name, step)

    async def deposit_local_tensor(
        self,
        *,
        transfer_code: int,
        step: int,
        sender_id: str,
        rank: int,
        tensor: TransportTensor,
    ) -> None:
        if self.worker_mailbox is None:
            raise ValueError(
                f"DelegatedWorkerMailbox has no worker mailbox for op={self.op_name}"
            )
        await self.worker_mailbox.deposit_local_tensor(
            self.op_name,
            transfer_code,
            step,
            sender_id,
            rank,
            tensor,
        )


def create_operator_mailbox(
    op_name: str,
    *,
    expected_direct_transfer_codes: set[int] | frozenset[int],
    on_step_ready: Callable[[str, int], None],
    worker_mailbox: Any,
    slots: dict[int, dict[int, DirectLocalInputSlot]] | None = None,
    ready_steps: set[int] | None = None,
) -> OperatorMailbox:
    """Factory creating the appropriate OperatorMailbox based on routing capabilities."""
    if expected_direct_transfer_codes:
        return DirectLocalMailbox(
            op_name=op_name,
            expected_transfer_codes=expected_direct_transfer_codes,
            on_step_ready=on_step_ready,
            slots=slots,
            ready_steps=ready_steps,
        )
    return DelegatedWorkerMailbox(
        op_name=op_name,
        worker_mailbox=worker_mailbox,
    )
