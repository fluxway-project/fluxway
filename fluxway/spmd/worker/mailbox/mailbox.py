"""
Mailbox module for dual-stream convergence.

Data plane: tensor shards from upstream workers, indexed by (op_type, name, step).
Control plane: state scalars from Driver.

Completion tracking is event-driven, not polled:
  - Each (op_type, step) has a StepBarrier composed of ArrivalBarriers.
  - When all expected shards and states arrive, the barrier fires automatically.
  - Worker registers an on_ready callback; no _check_and_enqueue polling needed.

Zero-copy shard writing is supported via TransportReceiver delegation.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from fluxway.common.types.configs import OpType
from fluxway.common.sync import ArrivalBarrier
from fluxway.transport.communication import TransportReceiver

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fluxway.common.types.tensor import TransportTensor


# ---------------------------------------------------------------------------
# StepBarrier — one per (op_type, step)
# ---------------------------------------------------------------------------


class StepBarrier:
    """
    Fires when all declared tensor names have received their expected shard
    counts AND all declared state names have been received.

    Internally composes one ArrivalBarrier per named stream.

    Args:
        expected_tensors: {tensor_name: required_shard_count}
        expected_states:  list of required state names
    """

    __slots__ = ("_tensor_barriers", "_state_barriers", "_combined")

    def __init__(
        self, expected_tensors: Dict[str, int], expected_states: List[str]
    ) -> None:
        self._tensor_barriers: Dict[str, ArrivalBarrier] = {
            name: ArrivalBarrier(count) for name, count in expected_tensors.items()
        }
        self._state_barriers: Dict[str, ArrivalBarrier] = {
            name: ArrivalBarrier(1) for name in expected_states
        }
        # Combined completion event
        self._combined = asyncio.Event()
        self._check()  # fire immediately if both dicts are empty

    # ------------------------------------------------------------------
    # Arrival notification
    # ------------------------------------------------------------------

    def tensor_arrived(self, name: str) -> bool:
        """Record one shard arriving for `name`. Returns True if barrier just fired."""
        barrier = self._tensor_barriers.get(name)
        if barrier is not None:
            barrier.arrive()
            return self._check()
        return False

    def state_arrived(self, name: str) -> bool:
        """Record state data arriving for `name`. Returns True if barrier just fired."""
        barrier = self._state_barriers.get(name)
        if barrier is not None:
            barrier.arrive()
            return self._check()
        return False

    # ------------------------------------------------------------------
    # Completion query
    # ------------------------------------------------------------------

    def _check(self) -> bool:
        if self._combined.is_set():
            return False
        tensors_ok = all(b.is_done() for b in self._tensor_barriers.values())
        states_ok = all(b.is_done() for b in self._state_barriers.values())
        if tensors_ok and states_ok:
            self._combined.set()
            return True
        return False

    def is_ready(self) -> bool:
        return self._combined.is_set()

    async def wait(self) -> None:
        await self._combined.wait()


# ---------------------------------------------------------------------------
# WorkerMailbox
# ---------------------------------------------------------------------------


class WorkerMailbox:
    """
    Mailbox for caching incoming data in an SPMD Worker.

    Control plane (state):
        {op_type → {name → {step → data}}}

    Data plane (tensors):
        {op_type → {name → {step → {rank → (data, meta)}}}}

    Completion tracking:
        One StepBarrier per (op_type, step), registered via declare_step().
        When a step becomes ready, the on_ready callback registered via
        set_ready_callback() is called immediately — no external polling.

    Zero-copy slots:
        register_zero_copy_slot() activates a TransportReceiver for a slot
        so deposit_shards() writes directly into pre-allocated buffers.
    """

    def __init__(self) -> None:
        # Control plane
        self._state_data: Dict[OpType, Dict[str, Dict[int, Any]]] = {
            op: {} for op in OpType
        }

        # Data plane (buffered path)
        self._tensor_slots: Dict[
            OpType, Dict[str, Dict[int, Dict[int, "TransportTensor"]]]
        ] = {op: {} for op in OpType}

        # Completion barriers: {op_type: {step: StepBarrier}}
        self._barriers: Dict[OpType, Dict[int, StepBarrier]] = {op: {} for op in OpType}

        # Zero-copy receivers (activated on demand)
        self._zc_receivers: Dict[OpType, Dict[str, Dict[int, TransportReceiver]]] = {
            op: {} for op in OpType
        }

        # Called with (op_type, step) when a step barrier fires
        self._on_ready: Optional[Callable[[OpType, int], None]] = None

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Step barrier lifecycle
    # ------------------------------------------------------------------

    def set_ready_callback(self, callback: Callable[[OpType, int], None]) -> None:
        """
        Register a callback fired when a step becomes ready.

        The callback is invoked inside the asyncio.Lock, synchronously —
        use asyncio.create_task() inside it for non-blocking work.

        Args:
            callback: callable(op_type, step)
        """
        self._on_ready = callback

    def declare_step(
        self,
        op_type: OpType,
        step: int,
        expected_tensors: Dict[str, int],
        expected_states: List[str],
    ) -> None:
        """
        Declare the expected arrivals for a step.

        Must be called before the first deposit_shards() / deposit_states() for
        that step.  If not called, deposit_* operations still work but no
        barrier fires.

        Args:
            expected_tensors: {tensor_name: required_shard_count}
            expected_states: list of required state names
        """
        barrier = StepBarrier(expected_tensors, expected_states)
        self._barriers[op_type][step] = barrier
        # In case expected counts are all zero the barrier is already ready
        if barrier.is_ready() and self._on_ready:
            self._on_ready(op_type, step)

    # ------------------------------------------------------------------
    # Control plane — state data
    # ------------------------------------------------------------------

    async def deposit_states(
        self, op_type: OpType, step: int, names: List[str], values: List[Any]
    ) -> None:
        """Cache state data and notify the step barrier."""
        if len(names) != len(values):
            raise ValueError("names and values must have the same length")

        async with self._lock:
            for name, data in zip(names, values, strict=True):
                self._state_data[op_type].setdefault(name, {})[step] = data
                barrier = self._barriers[op_type].get(step)
                if barrier is not None:
                    fired = barrier.state_arrived(name)
                    if fired and self._on_ready:
                        self._on_ready(op_type, step)
        logger.debug(
            "Mailbox deposit_states: op=%s names=%s step=%d", op_type.name, names, step
        )

    async def get_state(self, op_type: OpType, name: str, step: int) -> Optional[Any]:
        async with self._lock:
            return self._state_data[op_type].get(name, {}).get(step)

    async def get_all_states(self, op_type: OpType, step: int) -> Dict[str, Any]:
        async with self._lock:
            return {
                name: steps[step]
                for name, steps in self._state_data[op_type].items()
                if step in steps
            }

    async def has_state(self, op_type: OpType, name: str, step: int) -> bool:
        async with self._lock:
            return step in self._state_data[op_type].get(name, {})

    # ------------------------------------------------------------------
    # Data plane — tensor shards
    # ------------------------------------------------------------------

    def _buffered_slot(
        self, op_type: OpType, name: str, step: int
    ) -> Dict[int, "TransportTensor"]:
        return self._tensor_slots[op_type].setdefault(name, {}).setdefault(step, {})

    async def deposit_shards(
        self,
        op_type: OpType,
        name: str,
        step: int,
        rank: int,
        payloads: List[Any],
        counts: List[int],
        is_ragged: bool = False,
        has_indices: bool = False,
    ) -> None:
        """
        Cache a tensor shard and notify the step barrier.

        If a zero-copy slot is registered for (op_type, name, step), writes
        directly into the pre-allocated buffer instead.

        Args:
            op_type: Operation type
            name: Tensor name
            step: Execution step
            rank: Sender rank
            payloads: Raw data arrays (data, optional lengths, optional indices)
            counts: Number of rows for each payload
            is_ragged: Whether the payloads represent a ragged tensor (data + lengths)
            has_indices: Whether the payloads include routing indices
        """
        async with self._lock:
            zc = self._zc_receivers[op_type].get(name, {}).get(step)
            if zc is not None:
                # Zero-copy delegates raw payloads directly to the receiver.
                # Assuming one TransportTensor structure for simplicity in zero-copy routing.
                # If there are multiple co-located tensors in `payloads`, zero-copy setup
                # must match the exact payload sequence.
                zc.receive_slices(rank, payloads, counts)
            else:
                # Reconstruct TransportTensor(s) for the buffered slot
                from fluxway.common.types.tensor import TransportTensor
                from fluxway.common.backend import get_backend

                cursor = 0
                backend = get_backend()
                tensors = []

                # Handle multiple pairs if `payloads` contains more elements than a single tensor requires.
                # Generally, 1 tensor = 1 data + (0 or 1 lengths) + (0 or 1 indices).
                # To support lists (co-location), we iterate until payloads are consumed.
                while cursor < len(payloads):
                    count = counts[cursor] if cursor < len(counts) else 0
                    data_payload = (
                        payloads[cursor][:count] if count > 0 else payloads[cursor]
                    )
                    cursor += 1

                    lengths_payload = None
                    if is_ragged and cursor < len(payloads):
                        length_count = counts[cursor] if cursor < len(counts) else 0
                        lengths_payload = (
                            payloads[cursor][:length_count]
                            if length_count > 0
                            else payloads[cursor]
                        )
                        cursor += 1

                    indices_payload = None
                    if has_indices and cursor < len(payloads):
                        index_count = counts[cursor] if cursor < len(counts) else 0
                        indices_payload = (
                            payloads[cursor][:index_count]
                            if index_count > 0
                            else payloads[cursor]
                        )
                        cursor += 1

                    nested_splits = []
                    if lengths_payload is not None:
                        # Convert lengths back to splits via cumsum
                        cs = backend.cumsum(lengths_payload, dim=0)
                        zero = backend.zeros((1,), dtype=backend.int32())
                        splits = backend.cat([zero, cs], dim=0)
                        nested_splits = [splits]

                    tensors.append(
                        TransportTensor(
                            data_payload,
                            nested_splits=nested_splits,
                            indices=indices_payload,
                        )
                    )

                # Store single or list depending on original intention?
                # For compatibility, if only 1, store it as single
                if len(tensors) == 1:
                    self._buffered_slot(op_type, name, step)[rank] = tensors[0]
                else:
                    self._buffered_slot(op_type, name, step)[rank] = tensors

            barrier = self._barriers[op_type].get(step)
            if barrier is not None:
                fired = barrier.tensor_arrived(name)
                if fired and self._on_ready:
                    self._on_ready(op_type, step)

        logger.debug(
            "Mailbox deposit_shards: op=%s name=%s step=%d rank=%d",
            op_type.name,
            name,
            step,
            rank,
        )

    async def get_tensor_shard(
        self, op_type: OpType, name: str, step: int, rank: int
    ) -> Optional["TransportTensor"]:
        async with self._lock:
            return self._tensor_slots[op_type].get(name, {}).get(step, {}).get(rank)

    async def get_tensor_shards(
        self, op_type: OpType, name: str, step: int
    ) -> Dict[int, "TransportTensor"]:
        async with self._lock:
            return dict(self._tensor_slots[op_type].get(name, {}).get(step, {}))

    async def get_all_tensor_shards(
        self, op_type: OpType, step: int
    ) -> Dict[str, Dict[int, "TransportTensor"]]:
        async with self._lock:
            return {
                name: dict(step_dict[step])
                for name, step_dict in self._tensor_slots[op_type].items()
                if step in step_dict
            }

    async def get_tensor_shard_count(
        self, op_type: OpType, name: str, step: int
    ) -> int:
        async with self._lock:
            return len(self._tensor_slots[op_type].get(name, {}).get(step, {}))

    # ------------------------------------------------------------------
    # Step readiness
    # ------------------------------------------------------------------

    def is_step_ready(self, op_type: OpType, step: int) -> bool:
        """Non-blocking: True if the step barrier has fired."""
        barrier = self._barriers[op_type].get(step)
        return barrier is not None and barrier.is_ready()

    async def wait_step_ready(self, op_type: OpType, step: int) -> None:
        """Async wait until the step barrier fires.

        Raises RuntimeError if declare_step() was never called for this step.
        """
        barrier = self._barriers[op_type].get(step)
        if barrier is None:
            raise RuntimeError(
                f"declare_step() was not called for op={op_type.name} step={step}"
            )
        await barrier.wait()

    # ------------------------------------------------------------------
    # Zero-copy registration
    # ------------------------------------------------------------------

    async def register_zero_copy_slot(
        self,
        op_type: OpType,
        name: str,
        step: int,
        target_buffers: List[Any],
        expect_offsets: dict,
    ) -> None:
        """Enable zero-copy write-through for a tensor slot."""
        async with self._lock:
            receiver = TransportReceiver(wait_for_n_slices=0)
            receiver.register_zero_copy_buffers(target_buffers, expect_offsets)
            self._zc_receivers[op_type].setdefault(name, {})[step] = receiver

    async def get_zero_copy_buffers(
        self, op_type: OpType, name: str, step: int
    ) -> Optional[List[Any]]:
        async with self._lock:
            receiver = self._zc_receivers[op_type].get(name, {}).get(step)
            return receiver.target_buffers if receiver is not None else None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def clear_step(self, op_type: OpType, step: int) -> None:
        """Release all data for a completed step."""
        async with self._lock:
            for name, step_dict in list(self._tensor_slots[op_type].items()):
                step_dict.pop(step, None)
                if not step_dict:
                    del self._tensor_slots[op_type][name]

            for name, step_dict in list(self._zc_receivers[op_type].items()):
                step_dict.pop(step, None)
                if not step_dict:
                    del self._zc_receivers[op_type][name]

            for name, step_dict in list(self._state_data[op_type].items()):
                step_dict.pop(step, None)
                if not step_dict:
                    del self._state_data[op_type][name]

            self._barriers[op_type].pop(step, None)

    async def clear_all(self) -> None:
        async with self._lock:
            for op in OpType:
                self._state_data[op].clear()
                self._tensor_slots[op].clear()
                self._zc_receivers[op].clear()
                self._barriers[op].clear()

    # ------------------------------------------------------------------
    # Backpressure monitoring
    # ------------------------------------------------------------------

    async def get_size(self) -> int:
        async with self._lock:
            size = 0
            for op in OpType:
                for step_dict in self._state_data[op].values():
                    size += len(step_dict)
                for step_dict in self._tensor_slots[op].values():
                    for rank_dict in step_dict.values():
                        size += len(rank_dict)
            return size

    async def get_capacity(self) -> int:
        return 100
