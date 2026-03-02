"""
Mailbox module for dual-stream convergence.

Data plane: tensor shards from upstream workers, indexed by (op_type, name, step).
Control plane: state scalars from Driver.

Completion tracking is event-driven, not polled:
  - Each (op_type, step) has a StepBarrier composed of ArrivalBarriers.
  - When all expected shards and states arrive, the barrier fires automatically.
  - Worker registers an on_ready callback.

Zero-copy shard writing is supported via TransportReceiver delegation.
"""

from fluxway.common.types.configs import BarrierConfig
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

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

    __slots__ = ("_tensor_barriers", "_state_barriers", "_combined", "_periodic")

    def __init__(
        self,
        expected_tensors: Dict[str, int],
        expected_states: List[str],
        periodic: bool = False,
    ) -> None:
        self._periodic = periodic
        self._tensor_barriers: Dict[str, ArrivalBarrier] = {
            name: ArrivalBarrier(count, periodic=periodic)
            for name, count in expected_tensors.items()
        }
        self._state_barriers: Dict[str, ArrivalBarrier] = {
            name: ArrivalBarrier(1, periodic=periodic) for name in expected_states
        }
        # Combined completion event (one-shot mode only)
        self._combined = asyncio.Event()
        if not periodic:
            self._check()  # fire immediately if both dicts are empty

    @property
    def arrived_tensors(self) -> List[str]:
        return [
            name for name, b in self._tensor_barriers.items() if b.count >= b.target
        ]

    @property
    def expected_tensors(self) -> List[str]:
        return list(self._tensor_barriers.keys())

    @property
    def tensor_progress(self) -> Dict[str, str]:
        return {
            name: f"{b.count}/{b.target}" for name, b in self._tensor_barriers.items()
        }

    # ------------------------------------------------------------------
    # Arrival notification
    # ------------------------------------------------------------------

    def tensor_arrived(self, name: str) -> bool:
        """Record one shard arriving for `name`. Returns True if barrier just fired."""
        barrier = self._tensor_barriers.get(name)
        if barrier is not None:
            fired = barrier.arrive()
            logger.debug(
                f"[StepBarrier] TENSOR ARRIVED: name={name}, count={barrier.count}/{barrier.target}, fired={fired}"
            )
            if self._periodic:
                return fired
            return self._check()
        else:
            logger.warning(
                f"[StepBarrier] UNEXPECTED TENSOR: name={name}, expected={list(self._tensor_barriers.keys())}"
            )
        return False

    def state_arrived(self, name: str) -> bool:
        """Record state data arriving for `name`. Returns True if barrier just fired."""
        barrier = self._state_barriers.get(name)
        if barrier is not None:
            fired = barrier.arrive()
            logger.debug(
                f"[StepBarrier] STATE ARRIVED: name={name}, count={barrier.count}/{barrier.target}, fired={fired}"
            )
            if self._periodic:
                return fired
            return self._check()
        else:
            logger.warning(
                f"[StepBarrier] UNEXPECTED STATE: name={name}, expected={list(self._state_barriers.keys())}"
            )
        return False

    def satisfy_tensor(self, name: str) -> bool:
        """Force a tensor barrier to be satisfied (e.g., via staleness fallback)."""
        barrier = self._tensor_barriers.get(name)
        if barrier is not None and not barrier.is_done():
            # Force the ArrivalBarrier to be done by simulating arrivals
            remaining = barrier._n - barrier._count
            for _ in range(remaining):
                barrier.arrive()
            return self._check()
        return False

    def satisfy_state(self, name: str) -> bool:
        """Force a state barrier to be satisfied."""
        barrier = self._state_barriers.get(name)
        if barrier is not None and not barrier.is_done():
            # Force the ArrivalBarrier to be done by simulating arrivals
            remaining = barrier._n - barrier._count
            for _ in range(remaining):
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

        # Log why it's not ready
        waiting_tensors = [
            n for n, b in self._tensor_barriers.items() if not b.is_done()
        ]
        waiting_states = [n for n, b in self._state_barriers.items() if not b.is_done()]
        logger.debug(
            f"[StepBarrier] NOT READY: waiting_tensors={waiting_tensors}, waiting_states={waiting_states}"
        )

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
        self._state_data: Dict[str, Dict[str, Dict[int, Any]]] = {}

        # Data plane (buffered path)
        self._tensor_slots: Dict[
            str, Dict[str, Dict[int, Dict[Tuple[str, int], List["TransportTensor"]]]]
        ] = {}

        # Completion barriers: {op_name: {step: StepBarrier}}
        self._barriers: Dict[str, Dict[int, StepBarrier]] = {}

        # Zero-copy receivers (activated on demand)
        self._zc_receivers: Dict[str, Dict[str, Dict[int, TransportReceiver]]] = {}

        # Called with (op_name, step) when a step barrier fires
        self._on_ready: Optional[Callable[[str, int], None]] = None

        # Periodic mode state: {op_name: current_virtual_step}
        self._periodic_steps: Dict[str, int] = {}
        # Mapping from periodic op to its perpetual barrier
        self._periodic_barriers: Dict[str, StepBarrier] = {}

        # Tolerant mode (staleness) config: {op_name: {name: staleness_value}}
        self._staleness_config: Dict[str, Dict[str, int]] = {}
        # Expected counts for satisfying staleness: {op_name: {name: expected_count}}
        self._expected_tensor_counts: Dict[str, Dict[str, int]] = {}
        self._expected_state_names: Dict[str, List[str]] = {}
        # Retrieval policy for tolerant params: {op_name: {name: policy_string}}
        self._retrieval_policy: Dict[str, Dict[str, str]] = {}

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Step barrier lifecycle
    # ------------------------------------------------------------------

    def set_ready_callback(self, callback: Callable[[str, int], None]) -> None:
        """
        Register a callback fired when a step becomes ready.

        The callback is invoked inside the asyncio.Lock, synchronously —
        use asyncio.create_task() inside it for non-blocking work.

        Args:
            callback: callable(op_name, step)
        """
        self._on_ready = callback

    def declare_step(
        self,
        op_name: str,
        step: int,
        expected_tensors: Dict[str, int],
        expected_states: List[str],
        barrier_config: Optional["BarrierConfig"] = None,
    ) -> None:
        """
        Declare the expected arrivals for a step.

        Must be called before the first deposit_shards() / deposit_states() for
        that step.  If not called, deposit_* operations still work but no
        barrier fires.

        Args:
            op_name: Operation name
            step: Execution step (ignored for periodic ops)
            expected_tensors: {tensor_name: required_shard_count}
            expected_states: list of required state names
            barrier_config: Configuration for the step barrier
        """
        from fluxway.common.types.configs import BarrierType

        is_periodic = (
            barrier_config.type == BarrierType.PERIODIC if barrier_config else False
        )

        staleness_dict = {}
        if barrier_config and barrier_config.type == BarrierType.TOLERANT:
            from fluxway.common.types.configs import TolerantBarrierConfig

            if isinstance(barrier_config, TolerantBarrierConfig):
                staleness_dict = barrier_config.staleness
                policy = barrier_config.retrieval_policy
            else:
                staleness_dict = {}
                policy = "latest"

            self._staleness_config[op_name] = staleness_dict
            self._expected_tensor_counts[op_name] = expected_tensors
            self._expected_state_names[op_name] = expected_states

            # Currently apply one policy per operator for all its staleness configs
            self._retrieval_policy.setdefault(op_name, {})
            for name in staleness_dict:
                self._retrieval_policy[op_name][name] = policy

        if is_periodic:
            if op_name not in self._periodic_barriers:
                # Use n from barrier_config if available
                # Note: StepBarrier currently assumes n is the same for all inputs in periodic mode.
                # For now we use barrier_config.n if it is PeriodicBarrierConfig.
                from fluxway.common.types.configs import PeriodicBarrierConfig

                n = (
                    barrier_config.n
                    if isinstance(barrier_config, PeriodicBarrierConfig)
                    else 1
                )
                self._periodic_barriers[op_name] = StepBarrier(
                    expected_tensors, expected_states, periodic=True
                )
                # Ensure each ArrivalBarrier inside StepBarrier uses the correct n
                for b in self._periodic_barriers[op_name]._tensor_barriers.values():
                    b._n = n
                for b in self._periodic_barriers[op_name]._state_barriers.values():
                    b._n = n

                self._periodic_steps[op_name] = 0
            return

        barrier = StepBarrier(expected_tensors, expected_states)
        self._barriers.setdefault(op_name, {})[step] = barrier
        logger.debug(
            f"[WorkerMailbox] declare_step: op={op_name}, step={step}, expected_tensors={expected_tensors}, expected_states={expected_states}"
        )

        # Check if we can satisfy any tolerant dependencies using cached older data
        for name, max_staleness in staleness_dict.items():
            if max_staleness <= 0:
                continue

            # For tensors
            if name in expected_tensors:
                expected = expected_tensors[name]
                if expected > 0:
                    found_valid = False
                    for k in range(step, max(-1, step - max_staleness - 1), -1):
                        cached_shards = (
                            self._tensor_slots.get(op_name, {}).get(name, {}).get(k, {})
                        )
                        if len(cached_shards) >= expected:
                            found_valid = True
                            break
                    if found_valid:
                        barrier.satisfy_tensor(name)

            # For states
            elif name in expected_states:
                found_valid = False
                for k in range(step, max(-1, step - max_staleness - 1), -1):
                    if k in self._state_data.get(op_name, {}).get(name, {}):
                        found_valid = True
                        break
                if found_valid:
                    barrier.satisfy_state(name)

        # In case expected counts are all zero the barrier is already ready
        if barrier.is_ready() and self._on_ready:
            self._on_ready(op_name, step)

    # ------------------------------------------------------------------
    # Control plane — state data
    # ------------------------------------------------------------------

    async def deposit_states(
        self, op_name: str, step: int, names: List[str], values: List[Any]
    ) -> None:
        """Cache state data and notify the step barrier."""
        if len(names) != len(values):
            raise ValueError("names and values must have the same length")

        async with self._lock:
            # Handle periodic ops: use virtual step
            actual_step = step
            barrier = None
            if op_name in self._periodic_barriers:
                actual_step = self._periodic_steps[op_name]
                barrier = self._periodic_barriers[op_name]
            else:
                barrier = self._barriers.get(op_name, {}).get(step)

            for name, data in zip(names, values, strict=True):  # type: ignore
                self._state_data.setdefault(op_name, {}).setdefault(name, {})[
                    actual_step
                ] = data
                if barrier is not None:
                    fired = barrier.state_arrived(name)
                    if fired:
                        if self._on_ready:
                            self._on_ready(op_name, actual_step)
                        # Periodic op consumes one virtual step for the entire bundle
                        if op_name in self._periodic_barriers:
                            self._periodic_steps[op_name] += 1
                            actual_step = self._periodic_steps[op_name]

                # Check if this state satisfies any FUTURE barriers that tolerate staleness
                staleness = self._staleness_config.get(op_name, {}).get(name, 0)
                if staleness > 0:
                    for s in range(actual_step + 1, actual_step + staleness + 1):
                        future_barrier = self._barriers.get(op_name, {}).get(s)
                        if future_barrier is not None:
                            fired_future = future_barrier.satisfy_state(name)
                            if fired_future and self._on_ready:
                                self._on_ready(op_name, s)
        logger.debug(
            "Mailbox deposit_states: op=%s names=%s step=%d", op_name, names, step
        )

    async def get_state(self, op_name: str, name: str, step: int) -> Optional[Any]:
        async with self._lock:
            staleness = self._staleness_config.get(op_name, {}).get(name, 0)
            if staleness > 0:
                policy = self._retrieval_policy.get(op_name, {}).get(name, "latest")
                step_dict = self._state_data.get(op_name, {}).get(name, {})
                valid_steps = [
                    k
                    for k in range(max(0, step - staleness), step + 1)
                    if k in step_dict
                ]
                if valid_steps:
                    best_k = valid_steps[0] if policy == "oldest" else valid_steps[-1]
                    return step_dict[best_k]

            return self._state_data.get(op_name, {}).get(name, {}).get(step)

    async def get_all_states(self, op_name: str, step: int) -> Dict[str, Any]:
        async with self._lock:
            result = {}
            for name, step_dict in self._state_data.get(op_name, {}).items():
                staleness = self._staleness_config.get(op_name, {}).get(name, 0)
                if staleness > 0:
                    policy = self._retrieval_policy.get(op_name, {}).get(name, "latest")
                    valid_steps = [
                        k
                        for k in range(max(0, step - staleness), step + 1)
                        if k in step_dict
                    ]
                    if valid_steps:
                        best_k = (
                            valid_steps[0] if policy == "oldest" else valid_steps[-1]
                        )
                        result[name] = step_dict[best_k]
                        continue

                if step in step_dict:
                    result[name] = step_dict[step]
            return result

    async def has_state(self, op_name: str, name: str, step: int) -> bool:
        async with self._lock:
            return step in self._state_data.get(op_name, {}).get(name, {})

    # ------------------------------------------------------------------
    # Data plane — tensor shards
    # ------------------------------------------------------------------

    def _buffered_slot(
        self, op_name: str, name: str, step: int
    ) -> Dict[Tuple[str, int], List["TransportTensor"]]:
        return (
            self._tensor_slots.setdefault(op_name, {})
            .setdefault(name, {})
            .setdefault(step, {})
        )

    async def deposit_shards(
        self,
        op_name: str,
        names: List[str],
        step: int,
        sender_id: str,
        rank: int,
        payload_groups: List[List[Any]],
        counts_groups: List[List[int]],
        is_ragged: List[bool],
        has_spatial_indices: List[bool],
        has_origin_indices: List[bool],
    ) -> int:
        """
        Cache multiple tensor shards and notify the step barrier.

        Args:
            op_name: Operation type
            names: List of tensor names
            step: Execution step
            sender_id: ID of the sender
            rank: Sender rank
            payload_groups: List of lists of raw data arrays (data, optional lengths, optional spatial_indices, optional origin_indices)
            counts_groups: List of lists of counts for each payload
            is_ragged: List of bools indicating if ragged lengths are included for each tensor.
            has_spatial_indices: List of bools indicating if routing indices are included for each tensor.
            has_origin_indices: List of bools indicating if assembly indices are included for each tensor.
        """
        async with self._lock:
            # Determine actual step (periodic or standard)
            if op_name in self._periodic_barriers:
                actual_step = self._periodic_steps[op_name]
                # self._periodic_steps[op_name] += 1 # This is moved to after the loop
                barrier = self._periodic_barriers[op_name]
            else:
                actual_step = step
                barrier = self._barriers.get(op_name, {}).get(step)

            for i, name in enumerate(names):
                logger.debug(
                    f"[WorkerMailbox] deposit_shards: op={op_name}, name={name}, step={step}, sender={sender_id}:{rank}"
                )
                payloads = payload_groups[i]
                counts = counts_groups[i]
                ragged = is_ragged[i]
                has_idx = has_spatial_indices[i]
                has_asmb = has_origin_indices[i]

                # Strict Duplicate Validation
                slot = (
                    self._tensor_slots.get(op_name, {})
                    .get(name, {})
                    .get(actual_step, {})
                )
                key = (sender_id, rank)
                if key in slot:
                    msg = (
                        f"[WorkerMailbox Error] DUPLICATE SHARD RECEIVED: op={op_name} name={name} step={actual_step} sender={sender_id} rank={rank}. "
                        "This indicates that the same worker is receiving the same shard from the same sender twice. "
                        f"Current slot content: {slot[key]}"
                    )
                    raise ValueError(msg)

                # Route to Zero-Copy or Buffered path
                zc = self._zc_receivers.get(op_name, {}).get(name, {}).get(actual_step)
                if zc is not None:
                    # Note: Zero-copy bundling is not yet implemented; use first payload/count
                    zc.receive_slices(rank, payloads, counts)
                else:
                    self._deposit_buffered_shards(
                        op_name,
                        name,
                        actual_step,
                        sender_id,
                        rank,
                        payloads,
                        counts,
                        is_ragged=ragged,
                        has_spatial_indices=has_idx,
                        has_origin_indices=has_asmb,
                    )

                # Notify barrier ONCE per sender for this tensor in the bundle
                if barrier is not None:
                    if barrier.tensor_arrived(name):
                        if self._on_ready:
                            self._on_ready(op_name, actual_step)

                        # Periodic op consumes one virtual step for the entire bundle
                        if op_name in self._periodic_barriers:
                            self._periodic_steps[op_name] += 1
                            actual_step = self._periodic_steps[op_name]

                # Check if this tensor satisfies any FUTURE barriers that tolerate staleness
                staleness = self._staleness_config.get(op_name, {}).get(name, 0)
                if staleness > 0:
                    expected = self._expected_tensor_counts.get(op_name, {}).get(
                        name, 0
                    )
                    current_slot = (
                        self._tensor_slots.get(op_name, {})
                        .get(name, {})
                        .get(actual_step, {})
                    )
                    if expected > 0 and len(current_slot) >= expected:
                        for s in range(actual_step + 1, actual_step + staleness + 1):
                            future_barrier = self._barriers.get(op_name, {}).get(s)
                            if future_barrier is not None:
                                fired_future = future_barrier.satisfy_tensor(name)
                                if fired_future and self._on_ready:
                                    self._on_ready(op_name, s)

        return await self.get_size()

    def _deposit_buffered_shards(
        self,
        op_name: str,
        name: str,
        step: int,
        sender_id: str,
        rank: int,
        payloads: List[Any],
        counts: List[int],
        is_ragged: bool = False,
        has_spatial_indices: bool = False,
        has_origin_indices: bool = False,
    ) -> int:
        """Internal helper for buffered shard storage."""
        from fluxway.common.types.tensor import TransportTensor
        from fluxway.common.backend import get_backend

        cursor = 0
        backend = get_backend()
        tensors = []

        while cursor < len(payloads):
            count = counts[cursor] if cursor < len(counts) else 0
            data_payload = payloads[cursor]
            if count > 0:
                # Use count for slicing only if the tensor has a non-empty shape (is not a scalar)
                data_payload = (
                    data_payload[:count]
                    if backend.shape(data_payload)
                    else data_payload
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
            if has_spatial_indices and cursor < len(payloads):
                index_count = counts[cursor] if cursor < len(counts) else 0
                indices_payload = (
                    payloads[cursor][:index_count]
                    if index_count > 0
                    else payloads[cursor]
                )
                cursor += 1

            assembly_payload = None
            if has_origin_indices and cursor < len(payloads):
                asmb_count = counts[cursor] if cursor < len(counts) else 0
                assembly_payload = (
                    payloads[cursor][:asmb_count]
                    if asmb_count > 0
                    else payloads[cursor]
                )
                cursor += 1

            nested_splits = []
            if lengths_payload is not None:
                cs = backend.cumsum(lengths_payload, dim=0)
                zero = backend.zeros((1,), dtype=backend.int32())
                splits = backend.cat([zero, cs], dim=0)
                nested_splits = [splits]

            tensors.append(
                TransportTensor(
                    data_payload,
                    nested_splits=nested_splits,
                    spatial_indices=indices_payload,
                    origin_indices=assembly_payload,
                )
            )

        key = (sender_id, rank)
        self._buffered_slot(op_name, name, step)[key] = tensors

        # Debug Trace for barrier status
        barrier = self._barriers.get(op_name, {}).get(step)
        if barrier and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[WorkerMailbox] Shard stored: op={op_name} name={name} step={step} rank={rank}"
            )
        return int(counts[0]) if counts else 0

    async def get_all_tensor_shards(
        self, op_name: str, step: int
    ) -> Dict[str, Dict[Tuple[str, int], List["TransportTensor"]]]:
        """Get all received shards for a step."""
        async with self._lock:
            result = {}
            for name, step_dict in self._tensor_slots.get(op_name, {}).items():
                if step in step_dict:
                    result[name] = dict(step_dict[step])
            return result

    def _get_barrier(self, op_name: str, step: int) -> Optional[StepBarrier]:
        """Internal helper to get barrier."""
        if op_name in self._periodic_barriers:
            return self._periodic_barriers[op_name]
        return self._barriers.get(op_name, {}).get(step)

    async def get_tensor_shard(
        self,
        op_name: str,
        name: str,
        step: int,
        rank: int,
        sender_id: Optional[str] = None,
    ) -> Optional[List["TransportTensor"]]:
        async with self._lock:
            slot = self._tensor_slots.get(op_name, {}).get(name, {}).get(step, {})
            if sender_id is not None:
                return slot.get((sender_id, rank))

            # Search for the first key that matches the rank
            for (sid, r), tensors in slot.items():
                if r == rank:
                    return tensors
            return None

    async def get_tensor_shards(
        self, op_name: str, name: str, step: int
    ) -> Dict[Tuple[str, int], List["TransportTensor"]]:
        async with self._lock:
            staleness = self._staleness_config.get(op_name, {}).get(name, 0)
            if staleness > 0:
                expected = self._expected_tensor_counts.get(op_name, {}).get(name, 0)
                if expected > 0:
                    policy = self._retrieval_policy.get(op_name, {}).get(name, "latest")
                    step_dict = self._tensor_slots.get(op_name, {}).get(name, {})
                    valid_steps = [
                        k
                        for k in range(max(0, step - staleness), step + 1)
                        if len(step_dict.get(k, {})) >= expected
                    ]
                    if valid_steps:
                        best_k = (
                            valid_steps[0] if policy == "oldest" else valid_steps[-1]
                        )
                        return dict(step_dict[best_k])

            return dict(self._tensor_slots.get(op_name, {}).get(name, {}).get(step, {}))

    async def get_all_states_shards(self, op_name: str, step: int) -> Dict[str, Any]:
        async with self._lock:
            result = {}
            for name, step_dict in self._tensor_slots.get(op_name, {}).items():
                staleness = self._staleness_config.get(op_name, {}).get(name, 0)
                expected = self._expected_tensor_counts.get(op_name, {}).get(name, 0)
                if staleness > 0 and expected > 0:
                    policy = self._retrieval_policy.get(op_name, {}).get(name, "latest")
                    valid_steps = [
                        k
                        for k in range(max(0, step - staleness), step + 1)
                        if len(step_dict.get(k, {})) >= expected
                    ]
                    if valid_steps:
                        best_k = (
                            valid_steps[0] if policy == "oldest" else valid_steps[-1]
                        )
                        result[name] = dict(step_dict[best_k])
                        continue

                if step in step_dict:
                    result[name] = dict(step_dict[step])
            return result

    async def get_tensor_shard_count(self, op_name: str, name: str, step: int) -> int:
        async with self._lock:
            return len(self._tensor_slots.get(op_name, {}).get(name, {}).get(step, {}))

    # ------------------------------------------------------------------
    # Step readiness
    # ------------------------------------------------------------------

    def is_step_ready(self, op_name: str, step: int) -> bool:
        """Non-blocking: True if the step barrier has fired."""
        barrier = self._barriers.get(op_name, {}).get(step)
        return barrier is not None and barrier.is_ready()

    async def wait_step_ready(self, op_name: str, step: int) -> None:
        """Async wait until the step barrier fires.

        Raises RuntimeError if declare_step() was never called for this step.
        """
        barrier = self._get_barrier(op_name, step)
        if barrier is not None:
            await barrier.wait()
        else:
            raise RuntimeError(
                f"No barrier declared for op={op_name} step={step}. "
                f"Call declare_step() first."
            )

    # ------------------------------------------------------------------
    # Zero-copy registration
    # ------------------------------------------------------------------

    async def register_zero_copy_slot(
        self,
        op_name: str,
        name: str,
        step: int,
        target_buffers: List[Any],
        expect_offsets: dict,
    ) -> None:
        """Enable zero-copy write-through for a tensor slot."""
        async with self._lock:
            receiver = TransportReceiver(wait_for_n_slices=0)
            receiver.register_zero_copy_buffers(target_buffers, expect_offsets)
            self._zc_receivers.setdefault(op_name, {}).setdefault(name, {})[step] = (
                receiver
            )

    async def get_zero_copy_buffers(
        self, op_name: str, name: str, step: int
    ) -> Optional[List[Any]]:
        async with self._lock:
            receiver = self._zc_receivers.get(op_name, {}).get(name, {}).get(step)
            return receiver.target_buffers if receiver is not None else None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def clear_step(self, op_name: str, step: int) -> None:
        """Release all data for a completed step, respecting staleness retention."""
        async with self._lock:
            if op_name in self._tensor_slots:
                for name, step_dict in list(self._tensor_slots[op_name].items()):
                    staleness = self._staleness_config.get(op_name, {}).get(name, 0)
                    clear_target = step - staleness
                    step_dict.pop(clear_target, None)
                    if not step_dict:
                        del self._tensor_slots[op_name][name]

            if op_name in self._zc_receivers:
                for name, step_dict in list(self._zc_receivers[op_name].items()):
                    staleness = self._staleness_config.get(op_name, {}).get(name, 0)
                    clear_target = step - staleness
                    step_dict.pop(clear_target, None)
                    if not step_dict:
                        del self._zc_receivers[op_name][name]

            if op_name in self._state_data:
                for name, step_dict in list(self._state_data[op_name].items()):
                    staleness = self._staleness_config.get(op_name, {}).get(name, 0)
                    clear_target = step - staleness
                    step_dict.pop(clear_target, None)
                    if not step_dict:
                        del self._state_data[op_name][name]

            if op_name in self._barriers:
                self._barriers[op_name].pop(step, None)

    async def clear_all(self) -> None:
        async with self._lock:
            self._state_data.clear()
            self._tensor_slots.clear()
            self._zc_receivers.clear()
            self._barriers.clear()

    # ------------------------------------------------------------------
    # Backpressure monitoring
    # ------------------------------------------------------------------

    async def get_size(self) -> int:
        async with self._lock:
            size = 0
            for op_name_dict in self._state_data.values():
                for step_dict in op_name_dict.values():
                    size += len(step_dict)
            for op_name_dict in self._tensor_slots.values():
                for step_dict in op_name_dict.values():
                    for rank_dict in step_dict.values():
                        size += len(rank_dict)
            return size

    async def get_capacity(self) -> int:
        return 100
