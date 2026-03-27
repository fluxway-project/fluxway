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

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from fluxway.common.backend import get_backend
from fluxway.common.index_descriptor import ContinuousIndex
from fluxway.common.sync import ArrivalBarrier
from fluxway.common.types.configs import BarrierConfig, TolerantBarrierConfig
from fluxway.common.types.tensor import TransportTensor, Scalar
from fluxway.transport.communication import TransportReceiver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# StepBarrier — one per (op_type, step)
# ---------------------------------------------------------------------------


class StepBarrier:
    """
    Fires when all declared tensor names have received their expected shard
    counts AND all declared state names have been received.

    Internally composes one ArrivalBarrier per named stream.

    Args:
        expected_tensors: {tuuid: required_shard_count}
        expected_states:  list of required state names
    """

    __slots__ = (
        "_op_name",
        "_step",
        "_tensor_barriers",
        "_control_arg_barriers",
        "_combined",
        "_periodic",
    )

    def __init__(
        self,
        op_name: str,
        step: int,
        expected_tensors: Dict[int, int],
        expected_states: List[str],
        periodic: bool = False,
    ) -> None:
        self._op_name = op_name
        self._step = step
        self._periodic = periodic
        self._tensor_barriers: Dict[int, ArrivalBarrier] = {
            u: ArrivalBarrier(count, periodic=periodic)
            for u, count in expected_tensors.items()
        }
        self._control_arg_barriers: Dict[str, ArrivalBarrier] = {
            name: ArrivalBarrier(1, periodic=periodic) for name in expected_states
        }
        # Combined completion event (one-shot mode only)
        self._combined = asyncio.Event()
        if not periodic:
            self._check()  # fire immediately if both dicts are empty

    @property
    def arrived_tensors(self) -> List[int]:
        return [u for u, b in self._tensor_barriers.items() if b.count >= b.target]

    @property
    def expected_tensors(self) -> List[int]:
        return list(self._tensor_barriers.keys())

    @property
    def tensor_progress(self) -> Dict[int, str]:
        return {u: f"{b.count}/{b.target}" for u, b in self._tensor_barriers.items()}

    # ------------------------------------------------------------------
    # Arrival notification
    # ------------------------------------------------------------------

    def tensor_arrived(self, tuuid: int, current_step: Optional[int] = None) -> bool:
        """Record one shard arriving for tensor TUUID. Returns True if barrier just fired."""
        barrier = self._tensor_barriers.get(tuuid)
        if barrier is not None:
            fired = barrier.arrive()
            step_to_log = current_step if current_step is not None else self._step
            logger.debug(
                f"[StepBarrier {self._op_name}:{step_to_log}] TENSOR ARRIVED: tuuid={tuuid}, count={barrier.count}/{barrier.target}, fired={fired}"
            )
            if self._periodic:
                return fired
            return self._check(current_step=current_step)
        else:
            logger.warning(
                f"[StepBarrier] UNEXPECTED TENSOR: tuuid={tuuid}, expected={list(self._tensor_barriers.keys())}"
            )
        return False

    def control_arg_arrived(
        self, name: str, current_step: Optional[int] = None
    ) -> bool:
        """Record state data arriving for `name`. Returns True if barrier just fired."""
        barrier = self._control_arg_barriers.get(name)
        if barrier is not None:
            fired = barrier.arrive()
            step_to_log = current_step if current_step is not None else self._step
            logger.debug(
                f"[StepBarrier {self._op_name}:{step_to_log}] STATE ARRIVED: name={name}, count={barrier.count}/{barrier.target}, fired={fired}"
            )
            if self._periodic:
                return fired
            return self._check(current_step=current_step)
        else:
            logger.warning(
                f"[StepBarrier {self._op_name}:{self._step}] UNEXPECTED STATE: name={name}, expected={list(self._control_arg_barriers.keys())}"
            )
        return False

    def satisfy_tensor(self, tuuid: int, current_step: Optional[int] = None) -> bool:
        """Force a tensor barrier to be satisfied (e.g., via staleness fallback)."""
        barrier = self._tensor_barriers.get(tuuid)
        if barrier is not None and not barrier.is_done():
            # Force the ArrivalBarrier to be done by simulating arrivals
            remaining = barrier._n - barrier._count
            for _ in range(remaining):
                barrier.arrive()
            return self._check(current_step=current_step)
        return False

    def satisfy_control_arg(
        self, name: str, current_step: Optional[int] = None
    ) -> bool:
        """Force a state barrier to be satisfied."""
        barrier = self._control_arg_barriers.get(name)
        if barrier is not None and not barrier.is_done():
            # Force the ArrivalBarrier to be done by simulating arrivals
            remaining = barrier._n - barrier._count
            for _ in range(remaining):
                barrier.arrive()
            return self._check(current_step=current_step)
        return False

    # ------------------------------------------------------------------
    # Completion query
    # ------------------------------------------------------------------

    def _check(self, current_step: Optional[int] = None) -> bool:
        if self._combined.is_set():
            return False
        tensors_ok = all(b.is_done() for b in self._tensor_barriers.values())
        states_ok = all(b.is_done() for b in self._control_arg_barriers.values())

        step_to_log = current_step if current_step is not None else self._step

        if tensors_ok and states_ok:
            logger.debug(f"[StepBarrier {self._op_name}:{step_to_log}] FIRED.")
            self._combined.set()
            return True

        # Log why it's not ready
        waiting_tensors = [
            n for n, b in self._tensor_barriers.items() if not b.is_done()
        ]
        waiting_states = [
            n for n, b in self._control_arg_barriers.items() if not b.is_done()
        ]
        logger.debug(
            f"[StepBarrier {self._op_name}:{step_to_log}] NOT READY: waiting_tensors={waiting_tensors}, waiting_states={waiting_states}"
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
        self._control_arg_data: Dict[str, Dict[str, Dict[int, Any]]] = {}

        # Data plane (buffered path)
        self._tensor_slots: Dict[
            str, Dict[int, Dict[int, Dict[Tuple[str, int], List["TransportTensor"]]]]
        ] = {}

        # Completion barriers: {op_name: {step: StepBarrier}}
        self._barriers: Dict[str, Dict[int, StepBarrier]] = {}

        # Zero-copy receivers (activated on demand)
        self._zc_receivers: Dict[str, Dict[int, Dict[int, TransportReceiver]]] = {}

        # Called with (op_name, step) when a step barrier fires
        self._on_ready: Optional[Callable[[str, int], None]] = None

        # Periodic mode state: {op_name: current_virtual_step}
        self._periodic_steps: Dict[str, int] = {}
        # Mapping from periodic op to its perpetual barrier
        self._periodic_barriers: Dict[str, StepBarrier] = {}

        # Tolerant mode (staleness) config: {op_name: {name_or_uuid: staleness_value}}
        self._staleness_config: Dict[str, Dict[Any, int]] = {}
        # Expected counts for satisfying staleness: {op_name: {name_or_uuid: expected_count}}
        self._expected_tensor_counts: Dict[str, Dict[int, int]] = {}
        self._expected_control_arg_names: Dict[str, List[str]] = {}
        # Retrieval policy for tolerant params: {op_name: {name_or_uuid: policy_string}}
        self._retrieval_policy: Dict[str, Dict[Any, str]] = {}

        # Barrier configurations: {op_name: BarrierConfig}
        self._barrier_configs: Dict[str, "BarrierConfig"] = {}

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
        expected_tensors: Dict[int, int],
        expected_states: List[str],
        barrier_config: Optional["BarrierConfig"] = None,
    ) -> None:
        """
        Declare the expected arrivals for a step.

        Must be called before the first deposit_shards() / deposit_control_arguments() for
        that step.  If not called, deposit_* operations still work but no
        barrier fires.

        Args:
            op_name: Operation name
            step: Execution step (ignored for periodic ops)
            expected_tensors: {tuuid: required_shard_count}
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
                # Merge tensor and state staleness for internal lookback
                staleness_dict = {
                    **barrier_config.tensor_staleness,
                    **barrier_config.state_staleness,
                }
                policy = barrier_config.retrieval_policy
            else:
                staleness_dict = {}
                policy = "latest"

            self._staleness_config[op_name] = staleness_dict
            self._expected_tensor_counts[op_name] = expected_tensors
            self._expected_control_arg_names[op_name] = expected_states
            self._barrier_configs[op_name] = barrier_config

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
                    op_name, -1, expected_tensors, expected_states, periodic=True
                )
                # Ensure each ArrivalBarrier inside StepBarrier uses the correct n
                for b in self._periodic_barriers[op_name]._tensor_barriers.values():
                    b._n = n
                for b in self._periodic_barriers[
                    op_name
                ]._control_arg_barriers.values():
                    b._n = n

                self._periodic_steps[op_name] = 0
            return

        if op_name in self._barriers and step in self._barriers[op_name]:
            old_b = self._barriers[op_name][step]
            logger.warning(
                f"[WorkerMailbox] DUPLICATE declare_step: op={op_name}, step={step}. "
                f"Progress: {old_b.tensor_progress}"
            )

        barrier = StepBarrier(op_name, step, expected_tensors, expected_states)
        self._barriers.setdefault(op_name, {})[step] = barrier
        logger.debug(
            f"[WorkerMailbox] declare_step: op={op_name}, step={step}, expected_tensors={expected_tensors}, expected_states={expected_states}"
        )

        # Check if we can satisfy any tolerant dependencies using cached older data
        for ident, max_staleness in staleness_dict.items():
            if max_staleness <= 0:
                continue

            # For tensors: TUUIDs are ints
            if isinstance(ident, int) and ident in expected_tensors:
                expected = expected_tensors[ident]
                if expected > 0:
                    found_valid = False
                    for k in range(step, max(-1, step - max_staleness - 1), -1):
                        cached_shards = (
                            self._tensor_slots.get(op_name, {})
                            .get(ident, {})
                            .get(k, {})
                        )
                        # Count total SHARDS across all senders, not just the number of senders
                        shards_count = sum(
                            len(s_list) for s_list in cached_shards.values()
                        )
                        if shards_count >= expected:
                            found_valid = True
                            break
                    if found_valid:
                        barrier.satisfy_tensor(ident)

            # For control states: Names are strings
            elif isinstance(ident, str) and ident in expected_states:
                found_valid = False
                for k in range(step, max(-1, step - max_staleness - 1), -1):
                    if k in self._control_arg_data.get(op_name, {}).get(ident, {}):
                        found_valid = True
                        break
                if found_valid:
                    barrier.satisfy_control_arg(ident)

        # Check if any data for THIS EXACT step has already arrived.
        # This prevents progress loss if declare_step is called after some deposit_* calls.
        if expected_tensors:
            for tuuid, target in expected_tensors.items():
                arrived_shards = (
                    self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step, {})
                )
                for sender_shard_list in arrived_shards.values():
                    for _ in range(len(sender_shard_list)):
                        barrier.tensor_arrived(tuuid)
        # Check for states
        for name in expected_states:
            if step in self._control_arg_data.get(op_name, {}).get(name, {}):
                barrier.control_arg_arrived(name)

        # In case expected counts are all zero the barrier is already ready
        if barrier.is_ready() and self._on_ready:
            self._on_ready(op_name, step)

    # ------------------------------------------------------------------
    # Control plane — state data
    # ------------------------------------------------------------------

    async def deposit_control_arguments(
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

            any_fired = False
            for name, data in zip(names, values, strict=True):  # type: ignore
                self._control_arg_data.setdefault(op_name, {}).setdefault(name, {})[
                    actual_step
                ] = data
                if barrier is not None:
                    if barrier.control_arg_arrived(name, current_step=actual_step):
                        any_fired = True

                # Check if this state satisfies any FUTURE barriers that tolerate staleness
                staleness = self._staleness_config.get(op_name, {}).get(name, 0)
                if staleness > 0:
                    for s in range(actual_step + 1, actual_step + staleness + 1):
                        future_barrier = self._barriers.get(op_name, {}).get(s)
                        if future_barrier is not None:
                            fired_future = future_barrier.satisfy_control_arg(
                                name, current_step=s
                            )
                            if fired_future and self._on_ready:
                                self._on_ready(op_name, s)

            if any_fired:
                if op_name in self._periodic_barriers:
                    self._periodic_steps[op_name] += 1
                if self._on_ready:
                    self._on_ready(op_name, actual_step)
        logger.debug(
            "Mailbox deposit_control_arguments: op=%s names=%s step=%d",
            op_name,
            names,
            step,
        )

    async def get_control_argument(
        self, op_name: str, name: str, step: int
    ) -> Optional[Any]:
        async with self._lock:
            staleness = self._staleness_config.get(op_name, {}).get(name, 0)
            if staleness > 0:
                policy = self._retrieval_policy.get(op_name, {}).get(name, "latest")
                step_dict = self._control_arg_data.get(op_name, {}).get(name, {})
                valid_steps = [
                    k
                    for k in range(max(0, step - staleness), step + 1)
                    if k in step_dict
                ]
                if valid_steps:
                    best_k = valid_steps[0] if policy == "oldest" else valid_steps[-1]
                    return step_dict[best_k]

            return self._control_arg_data.get(op_name, {}).get(name, {}).get(step)

    async def get_all_control_arguments(
        self, op_name: str, step: int
    ) -> Dict[str, Any]:
        async with self._lock:
            result = {}
            for name, step_dict in self._control_arg_data.get(op_name, {}).items():
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

    async def has_control_argument(self, op_name: str, name: str, step: int) -> bool:
        async with self._lock:
            return step in self._control_arg_data.get(op_name, {}).get(name, {})

    # ------------------------------------------------------------------
    # Data plane — tensor shards
    # ------------------------------------------------------------------

    def _buffered_slot(
        self, op_name: str, tuuid: int, step: int
    ) -> Dict[Tuple[str, int], List["TransportTensor"]]:
        return (
            self._tensor_slots.setdefault(op_name, {})
            .setdefault(tuuid, {})
            .setdefault(step, {})
        )

    async def deposit_shards(
        self,
        op_name: str,
        tuuids: List[int],
        step: int,
        sender_id: str,
        rank: int,
        sender_world_size: int,
        payloads: List[Any],
        counts: List[int],
        global_shapes: Optional[List[Optional[Tuple[int, ...]]]] = None,
        meta_tensor_state_list: Optional[List[int]] = None,
        slice_compact_desc_params: Optional[List[Optional[List[int]]]] = None,
    ) -> int:
        """
        Cache multiple tensor shards and notify the step barrier.

        Args:
            op_name: Operation type
            tuuids: List of TUUIDs
            step: Execution step
            sender_id: ID of the sender
            rank: Sender rank
            payload_groups: List of lists of raw data arrays (data, lengths, routing_idx, origin_idx)
            counts_groups: List of lists of counts for each payload
            meta_tensor_state_list: List of ints representing the bitmask state of each tensor
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

            any_fired = False
            cursor = 0
            for i, u in enumerate(tuuids):
                logger.debug(
                    f"[WorkerMailbox] deposit_shards: op={op_name}, tuuid={u}, step={step}, sender={sender_id}:{rank}"
                )

                meta_state = meta_tensor_state_list[i] if meta_tensor_state_list else 0
                is_ragged = bool(meta_state & 1)
                has_origin_indices = bool(meta_state & 2)
                has_compact_index = bool(meta_state & 4)
                compact_params = (
                    slice_compact_desc_params[i] if slice_compact_desc_params else None
                )

                # Each TransportTensor can have multiple physical parts in the flat list:
                # 1. Tensor Data (Always)
                # 2. Ragged Lengths (if bit 1)
                # 3. Origin Indices (if bit 2 and NOT compact)
                n_elements = 1
                if is_ragged:
                    n_elements += 1
                if has_origin_indices:
                    n_elements += 1

                sub_payloads = payloads[cursor : cursor + n_elements]
                sub_counts = counts[cursor : cursor + n_elements]
                cursor += n_elements

                global_shape = (
                    global_shapes[i]
                    if global_shapes is not None and i < len(global_shapes)
                    else None
                )

                # Strict Duplicate Validation
                slot = (
                    self._tensor_slots.get(op_name, {}).get(u, {}).get(actual_step, {})
                )
                key = (sender_id, rank)
                if key in slot:
                    msg = (
                        f"[WorkerMailbox Error] DUPLICATE SHARD RECEIVED: op={op_name} tuuid={u} step={actual_step} sender={sender_id} rank={rank}. "
                        "This indicates that the same worker is receiving the same shard from the same sender twice. "
                        f"Current slot content: {slot[key]}"
                    )
                    raise ValueError(msg)

                # Route to Zero-Copy or Buffered path
                zc = self._zc_receivers.get(op_name, {}).get(u, {}).get(actual_step)
                if zc is not None:
                    # Note: Zero-copy bundling is not yet implemented; use flat sub-lists
                    zc.receive_slices(rank, sub_payloads, sub_counts)
                else:
                    self._deposit_buffered_shards(
                        op_name,
                        u,
                        actual_step,
                        sender_id,
                        rank,
                        sender_world_size,
                        sub_payloads,
                        sub_counts,
                        global_shape=global_shape,
                        is_ragged=is_ragged,
                        has_origin_indices=has_origin_indices,
                        has_compact_index=has_compact_index,
                        compact_desc_params=compact_params,
                    )

                # Notify barrier ONCE per sender for this tensor in the bundle
                if barrier is not None:
                    if barrier.tensor_arrived(u, current_step=actual_step):
                        any_fired = True

                # Check if this tensor satisfies any FUTURE barriers that tolerate staleness
                staleness = self._staleness_config.get(op_name, {}).get(u, 0)
                if staleness > 0:
                    expected = self._expected_tensor_counts.get(op_name, {}).get(u, 0)
                    current_slot = (
                        self._tensor_slots.get(op_name, {})
                        .get(u, {})
                        .get(actual_step, {})
                    )
                    if expected > 0 and len(current_slot) >= expected:
                        for s in range(actual_step + 1, actual_step + staleness + 1):
                            future_barrier = self._barriers.get(op_name, {}).get(s)
                            if future_barrier is not None:
                                fired_future = future_barrier.satisfy_tensor(
                                    u, current_step=s
                                )
                                if fired_future and self._on_ready:
                                    self._on_ready(op_name, s)

            # Fire readiness ONCE per bundle if any barrier satisfied
            if any_fired:
                if op_name in self._periodic_barriers:
                    # Periodic op consumes one virtual step for the entire bundle
                    self._periodic_steps[op_name] += 1

                if self._on_ready:
                    self._on_ready(op_name, actual_step)

        return self._get_size_lock_free()

    def _deposit_buffered_shards(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        sender_id: str,
        rank: int,
        sender_world_size: int,
        payloads: List[Any],
        counts: List[int],
        global_shape: Optional[Tuple[int, ...]] = None,
        is_ragged: bool = False,
        has_origin_indices: bool = False,
        has_compact_index: bool = False,
        compact_desc_params: Optional[List[int]] = None,
    ) -> int:
        """Internal helper for buffered shard storage."""

        cursor = 0
        backend = get_backend()
        tensors = []

        def _unwrap(p):
            return p.tensor if isinstance(p, Scalar) else p

        while cursor < len(payloads):
            count = counts[cursor] if cursor < len(counts) else 0
            data_payload = _unwrap(payloads[cursor])
            if backend.shape(data_payload):
                data_payload = data_payload[:count]

            cursor += 1

            lengths_payload = None
            if is_ragged and cursor < len(payloads):
                length_count = counts[cursor] if cursor < len(counts) else 0
                lengths_payload = _unwrap(payloads[cursor])
                if backend.shape(lengths_payload):
                    lengths_payload = lengths_payload[:length_count]
                cursor += 1

            origin_indices = None
            if has_origin_indices or has_compact_index:
                if has_compact_index and compact_desc_params:
                    origin_indices = ContinuousIndex.from_compact_params(
                        compact_desc_params
                    )
                elif cursor < len(payloads):
                    asmb_count = counts[cursor] if cursor < len(counts) else 0
                    origin_indices = _unwrap(payloads[cursor])
                    if backend.shape(origin_indices):
                        origin_indices = origin_indices[:asmb_count]
                    cursor += 1
                else:
                    raise ValueError(
                        f"[WorkerMailbox Error] MISSING ORIGIN INDICES: op={op_name} tuuid={tuuid} step={step} sender={sender_id} rank={rank}. "
                        f"Expected payload at cursor {cursor}, but total payloads is {len(payloads)}."
                    )

            # Assemble nested splits if ragged.
            # lengths_payload arrives as a host-level int32 array: either a numpy
            # ndarray or a backend tensor that must be moved to host before cumsum.
            # TransportTensor.nested_splits is typed List[np.ndarray], so we always
            # produce a numpy array here — no GPU kernels involved.
            nested_splits = []
            if lengths_payload is not None:
                if not hasattr(lengths_payload, "__len__"):
                    raise TypeError(
                        f"lengths_payload must be an array-like, got {type(lengths_payload)}"
                    )
                lengths_np = (
                    lengths_payload
                    if isinstance(lengths_payload, np.ndarray)
                    else backend.to_numpy(lengths_payload)
                )
                nested_splits = [
                    np.concatenate([np.zeros(1, dtype=np.int32), np.cumsum(lengths_np)])
                ]

            # Create local receipt for the shard (Zero-Overhead: reconstructed locally)
            from fluxway.common.types.tensor import SymmetryReceipt

            shard_receipt = SymmetryReceipt(sender_world_size, rank)

            tensors.append(
                TransportTensor(
                    data_payload,
                    nested_splits=nested_splits,
                    origin_indices=origin_indices,
                    routing_indices=None,
                    global_shape=global_shape,
                    receipt=shard_receipt,
                )
            )

        key = (sender_id, rank)
        self._buffered_slot(op_name, tuuid, step)[key] = tensors

        # Debug Trace for barrier status
        barrier = self._barriers.get(op_name, {}).get(step)
        if barrier and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[WorkerMailbox] Shard stored: op={op_name} tuuid={tuuid} step={step} rank={rank}"
            )
        return int(counts[0]) if counts else 0

    async def get_all_tensor_shards(
        self, op_name: str, step: int
    ) -> Dict[int, Dict[Tuple[str, int], List[TransportTensor]]]:
        """Get all received shards for a step, prioritizing Zero-Copy buffers."""
        async with self._lock:
            # Combine all TUUIDs for this operator
            all_tuuids = set(self._tensor_slots.get(op_name, {}).keys()) | set(
                self._zc_receivers.get(op_name, {}).keys()
            )

            result: Dict[int, Dict[Tuple[str, int], List[TransportTensor]]] = {}
            # We want to collect ALL tensors expected by this operator, including stale ones
            # First, check what TUUIDs this operator expects
            expected_counts = self._expected_tensor_counts.get(op_name, {})

            # Combine all TUUIDs from current slots and expected list
            all_tuuids = (
                set(self._tensor_slots.get(op_name, {}).keys())
                | set(self._zc_receivers.get(op_name, {}).keys())
                | set(expected_counts.keys())
            )

            for tuuid in all_tuuids:
                # 1. Try exact step (ZC first)
                zc = self._zc_receivers.get(op_name, {}).get(tuuid, {}).get(step)
                if zc and zc.target_buffers:
                    zc_tensor = TransportTensor(*zc.target_buffers)
                    result[tuuid] = {("__zero_copy__", -1): [zc_tensor]}
                    continue

                slotted = self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step)
                if slotted and len(slotted) >= expected_counts.get(tuuid, 0):
                    result[tuuid] = dict(slotted)
                    continue

                # 2. Try staleness lookback
                # Need to check the barrier config to be sure about TUUID staleness
                barrier_config = self._get_barrier_config(op_name)
                max_staleness = 0
                if isinstance(barrier_config, TolerantBarrierConfig):
                    max_staleness = barrier_config.tensor_staleness.get(tuuid, 0)

                if max_staleness > 0:
                    policy = self._retrieval_policy.get(op_name, {}).get(
                        tuuid, "latest"
                    )
                    step_slots = self._tensor_slots.get(op_name, {}).get(tuuid, {})
                    target_count = expected_counts.get(tuuid, 0)

                    valid_steps = [
                        k
                        for k in range(max(0, step - max_staleness), step + 1)
                        if k in step_slots and len(step_slots[k]) >= target_count
                    ]

                    if valid_steps:
                        best_k = (
                            valid_steps[0] if policy == "oldest" else valid_steps[-1]
                        )
                        result[tuuid] = dict(step_slots[best_k])
                        continue

                # 3. Last resort: just return whatever arrived for this step even if incomplete
                # (Standard behavior if no staleness defined)
                if slotted:
                    result[tuuid] = dict(slotted)

            return result

    def _get_barrier(self, op_name: str, step: int) -> Optional[StepBarrier]:
        """Internal helper to get barrier."""
        if op_name in self._periodic_barriers:
            return self._periodic_barriers[op_name]
        return self._barriers.get(op_name, {}).get(step)

    def _get_barrier_config(self, op_name: str) -> Optional["BarrierConfig"]:
        """Internal helper to get barrier config."""
        return self._barrier_configs.get(op_name)

    async def get_tensor_shard(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        rank: int,
        sender_id: Optional[str] = None,
    ) -> Optional[List["TransportTensor"]]:
        async with self._lock:
            slot = self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step, {})
            if sender_id is not None:
                return slot.get((sender_id, rank))

            # Search for the first key that matches the rank
            for (sid, r), tensors in slot.items():
                if r == rank:
                    return tensors
            return None

    async def get_tensor_shards(
        self, op_name: str, tuuid: int, step: int
    ) -> Dict[Tuple[str, int], List["TransportTensor"]]:
        async with self._lock:
            staleness = self._staleness_config.get(op_name, {}).get(tuuid, 0)
            if staleness > 0:
                expected = self._expected_tensor_counts.get(op_name, {}).get(tuuid, 0)
                if expected > 0:
                    policy = self._retrieval_policy.get(op_name, {}).get(
                        tuuid, "latest"
                    )
                    step_dict = self._tensor_slots.get(op_name, {}).get(tuuid, {})
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

            return dict(
                self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step, {})
            )

    async def get_all_tensor_shards_with_staleness(
        self, op_name: str, step: int
    ) -> Dict[int, Dict[Tuple[str, int], List[TransportTensor]]]:
        """Prioritized version of get_all_tensor_shards including staleness check."""
        async with self._lock:
            result = {}
            for tuuid, step_dict in self._tensor_slots.get(op_name, {}).items():
                staleness = self._staleness_config.get(op_name, {}).get(tuuid, 0)
                expected = self._expected_tensor_counts.get(op_name, {}).get(tuuid, 0)
                if staleness > 0 and expected > 0:
                    policy = self._retrieval_policy.get(op_name, {}).get(
                        tuuid, "latest"
                    )
                    valid_steps = [
                        k
                        for k in range(max(0, step - staleness), step + 1)
                        if len(step_dict.get(k, {})) >= expected
                    ]
                    if valid_steps:
                        best_k = (
                            valid_steps[0] if policy == "oldest" else valid_steps[-1]
                        )
                        result[tuuid] = dict(step_dict[best_k])
                        continue

                if step in step_dict:
                    result[tuuid] = dict(step_dict[step])
            return result

    async def get_tensor_shard_count(self, op_name: str, tuuid: int, step: int) -> int:
        async with self._lock:
            return len(self._tensor_slots.get(op_name, {}).get(tuuid, {}).get(step, {}))

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
        tuuid: int,
        step: int,
        target_buffers: List[Any],
        expect_offsets: dict,
    ) -> None:
        """Enable zero-copy write-through for a tensor slot."""
        async with self._lock:
            receiver = TransportReceiver(wait_for_n_slices=0)
            receiver.register_zero_copy_buffers(target_buffers, expect_offsets)
            self._zc_receivers.setdefault(op_name, {}).setdefault(tuuid, {})[step] = (
                receiver
            )

    async def get_zero_copy_buffers(
        self, op_name: str, tuuid: int, step: int
    ) -> Optional[List[Any]]:
        async with self._lock:
            receiver = self._zc_receivers.get(op_name, {}).get(tuuid, {}).get(step)
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

            if op_name in self._control_arg_data:
                for name, step_dict in list(self._control_arg_data[op_name].items()):
                    staleness = self._staleness_config.get(op_name, {}).get(name, 0)
                    clear_target = step - staleness
                    step_dict.pop(clear_target, None)
                    if not step_dict:
                        del self._control_arg_data[op_name][name]

            if op_name in self._barriers:
                self._barriers[op_name].pop(step, None)

    async def clear_all(self) -> None:
        async with self._lock:
            self._control_arg_data.clear()
            self._tensor_slots.clear()
            self._zc_receivers.clear()
            self._barriers.clear()

    # ------------------------------------------------------------------
    # Backpressure monitoring
    # ------------------------------------------------------------------

    async def get_size(self) -> int:
        async with self._lock:
            return self._get_size_lock_free()

    def _get_size_lock_free(self) -> int:
        """Internal helper to compute size without acquiring lock."""
        size = 0
        for op_name_dict in self._control_arg_data.values():
            for step_dict in op_name_dict.values():
                size += len(step_dict)
        for op_name_dict in self._tensor_slots.values():
            for step_dict in op_name_dict.values():
                for rank_dict in step_dict.values():
                    size += len(rank_dict)
        return size

    async def get_capacity(self) -> int:
        return 100
