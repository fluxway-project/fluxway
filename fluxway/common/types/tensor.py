import logging
from enum import Enum, auto
from typing import Any, Optional, Protocol

import numpy as np

from fluxway.common.allocation import (
    AllocationDescriptor,
    AllocationRelation,
)
from fluxway.common.coherence import (
    CoherenceStateTransitionError,
    TensorAccessIntent,
    CoherenceInstance,
    PhysicalMutability,
)

from fluxway.common.tensor_backend import force_writable, get_backend
from fluxway.common.index_descriptor import (
    ContinuousIndex,
    IndexDescriptor,
    ExplicitIndex,
)
from fluxway.common.tensor_components import (
    TensorLayout,
    TensorLineage,
    TensorPayload,
    TensorRouting,
    normalize_index_descriptor,
    routing_entity_count,
    validate_dense_global_shape,
    validate_metadata_lengths,
)

logger = logging.getLogger(__name__)


class TensorRuntimeState(Enum):
    VALID = auto()
    INVALIDATED = auto()
    RELEASED = auto()


class _ConsumerLeaseOwner(Protocol):
    def acquire_consumer_lease(
        self,
        instances: tuple[CoherenceInstance, ...],
        ticket_id: int | None = None,
    ) -> Any: ...

    def reserve_consumer_lease(
        self,
        instances: tuple[CoherenceInstance, ...],
        ticket_id: int | None = None,
    ) -> Any: ...


class _ConsumerLeaseReservationRef(Protocol):
    def claim(self) -> Any: ...

    def release_for_cleanup(self) -> None: ...


class TensorConsumerLease:
    __slots__ = ("_lease", "_tensor", "_released")

    def __init__(self, tensor: "TransportTensor", lease: Any) -> None:
        self._tensor = tensor
        self._lease = lease
        self._released = False

    def release(self) -> None:
        if self._released:
            raise CoherenceStateTransitionError(
                "tensor consumer lease was already released."
            )
        self._released = True
        self._lease.release()
        self._tensor._release_consumer_lease()

    def cancel(self) -> None:
        self.release()


class _ReleasedPayload:
    __slots__ = ()


_RELEASED_PAYLOAD = _ReleasedPayload()


class _LocalConsumerLease:
    __slots__ = ()

    def release(self) -> None:
        return None

    def cancel(self) -> None:
        self.release()


def runtime_allocation_descriptor(value: Any) -> AllocationDescriptor:
    if isinstance(value, TransportTensor):
        return value._runtime_allocation_descriptor()
    return get_backend().allocation_descriptor(value)


def runtime_allocation_relation(left: Any, right: Any) -> AllocationRelation:
    left_value = left._payload.values if isinstance(left, TransportTensor) else left
    right_value = right._payload.values if isinstance(right, TransportTensor) else right
    if left_value is right_value:
        return AllocationRelation.EXACT
    return runtime_allocation_descriptor(left).identity.relation(
        runtime_allocation_descriptor(right).identity
    )


# ============================================================
# TransportTensor — Unified tensor wrapper
# ============================================================


class SymmetryReceipt:
    """Stores the origin of each row in a TransportTensor.

    This acts as a 'Mirror Sharding' receipt for symmetric routing in the backward pass,
    allowing the framework to route data back to its source without re-calculating hashes.
    """

    def __init__(self, world_size: int, sender_rank_tags: Any):
        """
        Args:
            world_size: Total number of ranks in the source mesh.
            sender_rank_tags: Tensor or int containing target worker ranks.
        """
        if type(world_size) is not int or world_size <= 0:
            raise ValueError(
                f"SymmetryReceipt.world_size must be a positive int, got {world_size!r}."
            )
        if isinstance(sender_rank_tags, int | np.integer) and not (
            0 <= int(sender_rank_tags) < world_size
        ):
            raise ValueError(
                "SymmetryReceipt scalar rank is outside its rank domain: "
                f"rank={int(sender_rank_tags)} world_size={world_size}."
            )
        self._world_size = world_size
        self._sender_rank_tags = sender_rank_tags

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def sender_rank_tags(self) -> Any:
        return self._sender_rank_tags

    def is_scalar(self) -> bool:
        if isinstance(self._sender_rank_tags, int | np.integer):
            return True
        return getattr(self._sender_rank_tags, "shape", None) == ()

    def validate_entity_count(self, count: int) -> None:
        if self.is_scalar():
            return
        shape = get_backend().shape(self._sender_rank_tags)
        if len(shape) != 1:
            raise ValueError(
                f"SymmetryReceipt rank tags must be 1D, got shape={shape}."
            )
        if shape[0] != count:
            raise ValueError(
                f"SymmetryReceipt receipt length {shape[0]} does not match "
                f"tensor routing entity count {count}."
            )

    def __getitem__(self, key: Any) -> "SymmetryReceipt":
        """Slice or gather symmetry metadata."""
        if self.is_scalar():
            return self
        return SymmetryReceipt(self._world_size, self._sender_rank_tags[key])

    def materialize(self, length: int) -> Any:
        """Returns a full-sized tensor of rank tags."""
        if self.is_scalar():
            backend = get_backend()
            return backend.full(
                (length,), int(self._sender_rank_tags), dtype=backend.int32()
            )
        self.validate_entity_count(length)
        return self._sender_rank_tags

    @classmethod
    def concatenate(
        cls, receipts: list[Optional["SymmetryReceipt"]], lengths: list[int]
    ) -> Optional["SymmetryReceipt"]:
        """Concatenate multiple receipts into one."""
        if not receipts:
            return None
        if len(receipts) != len(lengths):
            raise ValueError(
                f"Receipt count {len(receipts)} does not match length count {len(lengths)}."
            )

        # Filter out None values and check for consistency
        valid_receipts = [r for r in receipts if r is not None]
        if not valid_receipts:
            return None

        if len(valid_receipts) != len(receipts):
            raise ValueError(
                f"Inconsistent provenance: Expected {len(receipts)} receipts, "
                f"but found {len(valid_receipts)}. All shards must carry receipts."
            )
        rank_domains = {receipt.world_size for receipt in valid_receipts}
        if len(rank_domains) != 1:
            raise ValueError(
                f"Cannot concatenate receipts from different rank domains: "
                f"{sorted(rank_domains)}."
            )
        if len(valid_receipts) == 1:
            valid_receipts[0].validate_entity_count(lengths[0])
            return valid_receipts[0]

        world_size = valid_receipts[0].world_size

        if all(receipt.is_scalar() for receipt in valid_receipts):
            first_rank = valid_receipts[0].sender_rank_tags
            if all(
                receipt.sender_rank_tags == first_rank for receipt in valid_receipts
            ):
                return cls(world_size, first_rank)

        backend = get_backend()
        tags = []
        for receipt, length in zip(valid_receipts, lengths, strict=True):  # type: ignore
            tags.append(receipt.materialize(length))

        merged_tags = backend.cat(tags, dim=0)
        return cls(world_size, merged_tags)


class Scalar:
    """Lightweight wrapper for 0-dim scalars to provide a compatible __len__ for transport."""

    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return 1


class TransportTensor:
    """Unified tensor wrapper for distributed data transport operations.

    This class abstracts the representation of both dense and ragged tensor layouts.
    Dense tensors are represented via multi-dimensional `flat_values` and empty splits,
    while ragged tensors use 1D `flat_values` accompanied by a `nested_splits` structure.
    """

    def __init__(
        self,
        flat_values: Any,
        nested_splits: list[Any] | None = None,
        origin_indices: Any | None = None,
        routing_indices: Any | None = None,
        global_shape: tuple[int, ...] | None = None,
        receipt: SymmetryReceipt | None = None,
        routing_axes: tuple[int, ...] = (0,),
        source_step: int | None = None,
    ):
        self._routing_axes: tuple[int, ...] = tuple(a for a in routing_axes)
        if len(self._routing_axes) != len(set(self._routing_axes)):
            raise ValueError(f"routing_axes must be unique, got {self._routing_axes}")
        if self._routing_axes != (0,):
            logger.warning(
                "TransportTensor.routing_axes=%s (non-default). "
                "Apply path requires non-leading binding support.",
                self._routing_axes,
            )

        self._payload = TensorPayload.create(flat_values)
        self._layout = TensorLayout.create(
            nested_splits, global_shape, self._routing_axes
        )
        self._flat_values = self._payload.values
        self._nested_splits = list(self._layout.nested_splits)
        self._global_shape = self._layout.global_shape
        backend = get_backend()
        validate_dense_global_shape(self._payload, self._layout, backend)

        data_len = routing_entity_count(self._payload, self._layout, backend)
        origin_routing = TensorRouting.create(
            normalize_index_descriptor(origin_indices, data_len)
        )
        self._lineage = TensorLineage(origin=origin_routing, receipt=receipt)

        self._routing = TensorRouting.create(
            normalize_index_descriptor(routing_indices, data_len)
        )
        self._routing_indices_materialize_cache = None
        self._origin_indices_materialize_cache = None
        self._routing_indices = self._routing.indices
        self._origin_indices = self._lineage.origin.indices
        self._receipt = self._lineage.receipt
        self._source_step = None
        self._runtime_access_intent = TensorAccessIntent.WRITABLE
        self._runtime_state = TensorRuntimeState.VALID
        self._consumer_lease_owner: _ConsumerLeaseOwner | None = None
        self._consumer_lease_reservation_ref: _ConsumerLeaseReservationRef | None = None
        self._active_consumer_leases = 0
        self._release_requested = False
        self._coherence_instances: tuple[CoherenceInstance, ...] = ()
        self._coherence_ticket_id: int | None = None
        self._allocation_descriptor: AllocationDescriptor | None = None
        self.source_step = source_step
        validate_metadata_lengths(
            self._payload,
            self._layout,
            self._routing,
            self._lineage,
            backend,
        )
        if receipt is not None:
            receipt.validate_entity_count(data_len)

    @classmethod
    def from_tensor(
        cls,
        tensor: Any,
        global_shape: tuple[int, ...] | None = None,
        origin_indices: Any | None = None,
        routing_indices: Any | None = None,
        receipt: SymmetryReceipt | None = None,
        routing_axes: tuple[int, ...] = (0,),
        logical_world_size: int | None = None,
        is_uniform_dense_partition: bool = False,
    ) -> "TransportTensor":
        """Creates a dense TransportTensor directly from a multi-dimensional tensor.

        Args:
            tensor: The dense data tensor.
            origin_indices: Global indices mapping. Can be a materialized tensor, an explicit
                     ContinuousIndex, or a python `slice` object for O(1) regular patterns.
            routing_indices: Global indices mapping. Can be a materialized tensor, an explicit
                     ContinuousIndex, or a python `slice` object for O(1) regular patterns.
            global_shape: Optional global shape of the full tensor.
            receipt: Optional symmetry receipt for provenance tracking.
            routing_axes: Which axes of `tensor` are bound to the routing
                domain. Default (0,) is correct for nearly all DP/MP/EP/SP/CP/TP
                layouts. The planner stays 1-D regardless; routing_axes only
                affects the apply (gather) step in the L1 distributor.
            is_uniform_dense_partition: If True, auto-infers `global_shape` by multiplying
                the leading dimension with `logical_world_size`. Must be False for ragged
                or filtered data.
        """
        # NOTE: We intentionally do NOT auto-fill origin_indices here.
        # `origin_indices` is per-origin-rank state ("which global rows do I hold"),
        # and the framework has no rank/partition context at construction time, so
        # any auto-synthesized value would be a wrong guess (e.g. all ranks claiming
        # start=0). The caller MUST set origin_indices explicitly when it is needed,
        # otherwise leave it as None and let the receipt/transport layer take over.
        backend = get_backend()
        if (
            global_shape is None
            and logical_world_size is not None
            and is_uniform_dense_partition
        ):
            p_shape = backend.shape(tensor)
            if len(p_shape) > 0:
                global_shape = (p_shape[0] * logical_world_size,) + tuple(p_shape[1:])

        return cls(
            flat_values=tensor,
            nested_splits=[],
            origin_indices=origin_indices,
            routing_indices=routing_indices,
            global_shape=global_shape,
            receipt=receipt,
            routing_axes=routing_axes,
        )

    @classmethod
    def from_lists(
        cls,
        data: list[Any],
        dtype: Any | None = None,
        global_shape: tuple[int, ...] | None = None,
        origin_indices: Any | None = None,
        routing_indices: Any | None = None,
        receipt: SymmetryReceipt | None = None,
        routing_axes: tuple[int, ...] = (0,),
        logical_world_size: int | None = None,
        is_uniform_dense_partition: bool = False,
    ) -> "TransportTensor":
        """Creates a TransportTensor from a nested list.

        This constructor automatically inspects the input structure. If the data is
        uniformly shaped, it optimizes the representation by creating a dense tensor.
        Otherwise, it constructs a ragged tensor with computed splits.
        """
        backend = get_backend()

        # Helper function to check if data has uniform shape
        def _is_uniform_shape(lst):
            if not isinstance(lst, list) or len(lst) == 0:
                return True, []

            if not isinstance(lst[0], list):
                return all(not isinstance(x, list) for x in lst), [len(lst)]

            first_len = len(lst[0])
            if not all(
                isinstance(item, list) and len(item) == first_len for item in lst
            ):
                return False, []

            is_uniform, child_shape = _is_uniform_shape(lst[0])
            if not is_uniform:
                return False, []

            return True, [len(lst)] + child_shape

        # Check if data has uniform shape
        is_uniform, shape = _is_uniform_shape(data)

        if is_uniform and len(shape) > 0:
            # Dense optimization: create multidimensional tensor directly
            if dtype is None:
                dtype = backend.float32()
            dense_tensor = backend.tensor(data, dtype=dtype)

            if (
                global_shape is None
                and logical_world_size is not None
                and is_uniform_dense_partition
            ):
                p_shape = backend.shape(dense_tensor)
                global_shape = (p_shape[0] * logical_world_size,) + tuple(p_shape[1:])

            return cls(
                dense_tensor,
                [],
                global_shape=global_shape,
                origin_indices=origin_indices,
                routing_indices=routing_indices,
                receipt=receipt,
                routing_axes=routing_axes,
            )

        # Ragged tensor: use iterative flattening
        current_level_lists = data
        all_splits = []

        while True:
            # Check if next level is lists
            if not current_level_lists or not isinstance(current_level_lists[0], list):
                break

            if len(current_level_lists) > 0 and not isinstance(
                current_level_lists[0], list
            ):
                break

            # Flatten one level
            next_level_lists = []
            current_splits = [0]
            offset = 0

            for item in current_level_lists:
                if not isinstance(item, list):
                    next_level_lists.append(item)
                    offset += 1
                else:
                    offset += len(item)
                    next_level_lists.extend(item)
                current_splits.append(offset)

            splits_tensor = backend.tensor(current_splits, dtype=backend.int32())
            all_splits.append(splits_tensor)

            current_level_lists = next_level_lists

            if len(current_level_lists) > 0 and not isinstance(
                current_level_lists[0], list
            ):
                break

        if dtype is None:
            dtype = backend.float32()

        flat_tensor = backend.tensor(current_level_lists, dtype=dtype)

        return cls(
            flat_tensor,
            all_splits,
            global_shape=global_shape,
            origin_indices=origin_indices,
            routing_indices=routing_indices,
            receipt=receipt,
            routing_axes=routing_axes,
        )

    @classmethod
    def from_row_partitioned(
        cls,
        rows: list[list[Any]],
        *,
        width: int,
        dtype: Any | None = None,
        global_shape: tuple[int, int] | None = None,
        origin_indices: Any | None = None,
        routing_axes: tuple[int, ...] = (0,),
    ) -> "TransportTensor":
        if width <= 0:
            raise ValueError("row-partitioned width must be positive")
        if not all(isinstance(row, list) for row in rows):
            raise TypeError("row-partitioned rows must be lists")
        if global_shape is None:
            raise ValueError("row-partitioned tensors require explicit global_shape")
        if tuple(global_shape)[1] != width:
            raise ValueError("global_shape width does not match row-partitioned width")
        backend = get_backend()
        flat = [value for row in rows for value in row]
        offsets = [0]
        for row in rows:
            offsets.append(offsets[-1] + len(row))
        if dtype is None:
            dtype = backend.float32()
        flat_tensor = backend.tensor(flat, dtype=dtype)
        splits = backend.tensor(offsets, dtype=backend.int32())
        return cls(
            flat_tensor,
            [splits],
            global_shape=global_shape,
            origin_indices=origin_indices,
            routing_axes=routing_axes,
        )

    def to_lists(self) -> list[Any]:
        """Converts back to a nested list."""
        backend = get_backend()

        if not self._nested_splits:
            return backend.to_numpy(self._flat_values).tolist()

        def build_list(level, start_idx, end_idx):
            if level == len(self._nested_splits):
                chunk = self._flat_values[start_idx:end_idx]
                return backend.to_numpy(chunk).tolist()

            splits = self._nested_splits[level]

            res = []
            for i in range(start_idx, end_idx):
                start = backend.item(splits[i])
                end = backend.item(splits[i + 1])

                child_list = build_list(level + 1, start, end)
                res.append(child_list)
            return res

        top_split = self._nested_splits[0]
        # Ensure we handle case where split might be empty or 1-element (empty tensor)
        if backend.shape(top_split)[0] == 0:
            return []
        num_splits = backend.shape(top_split)[0]

        return build_list(0, 0, num_splits - 1)

    @property
    def flat_values(self) -> Any:
        if self._runtime_state is not TensorRuntimeState.VALID:
            raise CoherenceStateTransitionError(
                f"tensor access is {self._runtime_state.name.lower()}."
            )
        values = self._payload.values
        if self._runtime_access_intent is TensorAccessIntent.READONLY:
            return get_backend().readonly_view(values)
        return values

    @property
    def is_readonly(self) -> bool:
        return self._runtime_access_intent is TensorAccessIntent.READONLY

    @property
    def physical_mutability(self) -> PhysicalMutability:
        return self._payload.physical_mutability

    def dlpack_values_for_consumer(self) -> Any:
        if self._runtime_state is not TensorRuntimeState.VALID:
            raise CoherenceStateTransitionError(
                f"tensor access is {self._runtime_state.name.lower()}."
            )
        if not self.is_readonly:
            raise CoherenceStateTransitionError(
                "DLPack consumer export requires a readonly coherence grant."
            )
        if self.physical_mutability is not PhysicalMutability.PROVEN_WRITABLE:
            raise CoherenceStateTransitionError(
                "DLPack consumer export requires a physically writable payload."
            )
        return force_writable(self._payload.values)

    @property
    def coherence_ticket_id(self) -> int | None:
        return self._coherence_ticket_id

    def _bind_coherence_ticket_id(self, ticket_id: int | None) -> None:
        if ticket_id is None:
            return
        if (
            self._coherence_ticket_id is not None
            and self._coherence_ticket_id != ticket_id
        ):
            raise CoherenceStateTransitionError(
                f"conflicting coherence ticket id: expected={self._coherence_ticket_id} actual={ticket_id}"
            )
        self._coherence_ticket_id = ticket_id

    def acquire_consumer_lease(
        self, ticket_id: int | None = None
    ) -> TensorConsumerLease:
        if self._runtime_state is not TensorRuntimeState.VALID:
            raise CoherenceStateTransitionError(
                f"cannot acquire a consumer lease while tensor access is "
                f"{self._runtime_state.name.lower()}."
            )
        owner = self._consumer_lease_owner
        if self._coherence_instances and owner is None:
            raise CoherenceStateTransitionError(
                "coherence-bound tensor has no consumer lease owner."
            )
        reservation_ref = self._consumer_lease_reservation_ref
        if reservation_ref is not None:
            lease = reservation_ref.claim()
            self._consumer_lease_reservation_ref = None
        else:
            target_ticket_id = (
                ticket_id if ticket_id is not None else self._coherence_ticket_id
            )
            lease = (
                owner.acquire_consumer_lease(
                    self._coherence_instances, ticket_id=target_ticket_id
                )
                if owner is not None
                else _LocalConsumerLease()
            )
        self._active_consumer_leases += 1
        return TensorConsumerLease(self, lease)

    def _reserve_consumer_lease(self, ticket_id: int | None = None) -> None:
        if self._runtime_state is not TensorRuntimeState.VALID:
            raise CoherenceStateTransitionError(
                "cannot reserve a consumer lease for invalid data."
            )
        owner = self._consumer_lease_owner
        if owner is None or not self._coherence_instances:
            raise CoherenceStateTransitionError(
                "coherence-bound tensor has no consumer lease owner."
            )
        if self._consumer_lease_reservation_ref is not None:
            raise CoherenceStateTransitionError(
                "tensor already has a consumer lease reservation."
            )
        target_ticket_id = (
            ticket_id if ticket_id is not None else self._coherence_ticket_id
        )
        self._consumer_lease_reservation_ref = owner.reserve_consumer_lease(
            self._coherence_instances,
            ticket_id=target_ticket_id,
        )

    def _cleanup_consumer_lease_reservation_ref(self) -> None:
        reservation_ref = self._consumer_lease_reservation_ref
        if reservation_ref is None:
            return
        self._consumer_lease_reservation_ref = None
        reservation_ref.release_for_cleanup()

    def _release_consumer_lease(self) -> None:
        if self._active_consumer_leases <= 0:
            raise CoherenceStateTransitionError(
                "tensor consumer lease count underflow."
            )
        self._active_consumer_leases -= 1
        if self._active_consumer_leases == 0 and self._release_requested:
            self._release_runtime_payload()

    def _bind_consumer_lease_owner(self, owner: _ConsumerLeaseOwner) -> None:
        if (
            self._consumer_lease_owner is not None
            and self._consumer_lease_owner is not owner
        ):
            raise CoherenceStateTransitionError(
                "tensor consumer lease owner is already bound."
            )
        self._consumer_lease_owner = owner

    def _set_runtime_access_intent(self, intent: TensorAccessIntent) -> None:
        if not isinstance(intent, TensorAccessIntent):
            raise TypeError(
                f"intent must be TensorAccessIntent, got {type(intent).__name__}."
            )
        if self._runtime_state is not TensorRuntimeState.VALID:
            raise CoherenceStateTransitionError("cannot grant access to invalid data.")
        self._runtime_access_intent = intent

    def _invalidate_runtime_access(self) -> None:
        if self._runtime_state is TensorRuntimeState.RELEASED:
            raise CoherenceStateTransitionError("tensor payload was already released.")
        self._runtime_state = TensorRuntimeState.INVALIDATED
        self._cleanup_consumer_lease_reservation_ref()
        if self._active_consumer_leases:
            self._release_requested = True

    def _release_runtime_payload(self) -> None:
        if self._runtime_state is TensorRuntimeState.VALID:
            raise CoherenceStateTransitionError(
                "tensor payload cannot be released while runtime access is valid."
            )
        if self._runtime_state is TensorRuntimeState.RELEASED:
            raise CoherenceStateTransitionError("tensor payload was already released.")
        if self._active_consumer_leases:
            self._release_requested = True
            return
        self._payload = TensorPayload(_RELEASED_PAYLOAD, PhysicalMutability.UNKNOWN)
        self._layout = TensorLayout((), None, self._routing_axes)
        self._flat_values = _RELEASED_PAYLOAD
        self._nested_splits.clear()
        self._routing = TensorRouting(None)
        self._lineage = TensorLineage(origin=TensorRouting(None), receipt=None)
        self._routing_indices = None
        self._origin_indices = None
        self._receipt = None
        self._coherence_instances = ()
        self._consumer_lease_reservation_ref = None
        self._routing_indices_materialize_cache = None
        self._origin_indices_materialize_cache = None
        self._allocation_descriptor = None
        self._runtime_state = TensorRuntimeState.RELEASED

    def _bind_coherence_instances(
        self, instances: tuple[CoherenceInstance, ...]
    ) -> None:
        if self._coherence_instances:
            raise CoherenceStateTransitionError(
                "tensor coherence instances are already bound."
            )
        if not instances or len(instances) != len(set(instances)):
            raise CoherenceStateTransitionError(
                "tensor coherence instances must be non-empty and unique."
            )
        self._coherence_instances = instances

    def _extend_coherence_instances(
        self,
        instances: tuple[CoherenceInstance, ...],
    ) -> None:
        if not instances:
            raise CoherenceStateTransitionError(
                "tensor coherence instances to extend must be non-empty."
            )
        if len(set(instances)) != len(instances):
            raise CoherenceStateTransitionError(
                "tensor coherence instances to extend must be unique."
            )
        merged_values = list(self._coherence_instances)
        seen = set(self._coherence_instances)
        for instance in instances:
            if instance not in seen:
                seen.add(instance)
                merged_values.append(instance)
        merged = tuple(merged_values)
        self._coherence_instances = merged

    def _handoff_coherence_instance(
        self,
        old: CoherenceInstance,
        new: CoherenceInstance,
    ) -> None:
        if self._coherence_instances != (old,):
            raise CoherenceStateTransitionError(
                "allocation handoff requires one exact coherence instance."
            )
        self._coherence_instances = (new,)

    def _runtime_allocation_descriptor(self) -> AllocationDescriptor:
        descriptor = self._allocation_descriptor
        if descriptor is None:
            descriptor = get_backend().allocation_descriptor(self._payload.values)
            self._allocation_descriptor = descriptor
        return descriptor

    @property
    def payload(self) -> TensorPayload:
        if self._runtime_state is not TensorRuntimeState.VALID:
            raise CoherenceStateTransitionError(
                f"tensor payload is {self._runtime_state.name.lower()}."
            )
        return self._payload

    @property
    def layout(self) -> TensorLayout:
        return self._layout

    @property
    def routing(self) -> TensorRouting:
        return self._routing

    @property
    def lineage(self) -> TensorLineage:
        return self._lineage

    @property
    def nested_splits(self) -> list[Any]:
        return list(self._layout.nested_splits)

    @property
    def is_dense(self) -> bool:
        """Returns True if this tensor is dense (no ragged dimensions)."""
        return self._layout.is_dense

    @property
    def is_ragged(self) -> bool:
        """Returns True if this tensor has ragged dimensions."""
        return self._layout.is_ragged

    @property
    def origin_indices(self) -> IndexDescriptor | None:
        """The underlying IndexDescriptor (metadata) for origin tracking."""
        return self._lineage.origin.indices

    @property
    def origin_indices_materialized(self) -> Any | None:
        """Global indices marking the original sequence/batch row.

        Lazy-materializes from descriptor if no explicit tensor is set.
        """
        if self._origin_indices is None:
            return None
        return self._lineage.origin.materialize()

    @property
    def routing_indices(self) -> IndexDescriptor | None:
        """The underlying IndexDescriptor (metadata) for spatial positioning."""
        return self._routing.indices

    @property
    def routing_indices_materialized(self) -> Any | None:
        """Routing indices to remote workers this local data corresponds to.

        Lazy-materializes from descriptor if no explicit tensor is set.
        Returns None if neither is available (implicit/zero-cost).
        """
        if self._routing_indices is None:
            return None
        return self._routing.materialize()

    @property
    def global_shape(self) -> tuple[int, ...] | None:
        """Global shape of the full tensor across all ranks."""
        return self._layout.global_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Local shape of the data shard (physical tensor shape with padding)."""
        backend = get_backend()
        return backend.shape(self._flat_values)

    @property
    def logical_shape(self) -> tuple[int | None, ...]:
        """Returns the logical shape, honoring ragged boundaries.
        Variable-length (ragged) dimensions are represented as `None`.
        """
        backend = get_backend()
        p_shape = backend.shape(self._flat_values)
        if self._nested_splits and len(self._nested_splits) > 0:
            top_split = self._nested_splits[0]
            if backend.shape(top_split)[0] > 0:
                bs = int(backend.shape(top_split)[0]) - 1
            else:
                bs = 0
            return (bs, None) + p_shape[1:]
        return p_shape

    @property
    def dtype(self) -> Any:
        """Data type of the tensor."""
        backend = get_backend()
        return backend.dtype(self._flat_values)

    @property
    def topology(self) -> list[Any] | None:
        """IndexDescriptor metadata derived from nested_splits. None for dense."""
        return self._nested_splits if self._nested_splits else None

    @property
    def receipt(self) -> SymmetryReceipt | None:
        return self._lineage.receipt

    @receipt.setter
    def receipt(self, value: SymmetryReceipt | None):
        if value is not None:
            value.validate_entity_count(self.routing_entity_count)
        self._lineage.receipt = value
        self._receipt = value

    def with_lineage(
        self,
        *,
        origin_indices: Any | None,
        receipt: SymmetryReceipt,
        global_shape: tuple[int, ...] | None,
    ) -> "TransportTensor":
        return TransportTensor(
            self.flat_values,
            nested_splits=self.nested_splits,
            routing_indices=self.routing_indices,
            global_shape=global_shape,
            origin_indices=origin_indices,
            receipt=receipt,
            routing_axes=self.routing_axes,
            source_step=self.source_step,
        )

    def zero_copy_alias(self) -> "TransportTensor":
        return self._zero_copy_alias(TensorAccessIntent.WRITABLE)

    def zero_copy_readonly_alias(self) -> "TransportTensor":
        return self._zero_copy_alias(TensorAccessIntent.READONLY)

    def _zero_copy_alias(self, intent: TensorAccessIntent) -> "TransportTensor":
        alias = TransportTensor(
            self._payload.values,
            nested_splits=self.nested_splits,
            origin_indices=self.origin_indices,
            routing_indices=self.routing_indices,
            global_shape=self.global_shape,
            receipt=self.receipt,
            routing_axes=self.routing_axes,
            source_step=self.source_step,
        )
        if alias._payload.values is not self._payload.values and (
            runtime_allocation_relation(alias, self) is not AllocationRelation.EXACT
        ):
            raise CoherenceStateTransitionError(
                "zero-copy alias construction changed the backing allocation."
            )
        alias._set_runtime_access_intent(intent)
        alias._coherence_ticket_id = self._coherence_ticket_id
        return alias

    @property
    def source_step(self) -> int | None:
        return self._source_step

    @source_step.setter
    def source_step(self, value: int | None):
        if value is not None and (type(value) is not int):
            raise TypeError(
                f"TransportTensor.source_step must be an int or None, got {type(value).__name__}."
            )
        if self._source_step is not None and value != self._source_step:
            raise ValueError(
                "TransportTensor.source_step is immutable once assigned; "
                f"current={self._source_step}, attempted={value}."
            )
        self._source_step = value

    @property
    def routing_axes(self) -> tuple[int, ...]:
        """Axes of `flat_values` bound to the routing domain.

        The planner (L2) is structurally 1-D and ignores this; it is read
        only by the L1 distributor at apply time to decide between the
        fast `take(axis=0)` path and the moveaxis+reshape fallback.
        """
        return self._routing_axes

    @property
    def routing_entity_count(self) -> int:
        backend = get_backend()
        return routing_entity_count(self._payload, self._layout, backend)

    @property
    def meta_state(self) -> int:
        """
        Computed bitmask/int representing the state of the tensor metadata.
        Used for efficient reconstruction in the mailbox.
        0: Dense
        1: Ragged
        2: Has Origin Indices (Materialized in payload)
        4: Has Compact Index (in slice_compact_desc_params)
        (Combine via BITWISE OR)
        """
        state = 0
        if self.is_ragged:
            state |= 1
        if self.origin_indices is not None:
            if isinstance(self.origin_indices, ContinuousIndex):
                state |= 4
            else:
                state |= 2
        return state

    # -------------------------------------------------------------------------
    # Transparent Container / Duck-typing Overloads
    # -------------------------------------------------------------------------

    def __getitem__(self, key: Any) -> "TransportTensor":
        """Support for slicing and gathering via []."""
        backend = get_backend()
        if isinstance(key, slice):
            return self.slice(key.start or 0, key.stop, unsafe=False)

        # If key is a tensor or TransportTensor, it's a gather
        if isinstance(key, TransportTensor) or backend.is_tensor(key):
            return self.gather(key)

        # Fallback to backend indexing for simple scalars if no splits
        if not self._nested_splits:
            return TransportTensor(
                self._flat_values[key],
                global_shape=self._global_shape,
                receipt=self._receipt[key] if self._receipt else None,
                routing_axes=self._routing_axes,
            )

        raise NotImplementedError(
            f"Indexing with {type(key)} not supported for ragged tensors"
        )

    def __len__(self) -> int:
        backend = get_backend()
        if backend.is_scalar(self._flat_values):
            return 1
        return int(backend.shape(self._flat_values)[0])

    def __iter__(self):
        # Very high-cost, but sometimes needed for duck-typing
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other: Any) -> "TransportTensor":
        return self._binary_op(other, "__add__")

    def __sub__(self, other: Any) -> "TransportTensor":
        return self._binary_op(other, "__sub__")

    def __mul__(self, other: Any) -> "TransportTensor":
        return self._binary_op(other, "__mul__")

    def __truediv__(self, other: Any) -> "TransportTensor":
        return self._binary_op(other, "__truediv__")

    def _binary_op(self, other: Any, op_name: str) -> "TransportTensor":
        other_val = other.flat_values if isinstance(other, TransportTensor) else other
        op = getattr(self._flat_values, op_name)
        res = op(other_val)
        # Inherit metadata from self
        return TransportTensor(
            res,
            nested_splits=self.nested_splits,
            routing_indices=self.routing_indices,
            global_shape=self._global_shape,
            origin_indices=self.origin_indices,
            receipt=self.receipt,
            routing_axes=self._routing_axes,
        )

    def sum(self, dim: Any = None) -> Any:
        backend = get_backend()
        return backend.sum(self._flat_values, dim=dim)

    def unique_with_routing(self) -> tuple["TransportTensor", "TransportTensor"]:
        """
        Finds the unique elements and creates a compact TransportTensor.
        Returns (compacted_tensor, inverse_indices_tensor).
        """
        backend = get_backend()
        uniques, first_indices, inverse = backend.unique(
            self.flat_values, return_index=True, return_inverse=True
        )

        # Inherit origin_indices directly from the parent tensor if they exist
        if self.origin_indices is not None:
            origin_indices = backend.take(
                self.origin_indices_materialized, first_indices
            )
        else:
            origin_indices = first_indices

        # Reshape inverse back to match original flat_values geometry
        inv_flat = backend.reshape(inverse, backend.shape(self.flat_values))
        inverse_tt = TransportTensor(
            flat_values=inv_flat,
            nested_splits=self.nested_splits,
            origin_indices=self.origin_indices,
            routing_indices=self.routing_indices,
            global_shape=self.global_shape,
            routing_axes=self._routing_axes,
        )

        compacted = TransportTensor(
            flat_values=uniques,
            routing_indices=uniques,
            origin_indices=origin_indices,
            global_shape=self.global_shape,
            routing_axes=self._routing_axes,
        )

        return compacted, inverse_tt

    def _propagate_indices_for_dense_slice(self, start: Any, size: Any) -> Any:
        """Derive new index info for a dense slice. Returns new IndexDescriptor or None."""
        backend = get_backend()
        from fluxway.common.index_descriptor import ContinuousIndex, ExplicitIndex

        if self._routing_indices is None:
            return None

        # Prefer descriptor path (O(1))
        if isinstance(self._routing_indices, ContinuousIndex):
            if backend.is_tensor(start):
                s = int(backend.item(start))
            else:
                s = int(start)
            if backend.is_tensor(size):
                sz = int(backend.item(size))
            else:
                sz = int(size)
            return self._routing_indices.slice(s, s + sz)

        # Explicit tensor path
        if isinstance(self._routing_indices, ExplicitIndex):
            new_indices = backend.dynamic_slice(
                self._routing_indices.indices, [start], [size]
            )
            return ExplicitIndex(new_indices)

        return None

    def _propagate_origin_indices_for_dense_slice(self, start: Any, size: Any) -> Any:
        """Derive new origin index info for a dense slice. Returns new IndexDescriptor or None."""
        backend = get_backend()

        if self._origin_indices is None:
            return None

        if isinstance(self._origin_indices, ContinuousIndex):
            if backend.is_tensor(start):
                s = int(backend.item(start))
            else:
                s = int(start)
            if backend.is_tensor(size):
                sz = int(backend.item(size))
            else:
                sz = int(size)
            return self._origin_indices.slice(s, s + sz)

        if isinstance(self._origin_indices, ExplicitIndex):
            new_indices = backend.dynamic_slice(
                self._origin_indices.indices, [start], [size]
            )
            return ExplicitIndex(new_indices)

        return None

    def dynamic_slice(
        self, start: Any, size_tensor: Any, unsafe: bool = False
    ) -> "TransportTensor":
        """Slice with tensor arguments using zero-sync backend primitives."""
        backend = get_backend()

        if not self._nested_splits:
            # Dense case
            x = self._flat_values
            full_shape = backend.shape(x)
            ndim = len(full_shape)

            starts = [start] + [0] * (ndim - 1)
            sizes = [size_tensor] + list(full_shape[1:])

            sliced_data = backend.dynamic_slice(x, starts, sizes)

            new_indices = self._propagate_indices_for_dense_slice(start, size_tensor)
            new_origin_indices = self._propagate_origin_indices_for_dense_slice(
                start, size_tensor
            )
            return TransportTensor(
                sliced_data,
                [],
                routing_indices=new_indices,
                origin_indices=new_origin_indices,
                global_shape=self._global_shape,
                receipt=(
                    self._receipt[start : start + size_tensor]
                    if self._receipt
                    else None
                ),
                routing_axes=self._routing_axes,
            )

        else:
            # Ragged case: requires processing nested splits.
            # We can now use slice() directly with tensors, as it avoids synchronization.
            return self.slice(start, start + size_tensor, unsafe=unsafe)

    def slice(
        self,
        start: Any,
        end: Any,
        static_shape: Any | None = None,
        unsafe: bool = False,
    ) -> "TransportTensor":
        """Slices the tensor along the first dimension.

        Args:
            start: Start index.
            end: End index.
            static_shape: Optional static shape constraint for JIT/vmap.
                          Can be an int (dim0 length) or a tuple (dim0 length, Values length, ...).
                          If tuple, enables JIT safe slicing for ragged tensors by padding values.
            unsafe: If True, skips bounds checking and assumes indices are valid and within bounds.
                    Enable this only when indices are guaranteed correct (e.g. from planner).
        """
        backend = get_backend()

        # Check for integer/scalar optimization path
        is_start_int = isinstance(start, int) or backend.is_scalar(start)
        is_end_int = isinstance(end, int) or backend.is_scalar(end)

        # Parse static shape
        dim0_static = None
        val_static = None
        if static_shape is not None:
            if isinstance(static_shape, tuple | list):
                dim0_static = static_shape[0]
                if len(static_shape) > 1:
                    val_static = static_shape[
                        -1
                    ]  # Simple assumption: (dim0, Vals) or (dim0, L1, Vals) - take last as Vals
            else:
                dim0_static = static_shape

        # Input normalization for Tensor operations
        # If unsafe is True and inputs are ints, we retention them as ints for native slicing
        if not unsafe:
            if not backend.is_tensor(start):
                start = backend.tensor(start, dtype=backend.int32())
            if not backend.is_tensor(end):
                end = backend.tensor(end, dtype=backend.int32())

        if not self._nested_splits:
            # Dense Slice
            size = end - start
            return self.dynamic_slice(start, size, unsafe=unsafe)

        # Ragged implementation
        # Slicing along Level 0
        dim0_splits = self._nested_splits[0]

        # Optimization: Use native slicing for integers if unsafe or backend supports it
        # This is CRITICAL for Eager mode performance to avoid JIT/Graph tracing overhead.
        if unsafe and is_start_int and is_end_int:
            # Fast path for known valid integer indices
            s = int(start)
            e = int(end)

            # Slice dim0 splits directly
            # For ragged, we need splits[start : end + 1]
            spl_slice = dim0_splits[s : e + 1]

            # Normalize splits (subtract offset)
            offset = spl_slice[0]
            new_dim0_splits = spl_slice - offset

            new_nested_splits = [new_dim0_splits]

            # Determined range for next level
            # range_start = dim0_splits[s] -> which is offset (already fetched)
            range_start = offset
            range_end = dim0_splits[e]

            curr_start = range_start
            curr_end = range_end

            # Loop for nested levels
            for i in range(1, len(self._nested_splits)):
                curr_splits = self._nested_splits[i]

                # For next levels, use dynamic slice logic but try to retention it cheap
                count = curr_end - curr_start + 1
                new_spl = backend.dynamic_slice(curr_splits, curr_start, count)

                # Normalize
                off = backend.take(new_spl, backend.tensor(0, dtype=backend.int32()))
                new_spl = new_spl - off

                new_nested_splits.append(new_spl)

                # Update next range
                curr_start = backend.take(curr_splits, curr_start)
                curr_end = backend.take(curr_splits, curr_end)

            # Slice Values
            val_count = curr_end - curr_start
            val_slice = backend.dynamic_slice(self._flat_values, curr_start, val_count)

            # Propagate indices
            new_indices = None
            if self._routing_indices is not None:
                new_indices = self._routing_indices.slice(s, e)

            # Propagate origin metadata
            new_origin_indices = None
            if self._origin_indices is not None:
                new_origin_indices = self._origin_indices.slice(s, e)

            return TransportTensor(
                val_slice,
                new_nested_splits,
                routing_indices=new_indices,
                origin_indices=new_origin_indices,
                global_shape=self._global_shape,
                receipt=self._receipt[s:e] if self._receipt else None,
                routing_axes=self._routing_axes,
            )

        # -------------------------------------------------------------------------
        # Standard / JIT-Safe Path
        # -------------------------------------------------------------------------

        # Make sure inputs are tensors for safe path
        if not backend.is_tensor(start):
            start = backend.tensor(start, dtype=backend.int32())
        if not backend.is_tensor(end):
            end = backend.tensor(end, dtype=backend.int32())

        dim0_len = backend.shape(dim0_splits)[0] - 1
        dim0_len_t = backend.tensor(dim0_len, dtype=backend.int32())
        zero_t = backend.tensor(0, dtype=backend.int32())

        # Clamp start/end without CPU sync (only if not unsafe)
        # If unsafe=True, we rely on caller guarantee and skip clamping to avoid kernels.

        if not unsafe:
            start = backend.where(start < zero_t, zero_t, start)
            start = backend.where(start > dim0_len_t, dim0_len_t, start)

        if static_shape is None:
            if not unsafe:
                end = backend.where(end > dim0_len_t, dim0_len_t, end)
                end = backend.where(end < start, start, end)

            # Dynamic shape path (using arange)
            dim0_indices = backend.arange(start, end + 1, dtype=backend.int32())
            spl_slice = backend.take(dim0_splits, dim0_indices)
        else:
            # Static shape path (JIT compatible)
            # Mask values beyond end using static_shape
            count = dim0_static + 1 if dim0_static is not None else dim0_len + 1
            base = backend.arange(0, count, dtype=backend.int32())
            dim0_indices = base + start

            # Clamp indices to valid range for 'take' safety
            safe_indices = backend.where(
                dim0_indices > dim0_len_t, dim0_len_t, dim0_indices
            )
            safe_indices = backend.where(safe_indices < zero_t, zero_t, safe_indices)

            spl_slice = backend.take(dim0_splits, safe_indices)

        offset = backend.take(spl_slice, zero_t)
        new_dim0_splits = spl_slice - offset

        new_nested_splits = [new_dim0_splits]

        # Determine range for next level
        range_start = backend.take(dim0_splits, start)
        range_end = backend.take(dim0_splits, end)

        curr_start = range_start
        curr_end = range_end

        for i in range(1, len(self._nested_splits)):
            curr_splits = self._nested_splits[i]

            # Current level range: [curr_start : curr_end + 1]
            s = curr_start
            e = curr_end

            # Use gather instead of dynamic_slice to avoid CPU sync
            idx = backend.arange(s, e + 1, dtype=backend.int32())
            spl_slice = backend.take(curr_splits, idx)

            # Normalize
            off = backend.take(spl_slice, zero_t)
            new_spl = spl_slice - off
            new_nested_splits.append(new_spl)

            # Update range for next level
            curr_start = backend.take(curr_splits, s)
            curr_end = backend.take(curr_splits, e)

        # Slice Values
        if val_static is not None:
            # Static Value Slice
            count = val_static
            base_idx = backend.arange(0, count, dtype=backend.int32())
            val_idx = base_idx + curr_start

            total_val_len = backend.shape(self._flat_values)[0]
            tvl_t = backend.tensor(total_val_len, dtype=backend.int32())

            # Safe clamping strategy for JIT
            safe_idx = backend.where(val_idx >= curr_end, curr_end - 1, val_idx)
            safe_idx = backend.where(safe_idx >= tvl_t, tvl_t - 1, safe_idx)
            safe_idx = backend.where(safe_idx < zero_t, zero_t, safe_idx)

            val_slice = backend.take(self._flat_values, safe_idx)
        else:
            val_idx = backend.arange(curr_start, curr_end, dtype=backend.int32())
            val_slice = backend.take(self._flat_values, val_idx)

        # Propagate origin metadata for ragged slice
        new_origin_indices = None
        if self._origin_indices is not None:
            new_origin_indices = self._origin_indices.slice(start, end)

        # Propagate indices for ragged slice
        new_indices = None
        if self._routing_indices is not None:
            new_indices = self._routing_indices.slice(start, end)

        return TransportTensor(
            val_slice,
            new_nested_splits,
            routing_indices=new_indices,
            origin_indices=new_origin_indices,
            global_shape=self._global_shape,
            receipt=self._receipt[start:end] if self._receipt else None,
            routing_axes=self._routing_axes,
        )

    def as_dlpack_slice(self, cursor: Any | None = None) -> Any:
        """Returns a zero-copy DLPack-compatible slice of the tensor.

        Args:
             cursor: Slice argument (e.g. length of first dimension)
        """
        backend = get_backend()
        return backend.get_dlpack_slice(self._flat_values, cursor)

    def compact(self) -> "TransportTensor":
        """Trims internal padding to match actual data size.

        This operations performs a data copy (dynamic_slice) to return a clean, minimal tensor.
        Useful before network transmission to avoid sending zero-padding.
        """
        backend = get_backend()

        if not self._nested_splits:
            return self

        last_splits = self._nested_splits[-1]
        # Check last element of splits. If masked/static sliced, the last element might be huge or repeated.
        # But we rely on the implementation of slice that preserves the last valid offset.
        total_elements = backend.take(
            last_splits, backend.tensor(-1, dtype=backend.int32())
        )
        current_val_size = backend.shape(self._flat_values)[0]

        # Optimize: if total_elements is small and buffer huge
        should_compact = backend.item(total_elements) < current_val_size

        if should_compact:
            # Note: dynamic_slice requires static start? [0] is static.
            # This method is Eager-only safe.

            ndim = len(backend.shape(self._flat_values))

            starts = [0] * ndim
            sizes = [total_elements] + list(backend.shape(self._flat_values)[1:])

            new_values = backend.dynamic_slice(self._flat_values, starts, sizes)
            return TransportTensor(
                new_values,
                self._nested_splits,
                routing_indices=self._routing_indices,
                origin_indices=self._origin_indices,
                global_shape=self._global_shape,
                routing_axes=self._routing_axes,
            )
        return self

    def gather(self, indices: Any) -> "TransportTensor":
        """Reorders the tensor along the first dimension according to indices."""
        backend = get_backend()

        idx_flat = (
            indices.flat_values if isinstance(indices, TransportTensor) else indices
        )

        # Propagate routing_indices through gather
        new_routing_indices = None
        src_indices = self.routing_indices_materialized
        if src_indices is not None:
            gathered_indices = backend.take(src_indices, idx_flat)
            new_routing_indices = ExplicitIndex(gathered_indices)

        # Propagate origin metadata through gather
        new_origin_indices = None
        src_origin = (
            self.origin_indices_materialized
        )  # Materializes descriptor if needed
        if src_origin is not None:
            gathered_origin = backend.take(src_origin, idx_flat)
            new_origin_indices = ExplicitIndex(gathered_origin)

        if not self._nested_splits:
            # Dense case (can be multi-dimensional)
            gathered = backend.take(self._flat_values, idx_flat)
            # If indices were ragged, the result should have the same ragged structure
            new_splits = (
                indices.nested_splits if isinstance(indices, TransportTensor) else []
            )
            return TransportTensor(
                gathered,
                new_splits,
                routing_indices=new_routing_indices,
                origin_indices=new_origin_indices,
                global_shape=self._global_shape,
                receipt=self._receipt[idx_flat] if self._receipt else None,
                routing_axes=self._routing_axes,
            )

        current_indices = idx_flat
        new_splits = []

        for level in range(len(self._nested_splits)):
            splits = self._nested_splits[level]
            lengths = splits[1:] - splits[:-1]

            # Gather lengths based on current level permutation
            gathered_lengths = backend.take(lengths, current_indices)

            # New splits: [0, cumsum(gathered_lengths)]
            cs = backend.cumsum(gathered_lengths, dim=0)
            zero = backend.zeros((1,), dtype=backend.int32())

            new_level_split = backend.cat([zero, cs], dim=0)
            new_splits.append(new_level_split)

            # Compute next level indices
            # generate the range of indices for the next level.
            # starts = splits[:-1][current_indices]
            starts = backend.take(splits[:-1], current_indices)

            # generate: [starts[0]...starts[0]+len[0], starts[1]...starts[1]+len[1], ...]
            total_elements = backend.sum(gathered_lengths)

            # If total is 0, handle empty case
            if backend.item(total_elements) == 0:
                current_indices = backend.zeros((0,), dtype=backend.int32())
                continue

            # bases = repeat(starts, counts)
            bases = backend.repeat_interleave(starts, gathered_lengths, dim=0)

            # offsets = global_seq - adjustments
            c = backend.cumsum(gathered_lengths, dim=0)
            group_starts = c - gathered_lengths

            adjustments = backend.repeat_interleave(
                group_starts, gathered_lengths, dim=0
            )

            global_seq = backend.arange(
                0, backend.item(total_elements), dtype=backend.int32()
            )

            offsets = global_seq - adjustments

            next_indices = bases + offsets
            current_indices = next_indices

        # Finally gather flat values
        new_values = backend.take(self._flat_values, current_indices)

        return TransportTensor(
            new_values,
            new_splits,
            routing_indices=new_routing_indices,
            origin_indices=new_origin_indices,
            global_shape=self._global_shape,
            routing_axes=self._routing_axes,
        )

    def __reduce__(self):
        """Serialization interface for pickle/Ray."""
        return (
            _reconstruct_transport_tensor,
            (
                self._flat_values,
                self._nested_splits,
                self._routing_indices,
                self._global_shape,
                self._origin_indices,
                self._receipt,
                self._routing_axes,
                self._source_step,
            ),
        )

    def __repr__(self):
        parts = [f"flat_values={self._flat_values}", f"splits={self._nested_splits}"]
        if self._routing_indices is not None:
            parts.append(f"routing_indices={self._routing_indices}")
        if self._origin_indices is not None:
            parts.append(f"origin_indices={self._origin_indices}")
        return f"TransportTensor({', '.join(parts)})"


def _reconstruct_transport_tensor(
    flat_values,
    nested_splits,
    routing_indices,
    global_shape=None,
    origin_indices=None,
    receipt=None,
    routing_axes=(0,),
    source_step=None,
):
    return TransportTensor(
        flat_values,
        nested_splits,
        routing_indices=routing_indices,
        global_shape=global_shape,
        origin_indices=origin_indices,
        receipt=receipt,
        routing_axes=routing_axes,
        source_step=source_step,
    )


try:
    import jax  # pyrefly: ignore [missing-import]

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

if _JAX_AVAILABLE:
    import jax  # pyrefly: ignore [missing-import]

    def _tt_flatten(v):
        children = (
            v._flat_values,
            v._nested_splits,
            v._routing_indices,
            v._global_shape,
            v._origin_indices,
        )
        aux_data = (v._routing_axes, v._source_step)
        return children, aux_data

    def _tt_unflatten(aux_data, children):
        flat_values, nested_splits, routing_indices, global_shape, origin_indices = (
            children
        )
        routing_axes, source_step = aux_data
        return TransportTensor(
            flat_values,
            nested_splits,
            routing_indices=routing_indices,
            global_shape=global_shape,
            origin_indices=origin_indices,
            routing_axes=routing_axes,
            source_step=source_step,
        )

    jax.tree_util.register_pytree_node(TransportTensor, _tt_flatten, _tt_unflatten)
