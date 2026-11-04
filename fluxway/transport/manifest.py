"""Flat transport manifest ABI for data-plane frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
from typing import Any, cast

import numpy as np
import ml_dtypes

from fluxway.common.coherence import CoherenceAction
from fluxway.common.tensor_protocol import DLPackTensor, dlpack_tensor_shape
from fluxway.transport.coherence import (
    entry_coherence_action,
    entry_layout_state,
    entry_tensor_kind,
    pack_entry_state_flags,
)
from fluxway.transport.kernels import FRAME_TABLE_KERNEL, FrameTableCapacity


INT32_MAX = np.iinfo(np.int32).max
INT64_MAX = np.iinfo(np.int64).max
INT32_NBYTES = np.dtype(np.int32).itemsize
INT64_NBYTES = np.dtype(np.int64).itemsize
TRANSPORT_ABI_TAG = 1


class TensorEntryKind(IntEnum):
    TENSOR = 0
    ROUTING_MISS = 1
    COMPLETED_WITHOUT_TENSOR = 2
    NOT_APPLICABLE = 3


class DependencyDecision(IntEnum):
    PROCEED = 1
    SKIP = 2


class TransportProtocolError(RuntimeError):
    pass


class DependencyDecisionMismatchError(TransportProtocolError):
    pass


class PayloadRole(IntEnum):
    VALUES = 0
    RAGGED_LENGTHS = 1
    ORIGIN = 2


class PayloadDType(IntEnum):
    INT32 = 1
    INT64 = 2
    UINT32 = 3
    UINT64 = 4
    FLOAT16 = 5
    FLOAT32 = 6
    FLOAT64 = 7
    BFLOAT16 = 8
    BOOL = 9


class SenderEmissionMode(IntEnum):
    STATIC_PRODUCERS_ONLY = 0
    ALL_EXPECTED_SENDERS = 1


class FrameFlag(IntEnum):
    TENSOR_TRANSFER = 0
    ORDERING_COMPLETION = 1
    DEPENDENCY_DECISION = 2
    COHERENCE = 3


FRAME_ABI_TAG_FIELD = 0
FRAME_STEP = 1
FRAME_SENDER_SPMD_CODE = 2
FRAME_SENDER_RANK = 3
FRAME_SENDER_WORLD_SIZE = 4
FRAME_ENTRY_COUNT = 5
FRAME_PAYLOAD_COUNT = 6
FRAME_FLAGS = 7
FRAME_WIDTH = 8

ENTRY_ROUTE_CODE = 0
ENTRY_STATE_FLAGS = 1
ENTRY_PAYLOAD_BEGIN = 2
ENTRY_PAYLOAD_COUNT = 3
ENTRY_EXPECTED_COUNT = 4
ENTRY_COMPACT_BEGIN = 5
ENTRY_GLOBAL_SHAPE_BEGIN = 6
ENTRY_GLOBAL_SHAPE_COUNT = 7
ENTRY_AUX = 8
ENTRY_WIDTH = 9

PAYLOAD_ROLE = 0
PAYLOAD_DTYPE = 1
PAYLOAD_NDIM = 2
PAYLOAD_SHAPE_BEGIN = 3
PAYLOAD_SHAPE_COUNT = 4
PAYLOAD_VALID_COUNT = 5
PAYLOAD_SEGMENT_ARG_INDEX = 6
PAYLOAD_FLAGS = 7
PAYLOAD_WIDTH = 8

COMPACT_PARAM_COUNT = 4


@dataclass(frozen=True, slots=True)
class TransportPlanManifest:
    expected_senders_per_dst: tuple[int, ...]
    sender_has_data_per_dst: tuple[bool, ...]
    emission_mode: SenderEmissionMode

    def __post_init__(self) -> None:
        if not isinstance(self.expected_senders_per_dst, tuple):
            raise TypeError("expected_senders_per_dst must be a tuple")
        if not isinstance(self.sender_has_data_per_dst, tuple):
            raise TypeError("sender_has_data_per_dst must be a tuple")
        if len(self.expected_senders_per_dst) != len(self.sender_has_data_per_dst):
            raise ValueError(
                "TransportPlanManifest expected_senders_per_dst and "
                "sender_has_data_per_dst must have the same length."
            )
        for value in self.expected_senders_per_dst:
            _as_i32(value, "expected_senders_per_dst entry")
        if not isinstance(self.emission_mode, SenderEmissionMode):
            raise TypeError("emission_mode must be a SenderEmissionMode")

    @property
    def target_world_size(self) -> int:
        return len(self.expected_senders_per_dst)

    def expected_count_for(self, dst_rank: int) -> int:
        if dst_rank < 0 or dst_rank >= self.target_world_size:
            raise IndexError(
                f"dst_rank={dst_rank} is outside manifest target world size {self.target_world_size}."
            )
        return self.expected_senders_per_dst[dst_rank]

    def should_emit_to(self, dst_rank: int) -> bool:
        if dst_rank < 0 or dst_rank >= self.target_world_size:
            raise IndexError(
                f"dst_rank={dst_rank} is outside manifest target world size {self.target_world_size}."
            )
        if self.expected_senders_per_dst[dst_rank] <= 0:
            return False
        if self.emission_mode is SenderEmissionMode.ALL_EXPECTED_SENDERS:
            return True
        return self.sender_has_data_per_dst[dst_rank]


def _as_integral(value: int, field: str) -> int:
    if isinstance(value, bool) or isinstance(value, np.bool_):
        raise TypeError(f"{field} must be integer scalar, got bool.")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be integer scalar, got {type(value).__name__}.")
    return value


def _as_i32(value: int, field: str) -> int:
    int_value = _as_integral(value, field)
    if int_value < 0 or int_value > INT32_MAX:
        raise OverflowError(
            f"{field}={int_value} is outside int32 range [0, {INT32_MAX}]."
        )
    return int_value


def _as_i64(value: int, field: str) -> int:
    int_value = _as_integral(value, field)
    if int_value < 0 or int_value > INT64_MAX:
        raise OverflowError(
            f"{field}={int_value} is outside int64 range [0, {INT64_MAX}]."
        )
    return int_value


def dtype_code(dtype: Any) -> int:
    dt = np.dtype(dtype)
    if dt == np.dtype(np.bool_):
        return int(PayloadDType.BOOL)
    if dt == np.dtype(np.int32):
        return int(PayloadDType.INT32)
    if dt == np.dtype(np.int64):
        return int(PayloadDType.INT64)
    if dt == np.dtype(np.uint32):
        return int(PayloadDType.UINT32)
    if dt == np.dtype(np.uint64):
        return int(PayloadDType.UINT64)
    if dt == np.dtype(np.float16):
        return int(PayloadDType.FLOAT16)
    if dt == np.dtype(np.float32):
        return int(PayloadDType.FLOAT32)
    if dt == np.dtype(np.float64):
        return int(PayloadDType.FLOAT64)
    if str(dt) == "bfloat16":
        return int(PayloadDType.BFLOAT16)
    raise TypeError(f"Unsupported transport payload dtype {dt}.")


def dtype_from_code(code: int) -> np.dtype:
    payload_dtype = PayloadDType(code)
    if payload_dtype is PayloadDType.INT32:
        return np.dtype(np.int32)
    if payload_dtype is PayloadDType.INT64:
        return np.dtype(np.int64)
    if payload_dtype is PayloadDType.UINT32:
        return np.dtype(np.uint32)
    if payload_dtype is PayloadDType.UINT64:
        return np.dtype(np.uint64)
    if payload_dtype is PayloadDType.FLOAT16:
        return np.dtype(np.float16)
    if payload_dtype is PayloadDType.FLOAT32:
        return np.dtype(np.float32)
    if payload_dtype is PayloadDType.FLOAT64:
        return np.dtype(np.float64)
    if payload_dtype is PayloadDType.BFLOAT16:
        return np.dtype(ml_dtypes.bfloat16)
    if payload_dtype is PayloadDType.BOOL:
        return np.dtype(np.bool_)
    raise ValueError(f"Unsupported transport payload dtype code {code}.")


def _require_nonnegative_i32_table(name: str, table: np.ndarray) -> None:
    if table.size and np.any(table < 0):
        first = np.argwhere(table < 0)[0]
        raise ValueError(
            f"{name} contains negative int32 metadata at index {tuple(int(x) for x in first)}."
        )


def _validate_enum(enum_type: type[IntEnum], value: int, field: str) -> None:
    try:
        enum_type(value)
    except ValueError as exc:
        raise ValueError(
            f"{field}={value} is not a valid {enum_type.__name__}."
        ) from exc


def _require_host_metadata_array(
    name: str,
    table: Any,
    *,
    dtype: Any,
    ndim: int,
    width: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if not isinstance(table, np.ndarray):
        raise TypeError(
            f"{name} must be a host numpy metadata array, got {type(table).__name__}."
        )
    if hasattr(table, "__cuda_array_interface__"):
        raise TypeError(f"{name} must be host metadata, not a CUDA array.")
    expected_dtype = np.dtype(dtype)
    if table.dtype != expected_dtype:
        raise ValueError(
            f"{name} must have dtype={expected_dtype}, got dtype={table.dtype}."
        )
    if table.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got ndim={table.ndim}.")
    if width is not None and (table.ndim != 2 or table.shape[1] != width):
        raise ValueError(f"{name} width must be {width}, got shape={table.shape}.")
    if shape is not None and table.shape != shape:
        raise ValueError(f"{name} must have shape={shape}, got shape={table.shape}.")
    return table


@dataclass(frozen=True, slots=True)
class TransportFrameHeader:
    frame_i32: np.ndarray
    entry_i32: np.ndarray
    payload_i32: np.ndarray
    shape_i32: np.ndarray
    global_shape_i64: np.ndarray
    compact_i64: np.ndarray

    def validate(self) -> None:
        _require_host_metadata_array(
            "frame_i32", self.frame_i32, dtype=np.int32, ndim=1, shape=(FRAME_WIDTH,)
        )
        _require_host_metadata_array(
            "entry_i32", self.entry_i32, dtype=np.int32, ndim=2, width=ENTRY_WIDTH
        )
        _require_host_metadata_array(
            "payload_i32",
            self.payload_i32,
            dtype=np.int32,
            ndim=2,
            width=PAYLOAD_WIDTH,
        )
        _require_host_metadata_array(
            "shape_i32", self.shape_i32, dtype=np.int32, ndim=1
        )
        _require_host_metadata_array(
            "global_shape_i64", self.global_shape_i64, dtype=np.int64, ndim=1
        )
        _require_host_metadata_array(
            "compact_i64", self.compact_i64, dtype=np.int64, ndim=1
        )
        _require_nonnegative_i32_table("frame_i32", self.frame_i32)
        _require_nonnegative_i32_table("entry_i32", self.entry_i32)
        _require_nonnegative_i32_table("payload_i32", self.payload_i32)
        _require_nonnegative_i32_table("shape_i32", self.shape_i32)
        sender_rank = int(self.frame_i32[FRAME_SENDER_RANK])
        sender_world_size = int(self.frame_i32[FRAME_SENDER_WORLD_SIZE])
        if sender_world_size <= 0 or sender_rank >= sender_world_size:
            raise ValueError(
                "Invalid sender rank domain: "
                f"rank={sender_rank} world_size={sender_world_size}."
            )
        if int(self.frame_i32[FRAME_ABI_TAG_FIELD]) != TRANSPORT_ABI_TAG:
            raise ValueError(
                f"Unsupported transport frame ABI tag {int(self.frame_i32[FRAME_ABI_TAG_FIELD])}; "
                f"expected {TRANSPORT_ABI_TAG}."
            )
        frame_flag_value = int(self.frame_i32[FRAME_FLAGS])
        _validate_enum(FrameFlag, frame_flag_value, "frame flags")
        frame_flag = FrameFlag(frame_flag_value)
        entry_count = int(self.frame_i32[FRAME_ENTRY_COUNT])
        payload_count = int(self.frame_i32[FRAME_PAYLOAD_COUNT])
        if frame_flag in (
            FrameFlag.ORDERING_COMPLETION,
            FrameFlag.DEPENDENCY_DECISION,
            FrameFlag.COHERENCE,
        ):
            if entry_count == 0:
                raise ValueError(
                    f"{frame_flag.name} transport frames require at least one entry."
                )
            if payload_count != 0:
                raise ValueError(
                    f"{frame_flag.name} transport frames cannot contain payloads."
                )
        if entry_count != self.entry_i32.shape[0]:
            raise ValueError(
                f"frame entry_count={entry_count} does not match entry_i32 entries={self.entry_i32.shape[0]}."
            )
        if payload_count != self.payload_i32.shape[0]:
            raise ValueError(
                f"frame payload_count={payload_count} does not match payload_i32 entries={self.payload_i32.shape[0]}."
            )
        decision_dependency_codes: set[int] = set()
        coherence_frame_action: CoherenceAction | None = None
        coherence_frame_epoch: int | None = None
        for entry_index, entry in enumerate(self.entry_i32):
            state_flags = int(entry[ENTRY_STATE_FLAGS])
            tensor_kind = TensorEntryKind(entry_tensor_kind(state_flags))
            coherence_action = entry_coherence_action(state_flags)
            layout_state = entry_layout_state(state_flags)
            aux_value = int(entry[ENTRY_AUX])
            payload_begin = int(entry[ENTRY_PAYLOAD_BEGIN])
            payload_count_row = int(entry[ENTRY_PAYLOAD_COUNT])
            compact_begin = int(entry[ENTRY_COMPACT_BEGIN])
            compact_count = COMPACT_PARAM_COUNT if layout_state & 4 else 0
            global_shape_begin = int(entry[ENTRY_GLOBAL_SHAPE_BEGIN])
            global_shape_count = int(entry[ENTRY_GLOBAL_SHAPE_COUNT])
            if frame_flag is FrameFlag.TENSOR_TRANSFER and payload_count_row == 0:
                raise ValueError(
                    f"tensor transfer entry {entry_index} requires at least one payload entry."
                )
            if frame_flag is FrameFlag.TENSOR_TRANSFER:
                if tensor_kind is TensorEntryKind.NOT_APPLICABLE:
                    raise ValueError(
                        f"tensor transfer entry {entry_index} requires a tensor entry kind."
                    )
                if coherence_action is not CoherenceAction.BUS_RD:
                    raise ValueError(
                        f"tensor transfer entry {entry_index} requires BUS_RD, "
                        f"got {coherence_action.name}."
                    )
            if frame_flag is FrameFlag.ORDERING_COMPLETION:
                if layout_state != 0:
                    raise ValueError(
                        f"ordering dependency entry {entry_index} cannot contain tensor metadata."
                    )
                if tensor_kind is not TensorEntryKind.NOT_APPLICABLE:
                    raise ValueError(
                        f"ordering dependency entry {entry_index} must use "
                        "NOT_APPLICABLE."
                    )
                if coherence_action is not CoherenceAction.NONE or aux_value != 0:
                    raise ValueError(
                        f"ordering dependency entry {entry_index} cannot contain aux metadata."
                    )
                if payload_count_row != 0:
                    raise ValueError(
                        f"ordering dependency entry {entry_index} cannot reference payload entries."
                    )
                if compact_count != 0 or global_shape_count != 0:
                    raise ValueError(
                        f"ordering dependency entry {entry_index} cannot contain tensor shapes."
                    )
            if frame_flag is FrameFlag.DEPENDENCY_DECISION:
                _validate_enum(
                    DependencyDecision,
                    aux_value,
                    f"frame entry {entry_index} dependency_decision",
                )
                dependency_code = int(entry[ENTRY_ROUTE_CODE])
                if dependency_code in decision_dependency_codes:
                    raise ValueError(
                        "DEPENDENCY_DECISION frame contains duplicate "
                        f"dependency_code={dependency_code}."
                    )
                decision_dependency_codes.add(dependency_code)
                if layout_state != 0:
                    raise ValueError(
                        f"dependency decision entry {entry_index} cannot contain tensor metadata."
                    )
                if tensor_kind is not TensorEntryKind.NOT_APPLICABLE:
                    raise ValueError(
                        f"dependency decision entry {entry_index} must use NOT_APPLICABLE."
                    )
                if payload_count_row != 0:
                    raise ValueError(
                        f"dependency decision entry {entry_index} cannot reference payload entries."
                    )
                if compact_count != 0 or global_shape_count != 0:
                    raise ValueError(
                        f"dependency decision entry {entry_index} cannot contain tensor shapes."
                    )
            if frame_flag is FrameFlag.COHERENCE:
                if layout_state != 0:
                    raise ValueError(
                        f"coherence entry {entry_index} cannot contain tensor metadata."
                    )
                if tensor_kind is not TensorEntryKind.NOT_APPLICABLE:
                    raise ValueError(
                        f"coherence entry {entry_index} must use NOT_APPLICABLE."
                    )
                if coherence_action is CoherenceAction.NONE:
                    raise ValueError(
                        f"coherence entry {entry_index} requires a coherence action."
                    )
                if coherence_frame_action is None:
                    coherence_frame_action = coherence_action
                    coherence_frame_epoch = aux_value
                elif (
                    coherence_action is not coherence_frame_action
                    or aux_value != coherence_frame_epoch
                ):
                    raise ValueError(
                        "coherence frame entries must use one action and one epoch."
                    )
                if aux_value != int(self.frame_i32[FRAME_STEP]):
                    raise ValueError(
                        f"coherence entry {entry_index} epoch={aux_value} does not "
                        f"match frame step={int(self.frame_i32[FRAME_STEP])}."
                    )
                if (
                    payload_count_row != 0
                    or compact_count != 0
                    or global_shape_count != 0
                ):
                    raise ValueError(
                        f"coherence entry {entry_index} cannot contain payload metadata."
                    )
            if payload_begin + payload_count_row > payload_count:
                raise ValueError(
                    f"frame entry {entry_index} payload range [{payload_begin}, "
                    f"{payload_begin + payload_count_row}) exceeds payload_count={payload_count}."
                )
            if frame_flag is FrameFlag.TENSOR_TRANSFER and tensor_kind in (
                TensorEntryKind.ROUTING_MISS,
                TensorEntryKind.COMPLETED_WITHOUT_TENSOR,
            ):
                for payload in self.payload_i32[
                    payload_begin : payload_begin + payload_count_row
                ]:
                    if int(payload[PAYLOAD_VALID_COUNT]) != 0:
                        raise ValueError(
                            f"tensor transfer entry {entry_index} kind "
                            f"{tensor_kind.name} requires zero payload valid counts."
                        )
            if compact_begin + compact_count > self.compact_i64.shape[0]:
                raise ValueError(
                    f"frame entry {entry_index} compact range [{compact_begin}, "
                    f"{compact_begin + compact_count}) exceeds compact_i64 size={self.compact_i64.shape[0]}."
                )
            if global_shape_begin + global_shape_count > self.global_shape_i64.shape[0]:
                raise ValueError(
                    f"frame entry {entry_index} global shape range [{global_shape_begin}, "
                    f"{global_shape_begin + global_shape_count}) exceeds "
                    f"global_shape_i64 size={self.global_shape_i64.shape[0]}."
                )
        for entry_index, entry in enumerate(self.payload_i32):
            _validate_enum(
                PayloadRole,
                int(entry[PAYLOAD_ROLE]),
                f"payload entry {entry_index} role",
            )
            _validate_enum(
                PayloadDType,
                int(entry[PAYLOAD_DTYPE]),
                f"payload entry {entry_index} dtype",
            )
            shape_begin = int(entry[PAYLOAD_SHAPE_BEGIN])
            shape_count = int(entry[PAYLOAD_SHAPE_COUNT])
            if shape_begin + shape_count > self.shape_i32.shape[0]:
                raise ValueError(
                    f"payload entry {entry_index} shape range [{shape_begin}, "
                    f"{shape_begin + shape_count}) exceeds shape_i32 size={self.shape_i32.shape[0]}."
                )

    def to_frame(self, payload_segments: tuple[DLPackTensor, ...]) -> TransportFrame:
        frame = TransportFrame(
            frame_i32=self.frame_i32,
            entry_i32=self.entry_i32,
            payload_i32=self.payload_i32,
            shape_i32=self.shape_i32,
            global_shape_i64=self.global_shape_i64,
            compact_i64=self.compact_i64,
            payload_segments=payload_segments,
        )
        frame.validate()
        return frame


@dataclass(frozen=True)
class TransportFrame:
    frame_i32: np.ndarray
    entry_i32: np.ndarray
    payload_i32: np.ndarray
    shape_i32: np.ndarray
    global_shape_i64: np.ndarray
    compact_i64: np.ndarray
    payload_segments: tuple[DLPackTensor, ...]

    @property
    def header(self) -> TransportFrameHeader:
        return TransportFrameHeader(
            frame_i32=self.frame_i32,
            entry_i32=self.entry_i32,
            payload_i32=self.payload_i32,
            shape_i32=self.shape_i32,
            global_shape_i64=self.global_shape_i64,
            compact_i64=self.compact_i64,
        )

    def validate(self) -> None:
        self.header.validate()
        for entry_index, entry in enumerate(self.payload_i32):
            segment_idx = int(entry[PAYLOAD_SEGMENT_ARG_INDEX])
            if segment_idx >= len(self.payload_segments):
                raise ValueError(
                    f"payload entry {entry_index} segment index {segment_idx} exceeds "
                    f"payload segment count {len(self.payload_segments)}."
                )
            segment = self.payload_segments[segment_idx]
            payload_shape = dlpack_tensor_shape(
                segment, name=f"payload segment {segment_idx}"
            )
            actual_dtype_code = dtype_code(segment.dtype)
            declared_dtype_code = int(entry[PAYLOAD_DTYPE])
            if actual_dtype_code != declared_dtype_code:
                raise ValueError(
                    f"payload entry {entry_index} dtype_code={declared_dtype_code} does not "
                    f"match segment dtype_code={actual_dtype_code}."
                )
            shape_begin = int(entry[PAYLOAD_SHAPE_BEGIN])
            shape_count = int(entry[PAYLOAD_SHAPE_COUNT])
            declared_shape = tuple(
                int(dimension)
                for dimension in self.shape_i32[shape_begin : shape_begin + shape_count]
            )
            if payload_shape != declared_shape:
                raise ValueError(
                    f"payload entry {entry_index} shape={declared_shape} does not "
                    f"match segment shape={payload_shape}."
                )


@dataclass(frozen=True, slots=True)
class TransportFrameTemplate:
    frame_i32: np.ndarray
    entry_i32: np.ndarray
    payload_i32: np.ndarray
    shape_i32: np.ndarray
    global_shape_i64: np.ndarray
    compact_i64: np.ndarray

    @classmethod
    def from_frame(cls, frame: TransportFrame) -> TransportFrameTemplate:
        frame.validate()
        return cls(
            frame_i32=frame.frame_i32.copy(),
            entry_i32=frame.entry_i32.copy(),
            payload_i32=frame.payload_i32.copy(),
            shape_i32=frame.shape_i32.copy(),
            global_shape_i64=frame.global_shape_i64.copy(),
            compact_i64=frame.compact_i64.copy(),
        )

    def instantiate(
        self,
        *,
        step: int,
        payload_segments: tuple[DLPackTensor, ...],
        payload_valid_counts: tuple[int, ...],
        global_shapes: tuple[tuple[int, ...] | None, ...],
        coherence_epochs: tuple[int, ...],
    ) -> TransportFrame:
        if len(payload_segments) != self.payload_i32.shape[0]:
            raise ValueError(
                "payload segment count does not match template payload entries: "
                f"segments={len(payload_segments)} entries={self.payload_i32.shape[0]}."
            )
        if len(payload_valid_counts) != self.payload_i32.shape[0]:
            raise ValueError(
                "payload valid count length does not match template payload entries: "
                f"counts={len(payload_valid_counts)} entries={self.payload_i32.shape[0]}."
            )
        if len(global_shapes) != self.entry_i32.shape[0]:
            raise ValueError(
                "global shape length does not match template entry entries: "
                f"shapes={len(global_shapes)} entries={self.entry_i32.shape[0]}."
            )
        if len(coherence_epochs) != self.entry_i32.shape[0]:
            raise ValueError(
                "coherence epoch length does not match template entry entries: "
                f"epochs={len(coherence_epochs)} entries={self.entry_i32.shape[0]}."
            )

        frame_i32 = self.frame_i32.copy()
        entry_i32 = self.entry_i32.copy()
        payload_i32 = self.payload_i32.copy()
        shape_i32 = self.shape_i32.copy()
        global_shape_i64 = self.global_shape_i64.copy()
        frame_i32[FRAME_STEP] = _as_i32(step, "step")

        for payload_index, segment in enumerate(payload_segments):
            payload_shape = dlpack_tensor_shape(
                segment, name=f"payload segment {payload_index}"
            )
            entry = payload_i32[payload_index]
            actual_dtype_code = dtype_code(segment.dtype)
            expected_dtype_code = int(entry[PAYLOAD_DTYPE])
            if actual_dtype_code != expected_dtype_code:
                raise ValueError(
                    f"payload entry {payload_index} dtype_code={expected_dtype_code} "
                    f"does not match segment dtype_code={actual_dtype_code}."
                )
            shape_count = int(entry[PAYLOAD_SHAPE_COUNT])
            if len(payload_shape) != shape_count:
                raise ValueError(
                    f"payload entry {payload_index} ndim={shape_count} does not match "
                    f"segment ndim={len(payload_shape)}."
                )
            shape_begin = int(entry[PAYLOAD_SHAPE_BEGIN])
            for offset, value in enumerate(payload_shape):
                shape_i32[shape_begin + offset] = _as_i32(
                    value, f"payload shape dim {offset}"
                )
            entry[PAYLOAD_VALID_COUNT] = _as_i32(
                payload_valid_counts[payload_index], "valid_count"
            )
            entry[PAYLOAD_SEGMENT_ARG_INDEX] = _as_i32(
                payload_index, "payload segment index"
            )

        for entry_index, (global_shape, coherence_epoch) in enumerate(
            zip(global_shapes, coherence_epochs, strict=True)
        ):
            entry = entry_i32[entry_index]
            entry[ENTRY_AUX] = _as_i32(coherence_epoch, "coherence_epoch")
            shape_count = int(entry[ENTRY_GLOBAL_SHAPE_COUNT])
            shape_begin = int(entry[ENTRY_GLOBAL_SHAPE_BEGIN])
            if global_shape is None:
                if shape_count != 0:
                    raise ValueError(
                        f"frame entry {entry_index} expected global_shape rank "
                        f"{shape_count}, got None."
                    )
                continue
            if len(global_shape) != shape_count:
                raise ValueError(
                    f"frame entry {entry_index} expected global_shape rank "
                    f"{shape_count}, got {len(global_shape)}."
                )
            for offset, value in enumerate(global_shape):
                global_shape_i64[shape_begin + offset] = _as_i64(
                    value, f"global_shape[{offset}]"
                )

        frame = TransportFrame(
            frame_i32=frame_i32,
            entry_i32=entry_i32,
            payload_i32=payload_i32,
            shape_i32=shape_i32,
            global_shape_i64=global_shape_i64,
            compact_i64=self.compact_i64.copy(),
            payload_segments=payload_segments,
        )
        frame.header.validate()
        return frame


class TransportFrameBuilder:
    _ENTRY_STRUCT = struct.Struct("=" + "i" * ENTRY_WIDTH)
    _PAYLOAD_STRUCT = struct.Struct("=" + "i" * PAYLOAD_WIDTH)
    _I32_SCALAR_STRUCT = struct.Struct("=i")

    def __init__(
        self,
        step: int,
        sender_spmd_code: int,
        sender_rank: int,
        sender_world_size: int,
        flags: FrameFlag = FrameFlag.TENSOR_TRANSFER,
        capacity: FrameTableCapacity | None = None,
    ):
        self._step = _as_i32(step, "step")
        self._sender_spmd_code = _as_i32(sender_spmd_code, "sender_spmd_code")
        self._sender_rank = _as_i32(sender_rank, "sender_rank")
        self._sender_world_size = _as_i32(sender_world_size, "sender_world_size")
        if self._sender_world_size <= 0 or self._sender_rank >= self._sender_world_size:
            raise ValueError(
                "Invalid sender rank domain: "
                f"rank={self._sender_rank} world_size={self._sender_world_size}."
            )
        if not isinstance(flags, FrameFlag):
            raise TypeError(f"flags must be FrameFlag, got {type(flags).__name__}.")
        self._frame_flag = flags
        self._flags = int(flags)
        self._capacity = capacity or FrameTableCapacity()
        self._entry_i32_buffer = FRAME_TABLE_KERNEL.empty_byte_buffer(
            self._capacity.entry_count * ENTRY_WIDTH,
            INT32_NBYTES,
            "entry_i32",
        )
        self._payload_i32_buffer = FRAME_TABLE_KERNEL.empty_byte_buffer(
            self._capacity.payload_count * PAYLOAD_WIDTH,
            INT32_NBYTES,
            "payload_i32",
        )
        self._shape_i32_buffer = FRAME_TABLE_KERNEL.empty_byte_buffer(
            self._capacity.shape_count, INT32_NBYTES, "shape_i32"
        )
        self._global_shape_i64_buffer = FRAME_TABLE_KERNEL.empty_byte_buffer(
            self._capacity.global_shape_count,
            INT64_NBYTES,
            "global_shape_i64",
        )
        self._compact_i64_buffer = FRAME_TABLE_KERNEL.empty_byte_buffer(
            self._capacity.compact_count,
            INT64_NBYTES,
            "compact_i64",
        )
        self._segments: list[DLPackTensor | None] = [
            None
        ] * self._capacity.segment_count
        self._entry_capacity = self._capacity.entry_count
        self._payload_capacity = self._capacity.payload_count
        self._shape_capacity = self._capacity.shape_count
        self._global_shape_capacity = self._capacity.global_shape_count
        self._compact_capacity = self._capacity.compact_count
        self._segment_capacity = self._capacity.segment_count
        self._entry_count = 0
        self._payload_count = 0
        self._shape_count = 0
        self._global_shape_count = 0
        self._compact_count = 0
        self._segment_count = 0
        self._decision_dependency_codes: set[int] = set()

    def add_tensor_transfer_entry(
        self,
        *,
        transfer_code: int,
        layout_state: int,
        tensor_kind: TensorEntryKind,
        expected_count: int,
        coherence_epoch: int,
        coherence_action: CoherenceAction = CoherenceAction.BUS_RD,
        compact_params: list[int] | tuple[int, ...] | None = None,
        global_shape: tuple[int, ...] | None = None,
    ) -> int:
        if self._frame_flag is not FrameFlag.TENSOR_TRANSFER:
            raise ValueError("Tensor transfers require a TENSOR_TRANSFER frame.")
        return self._add_route(
            route_code=transfer_code,
            layout_state=layout_state,
            tensor_kind=tensor_kind,
            expected_count=expected_count,
            aux=coherence_epoch,
            coherence_action=coherence_action,
            compact_params=compact_params,
            global_shape=global_shape,
        )

    def add_coherence_entry(
        self,
        *,
        route_code: int,
        coherence_epoch: int,
        coherence_action: CoherenceAction,
        expected_count: int = 1,
    ) -> int:
        if self._frame_flag is not FrameFlag.COHERENCE:
            raise ValueError("Coherence entries require a COHERENCE frame.")
        if coherence_action is CoherenceAction.NONE:
            raise ValueError("coherence_action must not be NONE.")
        return self._add_route(
            route_code=route_code,
            layout_state=0,
            tensor_kind=TensorEntryKind.NOT_APPLICABLE,
            expected_count=expected_count,
            aux=coherence_epoch,
            coherence_action=coherence_action,
        )

    def _add_route(
        self,
        *,
        route_code: int,
        layout_state: int,
        tensor_kind: TensorEntryKind,
        expected_count: int,
        aux: int,
        coherence_action: CoherenceAction = CoherenceAction.NONE,
        compact_params: list[int] | tuple[int, ...] | None = None,
        global_shape: tuple[int, ...] | None = None,
        dependency_decision: DependencyDecision | None = None,
    ) -> int:
        payload_begin = self._payload_count
        compact_begin = self.add_compact_params(compact_params)
        global_shape_begin = self._global_shape_count
        global_shape_count = 0
        if global_shape is not None:
            global_shape_values = tuple(
                _as_i64(value, f"global_shape[{i}]")
                for i, value in enumerate(global_shape)
            )
            self._ensure_global_shape_capacity(
                self._global_shape_count + len(global_shape_values)
            )
            struct.pack_into(
                "=" + "q" * len(global_shape_values),
                self._global_shape_i64_buffer,
                self._global_shape_count * INT64_NBYTES,
                *global_shape_values,
            )
            global_shape_count = len(global_shape)
            self._global_shape_count += global_shape_count
        entry_index = self._entry_count
        self._ensure_entry_capacity(entry_index + 1)
        self._ENTRY_STRUCT.pack_into(
            self._entry_i32_buffer,
            entry_index * self._ENTRY_STRUCT.size,
            _as_i32(route_code, "route_code"),
            pack_entry_state_flags(
                layout_state,
                int(tensor_kind),
                coherence_action,
            ),
            _as_i32(payload_begin, "payload_begin"),
            0,
            _as_i32(expected_count, "expected_count"),
            _as_i32(compact_begin, "compact_begin"),
            _as_i32(global_shape_begin, "global_shape_begin"),
            _as_i32(global_shape_count, "global_shape_count"),
            _as_i32(
                aux if dependency_decision is None else int(dependency_decision),
                "entry_aux",
            ),
        )
        self._entry_count += 1
        return entry_index

    def add_ordering_completion_entry(
        self,
        *,
        dependency_code: int,
        expected_count: int,
    ) -> int:
        if self._frame_flag is not FrameFlag.ORDERING_COMPLETION:
            raise ValueError(
                "Ordering completions require an ORDERING_COMPLETION frame."
            )
        return self._add_route(
            route_code=dependency_code,
            layout_state=0,
            tensor_kind=TensorEntryKind.NOT_APPLICABLE,
            expected_count=expected_count,
            aux=0,
        )

    def add_dependency_decision_entry(
        self,
        *,
        dependency_code: int,
        decision: DependencyDecision,
        expected_count: int,
    ) -> int:
        if self._frame_flag is not FrameFlag.DEPENDENCY_DECISION:
            raise ValueError(
                "Dependency decisions require a DEPENDENCY_DECISION transport frame."
            )
        if not isinstance(decision, DependencyDecision):
            raise TypeError(
                f"decision must be DependencyDecision, got {type(decision).__name__}."
            )
        if dependency_code in self._decision_dependency_codes:
            raise ValueError(
                "DEPENDENCY_DECISION frame has duplicate "
                f"dependency_code={dependency_code}."
            )
        self._decision_dependency_codes.add(dependency_code)
        return self._add_route(
            route_code=dependency_code,
            layout_state=0,
            tensor_kind=TensorEntryKind.NOT_APPLICABLE,
            expected_count=expected_count,
            aux=int(decision),
            dependency_decision=decision,
        )

    def add_payload(
        self,
        entry_index: int,
        role: PayloadRole,
        segment: DLPackTensor,
        valid_count: int,
    ) -> None:
        if self._frame_flag is not FrameFlag.TENSOR_TRANSFER:
            raise ValueError(
                f"{self._frame_flag.name} dependency frames cannot contain payloads."
            )
        if entry_index < 0 or entry_index >= self._entry_count:
            raise IndexError(f"entry_index {entry_index} is out of range.")
        shape = dlpack_tensor_shape(segment, name="payload segment")
        shape_begin = self._shape_count
        shape_values = tuple(
            _as_i32(value, f"payload shape dim {dim}")
            for dim, value in enumerate(shape)
        )
        self._ensure_shape_capacity(self._shape_count + len(shape_values))
        struct.pack_into(
            "=" + "i" * len(shape_values),
            self._shape_i32_buffer,
            self._shape_count * INT32_NBYTES,
            *shape_values,
        )
        self._shape_count += len(shape_values)

        segment_idx = self._segment_count
        self._ensure_segment_capacity(segment_idx + 1)
        self._segments[segment_idx] = segment
        self._segment_count += 1

        payload_index = self._payload_count
        self._ensure_payload_capacity(payload_index + 1)
        self._PAYLOAD_STRUCT.pack_into(
            self._payload_i32_buffer,
            payload_index * self._PAYLOAD_STRUCT.size,
            _as_i32(int(role), "payload role"),
            _as_i32(dtype_code(segment.dtype), "dtype_code"),
            _as_i32(len(shape), "ndim"),
            _as_i32(shape_begin, "shape_begin"),
            _as_i32(len(shape), "shape_count"),
            _as_i32(valid_count, "valid_count"),
            _as_i32(segment_idx, "segment_arg_index"),
            0,
        )
        self._payload_count += 1
        payload_count_byte = (
            entry_index * ENTRY_WIDTH + ENTRY_PAYLOAD_COUNT
        ) * INT32_NBYTES
        payload_count = (
            self._I32_SCALAR_STRUCT.unpack_from(
                self._entry_i32_buffer, payload_count_byte
            )[0]
            + 1
        )
        self._I32_SCALAR_STRUCT.pack_into(
            self._entry_i32_buffer, payload_count_byte, payload_count
        )
        _as_i32(payload_count, "payload_count")

    def add_compact_params(self, params: list[int] | tuple[int, ...] | None) -> int:
        if params is None:
            return 0
        if len(params) != COMPACT_PARAM_COUNT:
            raise ValueError(
                f"compact params must have {COMPACT_PARAM_COUNT} entries, got {len(params)}."
            )
        begin = self._compact_count
        compact_values = tuple(
            _as_i64(value, f"compact[{i}]") for i, value in enumerate(params)
        )
        self._ensure_compact_capacity(self._compact_count + COMPACT_PARAM_COUNT)
        struct.pack_into(
            "=" + "q" * len(compact_values),
            self._compact_i64_buffer,
            self._compact_count * INT64_NBYTES,
            *compact_values,
        )
        self._compact_count += COMPACT_PARAM_COUNT
        return begin

    def build(self) -> TransportFrame:
        entry_count = _as_i32(self._entry_count, "entry_count")
        payload_count = _as_i32(self._payload_count, "payload_count")
        frame_i32 = np.array(
            [
                TRANSPORT_ABI_TAG,
                self._step,
                self._sender_spmd_code,
                self._sender_rank,
                self._sender_world_size,
                entry_count,
                payload_count,
                self._flags,
            ],
            dtype=np.int32,
        )
        entry_i32 = FRAME_TABLE_KERNEL.i32_table_from_byte_buffer(
            self._entry_i32_buffer, entry_count, ENTRY_WIDTH, "entry_i32"
        )
        payload_i32 = FRAME_TABLE_KERNEL.i32_table_from_byte_buffer(
            self._payload_i32_buffer, payload_count, PAYLOAD_WIDTH, "payload_i32"
        )
        shape_i32 = FRAME_TABLE_KERNEL.i32_vector_from_byte_buffer(
            self._shape_i32_buffer, self._shape_count, "shape_i32"
        )
        global_shape_i64 = FRAME_TABLE_KERNEL.i64_vector_from_byte_buffer(
            self._global_shape_i64_buffer,
            self._global_shape_count,
            "global_shape_i64",
        )
        compact_i64 = FRAME_TABLE_KERNEL.i64_vector_from_byte_buffer(
            self._compact_i64_buffer, self._compact_count, "compact_i64"
        )
        frame = TransportFrame(
            frame_i32=frame_i32,
            entry_i32=entry_i32,
            payload_i32=payload_i32,
            shape_i32=shape_i32,
            global_shape_i64=global_shape_i64,
            compact_i64=compact_i64,
            payload_segments=cast(
                tuple[DLPackTensor, ...],
                tuple(self._segments[: self._segment_count]),
            ),
        )
        frame.header.validate()
        return frame

    def _ensure_entry_capacity(self, needed: int) -> None:
        if needed <= self._entry_capacity:
            return
        next_capacity = self._next_capacity(self._entry_capacity, needed)
        self._entry_i32_buffer = FRAME_TABLE_KERNEL.grow_byte_buffer(
            self._entry_i32_buffer,
            next_capacity * ENTRY_WIDTH,
            INT32_NBYTES,
            "entry_i32",
        )
        self._entry_capacity = next_capacity

    def _ensure_payload_capacity(self, needed: int) -> None:
        if needed <= self._payload_capacity:
            return
        next_capacity = self._next_capacity(self._payload_capacity, needed)
        self._payload_i32_buffer = FRAME_TABLE_KERNEL.grow_byte_buffer(
            self._payload_i32_buffer,
            next_capacity * PAYLOAD_WIDTH,
            INT32_NBYTES,
            "payload_i32",
        )
        self._payload_capacity = next_capacity

    def _ensure_shape_capacity(self, needed: int) -> None:
        if needed <= self._shape_capacity:
            return
        next_capacity = self._next_capacity(self._shape_capacity, needed)
        self._shape_i32_buffer = FRAME_TABLE_KERNEL.grow_byte_buffer(
            self._shape_i32_buffer,
            next_capacity,
            INT32_NBYTES,
            "shape_i32",
        )
        self._shape_capacity = next_capacity

    def _ensure_global_shape_capacity(self, needed: int) -> None:
        if needed <= self._global_shape_capacity:
            return
        next_capacity = self._next_capacity(self._global_shape_capacity, needed)
        self._global_shape_i64_buffer = FRAME_TABLE_KERNEL.grow_byte_buffer(
            self._global_shape_i64_buffer,
            next_capacity,
            INT64_NBYTES,
            "global_shape_i64",
        )
        self._global_shape_capacity = next_capacity

    def _ensure_compact_capacity(self, needed: int) -> None:
        if needed <= self._compact_capacity:
            return
        next_capacity = self._next_capacity(self._compact_capacity, needed)
        self._compact_i64_buffer = FRAME_TABLE_KERNEL.grow_byte_buffer(
            self._compact_i64_buffer,
            next_capacity,
            INT64_NBYTES,
            "compact_i64",
        )
        self._compact_capacity = next_capacity

    def _ensure_segment_capacity(self, needed: int) -> None:
        if needed <= self._segment_capacity:
            return
        next_capacity = self._next_capacity(self._segment_capacity, needed)
        self._segments.extend([None] * (next_capacity - self._segment_capacity))
        self._segment_capacity = next_capacity

    def _next_capacity(self, current: int, needed: int) -> int:
        if needed <= current:
            return current
        if current == 0:
            return needed
        return max(needed, current * 2)


def metadata_nbytes(frame: TransportFrame) -> int:
    return (
        frame.frame_i32.nbytes
        + frame.entry_i32.nbytes
        + frame.payload_i32.nbytes
        + frame.shape_i32.nbytes
        + frame.global_shape_i64.nbytes
        + frame.compact_i64.nbytes
    )
