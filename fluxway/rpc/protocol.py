"""
Fluxway Data Plane RPC Protocol.
Used for high-performance peer-to-peer data transmission (Tensors).
Control plane operations (scheduling, heartbeats) still use native Ray.
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from enum import StrEnum
from inspect import isawaitable
from typing import Any, ClassVar

import ray

from fluxway.common.tensor_backend import ensure_cpu_serializable
from fluxway.common.tensor_protocol import DLPackTensor, dlpack_device
from fluxway.transport.manifest import TransportFrameHeader


@dataclass(frozen=True, slots=True)
class TransportFrameEnvelope:
    header: TransportFrameHeader
    payload_segments: tuple[DLPackTensor, ...]


@dataclass(frozen=True, slots=True)
class ReceiverIngressCompletion:
    """Carrier proof that receiver ingress completed for one frame batch."""

    result: Any
    receiver_accepted: bool = True
    producer_retention_released: bool = True

    def __post_init__(self) -> None:
        if type(self.receiver_accepted) is not bool:
            raise TypeError("receiver_accepted must be bool.")
        if type(self.producer_retention_released) is not bool:
            raise TypeError("producer_retention_released must be bool.")


class PayloadDeviceClass(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class AxonCompletionMode(StrEnum):
    RECEIVER_SUCCESS = "receiver_success"
    SEND_ENQUEUED = "send_enqueued"


_DLPACK_CPU_DEVICE_TYPE = 1
_DLPACK_CUDA_DEVICE_TYPE = 2


def _coerce_payload_device_class(value: PayloadDeviceClass | str) -> PayloadDeviceClass:
    if isinstance(value, PayloadDeviceClass):
        return value
    try:
        return PayloadDeviceClass(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported payload device class {value!r}.") from exc


def _coerce_axon_completion_mode(value: AxonCompletionMode | str) -> AxonCompletionMode:
    if isinstance(value, AxonCompletionMode):
        return value
    try:
        return AxonCompletionMode(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported Axon completion mode {value!r}.") from exc


def payload_device_class(payload_iov: Any) -> PayloadDeviceClass:
    device_type, _ = dlpack_device(payload_iov, name="Axon payload iov")
    if device_type == _DLPACK_CPU_DEVICE_TYPE:
        return PayloadDeviceClass.CPU
    if device_type == _DLPACK_CUDA_DEVICE_TYPE:
        return PayloadDeviceClass.CUDA
    raise TypeError(
        f"Unsupported DLPack device type {device_type} for Axon payload iov."
    )


def validate_payload_iov_device_class(
    payload_iovs: tuple[Any, ...],
    *,
    expected_device_class: PayloadDeviceClass | str | None = None,
) -> PayloadDeviceClass | None:
    expected = (
        None
        if expected_device_class is None
        else _coerce_payload_device_class(expected_device_class)
    )
    actual: PayloadDeviceClass | None = None
    for index, payload_iov in enumerate(payload_iovs):
        current = payload_device_class(payload_iov)
        if actual is None:
            actual = current
        elif current is not actual:
            raise ValueError(
                "Axon payload iovs must share one device class; "
                f"payload 0 is {actual.value}, payload {index} is {current.value}."
            )

    if expected is not None and actual is not None and actual is not expected:
        raise ValueError(
            "Axon payload iov source device class does not match destination "
            f"device class: source={actual.value} destination={expected.value}."
        )
    return expected if actual is None else actual


def _validated_transport_header(header: TransportFrameHeader) -> TransportFrameHeader:
    if not isinstance(header, TransportFrameHeader):
        raise TypeError(
            f"header must be TransportFrameHeader, got {type(header).__name__}."
        )
    header.validate()
    return header


async def _await_required(result: Any, method_name: str) -> Any:
    if not isawaitable(result):
        raise TypeError(f"Axon handle {method_name} must return an awaitable.")
    return await result


async def _await_ray_object_ref(ref: Any) -> Any:
    if not isinstance(ref, ray.ObjectRef):
        raise TypeError(
            f"Ray ObjectRef wait requires ObjectRef, got {type(ref).__name__}."
        )

    try:
        return await asyncio.to_thread(ray.get, ref)
    except asyncio.CancelledError:
        ray.cancel(ref, force=False)
        raise


class DataPlaneCallable(ABC):
    """
    Abstract base class providing a decoupled interface for data transmission.
    Allows replacing Ray's .remote() with other RPC libraries without changing worker logic.
    """

    @abstractmethod
    def __init__(self, handle: Any, sender_spmd_code: int):
        pass

    @abstractmethod
    async def receive_transport_frames(
        self,
        frames: tuple[TransportFrameEnvelope, ...],
    ) -> ReceiverIngressCompletion:
        """Send a batch and return the carrier's receiver-ingress proof."""
        pass

    @abstractmethod
    async def receive_control(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> Any:
        """
        Data Plane: Method for state data transmission.
        """
        pass


class RayActorDataPlane(DataPlaneCallable):
    """
    Ray-specific implementation of DataPlaneCallable.
    """

    def __init__(self, handle: Any, sender_spmd_code: int):
        self._handle = handle
        if not isinstance(sender_spmd_code, int) or sender_spmd_code < 0:
            raise ValueError(
                f"sender_spmd_code must be a non-negative int, got {sender_spmd_code!r}."
            )
        self._sender_spmd_code = sender_spmd_code

    async def receive_transport_frames(
        self,
        frames: tuple[TransportFrameEnvelope, ...],
    ) -> ReceiverIngressCompletion:
        return await self.receive_transport_frames_profiled(
            frames, profiler_recorder=None
        )

    async def receive_transport_frames_profiled(
        self,
        frames: tuple[TransportFrameEnvelope, ...],
        *,
        profiler_recorder: Any | None,
    ) -> ReceiverIngressCompletion:
        if not frames:
            raise ValueError("receive_transport_frames requires at least one frame.")
        payload_count = sum(len(frame.payload_segments) for frame in frames)

        def serialize_frames() -> tuple[TransportFrameEnvelope, ...]:
            return tuple(
                TransportFrameEnvelope(
                    header=_validated_transport_header(frame.header),
                    payload_segments=tuple(
                        ensure_cpu_serializable(payload)
                        for payload in frame.payload_segments
                    ),
                )
                for frame in frames
            )

        if profiler_recorder is None:
            cpu_frames = serialize_frames()
            result = await _await_ray_object_ref(
                self._handle.receive_transport_frames.remote(cpu_frames),
            )
            return ReceiverIngressCompletion(result)

        with profiler_recorder.data_plane_ray_payload_serialize(
            frame_count=len(frames),
            payload_count=payload_count,
        ):
            cpu_frames = serialize_frames()
        with profiler_recorder.data_plane_ray_rpc_submit(frame_count=len(frames)):
            ref = self._handle.receive_transport_frames.remote(cpu_frames)
        async with profiler_recorder.data_plane_ray_rpc_wait(frame_count=len(frames)):
            return ReceiverIngressCompletion(await _await_ray_object_ref(ref))

    async def receive_control(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> Any:
        ref = self._handle.receive_control.remote(op_name, step, names, values)
        return await _await_ray_object_ref(ref)

    def __repr__(self):
        return f"RayActorDataPlane(handle={self._handle})"


class LocalWorkerDataPlane(DataPlaneCallable):
    def __init__(self, handle: Any, sender_spmd_code: int):
        self._handle = handle
        if not isinstance(sender_spmd_code, int) or sender_spmd_code < 0:
            raise ValueError(
                f"sender_spmd_code must be a non-negative int, got {sender_spmd_code!r}."
            )
        if getattr(handle, "receive_transport_frames", None) is None:
            raise TypeError(
                "LocalWorkerDataPlane handle must expose receive_transport_frames."
            )
        if getattr(handle, "receive_control", None) is None:
            raise TypeError("LocalWorkerDataPlane handle must expose receive_control.")
        self._sender_spmd_code = sender_spmd_code

    async def receive_transport_frames(
        self,
        frames: tuple[TransportFrameEnvelope, ...],
    ) -> ReceiverIngressCompletion:
        if not frames:
            raise ValueError("receive_transport_frames requires at least one frame.")
        validated_frames = tuple(
            TransportFrameEnvelope(
                header=_validated_transport_header(frame.header),
                payload_segments=tuple(frame.payload_segments),
            )
            for frame in frames
        )
        result = self._handle.receive_transport_frames(validated_frames)
        if not isawaitable(result):
            raise TypeError(
                "LocalWorkerDataPlane receive_transport_frames must return an awaitable."
            )
        return ReceiverIngressCompletion(await result)

    async def receive_control(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> Any:
        result = self._handle.receive_control(op_name, step, names, values)
        if not isawaitable(result):
            raise TypeError(
                "LocalWorkerDataPlane receive_control must return an awaitable."
            )
        return await result

    def __repr__(self):
        return f"LocalWorkerDataPlane(handle={self._handle})"


class AxonDataPlane(DataPlaneCallable):
    def __init__(self, handle: Any, sender_spmd_code: int):
        self._handle = handle
        if not isinstance(sender_spmd_code, int) or sender_spmd_code < 0:
            raise ValueError(
                f"sender_spmd_code must be a non-negative int, got {sender_spmd_code!r}."
            )
        destination_device_class = getattr(handle, "payload_device_class", None)
        if destination_device_class is None:
            raise TypeError(
                "AxonDataPlane handle must expose payload_device_class for "
                "source/destination iov device validation."
            )
        self._sender_spmd_code = sender_spmd_code
        self._payload_device_class = _coerce_payload_device_class(
            destination_device_class
        )
        completion_mode = getattr(handle, "completion_mode", None)
        if completion_mode is None:
            raise TypeError(
                "AxonDataPlane handle must expose completion_mode="
                "'receiver_success' for Local Deposit Boundary semantics."
            )
        self._completion_mode = _coerce_axon_completion_mode(completion_mode)
        if self._completion_mode is not AxonCompletionMode.RECEIVER_SUCCESS:
            raise ValueError(
                "AxonDataPlane requires completion_mode='receiver_success'; "
                f"got {self._completion_mode.value!r}. send_enqueued hides receiver "
                "deposit/assemble failures from the current step."
            )

    async def receive_transport_frames(
        self,
        frames: tuple[TransportFrameEnvelope, ...],
    ) -> ReceiverIngressCompletion:
        if not frames:
            raise ValueError("receive_transport_frames requires at least one frame.")
        validated_frames = tuple(
            TransportFrameEnvelope(
                header=_validated_transport_header(frame.header),
                payload_segments=tuple(frame.payload_segments),
            )
            for frame in frames
        )
        payload_class = validate_payload_iov_device_class(
            tuple(
                payload
                for frame in validated_frames
                for payload in frame.payload_segments
            ),
            expected_device_class=self._payload_device_class,
        )
        invoke = getattr(self._handle, "invoke_transport_frames", None)
        if invoke is None:
            raise TypeError("AxonDataPlane handle must expose invoke_transport_frames.")
        assert payload_class is not None, (
            "payload_class must not be None when expected_device_class is provided"
        )
        result = await _await_required(
            invoke(
                validated_frames,
                payload_class.value,
                self._sender_spmd_code,
            ),
            "invoke_transport_frames",
        )
        return ReceiverIngressCompletion(result)

    async def receive_control(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> Any:
        invoke = getattr(self._handle, "invoke_control", None)
        if invoke is None:
            raise TypeError("AxonDataPlane handle must expose invoke_control.")
        return await _await_required(
            invoke(op_name, step, names, values, self._sender_spmd_code),
            "invoke_control",
        )

    def __repr__(self):
        return (
            "AxonDataPlane("
            f"handle={self._handle}, payload_device_class={self._payload_device_class.value}, "
            f"completion_mode={self._completion_mode.value})"
        )


class DataPlaneRegistry:
    """Registry for data plane implementations."""

    _registry: ClassVar[dict[str, type[DataPlaneCallable]]] = {}

    @classmethod
    def register(cls, name: str, impl_cls: type[DataPlaneCallable]) -> None:
        cls._registry[name] = impl_cls

    @classmethod
    def create(
        cls,
        transport_type: str,
        handle: Any,
        sender_spmd_code: int,
    ) -> DataPlaneCallable:
        impl_cls = cls._registry.get(transport_type)
        if impl_cls is None:
            raise ValueError(f"Unknown transport_type: {transport_type}")
        return impl_cls(handle, sender_spmd_code=sender_spmd_code)


# Register default Ray implementation
DataPlaneRegistry.register("ray", RayActorDataPlane)
DataPlaneRegistry.register("axon", AxonDataPlane)
