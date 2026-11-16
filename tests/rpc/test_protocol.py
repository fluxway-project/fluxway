import numpy as np
import asyncio
import pytest
import ray
import threading

from fluxway.common.tensor_backend import ensure_cpu_serializable
from fluxway.rpc.protocol import (
    AxonDataPlane,
    DataPlaneRegistry,
    PayloadDeviceClass,
    RayActorDataPlane,
    _await_ray_object_ref,
    TransportFrameEnvelope,
    payload_device_class,
    validate_payload_iov_device_class,
)
from fluxway.transport.manifest import (
    FrameFlag,
    PayloadRole,
    TensorEntryKind,
    TransportFrameBuilder,
    TransportFrameHeader,
)


def _make_header(payload_segments: tuple[np.ndarray, ...]) -> TransportFrameHeader:
    builder = TransportFrameBuilder(
        step=0,
        sender_spmd_code=0,
        sender_rank=0,
        sender_world_size=1,
        flags=FrameFlag.TENSOR_TRANSFER,
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=0,
        layout_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        coherence_epoch=0,
    )
    for payload in payload_segments:
        builder.add_payload(
            entry_index,
            PayloadRole.VALUES,
            payload,
            valid_count=int(payload.shape[0]),
        )
    return builder.build().header


class _CudaArrayInterfaceOnlyPayload:
    dtype = np.dtype(np.float32)
    __cuda_array_interface__ = {
        "shape": (1,),
        "typestr": "<f4",
        "data": (1, False),
        "version": 3,
    }


class _DLPackOnlyPayload:
    dtype = np.dtype(np.float32)
    shape = (1,)

    def __dlpack__(self, stream=None):
        return object()


class _DLPackCpuPayload(_DLPackOnlyPayload):
    def __dlpack_device__(self):
        return (1, 0)


class _DLPackCudaPayload(_DLPackOnlyPayload):
    def __dlpack_device__(self):
        return (2, 0)


def test_registry():
    class MockHandle:
        pass

    handle = MockHandle()
    dp = DataPlaneRegistry.create("ray", handle, sender_spmd_code=0)
    assert isinstance(dp, RayActorDataPlane)


def test_registry_creates_axon_data_plane():
    class MockHandle:
        payload_device_class = "cpu"
        completion_mode = "receiver_success"

        async def invoke_transport_frames(
            self, frames, payload_device_class, sender_spmd_code
        ):
            return True

        async def invoke_control(self, op_name, step, names, values, sender_spmd_code):
            return True

    dp = DataPlaneRegistry.create("axon", MockHandle(), sender_spmd_code=0)
    assert isinstance(dp, AxonDataPlane)


def test_axon_data_plane_requires_explicit_receiver_success_completion_mode():
    class MissingCompletionMode:
        payload_device_class = "cpu"

    with pytest.raises(TypeError, match="completion_mode"):
        AxonDataPlane(MissingCompletionMode(), sender_spmd_code=0)

    class SendEnqueuedHandle:
        payload_device_class = "cpu"
        completion_mode = "send_enqueued"

    with pytest.raises(ValueError, match="receiver_success"):
        AxonDataPlane(SendEnqueuedHandle(), sender_spmd_code=0)


def test_ensure_cpu_serializable():
    # Test 1: normal array (should pass through)
    a = np.array([1, 2, 3])
    b = ensure_cpu_serializable(a)
    assert a is b

    # Test 2: memory view without owndata
    # create a large array and take a view
    base = np.zeros(1000)
    view = base[10:20]

    assert not view.flags.owndata

    fixed = ensure_cpu_serializable(view)

    assert fixed is not view
    assert fixed.flags.owndata
    assert np.array_equal(fixed, view)


@pytest.mark.asyncio
async def test_ray_actor_data_plane_sends_header_and_payload_buffers(monkeypatch):
    class RemoteMethod:
        def __init__(self):
            self.args = None

        def remote(self, *args):
            self.args = args
            return ray.ObjectRef.from_random()

    class MockHandle:
        def __init__(self):
            self.receive_transport_frames = RemoteMethod()

    handle = MockHandle()
    monkeypatch.setattr(ray, ray.get.__name__, lambda ref: True)
    adapter = RayActorDataPlane(handle, sender_spmd_code=7)
    ids = np.array([2**40], dtype=np.uint64)
    emb = np.arange(4, dtype=np.float32)
    payload_segments = (ids, emb)
    header = _make_header(payload_segments)
    await adapter.receive_transport_frames(
        (
            TransportFrameEnvelope(
                header=header,
                payload_segments=payload_segments,
            ),
        )
    )

    args = handle.receive_transport_frames.args
    assert args is not None
    frames = args[0]
    assert len(frames) == 1
    assert frames[0].header is header
    assert isinstance(frames[0].header, TransportFrameHeader)
    assert isinstance(frames[0].payload_segments[0], np.ndarray)
    assert isinstance(frames[0].payload_segments[1], np.ndarray)
    assert frames[0].payload_segments[0].dtype == np.uint64
    assert not any(
        isinstance(arg, (str, list, dict))
        for frame in frames
        for arg in frame.payload_segments
    )


def test_payload_device_class_detects_cpu_cuda_and_dlpack_cpu():
    assert payload_device_class(np.zeros(1, dtype=np.float32)) is PayloadDeviceClass.CPU
    assert payload_device_class(_DLPackCudaPayload()) is PayloadDeviceClass.CUDA
    assert payload_device_class(_DLPackCpuPayload()) is PayloadDeviceClass.CPU


def test_payload_device_class_rejects_unverifiable_dlpack_payload():
    for payload in (_DLPackOnlyPayload(), _CudaArrayInterfaceOnlyPayload()):
        with pytest.raises(TypeError, match="__dlpack_device__"):
            payload_device_class(payload)


def test_payload_device_class_rejects_non_tensor_dlpack_object():
    class DLPackObjectWithoutTensorMetadata:
        def __dlpack__(self, stream=None):
            return object()

        def __dlpack_device__(self):
            return (1, 0)

    with pytest.raises(TypeError, match="dtype and shape"):
        payload_device_class(DLPackObjectWithoutTensorMetadata())


def test_payload_device_class_does_not_export_dlpack_capsule():
    class InstrumentedDLPackTensor:
        dtype = np.dtype(np.float32)
        shape = (1,)

        def __init__(self):
            self.device_queries = 0

        def __dlpack__(self, stream=None):
            raise AssertionError("classification must not export a DLPack capsule")

        def __dlpack_device__(self):
            self.device_queries += 1
            return (1, 0)

    payload = InstrumentedDLPackTensor()

    assert payload_device_class(payload) is PayloadDeviceClass.CPU
    assert payload.device_queries == 1


def test_validate_payload_iov_device_class_rejects_mixed_payloads():
    with pytest.raises(ValueError, match="must share one device class"):
        validate_payload_iov_device_class(
            (np.zeros(1, dtype=np.float32), _DLPackCudaPayload())
        )


def test_validate_payload_iov_device_class_rejects_destination_mismatch():
    with pytest.raises(ValueError, match="source=cpu destination=cuda"):
        validate_payload_iov_device_class(
            (np.zeros(1, dtype=np.float32),),
            expected_device_class=PayloadDeviceClass.CUDA,
        )


@pytest.mark.asyncio
async def test_axon_data_plane_validates_and_invokes_flat_payload_iovs():
    class MockHandle:
        payload_device_class = "cpu"
        completion_mode = "receiver_success"

        def __init__(self):
            self.args = None

        async def invoke_transport_frames(
            self, frames, payload_device_class, sender_spmd_code
        ):
            self.args = (frames, payload_device_class, sender_spmd_code)
            return True

    handle = MockHandle()
    adapter = AxonDataPlane(handle, sender_spmd_code=11)
    ids = np.array([2**40], dtype=np.uint64)
    emb = np.arange(4, dtype=np.float32)
    payload_segments = (ids, emb)
    header = _make_header(payload_segments)

    result = await adapter.receive_transport_frames(
        (TransportFrameEnvelope(header=header, payload_segments=payload_segments),)
    )

    assert result.result is True
    assert handle.args is not None
    assert handle.args[0][0].header is header
    assert handle.args[1] == "cpu"
    assert handle.args[2] == 11
    assert handle.args[0][0].payload_segments[0] is ids
    assert handle.args[0][0].payload_segments[1] is emb


@pytest.mark.asyncio
async def test_axon_data_plane_rejects_payload_before_invoking_handle():
    class MockHandle:
        payload_device_class = "cuda"
        completion_mode = "receiver_success"

        def __init__(self):
            self.called = False

        async def invoke_transport_frames(
            self, frames, payload_device_class, sender_spmd_code
        ):
            self.called = True
            return True

    handle = MockHandle()
    adapter = AxonDataPlane(handle, sender_spmd_code=0)
    payload_segments = (
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="source=cpu destination=cuda"):
        await adapter.receive_transport_frames(
            (
                TransportFrameEnvelope(
                    header=_make_header(payload_segments),
                    payload_segments=payload_segments,
                ),
            )
        )
    assert handle.called is False


@pytest.mark.asyncio
async def test_ray_object_ref_wait_runs_off_owner_loop(monkeypatch):
    owner_thread = threading.get_ident()
    get_threads: list[int] = []

    def fake_get(ref):
        assert isinstance(ref, ray.ObjectRef)
        get_threads.append(threading.get_ident())
        return True

    monkeypatch.setattr(ray, ray.get.__name__, fake_get)
    assert await _await_ray_object_ref(ray.ObjectRef.from_random()) is True
    assert get_threads[0] != owner_thread


@pytest.mark.asyncio
async def test_ray_actor_data_plane_receive_control_awaits_object_ref(monkeypatch):
    result = {"accepted": True}

    class RemoteMethod:
        def remote(self, *args):
            assert args == ("update", 3, ["weight"], [result])
            return ray.ObjectRef.from_random()

    class MockHandle:
        receive_control = RemoteMethod()

    monkeypatch.setattr(ray, ray.get.__name__, lambda ref: result)
    adapter = RayActorDataPlane(MockHandle(), sender_spmd_code=0)

    assert await adapter.receive_control("update", 3, ["weight"], [result]) is result


@pytest.mark.asyncio
async def test_ray_actor_data_plane_receive_control_propagates_ray_failure(monkeypatch):
    failure = RuntimeError("control RPC failed")

    class RemoteMethod:
        def remote(self, *_args):
            return ray.ObjectRef.from_random()

    class MockHandle:
        receive_control = RemoteMethod()

    def fail_get(_ref):
        raise failure

    monkeypatch.setattr(ray, ray.get.__name__, fail_get)
    adapter = RayActorDataPlane(MockHandle(), sender_spmd_code=0)

    with pytest.raises(RuntimeError) as error:
        await adapter.receive_control("update", 3, [], [])

    assert error.value is failure


@pytest.mark.asyncio
async def test_ray_actor_data_plane_receive_control_cancellation_cancels_object_ref(
    monkeypatch,
):
    wait_started = threading.Event()
    release_wait = threading.Event()
    ref = ray.ObjectRef.from_random()
    cancelled_refs = []

    class RemoteMethod:
        def remote(self, *_args):
            return ref

    class MockHandle:
        receive_control = RemoteMethod()

    def blocking_get(received_ref):
        assert received_ref is ref
        wait_started.set()
        assert release_wait.wait(timeout=5)
        return True

    monkeypatch.setattr(ray, ray.get.__name__, blocking_get)
    monkeypatch.setattr(
        ray,
        ray.cancel.__name__,
        lambda received_ref, force: cancelled_refs.append((received_ref, force)),
    )
    adapter = RayActorDataPlane(MockHandle(), sender_spmd_code=0)
    task = asyncio.create_task(adapter.receive_control("update", 3, [], []))

    try:
        assert await asyncio.to_thread(wait_started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert cancelled_refs == [(ref, False)]
    finally:
        release_wait.set()
