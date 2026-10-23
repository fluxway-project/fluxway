from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable

import pytest

from fluxway.common.ingestion import SourceBindingIdentity
from fluxway.control_plane.ingestion import TailLayoutMode
from fluxway.control_plane.ingestion.adapter import SourceScanBinding
from fluxway.control_plane.ingestion.planning_component import (
    IngestionLayoutPlanningComponent,
)


class ResourceAdapter:
    def __init__(self) -> None:
        self.closed = 0

    def scan_next(self, cursor, budget):
        raise AssertionError("construction must not read data")

    def advance_cursor(self, cursor, records):
        raise AssertionError("construction must not advance the source")

    def close(self) -> None | Awaitable[None]:
        self.closed += 1


def binding(adapter):
    return SourceScanBinding(
        adapter=adapter,
        identity=SourceBindingIdentity("configuration", "source", "finite", "native"),
        initial_cursor_factory=lambda: None,
        topology=object(),
        tail_layout_mode=TailLayoutMode.DISCARD_REMAINDER,
    )


@pytest.mark.asyncio
async def test_concurrent_binding_construction_is_owned_before_scan_activation():
    component = IngestionLayoutPlanningComponent("resource-owner")
    adapter = ResourceAdapter()
    calls = 0

    async def build():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return binding(adapter)

    first, second = await asyncio.gather(
        component.open_source_binding(build), component.open_source_binding(build)
    )
    assert first is second
    assert calls == 1
    await component.close()
    await component.close()
    assert adapter.closed == 1


@pytest.mark.asyncio
async def test_cancelled_binding_caller_does_not_abandon_created_resource():
    component = IngestionLayoutPlanningComponent("cancelled-resource-owner")
    adapter = ResourceAdapter()
    entered = threading.Event()
    release = threading.Event()

    async def build():
        entered.set()
        await asyncio.to_thread(release.wait)
        return binding(adapter)

    opening = asyncio.create_task(component.open_source_binding(build))
    await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=3)
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    closing = asyncio.create_task(component.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await asyncio.wait_for(closing, timeout=3)
    assert adapter.closed == 1


@pytest.mark.asyncio
async def test_native_async_close_runs_on_the_adapter_lane_and_is_shielded():
    component = IngestionLayoutPlanningComponent("async-resource-owner")
    entered = threading.Event()
    release = threading.Event()
    threads = []

    class Adapter(ResourceAdapter):
        async def close(self):
            threads.append(threading.get_ident())
            entered.set()
            await asyncio.to_thread(release.wait)
            self.closed += 1

    adapter = Adapter()

    def build():
        threads.append(threading.get_ident())
        return binding(adapter)

    await component.open_source_binding(build)
    closing = asyncio.create_task(component.close())
    await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=3)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    release.set()
    await asyncio.wait_for(component.close(), timeout=3)
    assert adapter.closed == 1
    assert threads[0] == threads[1] != threading.get_ident()


@pytest.mark.asyncio
async def test_close_failure_does_not_leave_the_adapter_lane_running():
    component = IngestionLayoutPlanningComponent("failed-resource-owner")

    class Adapter(ResourceAdapter):
        def close(self):
            self.closed += 1
            raise ValueError("source release failed")

    adapter = Adapter()
    await component.open_source_binding(lambda: binding(adapter))
    with pytest.raises(BaseExceptionGroup, match="source planning close failed"):
        await component.close()
    assert adapter.closed == 1
    with pytest.raises(RuntimeError, match="closing"):
        await component._adapter_lane.call(lambda: None)
    with pytest.raises(RuntimeError, match="closing"):
        await component.open_source_binding(lambda: binding(ResourceAdapter()))
