import asyncio

import pytest

from fluxway.common.async_close import AsyncCloseOnce


@pytest.mark.asyncio
@pytest.mark.parametrize("child_task", [False, True])
async def test_close_rejects_cleanup_waiting_for_itself(child_task):
    async def cleanup():
        if child_task:
            await asyncio.create_task(owner.close())
        else:
            await owner.close()

    owner = AsyncCloseOnce(cleanup)
    with pytest.raises(RuntimeError, match="cannot await itself"):
        await asyncio.wait_for(owner.close(), timeout=2)


@pytest.mark.asyncio
async def test_close_keeps_one_failure_after_caller_cancellation():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def cleanup():
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        raise ValueError("cleanup failed")

    owner = AsyncCloseOnce(cleanup)
    caller = asyncio.create_task(owner.close())
    await entered.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    release.set()
    for _ in range(2):
        with pytest.raises(ValueError, match="cleanup failed"):
            await owner.close()
    assert calls == 1
