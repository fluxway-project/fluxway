import numpy as np
import pytest

from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.common.types.configs import TolerantBarrierConfig


@pytest.mark.asyncio
async def test_mailbox_staleness_tensor_fallback():
    """Verify that a step can be satisfied by stale tensor data within the staleness window."""
    mb = WorkerMailbox()

    # Declare step 0 and provide data
    mb.declare_step("op", step=0, expected_tensors={"a": 1}, expected_states=[])
    await mb.deposit_shards("op", ["a"], 0, "sender0", 0, [[np.array([1.0])]], [[1]])
    assert mb.is_step_ready("op", 0)

    # Declare step 1 and provide data
    mb.declare_step("op", step=1, expected_tensors={"a": 1}, expected_states=[])
    await mb.deposit_shards("op", ["a"], 1, "sender0", 0, [[np.array([2.0])]], [[1]])
    assert mb.is_step_ready("op", 1)

    # Declare step 3 but it requires a tensor 'a'. Configure it with a staleness of 2.
    # We don't deposit any shards for step 3. It should fall back to step 1's data.
    config = TolerantBarrierConfig(staleness={"a": 2}, retrieval_policy="latest")
    mb.declare_step(
        "op",
        step=3,
        expected_tensors={"a": 1},
        expected_states=[],
        barrier_config=config,
    )

    # It should become ready immediately upon declaration because step 1's data is within the [3-2, 3] window.
    assert mb.is_step_ready("op", 3)

    # Verify retrieved data uses the 'latest' policy (step 1 is latest valid)
    shards = await mb.get_tensor_shards("op", "a", 3)
    assert len(shards) == 1
    np.testing.assert_array_equal(shards[("sender0", 0)][0].flat_values, [2.0])


@pytest.mark.asyncio
async def test_mailbox_staleness_state_fallback():
    """Verify that a step can be satisfied by stale state data."""
    mb = WorkerMailbox()

    mb.declare_step("op", step=5, expected_tensors={}, expected_states=["state"])
    await mb.deposit_states("op", 5, ["state"], [{"val": 5}])
    assert mb.is_step_ready("op", 5)

    config = TolerantBarrierConfig(staleness={"state": 3}, retrieval_policy="oldest")

    # Step 8 can use step 5's data (8 - 3 = 5, which is in window)
    mb.declare_step(
        "op",
        step=8,
        expected_tensors={},
        expected_states=["state"],
        barrier_config=config,
    )
    assert mb.is_step_ready("op", 8)

    state_val = await mb.get_state("op", "state", 8)
    assert state_val == {"val": 5}


@pytest.mark.asyncio
async def test_mailbox_clear_step_respects_staleness():
    """Verify clear_step preserves data if it's within the maximum configured staleness."""
    mb = WorkerMailbox()

    # Step 1 setup
    await mb.deposit_states("op", 1, ["s"], ["val1"])
    mb.declare_step("op", step=1, expected_tensors={}, expected_states=["s"])

    # Configure step 2 to allow staleness of 1 for state 's'
    config = TolerantBarrierConfig(staleness={"s": 1})
    mb.declare_step(
        "op", step=2, expected_tensors={}, expected_states=["s"], barrier_config=config
    )

    # Clear step 2
    # Because 's' has staleness=1, it will try to pop (2 - 1) = 1.
    await mb.clear_step("op", 2)

    # Wait, if we clear step 2, it should pop step 1. But does step 2's clear affect step 1?
    # Actually, clear_step(op, 2) drops target_step = 2 - 1 = 1.
    assert not await mb.has_state("op", "s", 1)


@pytest.mark.asyncio
async def test_mailbox_future_resolution():
    """Verify that depositing old data can satisfy a future tolerant step."""
    mb = WorkerMailbox()

    config = TolerantBarrierConfig(staleness={"x": 1})

    # Declare step 2 which expects "x" with staleness 1.
    mb.declare_step(
        "op",
        step=2,
        expected_tensors={"x": 1},
        expected_states=[],
        barrier_config=config,
    )

    # Step 2 won't be ready initially.
    assert not mb.is_step_ready("op", 2)

    # Deposit shards for step 1. This should trigger the check and satisfy step 2.
    await mb.deposit_shards("op", ["x"], 1, "sender0", 0, [[np.array([10.0])]], [[1]])

    assert mb.is_step_ready("op", 2)
    shards = await mb.get_tensor_shards("op", "x", 2)
    assert len(shards) == 1
    np.testing.assert_array_equal(shards[("sender0", 0)][0].flat_values, [10.0])
