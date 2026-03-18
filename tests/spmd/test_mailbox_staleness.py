import numpy as np
import pytest

from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.common.types.configs import TolerantBarrierConfig

TUUID_A = 1001
TUUID_X = 1002


@pytest.mark.asyncio
async def test_mailbox_staleness_tensor_fallback():
    """Verify that a step can be satisfied by stale tensor data within the staleness window."""
    mb = WorkerMailbox()

    mb.declare_step("op", step=0, expected_tensors={TUUID_A: 1}, expected_states=[])
    await mb.deposit_shards("op", [TUUID_A], 0, "sender0", 0, 1, [np.array([1.0])], [1])
    assert mb.is_step_ready("op", 0)

    mb.declare_step("op", step=1, expected_tensors={TUUID_A: 1}, expected_states=[])
    await mb.deposit_shards("op", [TUUID_A], 1, "sender0", 0, 1, [np.array([2.0])], [1])
    assert mb.is_step_ready("op", 1)

    config = TolerantBarrierConfig(
        tensor_staleness={TUUID_A: 2}, retrieval_policy="latest"
    )
    mb.declare_step(
        "op",
        step=3,
        expected_tensors={TUUID_A: 1},
        expected_states=[],
        barrier_config=config,
    )

    assert mb.is_step_ready("op", 3)

    shards = await mb.get_tensor_shards("op", TUUID_A, 3)
    assert len(shards) == 1
    np.testing.assert_array_equal(shards[("sender0", 0)][0].flat_values, [2.0])


@pytest.mark.asyncio
async def test_mailbox_staleness_state_fallback():
    """Verify that a step can be satisfied by stale state data."""
    mb = WorkerMailbox()

    mb.declare_step("op", step=5, expected_tensors={}, expected_states=["state"])
    await mb.deposit_states("op", 5, ["state"], [{"val": 5}])
    assert mb.is_step_ready("op", 5)

    config = TolerantBarrierConfig(
        state_staleness={"state": 3}, retrieval_policy="oldest"
    )
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

    await mb.deposit_states("op", 1, ["s"], ["val1"])
    mb.declare_step("op", step=1, expected_tensors={}, expected_states=["s"])

    config = TolerantBarrierConfig(state_staleness={"s": 1})
    mb.declare_step(
        "op", step=2, expected_tensors={}, expected_states=["s"], barrier_config=config
    )

    await mb.clear_step("op", 2)
    assert not await mb.has_state("op", "s", 1)


@pytest.mark.asyncio
async def test_mailbox_future_resolution():
    """Verify that depositing old data can satisfy a future tolerant step."""
    mb = WorkerMailbox()

    config = TolerantBarrierConfig(tensor_staleness={TUUID_X: 1})
    mb.declare_step(
        "op",
        step=2,
        expected_tensors={TUUID_X: 1},
        expected_states=[],
        barrier_config=config,
    )

    assert not mb.is_step_ready("op", 2)

    await mb.deposit_shards(
        "op", [TUUID_X], 1, "sender0", 0, 1, [np.array([10.0])], [1]
    )

    assert mb.is_step_ready("op", 2)
    shards = await mb.get_tensor_shards("op", TUUID_X, 2)
    assert len(shards) == 1
    np.testing.assert_array_equal(shards[("sender0", 0)][0].flat_values, [10.0])
