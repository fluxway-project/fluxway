import numpy as np
import pytest

from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.common.types.configs import TolerantBarrierConfig

TUUID_EMB = 2001
TUUID_BATCH = 2002


@pytest.mark.asyncio
async def test_off_policy_staleness_integration():
    """Simulate off-policy training where sparse_embedding can be up to 2 steps stale."""
    mb = WorkerMailbox()

    config = TolerantBarrierConfig(
        tensor_staleness={TUUID_EMB: 2, TUUID_BATCH: 0}, retrieval_policy="latest"
    )

    def declare(step):
        mb.declare_step(
            "compute",
            step=step,
            expected_tensors={TUUID_EMB: 1, TUUID_BATCH: 1},
            expected_states=[],
            barrier_config=config,
        )

    async def deposit(tuuid, step, val):
        await mb.deposit_shards(
            "compute", [tuuid], step, "A", 0, 1, [np.array([val])], [1]
        )

    # Step 1: both arrive
    declare(1)
    await deposit(TUUID_EMB, 1, 0.1)
    await deposit(TUUID_BATCH, 1, 10.0)
    assert mb.is_step_ready("compute", 1), "Step 1 should be ready"

    # Step 2: both arrive
    declare(2)
    await deposit(TUUID_EMB, 2, 0.2)
    await deposit(TUUID_BATCH, 2, 20.0)
    assert mb.is_step_ready("compute", 2), "Step 2 should be ready"

    # Step 3: sparse_embedding delayed, uses step 2 (1 step stale <=2)
    declare(3)
    await deposit(TUUID_BATCH, 3, 30.0)
    assert mb.is_step_ready("compute", 3), "Step 3 should be ready via stale fallback"

    shards_emb = await mb.get_tensor_shards("compute", TUUID_EMB, 3)
    flat = [s for sl in shards_emb.values() for s in sl]
    np.testing.assert_array_equal(flat[0].flat_values, [0.2])

    shards_batch = await mb.get_tensor_shards("compute", TUUID_BATCH, 3)
    flat_b = [s for sl in shards_batch.values() for s in sl]
    np.testing.assert_array_equal(flat_b[0].flat_values, [30.0])

    # Step 4: sparse_embedding delayed again, uses step 2 (2 steps stale <=2)
    declare(4)
    await deposit(TUUID_BATCH, 4, 40.0)
    assert mb.is_step_ready("compute", 4), "Step 4 should be ready via stale fallback"

    shards_emb_4 = await mb.get_tensor_shards("compute", TUUID_EMB, 4)
    flat4 = [s for sl in shards_emb_4.values() for s in sl]
    np.testing.assert_array_equal(flat4[0].flat_values, [0.2])

    # Step 5: step 2 is 3 steps stale >2 → NOT ready
    declare(5)
    await deposit(TUUID_BATCH, 5, 50.0)
    assert not mb.is_step_ready("compute", 5), "Step 5 should NOT be ready (too stale)"

    # Deposit step 3 embedding → step 5 becomes ready (3 steps stale <=2 from step 3)
    await deposit(TUUID_EMB, 3, 0.3)
    assert mb.is_step_ready("compute", 5), "Step 5 should be ready after new deposit"
