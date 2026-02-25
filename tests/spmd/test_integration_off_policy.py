import numpy as np
import pytest

from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.common.types.configs import TolerantBarrierConfig


@pytest.mark.asyncio
async def test_off_policy_staleness_integration():
    """Simulate off-policy training where one SPMD (e.g. sparse_embedding) provides stale data."""
    mb = WorkerMailbox()

    # Configuration:
    # Sparse embedding can be up to 2 steps stale.
    # Batch data is the anchor tensor (staleness 0, drives the step).
    config = TolerantBarrierConfig(
        staleness={"sparse_embedding": 2, "batch_data": 0},
        anchor_tensor="batch_data",
        retrieval_policy="latest",
    )

    # 1. Step 1 arrives normally
    mb.declare_step(
        "compute",
        step=1,
        expected_tensors={"sparse_embedding": 1, "batch_data": 1},
        expected_states=[],
        barrier_config=config,
    )
    await mb.deposit_shards("compute", "sparse_embedding", 1, 0, [np.array([0.1])], [1])
    await mb.deposit_shards("compute", "batch_data", 1, 0, [np.array([10])], [1])
    assert mb.is_step_ready("compute", 1), "Step 1 should be ready"

    # 2. Step 2 arrives normally
    mb.declare_step(
        "compute",
        step=2,
        expected_tensors={"sparse_embedding": 1, "batch_data": 1},
        expected_states=[],
        barrier_config=config,
    )
    await mb.deposit_shards("compute", "sparse_embedding", 2, 0, [np.array([0.2])], [1])
    await mb.deposit_shards("compute", "batch_data", 2, 0, [np.array([20])], [1])
    assert mb.is_step_ready("compute", 2), "Step 2 should be ready"

    # 3. Simulate step 3, where sparse embedding is delayed.
    mb.declare_step(
        "compute",
        step=3,
        expected_tensors={"sparse_embedding": 1, "batch_data": 1},
        expected_states=[],
        barrier_config=config,
    )

    # We only deposit batch data for step 3
    await mb.deposit_shards("compute", "batch_data", 3, 0, [np.array([30])], [1])

    # Step 3 should spring ready instantly! Because sparse_embedding from step 2 is 1 step stale, <= 2.
    assert mb.is_step_ready("compute", 3), (
        "Step 3 should be ready via stale data fallback"
    )

    # Verify retrieved data for step 3
    shards_emb = await mb.get_tensor_shards("compute", "sparse_embedding", 3)
    np.testing.assert_array_equal(shards_emb[0].flat_values, [0.2])

    shards_batch = await mb.get_tensor_shards("compute", "batch_data", 3)
    np.testing.assert_array_equal(shards_batch[0].flat_values, [30.0])

    # 4. Simulate step 4, where sparse embedding is delayed again.
    mb.declare_step(
        "compute",
        step=4,
        expected_tensors={"sparse_embedding": 1, "batch_data": 1},
        expected_states=[],
        barrier_config=config,
    )
    await mb.deposit_shards("compute", "batch_data", 4, 0, [np.array([40])], [1])

    # Step 4 should also spring ready! Because sparse_embedding from step 2 is 2 steps stale, <= 2.
    assert mb.is_step_ready("compute", 4), (
        "Step 4 should be ready via stale data fallback"
    )

    shards_emb_4 = await mb.get_tensor_shards("compute", "sparse_embedding", 4)
    np.testing.assert_array_equal(shards_emb_4[0].flat_values, [0.2])

    # 5. Simulate step 5, where sparse embedding is delayed YET again.
    mb.declare_step(
        "compute",
        step=5,
        expected_tensors={"sparse_embedding": 1, "batch_data": 1},
        expected_states=[],
        barrier_config=config,
    )
    await mb.deposit_shards("compute", "batch_data", 5, 0, [np.array([50])], [1])

    # Step 5 should NOT be ready! Because sparse_embedding from step 2 is 3 steps stale, which > 2.
    assert not mb.is_step_ready("compute", 5), (
        "Step 5 should NOT be ready because data is too stale"
    )

    # Once we deposit step 3's embedding, it becomes ready
    await mb.deposit_shards("compute", "sparse_embedding", 3, 0, [np.array([0.3])], [1])
    assert mb.is_step_ready("compute", 5), (
        "Step 5 should be ready after depositing new embedding data"
    )
