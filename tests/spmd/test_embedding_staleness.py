"""
Tests for asynchronous Embedding training with TolerantBarrier.

This test simulates a real-world scenario where:
1. An Embedding Forward operator runs ahead of the Optimizer.
2. We use TolerantBarrier(staleness=2) to allow Forward(t+2) to run if Weights(t) is ready.
3. We verify that Forward(t+3) is BLOCKED if Weights(t) hasn't arrived.
"""

import pytest
import numpy as np
from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.common.types.configs import TolerantBarrierConfig


@pytest.mark.asyncio
async def test_embedding_forward_staleness_logic():
    """
    Scenario:
    - Forward step t depends on Weights.
    - Config: TolerantBarrier with staleness=2.
    - Forward(0), Forward(1), Forward(2) should be allowed if Weights(0) is available.
    - Forward(3) should be blocked until Weights(1) arrives.
    """
    mb = WorkerMailbox()
    ready_steps = []

    def on_ready(op, step):
        ready_steps.append(step)

    mb.set_ready_callback(on_ready)

    # Define an operator with TolerantBarrier
    # Staleness = 2 means for step T, we need data from at least step T-2.
    barrier_cfg = TolerantBarrierConfig(state_staleness={"weights": 2})

    # 1. Initialize Weights at step 0
    await mb.deposit_states("forward", step=0, names=["weights"], values=[np.zeros(10)])

    # 2. Declare steps 0 to 4 for the forward operator
    for s in range(5):
        mb.declare_step(
            "forward",
            step=s,
            expected_tensors={},
            expected_states=["weights"],
            barrier_config=barrier_cfg,
        )

    # Check Step 0: Should be ready (needs weights step >= 0-2=0)
    assert mb.is_step_ready("forward", 0)

    # Check Step 1: Should be ready (needs weights step >= 1-2=-1, so 0 is fine)
    assert mb.is_step_ready("forward", 1)

    # Check Step 2: Should be ready (needs weights step >= 2-2=0)
    assert mb.is_step_ready("forward", 2)

    # Check Step 3: Should NOT be ready (needs weights step >= 3-2=1)
    # Even though Weights(0) is there, it's too stale (staleness=3 > 2)
    assert not mb.is_step_ready("forward", 3)

    # 3. Supply Weights(1)
    await mb.deposit_states("forward", step=1, names=["weights"], values=[np.ones(10)])

    # Now Step 3 should be ready (min required step is 3-2=1, and we have 1)
    assert mb.is_step_ready("forward", 3)

    # Step 4 should NOT be ready (needs weights step >= 4-2=2)
    assert not mb.is_step_ready("forward", 4)

    assert ready_steps == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_api_renames_in_synchronizer():
    """Verify that SPMDSynchronizer blocks workers correctly with MaxSkewSyncPolicy."""
    from fluxway.spmd.driver.synchronizer import SPMDSynchronizer, MaxSkewSyncPolicy

    sync = SPMDSynchronizer("test", num_workers=2, sync_policy=MaxSkewSyncPolicy(1))
    blocked = []
    sync.set_callbacks(block_worker_op=lambda r, op: blocked.append(r))

    # Worker 0 completes step 0, 1
    sync.report_worker_op_completed(0, "op", 0)
    sync.report_worker_op_completed(0, "op", 1)

    # Worker 1 is still at -1 (implicit). Diff = 1 - (-1)? Wait, initial is 0.
    # Worker 1 hasn't reported.
    # Logic in sync: steps = [worker_steps[r].get(op, 0) ...]
    # So steps = [1, 0]. Diff = 1. Not > 1.
    assert not blocked

    # Worker 0 completes step 2. Diff = 2 - 0 = 2. 2 > 1.
    sync.report_worker_op_completed(0, "op", 2)
    assert 0 in blocked
