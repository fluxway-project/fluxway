"""
Tests for intra-SPMD and inter-SPMD asynchronous execution mechanisms.

These tests verify:
1. max_worker_step_skew: Limits step difference between workers in the same SPMD.
2. max_inflight_steps: Limits step difference between connected SPMDs in the graph.
"""

from fluxway.spmd.driver.synchronizer.worker_sync import WorkerSynchronizer
from fluxway.control_plane.async_manager.manager import AsyncStepManager
from fluxway.common.types.configs import (
    Graph,
    Node,
    Edge,
    SPMDConfig,
    OperatorMeta,
    AsyncStepConfig,
)


# =============================================================================
# 1. Intra-SPMD: WorkerSynchronizer Tests
# =============================================================================


def test_worker_sync_skew_limit():
    """
    Verify that WorkerSynchronizer blocks workers that exceed max_worker_step_skew.
    """
    num_workers = 3
    max_diff = 2
    sync = WorkerSynchronizer(
        spmd_id="test_spmd", num_workers=num_workers, max_worker_step_skew=max_diff
    )

    blocked_workers = set()
    unblocked_workers = set()

    def block_cb(rank, op):
        blocked_workers.add(rank)
        if rank in unblocked_workers:
            unblocked_workers.remove(rank)

    def unblock_cb(rank, op):
        unblocked_workers.add(rank)
        if rank in blocked_workers:
            blocked_workers.remove(rank)

    sync.set_callbacks(block_worker_op=block_cb, unblock_worker_op=unblock_cb)

    op = "compute"

    # 1. All workers at step 0
    sync.report_worker_op_completed(0, op, 0)
    sync.report_worker_op_completed(1, op, 0)
    sync.report_worker_op_completed(2, op, 0)
    assert not blocked_workers
    assert sync.get_spmd_step(op) == 0

    # 2. Worker 0 and 1 move to step 2
    sync.report_worker_op_completed(0, op, 1)
    sync.report_worker_op_completed(0, op, 2)
    sync.report_worker_op_completed(1, op, 1)
    sync.report_worker_op_completed(1, op, 2)
    # diff is 2 - 0 = 2, which is NOT > max_diff (2)
    assert not blocked_workers

    # 3. Worker 0 moves to step 3
    sync.report_worker_op_completed(0, op, 3)
    # diff is 3 - 0 = 3, which is > max_diff (2)
    # Worker 0 (step 3) and possibly 1 (step 2) are ahead?
    assert 0 in blocked_workers
    assert 1 not in blocked_workers  # Worker 1 is at 2, not >= 3

    # 4. Worker 2 catches up to step 1
    sync.report_worker_op_completed(2, op, 1)
    # diff is 3 - 1 = 2, which is <= max_diff.
    # But unblock only happens when diff <= max_diff // 2 (i.e. 1)
    assert 0 in blocked_workers

    # 5. Worker 2 catches up to step 2
    sync.report_worker_op_completed(2, op, 2)
    # diff is 3 - 2 = 1, which is <= 2 // 2. Unblock should trigger.
    assert 0 not in blocked_workers
    assert 0 in unblocked_workers

    # 6. SPMD step should be 2 (minimum of all workers)
    assert sync.get_spmd_step(op) == 2


def test_worker_sync_controller_lock():
    """Verify that controller-requested locks work alongside drift locks."""
    sync = WorkerSynchronizer(
        spmd_id="test_spmd", num_workers=2, max_worker_step_skew=10
    )

    blocked = []
    sync.set_callbacks(block_worker_op=lambda r, o: blocked.append(r))

    sync.block_spmd_op("train")
    assert len(blocked) == 2
    assert sync.is_op_locked("train")


# =============================================================================
# 2. Inter-SPMD: AsyncStepManager Tests
# =============================================================================


def test_async_manager_edge_limit():
    """
    Verify that AsyncStepManager locks source SPMD when it gets too far ahead of target.
    """
    # Setup graph: A -> B
    op_a = "produce"
    op_b = "consume"

    # Node A
    meta_a = OperatorMeta(name=op_a)
    node_a = Node(name="A", spmd_config=SPMDConfig(operator_metas={op_a: meta_a}))

    # Node B with max_inflight_steps = 2
    meta_b = OperatorMeta(name=op_b, max_inflight_steps=2)
    node_b = Node(name="B", spmd_config=SPMDConfig(operator_metas={op_b: meta_b}))

    edge = Edge(source_node="A", target_node="B", source_op=op_a, target_op=op_b)
    graph = Graph(nodes={"A": node_a, "B": node_b}, edges=[edge])

    manager = AsyncStepManager(graph=graph)

    # 1. Both at step 0
    manager.report_op_completed("A", op_a, 0)
    manager.report_op_completed("B", op_b, 0)
    actions = manager.check_and_enforce_limits()
    assert not actions

    # 2. A moves to step 2
    manager.report_op_completed("A", op_a, 2)
    actions = manager.check_and_enforce_limits()
    # diff = 2 - 0 = 2, which is NOT > limit (2)
    assert not actions

    # 3. A moves to step 3
    manager.report_op_completed("A", op_a, 3)
    actions = manager.check_and_enforce_limits()
    # diff = 3 - 0 = 3, which IS > limit (2)
    # Should lock A.produce
    assert ("A", op_a, True) in actions
    assert manager.is_op_locked("A", op_a)

    # 4. B moves to step 1
    manager.report_op_completed("B", op_b, 1)
    actions = manager.check_and_enforce_limits()
    # diff = 3 - 1 = 2. Still > limit // 2 (1). Stay locked.
    assert not actions
    assert manager.is_op_locked("A", op_a)

    # 5. B moves to step 2
    manager.report_op_completed("B", op_b, 2)
    # diff = 3 - 2 = 1. <= limit // 2 (1). Unlock!
    manager._config.min_lock_duration = 0
    actions = manager.check_and_enforce_limits()
    assert ("A", op_a, False) in actions
    assert not manager.is_op_locked("A", op_a)


def test_async_manager_global_limit():
    """Verify that AsyncStepManager locks all ops when max_steps is reached."""
    meta = OperatorMeta(name="run")
    node = Node(name="N", spmd_config=SPMDConfig(operator_metas={"run": meta}))
    graph = Graph(nodes={"N": node})

    config = AsyncStepConfig(max_steps=5)
    manager = AsyncStepManager(graph=graph, config=config)

    manager.report_op_completed("N", "run", 4)
    actions = manager.check_and_enforce_limits()
    assert not actions

    manager.report_op_completed("N", "run", 5)
    actions = manager.check_and_enforce_limits()
    assert ("N", "run", True) in actions
    assert manager.is_op_locked("N", "run")
    assert manager.is_all_completed()
