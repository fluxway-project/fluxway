"""
Tests for intra-SPMD and inter-SPMD asynchronous execution mechanisms.

These tests verify:
1. max_worker_step_skew: Limits step difference between workers in the same SPMD.
2. max_inflight_steps: Limits step difference between connected SPMDs in the graph.
"""

from fluxway.spmd.driver.synchronizer import SPMDSynchronizer, MaxSkewSyncPolicy
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
# 1. Intra-SPMD: SPMDSynchronizer Tests
# =============================================================================


def test_worker_sync_skew_limit():
    """
    Verify that SPMDSynchronizer blocks workers that exceed max_worker_step_skew.
    """
    num_workers = 3
    max_diff = 2
    sync = SPMDSynchronizer(
        spmd_id="test_spmd",
        num_workers=num_workers,
        sync_policy=MaxSkewSyncPolicy(max_diff),
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
    # diff is 3 - 1 = 2, which is <= max_diff. Unblocked!
    assert 0 not in blocked_workers

    # 5. Worker 2 catches up to step 2
    sync.report_worker_op_completed(2, op, 2)
    # diff is 3 - 2 = 1, which is <= 2 // 2. Unblock should trigger.
    assert 0 not in blocked_workers
    assert 0 in unblocked_workers

    # 6. SPMD step should be 2 (minimum of all workers)
    assert sync.get_spmd_step(op) == 2


def test_worker_sync_controller_lock():
    """Verify that controller-requested locks work alongside drift locks."""
    sync = SPMDSynchronizer(
        spmd_id="test_spmd", num_workers=2, sync_policy=MaxSkewSyncPolicy(10)
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
    updates = manager.check_and_enforce_limits()
    # allowed_step = target_min_step (0) + limit (2) = 2
    assert ("A", op_a, 2) in updates

    # 2. A moves to step 2
    manager.report_op_completed("A", op_a, 2)
    updates = manager.check_and_enforce_limits()
    # A is at 2, B is at 0. diff=2. allowed_step = B(0) + limit(2) = 2.
    # Cached, should not return update
    assert not any(u for u in updates if u[0] == "A" and u[1] == op_a)

    # 3. A moves to step 3
    manager.report_op_completed("A", op_a, 3)
    updates = manager.check_and_enforce_limits()
    # Still allowed_step = 2, cached.
    assert not any(u for u in updates if u[0] == "A" and u[1] == op_a)

    # 4. B moves to step 1
    manager.report_op_completed("B", op_b, 1)
    updates = manager.check_and_enforce_limits()
    # target B.consume is at 1, limit is 2. allowed_step = 1 + 2 = 3
    assert ("A", op_a, 3) in updates

    # 5. B moves to step 2
    manager.report_op_completed("B", op_b, 2)
    # target B.consume is at 2, limit 2. allowed_step = 2 + 2 = 4
    updates = manager.check_and_enforce_limits()
    assert ("A", op_a, 4) in updates


def test_async_manager_global_limit():
    """Verify that AsyncStepManager locks all ops when max_steps is reached."""
    meta = OperatorMeta(name="run")
    node = Node(name="N", spmd_config=SPMDConfig(operator_metas={"run": meta}))
    graph = Graph(nodes={"N": node})

    config = AsyncStepConfig(max_steps=5)
    manager = AsyncStepManager(graph=graph, config=config)

    manager.report_op_completed("N", "run", 4)
    updates = manager.check_and_enforce_limits()
    # allowed_step should be 5 because max_steps=5 and global_step is 4
    # Wait, Global Limit code:
    # `allowed_step = min(allowed_step, self._config.max_steps)`
    # actually it should allow step 5 if we just completed 4.
    assert ("N", "run", 5) in [u for u in updates if u[0] == "N"]

    manager.report_op_completed("N", "run", 5)
    updates = manager.check_and_enforce_limits()
    # Limit hasn't changed, should be empty
    assert not any(u for u in updates if u[0] == "N" and u[1] == "run")


def test_async_manager_reports_all_completed_at_max_steps():
    """Completion should only trip once every tracked op reaches max_steps."""
    op_a = "produce"
    op_b = "consume"
    node_a = Node(
        name="A",
        spmd_config=SPMDConfig(operator_metas={op_a: OperatorMeta(name=op_a)}),
    )
    node_b = Node(
        name="B",
        spmd_config=SPMDConfig(operator_metas={op_b: OperatorMeta(name=op_b)}),
    )
    graph = Graph(nodes={"A": node_a, "B": node_b})
    manager = AsyncStepManager(graph=graph, config=AsyncStepConfig(max_steps=3))

    assert manager.is_all_completed() is False

    manager.report_op_completed("A", op_a, 3)
    assert manager.is_all_completed() is False

    manager.report_op_completed("B", op_b, 2)
    assert manager.is_all_completed() is False

    manager.report_op_completed("B", op_b, 3)
    assert manager.is_all_completed() is True
