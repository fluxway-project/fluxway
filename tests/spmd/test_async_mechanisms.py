"""
Tests for intra-SPMD and inter-SPMD asynchronous execution mechanisms.

These tests verify:
1. max_rank_step_skew: Limits completed-step difference between ranks in one SPMD.
2. max_inflight_steps: Limits step difference between connected SPMDs in the graph.
"""

from collections.abc import Mapping
from unittest.mock import MagicMock

import pytest

from fluxway.control_plane.flow_control.manager import FlowControlManager
from fluxway.spmd.driver.synchronizer import (
    MaxRankStepSkewSyncPolicy,
    SPMDSynchronizer,
)
from fluxway.spmd.driver.driver import SPMDDriver
from fluxway.spmd.driver.credit import (
    CreditEdge,
    CreditUpdate,
    CreditWindow,
    UpstreamCreditState,
    build_credit_edges,
    credit_topology_for_driver,
)
from fluxway.control_plane.async_manager.manager import AsyncStepManager
from fluxway.common.compiled_graph import (
    DependencySpec,
    OperatorKey,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import (
    Graph,
    Node,
    AsyncStepConfig,
    DriverHeartbeat,
    DriverState,
    DriverStatus,
    FlowControlState,
    WorkerHeartbeat,
    WorkerStatus,
)
from fluxway import OperatorSpec
from fluxway.common.operator_policy import AsyncPolicy
from tests.compiled_graph_fixtures import (
    compile_test_graph,
    single_spmd_config,
    spmd_config_from_graph,
)


def _dependency_edge(
    source_node: str,
    source_op: str,
    target_node: str,
    target_op: str,
    *,
    local_only: bool = False,
) -> DependencySpec:
    return DependencySpec(
        source=OperatorKey(spmd_id=source_node, op_id=source_op),
        target=OperatorKey(spmd_id=target_node, op_id=target_op),
        local_only=local_only,
    )


def _planned_graph(
    *,
    name: str,
    operators_by_spmd: Mapping[str, Mapping[str, OperatorSpec]],
    edges: tuple[DependencySpec, ...] = (),
) -> Graph:
    mesh = DeviceMesh(axis_names=("worker",), device_shape=(1,))
    compiled_graph = compile_test_graph(
        name=name,
        spmds={
            spmd_id: (mesh, operators)
            for spmd_id, operators in operators_by_spmd.items()
        },
        dependencies=edges,
    )
    return Graph(
        name=name,
        nodes={
            spmd_id: Node(
                spmd_class=object,
                name=spmd_id,
                resource_profile=compiled_graph.spmd(spmd_id).resource_profile,
                spmd_config=spmd_config_from_graph(
                    compiled_graph,
                    spmd_id,
                ),
            )
            for spmd_id in operators_by_spmd
        },
        compiled_graph=compiled_graph,
    )


# =============================================================================
# 1. Intra-SPMD: SPMDSynchronizer Tests
# =============================================================================


def test_worker_sync_skew_limit():
    """
    Verify that SPMDSynchronizer blocks ranks that exceed max_rank_step_skew.
    """
    num_workers = 3
    max_diff = 2
    sync = SPMDSynchronizer(
        spmd_id="test_spmd",
        num_workers=num_workers,
        sync_policy=MaxRankStepSkewSyncPolicy(max_diff),
    )

    blocked_workers: set[int] = set()
    unblocked_workers: set[int] = set()

    versions = []

    def block_cb(rank, op, version):
        versions.append((rank, op, version))
        blocked_workers.add(rank)
        if rank in unblocked_workers:
            unblocked_workers.remove(rank)

    def unblock_cb(rank, op, version):
        versions.append((rank, op, version))
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
    assert versions == sorted(versions, key=lambda item: item[2])


def test_worker_sync_controller_lock():
    """Verify that controller-requested locks work alongside drift locks."""
    sync = SPMDSynchronizer(
        spmd_id="test_spmd", num_workers=2, sync_policy=MaxRankStepSkewSyncPolicy(10)
    )

    blocked = []
    sync.set_callbacks(block_worker_op=lambda r, o, v: blocked.append((r, v)))

    sync.block_spmd_op("train")
    assert len(blocked) == 2
    assert [version for _, version in blocked] == [0, 0]
    assert sync.is_op_locked("train")


def test_worker_sync_op_runnable_epochs_are_monotonic_per_worker_op():
    sync = SPMDSynchronizer(
        spmd_id="test_spmd", num_workers=2, sync_policy=MaxRankStepSkewSyncPolicy(0)
    )
    events = []
    sync.set_callbacks(
        block_worker_op=lambda r, op, v: events.append(("block", r, op, v)),
        unblock_worker_op=lambda r, op, v: events.append(("unblock", r, op, v)),
    )

    sync.report_worker_op_completed(0, "train", 1)
    sync.report_worker_op_completed(1, "train", 1)
    sync.report_worker_op_completed(0, "train", 2)
    sync.report_worker_op_completed(1, "train", 2)

    epochs_by_worker: dict[tuple[int, str], list[int]] = {}
    for _kind, rank, op_name, epoch in events:
        epochs_by_worker.setdefault((rank, op_name), []).append(epoch)
    for epochs in epochs_by_worker.values():
        assert epochs == sorted(epochs)
        assert len(epochs) == len(set(epochs))


def test_max_rank_step_skew_sync_policy_invariants():
    policy = MaxRankStepSkewSyncPolicy()
    assert policy.max_rank_step_skew == 0

    with pytest.raises(ValueError, match="non-negative integer"):
        MaxRankStepSkewSyncPolicy(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        MaxRankStepSkewSyncPolicy(True)
    with pytest.raises(ValueError, match="non-negative integer"):
        MaxRankStepSkewSyncPolicy(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative integer"):
        MaxRankStepSkewSyncPolicy("0")  # type: ignore[arg-type]

    steps = {0: {"train": 1}, 1: {"train": 0}}
    assert (
        policy.evaluate(
            steps,
            num_workers=2,
            rank=0,
            op_name="train",
            max_step=1,
            min_step=0,
        )
        is True
    )
    assert (
        policy.evaluate(
            steps,
            num_workers=2,
            rank=1,
            op_name="train",
            max_step=1,
            min_step=0,
        )
        is False
    )


# =============================================================================
# 2. Inter-SPMD: Driver Credit Tests
# =============================================================================


def test_driver_credit_edge_limit():
    op_a = "produce"
    op_b = "consume"
    meta_a = OperatorSpec(name=op_a)
    meta_b = OperatorSpec(name=op_b, async_policy=AsyncPolicy.pipelined(depth=3))
    edge = _dependency_edge("A", op_a, "B", op_b)
    graph = _planned_graph(
        name="driver-credit-edge-limit",
        operators_by_spmd={"A": {op_a: meta_a}, "B": {op_b: meta_b}},
        edges=(edge,),
    )

    credit_edges = build_credit_edges(
        graph.require_compiled_graph(),
        AsyncStepConfig(),
    )
    upstream_edges, downstream_edges = credit_topology_for_driver("A", credit_edges)
    assert downstream_edges == ()
    state = UpstreamCreditState(upstream_edges, epoch=0)
    assert state.allowed_steps() == {op_a: 2}

    edge_id = credit_edges[0].edge_id
    assert state.receive(edge_id, CreditUpdate(epoch=0, sequence=1, allowed_step=3))
    assert state.allowed_step(op_a) == 3
    assert state.receive(edge_id, CreditUpdate(epoch=0, sequence=2, allowed_step=4))
    assert state.allowed_step(op_a) == 4


def test_driver_credit_ignores_local_only_edges_for_flow_control():
    op_a = "prepare"
    op_b = "train"
    meta_a = OperatorSpec(name=op_a)
    meta_b = OperatorSpec(name=op_b)
    edge = _dependency_edge(
        "Dense",
        op_a,
        "Dense",
        op_b,
        local_only=True,
    )
    graph = _planned_graph(
        name="driver-credit-local-only",
        operators_by_spmd={"Dense": {op_a: meta_a, op_b: meta_b}},
        edges=(edge,),
    )
    assert build_credit_edges(graph.require_compiled_graph(), AsyncStepConfig()) == ()


def test_driver_credit_uses_min_per_source_op_and_keeps_other_ops_independent():
    graph = _planned_graph(
        name="driver-credit-source-op-minimum",
        operators_by_spmd={
            "A": {
                "forward": OperatorSpec(name="forward"),
                "metrics": OperatorSpec(name="metrics"),
            },
            "B": {
                "train": OperatorSpec(
                    name="train", async_policy=AsyncPolicy.pipelined(depth=4)
                )
            },
            "C": {
                "train": OperatorSpec(
                    name="train", async_policy=AsyncPolicy.pipelined(depth=2)
                )
            },
            "D": {
                "consume": OperatorSpec(
                    name="consume", async_policy=AsyncPolicy.pipelined(depth=10)
                )
            },
        },
        edges=(
            _dependency_edge("A", "forward", "B", "train"),
            _dependency_edge("A", "forward", "C", "train"),
            _dependency_edge("A", "metrics", "D", "consume"),
        ),
    )
    credit_edges = build_credit_edges(
        graph.require_compiled_graph(),
        AsyncStepConfig(),
    )
    upstream_edges, _downstream_edges = credit_topology_for_driver("A", credit_edges)
    state = UpstreamCreditState(upstream_edges, epoch=0)

    assert state.allowed_steps() == {"forward": 1, "metrics": 9}
    by_id = {edge.edge_id: edge for edge in credit_edges}
    edge_b = by_id["A.forward->B.train"]
    edge_c = by_id["A.forward->C.train"]
    state.receive(
        edge_b.edge_id,
        CreditUpdate(epoch=0, sequence=1, allowed_step=8),
    )
    assert state.allowed_steps() == {"forward": 1, "metrics": 9}
    state.receive(
        edge_c.edge_id,
        CreditUpdate(epoch=0, sequence=1, allowed_step=4),
    )
    assert state.allowed_steps() == {"forward": 4, "metrics": 9}


def test_heartbeat_contract_excludes_progress_and_backpressure_fields():
    worker_heartbeat = WorkerHeartbeat(
        rank=0,
        timestamp=1.0,
        status=WorkerStatus.RUNNING,
    )
    driver_heartbeat = DriverHeartbeat(
        spmd_id="A",
        timestamp=1.0,
        status=DriverStatus.HEALTHY,
        state=DriverState.RUNNING,
        num_workers=1,
        healthy_num_workers=1,
        workers_ready=True,
        topology_ready=True,
    )

    removed_worker_fields = {
        "completed_step_count",
        "telemetry",
        "mailbox_status",
    }
    removed_driver_fields = {
        "completed_step_count",
        "blocked_ops",
        "is_downstream_busy",
        "source_state",
        "busy_worker_ratio",
    }

    assert removed_worker_fields.isdisjoint(worker_heartbeat.as_dict())
    assert removed_driver_fields.isdisjoint(driver_heartbeat.as_dict())


def test_flow_control_manager_produces_controller_authoritative_windows():
    graph = _planned_graph(
        name="flow-control-authoritative-windows",
        operators_by_spmd={
            "A": {"produce": OperatorSpec(name="produce")},
            "B": {
                "consume": OperatorSpec(
                    name="consume", async_policy=AsyncPolicy.pipelined(depth=3)
                )
            },
        },
        edges=(_dependency_edge("A", "produce", "B", "consume"),),
    )
    manager = FlowControlManager(graph, AsyncStepConfig())

    assert manager.initial_windows_for_driver("A") == (
        CreditWindow(epoch=0, op_name="produce", allowed_step=2),
    )
    assert manager.observe_op_completed("B", "consume", 3) == {
        "A": (CreditWindow(epoch=0, op_name="produce", allowed_step=5),)
    }

    manager.report_state(
        FlowControlState(spmd_id="A", timestamp=1.0, paused=True, epoch=0)
    )
    state = manager.state_for_driver("A")
    assert state is not None
    assert state.paused is True
    with pytest.raises(ValueError, match="epoch mismatch"):
        manager.report_state(
            FlowControlState(spmd_id="A", timestamp=1.0, paused=False, epoch=1)
        )


def test_flow_control_frontier_follows_logical_completion_monotonically():
    graph = _planned_graph(
        name="flow-control-restore-frontier",
        operators_by_spmd={
            "A": {"produce": OperatorSpec(name="produce")},
            "B": {
                "consume": OperatorSpec(
                    name="consume", async_policy=AsyncPolicy.pipelined(depth=3)
                )
            },
        },
        edges=(_dependency_edge("A", "produce", "B", "consume"),),
    )
    manager = FlowControlManager(graph, AsyncStepConfig())
    manager.observe_op_completed("B", "consume", 10)

    windows = manager.observe_op_completed("B", "consume", 6)

    assert manager.epoch == 0
    assert windows == {}
    assert manager.observe_op_completed("B", "consume", 6) == {}


def _make_driver(spmd_id: str, controller_handle=None):
    mesh = DeviceMesh(axis_names=("worker",), device_shape=(1,))
    operators = {
        "produce": OperatorSpec(name="produce"),
        "consume": OperatorSpec(name="consume"),
    }
    config = single_spmd_config(
        name="async-mechanisms-driver-test",
        spmd_id=spmd_id,
        mesh=mesh,
        operators=operators,
    )
    # pyrefly: ignore [missing-attribute]
    return SPMDDriver.__ray_actor_class__(
        spmd_id=spmd_id,
        config=config,
        placement_group=object(),
        worker_bundle_indices=[1],
        controller_handle=controller_handle,
        graph_topology={},
        worker_resource={"CPU": 0},
        expected_node_ids=["node-a"],
    )


def test_driver_combines_controller_window_and_peer_hint_by_min():
    driver = _make_driver("A", MagicMock())
    driver.spmd_synchronizer.update_allowed_step = MagicMock()
    edge = CreditEdge(
        edge_id="A.produce->B.consume",
        source_spmd_id="A",
        source_op="produce",
        target_spmd_id="B",
        target_op="consume",
        max_inflight_steps=2,
    )

    driver.register_credit_topology({}, (edge,), (), epoch=0)
    driver.spmd_synchronizer.update_allowed_step.assert_called_with("produce", 2)

    driver.spmd_synchronizer.update_allowed_step.reset_mock()
    driver.receive_credit_update(
        "B",
        edge.edge_id,
        CreditUpdate(epoch=0, sequence=1, allowed_step=9),
    )
    driver.spmd_synchronizer.update_allowed_step.assert_called_once_with("produce", 2)

    driver.spmd_synchronizer.update_allowed_step.reset_mock()
    driver.apply_credit_window(
        (CreditWindow(epoch=0, op_name="produce", allowed_step=5),)
    )
    driver.spmd_synchronizer.update_allowed_step.assert_called_once_with("produce", 5)

    driver.spmd_synchronizer.update_allowed_step.reset_mock()
    driver.apply_credit_window(
        (CreditWindow(epoch=0, op_name="produce", allowed_step=4),)
    )
    driver.spmd_synchronizer.update_allowed_step.assert_not_called()
    assert driver.credit_manager._controller_credit_windows_by_op == {"produce": 5}

    with pytest.raises(ValueError, match="epoch mismatch"):
        driver.apply_credit_window(
            (CreditWindow(epoch=1, op_name="produce", allowed_step=6),)
        )


def test_driver_completion_reports_controller_and_keeps_peer_fast_path():
    controller = MagicMock()
    upstream_driver = MagicMock()
    driver = _make_driver("B", controller)
    edge = CreditEdge(
        edge_id="A.produce->B.consume",
        source_spmd_id="A",
        source_op="produce",
        target_spmd_id="B",
        target_op="consume",
        max_inflight_steps=2,
    )
    driver.register_credit_topology({"A": upstream_driver}, (), (edge,), epoch=0)

    driver._on_spmd_op_completed("B", "consume", 3)

    controller.report_op_completed.remote.assert_called_once_with("B", "consume", 3)
    upstream_driver.receive_credit_update.remote.assert_called_once()
    downstream_id, edge_id, update = (
        upstream_driver.receive_credit_update.remote.call_args.args
    )
    assert downstream_id == "B"
    assert edge_id == edge.edge_id
    assert update == CreditUpdate(epoch=0, sequence=1, allowed_step=5)


def test_driver_reports_flow_control_state_outside_heartbeat():
    controller = MagicMock()
    driver = _make_driver("A", controller)

    driver.pause_sources()
    paused_state = controller.report_flow_control_state.remote.call_args.args[0]
    assert paused_state.spmd_id == "A"
    assert paused_state.paused is True
    assert paused_state.epoch == 0

    driver.resume_sources()
    resumed_state = controller.report_flow_control_state.remote.call_args.args[0]
    assert resumed_state.spmd_id == "A"
    assert resumed_state.paused is False
    assert resumed_state.epoch == 0


def test_async_manager_reports_all_completed_at_max_steps():
    """Completion should only trip once every tracked op reaches max_steps."""
    op_a = "produce"
    op_b = "consume"
    graph = _planned_graph(
        name="async-manager-test",
        operators_by_spmd={
            "A": {op_a: OperatorSpec(name=op_a)},
            "B": {op_b: OperatorSpec(name=op_b)},
        },
    )
    manager = AsyncStepManager(graph=graph, config=AsyncStepConfig(max_steps=3))

    assert manager.is_all_completed() is False

    manager.report_op_completed("A", op_a, 3)
    assert manager.is_all_completed() is False

    manager.report_op_completed("B", op_b, 2)
    assert manager.is_all_completed() is False

    manager.report_op_completed("B", op_b, 3)
    assert manager.is_all_completed() is True


class TestResumeSkewRecheck:
    """Prove root cause and fix: stale policy locks after snapshot resume."""

    def _make_sync(self, max_rank_step_skew: int = 1, num_workers: int = 2):
        sync = SPMDSynchronizer(
            "EmbeddingTable2",
            num_workers=num_workers,
            sync_policy=MaxRankStepSkewSyncPolicy(
                max_rank_step_skew=max_rank_step_skew
            ),
        )
        blocks: list[tuple[int, str]] = []
        unblocks: list[tuple[int, str]] = []
        sync.set_callbacks(
            block_worker_op=lambda r, o, e: blocks.append((r, o)),
            unblock_worker_op=lambda r, o, e: unblocks.append((r, o)),
        )
        sync.register_expected_ops(["lookup"])
        sync.unblock_all_ops()
        return sync, blocks, unblocks

    def test_policy_lock_persists_after_clear_drain(self):
        """Root cause: clear_drain() does not release worker-level policy locks."""
        sync, blocks, _ = self._make_sync()

        for step in range(1, 5):
            sync.report_worker_op_completed(0, "lookup", step)
            sync.report_worker_op_completed(1, "lookup", step)

        # rank 0 at step 6, rank 1 at step 4 -> skew=2 > 1 -> BLOCK rank 0
        sync.report_worker_op_completed(0, "lookup", 6)
        assert sync._worker_op_locked[0].get("lookup", False), (
            "rank 0 must be policy-blocked"
        )

        blocks.clear()
        sync.clear_drain()

        assert sync._worker_op_locked[0].get("lookup", False), (
            "ROOT CAUSE: clear_drain() does not release worker-level policy locks; "
            "workers remain stuck after snapshot resume"
        )

    def test_clear_drain_resolves_stale_lock_when_skew_resolved(self):
        """Clearing a drain reconciles worker locks with the current skew."""
        sync, _, unblocks = self._make_sync()

        # Both workers reach the drain boundary step=6
        for step in range(1, 7):
            sync.report_worker_op_completed(0, "lookup", step)
            sync.report_worker_op_completed(1, "lookup", step)

        # Manually introduce stale lock (as if it was set mid-drain)
        sync._worker_op_locked[0]["lookup"] = True

        unblocks.clear()
        sync.clear_drain()

        assert not sync._worker_op_locked[0].get("lookup", False), (
            "With equal step counts (skew=0), recheck must unblock rank 0"
        )
        assert any(r == 0 and o == "lookup" for r, o in unblocks), (
            "unblock callback must fire to wake the worker's execution loop"
        )

    def test_clear_drain_preserves_genuine_skew_block(self):
        """Clearing a drain keeps a genuine skew block."""
        sync, _, _ = self._make_sync()

        # rank 0 at step 6, rank 1 at step 4 -> real skew=2 > 1
        sync.report_worker_op_completed(0, "lookup", 6)
        sync.report_worker_op_completed(1, "lookup", 4)

        sync.clear_drain()

        assert sync._worker_op_locked[0].get("lookup", False), (
            "Genuine skew=2 > max_rank_step_skew=1 must remain blocked after recheck"
        )
