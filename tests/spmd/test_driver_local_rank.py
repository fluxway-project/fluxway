from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import SPMDConfig
from fluxway.spmd.driver.driver import SPMDDriver
import pytest


def _driver_cls():
    return SPMDDriver.__ray_metadata__.modified_class


def _make_driver():
    mesh = DeviceMesh(axis_names=("dp",), device_shape=(4,))
    config = SPMDConfig(mesh=mesh, operators={}, operator_metas={})
    driver_cls = _driver_cls()
    return driver_cls(
        spmd_id="spmd-a",
        config=config,
        placement_group=None,
        worker_bundle_indices=[],
        controller_handle=None,
        graph_topology={"predecessors": [], "successors": []},
    )


def test_compute_rank_info_single_node():
    driver = _make_driver()
    node_ids = {0: "node-a", 1: "node-a", 2: "node-a"}

    assert driver._compute_rank_info(node_ids) == {
        0: {"local_rank": 0, "local_world_size": 3, "node_rank": 0, "num_nodes": 1},
        1: {"local_rank": 1, "local_world_size": 3, "node_rank": 0, "num_nodes": 1},
        2: {"local_rank": 2, "local_world_size": 3, "node_rank": 0, "num_nodes": 1},
    }


def test_compute_rank_info_partial_fill_orders_node_rank_by_smallest_rank():
    driver = _make_driver()
    node_ids = {0: "node-b", 1: "node-b", 2: "node-a", 3: "node-a", 4: "node-a"}

    assert driver._compute_rank_info(node_ids) == {
        0: {"local_rank": 0, "local_world_size": 2, "node_rank": 0, "num_nodes": 2},
        1: {"local_rank": 1, "local_world_size": 2, "node_rank": 0, "num_nodes": 2},
        2: {"local_rank": 0, "local_world_size": 3, "node_rank": 1, "num_nodes": 2},
        3: {"local_rank": 1, "local_world_size": 3, "node_rank": 1, "num_nodes": 2},
        4: {"local_rank": 2, "local_world_size": 3, "node_rank": 1, "num_nodes": 2},
    }


class _FakeAsyncRemoteCall:
    def __init__(self, func):
        async def remote(*args, **kwargs):
            return func(*args, **kwargs)

        self.remote = remote


class _FakeSyncRemoteCall:
    def __init__(self, func):
        self.remote = func


class FakeWorker:
    def __init__(self, node_id):
        self.node_id = node_id
        self.setup_calls = []
        self.get_node_id = _FakeAsyncRemoteCall(lambda: node_id)
        self.setup_and_start = _FakeSyncRemoteCall(
            lambda **kwargs: self.setup_calls.append(kwargs)
        )


@pytest.mark.asyncio
async def test_start_workers_queries_node_ids_before_setup(monkeypatch):
    driver = _make_driver()
    driver.worker_bundle_indices = [0, 1, 2]
    fake_workers = [FakeWorker("node-a"), FakeWorker("node-a"), FakeWorker("node-b")]
    worker_iter = iter(fake_workers)

    class _FakeActorClass:
        def options(self, **kwargs):
            return self

        def remote(self, **kwargs):
            return next(worker_iter)

    monkeypatch.setattr(
        "fluxway.spmd.driver.driver.SPMDWorker",
        _FakeActorClass(),
    )
    monkeypatch.setattr(
        "fluxway.spmd.driver.driver.ray.get_runtime_context",
        lambda: type("Ctx", (), {"current_actor": object()})(),
    )
    monkeypatch.setattr("fluxway.spmd.driver.driver.ray.get", lambda refs: refs)

    await driver._start_workers()

    assert fake_workers[0].setup_calls == [
        {"local_rank": 0, "local_world_size": 2, "node_rank": 0, "num_nodes": 2}
    ]
    assert fake_workers[1].setup_calls == [
        {"local_rank": 1, "local_world_size": 2, "node_rank": 0, "num_nodes": 2}
    ]
    assert fake_workers[2].setup_calls == [
        {"local_rank": 0, "local_world_size": 1, "node_rank": 1, "num_nodes": 2}
    ]
