from unittest.mock import MagicMock

from fluxway.common.types import DriverResourceSpec, ResourceSpec
from fluxway.control_plane.controller import ControllerActor


def test_controller_exposes_resource_allocation_summary():
    graph_stub = MagicMock()
    graph_stub.nodes = {}
    graph_stub.edges = []
    graph_stub.source_nodes = []

    ctrl = ControllerActor.__ray_actor_class__(graph=graph_stub)
    ctrl._scheduler.pg_manager._spmd_to_resource["dense"] = ResourceSpec(
        num_cpus=2, memory_gb=1.0
    )
    ctrl._scheduler.pg_manager._spmd_to_num_workers["dense"] = 6
    ctrl._scheduler.pg_manager._spmd_to_driver_resource["dense"] = DriverResourceSpec(
        num_cpus=1, memory_gb=1.0
    )

    summary = ctrl.get_resource_allocation_summary()

    assert summary["dense"]["bundle_count"] == 7
    assert summary["dense"]["worker_count"] == 6
    assert summary["dense"]["total_memory_gb"] == 7.0
    assert summary["dense"]["total_cpu"] == 13.0
