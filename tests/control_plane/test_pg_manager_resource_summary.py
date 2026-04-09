import pytest

from fluxway.common.types import DriverResourceSpec, ResourceSpec
from fluxway.control_plane.scheduler import PGManager


def test_pg_manager_resource_allocation_summary():
    pg_manager = PGManager()
    pg_manager._spmd_to_resource["summary_spmd"] = ResourceSpec(
        num_cpus=1, memory_gb=1.5
    )
    pg_manager._spmd_to_num_workers["summary_spmd"] = 3
    pg_manager._spmd_to_driver_resource["summary_spmd"] = DriverResourceSpec(
        num_cpus=1, memory_gb=0.5
    )

    summary = pg_manager.get_resource_allocation_summary("summary_spmd")

    assert summary is not None
    assert summary["spmd_id"] == "summary_spmd"
    assert summary["worker_count"] == 3
    assert summary["bundle_count"] == 4
    assert summary["driver_bundle"]["memory_gb"] == 0.5
    assert summary["worker_bundle"]["memory_gb"] == 1.5
    assert summary["total_memory_gb"] == pytest.approx(5.0)
    assert summary["total_cpu"] == pytest.approx(4.0)
