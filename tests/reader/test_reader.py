"""
End-to-end test for the Reader → Processor graph pipeline.

Verifies the full lifecycle: graph compilation → controller startup →
topology negotiation → actual data flow between a GrainReader SPMD and a
downstream Processor SPMD.
"""

import asyncio
import os

import pytest
import ray

from fluxway.control_plane.controller import create_controller
from fluxway.control_plane.graph.functional import FunctionalGraph

# Import helper nodes from the separate module so Ray can deserialize them.
from tests.reader.helper_nodes import MockDataSource, SimpleProcessor, TypedGrainReader

os.environ["FLUXWAY_BACKEND"] = "jax"

_RUNTIME_ENV = {"working_dir": "."}
_DATA_SIZE = 10
_POLL_INTERVAL_S = 1.0
_MAX_POLL_ROUNDS = 30


@pytest.fixture(scope="module", autouse=True)
def ray_session():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=True, runtime_env=_RUNTIME_ENV)
    yield
    # Ray is intentionally kept alive across tests in this module to avoid
    # repeated cluster startup cost.


def _build_graph() -> FunctionalGraph:
    graph = FunctionalGraph()
    reader = graph.reader(
        "Reader",
        reader_class=TypedGrainReader,
        mesh_axis_names=("data",),
        mesh_shape=(1,),
        data_source=MockDataSource(_DATA_SIZE),
        shuffle=False,
        num_epochs=1,
    )
    processor = graph.node("Processor", SimpleProcessor)
    processor.process(batch=reader.read())
    return graph


@pytest.mark.asyncio
async def test_reader_processor_data_flows():
    """Both Reader and Processor must make progress within the timeout."""
    task_graph = _build_graph().compile("ReaderE2E")

    reader_id = next(n for n in task_graph.nodes if "Reader" in n)
    proc_id = next(n for n in task_graph.nodes if "Processor" in n)

    controller = await create_controller(task_graph, runtime_env=_RUNTIME_ENV)
    try:
        for _ in range(_MAX_POLL_ROUNDS):
            await asyncio.sleep(_POLL_INTERVAL_S)
            heartbeats = await controller.get_all_heartbeats.remote()
            reader_steps = heartbeats.get(reader_id, {}).get("op_steps", {})
            proc_steps = heartbeats.get(proc_id, {}).get("op_steps", {})
            if reader_steps.get("read", 0) > 0 and proc_steps.get("process", 0) > 0:
                break

        heartbeats = await controller.get_all_heartbeats.remote()
        reader_steps = heartbeats.get(reader_id, {}).get("op_steps", {})
        proc_steps = heartbeats.get(proc_id, {}).get("op_steps", {})

        assert reader_steps.get("read", 0) > 0, (
            f"Reader {reader_id} never produced data. Heartbeat: {heartbeats.get(reader_id)}"
        )
        assert proc_steps.get("process", 0) > 0, (
            f"Processor {proc_id} never consumed data. Heartbeat: {heartbeats.get(proc_id)}"
        )
    finally:
        await controller.stop.remote()


if __name__ == "__main__":
    asyncio.run(test_reader_processor_data_flows())
