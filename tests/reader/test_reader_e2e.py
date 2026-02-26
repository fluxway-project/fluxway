import pytest
import ray
import asyncio
import os
import json
import tempfile

# Set backend before imports that might trigger backend selection
os.environ["FLUXWAY_BACKEND"] = "jax"

from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.control_plane.controller import create_controller

# Import helper nodes from the separate module to ensure Ray can deserialize them
from tests.reader.helper_nodes import SimpleProcessor


@pytest.mark.asyncio
async def test_reader_e2e():
    # Ensure current working directory is in Ray's runtime environment
    # so workers can find 'tests.reader.helper_nodes'
    runtime_env = {"working_dir": "."}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=True, runtime_env=runtime_env)

    # 1. Prepare dummy data
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_file = os.path.join(tmp_dir, "data.json")
        with open(data_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"id": i, "value": i * 1.5}) + "\n")

        # 2. Build Graph
        graph = FunctionalGraph()

        # Use our new Reader (default is GrainReader)
        reader = graph.reader(
            "MyReader", src_dir=tmp_dir, file_pattern="*.json", batch_size=2
        )

        processor = graph.node("Processor", SimpleProcessor)

        # Connect
        raw_data = reader.read()
        _ = processor.process(data=raw_data)

        # 3. Compile
        task_graph = graph.compile("ReaderTest")

        # 4. Run Controller
        controller = await create_controller(task_graph, runtime_env=runtime_env)

        # Monitor cluster status and heartbeats
        print("\nMonitoring cluster status...")
        reader_id = [node_id for node_id in task_graph.nodes if "MyReader" in node_id][
            0
        ]
        proc_id = [node_id for node_id in task_graph.nodes if "Processor" in node_id][0]

        for i in range(30):
            await asyncio.sleep(1.0)
            status = await controller.get_status.remote()
            print(f"[{i:02d}] Cluster status: {status}")

            # Print heartbeats for debugging
            heartbeats = await controller.get_all_heartbeats.remote()
            reader_hb = heartbeats.get(reader_id, {})
            proc_hb = heartbeats.get(proc_id, {})

            print(
                f"    - Reader: status={reader_hb.get('status')}, ready={reader_hb.get('workers_ready')}, steps={reader_hb.get('op_steps')}"
            )
            print(
                f"    - Processor: status={proc_hb.get('status')}, ready={proc_hb.get('workers_ready')}, steps={proc_hb.get('op_steps')}"
            )

            # Check if both have progressed
            reader_steps = reader_hb.get("op_steps", {})
            proc_steps = proc_hb.get("op_steps", {})

            if reader_steps.get("read", 0) > 0 and proc_steps.get("process", 0) > 0:
                print("Progress detected!")
                break

        # Verify final results
        heartbeats = await controller.get_all_heartbeats.remote()
        reader_hb = heartbeats.get(reader_id, {})
        proc_hb = heartbeats.get(proc_id, {})

        reader_steps = reader_hb.get("op_steps", {})
        proc_steps = proc_hb.get("op_steps", {})

        print("\nFinal Summary:")
        print(
            f"Reader {reader_id}: status={reader_hb.get('status')}, steps={reader_steps}"
        )
        print(
            f"Processor {proc_id}: status={proc_hb.get('status')}, steps={proc_steps}"
        )

        # Check that they have moved
        assert reader_steps.get("read", 0) > 0, (
            f"Reader {reader_id} should have progressed. HB: {reader_hb}"
        )
        assert proc_steps.get("process", 0) > 0, (
            f"Processor {proc_id} should have progressed. HB: {proc_hb}"
        )

        # Cleanup
        await controller.stop.remote()


if __name__ == "__main__":
    asyncio.run(test_reader_e2e())
