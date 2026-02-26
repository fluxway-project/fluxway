import ray
import asyncio
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.control_plane.controller import create_controller

from .data_types import EmbeddingInput
from .sparse_spmd import SparseEmbeddingSPMD
from .dense_spmd import DenseSPMD


async def run_training():
    # 1. Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 2. Initialize Functional Graph
    graph = FunctionalGraph()

    # 3. Define SPMD Nodes
    sparse1 = graph.node("Sparse1", SparseEmbeddingSPMD)
    sparse2 = graph.node("Sparse2", SparseEmbeddingSPMD)
    dense = graph.node("Dense", DenseSPMD)

    # 4. Define Data Flow (Forward)
    reader1 = graph.reader(
        "Input1", config={"src_dir": "/tmp/data1"}, output_data_type=EmbeddingInput
    )
    reader2 = graph.reader(
        "Input2", config={"src_dir": "/tmp/data2"}, output_data_type=EmbeddingInput
    )

    h_input1 = reader1.read()
    h_input2 = reader2.read()

    h_emb1 = sparse1.lookup(h_input1)
    h_emb2 = sparse2.lookup(h_input2)

    h_forward = dense.forward(emb1=h_emb1, emb2=h_emb2)

    # 5. Define Data Flow (Backward)
    # The backward operator returns grad1, grad2
    h_grads = dense.backward(h_forward)

    # Route gradients back to Sparse SPMDs
    # In FunctionalGraph, we can select specific outputs if an op returns multiple
    # but for simplicity in this demo, we use the returned handle
    _ = sparse1.update(h_grads)
    _ = sparse2.update(h_grads)

    # 6. Compile the Graph
    task_graph = graph.compile("Recommendation_Training")

    print("Success: Recommendation training graph compiled!")
    print(f"Nodes: {list(task_graph.nodes.keys())}")

    # 7. Start Distributed Execution via Controller
    print("\nStarting Fluxway Controller (Ray Actors)...")
    try:
        _ = await create_controller(task_graph)
        print("Controller started. Waiting for execution... (10s)")
        await asyncio.sleep(10)

        print("Trial execution completed.")
        print(
            "Check the logs above for [SparseSPMD] and [DenseSPMD] prints to verify communication."
        )

    except Exception as e:
        print(f"Controller Startup Failed (likely due to resource constraints): {e}")


def main():
    asyncio.run(run_training())


if __name__ == "__main__":
    main()
