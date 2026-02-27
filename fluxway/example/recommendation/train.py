import ray
import asyncio
import argparse
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.control_plane.controller import create_controller
from fluxway.common.types.configs import AsyncStepConfig

from .input_spmd import InputSPMD
from .sparse_spmd import SparseEmbeddingSPMD
from .dense_spmd import DenseSPMD
from .reader import RecommendationReader


async def run_training(args):
    # Set JAX_PLATFORMS=cpu in runtime_env to ensure all Ray actors use CPU
    runtime_env = {"env_vars": {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "-1"}}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    graph = FunctionalGraph()

    # 1. Define Nodes
    input_node = graph.node("InputProcessor", InputSPMD)
    table1 = graph.node("EmbeddingTable1", SparseEmbeddingSPMD)
    table2 = graph.node("EmbeddingTable2", SparseEmbeddingSPMD)
    dense = graph.node("DenseLayers", DenseSPMD)

    # 2. Source Data (Custom Reader)
    reader = graph.node("TrainingReader", RecommendationReader)
    h_raw = reader.read()

    # 3. Forward Pass
    h_proc = input_node.process(h_raw)

    h_emb1 = table1.lookup(unique_ids=h_proc.unique_ids1)
    h_emb2 = table2.lookup(unique_ids=h_proc.unique_ids2)

    h_forward = dense.forward(
        emb1=h_emb1, inv1=h_proc.inv1, emb2=h_emb2, inv2=h_proc.inv2
    )

    # 4. Backward Pass
    h_grads = dense.backward(context=h_forward)

    _ = table1.update(grads=h_grads)
    _ = table2.update(grads=h_grads)

    # 6. Compile and Start
    task_graph = graph.compile("End-to-End_Training_Flow")

    print("\n[Fluxway] Recommendation Graph Compiled.")
    print(f"Nodes: {list(task_graph.nodes.keys())}")

    # Set max steps
    async_step_config = AsyncStepConfig(max_steps=args.steps)

    try:
        controller = await create_controller(
            task_graph, async_step_config=async_step_config, runtime_env=runtime_env
        )
        print(f"[Fluxway] System running for {args.steps} steps...")

        # Wait for completion
        completed = await controller.wait_for_completion.remote()
        if completed:
            print(
                f"\n[Fluxway] Training completed successfully after {args.steps} steps."
            )
        else:
            print("\n[Fluxway] Training failed or timed out.")

    except Exception as e:
        print(f"Controller Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fluxway Recommendation Example")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to run")
    args = parser.parse_args()

    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
