import os

# Force CPU and hide GPUs before any imports (JAX/Ray)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import ray
import asyncio
import argparse
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.control_plane.controller import create_controller
from fluxway.common.types.configs import AsyncStepConfig

from fluxway.example.recommendation.input_spmd import InputSPMD
from fluxway.example.recommendation.sparse_spmd import SparseEmbeddingSPMD
from fluxway.example.recommendation.dense_spmd import DenseSPMD
from fluxway.example.recommendation.reader import RecommendationReader


async def run_training(args):
    # Set JAX_PLATFORMS=cpu in runtime_env to ensure all Ray actors use CPU
    runtime_env = {"env_vars": {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    graph = FunctionalGraph()

    # 1. Define Nodes
    reader = graph.node("TrainingReader", RecommendationReader)
    input_node = graph.node("InputProcessor", InputSPMD)
    table1 = graph.node("EmbeddingTable1", SparseEmbeddingSPMD)
    table2 = graph.node("EmbeddingTable2", SparseEmbeddingSPMD)
    dense = graph.node("DenseLayers", DenseSPMD)

    r"""
            reader
              |
          input_node
        /     |      \
      |       |        |
    table1  table2     |
      |       |        |
      ------------------
              |
            dense
    """

    # 2. Source Data (Custom Reader)
    h_raw = reader.read()

    # 3. Forward Pass
    h_proc = input_node.process(h_raw)

    h_emb1 = table1.lookup(unique_ids=h_proc.unique_ids1).sparse_features
    h_emb2 = table2.lookup(unique_ids=h_proc.unique_ids2).sparse_features

    h_forward = dense.forward(
        emb1=h_emb1, inv1=h_proc.inv1, emb2=h_emb2, inv2=h_proc.inv2
    )

    # 4. Backward Pass
    h_grads = dense.backward(context=h_forward)

    _ = table1.update(grads=h_grads.grad1)
    _ = table2.update(grads=h_grads.grad2)

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
