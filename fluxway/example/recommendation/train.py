import os

# Force CPU and hide GPUs before any imports (JAX/Ray)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import asyncio
import argparse

import ray

from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.control_plane.controller import create_controller
from fluxway.common.types.configs import AsyncStepConfig

from fluxway.example.recommendation.input_spmd import InputSPMD
from fluxway.example.recommendation.sparse_spmd import SparseEmbeddingSPMD
from fluxway.example.recommendation.dense_spmd import DenseSPMD
from fluxway.example.recommendation.reader import RecommendationReader


async def run_training(args):
    # Set JAX_PLATFORMS=cpu in runtime_env to ensure all Ray actors use CPU
    runtime_env = {"env_vars": {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    graph = FunctionalGraph()
    from fluxway.common.mesh import DeviceMesh

    # 1. Define Nodes with distinct worker counts
    # TrainingReader: 5 workers
    reader_mesh = DeviceMesh(device_shape=(5,), axis_names=("worker",))
    # InputProcessor: 2 workers
    input_mesh = DeviceMesh(device_shape=(2,), axis_names=("dp",))
    # Table1: 3 workers
    table1_mesh = DeviceMesh(device_shape=(3,), axis_names=("shard",))
    # Table2: 4 workers
    table2_mesh = DeviceMesh(device_shape=(4,), axis_names=("shard",))
    # DenseLayers: 6 worker
    dense_mesh = DeviceMesh(device_shape=(6,), axis_names=("dp",))

    reader = graph.node("TrainingReader", RecommendationReader, mesh=reader_mesh)
    input_node = graph.node("InputProcessor", InputSPMD, mesh=input_mesh)
    table1 = graph.node("EmbeddingTable1", SparseEmbeddingSPMD, mesh=table1_mesh)
    table2 = graph.node("EmbeddingTable2", SparseEmbeddingSPMD, mesh=table2_mesh)
    dense = graph.node("DenseLayers", DenseSPMD, mesh=dense_mesh)

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

    h_emb1 = table1.lookup(unique_ids=h_proc.unique_user_ids).sparse_features
    h_emb2 = table2.lookup(unique_ids=h_proc.unique_item_ids).sparse_features

    h_forward = dense.forward(
        emb1=h_emb1,
        inv1=h_proc.user_inv,
        emb2=h_emb2,
        inv2=h_proc.item_inv,
        dp_data=h_proc.dp_data,
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
