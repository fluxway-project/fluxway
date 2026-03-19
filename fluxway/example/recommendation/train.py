import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import asyncio
import argparse

import ray

from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import AsyncStepConfig
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.control_plane.controller import create_controller

from fluxway.example.recommendation.reader import RecommendationReader
from fluxway.example.recommendation.input_spmd import InputSPMD
from fluxway.example.recommendation.sparse_spmd import SparseEmbeddingSPMD
from fluxway.example.recommendation.dense_spmd import DenseSPMD


async def run_training(args):
    runtime_env = {"env_vars": {"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}}

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    graph = FunctionalGraph()

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

    reader_mesh = DeviceMesh(device_shape=(5,), axis_names=("worker",))
    input_mesh = DeviceMesh(device_shape=(6,), axis_names=("dp",))
    table1_mesh = DeviceMesh(device_shape=(3,), axis_names=("shard",))
    table2_mesh = DeviceMesh(device_shape=(4,), axis_names=("shard",))
    dense_mesh = DeviceMesh(device_shape=(6,), axis_names=("dp",))

    reader = graph.node("TrainingReader", RecommendationReader, mesh=reader_mesh)
    input_node = graph.node("InputProcessor", InputSPMD, mesh=input_mesh)
    table1 = graph.node(
        "EmbeddingTable1",
        SparseEmbeddingSPMD,
        mesh=table1_mesh,
        vocab_size=2000,
        embed_dim=16,
    )
    table2 = graph.node(
        "EmbeddingTable2",
        SparseEmbeddingSPMD,
        mesh=table2_mesh,
        vocab_size=2000,
        embed_dim=16,
    )
    dense = graph.node(
        "DenseLayers", DenseSPMD, mesh=dense_mesh, embed_dim=16, dense_stats_dim=2
    )

    h_raw = reader.read()
    h_proc = input_node.process(h_raw)

    h_emb1 = table1.lookup(unique_ids=h_proc.unique_user_ids).sparse_features
    h_emb2 = table2.lookup(unique_ids=h_proc.unique_item_ids).sparse_features

    h_step = dense.step(
        emb1=h_emb1,
        inv1=h_proc.user_inv,
        emb2=h_emb2,
        inv2=h_proc.item_inv,
        dp_data=h_proc.dp_data,
    )

    table1.update(grads=h_step.grad1)
    table2.update(grads=h_step.grad2)
    dense.update(dense_grad=h_step.dense_grad)

    task_graph = graph.compile("End-to-End_Training_Flow")
    print(f"[Fluxway] Graph compiled. Nodes: {list(task_graph.nodes.keys())}")

    async_step_config = AsyncStepConfig(max_steps=args.steps)
    controller = await create_controller(
        task_graph, async_step_config=async_step_config, runtime_env=runtime_env
    )
    print(f"[Fluxway] Running for {args.steps} steps ...")

    completed = await controller.wait_for_completion.remote()
    if completed:
        print(f"[Fluxway] Training completed after {args.steps} steps.")
    else:
        print("[Fluxway] Training failed or timed out.")


def main():
    parser = argparse.ArgumentParser(description="Fluxway Recommendation Example")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to run")
    asyncio.run(run_training(parser.parse_args()))


if __name__ == "__main__":
    main()
