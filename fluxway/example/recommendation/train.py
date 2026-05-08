import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse

from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import AsyncStepConfig
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.control_plane.controller import create_controller
from fluxway.common.types import CheckpointConfig

from fluxway.example.recommendation.reader import RecommendationReader
from fluxway.example.recommendation.input_spmd import InputSPMD
from fluxway.example.recommendation.sparse_spmd import SparseEmbeddingSPMD
from fluxway.example.recommendation.dense_spmd import DenseSPMD

from fluxway import launch


async def _print_resource_allocation_summary(controller):
    summary = await controller.get_resource_allocation_summary.remote()
    if not summary:
        print("[Fluxway] Controller resource summary is empty.")
        return

    print("[Fluxway] Controller resource summary:")
    for spmd_id in sorted(summary):
        item = summary[spmd_id]
        print(
            f"  - {spmd_id}: bundles={item['bundle_count']} "
            f"workers={item['worker_count']} total_cpu={item['total_cpu']} "
            f"total_memory_gb={item['total_memory_gb']} "
            f"driver={item['driver_bundle']} worker={item['worker_bundle']}"
        )


async def run_training(args):
    cpu_runtime_env = {
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    }
    sparse_runtime_env = {
        "env_vars": {
            **cpu_runtime_env["env_vars"],
            "JAX_ENABLE_X64": "1",
        }
    }
    gpu_runtime_env = {"env_vars": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}}

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

    reader_mesh = DeviceMesh(device_shape=(1,), axis_names=("worker",))
    input_mesh = DeviceMesh(device_shape=(args.dp,), axis_names=("dp",))
    table1_mesh = DeviceMesh(device_shape=(args.table1_shards,), axis_names=("shard",))
    table2_mesh = DeviceMesh(device_shape=(args.table2_shards,), axis_names=("shard",))
    dense_mesh = DeviceMesh(device_shape=(args.dp,), axis_names=("dp",))

    with FunctionalGraph("End-to-End_Training_Flow") as graph:
        reader = RecommendationReader(
            name="TrainingReader", mesh=reader_mesh, runtime_env=cpu_runtime_env
        )
        input_node = InputSPMD(
            name="InputProcessor", mesh=input_mesh, runtime_env=cpu_runtime_env
        )
        table1 = SparseEmbeddingSPMD(
            name="EmbeddingTable1",
            mesh=table1_mesh,
            vocab_size=2000,
            embed_dim=16,
            num_threads=args.num_threads,
            runtime_env=sparse_runtime_env,
        )
        table2 = SparseEmbeddingSPMD(
            name="EmbeddingTable2",
            mesh=table2_mesh,
            vocab_size=2000,
            embed_dim=16,
            num_threads=args.num_threads,
            runtime_env=sparse_runtime_env,
        )
        dense = DenseSPMD(
            name="DenseLayers",
            mesh=dense_mesh,
            embed_dim=16,
            dense_stats_dim=2,
            runtime_env=gpu_runtime_env,
            resource={"num_gpus": 1},
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

        task_graph = graph.compile()

    print(f"[Fluxway] Graph compiled. Nodes: {list(task_graph.nodes.keys())}")

    async_step_config = AsyncStepConfig(max_steps=args.max_steps)
    checkpoint_config = CheckpointConfig(
        force_write=True,
        backend="orbax",
        ckpt_dir="/tmp/fluxway_recsys_ckpt",
        interval_steps=args.checkpoint_interval,
    )

    controller = await create_controller(
        task_graph,
        async_step_config=async_step_config,
        checkpoint_config=checkpoint_config,
    )
    await _print_resource_allocation_summary(controller)

    restored_step = await controller.restore_checkpoint.remote()
    if restored_step is not None:
        print(f"[Fluxway] Resumed from checkpoint at step {restored_step}.")

    print(f"[Fluxway] Running to reach total {args.max_steps} steps …")
    completed = await controller.wait_for_completion.remote()

    await controller.stop_checkpointer.remote()

    if completed:
        print("[Fluxway] Training completed up to target steps.")
    else:
        print("[Fluxway] Training failed or timed out.")


def main():
    parser = argparse.ArgumentParser(description="Fluxway Recommendation Example")
    parser.add_argument(
        "--max-steps", type=int, default=10, help="Max number of steps to run"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2,
        dest="checkpoint_interval",
        help="Save a checkpoint every N steps",
    )
    parser.add_argument(
        "--dp",
        type=int,
        default=4,
        help="Number of data-parallel ranks for DenseSPMD (one GPU per rank)",
    )
    parser.add_argument(
        "--table1-shards",
        type=int,
        default=2,
        dest="table1_shards",
        help="Number of shards for EmbeddingTable1",
    )
    parser.add_argument(
        "--table2-shards",
        type=int,
        default=2,
        dest="table2_shards",
        help="Number of shards for EmbeddingTable2",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        dest="num_threads",
        help="Number of threads for SparseEmbedding store",
    )
    launch(
        run_training,
        parser.parse_args(),
    )


if __name__ == "__main__":
    main()
