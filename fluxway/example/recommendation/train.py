import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
from pathlib import Path
import time

from fluxway.common.mesh import DeviceMesh
from fluxway.common.decorators import spmd_source
from fluxway.common.transfer import WholeUnitRouting
from fluxway.example.recommendation.source_adapter import RecommendationSourceDescriptor
from fluxway.common.types.configs import AsyncStepConfig
import fluxway
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.control_plane.checkpointer.directory import CheckpointDirectory
from fluxway.common.types import CheckpointConfig, SourceControlConfig
from fluxway.profiler import ProfilerDiagnostic
from fluxway.placement import (
    AFFINITY_FIRST,
    PlacementRecipe,
    ResourceProfile,
    equal,
    rank,
    support,
    workers,
)

from fluxway import run_in_sandbox


async def _print_resource_allocation_summary(execution):
    summary = execution.allocation
    if not summary:
        print("[Fluxway] Controller resource summary is empty.")
        return

    print(f"[Fluxway] Controller resource summary: recipe={summary['recipe_id']}")
    counts = {}
    for assignment in summary["assignments"]:
        key = (assignment["spmd_id"], assignment["node_id"])
        counts[key] = counts.get(key, 0) + 1
    for (spmd_id, node_id), worker_count in sorted(counts.items()):
        print(f"  - {spmd_id}: node={node_id} workers={worker_count}")


def _build_profiler_config(args):
    if args.profiler_trace_dir is None:
        return None

    from fluxway.profiler import ProfilerConfig

    run_id = args.profiler_run_id
    if run_id is None:
        run_id = f"recommendation_{time.time_ns()}"

    config = ProfilerConfig(
        trace_dir=args.profiler_trace_dir,
        run_id=run_id,
        buffer_bytes_per_process=args.profiler_buffer_mib * 1024 * 1024,
        clock_sync_rounds=args.profiler_clock_sync_rounds,
        max_clock_uncertainty_ns=args.profiler_max_clock_uncertainty_us * 1000,
        output_file=args.profiler_output_file,
        diagnostics=frozenset(
            ProfilerDiagnostic(item) for item in args.profiler_diagnostics
        ),
        skip_initial_steps=args.profiler_skip_initial_steps,
        steady_state_only=args.profiler_steady_state_only,
    )
    config.run_dir.mkdir(parents=True, exist_ok=False)
    return config


def _apply_data_plane_config(task_graph, args) -> None:
    batch_max_frames = args.data_plane_batch_max_frames
    batch_flush_us = args.data_plane_batch_flush_us
    max_in_flight = args.data_plane_max_in_flight
    if batch_max_frames is not None and batch_max_frames <= 0:
        raise ValueError(
            f"--data-plane-batch-max-frames must be positive, got {batch_max_frames}."
        )
    if batch_flush_us is not None and batch_flush_us < 0:
        raise ValueError(
            f"--data-plane-batch-flush-us must be non-negative, got {batch_flush_us}."
        )
    if max_in_flight is not None and max_in_flight <= 0:
        raise ValueError(
            f"--data-plane-max-in-flight must be positive, got {max_in_flight}."
        )
    for node in task_graph.nodes.values():
        config = node.spmd_config
        if batch_max_frames is not None:
            config.data_plane_batch_max_frames = batch_max_frames
        if batch_flush_us is not None:
            config.data_plane_batch_flush_us = batch_flush_us
        if max_in_flight is not None:
            config.data_plane_max_in_flight = max_in_flight


def _build_ray_init_kwargs(args):
    ray_init_kwargs = {}
    if args.disable_ray_dashboard:
        ray_init_kwargs["include_dashboard"] = False
    if args.ray_num_cpus is not None:
        ray_init_kwargs["num_cpus"] = args.ray_num_cpus
    if args.ray_num_gpus is not None:
        ray_init_kwargs["num_gpus"] = args.ray_num_gpus
    return ray_init_kwargs or None


def _clear_checkpoint_dir(checkpoint_dir: str | Path) -> None:
    CheckpointDirectory.clear(checkpoint_dir)


def _jax_runtime_env(platform: str) -> dict[str, dict[str, str]]:
    return {
        "env_vars": {
            "JAX_PLATFORMS": platform,
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "NCCL_MULTI_RANK_GPU_ENABLE": "1",
        }
    }


def _dense_runtime_env(num_gpus: int) -> dict[str, dict[str, str]]:
    if num_gpus < 0:
        raise ValueError(f"num_gpus must be non-negative, got {num_gpus}")
    return _jax_runtime_env("cuda" if num_gpus > 0 else "cpu")


def _export_graph_visualization(task_graph, args) -> None:
    if args.graph_visualization_dir is None:
        return
    from fluxway import visualize_graph

    bundle = visualize_graph(
        task_graph,
        args.graph_visualization_dir,
        overwrite=args.overwrite_graph_visualization,
    )
    print(f"[Fluxway] Graph visualization bundle: {bundle}")
    print(f"[Fluxway] View with: model-explorer {bundle / 'model_explorer.json'}")


async def run_training(args):
    from fluxway.example.recommendation.reader import RecommendationReader
    from fluxway.example.recommendation.input_spmd import InputSPMD
    from fluxway.example.recommendation.sparse_spmd import SparseEmbeddingSPMD
    from fluxway.example.recommendation.dense_spmd import DenseSPMD
    from fluxway.example.recommendation.sparse_checkpoint import (
        compact_sparse_checkpoints,
    )
    from fluxway.common.bind import bind_spmd
    from fluxway.integrations.sparse_embedding.checkpoint import (
        LocalLanceStorageFactory,
        SparseEmbeddingCheckpointComponent,
        CheckpointSaveMode,
    )

    if getattr(args, "compact_only", False):
        for spmd_id in ("EmbeddingTable1", "EmbeddingTable2"):
            print(
                f"[Fluxway] Compacting sparse checkpoints for {spmd_id} in {args.checkpoint_dir} …"
            )
            await compact_sparse_checkpoints(
                repository_root=args.checkpoint_dir,
                spmd_id=spmd_id,
                lineage_name="embedding",
            )
        print("[Fluxway] Sparse checkpoint compaction completed.")
        return

    if args.num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {args.num_epochs}")
    if args.dense_step_delay_ms < 0:
        raise ValueError(
            f"--dense-step-delay-ms must be non-negative, got {args.dense_step_delay_ms}."
        )
    cpu_runtime_env = _jax_runtime_env("cpu")
    dense_runtime_env = _dense_runtime_env(args.num_gpus)

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

    reader_mesh = DeviceMesh(device_shape=(args.reader_shards,), axis_names=("worker",))
    input_mesh = DeviceMesh(device_shape=(args.dp,), axis_names=("dp",))
    table1_mesh = DeviceMesh(device_shape=(args.table1_shards,), axis_names=("shard",))
    table2_mesh = DeviceMesh(device_shape=(args.table2_shards,), axis_names=("shard",))
    dense_mesh = DeviceMesh(device_shape=(args.dp,), axis_names=("dp",))

    reader_type = spmd_source(
        resource_profile=RecommendationReader.get_resource_profile(),
        mesh_axis_names=("worker",),
        interleaved=True,
        interleave_unit_size=args.dp,
        execution_groups=tuple(
            tuple(
                lane * (args.reader_shards // args.dp) + slot for lane in range(args.dp)
            )
            for slot in range(args.reader_shards // args.dp)
        ),
        source=RecommendationSourceDescriptor(),
    )(type("TrainingReader", (RecommendationReader,), {"__module__": __name__}))

    with FunctionalGraph("End-to-End_Training_Flow") as graph:
        reader = reader_type(
            name="TrainingReader",
            mesh=reader_mesh,
            runtime_env=cpu_runtime_env,
            num_epochs=args.num_epochs,
        )
        input_node = InputSPMD(
            name="InputProcessor", mesh=input_mesh, runtime_env=cpu_runtime_env
        )
        table1_type = bind_spmd(
            SparseEmbeddingSPMD,
            resource_profile=SparseEmbeddingSPMD.get_resource_profile(),
            mesh_axis_names=("shard",),
            checkpoint_components=(
                SparseEmbeddingCheckpointComponent(
                    LocalLanceStorageFactory(),
                    weight_group="table1",
                    lineage_name="embedding",
                    mode=CheckpointSaveMode.INCREMENTAL,
                    initial_mode=CheckpointSaveMode.FULL,
                ).registration(),
            ),
        )
        table2_type = bind_spmd(
            SparseEmbeddingSPMD,
            resource_profile=SparseEmbeddingSPMD.get_resource_profile(),
            mesh_axis_names=("shard",),
            checkpoint_components=(
                SparseEmbeddingCheckpointComponent(
                    LocalLanceStorageFactory(),
                    weight_group="table2",
                    lineage_name="embedding",
                    mode=CheckpointSaveMode.INCREMENTAL,
                    initial_mode=CheckpointSaveMode.FULL,
                ).registration(),
            ),
        )
        table1 = table1_type(
            name="EmbeddingTable1",
            mesh=table1_mesh,
            vocab_size=2000,
            embed_dim=16,
            num_threads=args.num_threads,
            value_dtype=args.dtype,
            runtime_env=cpu_runtime_env,
        )
        table2 = table2_type(
            name="EmbeddingTable2",
            mesh=table2_mesh,
            vocab_size=2000,
            embed_dim=16,
            num_threads=args.num_threads,
            value_dtype=args.dtype,
            runtime_env=cpu_runtime_env,
        )
        dense = DenseSPMD(
            name="DenseLayers",
            mesh=dense_mesh,
            embed_dim=16,
            dense_stats_dim=2,
            diagnostic_step_delay_ms=args.dense_step_delay_ms,
            runtime_env=dense_runtime_env,
            resource_profile=ResourceProfile(
                base={
                    "CPU": 1,
                    "GPU": args.num_gpus,
                    "memory": 1024**3,
                }
            ),
        )

        h_raw = reader.read()
        h_proc = input_node.process(
            h_raw, unit_routing=WholeUnitRouting(tuple(range(args.dp)))
        )

        h_emb1 = table1.lookup(
            unique_ids=h_proc.unique_user_ids  # type: ignore
        ).sparse_features
        h_emb2 = table2.lookup(
            unique_ids=h_proc.unique_target_ids  # type: ignore
        ).sparse_features

        h_prepare = dense.prepare_dense_inputs(
            emb1=h_emb1,
            inv1=h_proc.user_inv,  # type: ignore
            emb2=h_emb2,
            inv2=h_proc.item_inv,  # type: ignore
            dp_data=h_proc.dp_data,  # type: ignore
        )

        h_core = dense.train_core(
            f1_rec=h_prepare.f1_rec,  # type: ignore
            f2_rec=h_prepare.f2_rec,  # type: ignore
            dense_stats=h_prepare.dense_stats,  # type: ignore
            labels=h_prepare.labels,  # type: ignore
            local=True,
        )

        h_scatter = dense.scatter_sparse_grads(
            grad_f1=h_core.grad_f1,  # type: ignore
            grad_f2=h_core.grad_f2,  # type: ignore
            dense_grad=h_core.dense_grad,  # type: ignore
            emb1_ref=h_prepare.emb1_ref,  # type: ignore
            emb2_ref=h_prepare.emb2_ref,  # type: ignore
            inv1_dense=h_prepare.inv1_dense,  # type: ignore
            inv2_dense=h_prepare.inv2_dense,  # type: ignore
            local=True,
        )

        table1.update(grads=h_scatter.grad1)  # type: ignore
        table2.update(grads=h_scatter.grad2)  # type: ignore
        dense.update(dense_grad=h_scatter.dense_grad, local=True)  # type: ignore

        input_dense_constraints = tuple(
            equal(
                support(rank(workers(InputSPMD), i)),
                support(rank(workers(DenseSPMD), i)),
            )
            for i in range(args.dp)
        )
        task_graph = graph.build()
        reader_pairs = set()
        for dependency in task_graph.require_compiled_graph().dependencies:
            if dependency.spec.source.spmd_id != "TrainingReader":
                continue
            for transfer in dependency.transfers:
                relation = dependency.unit_relation(transfer.transfer_code)
                if relation is None:
                    raise ValueError(
                        "recommendation Reader requires compiled unit routing"
                    )
                for group in dependency.source_operator.execution_layout.cycle:
                    for consumer, producers in enumerate(relation.producers_by_target):
                        reader_pairs.update(
                            (group[producer], consumer) for producer in producers
                        )
        placement_recipe = PlacementRecipe(
            recipe_id="recommendation-training",
            constraints=(
                *(
                    equal(
                        support(rank(workers(reader_type), producer)),
                        support(rank(workers(InputSPMD), consumer)),
                    )
                    for producer, consumer in sorted(reader_pairs)
                ),
                *input_dense_constraints,
            ),
            objective=AFFINITY_FIRST,
        )

    _apply_data_plane_config(task_graph, args)
    _export_graph_visualization(task_graph, args)
    print(f"[Fluxway] Graph compiled. Nodes: {list(task_graph.nodes.keys())}")

    async_step_config = AsyncStepConfig(max_steps=args.max_steps)
    source_max_in_flight = args.source_max_in_flight
    source_control = (
        SourceControlConfig(
            max_in_flight_steps=source_max_in_flight,
        )
        if task_graph.requires_source_control_runtime
        else None
    )
    checkpoint_config = (
        CheckpointConfig(
            repository_dir=args.checkpoint_dir,
            interval_steps=args.checkpoint_interval,
        )
        if args.checkpoint_interval > 0
        else None
    )
    profiler_config = _build_profiler_config(args)
    if profiler_config is not None:
        print(f"[Fluxway] Profiler trace run directory: {profiler_config.run_dir}")
        print(f"[Fluxway] Profiler trace output file: {profiler_config.output_path}")

    execution = await fluxway.start(
        task_graph,
        placement_recipe=placement_recipe,
        async_step_config=async_step_config,
        source_control=source_control,
        checkpoint_config=checkpoint_config,
        profiler_config=profiler_config,
        resume="latest" if checkpoint_config is not None else None,
    )
    await _print_resource_allocation_summary(execution)

    restored_step = execution.restored_step
    if restored_step is not None:
        print(f"[Fluxway] Resumed from checkpoint at step {restored_step}.")

    print(f"[Fluxway] Running to reach total {args.max_steps} steps …")
    await execution.wait()

    await execution.cancel()
    print("[Fluxway] Training completed up to max steps.")

    if getattr(args, "compact_checkpoints", False):
        print(
            "[Fluxway] Training finished. Triggering manual sparse checkpoint compaction …"
        )
        for spmd_id in ("EmbeddingTable1", "EmbeddingTable2"):
            print(
                f"[Fluxway] Compacting sparse checkpoints for {spmd_id} in {args.checkpoint_dir} …"
            )
            await compact_sparse_checkpoints(
                repository_root=args.checkpoint_dir,
                spmd_id=spmd_id,
                lineage_name="embedding",
            )
        print("[Fluxway] Manual sparse checkpoint compaction completed.")


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
        "--num-epochs",
        type=int,
        default=1,
        help="Number of synthetic reader epochs",
    )
    parser.add_argument(
        "--source-max-in-flight",
        type=int,
        default=4,
        help="Maximum logical source work steps acquired in flight",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/tmp/fluxway_recsys_ckpt",
        help="Checkpoint directory for this training run",
    )
    parser.add_argument(
        "--reader-shards",
        type=int,
        default=4,
        help="Number of Reader SPMD workers",
    )
    parser.add_argument(
        "--dp",
        type=int,
        default=4,
        help="Number of data-parallel ranks for DenseSPMD",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs per DenseSPMD rank (default: 1, use 0 for CPU-only execution)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for embeddings (default: bfloat16)",
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
    parser.add_argument(
        "--dense-step-delay-ms",
        type=int,
        default=0,
        help="Diagnostic delay inside DenseSPMD.train_core for overlap proof runs",
    )
    parser.add_argument(
        "--clear-ckpt",
        action="store_true",
        help="Clear existing checkpoints before starting",
    )
    parser.add_argument(
        "--profiler-trace-dir",
        default=None,
        help="Enable profiler and configure the MPMD trace output root directory",
    )
    parser.add_argument(
        "--graph-visualization-dir",
        default=None,
        help="Export a Model Explorer graph bundle after graph compilation",
    )
    parser.add_argument(
        "--overwrite-graph-visualization",
        action="store_true",
        help="Replace an existing verified Fluxway graph bundle",
    )
    parser.add_argument(
        "--profiler-run-id",
        default=None,
        help="Profiler run id under --profiler-trace-dir",
    )
    parser.add_argument(
        "--profiler-buffer-mib",
        type=int,
        default=64,
        help="Profiler arena size per process in MiB",
    )
    parser.add_argument(
        "--profiler-clock-sync-rounds",
        type=int,
        default=8,
        help="Clock sync ping-pong rounds per controller/driver/worker edge",
    )
    parser.add_argument(
        "--profiler-max-clock-uncertainty-us",
        type=int,
        default=1000,
        help="Maximum accepted physical clock uncertainty in microseconds",
    )
    parser.add_argument(
        "--profiler-output-file",
        default=None,
        help="Profiler Perfetto trace output path",
    )
    parser.add_argument(
        "--profiler-diagnostics",
        action="append",
        choices=[item.value for item in ProfilerDiagnostic],
        default=[],
        help="Optional non-hot diagnostic trace domains to include",
    )
    parser.add_argument(
        "--profiler-skip-initial-steps",
        type=int,
        default=1,
        help="Drop step-level profiler events before this step index",
    )
    parser.add_argument(
        "--profiler-steady-state-only",
        action="store_true",
        help="Drop lifecycle events from the exported profiler trace",
    )
    parser.add_argument(
        "--data-plane-batch-max-frames",
        type=int,
        default=None,
        help="Override max transport frames per batch RPC for this run",
    )
    parser.add_argument(
        "--data-plane-batch-flush-us",
        type=int,
        default=None,
        help="Override data-plane batch flush window in microseconds for this run",
    )
    parser.add_argument(
        "--data-plane-max-in-flight",
        type=int,
        default=None,
        help="Override max in-flight background delivery tasks per worker",
    )
    parser.add_argument(
        "--disable-ray-dashboard",
        action="store_true",
        help="Start local Ray without the dashboard/agent HTTP server",
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=None,
        help="Override local ray.init num_cpus for this example run",
    )
    parser.add_argument(
        "--ray-num-gpus",
        type=int,
        default=None,
        help="Override local ray.init num_gpus for this example run",
    )
    parser.add_argument(
        "--compact-checkpoints",
        action="store_true",
        help="Manually trigger sparse embedding checkpoint compaction after training",
    )
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="Manually trigger sparse embedding checkpoint compaction without training",
    )
    args = parser.parse_args()
    if args.clear_ckpt:
        _clear_checkpoint_dir(args.checkpoint_dir)

    run_in_sandbox(
        run_training,
        args,
        ray_init_kwargs=_build_ray_init_kwargs(args),
    )


if __name__ == "__main__":
    main()
