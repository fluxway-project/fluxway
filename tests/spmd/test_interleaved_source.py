import numpy as np
import pytest
import asyncio
from unittest.mock import AsyncMock, patch

from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.common.types.configs import (
    SPMDConfig,
    OperatorMeta,
    EdgeShardingInfo,
    ArgumentType,
    SPMDTopology,
    SPMDTopologyInfo,
)
from fluxway.common.types.sharding import ShardingSpec, PartitionSpec
from fluxway.common.mesh import DeviceMesh


def _meta_to_topology(meta: OperatorMeta) -> dict[str, SPMDTopology]:
    """Helper for unit tests to derive topology from OperatorMeta."""
    top_dict = {}
    top = SPMDTopology()
    for pred_id, edges in meta.input_pre_sharding.items():
        if edges:
            e = edges[0]
            top.predecessors[pred_id] = SPMDTopologyInfo(
                spmd_id=pred_id,
                tuuid_sharding=e.tuuid_sharding,
                tuuid_map=e.tuuid_map,
                is_interleaved=e.is_interleaved,
                interleave_unit_size=e.interleave_unit_size,
                mesh=e.mesh,
                world_size=e.mesh.size if getattr(e, "mesh", None) else 0,
            )
    for succ_id, edges in meta.output_post_sharding.items():
        if edges:
            e = edges[0]
            top.successors[succ_id] = SPMDTopologyInfo(
                spmd_id=succ_id,
                tuuid_sharding=e.tuuid_sharding,
                tuuid_map=e.tuuid_map,
                is_interleaved=e.is_interleaved,
                interleave_unit_size=e.interleave_unit_size,
                mesh=e.mesh,
                world_size=e.mesh.size if getattr(e, "mesh", None) else 0,
            )
    top_dict[meta.name] = top
    return top_dict


@pytest.mark.asyncio
async def test_multi_worker_interleaved_unit_mesh_inheritance():
    """
    Verify that interleaved units correctly inherit the sub-mesh structure.
    Loader (4 ranks, mesh (2,2)) -> Processor (2 ranks)
    """
    loader_mesh = DeviceMesh(
        axis_names=("data", "mp"), device_shape=(2, 2), device_ids=[0, 1, 2, 3]
    )
    processor_mesh = DeviceMesh(
        axis_names=("data",), device_shape=(2,), device_ids=[0, 1]
    )

    sharding = ShardingSpec.chunk(PartitionSpec("data", "mp"))
    edge_info = EdgeShardingInfo(
        is_interleaved=True,
        interleave_unit_size=2,
        tuuid_sharding={1001: sharding},
        tuuid_map={"t1": 1001},
        mesh=loader_mesh,
    )

    loader_meta = OperatorMeta(
        name="loader",
        is_source=True,
        is_interleaved=True,
        interleave_unit_size=2,
        output_arguments={ArgumentType.DATA.value: ["t1"]},
        output_uuids=[1001],
        output_sharding={1001: sharding},
        output_post_sharding={"processor": [edge_info]},
    )

    processor_meta = OperatorMeta(
        name="processor",
        input_arguments={ArgumentType.DATA.value: ["t1"]},
        input_sharding={"t1": ShardingSpec.chunk(PartitionSpec())},
        input_pre_sharding={"loader": [edge_info]},
    )

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker

    def dummy_processor(t1: np.ndarray = None):
        pass

    # Instantiate Loader Workers
    loader_workers = []
    for rank in range(4):
        config = SPMDConfig(
            mesh=loader_mesh,
            spmd_id="loader",
            operators={"loader": lambda: {"t1": np.ones((10, 10))}},
            operator_metas={"loader": loader_meta, "processor": processor_meta},
            topology=_meta_to_topology(loader_meta),
        )
        w = cls(config, rank=rank)
        w._successor_handlers = {"processor": {0: AsyncMock(), 1: AsyncMock()}}
        loader_workers.append(w)

    # Instantiate Processor Workers (for shard count check)
    processor_workers = []
    for rank in range(2):
        config = SPMDConfig(
            mesh=processor_mesh,
            spmd_id="processor",
            operators={"processor": dummy_processor},
            operator_metas={"loader": loader_meta, "processor": processor_meta},
            topology=_meta_to_topology(processor_meta),
        )
        w = cls(config, rank=rank)
        processor_workers.append(w)

    shards_dict = processor_workers[0]._op_manager.get_expected_tensor_shards(
        "processor"
    )
    assert (
        shards_dict[1001] == 2
    ), f"Target processor should expect 2 shards, got {shards_dict}"

    # 2. RUN STEP 0: Unit 0 (Ranks 0, 1) active
    with patch("fluxway.spmd.worker.worker.AsyncDataDistributor") as mock_dist_cls:
        mock_dist = mock_dist_cls.return_value
        mock_dist.distribute = AsyncMock()

        await asyncio.gather(
            *[w._execute_operator("loader", 0) for w in loader_workers]
        )

        assert mock_dist.plan_and_shard.call_count == 2
        assert mock_dist.distribute.call_count == 2
        found_u0 = False
        for call in mock_dist.plan_and_shard.call_args_list:
            source_spec = call.args[1]
            if list(source_spec.mesh.device_ids.flatten()) == [0, 1]:
                assert source_spec.mesh.shape == (1, 2)
                assert source_spec.mesh.axis_names == ("data", "mp")
                found_u0 = True
        assert found_u0, "Unit 0 sub-mesh distribution not found"

    # 3. RUN STEP 1: Unit 1 (Ranks 2, 3) active
    with patch("fluxway.spmd.worker.worker.AsyncDataDistributor") as mock_dist_cls:
        mock_dist = mock_dist_cls.return_value
        mock_dist.distribute = AsyncMock()

        await asyncio.gather(
            *[w._execute_operator("loader", 1) for w in loader_workers]
        )

        assert mock_dist.plan_and_shard.call_count == 2
        assert mock_dist.distribute.call_count == 2
        found_u1 = False
        for call in mock_dist.plan_and_shard.call_args_list:
            source_spec = call.args[1]
            if list(source_spec.mesh.device_ids.flatten()) == [2, 3]:
                found_u1 = True
        assert found_u1, "Unit 1 sub-mesh distribution not found"


@pytest.mark.asyncio
async def test_single_worker_interleaved_unit():
    loader_mesh = DeviceMesh(axis_names=("data",), device_shape=(4,))
    sharding = ShardingSpec.chunk(PartitionSpec("data"))
    edge = EdgeShardingInfo(
        is_interleaved=True,
        interleave_unit_size=4,
        tuuid_sharding={2001: sharding},
        tuuid_map={"t1": 2001},
        mesh=loader_mesh,
    )
    loader_meta = OperatorMeta(
        name="loader",
        is_source=True,
        is_interleaved=True,
        interleave_unit_size=1,
        output_arguments={ArgumentType.DATA.value: ["t1"]},
        output_uuids=[2001],
        output_sharding={2001: sharding},
        output_post_sharding={"processor": [edge]},
    )

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker

    loader_workers = []
    for rank in range(4):
        config = SPMDConfig(
            mesh=loader_mesh,
            spmd_id="loader",
            operators={"loader": lambda: {"t1": np.zeros(1)}},
            operator_metas={"loader": loader_meta},
            topology=_meta_to_topology(loader_meta),
        )
        w = cls(config, rank=rank)
        w._successor_handlers = {"processor": {0: AsyncMock()}}
        loader_workers.append(w)

    with patch("fluxway.spmd.worker.worker.AsyncDataDistributor") as mock_dist_cls:
        mock_dist = mock_dist_cls.return_value
        mock_dist.distribute = AsyncMock()

        await asyncio.gather(
            *[w._execute_operator("loader", 0) for w in loader_workers]
        )
        assert mock_dist.plan_and_shard.call_count == 1
        assert mock_dist.distribute.call_count == 1

        source_spec = mock_dist.plan_and_shard.call_args.args[1]
        assert source_spec.mesh.shape == (1,)
        assert list(source_spec.mesh.device_ids.flatten()) == [0]
