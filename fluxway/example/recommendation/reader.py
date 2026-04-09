import json
import pathlib
import jax.numpy as jnp
from typing import Any, Dict, Optional

from fluxway.common.decorators import spmd_source, operator
from fluxway.common.index_descriptor import ContinuousIndex
from fluxway.common.types.tensor import TransportTensor
from fluxway.reader import GrainReader
from .data_types import UserBatch, ActiveFeatures, PassiveFeatures, LabelFeatures


@spmd_source(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("worker",),
    mesh_axis_shapes=(1,),
    interleaved=True,
    interleave_unit_size=1,
)
class RecommendationReader(GrainReader):
    """
    Produces batches of UserBatch in col-store layout (Active / Passive / Label).
    When no data_source is provided, generates a 1000-sample synthetic dataset.
    """

    def _create_data_loader(
        self,
        static_args: Dict[str, Any],
        rank: int,
        world_size: int,
        shard_index: Optional[int] = None,
        shard_count: Optional[int] = None,
    ):
        if "data_source" not in static_args:
            data = [
                {
                    "user_id": [1000 + i],
                    "user_age_level": [i % 5],
                    "request_hour": [18],
                    "hist_click_item_seq": [200, 201, 202] if i % 2 == 0 else [210],
                    "hist_seq_length": 3 if i % 2 == 0 else 1,
                    "target_item_id": [500 + i],
                    "target_item_category": [i % 10],
                    "target_item_stats": [0.05, 19.9],
                    "is_click": [1 if i % 3 == 0 else 0],
                }
                for i in range(1000)
            ]
            static_args["data_source"] = data
            static_args.setdefault("batch_size", 24)
            static_args.setdefault("num_epochs", 1)
            static_args.setdefault("shuffle", False)

        super()._create_data_loader(
            static_args, rank, world_size, shard_index, shard_count
        )
        self._completed_steps = 0

    @operator(
        "read",
        output_sharding={
            "active": {"type": "static", "axis": ["worker"]},
            "passive": {"type": "static", "axis": ["worker"]},
            "labels": {"type": "static", "axis": ["worker"]},
        },
    )
    def read(self) -> Optional[UserBatch]:
        static_args = getattr(self, "static_args", {})
        batch_size = static_args.get("batch_size", 24)

        samples = []
        for _ in range(batch_size):
            item = self.next_item()
            if item is None:
                break
            samples.append(item)

        if not samples:
            return None

        self._completed_steps += 1

        logical_world = self.world_size if hasattr(self, "world_size") else 1
        global_batch_size = batch_size * logical_world
        g_shape = (global_batch_size,)
        origins = ContinuousIndex.chunk(0, global_batch_size)

        active = ActiveFeatures(
            user_id=TransportTensor.from_lists(
                [s["user_id"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            user_age_level=TransportTensor.from_lists(
                [s["user_age_level"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            request_hour=TransportTensor.from_lists(
                [s["request_hour"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            hist_click_item_seq=TransportTensor.from_lists(
                [s["hist_click_item_seq"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            hist_seq_length=TransportTensor.from_lists(
                [[s["hist_seq_length"]] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
        )

        passive = PassiveFeatures(
            target_item_id=TransportTensor.from_lists(
                [s["target_item_id"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            target_item_category=TransportTensor.from_lists(
                [s["target_item_category"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            target_item_stats=TransportTensor.from_lists(
                [s["target_item_stats"] for s in samples],
                dtype=jnp.float32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
        )

        labels = LabelFeatures(
            is_click=TransportTensor.from_lists(
                [s["is_click"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
                origin_indices=origins,
            )
        )

        return UserBatch(active=active, passive=passive, labels=labels)

    def save_state(self, ckpt_dir: str) -> Dict[str, Any]:
        shard_count = getattr(self, "world_size", 1)
        static_args = getattr(self, "static_args", {})
        path = pathlib.Path(ckpt_dir) / f"reader_rank_{self.rank}_of_{shard_count}.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "completed_steps": getattr(self, "_completed_steps", 0),
                    "batch_size": static_args.get("batch_size", 24),
                },
                f,
            )
        return {}

    def restore_state(self, state: Dict[str, Any], ckpt_dir: str) -> None:
        # completed_steps is identical across all ranks, so take the max.
        # Pair it with the saved batch_size to get the exact items-per-rank to skip,
        # regardless of the new world_size or batch_size.
        completed_steps = 0
        saved_batch_size = None
        for p in sorted(pathlib.Path(ckpt_dir).glob("reader_rank_*_of_*.json")):
            with open(p) as f:
                data = json.load(f)
            steps = data.get("completed_steps", 0)
            if steps > completed_steps:
                completed_steps = steps
                saved_batch_size = data.get("batch_size")

        new_shard_count = getattr(self, "world_size", 1)
        static_args = getattr(self, "static_args", {})
        if saved_batch_size is None:
            saved_batch_size = static_args.get("batch_size", 24)

        per_rank_skip = completed_steps * saved_batch_size

        self._create_data_loader(static_args, getattr(self, "rank", 0), new_shard_count)
        for _ in range(per_rank_skip):
            self.next_item()

        self._completed_steps = completed_steps
