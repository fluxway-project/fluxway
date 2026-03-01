import jax.numpy as jnp
from typing import Any, Dict, Optional
from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from fluxway.reader import GrainReader
from .data_types import UserBatch, ActiveFeatures, PassiveFeatures, LabelFeatures


@spmd(mesh_axis_names=("worker",), mesh_axis_shapes=(1,))
class RecommendationReader(GrainReader):
    """
    Refactored Reader SPMD for Recommendation Example.
    Uses col-store (Active/Passive/Label) schema.
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
            # Synthetic dataset following the Active/Passive/Label pattern
            # Note: We use lists to represent ragged/variable length data
            data = []
            for i in range(100):
                sample = {
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
                data.append(sample)
            static_args["data_source"] = data
            static_args.setdefault("batch_size", 4)
            static_args.setdefault("num_epochs", 1)
            static_args.setdefault("shuffle", False)

        super()._create_data_loader(
            static_args, rank, world_size, shard_index, shard_count
        )

    @operator(
        "read",
        output_sharding={
            "active": {"type": "static", "axis": ["worker"]},
            "passive": {"type": "static", "axis": ["worker"]},
            "labels": {"type": "static", "axis": ["worker"]},
        },
    )
    def read(self) -> Optional[UserBatch]:
        """Produces a batch of UserBatch."""
        static_args = getattr(self, "static_args", {})
        batch_size = static_args.get("batch_size", 4)
        samples = []

        for i in range(batch_size):
            item = self.next_item()
            if item is None:
                break
            samples.append(item)

        if not samples:
            return None

        # Calculate global batch size
        global_batch_size = batch_size * self.world_size
        g_shape = (global_batch_size,)

        # 1. Active Features
        active = ActiveFeatures(
            user_id=TransportTensor.from_lists(
                [s["user_id"] for s in samples], dtype=jnp.int32, global_shape=g_shape
            ),
            user_age_level=TransportTensor.from_lists(
                [s["user_age_level"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
            ),
            request_hour=TransportTensor.from_lists(
                [s["request_hour"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
            ),
            hist_click_item_seq=TransportTensor.from_lists(
                [s["hist_click_item_seq"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
            ),
            hist_seq_length=TransportTensor.from_lists(
                [[s["hist_seq_length"]] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
            ),
        )

        # 2. Passive Features
        passive = PassiveFeatures(
            target_item_id=TransportTensor.from_lists(
                [s["target_item_id"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
            ),
            target_item_category=TransportTensor.from_lists(
                [s["target_item_category"] for s in samples],
                dtype=jnp.int32,
                global_shape=g_shape,
            ),
            target_item_stats=TransportTensor.from_lists(
                [s["target_item_stats"] for s in samples],
                dtype=jnp.float32,
                global_shape=g_shape,
            ),
        )

        # 3. Label Features
        labels = LabelFeatures(
            is_click=TransportTensor.from_lists(
                [s["is_click"] for s in samples], dtype=jnp.int32, global_shape=g_shape
            )
        )

        return UserBatch(active=active, passive=passive, labels=labels)
