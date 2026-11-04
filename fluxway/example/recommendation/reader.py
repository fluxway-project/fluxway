import numpy as np
from typing import Any

from fluxway.common.decorators import spmd_source, operator, requires_worker_process
from fluxway.common.operator_policy import AsyncPolicy
from fluxway.example.recommendation.source_adapter import RecommendationSourceDescriptor
from fluxway.reader import BaseReader
from fluxway import ResourceProfile
from .data_types import UserBatch, ActiveFeatures, PassiveFeatures, LabelFeatures


@requires_worker_process
@spmd_source(
    resource_profile=ResourceProfile(base={"CPU": 1, "memory": 1024**3}),
    mesh_axis_names=("worker",),
    interleaved=True,
    interleave_unit_size=1,
    source=RecommendationSourceDescriptor(),
)
class RecommendationReader(BaseReader):
    rank: int
    world_size: int
    """
    Produces batches of UserBatch in col-store layout (Active / Passive / Label).
    When no data_source is provided, generates a 1000-sample synthetic dataset.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._data_source = self._create_data_source(self.static_args)

    def _create_data_source(
        self,
        static_args: dict[str, Any],
    ):
        if "data_source" not in static_args:
            # Synthesize a recommendation corpus whose item / user popularity
            # follows a power-law (Zipf-like) distribution, mirroring real
            # traffic where a small set of "hot" entities dominates impressions.
            # This guarantees that different DP ranks observe overlapping sparse
            # IDs and thereby exercise the cross-DP embedding routing path
            # (e.g. the same item shows up on multiple input workers, gets routed to a single
            # sparse worker, and its embedding fans out to multiple dense
            # workers).
            seed = static_args.get("seed", 42)
            rng = np.random.default_rng(seed)

            vocab_size_user = static_args.get("vocab_size_user", 100)
            vocab_size_item = static_args.get("vocab_size_item", 2000)
            num_samples = static_args.get("num_samples", 1000)
            item_alpha = static_args.get("item_zipf_alpha", 1.2)
            user_alpha = static_args.get("user_zipf_alpha", 1.0)
            max_seq_len = static_args.get("max_seq_len", 5)

            item_probs = self._zipf_probs(vocab_size_item, item_alpha, rng)
            user_probs = self._zipf_probs(vocab_size_user, user_alpha, rng)

            # Heavier-tailed sequence length: most users have short history,
            # a few have longer ones.
            seq_len_choices = np.arange(1, max_seq_len + 1)
            seq_len_probs = 1.0 / seq_len_choices.astype(np.float64)
            seq_len_probs /= seq_len_probs.sum()

            user_ids = rng.choice(vocab_size_user, size=num_samples, p=user_probs)
            target_target_ids = rng.choice(
                vocab_size_item, size=num_samples, p=item_probs
            )
            seq_lens = rng.choice(seq_len_choices, size=num_samples, p=seq_len_probs)

            data = []
            for i in range(num_samples):
                sl = int(seq_lens[i])
                hist_seq = rng.choice(vocab_size_item, size=sl, p=item_probs).tolist()

                data.append(
                    {
                        "user_id": [int(user_ids[i])],
                        "user_age_level": [int(rng.integers(0, 5))],
                        "request_hour": [int(rng.integers(0, 24))],
                        "hist_click_item_seq": [int(x) for x in hist_seq],
                        "hist_seq_length": [sl],
                        "target_target_id": [int(target_target_ids[i])],
                        "target_item_category": [int(rng.integers(0, 10))],
                        "target_item_stats": [
                            rng.random(),
                            (rng.random() * 20),
                        ],
                        "is_click": [int(rng.integers(0, 2))],
                    }
                )

            return data
        return static_args["data_source"]

    @staticmethod
    def _zipf_probs(
        vocab_size: int, alpha: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Return a permuted Zipf-like probability vector over [0, vocab_size).

        The raw 1 / (rank + 1) ** alpha mass is then permuted so that "hot"
        IDs are scattered across the vocabulary instead of being clustered at
        small numeric values.  This decouples ID popularity from ID encoding,
        which is closer to real-world scenarios and prevents the downstream
        ``ModuloPartitioner`` from skewing all hot IDs onto a single shard.
        """
        ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
        p = 1.0 / (ranks**alpha)
        p /= p.sum()
        perm = rng.permutation(vocab_size)
        out = np.empty_like(p)
        out[perm] = p
        return out

    @operator(
        "read",
        async_policy=AsyncPolicy.pipelined(depth=4),
        output_sharding={
            "active": {"type": "static", "axis": ["worker"]},
            "passive": {"type": "static", "axis": ["worker"]},
            "labels": {"type": "static", "axis": ["worker"]},
        },
    )
    def read(self) -> UserBatch | None:
        if self.ctx is None:
            raise RuntimeError(
                "RecommendationReader.read requires an active OperatorContext"
            )
        work = self.ctx.source_work_items
        if not work:
            raise RuntimeError("RecommendationReader requires acquired source work")
        samples = [
            self._data_source[index % len(self._data_source)]
            for item in work
            for index in range(item.logical_start, item.logical_end)
        ]

        # -- Invariants on per-sample structure --
        required_keys = (
            "user_id",
            "user_age_level",
            "request_hour",
            "hist_click_item_seq",
            "hist_seq_length",
            "target_target_id",
            "target_item_category",
            "target_item_stats",
            "is_click",
        )
        for k in required_keys:
            assert k in samples[0], f"[Reader] sample missing required key '{k}'"

        # Determine feature dimensions from first sample
        stats_dim = len(samples[0]["target_item_stats"])
        assert stats_dim > 0, (
            f"[Reader] target_item_stats must be non-empty, got dim={stats_dim}"
        )
        # All samples must agree on stats dimensionality.
        for i, s in enumerate(samples):
            assert len(s["target_item_stats"]) == stats_dim, (
                f"[Reader] sample {i} target_item_stats dim "
                f"{len(s['target_item_stats'])} != expected {stats_dim}"
            )
            assert s["is_click"][0] in (0, 1), (
                f"[Reader] sample {i} is_click must be 0 or 1, got {s['is_click'][0]}"
            )

        active = ActiveFeatures(
            user_id=self.create_row_sparse_tensor(
                [[s["user_id"][0]] for s in samples], width=1, dtype=np.int32
            ),
            user_age_level=self.create_tensor(
                [s["user_age_level"][0] for s in samples],
                dtype=np.int32,
                uniform_dense_partition=True,
            ),
            request_hour=self.create_tensor(
                [s["request_hour"][0] for s in samples],
                dtype=np.int32,
                uniform_dense_partition=True,
            ),
            hist_click_item_seq=self.create_row_sparse_tensor(
                [s["hist_click_item_seq"] for s in samples],
                width=self.static_args.get("max_seq_len", 5),
                dtype=np.int32,
            ),
            hist_seq_length=self.create_tensor(
                [s["hist_seq_length"][0] for s in samples],
                dtype=np.int32,
                uniform_dense_partition=True,
            ),
        )

        passive = PassiveFeatures(
            target_target_id=self.create_row_sparse_tensor(
                [
                    [
                        s["target_target_id"][0],
                        *s["hist_click_item_seq"],
                    ]
                    for s in samples
                ],
                width=self.static_args.get("max_seq_len", 5) + 1,
                dtype=np.int32,
            ),
            target_item_category=self.create_tensor(
                [s["target_item_category"][0] for s in samples],
                dtype=np.int32,
                uniform_dense_partition=True,
            ),
            target_item_stats=self.create_tensor(
                [s["target_item_stats"] for s in samples],
                dtype=np.float32,
                uniform_dense_partition=True,
            ),
        )

        labels = LabelFeatures(
            is_click=self.create_tensor(
                [s["is_click"][0] for s in samples],
                dtype=np.int32,
                uniform_dense_partition=True,
            )
        )

        return UserBatch(active=active, passive=passive, labels=labels)
