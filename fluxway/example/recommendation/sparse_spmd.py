from typing import Any
import os

import numpy as np
import sparse_embedding as se  # type: ignore
from sparse_embedding.checkpoint import CheckpointSaveMode

from fluxway.common.tensor_backend import get_backend, prepare_dlpack_input
from fluxway.common.decorators import spmd, operator, requires_worker_process
from fluxway.common.operator_policy import AsyncPolicy, SchedulePolicy, StatePolicy
from fluxway.common.utils.logging import get_logger
from fluxway.common.types.tensor import TransportTensor
from fluxway import ResourceProfile
from fluxway.example.recommendation.data_types import LookupOutput
from fluxway.integrations.sparse_embedding.checkpoint import (
    LocalLanceStorageFactory,
    SparseEmbeddingCheckpointComponent,
)

logger = get_logger(__name__)


def _balanced_counts(total: int, shard_count: int) -> list[int]:
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    if shard_count <= 0:
        raise ValueError(f"shard_count must be positive, got {shard_count}")
    base, rem = divmod(total, shard_count)
    return [base + (1 if shard_id < rem else 0) for shard_id in range(shard_count)]


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1, "memory": 1024**3}),
    mesh_axis_names=("shard",),
    checkpoint_components=(
        SparseEmbeddingCheckpointComponent(
            LocalLanceStorageFactory(),
            lineage_name="embedding",
            mode=CheckpointSaveMode.INCREMENTAL,
            initial_mode=CheckpointSaveMode.FULL,
        ).registration(),
    ),
)
@requires_worker_process
class SparseEmbeddingSPMD:
    rank: int
    world_size: int
    ctx: "Any"
    """Sharded embedding table backed by sparse_embedding.EmbeddingStore.

    Each shard owns the embeddings for keys hash-routed to it.  Forward lookup
    uses EmbeddingStore.lookup_or_insert (lazy initialisation on first access),
    and the backward pass accumulates gradients with Adam then calls
    optimizer_step once per training step.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 3,
        num_threads: int | None = None,
        name: str = "",
        mesh: "Any" = None,
        runtime_env: "Any" = None,
        resource: "Any" = None,
        **kwargs: "Any",
    ):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_threads = num_threads or os.cpu_count() or 4

        # -- Invariants on configuration --
        assert isinstance(embed_dim, int) and embed_dim > 0, (
            f"[SparseSPMD] embed_dim must be a positive int, got {embed_dim!r}"
        )
        assert isinstance(vocab_size, int) and vocab_size > 0, (
            f"[SparseSPMD] vocab_size must be a positive int, got {vocab_size!r}"
        )
        assert self.num_threads > 0, (
            f"[SparseSPMD] num_threads must be positive, got {self.num_threads}"
        )

        self.store = se.EmbeddingStore(
            embedding_config=se.EmbeddingConfig(
                dims=[embed_dim],
                thread_budget=self.num_threads,
                value_dtype="float32",
            ),
            initializer_config=se.Uniform(min_val=-0.01, max_val=0.01),
            optimizer_config=se.Adam(lr=0.01),
        )
        internal_shard_count = self.store.internal_shard_count()
        reserve_counts = _balanced_counts(vocab_size, internal_shard_count)
        self.store.reserve_per_shard(reserve_counts, dim=embed_dim)
        self._reserved_key_capacity = sum(reserve_counts)
        self._internal_shard_count = internal_shard_count
        self._forward_keys_cache: dict[int, np.ndarray] = {}

        logger.debug(
            "SparseEmbeddingSPMD_init rank=%s world_size=%s vocab_size=%s embed_dim=%s reserve_capacity=%s internal_shards=%s",
            getattr(self, "rank", None),
            getattr(self, "world_size", None),
            vocab_size,
            embed_dim,
            self._reserved_key_capacity,
            self._internal_shard_count,
        )

    # Forward: lookup
    # ------------------------------------------------------------------

    @operator(
        name="lookup",
        async_policy=AsyncPolicy.bounded_stale(
            depth=8,
            max_staleness=7,
            commit="relaxed",
            schedule=SchedulePolicy.deadline_boost(),
            state=StatePolicy.thread_safe(),
        ),
        data_argument=["unique_ids"],
        input_sharding={
            "unique_ids": {
                "type": "dynamic",
                "axis": ["shard"],
                "strategy": "modulo",
            }
        },
        output_sharding={
            "sparse_features": {
                "type": "mirror",
                "reference": "lookup.unique_ids",
                "axis": [],
            }
        },
    )
    def lookup(self, unique_ids: TransportTensor) -> LookupOutput:
        step = self.ctx.step
        ids_np = np.asarray(unique_ids.flat_values, dtype=np.int64).ravel()
        ids_np = prepare_dlpack_input(ids_np)

        # -- Invariants on input ids --
        assert ids_np.ndim == 1, (
            f"[SparseSPMD.lookup] ids must be 1D after ravel, got shape={ids_np.shape}"
        )
        assert ids_np.size >= 0, (
            f"[SparseSPMD.lookup] ids size must be non-negative, got {ids_np.size}"
        )

        # Lazy-insert new keys, return embeddings for all keys in this shard.
        vals_dlm, _ = self.store.lookup_or_insert(ids_np, dim=self.embed_dim)
        backend = get_backend()
        emb_np = backend.from_dlpack(vals_dlm).reshape(ids_np.size, self.embed_dim)

        # -- Invariants on lookup output --
        assert emb_np.shape == (ids_np.size, self.embed_dim), (
            f"[SparseSPMD.lookup] emb shape {emb_np.shape} "
            f"!= expected ({ids_np.size}, {self.embed_dim})"
        )

        self._forward_keys_cache[step] = ids_np

        # -- Numerical correctness check: lookup keys match received unique_ids --
        np.testing.assert_array_equal(
            ids_np,
            unique_ids.flat_values.ravel().astype(np.int64),
            err_msg=f"[SparseSPMD] rank={self.rank} lookup keys mismatch unique_ids",
        )

        # Mirror auto-inherit: the operator declares ``output_sharding`` as
        # ``mirror reference="lookup.unique_ids"``, so the framework's worker
        # hook (see ``Worker._apply_mirror_inheritance``) automatically copies
        # both ``receipt`` (the SymmetryReceipt that drives mirror routing)
        # and ``origin_indices`` from the cached ``unique_ids`` source onto
        # this output.  Manually re-attaching ``unique_ids.origin_indices``
        # here is therefore redundant -- and risky, since any future change
        # in row count would silently mismatch the framework's length check
        # in ``Distributor`` (see communication.py "origin_indices length
        # mismatch" error).  Leave the output naked and let mirror inherit.
        output = TransportTensor(emb_np)
        print(
            f"[SparseEmbeddingSPMD.lookup] rank={self.rank} step={step} "
            f"keys={ids_np} emb_norm={float(np.linalg.norm(emb_np)):.4f}"
        )
        return LookupOutput(sparse_features=output)

    # ------------------------------------------------------------------
    # Backward: accumulate gradients and take an optimizer step
    # ------------------------------------------------------------------

    @operator(
        name="update",
        async_policy=AsyncPolicy.bounded_stale(
            depth=8,
            max_staleness=7,
            commit="relaxed",
            schedule=SchedulePolicy.deadline_boost(boost_priority=200),
            state=StatePolicy.thread_safe(),
        ),
        data_argument=["grads"],
        input_sharding={
            "grads": {"type": "mirror", "reference": "lookup.sparse_features"}
        },
    )
    def update(self, grads: TransportTensor) -> None:
        step = self.ctx.step
        grad_step = grads.source_step
        if grad_step is None:
            raise ValueError(
                f"SparseEmbeddingSPMD rank={self.rank}: "
                "grads.source_step is required for bounded-stale update"
            )
        keys_np = self._forward_keys_cache.pop(grad_step, None)

        if keys_np is None:
            raise ValueError(
                f"SparseEmbeddingSPMD rank={self.rank}: "
                f"missing forward cache for gradient source step {grad_step}"
            )

        grads_np = prepare_dlpack_input(
            grads.flat_values.reshape(keys_np.size, self.embed_dim)
        )

        # -- Numerical correctness check: gradient layout matches keys --
        assert grads_np.shape == (keys_np.size, self.embed_dim), (
            f"[SparseSPMD] update grads shape {grads_np.shape} "
            f"!= expected ({keys_np.size}, {self.embed_dim})"
        )
        assert keys_np.ndim == 1, (
            f"[SparseSPMD] update keys must be 1D, got shape={keys_np.shape}"
        )

        self.store.accumulate_gradients(keys_np, grads_np)
        self.store.optimizer_step(1)

        print(
            f"[SparseEmbeddingSPMD.update] rank={self.rank} step={step} "
            f"grad_norm={float(np.linalg.norm(grads_np)):.4f}"
        )
