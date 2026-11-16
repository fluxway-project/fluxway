from typing import Any
import asyncio

import numpy as np
import sparse_embedding as se  # type: ignore

import fluxway as fw
from fluxway.common.tensor_backend import get_backend, prepare_dlpack_input
from fluxway.common.decorators import spmd, operator, requires_worker_process
from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.operator_policy import AsyncPolicy, StatePolicy
from fluxway.common.tensor_components import (
    dense_path_strides,
    nested_global_positions_kernel,
    nested_path_position_maps,
)
from fluxway.common.types.tensor import TransportTensor
from fluxway import ResourceProfile
from .data_types import UserBatch, InputNodeOutput, DPFeatures


def _unique_global_flat_positions(
    nested_splits,
    flat_first_indices,
    widths,
    origin_indices,
    group_name,
):
    if len(nested_splits) != len(widths):
        raise ValueError(
            f"[InputSPMD._sparse_unique:{group_name}] nested levels "
            f"{len(nested_splits)} != dense widths {len(widths)}."
        )
    local_rows, path_offsets = nested_path_position_maps(
        nested_splits, flat_first_indices, widths
    )
    if local_rows.size and int(local_rows.max()) >= origin_indices.shape[0]:
        raise ValueError(
            f"[InputSPMD._sparse_unique:{group_name}] local row index "
            f"{int(local_rows.max())} exceeds origin length {origin_indices.shape[0]}."
        )
    out = np.empty(flat_first_indices.shape[0], dtype=np.int64)
    nested_global_positions_kernel(
        local_rows.astype(np.int64, copy=False),
        path_offsets,
        origin_indices,
        dense_path_strides(widths),
        out,
    )
    return out


def _materialize_sparse_unique_tensors(
    unique_raw: Any,
    inv_raw: Any,
    first_raw: Any,
    original_ids: np.ndarray,
    global_shape: tuple[int, ...],
    nested_splits: tuple[np.ndarray, ...],
    source_origins: np.ndarray,
    group_name: str,
) -> tuple[TransportTensor, TransportTensor]:
    if hasattr(unique_raw, "__dlpack__") or hasattr(unique_raw, "__dlpack_device__"):
        backend = get_backend()
        unique_np = backend.from_dlpack(unique_raw).astype(np.int64)
        inv_1d = backend.from_dlpack(inv_raw).astype(np.int32).reshape(-1)
        first_local = backend.from_dlpack(first_raw).astype(np.int64).reshape(-1)
    else:
        unique_np = np.asarray(unique_raw, dtype=np.int64)
        inv_1d = np.asarray(inv_raw, dtype=np.int32).reshape(-1)
        first_local = np.asarray(first_raw, dtype=np.int64).reshape(-1)

    assert unique_np.ndim == 1, (
        f"[InputSPMD._sparse_unique:{group_name}] unique must be 1D, got shape={unique_np.shape}"
    )
    assert np.unique(unique_np).size == unique_np.size, (
        f"[InputSPMD._sparse_unique:{group_name}] async_unique output contains duplicates"
    )
    assert first_local.shape == unique_np.shape, (
        f"[InputSPMD._sparse_unique:{group_name}] first_indices shape {first_local.shape} "
        f"!= unique shape {unique_np.shape}"
    )
    if first_local.size > 1:
        assert bool(np.all(np.diff(first_local) > 0)), (
            f"[InputSPMD._sparse_unique:{group_name}] first_indices not strictly monotonic; "
            "async_unique's first-occurrence ordering contract was violated"
        )

    assert inv_1d.shape[0] == original_ids.shape[0], (
        f"[InputSPMD._sparse_unique:{group_name}] inv length {inv_1d.shape[0]} "
        f"!= original length {original_ids.shape[0]}"
    )
    if inv_1d.size > 0:
        assert int(inv_1d.min()) >= 0 and int(inv_1d.max()) < unique_np.size, (
            f"[InputSPMD._sparse_unique:{group_name}] inv out of range: "
            f"min={int(inv_1d.min())}, max={int(inv_1d.max())}, "
            f"unique_size={unique_np.size}"
        )
    np.testing.assert_array_equal(
        unique_np[inv_1d],
        original_ids,
        err_msg=f"[InputSPMD:{group_name}] unique+inv reconstruction failed",
    )
    np.testing.assert_array_equal(
        original_ids[first_local],
        unique_np,
        err_msg=(f"[InputSPMD:{group_name}] first_indices contract violated"),
    )

    widths = tuple(w for w in global_shape[1:])
    origins = _unique_global_flat_positions(
        nested_splits,
        first_local,
        widths,
        source_origins,
        group_name,
    )
    unique_global_shape = (int(np.prod(global_shape, dtype=np.int64)),)

    unique_tt = TransportTensor(
        flat_values=unique_np,
        origin_indices=ExplicitIndex(origins),
        global_shape=unique_global_shape,
    )
    inv_tt = TransportTensor(
        flat_values=inv_1d.astype(np.int64, copy=False),
        nested_splits=list(nested_splits),
        global_shape=global_shape,
    )
    return unique_tt, inv_tt


async def _sparse_unique(
    tt: TransportTensor,
    rank: int,
    world_size: int,
    group_name: str,
) -> tuple[TransportTensor, TransportTensor]:
    """Deduplicate ids via sparse_embedding.async_unique (C++ high-perf).

    Returns ``(unique_tt, inv_tt)`` carrying the *minimum* metadata required
    for correct cross-mesh reassembly:

    * ``unique_tt`` carries a per-row ``ExplicitIndex`` inherited from the
      source tensor's origin lineage at each unique row's first occurrence.
      This retentions dynamic unique ids, embedding lookup output, and the inverse
      tensor tied to the same logical input rows across steps.
    * ``inv_tt`` retentions ``global_shape`` only; the framework's static-DP
      path handles its placement deterministically, no extra meta required.
    """
    ids_np = tt.flat_values

    # -- Invariants on input layout --
    assert ids_np is not None, (
        f"[InputSPMD._sparse_unique:{group_name}] ids_np must not be None, got "
        f"shape={getattr(ids_np, 'shape', None)}"
    )
    assert isinstance(world_size, int) and world_size >= 1, (
        f"[InputSPMD._sparse_unique:{group_name}] world_size must be a positive int, "
        f"got {world_size!r}"
    )
    assert isinstance(rank, int) and 0 <= rank < world_size, (
        f"[InputSPMD._sparse_unique:{group_name}] rank {rank!r} out of range "
        f"[0, world_size={world_size})"
    )

    source_origins = tt.origin_indices_materialized
    if source_origins is None or len(source_origins) == 0:
        raise ValueError(
            f"[InputSPMD._sparse_unique:{group_name}] sparse ids require non-empty "
            "input origin_indices."
        )
    if not tt.nested_splits:
        raise ValueError(
            f"[InputSPMD._sparse_unique:{group_name}] sparse ids must use the "
            "row-partitioned contract: flat_values is an occurrence stream and "
            "nested_splits[0] maps batch rows to occurrence ranges."
        )
    if not tt.global_shape or len(tt.global_shape) != len(tt.nested_splits) + 1:
        raise ValueError(
            f"[InputSPMD._sparse_unique:{group_name}] row-partitioned sparse ids "
            f"require global_shape=(global_rows, *nested_widths), got {tt.global_shape}."
        )
    if len(tt.nested_splits[0]) - 1 != len(source_origins):
        raise ValueError(
            f"[InputSPMD._sparse_unique:{group_name}] origin length "
            f"{len(source_origins)} != sparse row count {len(tt.nested_splits[0]) - 1}."
        )
    nested_splits = tuple(np.asarray(s, dtype=np.int64) for s in tt.nested_splits)

    if ids_np.size == 0:
        unique_raw = np.empty((0,), dtype=np.int64)
        inv_raw = np.empty((0,), dtype=np.int32)
        first_raw = np.empty((0,), dtype=np.int64)
    else:
        unique_raw, inv_raw, first_raw = await se.async_unique(
            prepare_dlpack_input(ids_np.ravel()),
            return_inverse=True,
            return_first_indices=True,
        )

    original_ids = ids_np.ravel()
    return _materialize_sparse_unique_tensors(
        unique_raw,
        inv_raw,
        first_raw,
        original_ids,
        tt.global_shape,
        nested_splits,
        np.asarray(source_origins, dtype=np.int64),
        group_name,
    )


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1, "memory": 1024**3}),
    mesh_axis_names=("dp",),
)
@requires_worker_process
class InputSPMD:
    rank: int
    world_size: int
    ctx: "Any"
    """Deduplicates features via sparse_embedding.async_unique and routes to embedding tables."""

    def __init__(
        self,
        name: str = "",
        mesh: "Any" = None,
        runtime_env: "Any" = None,
        resource: "Any" = None,
        **kwargs: "Any",
    ):
        pass

    @operator(
        name="process",
        async_policy=AsyncPolicy.pipelined(
            depth=4,
            state=StatePolicy.thread_safe(),
        ),
        input_sharding={"data": {"type": "static", "axis": ["dp"]}},
        output_sharding={
            "unique_user_ids": {
                "type": "dynamic",
                "axis": ["dp"],
                "strategy": "modulo",
            },
            "unique_target_ids": {
                "type": "dynamic",
                "axis": ["dp"],
                "strategy": "modulo",
            },
            "user_inv": {"type": "static", "axis": ["dp"]},
            "item_inv": {"type": "static", "axis": ["dp"]},
            "dp_data": {"type": "static", "axis": ["dp"]},
        },
    )
    async def process(self, data: UserBatch) -> InputNodeOutput:
        # -- Invariants on input feature layout --
        assert data is not None, "[InputSPMD.process] data is None"
        assert data.active.user_id.flat_values.ndim >= 1, (
            f"[InputSPMD.process] user_id must be at least 1D, "
            f"got shape={data.active.user_id.flat_values.shape}"
        )
        assert data.passive.target_target_id.flat_values.ndim >= 1, (
            f"[InputSPMD.process] target_target_id must be at least 1D, "
            f"got shape={data.passive.target_target_id.flat_values.shape}"
        )

        user_task = _sparse_unique(
            data.active.user_id,
            rank=self.rank,
            world_size=self.world_size,
            group_name="user",
        )
        item_task = _sparse_unique(
            data.passive.target_target_id,
            rank=self.rank,
            world_size=self.world_size,
            group_name="item",
        )

        with fw.profiler.scope(
            "input_process.sparse_unique",
            dp_rank=self.rank,
            dp_world_size=self.world_size,
        ):
            (u_user, inv_user), (u_item, inv_item) = await asyncio.gather(
                user_task, item_task
            )

        fw.profiler.event(
            "input_process.unique_ready",
            unique_user_count=int(u_user.flat_values.size),
            unique_item_count=int(u_item.flat_values.size),
        )

        print(
            f"[InputSPMD.process] rank={self.rank} "
            f"unique_users={u_user.flat_values} user_inv={inv_user.flat_values} "
            f"unique_items={u_item.flat_values} item_inv={inv_item.flat_values}"
        )

        return InputNodeOutput(
            unique_user_ids=u_user,
            user_inv=inv_user,
            unique_target_ids=u_item,
            item_inv=inv_item,
            dp_data=DPFeatures(
                labels=data.labels, dense_stats=data.passive.target_item_stats
            ),
        )
