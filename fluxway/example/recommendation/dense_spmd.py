from typing import Any
import time
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from numba import njit

from fluxway.common.decorators import spmd, operator, requires_worker_process
from fluxway.common.operator_policy import AsyncPolicy, SchedulePolicy, StatePolicy
from fluxway.common.tensor_components import (
    dense_path_strides,
    fill_nested_dense_inv_with_pad,
)
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.tensor_backend import get_backend
from fluxway.common.types import SaveSession, RestoreSession
from fluxway.control_plane.checkpointer import OrbaxCheckpointComponent
from fluxway import ResourceProfile
from .data_types import DPFeatures

_DENSE_PHASE_ORDER = (
    "prepare_dense_inputs",
    "train_core",
    "scatter_sparse_grads",
    "update",
)
_DENSE_PHASE_DEPENDENCIES = (
    ("prepare_dense_inputs", "train_core", 0),
    ("train_core", "scatter_sparse_grads", 0),
    ("train_core", "update", 0),
    ("update", "train_core", 1),
)
_DENSE_LANE_ORDER = ("compute", "data")
_DENSE_LANE_PRIORITIES = {"compute": 1000, "data": 100}
_DENSE_PHASE_LANES = {
    "prepare_dense_inputs": "data",
    "train_core": "compute",
    "scatter_sparse_grads": "data",
    "update": "compute",
}
_DENSE_PHASE_PRIORITIES = {
    "train_core": 500,
    "update": 490,
    "prepare_dense_inputs": 300,
    "scatter_sparse_grads": 20,
}


def _dense_phase_schedule(phase: str) -> SchedulePolicy:
    if phase not in _DENSE_PHASE_LANES:
        raise ValueError(f"Unknown Dense phase {phase!r}.")
    return SchedulePolicy.phase_graph(
        phase=phase,
        phase_order=_DENSE_PHASE_ORDER,
        dependencies=_DENSE_PHASE_DEPENDENCIES,
        lane=_DENSE_PHASE_LANES[phase],
        lane_order=_DENSE_LANE_ORDER,
        lane_priorities=_DENSE_LANE_PRIORITIES,
        phase_priorities=_DENSE_PHASE_PRIORITIES,
    )


def _xp() -> Any:
    platform = jax.default_backend()
    if platform == "cpu":
        return np
    if platform == "gpu":
        import cupy

        return cupy
    raise ValueError(f"unsupported Dense JAX platform {platform!r}")


def _local_jax_to_native(value: jax.Array) -> Any:
    if not isinstance(value, jax.Array):
        raise TypeError("Dense export requires a JAX array")
    shards = value.addressable_shards
    if len(shards) != 1:
        raise ValueError("Dense export requires exactly one addressable shard")
    local = shards[0].data
    platform = shards[0].device.platform
    if platform not in ("cpu", "gpu"):
        raise ValueError(f"unsupported Dense export platform {platform!r}")
    device = "cuda" if platform == "gpu" else "cpu"
    return get_backend().asarray(local, copy=False, device=device)


def _place_gradient_like(gradient: jax.Array, reference: jax.Array) -> jax.Array:
    if not isinstance(reference, jax.Array):
        raise TypeError(
            f"gradient reference must be a jax.Array, got {type(reference).__name__}"
        )
    if gradient.shape != reference.shape:
        raise ValueError(
            f"gradient shape {gradient.shape} does not match reference {reference.shape}"
        )
    if gradient.dtype != reference.dtype:
        raise TypeError(
            f"gradient dtype {gradient.dtype} does not match reference {reference.dtype}"
        )
    addressable_gradients = tuple(
        jax.device_put(gradient, shard.device) for shard in reference.addressable_shards
    )
    if not addressable_gradients:
        raise RuntimeError("gradient reference has no addressable shards")
    return jax.make_array_from_single_device_arrays(
        shape=reference.shape,
        sharding=reference.sharding,
        arrays=addressable_gradients,
    )


def _host_array(x, dtype=None):
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x, dtype=dtype)


# ---------------------------------------------------------------------------
# Numba JIT kernels for CPU acceleration
# ---------------------------------------------------------------------------


@njit(nogil=True, cache=True)
def _gather_2d_kernel(src, indices, out):
    """out[i, :] = src[indices[i], :] — 2D gather by row index.

    Replaces NumPy fancy indexing ``src[indices]`` with a compiled loop
    for consistent CPU throughput on large batches.
    """
    for i in range(indices.shape[0]):
        idx = indices[i]
        for d in range(src.shape[1]):
            out[i, d] = src[idx, d]


@njit(nogil=True, cache=True)
def _gather_2d_kernel_u16(src, indices, out):
    """Bitwise 16-bit 2D gather for half-precision floating point types."""
    for i in range(indices.shape[0]):
        idx = indices[i]
        for d in range(src.shape[1]):
            out[i, d] = src[idx, d]


@njit(nogil=True, cache=True)
def _scatter_add_2d_kernel(out, idx, vals):
    """In-place ``out[idx] += vals`` for 2D ``out``/``vals`` and 1D ``idx``.

    Serial implementation to preserve np.add.at semantics under duplicate indices.
    """
    n = idx.shape[0]
    d = vals.shape[1]
    for i in range(n):
        k = idx[i]
        for j in range(d):
            out[k, j] += vals[i, j]


def _dense_inverse_for_sparse_group(
    inv_tt: TransportTensor,
    pad_index: int,
    batch_size: int,
    gname: str = "",
):
    if not inv_tt.nested_splits:
        inv_dense = _host_array(inv_tt.flat_values)
        if inv_dense.shape[0] != batch_size:
            raise ValueError(
                f"Sparse inverse for group {gname} is dense but first dim "
                f"{inv_dense.shape[0]} != batch_size {batch_size}."
            )
        return inv_dense

    if (
        not inv_tt.global_shape
        or len(inv_tt.global_shape) != len(inv_tt.nested_splits) + 1
    ):
        raise ValueError(
            f"Sparse inverse for group {gname} is ragged but lacks dense consumer "
            f"global_shape. global_shape={inv_tt.global_shape}, "
            f"nested_levels={len(inv_tt.nested_splits)}."
        )

    widths = tuple(w for w in inv_tt.global_shape[1:])
    if inv_tt.global_shape[0] % batch_size != 0:
        raise ValueError(
            f"Sparse inverse for group {gname} global rows {inv_tt.global_shape[0]} "
            f"are not divisible by local batch_size {batch_size}."
        )
    if len(inv_tt.nested_splits[0]) - 1 != batch_size:
        raise ValueError(
            f"Sparse inverse for group {gname} has {len(inv_tt.nested_splits[0]) - 1} "
            f"ragged rows, expected batch_size {batch_size}."
        )

    out = np.empty((batch_size, *widths), dtype=np.int64)
    splits = tuple(_host_array(s, dtype=np.int64) for s in inv_tt.nested_splits)
    fill_nested_dense_inv_with_pad(
        _host_array(inv_tt.flat_values, dtype=np.int64),
        splits,
        dense_path_strides(widths),
        pad_index,
        out,
    )
    return out


def _gather_sparse_input(emb_flat, inv, batch_size, emb_dim, gname=""):
    """Reconstruct batch tensor from unique embeddings via inverse indices.

    Equivalent to ``emb_2d[inv_1d].reshape(batch_size, -1, emb_dim)``
    but uses Numba on CPU and CuPy fancy indexing on GPU for zero-copy
    data paths.
    """
    xp = _xp()

    emb_2d = xp.asarray(emb_flat).reshape(-1, emb_dim)
    pad_index = emb_2d.shape[0]
    if isinstance(inv, TransportTensor):
        inv_dense = _dense_inverse_for_sparse_group(inv, pad_index, batch_size, gname)
    else:
        inv_dense = inv
    inv_1d = xp.asarray(inv_dense).reshape(-1)

    if pad_index == 0:
        raise ValueError(
            f"Sparse embedding unique payload 'emb_flat' for group {gname} cannot be empty! "
            f"emb_flat.shape={emb_flat.shape}, inv_shape={getattr(inv_dense, 'shape', None)}, "
            f"batch_size={batch_size}, emb_dim={emb_dim}. "
            "This indicates a bug in the fluxway routing layer."
        )

    if inv_1d.size > 0:
        inv_min = int(xp.min(inv_1d))
        inv_max = int(xp.max(inv_1d))
        if inv_min < 0 or inv_max > pad_index:
            raise ValueError(
                f"Sparse embedding gather index out of range for group {gname}: "
                f"local_index_min={inv_min}, local_index_max={inv_max}, "
                f"emb_rows={emb_2d.shape[0]}. "
                "Valid compact inverse values are unique rows plus one pad row."
            )

    zero_row = xp.zeros((1, emb_dim), dtype=emb_2d.dtype)
    emb_for_gather = xp.concatenate([emb_2d, zero_row], axis=0)

    if xp is not np:
        rec = emb_for_gather[inv_1d]
    else:
        rec = np.empty((inv_1d.shape[0], emb_dim), dtype=emb_2d.dtype)
        if rec.itemsize == 2:
            _gather_2d_kernel_u16(
                np.asarray(emb_for_gather).view(np.uint16),
                np.asarray(inv_1d),
                rec.view(np.uint16),
            )
        else:
            _gather_2d_kernel(np.asarray(emb_for_gather), np.asarray(inv_1d), rec)
    return rec.reshape(batch_size, -1, emb_dim)


def _pooled_grad_slots(inv_dense, pad_index: int, grad_local):
    xp = _xp()
    inv_arr = xp.asarray(inv_dense)
    grad_arr = xp.asarray(grad_local)
    if inv_arr.ndim < 1:
        raise ValueError(
            f"inverse must have at least one dimension, got {inv_arr.shape}"
        )
    slots_per_sample = int(np.prod(tuple(int(v) for v in inv_arr.shape[1:])) or 1)
    grad_slots = xp.repeat(
        grad_arr.reshape(grad_arr.shape[0], 1, -1), slots_per_sample, axis=1
    )
    grad_slots = grad_slots.reshape(-1, grad_arr.shape[-1])
    inv_1d = inv_arr.reshape(-1)
    if inv_1d.size != grad_slots.shape[0]:
        raise ValueError(
            f"inverse slot count {inv_1d.size} does not match expanded gradient "
            f"slot count {grad_slots.shape[0]}"
        )
    if inv_1d.size:
        grad_slots = grad_slots.copy()
        grad_slots[inv_1d == pad_index] = 0
    return grad_slots


def _scatter_add_grad(emb_2d, inv_1d, grad_local):
    """Scatter-add per-sample gradients back to unique embedding rows.

    Equivalent to ``jnp.zeros(shape).at[inv].add(grad)`` but in-place.
    Uses cupyx.scatter_add on GPU and Numba _scatter_add_2d_kernel on CPU.
    """
    xp = _xp()
    if emb_2d.shape[0] == 0:
        raise ValueError(
            f"_scatter_add_grad received empty embedding tensor: "
            f"emb_2d.shape={emb_2d.shape}, inv_1d.shape={inv_1d.shape}, "
            f"grad_local.shape={grad_local.shape}. "
            "This indicates a bug in the fluxway routing layer."
        )

    if xp is not np:
        import cupyx

        if emb_2d.dtype == xp.dtype("bfloat16"):
            g_f32 = xp.zeros(emb_2d.shape, dtype=xp.float32)
            cupyx.scatter_add(
                g_f32,
                xp.asarray(inv_1d),
                xp.asarray(grad_local).astype(xp.float32),
            )
            return g_f32.astype(emb_2d.dtype)

        g_unique = xp.zeros_like(emb_2d)
        cupyx.scatter_add(
            g_unique,
            xp.asarray(inv_1d),
            grad_local,
        )
        return g_unique

    g_unique = xp.zeros_like(emb_2d)
    if emb_2d.dtype in (np.float16, np.dtype("bfloat16")):
        np.add.at(
            np.asarray(g_unique),
            np.asarray(inv_1d),
            np.asarray(grad_local),
        )
    else:
        _scatter_add_2d_kernel(
            np.asarray(g_unique),
            np.asarray(inv_1d),
            np.asarray(grad_local),
        )
    return g_unique


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1, "memory": 1024**3}),
    mesh_axis_names=("dp",),
    checkpoint_components=(
        OrbaxCheckpointComponent(
            "dense",
            lambda owner, context: owner.save_checkpoint_state(context),
            lambda owner, state, context: owner.restore_checkpoint_state(
                state, context
            ),
            lambda owner, context: owner.restore_checkpoint_layout(context),
            finalize_state=lambda owner, context: owner.finalize_checkpoint_state(
                context
            ),
        ).registration(),
    ),
)
@requires_worker_process
class DenseSPMD:
    rank: int
    world_size: int
    ctx: "Any"
    rank0_address: str
    """Final computation node: forward pass, loss, and dense/embedding gradient computation."""

    def __init__(
        self,
        embed_dim: int = 3,
        dense_stats_dim: int = 2,
        grad_accum_steps: int = 2,
        diagnostic_step_delay_ms: int = 0,
        name: str = "",
        mesh: "Any" = None,
        runtime_env: "Any" = None,
        resource: "Any" = None,
        **kwargs: "Any",
    ):
        self.embed_dim = embed_dim
        self.dense_stats_dim = dense_stats_dim
        self.grad_accum_steps = grad_accum_steps
        self.diagnostic_step_delay_ms = diagnostic_step_delay_ms
        self._pending_accum_steps = 0

        # -- Invariants on configuration --
        assert isinstance(embed_dim, int) and embed_dim > 0, (
            f"[DenseSPMD] embed_dim must be a positive int, got {embed_dim!r}"
        )
        assert isinstance(dense_stats_dim, int) and dense_stats_dim > 0, (
            f"[DenseSPMD] dense_stats_dim must be a positive int, got {dense_stats_dim!r}"
        )
        assert isinstance(grad_accum_steps, int) and grad_accum_steps >= 1, (
            f"[DenseSPMD] grad_accum_steps must be >= 1, got {grad_accum_steps!r}"
        )
        assert (
            isinstance(diagnostic_step_delay_ms, int) and diagnostic_step_delay_ms >= 0
        ), (
            "[DenseSPMD] diagnostic_step_delay_ms must be a non-negative int, "
            f"got {diagnostic_step_delay_ms!r}"
        )

        print(
            f"[DenseSPMD] init jax.distributed rank={self.rank} "
            f"world_size={self.world_size} coordinator={self.rank0_address}",
            flush=True,
        )
        jax.distributed.initialize(
            coordinator_address=self.rank0_address,
            num_processes=self.world_size,
            process_id=self.rank,
            local_device_ids=[0],
        )

        n_global = len(jax.devices())
        print(
            f"[DenseSPMD] jax.distributed ok rank={self.rank} "
            f"global_devices={n_global} local_devices={len(jax.local_devices())}"
        )
        self.mesh = Mesh(jax.devices(), ("dp",))

        input_dim = self.embed_dim * 2 + self.dense_stats_dim

        class DenseLayer(nnx.Module):
            def __init__(self, in_dim):
                self.linear = nnx.Linear(in_dim, 1, rngs=nnx.Rngs(0))

            def __call__(self, x):
                return self.linear(x).squeeze(-1)

        self.model = DenseLayer(input_dim)

        # Model params must live on the distributed mesh from creation so that
        # checkpoint save/restore sees a consistent pytree structure (all leaves
        # are globally-sharded JAX arrays, never SingleDeviceSharding).
        replicated = NamedSharding(self.mesh, P())
        state = nnx.state(self.model)
        nnx.update(
            self.model,
            jtu.tree_map(lambda p: jax.device_put(p, replicated), state),
        )
        self.grad_accum = jtu.tree_map(
            lambda p: jax.device_put(jnp.zeros_like(p), replicated),
            nnx.state(self.model),
        )

    def _checkpoint_state(self) -> dict:
        return {"model_state": nnx.state(self.model)}

    def finalize_checkpoint_state(self, context: SaveSession) -> None:
        if self._pending_accum_steps > 0:
            scale = 1.0 / self._pending_accum_steps
            params = nnx.state(self.model)
            updated = jtu.tree_map(
                lambda p, g: p - 0.01 * (g * scale), params, self.grad_accum
            )
            nnx.update(self.model, updated)
            self.grad_accum = jtu.tree_map(jnp.zeros_like, self.grad_accum)
            print(
                f"[DenseSPMD.finalize] rank={self.rank} flushed "
                f"{self._pending_accum_steps} pending accumulated steps into model"
            )
            self._pending_accum_steps = 0

    def save_checkpoint_state(self, context: SaveSession) -> dict:
        return self._checkpoint_state()

    def restore_checkpoint_layout(self, context: RestoreSession) -> dict:
        return self._checkpoint_state()

    def restore_checkpoint_state(self, state: dict, context: RestoreSession) -> None:
        nnx.update(self.model, state["model_state"])
        self.grad_accum = jtu.tree_map(jnp.zeros_like, nnx.state(self.model))
        self._pending_accum_steps = 0

    @operator(
        name="prepare_dense_inputs",
        async_policy=AsyncPolicy.pipelined(
            depth=2,
            schedule=_dense_phase_schedule("prepare_dense_inputs"),
            state=StatePolicy.thread_safe(),
        ),
        input_sharding={
            "emb1": {"type": "dynamic", "axis": ["dp"]},
            "inv1": {"type": "static", "axis": ["dp"]},
            "emb2": {"type": "dynamic", "axis": ["dp"]},
            "inv2": {"type": "static", "axis": ["dp"]},
            "dp_data": {"type": "static", "axis": ["dp"]},
        },
        output_sharding={
            "f1_rec": {"type": "static", "axis": []},
            "f2_rec": {"type": "static", "axis": []},
            "dense_stats": {"type": "static", "axis": []},
            "labels": {"type": "static", "axis": []},
            "emb1_ref": {"type": "static", "axis": []},
            "emb2_ref": {"type": "static", "axis": []},
            "inv1_dense": {"type": "static", "axis": []},
            "inv2_dense": {"type": "static", "axis": []},
        },
    )
    def prepare_dense_inputs(
        self,
        emb1: TransportTensor,
        inv1: TransportTensor,
        emb2: TransportTensor,
        inv2: TransportTensor,
        dp_data: DPFeatures,
    ) -> dict:
        xp = _xp()
        emb1_2d = xp.asarray(emb1.flat_values).reshape(-1, emb1.flat_values.shape[-1])  # type: ignore
        emb2_2d = xp.asarray(emb2.flat_values).reshape(-1, emb2.flat_values.shape[-1])  # type: ignore

        assert emb1_2d.shape[1] == self.embed_dim, (
            f"[DenseSPMD] emb1 last dim {emb1_2d.shape[1]} != configured embed_dim {self.embed_dim}"
        )
        assert emb2_2d.shape[1] == self.embed_dim, (
            f"[DenseSPMD] emb2 last dim {emb2_2d.shape[1]} != configured embed_dim {self.embed_dim}"
        )

        dense_stats = xp.asarray(dp_data.dense_stats.flat_values).reshape(  # type: ignore
            -1, dp_data.dense_stats.flat_values.shape[-1]
        )
        labels_local = (
            xp.asarray(dp_data.labels.is_click.flat_values)  # type: ignore
            .reshape(-1)
            .astype(xp.float32)  # type: ignore
        )
        local_batch = int(dense_stats.shape[0])

        inv1_dense = _dense_inverse_for_sparse_group(
            inv1, emb1_2d.shape[0], local_batch, "f1"
        )
        inv2_dense = _dense_inverse_for_sparse_group(
            inv2, emb2_2d.shape[0], local_batch, "f2"
        )
        inv1_1d = xp.asarray(inv1_dense).reshape(-1)  # type: ignore
        inv2_1d = xp.asarray(inv2_dense).reshape(-1)  # type: ignore

        if inv1_1d.size > 0:
            inv1_min = int(xp.min(inv1_1d))
            inv1_max = int(xp.max(inv1_1d))
            assert inv1_min >= 0 and inv1_max <= emb1_2d.shape[0], (
                f"[DenseSPMD] inv1 out of range: min={inv1_min}, "
                f"max={inv1_max}, emb1_unique_count={emb1_2d.shape[0]}"
            )
        if inv2_1d.size > 0:
            inv2_min = int(xp.min(inv2_1d))
            inv2_max = int(xp.max(inv2_1d))
            assert inv2_min >= 0 and inv2_max <= emb2_2d.shape[0], (
                f"[DenseSPMD] inv2 out of range: min={inv2_min}, "
                f"max={inv2_max}, emb2_unique_count={emb2_2d.shape[0]}"
            )
        assert bool(xp.all(xp.isfinite(emb1_2d))), (  # type: ignore
            "[DenseSPMD] emb1 contains NaN/Inf -- possible cross-mesh "
            "transport corruption between SparseEmbeddingSPMD and DenseSPMD"
        )
        assert bool(xp.all(xp.isfinite(emb2_2d))), (  # type: ignore
            "[DenseSPMD] emb2 contains NaN/Inf -- possible cross-mesh "
            "transport corruption between SparseEmbeddingSPMD and DenseSPMD"
        )

        f1_rec_3d = _gather_sparse_input(
            emb1.flat_values, inv1_dense, local_batch, self.embed_dim, "f1"
        )
        f2_rec_3d = _gather_sparse_input(
            emb2.flat_values, inv2_dense, local_batch, self.embed_dim, "f2"
        )
        f1_rec = xp.sum(f1_rec_3d, axis=1)  # type: ignore
        f2_rec = xp.sum(f2_rec_3d, axis=1)  # type: ignore

        assert dense_stats.shape[1] == self.dense_stats_dim, (
            f"[DenseSPMD] dense_stats last dim {dense_stats.shape[1]} "
            f"!= configured dense_stats_dim {self.dense_stats_dim}"
        )
        assert dense_stats.shape[0] == local_batch, (
            f"[DenseSPMD] dense_stats batch {dense_stats.shape[0]} "
            f"!= local_batch {local_batch}"
        )
        assert labels_local.shape[0] == local_batch, (
            f"[DenseSPMD] labels batch {labels_local.shape[0]} "
            f"!= local_batch {local_batch}"
        )
        assert f1_rec.shape[0] == local_batch, (
            f"[DenseSPMD] f1_rec batch size {f1_rec.shape[0]} != local_batch {local_batch}"
        )
        assert f2_rec.shape[0] == local_batch, (
            f"[DenseSPMD] f2_rec batch size {f2_rec.shape[0]} != local_batch {local_batch}"
        )

        return {
            "f1_rec": f1_rec,
            "f2_rec": f2_rec,
            "dense_stats": dense_stats,
            "labels": labels_local,
            "emb1_ref": emb1.zero_copy_alias(),
            "emb2_ref": emb2.zero_copy_alias(),
            "inv1_dense": inv1_dense,
            "inv2_dense": inv2_dense,
        }

    @operator(
        name="train_core",
        async_policy=AsyncPolicy.pipelined(
            depth=1,
            schedule=_dense_phase_schedule("train_core"),
        ),
        input_sharding={
            "f1_rec": {"type": "static", "axis": []},
            "f2_rec": {"type": "static", "axis": []},
            "dense_stats": {"type": "static", "axis": []},
            "labels": {"type": "static", "axis": []},
        },
        output_sharding={
            "grad_f1": {"type": "static", "axis": []},
            "grad_f2": {"type": "static", "axis": []},
            "dense_grad": {"type": "static", "axis": []},
        },
    )
    def train_core(
        self,
        f1_rec: TransportTensor,
        f2_rec: TransportTensor,
        dense_stats: TransportTensor,
        labels: TransportTensor,
    ) -> dict:
        xp = _xp()
        f1_local = xp.asarray(f1_rec.flat_values)
        f2_local = xp.asarray(f2_rec.flat_values)
        stats_local = xp.asarray(dense_stats.flat_values)
        labels_local = xp.asarray(labels.flat_values).reshape(-1).astype(xp.float32)  # type: ignore
        local_batch = int(stats_local.shape[0])

        dp_sharding = NamedSharding(self.mesh, P("dp"))
        replicated = NamedSharding(self.mesh, P())

        f1_jax = jnp.asarray(f1_local)
        f2_jax = jnp.asarray(f2_local)
        stats_jax = jnp.asarray(stats_local)
        labels_jax = jnp.asarray(labels_local)

        jax_world_size = self.world_size

        global_f1 = jax.make_array_from_single_device_arrays(
            shape=(jax_world_size * local_batch, f1_jax.shape[1]),
            sharding=dp_sharding,
            arrays=[f1_jax],
        )
        global_f2 = jax.make_array_from_single_device_arrays(
            shape=(jax_world_size * local_batch, f2_jax.shape[1]),
            sharding=dp_sharding,
            arrays=[f2_jax],
        )
        global_stats = jax.make_array_from_single_device_arrays(
            shape=(jax_world_size * local_batch, stats_jax.shape[1]),
            sharding=dp_sharding,
            arrays=[stats_jax],
        )
        global_labels = jax.make_array_from_single_device_arrays(
            shape=(jax_world_size * local_batch,),
            sharding=dp_sharding,
            arrays=[labels_jax],
        )

        @nnx.jit(
            in_shardings=(None, dp_sharding, dp_sharding, dp_sharding, dp_sharding),
            out_shardings=(None, dp_sharding, dp_sharding, replicated),
        )
        def forward_and_grad(model, f1, f2, stats, labels):
            x = jnp.concatenate([f1, f2, stats], axis=1)

            def loss_fn(m, input_x, y):
                pred = m(input_x)
                return jnp.mean((pred - y) ** 2)

            # dense_param_grads: replicated (same on all ranks after jit+allreduce)
            # x_grad: dp-sharded (each rank gets its local slice)
            dense_param_grads, x_grad = nnx.grad(loss_fn, argnums=(0, 1))(
                model, x, labels
            )
            loss_val = loss_fn(model, x, labels)
            emb1_dim = f1.shape[1]
            emb2_dim = f2.shape[1]
            return (
                dense_param_grads,
                x_grad[:, :emb1_dim],
                x_grad[:, emb1_dim : emb1_dim + emb2_dim],
                loss_val,
            )

        dense_param_grads, grad_f1_global, grad_f2_global, loss_val = forward_and_grad(
            self.model, global_f1, global_f2, global_stats, global_labels
        )
        if self.diagnostic_step_delay_ms:
            time.sleep(self.diagnostic_step_delay_ms / 1000.0)

        grad_f1_local = _local_jax_to_native(grad_f1_global)
        grad_f2_local = _local_jax_to_native(grad_f2_global)

        dense_grad_leaves = jtu.tree_leaves(dense_param_grads)
        if any(not gradient.is_fully_replicated for gradient in dense_grad_leaves):
            raise ValueError(
                "Dense parameter gradients must be replicated before export"
            )
        native_gradients = [
            _local_jax_to_native(gradient) for gradient in dense_grad_leaves
        ]
        dense_grad_flat = xp.concatenate(
            [gradient.ravel() for gradient in native_gradients]
        )

        print(
            f"[DenseSPMD.train_core] rank={self.rank} "
            f"f1_rec={f1_local.shape} f2_rec={f2_local.shape} "
            f"loss={float(loss_val):.4f} "
            f"dense_grad_norms={[float(xp.linalg.norm(gradient)) for gradient in native_gradients]}"
        )

        return {
            "grad_f1": grad_f1_local,
            "grad_f2": grad_f2_local,
            "dense_grad": TransportTensor(
                dense_grad_flat, global_shape=(int(dense_grad_flat.shape[0]),)
            ),
        }

    @operator(
        name="scatter_sparse_grads",
        async_policy=AsyncPolicy.pipelined(
            depth=2,
            schedule=_dense_phase_schedule("scatter_sparse_grads"),
            state=StatePolicy.thread_safe(),
        ),
        input_sharding={
            "grad_f1": {"type": "static", "axis": []},
            "grad_f2": {"type": "static", "axis": []},
            "dense_grad": {"type": "static", "axis": []},
            "emb1_ref": {"type": "static", "axis": []},
            "emb2_ref": {"type": "static", "axis": []},
            "inv1_dense": {"type": "static", "axis": []},
            "inv2_dense": {"type": "static", "axis": []},
        },
        output_sharding={
            "grad1": {"type": "mirror", "reference": "scatter_sparse_grads.emb1_ref"},
            "grad2": {"type": "mirror", "reference": "scatter_sparse_grads.emb2_ref"},
            "dense_grad": {"type": "static", "axis": []},
        },
    )
    def scatter_sparse_grads(
        self,
        grad_f1: TransportTensor,
        grad_f2: TransportTensor,
        dense_grad: TransportTensor,
        emb1_ref: TransportTensor,
        emb2_ref: TransportTensor,
        inv1_dense: TransportTensor,
        inv2_dense: TransportTensor,
    ) -> dict:
        xp = _xp()
        emb1_2d = xp.asarray(emb1_ref.flat_values).reshape(
            -1, emb1_ref.flat_values.shape[-1]
        )  # type: ignore
        emb2_2d = xp.asarray(emb2_ref.flat_values).reshape(
            -1, emb2_ref.flat_values.shape[-1]
        )  # type: ignore
        inv1_arr = xp.asarray(inv1_dense.flat_values)
        inv2_arr = xp.asarray(inv2_dense.flat_values)
        inv1_1d = inv1_arr.reshape(-1)
        inv2_1d = inv2_arr.reshape(-1)
        grad_f1_local = xp.asarray(grad_f1.flat_values)
        grad_f2_local = xp.asarray(grad_f2.flat_values)

        emb1_for_grad = xp.concatenate(  # type: ignore
            [emb1_2d, xp.zeros((1, self.embed_dim), dtype=emb1_2d.dtype)], axis=0
        )
        emb2_for_grad = xp.concatenate(  # type: ignore
            [emb2_2d, xp.zeros((1, self.embed_dim), dtype=emb2_2d.dtype)], axis=0
        )
        g1_unique = _scatter_add_grad(
            emb1_for_grad,
            inv1_1d,
            _pooled_grad_slots(inv1_arr, emb1_2d.shape[0], grad_f1_local),
        )[: emb1_2d.shape[0]]
        g2_unique = _scatter_add_grad(
            emb2_for_grad,
            inv2_1d,
            _pooled_grad_slots(inv2_arr, emb2_2d.shape[0], grad_f2_local),
        )[: emb2_2d.shape[0]]

        g1_nonzero_rows = int(xp.count_nonzero(xp.any(g1_unique != 0, axis=1)))
        g2_nonzero_rows = int(xp.count_nonzero(xp.any(g2_unique != 0, axis=1)))
        assert g1_nonzero_rows <= emb1_2d.shape[0], (
            f"[DenseSPMD] grad1 scatter-add non-zero rows {g1_nonzero_rows} "
            f"exceeds unique keys count {emb1_2d.shape[0]}"
        )
        assert g2_nonzero_rows <= emb2_2d.shape[0], (
            f"[DenseSPMD] grad2 scatter-add non-zero rows {g2_nonzero_rows} "
            f"exceeds unique keys count {emb2_2d.shape[0]}"
        )

        return {
            "grad1": g1_unique,
            "grad2": g2_unique,
            "dense_grad": dense_grad.zero_copy_alias(),
        }

    @operator(
        name="update",
        async_policy=AsyncPolicy.pipelined(
            depth=1,
            schedule=_dense_phase_schedule("update"),
        ),
        input_sharding={"dense_grad": {"type": "static", "axis": []}},
    )
    def update(self, dense_grad: TransportTensor) -> None:
        step = self.ctx.step
        treedef = jtu.tree_structure(self.grad_accum)
        param_leaves = jtu.tree_leaves(nnx.state(self.model))
        flat = jnp.asarray(dense_grad.flat_values)

        # -- Invariants on dense gradient flat layout --
        expected_size = sum(int(leaf.size) for leaf in param_leaves)
        assert flat.size == expected_size, (
            f"[DenseSPMD.update] dense_grad flat size {flat.size} "
            f"!= sum of param leaf sizes {expected_size}"
        )
        assert flat.ndim == 1, (
            f"[DenseSPMD.update] dense_grad must be 1D after ravel, got shape={flat.shape}"
        )

        cursor = 0
        split_leaves = []
        for leaf in param_leaves:
            local_gradient = flat[cursor : cursor + leaf.size].reshape(leaf.shape)
            split_leaves.append(_place_gradient_like(local_gradient, leaf))
            cursor += leaf.size

        self.grad_accum = jtu.tree_map(
            lambda acc, g: acc + g,
            self.grad_accum,
            jtu.tree_unflatten(treedef, split_leaves),
        )
        self._pending_accum_steps += 1

        if step % self.grad_accum_steps == 0:
            params = nnx.state(self.model)
            scale = 1.0 / self._pending_accum_steps
            updated = jtu.tree_map(
                lambda p, g: p - 0.01 * (g * scale), params, self.grad_accum
            )
            nnx.update(self.model, updated)
            self.grad_accum = jtu.tree_map(jnp.zeros_like, self.grad_accum)
            self._pending_accum_steps = 0
            print(
                f"[DenseSPMD.update] rank={self.rank} step={step} updated model params"
            )
        else:
            print(f"[DenseSPMD.update] rank={self.rank} step={step} accumulating grads")
