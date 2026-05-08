import asyncio

import jax
import jax.numpy as jnp
import sparse_embedding as se

from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.index_descriptor import ExplicitIndex
from .data_types import UserBatch, InputNodeOutput, DPFeatures


def hash_routing(tensor, mesh):
    val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    if len(val.shape) > 1:
        val = val[:, 0]
    return jnp.mod(val, mesh.size)


async def _sparse_unique(
    tt: TransportTensor,
) -> tuple[TransportTensor, TransportTensor]:
    """Uses sparse_embedding.async_unique (C++ high-perf) to deduplicate ids.

    Returns (unique_tt, inverse_tt) with the same semantic as
    TransportTensor.unique_with_routing() but backed by the C++ unique kernel.
    """
    ids_jax = tt.flat_values
    unique_dlm, inv_dlm, first_index_dlm = await se.async_unique(
        ids_jax.ravel(), return_inverse=True, return_first_indices=True
    )

    # Zero-copy DLPack transfer directly to JAX
    unique_jax = jax.dlpack.from_dlpack(unique_dlm).astype(jnp.int64)
    inv_jax = jax.dlpack.from_dlpack(inv_dlm).astype(jnp.int32)
    first_jax = jax.dlpack.from_dlpack(first_index_dlm).astype(jnp.int32)

    if tt.origin_indices is not None:
        global_first_jax = jnp.take(tt.origin_indices_materialized, first_jax)
    else:
        global_first_jax = first_jax

    unique_tt = TransportTensor(
        flat_values=unique_jax,
        routing_indices=ExplicitIndex(unique_jax),
        origin_indices=ExplicitIndex(global_first_jax),
        global_shape=tt.global_shape,
    )
    inv_tt = TransportTensor(flat_values=inv_jax, global_shape=tt.global_shape)
    return unique_tt, inv_tt


@spmd(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("dp",),
)
class InputSPMD:
    """Deduplicates features via sparse_embedding.async_unique and routes to embedding tables."""

    @operator(
        name="process",
        input_sharding={"data": {"type": "static", "axis": ["dp"]}},
        output_sharding={
            "unique_user_ids": {
                "type": "dynamic",
                "axis": ["dp"],
                "fn": "hash_routing",
            },
            "unique_item_ids": {
                "type": "dynamic",
                "axis": ["dp"],
                "fn": "hash_routing",
            },
            "user_inv": {"type": "static", "axis": ["dp"]},
            "item_inv": {"type": "static", "axis": ["dp"]},
            "dp_data": {"type": "static", "axis": ["dp"]},
        },
    )
    async def process(self, data: UserBatch) -> InputNodeOutput:
        user_task = _sparse_unique(data.active.user_id)
        item_task = _sparse_unique(data.passive.target_item_id)

        (u_user, inv_user), (u_item, inv_item) = await asyncio.gather(
            user_task, item_task
        )

        print(
            f"[InputSPMD.process] rank={self.rank} "
            f"unique_users={u_user.flat_values} user_inv={inv_user.flat_values} "
            f"unique_items={u_item.flat_values} item_inv={inv_item.flat_values}"
        )

        return InputNodeOutput(
            unique_user_ids=u_user,
            user_inv=inv_user,
            unique_item_ids=u_item,
            item_inv=inv_item,
            dp_data=DPFeatures(
                labels=data.labels, dense_stats=data.passive.target_item_stats
            ),
        )
