import jax.numpy as jnp
from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from .data_types import DenseOutput, DPFeatures


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class DenseSPMD:
    """
    Final computation node.
    """

    @operator(
        name="forward",
        input_sharding={
            "emb1": {"type": "dynamic", "axis": ["dp"]},
            "inv1": {"type": "static", "axis": ["dp"]},
            "emb2": {"type": "dynamic", "axis": ["dp"]},
            "inv2": {"type": "static", "axis": ["dp"]},
            "dp_data": {"type": "static", "axis": ["dp"]},
        },
        output_sharding={
            "prediction": {"type": "static", "axis": []},  # Scalar is replicated
            "user_inv": {"type": "static", "axis": ["dp"]},
            "item_inv": {"type": "static", "axis": ["dp"]},
            "user_emb_unique": {"type": "static", "axis": ["dp"]},
            "item_emb_unique": {"type": "static", "axis": ["dp"]},
            "dp_data": {"type": "static", "axis": ["dp"]},
        },
    )
    def forward(
        self,
        emb1: TransportTensor,
        inv1: TransportTensor,
        emb2: TransportTensor,
        inv2: TransportTensor,
        dp_data: DPFeatures,
    ) -> DenseOutput:
        rank = self.rank  # type: ignore

        # Reconstructions using indexing
        f1_rec = emb1[inv1]
        f2_rec = emb2[inv2]

        # Simple summation + dense stats
        prediction = f1_rec.sum() + f2_rec.sum() + dp_data.dense_stats.sum()

        print(
            f"[DenseSPMD] forward input rank={rank}, inv1={inv1.flat_values}, inv2={inv2.flat_values}"
        )

        print(
            f"[DenseSPMD]forward output rank={rank}, emb1:user_emb_unique={emb1.flat_values}, emb1.global_shape={emb1.global_shape}, emb2:item_emb_unique={emb2.flat_values}, emb2.global_shape={emb2.global_shape}"
        )

        return DenseOutput(
            prediction=prediction,
            user_inv=inv1,
            item_inv=inv2,
            user_emb_unique=emb1,
            item_emb_unique=emb2,
            dp_data=dp_data,
        )

    @operator(
        name="backward",
        input_sharding={"context": {"type": "static", "axis": ["dp"]}},
        output_sharding={
            "grad1": {"type": "mirror", "source": "forward.emb1"},
            "grad2": {"type": "mirror", "source": "forward.emb2"},
        },
    )
    def backward(self, context: DenseOutput) -> dict:
        rank = self.rank  # type: ignore
        # 1. Produce gradients for the "gathered" embeddings (per-sample)
        print(
            f"[DenseSPMD] backward rank={rank}, context.user_inv={context.user_inv.flat_values}, context.user_emb_unique.flat_values={context.user_emb_unique.flat_values}"
        )
        f1_shape = (len(context.user_inv), context.user_emb_unique.flat_values.shape[1])
        f2_shape = (len(context.item_inv), context.item_emb_unique.flat_values.shape[1])

        g1_gathered = jnp.ones(f1_shape, dtype=jnp.float32) * 0.1
        g2_gathered = jnp.ones(f2_shape, dtype=jnp.float32) * 0.2

        # 2. Aggregate gradients for unique IDs using inverse indices
        num_uniques1 = len(context.user_emb_unique)
        num_uniques2 = len(context.item_emb_unique)

        g1_unique = (
            jnp.zeros((num_uniques1, f1_shape[1]))
            .at[context.user_inv.flat_values.reshape(-1)]
            .add(g1_gathered)
        )
        g2_unique = (
            jnp.zeros((num_uniques2, f2_shape[1]))
            .at[context.item_inv.flat_values.reshape(-1)]
            .add(g2_gathered)
        )

        # 3. Create TransportTensors.
        # We don't even need to manually set routing_indices if the receiver
        # can infer them from provenance, but keeping them doesn't hurt.
        return {"grad1": g1_unique, "grad2": g2_unique}
