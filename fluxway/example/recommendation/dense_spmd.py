import jax.numpy as jnp
from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from .data_types import DenseOutput


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class DenseSPMD:
    """
    Final computation node.
    """

    @operator(
        name="forward",
        data_argument=["emb1", "inv1", "emb2", "inv2"],
        input_sharding={
            "emb1": {"type": "static", "axis": ["dp"]},
            "inv1": {"type": "static", "axis": ["dp"]},
            "emb2": {"type": "static", "axis": ["dp"]},
            "inv2": {"type": "static", "axis": ["dp"]},
        },
        output_sharding={
            "prediction": {"type": "static", "axis": []},
            "context": {"type": "static", "axis": []},
            "inv1": {"type": "static", "axis": []},
            "inv2": {"type": "static", "axis": []},
            "emb1_unique": {"type": "static", "axis": []},
            "emb2_unique": {"type": "static", "axis": []},
        },
    )
    def forward(
        self,
        emb1: TransportTensor,
        inv1: TransportTensor,
        emb2: TransportTensor,
        inv2: TransportTensor,
    ) -> DenseOutput:
        rank = self.rank

        # Reconstructions using gather
        f1_rec = emb1.gather(inv1)
        f2_rec = emb2.gather(inv2)

        # Simple summation for demo prediction
        prediction = jnp.sum(f1_rec.flat_values) + jnp.sum(f2_rec.flat_values)
        print(f"[DenseSPMD] Worker {rank} Forward complete. Prediction: {prediction}")
        return DenseOutput(
            prediction=prediction,
            inv1=inv1,
            inv2=inv2,
            emb1_unique=emb1,
            emb2_unique=emb2,
        )

    @operator(
        name="backward",
        data_argument=["context"],
        input_sharding={"context": {"type": "static", "axis": []}},
        output_sharding={
            "grad1": {"type": "dynamic", "axis": [0], "fn": "hash_routing"},
            "grad2": {"type": "dynamic", "axis": [0], "fn": "hash_routing"},
        },
    )
    def backward(self, context: DenseOutput) -> dict:
        rank = self.rank
        print(f"[DenseSPMD] Worker {rank} Backward starting.")

        # 1. Produce gradients for the "gathered" embeddings (per-sample)
        # Assuming we have 2 features per sample on average for demo
        f1_shape = (
            context.inv1.flat_values.shape[0],
            context.emb1_unique.flat_values.shape[1],
        )
        f2_shape = (
            context.inv2.flat_values.shape[0],
            context.emb2_unique.flat_values.shape[1],
        )

        g1_gathered = jnp.ones(f1_shape) * 0.1
        g2_gathered = jnp.ones(f2_shape) * 0.2

        # 2. Aggregate gradients for unique IDs using inverse indices
        # Number of uniques is known from the unique embedding tensor shape
        num_uniques1 = context.emb1_unique.flat_values.shape[0]
        num_uniques2 = context.emb2_unique.flat_values.shape[0]

        g1_unique = (
            jnp.zeros((num_uniques1, f1_shape[1]))
            .at[context.inv1.flat_values]
            .add(g1_gathered)
        )
        g2_unique = (
            jnp.zeros((num_uniques2, f2_shape[1]))
            .at[context.inv2.flat_values]
            .add(g2_gathered)
        )

        # 3. Create TransportTensors with proper global IDs for routing
        # Use the global_indices (original IDs) from the unique embeddings
        # Routing will be based on these IDs via hash_routing
        grad1 = TransportTensor(g1_unique, indices=context.emb1_unique.global_indices)
        grad2 = TransportTensor(g2_unique, indices=context.emb2_unique.global_indices)

        print(
            f"[DenseSPMD] Worker {rank} Produced aggregated unique gradients sum: {jnp.sum(g1_unique) + jnp.sum(g2_unique)}"
        )
        return {"grad1": grad1, "grad2": grad2}


def hash_routing(tensor, mesh):
    val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    return jnp.mod(val, mesh.size)
