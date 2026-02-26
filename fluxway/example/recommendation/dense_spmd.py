import jax
import jax.numpy as jnp
from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from .data_types import LookupOutput, DenseOutput


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
    )
    def forward(
        self,
        emb1: LookupOutput,
        inv1: TransportTensor,
        emb2: LookupOutput,
        inv2: TransportTensor,
    ) -> DenseOutput:
        rank = jax.process_index() if hasattr(jax, "process_index") else 0

        # O(1) Reconstructions
        f1_rec = emb1.sparse_features.gather(inv1.flat_values)
        f2_rec = emb2.sparse_features.gather(inv2.flat_values)

        # Simple summation for demo prediction
        prediction = jnp.sum(f1_rec.flat_values) + jnp.sum(f2_rec.flat_values)
        print(f"[DenseSPMD] Worker {rank} Forward complete. Prediction: {prediction}")
        return DenseOutput(prediction=prediction, inv1=inv1, inv2=inv2)

    @operator(
        name="backward",
        data_argument=["context"],
        output_sharding={
            "grad1": {"type": "dynamic", "axis": [0], "fn": "hash_routing"},
            "grad2": {"type": "dynamic", "axis": [0], "fn": "hash_routing"},
        },
    )
    def backward(self, context: DenseOutput) -> dict:
        rank = jax.process_index() if hasattr(jax, "process_index") else 0
        print(f"[DenseSPMD] Worker {rank} Backward starting.")

        # Simulate backprop to unique ID space
        g1 = TransportTensor.from_tensor(jnp.ones_like(context.inv1.flat_values) * 0.1)
        g2 = TransportTensor.from_tensor(jnp.ones_like(context.inv2.flat_values) * 0.2)

        print(
            f"[DenseSPMD] Worker {rank} Produced gradients sum: {jnp.sum(g1.flat_values) + jnp.sum(g2.flat_values)}"
        )
        return {"grad1": g1, "grad2": g2}


def hash_routing(tensor, mesh):
    val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    return jnp.mod(val, mesh.size)
