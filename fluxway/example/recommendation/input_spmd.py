import jax.numpy as jnp
from fluxway.common.decorators import spmd, operator
from .data_types import RaggedInput, InputNodeOutput


def hash_routing(tensor, mesh):
    """Modulo-based routing for P2P."""
    return jnp.mod(tensor.flat_values, mesh.size)


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class InputSPMD:
    """
    Deduplicates multiple features and routes them to respective tables.
    """

    @operator(
        name="process",
        data_argument=["feature1", "feature2"],
        output_sharding={
            "unique_ids1": {"type": "dynamic", "axis": [0], "fn": "hash_routing"},
            "unique_ids2": {"type": "dynamic", "axis": [0], "fn": "hash_routing"},
            "inv1": {"type": "static", "axis": ["dp"]},
            "inv2": {"type": "static", "axis": ["dp"]},
        },
    )
    def process(self, data: RaggedInput) -> InputNodeOutput:
        u1, inv1 = data.feature1.unique_with_routing()
        u2, inv2 = data.feature2.unique_with_routing()

        print(f"[InputSPMD] F1 Unique IDs: {u1.flat_values}")
        print(f"[InputSPMD] F1 Inv Map: {inv1.flat_values}")
        print(f"[InputSPMD] F2 Unique IDs: {u2.flat_values}")
        print(f"[InputSPMD] F2 Inv Map: {inv2.flat_values}")

        return InputNodeOutput(unique_ids1=u1, inv1=inv1, unique_ids2=u2, inv2=inv2)
