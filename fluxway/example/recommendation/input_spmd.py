import jax.numpy as jnp
from fluxway.common.decorators import spmd, operator
from .data_types import UserBatch, InputNodeOutput, DPFeatures


def hash_routing(tensor, mesh):
    from fluxway.common.types.tensor import TransportTensor

    # Feature IDs are stored in flat_values for input sharding
    val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor

    # If it's a 2D matrix (like ID batch or gradients), we hash based on the first column (the IDs)
    if len(val.shape) > 1:
        val = val[:, 0]

    return jnp.mod(val, mesh.size)


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class InputSPMD:
    """
    Deduplicates multiple features and routes them to respective tables.
    """

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
    def process(self, data: UserBatch) -> InputNodeOutput:
        # 1. Handle User ID (Routing to Table 1)
        u_user, inv_user = data.active.user_id.unique_with_routing()

        # 2. Handle Item ID (Routing to Table 2)
        u_item, inv_item = data.passive.target_item_id.unique_with_routing()

        # 3. Features to pass through as-is (DP aligned)
        dp_data = DPFeatures(
            labels=data.labels, dense_stats=data.passive.target_item_stats
        )

        return InputNodeOutput(
            unique_user_ids=u_user,
            user_inv=inv_user,
            unique_item_ids=u_item,
            item_inv=inv_item,
            dp_data=dp_data,
        )
