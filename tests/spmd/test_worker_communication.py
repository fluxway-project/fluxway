import unittest
import numpy as np
import jax.numpy as jnp
from unittest.mock import AsyncMock

from fluxway.common.types.tensor import TransportTensor, SymmetryReceipt
from fluxway.common.types.sharding import ShardingSpec, MirrorPartition, PartitionSpec
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.communication import BaseDataDistributor


class TestWorkerCommunicationLogic(unittest.IsolatedAsyncioTestCase):
    async def test_self_receive_data_logic(self):
        """Verify the logic in _self_receive_data correctly includes indices."""

        # Logic to be tested (extracted from worker.py)
        async def mock_self_receive_data(
            target_op, name, step, tensor, mailbox_deposit_shards_fn, rank, spmd_id
        ):
            group = [tensor.flat_values]
            if tensor.is_ragged:
                # Reconstruct lengths from splits
                splits = tensor.nested_splits[0]
                lengths = splits[1:] - splits[:-1]
                group.append(lengths)
            if tensor.routing_indices_materialized is not None:
                group.append(tensor.routing_indices_materialized)
            if tensor.origin_indices is not None:
                group.append(tensor.origin_indices)

            # meta_state: bit0=ragged, bit1=origin, bit2=routing
            meta_state = 0
            if tensor.is_ragged:
                meta_state |= 1
            if tensor.origin_indices is not None:
                meta_state |= 2
            if tensor.routing_indices is not None:
                meta_state |= 4

            row_count = int(tensor.shape[0]) if tensor.shape else 1
            await mailbox_deposit_shards_fn(
                target_op,
                [name],
                step,
                spmd_id,
                rank,
                [group],
                [[row_count]],
                meta_tensor_state_list=[meta_state],
            )
            # Removed outdated keyword arguments is_ragged, etc.

        # Test case
        data = jnp.array([1.0, 2.0, 3.0])
        spatial_idx = jnp.array([10, 11, 12])
        origin_idx = jnp.array([100, 101, 102])

        tt = TransportTensor(
            data, routing_indices=spatial_idx, origin_indices=origin_idx
        )

        deposit_mock = AsyncMock(return_value=0)
        await mock_self_receive_data(
            "target_op", "tensor_name", 0, tt, deposit_mock, rank=0, spmd_id="w0"
        )

        deposit_mock.assert_called_once()
        args, kwargs = deposit_mock.call_args

        payload_group = args[5][0]
        self.assertEqual(len(payload_group), 3)  # data, spatial, origin
        np.testing.assert_array_equal(payload_group[1], spatial_idx)
        np.testing.assert_array_equal(payload_group[2], origin_idx)

        # Removed outdated keyword assertions

    async def test_send_fn_unflattening_logic(self):
        """Verify the logic in _send_fn correctly unflattens a complex batch."""

        # Logic to be tested (extracted from worker.py L1525+)
        # Updated logic to match worker.py's new deduplicated unbundling
        async def mock_unflatten_logic(tensors, counts, meta_list, _batch_tensors):
            # Phase 1: Unique indices map
            unique_indices_in_flat = {}
            cursor = 0
            for i, meta_state in enumerate(meta_list):
                cursor += 1  # data
                if bool(meta_state & 1):
                    cursor += 1  # lengths

            for i, tt in enumerate(_batch_tensors):
                if tt.origin_indices is not None:
                    if id(tt.origin_indices) not in unique_indices_in_flat:
                        unique_indices_in_flat[id(tt.origin_indices)] = cursor
                        cursor += 1
                if tt.routing_indices is not None:
                    if id(tt.routing_indices) not in unique_indices_in_flat:
                        unique_indices_in_flat[id(tt.routing_indices)] = cursor
                        cursor += 1

            payload_groups = []
            cursor = 0
            for i, meta_state in enumerate(meta_list):
                group = []
                group.append(tensors[cursor])  # data
                cursor += 1
                if bool(meta_state & 1):  # ragged
                    group.append(tensors[cursor])
                    cursor += 1
                if bool(meta_state & 2):  # origin
                    idx = unique_indices_in_flat[id(_batch_tensors[i].origin_indices)]
                    group.append(tensors[idx])
                if bool(meta_state & 4):  # routing
                    idx = unique_indices_in_flat[id(_batch_tensors[i].routing_indices)]
                    group.append(tensors[idx])
                payload_groups.append(group)
            return payload_groups

        # Updated Test case: [t1(ragged, routing), t2(origin)]
        data1 = jnp.array([1.0, 2.0])
        len1 = jnp.array([1, 1])
        spatial1 = jnp.array([0, 1])
        data2 = jnp.array([3.0])
        asmb2 = jnp.array([100])

        tt1 = TransportTensor(
            data1, nested_splits=[jnp.array([0, 1, 2])], routing_indices=spatial1
        )
        tt2 = TransportTensor(data2, origin_indices=asmb2)

        # Meta: t1=1(ragged)+4(routing)=5, t2=2(origin)
        meta_list = [5, 2]
        # order in tensors list must match unique indices order (as packed by communication.py)
        # Unique indices are added after all data/lengths.
        # Phase 1: spatial1, Phase 2: asmb2
        tensors = [data1, len1, data2, spatial1, asmb2]

        p_groups = await mock_unflatten_logic(tensors, None, meta_list, [tt1, tt2])

        # Verify t1: [data1, len1, spatial1]
        self.assertEqual(len(p_groups[0]), 3)
        np.testing.assert_array_equal(p_groups[0][0], data1)
        np.testing.assert_array_equal(p_groups[0][1], len1)
        np.testing.assert_array_equal(p_groups[0][2], spatial1)

        # Verify t2: [data2, asmb2]
        self.assertEqual(len(p_groups[1]), 2)
        np.testing.assert_array_equal(p_groups[1][0], data2)
        np.testing.assert_array_equal(p_groups[1][1], asmb2)

    def test_base_distributor_mirror_routing(self):
        """Verify the BaseDataDistributor correctly handles mirror routing."""
        mesh = DeviceMesh(device_shape=(1,), axis_names=("worker",))

        # 1. Simulate a tensor with a receipt (e.g. received from rank 42)
        # SymmetryReceipt: (world_size, sender_rank_tags)
        receipt = SymmetryReceipt(
            world_size=43, sender_rank_tags=jnp.array([42, 42, 42])
        )

        data = jnp.array([[1.0], [2.0], [3.0]])
        tt = TransportTensor(data, receipt=receipt)

        # 2. Define a Mirror Sharding Spec
        mirror_strategy = MirrorPartition(source="some_op.input_tensor")
        spec = ShardingSpec(
            strategy=mirror_strategy, mesh=mesh, partition=PartitionSpec()
        )

        distributor = BaseDataDistributor(local_rank=0)

        # 3. Plan and Shard
        distributor.plan_and_shard(
            [tt],
            source=spec,  # Using mirror spec as source triggers the logic
            target=spec,  # target is also mirror
            tuuid=[123],
        )
        # In Late Aggregation, we get results from the internal _dest_agg
        # Each entry in _dest_agg[rank] is a 4-tuple: (tensors, counts, shapes, meta)
        (
            payload_tensors,
            payload_counts,
            global_shapes_list,
            meta_tensor_state_list,
        ) = distributor._dest_agg[42][0]

        # Verification
        self.assertEqual(len(payload_tensors), 1)
        # In current BaseDataDistributor._plan_and_shard mirror path:
        # It calls _stitch_with_prime and then _get_rank_data_from_tree
        # payload_tensors should be [data]
        self.assertEqual(len(payload_tensors), 1)
        np.testing.assert_allclose(payload_tensors[0], data)
        self.assertEqual(payload_counts, [3])
        self.assertEqual(
            meta_tensor_state_list, [0]
        )  # Mirror Path currently doesn't propagate indices?


if __name__ == "__main__":
    unittest.main()
