import unittest
import asyncio
from unittest.mock import MagicMock
import jax.numpy as jnp
import numpy as np

from fluxway.transport.communication import SyncDataDistributor, AsyncDataDistributor
from fluxway.common.types.sharding import ShardingSpec, P
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.tensor import TransportTensor


class TestDataDistributor(unittest.TestCase):
    def setUp(self):
        self.local_rank = 0
        self.num_workers = 2

        # Setup simple data
        self.data = jnp.arange(10, dtype=jnp.float32)
        self.transport_tensor = TransportTensor(self.data, global_shape=(20,))

        # Specs with Mesh for inference
        self.mesh = DeviceMesh(
            axis_names=("x",), device_shape=(2,), device_ids=np.array([0, 1])
        )

        self.src_spec = ShardingSpec.chunk(mesh=self.mesh, partition=P("x"))
        self.dst_spec = ShardingSpec.chunk(mesh=self.mesh, partition=P("x"))

    def test_sync_distributor(self):
        # New API: init only needs local_rank
        dist = SyncDataDistributor(self.local_rank)

        # Mock send_fn
        send_fn = MagicMock(return_value="ok")

        try:
            # New API: distribute(local_data, source, target, send_fn)
            dist.distribute(
                local_data=self.transport_tensor,
                source=self.src_spec,
                target=self.dst_spec,
                send_fn=send_fn,
            )

            # Should have called send_fn
            self.assertTrue(send_fn.called)

        except ImportError:
            print("Skipping due to missing backend")

    def test_async_distributor(self):
        dist = AsyncDataDistributor(self.local_rank)

        async def async_send_fn(
            device_id, tensors, counts, global_shapes, meta_tensor_state_list
        ):
            await asyncio.sleep(0.01)
            return f"ok_{device_id}"

        async def run_test():
            results, _ = await dist.distribute(
                local_data=self.transport_tensor,
                source=self.src_spec,
                target=self.dst_spec,
                send_fn=async_send_fn,
            )
            return results

        try:
            results = asyncio.run(run_test())
            self.assertTrue(len(results) > 0)
            for r in results:
                self.assertTrue(r.startswith("ok_"))

        except ImportError:
            print("Skipping due to missing backend")

    def test_async_distributor_complex_batch(self):
        dist = AsyncDataDistributor(self.local_rank)

        # Create a complex batch
        dense_data = jnp.arange(10, dtype=jnp.float32)

        ragged_data = jnp.arange(6, dtype=jnp.float32)
        # lengths [2, 1, 3] -> splits [0, 2, 3, 6]
        ragged_tt = TransportTensor(
            ragged_data, nested_splits=[jnp.array([0, 2, 3, 6])], global_shape=(12,)
        )

        routing_indices = jnp.arange(10, dtype=jnp.int32)
        spatial_tt = TransportTensor(
            dense_data, routing_indices=routing_indices, global_shape=(20,)
        )

        origin_indices = jnp.array([100, 101, 102], dtype=jnp.int32)
        # If we have origin indices, the count should match
        origin_data = jnp.arange(3, dtype=jnp.float32)
        origin_tt = TransportTensor(
            origin_data, origin_indices=origin_indices, global_shape=(6,)
        )

        batch = [spatial_tt, ragged_tt, origin_tt]

        async def async_send_fn(
            device_id, tensors, counts, global_shapes, meta_tensor_state_list
        ):
            self.assertEqual(len(meta_tensor_state_list), 3)

            # spatial_tt is first (primary)
            self.assertTrue(bool(meta_tensor_state_list[0] & 4))
            # ragged_tt is second
            self.assertTrue(bool(meta_tensor_state_list[1] & 1))
            # origin_tt is third
            self.assertTrue(bool(meta_tensor_state_list[2] & 2))

            return f"ok_{device_id}"

        async def run_test():
            results, _ = await dist.distribute(
                local_data=batch,
                source=self.src_spec,
                target=self.dst_spec,
                send_fn=async_send_fn,
            )
            return results

        try:
            asyncio.run(run_test())
        except ImportError:
            print("Skipping complex batch test due to missing backend")

    def test_async_distributor_all_with_indices(self):
        """Test a batch where all tensors have spatial indices."""
        dist = AsyncDataDistributor(self.local_rank)

        routing_indices = jnp.arange(10, dtype=jnp.int32)
        tt1 = TransportTensor(
            jnp.arange(10), routing_indices=routing_indices, global_shape=(20,)
        )
        tt2 = TransportTensor(
            jnp.arange(10) + 10, routing_indices=routing_indices, global_shape=(20,)
        )

        batch = [tt1, tt2]

        async def async_send_fn(
            device_id, tensors, counts, global_shapes, meta_tensor_state_list
        ):
            self.assertEqual(meta_tensor_state_list, [4, 4])
            # In current implementation, we only send ONE shared routing_indices tensor at the end
            # so flat tensors should be: [tt1_data, tt2_data, shared_spatial_idx]
            self.assertEqual(len(tensors), 3)
            return f"ok_{device_id}"

        async def run_test():
            await dist.distribute(batch, self.src_spec, self.dst_spec, async_send_fn)

        asyncio.run(run_test())

    def test_async_distributor_multiple_mixed(self):
        """Test a mixed batch: [dense, ragged+indices, dense+indices]."""
        dist = AsyncDataDistributor(self.local_rank)

        idx = jnp.arange(5, dtype=jnp.int32)
        t1 = TransportTensor(jnp.zeros(5), global_shape=(10,))  # dense
        t2 = TransportTensor(
            jnp.ones(5),
            nested_splits=[jnp.array([0, 2, 5])],
            routing_indices=idx,
            global_shape=(10,),
        )  # ragged+idx
        t3 = TransportTensor(
            jnp.ones(5) * 2, routing_indices=idx, global_shape=(10,)
        )  # dense+idx

        batch = [t1, t2, t3]

        async def async_send_fn(
            device_id, tensors, counts, global_shapes, meta_tensor_state_list
        ):
            self.assertEqual(meta_tensor_state_list, [0, 5, 4])
            # flat tensors: [t1_data, t2_data, t2_len, t3_data, shared_idx]
            self.assertEqual(len(tensors), 5)
            self.assertEqual(len(counts), 5)
            return f"ok_{device_id}"

        async def run_test():
            await dist.distribute(batch, self.src_spec, self.dst_spec, async_send_fn)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
