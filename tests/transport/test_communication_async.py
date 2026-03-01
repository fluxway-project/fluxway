import unittest
import asyncio
from unittest.mock import MagicMock
import jax.numpy as jnp

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
        self.transport_tensor = TransportTensor(self.data)

        # Specs with Mesh for inference
        self.mesh = DeviceMesh((2,), ("x",), jnp.array([0, 1]))

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
            # send_fn calls depend on plan. With chunks matching, maybe no communication?
            # Or self-send?
            print(f"Sync distribute called send_fn {send_fn.call_count} times.")

        except ImportError:
            print("Skipping due to missing backend")

    def test_async_distributor(self):
        dist = AsyncDataDistributor(self.local_rank)

        # Updated API: send_fn takes 6 arguments
        async def async_send_fn(
            device_id,
            tensors,
            counts,
            has_spatial_indices_list,
            is_ragged_list,
            has_origin_indices_list,
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
            print(f"Async distribute returned {len(results)} results.")
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
            ragged_data, nested_splits=[jnp.array([0, 2, 3, 6])]
        )

        spatial_indices = jnp.arange(10, dtype=jnp.int32)
        spatial_tt = TransportTensor(dense_data, spatial_indices=spatial_indices)

        origin_indices = jnp.array([100, 101, 102], dtype=jnp.int32)
        # If we have origin indices, the count should match
        origin_data = jnp.arange(3, dtype=jnp.float32)
        origin_tt = TransportTensor(origin_data, origin_indices=origin_indices)

        batch = [spatial_tt, ragged_tt, origin_tt]

        async def async_send_fn(
            device_id,
            tensors,
            counts,
            has_spatial_indices_list,
            is_ragged_list,
            has_origin_indices_list,
        ):
            self.assertEqual(len(has_spatial_indices_list), 3)
            self.assertEqual(len(is_ragged_list), 3)
            self.assertEqual(len(has_origin_indices_list), 3)

            # spatial_tt is first (primary)
            self.assertTrue(has_spatial_indices_list[0])
            # ragged_tt is second
            self.assertTrue(is_ragged_list[1])
            # origin_tt is third
            self.assertTrue(has_origin_indices_list[2])

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
            print("Complex batch async distribute test passed!")
        except ImportError:
            print("Skipping complex batch test due to missing backend")

    def test_async_distributor_all_with_indices(self):
        """Test a batch where all tensors have spatial indices."""
        dist = AsyncDataDistributor(self.local_rank)

        spatial_indices = jnp.arange(10, dtype=jnp.int32)
        tt1 = TransportTensor(jnp.arange(10), spatial_indices=spatial_indices)
        tt2 = TransportTensor(jnp.arange(10) + 10, spatial_indices=spatial_indices)

        batch = [tt1, tt2]

        async def async_send_fn(
            device_id,
            tensors,
            counts,
            has_spatial_list,
            is_ragged_list,
            has_origin_list,
        ):
            self.assertEqual(has_spatial_list, [True, True])
            # In current implementation, we only send ONE shared spatial_indices tensor at the end
            # so flat tensors should be: [tt1_data, tt2_data, shared_spatial_idx]
            self.assertEqual(len(tensors), 3)
            return f"ok_{device_id}"

        async def run_test():
            await dist.distribute(batch, self.src_spec, self.dst_spec, async_send_fn)

        asyncio.run(run_test())
        print("All-with-indices batch test passed!")

    def test_async_distributor_multiple_mixed(self):
        """Test a mixed batch: [dense, ragged+indices, dense+indices]."""
        dist = AsyncDataDistributor(self.local_rank)

        idx = jnp.arange(5, dtype=jnp.int32)
        t1 = TransportTensor(jnp.zeros(5))  # dense
        t2 = TransportTensor(
            jnp.ones(5), nested_splits=[jnp.array([0, 2, 5])], spatial_indices=idx
        )  # ragged+idx
        t3 = TransportTensor(jnp.ones(5) * 2, spatial_indices=idx)  # dense+idx

        batch = [t1, t2, t3]

        async def async_send_fn(
            device_id,
            tensors,
            counts,
            has_spatial_list,
            is_ragged_list,
            has_origin_list,
        ):
            self.assertEqual(has_spatial_list, [False, True, True])
            self.assertEqual(is_ragged_list, [False, True, False])
            # flat tensors: [t1_data, t2_data, t2_len, t3_data, shared_idx]
            self.assertEqual(len(tensors), 5)
            self.assertEqual(len(counts), 5)
            return f"ok_{device_id}"

        async def run_test():
            await dist.distribute(batch, self.src_spec, self.dst_spec, async_send_fn)

        asyncio.run(run_test())
        print("Multiple mixed batch test passed!")


if __name__ == "__main__":
    unittest.main()
