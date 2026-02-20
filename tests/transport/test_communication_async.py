import unittest
import asyncio
from unittest.mock import MagicMock
import jax.numpy as jnp

from fluxway.transport.communication import SyncDataDistributor, AsyncDataDistributor
from fluxway.transport.sharding import ShardingSpec, P
from fluxway.transport.mesh import DeviceMesh
from fluxway.transport.tensor import TransportTensor


class TestDataDistributor(unittest.TestCase):
    def setUp(self):
        self.local_rank = 0
        self.world_size = 2

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

        # Mock async send_fn
        async def async_send_fn(device_id, tensors, counts):
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


if __name__ == "__main__":
    unittest.main()
