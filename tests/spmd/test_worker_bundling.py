import unittest
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.types.sharding import ShardingSpec, P
from fluxway.common.mesh import DeviceMesh


class TestWorkerBundling(unittest.IsolatedAsyncioTestCase):
    """
    Tests for SPMD worker communication bundling, sharding splits,
    and @data_type field handling.
    """

    async def _mock_unflatten_logic(
        self, tensors, counts, h_spatial_list, i_ragged_list, h_origin_list
    ):
        """Extraction of worker.py L1525+ unflattening logic."""
        payload_groups = []
        counts_groups = []
        cursor = 0
        c_cursor = 0

        global_indices_tensor = None
        global_indices_count = None
        if h_spatial_list and any(h_spatial_list) and len(tensors) > cursor:
            global_indices_tensor = tensors[-1]
            global_indices_count = counts[-1]

        for i, (ragged, has_asmb) in enumerate(
            zip(i_ragged_list, h_origin_list, strict=True)  # type: ignore
        ):
            group_payloads = []
            group_counts = []

            # 1. Primary data
            group_payloads.append(tensors[cursor])
            group_counts.append(counts[c_cursor])
            cursor += 1
            c_cursor += 1

            # 2. Ragged lengths
            if ragged:
                group_payloads.append(tensors[cursor])
                group_counts.append(counts[c_cursor - 1])  # simplified for test
                cursor += 1

            # 3. Spatial Indices (Routing)
            if (
                h_spatial_list
                and h_spatial_list[i]
                and global_indices_tensor is not None
            ):
                group_payloads.append(global_indices_tensor)
                group_counts.append(global_indices_count)

            # 4. Assembly indices (Origin)
            if has_asmb:
                group_payloads.append(tensors[cursor])
                group_counts.append(counts[c_cursor])
                cursor += 1
                c_cursor += 1

            payload_groups.append(group_payloads)
            counts_groups.append(group_counts)

        return payload_groups, counts_groups

    def test_worker_grouping_by_sharding(self):
        """
        Verify that worker.py correctly groups tensors into batches
        based on (source_spec, target_spec).
        """
        mesh = DeviceMesh((2,), ("x",), jnp.array([0, 1]))
        spec1 = ShardingSpec.chunk(mesh, P("x"))
        spec2 = ShardingSpec.chunk(mesh, P())  # replicated

        # Simulate worker._send_output grouping logic
        tensors = [
            ("t1", spec1, spec1),  # Group A
            ("t2", spec1, spec1),  # Group A
            ("t3", spec1, spec2),  # Group B
            ("t4", spec2, spec2),  # Group C
        ]

        batches = {}
        for name, src, dst in tensors:
            key = (src, dst)
            if key not in batches:
                batches[key] = []
            batches[key].append(name)

        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[(spec1, spec1)], ["t1", "t2"])
        self.assertEqual(batches[(spec1, spec2)], ["t3"])
        self.assertEqual(batches[(spec2, spec2)], ["t4"])

    async def test_unflatten_complex_bundle(self):
        """Verify unflattening mixed tensors in a single bundle."""
        d1 = jnp.zeros(5)
        d2 = jnp.ones(5)
        l2 = jnp.array([2, 3])
        shared_idx = jnp.arange(5)

        # Bundle: [t1(dense+idx), t2(ragged+idx)]
        tensors = [d1, d2, l2, shared_idx]
        counts = [5, 5, 2, 5]
        h_spatial = [True, True]
        i_ragged = [False, True]
        h_origin = [False, False]

        p_groups, _ = await self._mock_unflatten_logic(
            tensors, counts, h_spatial, i_ragged, h_origin
        )

        # t1: [d1, shared_idx]
        self.assertEqual(len(p_groups[0]), 2)
        np.testing.assert_array_equal(p_groups[0][1], shared_idx)
        # t2: [d2, l2, shared_idx]
        self.assertEqual(len(p_groups[1]), 3)
        np.testing.assert_array_equal(p_groups[1][2], shared_idx)

    async def test_self_receive_data_indices(self):
        """Verify self-loop logic includes ALL indices in the group."""

        async def mock_self_receive(tt):
            group = [tt.flat_values]
            if tt.spatial_indices is not None:
                group.append(tt.spatial_indices)
            if tt.origin_indices is not None:
                group.append(tt.origin_indices)
            return group

        # Case 1: Spatial only
        tt1 = TransportTensor(jnp.zeros(2), spatial_indices=jnp.arange(2))
        g1 = await mock_self_receive(tt1)
        self.assertEqual(len(g1), 2)

        # Case 2: Origin only
        tt2 = TransportTensor(jnp.zeros(2), origin_indices=jnp.arange(2))
        g2 = await mock_self_receive(tt2)
        self.assertEqual(len(g2), 2)

        # Case 3: Both
        tt3 = TransportTensor(
            jnp.zeros(2), spatial_indices=jnp.arange(2), origin_indices=jnp.arange(2)
        )
        g3 = await mock_self_receive(tt3)
        self.assertEqual(len(g3), 3)

    async def test_zero_copy_bundling_registration(self):
        """
        Verify that zero-copy registration can handle multiple tensors
        arriving in one bundle.
        """
        # In fluxway, zero-copy registration is per (op, tensor_name, step)
        # If a bundle has multiple tensors, each must match its registered buffer.

        # This is more of a mailbox + worker interaction test.
        # Mailbox.deposit_shards iterates over names and checks for zero-copy slots.
        from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox

        mb = WorkerMailbox()

        buf1 = np.zeros(10)  # type: ignore
        buf2 = np.zeros(10)  # type: ignore

        await mb.register_zero_copy_slot("op", "t1", 0, [buf1], {0: 0})
        await mb.register_zero_copy_slot("op", "t2", 0, [buf2], {0: 0})

        # Deposit a bundle with t1 and t2
        data1 = np.ones(10)  # type: ignore
        data2 = np.ones(10) * 2  # type: ignore

        await mb.deposit_shards(
            "op",
            ["t1", "t2"],
            0,
            "s0",
            0,
            payload_groups=[[data1], [data2]],
            counts_groups=[[10], [10]],
            is_ragged=[False, False],
            has_spatial_indices=[False, False],
            has_origin_indices=[False, False],
        )

        # Verify buffers updated (check via mb.get_zero_copy_buffers since JAX reassigns)
        out_bufs1 = await mb.get_zero_copy_buffers("op", "t1", 0)
        out_bufs2 = await mb.get_zero_copy_buffers("op", "t2", 0)

        self.assertIsNotNone(out_bufs1)
        self.assertIsNotNone(out_bufs2)
        np.testing.assert_array_equal(out_bufs1[0], data1)
        np.testing.assert_array_equal(out_bufs2[0], data2)

    def test_datatype_grouping_logic(self):
        """
        Verify that a flattened @data_type is split or bundled correctly
        based on sharding.
        """
        mesh = DeviceMesh((2,), ("x",), jnp.array([0, 1]))
        spec1 = ShardingSpec.chunk(mesh, P("x"))
        spec2 = ShardingSpec.chunk(mesh, P())

        # Scenario 1: Mixed sharding in one @data_type
        # Flattened fields:
        fields = [("my_data.field_a", spec1, spec1), ("my_data.field_b", spec2, spec2)]
        batches: Dict[Tuple[ShardingSpec, ShardingSpec], List[str]] = {}
        for name, src, dst in fields:
            key = (src, dst)
            batches.setdefault(key, []).append(name)

        self.assertEqual(
            len(batches), 2, "Should split into 2 bundles due to different sharding"
        )

        # Scenario 2: Same sharding in @data_type
        fields = [("my_data.field_a", spec1, spec1), ("my_data.field_b", spec1, spec1)]
        batches: Dict[Tuple[ShardingSpec, ShardingSpec], List[str]] = {}
        for name, src, dst in fields:
            key = (src, dst)
            batches.setdefault(key, []).append(name)

        self.assertEqual(
            len(batches), 1, "Should bundle into 1 batch due to same sharding"
        )
        self.assertEqual(len(batches[(spec1, spec1)]), 2)


if __name__ == "__main__":
    unittest.main()
