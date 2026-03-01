import unittest
import numpy as np
import jax.numpy as jnp
from unittest.mock import AsyncMock

from fluxway.common.types.tensor import TransportTensor


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
            if tensor.spatial_indices is not None:
                group.append(tensor.spatial_indices)
            if tensor.origin_indices is not None:
                group.append(tensor.origin_indices)

            row_count = int(tensor.shape[0]) if tensor.shape else 1
            await mailbox_deposit_shards_fn(
                target_op,
                [name],
                step,
                spmd_id,
                rank,
                [group],
                [[row_count]],
                is_ragged=[tensor.is_ragged],
                has_spatial_indices=[tensor.spatial_indices is not None],
                has_origin_indices=[tensor.origin_indices is not None],
            )

        # Test case
        data = jnp.array([1.0, 2.0, 3.0])
        spatial_idx = jnp.array([10, 11, 12])
        origin_idx = jnp.array([100, 101, 102])

        tt = TransportTensor(
            data, spatial_indices=spatial_idx, origin_indices=origin_idx
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

        self.assertEqual(kwargs["has_spatial_indices"], [True])
        self.assertEqual(kwargs["has_origin_indices"], [True])

    async def test_send_fn_unflattening_logic(self):
        """Verify the logic in _send_fn correctly unflattens a complex batch."""

        # Logic to be tested (extracted from worker.py L1525+)
        async def mock_unflatten_logic(
            tensors, counts, h_spatial_list, i_ragged_list, h_origin_list
        ):
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

                # 1. Data
                group_payloads.append(tensors[cursor])
                group_counts.append(counts[c_cursor])
                cursor += 1
                c_cursor += 1

                # 2. Lengths
                if ragged:
                    group_payloads.append(tensors[cursor])
                    group_counts.append(counts[c_cursor])
                    cursor += 1
                    c_cursor += 1

                # 3. Spatial
                if (
                    h_spatial_list
                    and h_spatial_list[i]
                    and global_indices_tensor is not None
                ):
                    group_payloads.append(global_indices_tensor)
                    group_counts.append(global_indices_count)

                # 4. Origin
                if has_asmb:
                    group_payloads.append(tensors[cursor])
                    group_counts.append(counts[c_cursor])
                    cursor += 1
                    c_cursor += 1

                payload_groups.append(group_payloads)
                counts_groups.append(group_counts)

            return payload_groups, counts_groups

        # Test case: [t1(ragged, spatial), t2(origin)]
        data1 = jnp.array([1.0, 2.0])
        len1 = jnp.array([1, 1])
        data2 = jnp.array([3.0])
        asmb2 = jnp.array([100])
        spatial_idx = jnp.array([0, 1])

        tensors = [data1, len1, data2, asmb2, spatial_idx]
        counts = [2, 2, 1, 1, 2]
        h_spatial = [True, False]
        i_ragged = [True, False]
        h_origin = [False, True]

        p_groups, c_groups = await mock_unflatten_logic(
            tensors, counts, h_spatial, i_ragged, h_origin
        )

        # Verify t1: [data1, len1, spatial_idx]
        self.assertEqual(len(p_groups[0]), 3)
        np.testing.assert_array_equal(p_groups[0][0], data1)
        np.testing.assert_array_equal(p_groups[0][1], len1)
        np.testing.assert_array_equal(p_groups[0][2], spatial_idx)

        # Verify t2: [data2, asmb2]
        self.assertEqual(len(p_groups[1]), 2)
        np.testing.assert_array_equal(p_groups[1][0], data2)
        np.testing.assert_array_equal(p_groups[1][1], asmb2)


if __name__ == "__main__":
    unittest.main()
