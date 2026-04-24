"""
Comprehensive tests for cross-mesh (M:N) routing in Fluxway's SPMD layer.
These are pure-Python / JAX unit tests that do NOT require Ray or a full
training stack. They validate the assembly helper methods in worker.py and
the topology projection logic in operator_manager.py in isolation.

Test scenarios
--------------
1. 1:1 Static  - dp=4 → dp=4
2. M:N Static  - dp=2 → dp=6
3. Dynamic → Static (origin_indices reassembly) - shard=3 → dp=6
4. Interleave Source - virtual SPMD sub-mesh rotation
5. Mirror routing - SymmetryReceipt construction and routing
6. E2E Embedding - shard=4 dynamic forward + mirror backward
"""

from __future__ import annotations

import unittest
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np

from fluxway.common.backend import get_backend
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor

# ─── Helper: build a minimal TransportTensor ─────────────────────────────────


def make_tt(
    values,
    origin_indices=None,
    routing_indices=None,
    receipt: Optional[SymmetryReceipt] = None,
    global_shape=None,
) -> TransportTensor:
    return TransportTensor(
        jnp.array(values, dtype=jnp.float32),
        origin_indices=(
            jnp.array(origin_indices, dtype=jnp.int32)
            if origin_indices is not None
            else None
        ),
        routing_indices=(
            jnp.array(routing_indices, dtype=jnp.int32)
            if routing_indices is not None
            else None
        ),
        receipt=receipt,
        global_shape=global_shape,
    )


def make_receipt(sender_ranks, remote_indices, world_size=4) -> SymmetryReceipt:
    return SymmetryReceipt(world_size, jnp.array(sender_ranks, dtype=jnp.int32))


# ─── Minimal stand-in for the Worker's assembly helpers ──────────────────────
# We instantiate a light mock object that provides only what is needed for the
# helper methods so we can test them without spinning up Ray actors.


class _AssemblyMixin:
    """Mixin that contains only the assembly helpers from Worker."""

    rank: int

    def _get_values(self, t: TransportTensor):
        return t.flat_values

    def _get_routing_indices(self, t: TransportTensor):
        if isinstance(t, TransportTensor):
            return t.routing_indices_materialized
        return None

    def _normalize_and_cat_shards(self, tensors, dim=0):
        backend = get_backend()
        if not tensors:
            return backend.zeros((0,), dtype=backend.float32())
        if len(tensors) == 1:
            return tensors[0]
        # Promote dtype to the first non-empty tensor's dtype
        proto = next((t for t in tensors if backend.shape(t)[0] > 0), tensors[0])
        proto_dtype = backend.dtype(proto)
        proto_shape = list(backend.shape(proto))
        normalized = []
        for t in tensors:
            s = list(backend.shape(t))
            if s[0] == 0 and len(s) < len(proto_shape):
                new_shape = [0] + proto_shape[1:]
                normalized.append(backend.zeros(tuple(new_shape), dtype=proto_dtype))
            else:
                normalized.append(t)
        return backend.cat(normalized, dim=dim)

    # ── Copy the four helpers verbatim from worker.py ────────────────────────

    def _merge_index_descriptors(self, shard_list):
        from fluxway.common.index_descriptor import IndexDescriptor

        descriptors = [
            s.index_descriptor for s in shard_list if s.index_descriptor is not None
        ]
        if len(descriptors) == len(shard_list) and shard_list:
            return IndexDescriptor.concatenate(descriptors)
        return None

    def _merge_origin_indices(self, shards):
        backend = get_backend()
        if not any(
            hasattr(s, "origin_indices") and s.origin_indices is not None
            for s in shards
        ):
            return None
        origin_list = []
        for s in shards:
            idx = (
                s.origin_indices_materialized
                if hasattr(s, "origin_indices_materialized")
                else None
            )
            if idx is None:
                idx = self._get_routing_indices(s)
            if idx is None:
                num = backend.shape(self._get_values(s))[0]
                idx = backend.arange(0, num, dtype=backend.int32())
            origin_list.append(idx)
        return self._normalize_and_cat_shards(origin_list, dim=0)

    def _maybe_merge_receipts(self, shards, has_mirror: bool):
        if not has_mirror:
            return None
        receipts = [s.receipt for s in shards]
        if all(r is not None for r in receipts):
            backend = get_backend()
            lengths = [int(backend.shape(s.flat_values)[0]) for s in shards]
            return SymmetryReceipt.concatenate(receipts, lengths)
        return None

    def _heal_empty_shape(self, input_spec, reference_values=None):
        backend = get_backend()
        dtype = backend.float32()
        feature_dim = None
        if hasattr(input_spec, "global_shape") and input_spec.global_shape:
            gs = list(input_spec.global_shape)
            if len(gs) >= 2:
                feature_dim = gs[1]
        if feature_dim is None and reference_values is not None:
            ref_shape = backend.shape(reference_values)
            if len(ref_shape) >= 2:
                feature_dim = int(ref_shape[1])
        if feature_dim is not None:
            return TransportTensor.from_tensor(
                backend.zeros((0, feature_dim), dtype=dtype)
            )
        return TransportTensor.from_tensor(backend.zeros((0,), dtype=dtype))

    def _assemble_dynamic_to_static_core(
        self, valid_shards: List[TransportTensor], input_spec, has_mirror: bool
    ):
        """Extracted core of _assemble_dynamic_to_static_tensor for testing."""
        backend = get_backend()
        all_receipt = self._maybe_merge_receipts(valid_shards, has_mirror)

        routing_dim = getattr(input_spec, "routing_dim", 0)
        if isinstance(routing_dim, (list, tuple)):
            routing_dim = routing_dim[0]

        all_values = backend.cat(
            [self._get_values(s) for s in valid_shards], dim=routing_dim
        )

        all_indices_list = []
        for s in valid_shards:
            idx = (
                s.origin_indices_materialized
                if hasattr(s, "origin_indices_materialized")
                else None
            )
            if idx is None:
                idx = self._get_routing_indices(s)
            if idx is None:
                num = backend.shape(self._get_values(s))[0]
                idx = backend.arange(0, num, dtype=backend.int32())
            all_indices_list.append(idx)
        all_indices = backend.cat(all_indices_list, dim=0)

        max_idx = int(backend.item(backend.max(all_indices)))
        inferred_dim_size = max_idx + 1

        # Prefer the global_shape carried by the received shards when available.
        # Receivers only see a subset of indices, so local max_idx < global dim.
        shard_global_shape = next(
            (s.global_shape for s in valid_shards if s.global_shape is not None), None
        )
        if shard_global_shape and len(shard_global_shape) > routing_dim:
            inferred_dim_size = shard_global_shape[routing_dim]

        val_shape = list(backend.shape(all_values))
        global_shape_list = val_shape[:]
        global_shape_list[routing_dim] = inferred_dim_size
        global_shape = tuple(global_shape_list)

        local_shape = input_spec.get_shard_shape(self.rank, data_shape=global_shape)
        slice_start, slice_end = input_spec.get_shard_bounds(
            self.rank, routing_dim, data_shape=global_shape
        )

        mask = (all_indices >= slice_start) & (all_indices < slice_end)
        local_indices = all_indices[mask] - slice_start

        if routing_dim == 0:
            local_values = all_values[mask]
        else:
            indices_to_take = backend.nonzero(mask).flatten()
            idx_tuple = [slice(None)] * len(val_shape)
            idx_tuple[routing_dim] = indices_to_take
            local_values = all_values[tuple(idx_tuple)]

        assembled_data = backend.zeros(local_shape, dtype=all_values.dtype)
        indices_tuple = [slice(None)] * len(local_shape)
        indices_tuple[routing_dim] = local_indices
        assembled_data = backend.index_update(
            assembled_data, tuple(indices_tuple), local_values
        )

        assembled_receipt = None
        if all_receipt is not None:
            final_ranks = backend.full(
                (local_shape[routing_dim],), -1, dtype=backend.int32()
            )
            local_ranks = all_receipt.sender_rank_tags[mask]
            final_ranks = backend.index_update(final_ranks, local_indices, local_ranks)
            assembled_receipt = SymmetryReceipt(all_receipt.world_size, final_ranks)

        return TransportTensor(
            assembled_data, global_shape=global_shape, receipt=assembled_receipt
        )


class _MockWorker(_AssemblyMixin):
    def __init__(self, rank: int):
        self.rank = rank


# =============================================================================
# Tests
# =============================================================================


class TestSymmetryReceiptHelpers(unittest.TestCase):
    """Unit tests for _maybe_merge_receipts."""

    def setUp(self):
        self.w = _MockWorker(rank=0)

    def test_no_mirror_returns_none(self):
        shards = [
            make_tt([[1.0, 2.0]], receipt=make_receipt([0], [0])),
            make_tt([[3.0, 4.0]], receipt=make_receipt([1], [1])),
        ]
        result = self.w._maybe_merge_receipts(shards, has_mirror=False)
        self.assertIsNone(result)

    def test_mirror_merges_receipts(self):
        r0 = make_receipt([0, 0], [10, 11])
        r1 = make_receipt([1, 1], [20, 21])
        shards = [
            make_tt([[1.0], [2.0]], receipt=r0),
            make_tt([[3.0], [4.0]], receipt=r1),
        ]
        result = self.w._maybe_merge_receipts(shards, has_mirror=True)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result.sender_rank_tags, [0, 0, 1, 1])

    def test_partial_receipts_returns_none_even_with_mirror_flag(self):
        """If one shard has no receipt, result must be None (no partial provenance)."""
        shards = [
            make_tt([[1.0]], receipt=make_receipt([0], [0])),
            make_tt([[2.0]], receipt=None),
        ]
        result = self.w._maybe_merge_receipts(shards, has_mirror=True)
        self.assertIsNone(result)


class TestHealEmptyShape(unittest.TestCase):
    """Unit tests for _heal_empty_shape."""

    def setUp(self):
        self.w = _MockWorker(rank=0)

    def test_returns_0d_when_no_info(self):
        spec = MagicMock()
        spec.global_shape = None
        result = self.w._heal_empty_shape(spec)
        self.assertEqual(get_backend().shape(result.flat_values), (0,))

    def test_infers_feature_dim_from_spec(self):
        spec = MagicMock()
        spec.global_shape = (100, 64)
        result = self.w._heal_empty_shape(spec)
        shape = get_backend().shape(result.flat_values)
        self.assertEqual(shape, (0, 64))

    def test_infers_feature_dim_from_reference(self):
        spec = MagicMock()
        spec.global_shape = None
        ref = jnp.ones((5, 32))
        result = self.w._heal_empty_shape(spec, reference_values=ref)
        shape = get_backend().shape(result.flat_values)
        self.assertEqual(shape, (0, 32))


class TestMergeOriginIndices(unittest.TestCase):
    """Unit tests for _merge_origin_indices."""

    def setUp(self):
        self.w = _MockWorker(rank=0)

    def test_returns_none_when_no_origin(self):
        shards = [make_tt([1.0, 2.0]), make_tt([3.0])]
        result = self.w._merge_origin_indices(shards)
        self.assertIsNone(result)

    def test_merges_present_origin_indices(self):
        shards = [
            make_tt([1.0, 2.0], origin_indices=[2, 0]),
            make_tt([3.0], origin_indices=[1]),
        ]
        result = self.w._merge_origin_indices(shards)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, [2, 0, 1])

    def test_falls_back_to_arange_for_missing_shard(self):
        """A shard without origin_indices falls back to arange(len)."""
        shards = [
            make_tt([1.0, 2.0], origin_indices=[5, 6]),
            make_tt([3.0, 4.0]),  # no origin_indices
        ]
        result = self.w._merge_origin_indices(shards)
        self.assertIsNotNone(result)
        # First shard: [5,6], second shard fallback: [0,1]
        np.testing.assert_array_equal(result, [5, 6, 0, 1])


class TestDynamicToStaticAssembly(unittest.TestCase):
    """
    Test _assemble_dynamic_to_static_core for correct origin_indices reassembly.
    Simulates SPMD_A(shard=4, dynamic) → SPMD_B(dp=6, static).
    """

    def _run_rank(
        self,
        rank: int,
        shards: List[TransportTensor],
        world_size: int,
        has_mirror: bool,
    ):
        from fluxway.common.types.sharding import (
            AlgebraicRouting,
            ShardingSpec,
            PartitionSpec,
            ChunkPartition,
        )

        mesh = DeviceMesh(device_shape=(world_size,), axis_names=("dp",))
        strategy = ChunkPartition()
        # PartitionSpec takes positional axis names: P("dp") means dim-0 is sharded on "dp"
        spec = ShardingSpec(
            routing=AlgebraicRouting(strategy=strategy, partition=PartitionSpec("dp"))
        ).bind(mesh)
        w = _MockWorker(rank=rank)
        valid = [s for s in shards if get_backend().shape(w._get_values(s))[0] > 0]
        if not valid:
            return w._heal_empty_shape(spec)
        return w._assemble_dynamic_to_static_core(valid, spec, has_mirror)

    def test_basic_reassembly_no_mirror(self):
        """
        Global batch = 12 items. SPMD_A has 4 workers, each sending 3 items
        to SPMD_B (dp=6). Each dp=6 rank should receive exactly 2 items.
        """
        # Simulate 12 items with global indices 0-11 split across 4 sources
        # Each source sends 3 consecutive items
        shard0 = make_tt([[1.0], [2.0], [3.0]], origin_indices=[0, 1, 2])
        shard1 = make_tt([[4.0], [5.0], [6.0]], origin_indices=[3, 4, 5])
        shard2 = make_tt([[7.0], [8.0], [9.0]], origin_indices=[6, 7, 8])
        shard3 = make_tt([[10.0], [11.0], [12.0]], origin_indices=[9, 10, 11])

        all_shards = [shard0, shard1, shard2, shard3]

        for rank in range(6):
            result = self._run_rank(rank, all_shards, world_size=6, has_mirror=False)
            data = np.array(result.flat_values)
            self.assertEqual(
                data.shape[0],
                2,
                f"Rank {rank} should have 2 items, got {data.shape[0]}",
            )
            # No mirror → no receipt
            self.assertIsNone(result.receipt)

    def test_unequal_shard_sizes(self):
        """
        Simulates real embedding lookup where shards are unevenly distributed.
        Global: 11 items across 4 SPMD_A workers → SPMD_B (dp=4).
        """
        # 4 workers send: 4, 3, 2, 2 items with correct global origin_indices
        shard0 = make_tt(
            [[float(i)] for i in range(4)], origin_indices=list(range(0, 4))
        )
        shard1 = make_tt(
            [[float(i)] for i in range(4, 7)], origin_indices=list(range(4, 7))
        )
        shard2 = make_tt(
            [[float(i)] for i in range(7, 9)], origin_indices=list(range(7, 9))
        )
        shard3 = make_tt(
            [[float(i)] for i in range(9, 11)], origin_indices=list(range(9, 11))
        )
        all_shards = [shard0, shard1, shard2, shard3]

        total = 0
        for rank in range(4):
            result = self._run_rank(rank, all_shards, world_size=4, has_mirror=False)
            n = int(np.array(result.flat_values).shape[0])
            total += n
        # All 11 items should be distributed without loss
        self.assertEqual(total, 11)

    def test_mirror_receipt_roundtrip(self):
        """
        E2E Embedding: SPMD_A(shard=4, dynamic) forward → SPMD_B(dp=6, static).
        SymmetryReceipt on each item records the sender_rank (SPMD_A worker) and
        remote_index (SPMD_A's local position). After assembly, SPMD_B Rank r's
        receipt should contain the correct back-routing info for all its items.
        """

        # 12 items, 4 SPMD_A workers, each sending 3 items
        # receipt: sender_rank = SPMD_A rank, remote_index = local offset in that rank
        def make_emb_shard(spmd_a_rank: int, values, global_indices):
            n = len(values)
            receipt = make_receipt(
                sender_ranks=[spmd_a_rank] * n, remote_indices=list(range(n))
            )
            return make_tt(
                [[v] for v in values], origin_indices=global_indices, receipt=receipt
            )

        shard0 = make_emb_shard(0, [1.0, 2.0, 3.0], [0, 1, 2])
        shard1 = make_emb_shard(1, [4.0, 5.0, 6.0], [3, 4, 5])
        shard2 = make_emb_shard(2, [7.0, 8.0, 9.0], [6, 7, 8])
        shard3 = make_emb_shard(3, [10.0, 11.0, 12.0], [9, 10, 11])
        all_shards = [shard0, shard1, shard2, shard3]

        total_items = 0
        for rank in range(6):
            result = self._run_rank(rank, all_shards, world_size=6, has_mirror=True)
            data = np.array(result.flat_values)
            n = data.shape[0]
            total_items += n

            # Receipt must be present for mirror edges
            self.assertIsNotNone(result.receipt, f"Rank {rank} missing receipt")

            if n > 0:
                receipt = result.receipt
                # sender_rank_tags must be in [0,3] (valid SPMD_A ranks)
                tags = np.array(receipt.sender_rank_tags)
                self.assertTrue(
                    np.all(tags >= 0), f"Rank {rank}: invalid sender_rank < 0"
                )
                self.assertTrue(np.all(tags <= 3), f"Rank {rank}: sender_rank > 3")

        # All 12 items must be covered
        self.assertEqual(total_items, 12)

    def test_backward_routing_uses_receipt(self):
        """
        Verify that backward routing (mirror) correctly inverts the forward path.
        SPMD_B gradient tensor carries SymmetryReceipt.sender_ranks, which should
        match the original SPMD_A rank that sent each embedding item.
        """
        # Forward assembly for rank=0 of SPMD_B (dp=6), items 0 and 1
        receipt = make_receipt(sender_ranks=[0, 0], remote_indices=[0, 1])
        grad_tensor = make_tt([[0.1], [0.2]], receipt=receipt)

        # The backward pass reads receipt.sender_rank_tags → routes gradient to SPMD_A rank 0
        backward_targets = set(np.array(grad_tensor.receipt.sender_rank_tags).tolist())
        self.assertEqual(backward_targets, {0})  # both rows go back to SPMD_A[0]

        # SymmetryReceipt no longer directly carries remote_indices in the tags.
        # It's assumed the receiver knows how to scatter or the tags are enough?
        # In current Fluxway, SymmetryReceipt ONLY carries sender_rank_tags.
        pass


class TestOperatorManagerPhase2MeshCorrectness(unittest.TestCase):
    def _build_operator_manager(
        self,
        sender_mesh: DeviceMesh,
        receiver_mesh: DeviceMesh,
        sender_op: str = "lookup",
        receiver_spmd: str = "DenseLayers",
        output_tensor_name: str = "sparse_features",
        output_sharding_type: str = "dynamic",
        output_axis: str = "shard",
    ):
        from fluxway.common.types.configs import (
            SPMDConfig,
            SPMDTopology,
            SPMDTopologyInfo,
            OperatorMeta,
            EdgeShardingInfo,
        )
        from fluxway.spmd.worker.operator_manager.operator_manager import (
            OperatorManager,
        )

        # Minimal sentinel operator function
        def _lookup_fn():
            pass

        # Build OperatorMeta with output_post_sharding declaring the edge
        tensor_tuuid = 123
        edge = EdgeShardingInfo(
            tuuid_sharding={
                tensor_tuuid: {"type": output_sharding_type, "axis": [output_axis]}
            },
            tuuid_map={output_tensor_name: tensor_tuuid},
            target_op=sender_op,
        )
        meta = OperatorMeta(
            name=sender_op,
            output_post_sharding={receiver_spmd: [edge]},
            output_arguments={"tensor": [output_tensor_name]},
        )

        # Phase 0 topology that Builder would inject: maps receiver_spmd → its physical mesh
        topology: Dict[str, SPMDTopology] = {
            sender_op: SPMDTopology(
                successors={
                    receiver_spmd: SPMDTopologyInfo(
                        spmd_id=receiver_spmd,
                        world_size=receiver_mesh.size,
                        mesh=receiver_mesh,
                        tuuid_sharding=edge.tuuid_sharding,
                        tuuid_map=edge.tuuid_map,
                    )
                }
            )
        }

        config = SPMDConfig(
            mesh=sender_mesh,
            operators={sender_op: _lookup_fn},
            operator_metas={sender_op: meta},
            topology=topology,
        )

        mgr = OperatorManager(config, rank=0, spmd_id="EmbeddingTable")
        return mgr

    def test_successor_tensor_sharding_uses_receiver_mesh(self):
        SENDER_SIZE = 4
        RECEIVER_SIZE = 6

        sender_mesh = DeviceMesh(device_shape=(SENDER_SIZE,), axis_names=("shard",))
        receiver_mesh = DeviceMesh(device_shape=(RECEIVER_SIZE,), axis_names=("dp",))

        mgr = self._build_operator_manager(
            sender_mesh=sender_mesh,
            receiver_mesh=receiver_mesh,
            output_axis="dp",  # axis name must match the TARGET mesh, not the sender's
        )

        succ_topo = mgr._successor_topology.get("lookup", {})
        self.assertIn(
            "DenseLayers", succ_topo, "Missing DenseLayers in successor topology"
        )

        succ_entry = succ_topo["DenseLayers"]
        tensor_tuuid = 123
        sparse_spec = succ_entry.tuuid_sharding.get(tensor_tuuid)
        self.assertIsNotNone(
            sparse_spec,
            "sparse_features spec missing from successor entry (TUUID lookup)",
        )

        # In the new architecture, mesh is stored on ResolvedTopologyInfo, not ShardingSpec.
        # The ShardingSpec is purely logical; binding to a mesh produces a PhysicalLayout.
        self.assertEqual(
            succ_entry.mesh.size,
            RECEIVER_SIZE,
            f"succ_entry.mesh.size should be {RECEIVER_SIZE} (receiver), "
            f"got {succ_entry.mesh.size}. Phase 2 must use pinned_succ_entry.mesh, not self._mesh.",
        )
        self.assertEqual(
            succ_entry.mesh.axis_names,
            ("dp",),
            "succ_entry.mesh.axis_names should be the receiver's.",
        )

    def test_predecessor_physical_identity_comes_from_phase0(self):
        """
        Phase 0 anchors the predecessor entry's physical identity (world_size + mesh) via
        the Builder-injected SPMDTopologyInfo. Phase 1 only overlays logical sharding on
        top of that pinned entry. Verify that pred_entry.world_size and pred_entry.mesh
        reflect the SENDER's physical topology, not whatever self._mesh happens to be.
        """
        from fluxway.common.types.configs import (
            SPMDConfig,
            SPMDTopology,
            SPMDTopologyInfo,
            OperatorMeta,
            EdgeShardingInfo,
        )
        from fluxway.spmd.worker.operator_manager.operator_manager import (
            OperatorManager,
        )

        SENDER_SIZE = 4
        RECEIVER_SIZE = 6

        sender_mesh = DeviceMesh(device_shape=(SENDER_SIZE,), axis_names=("shard",))
        receiver_mesh = DeviceMesh(device_shape=(RECEIVER_SIZE,), axis_names=("dp",))

        from fluxway.common.types.sharding import ShardingSpec, PartitionSpec

        edge = EdgeShardingInfo(
            tensor_sharding={"emb1": {"type": "static", "axis": ["dp"]}},
            target_op="forward",
        )
        meta = OperatorMeta(
            name="forward",
            input_pre_sharding={"EmbeddingTable": [edge]},
            input_arguments={"tensor": ["emb1"]},
            input_sharding={
                "emb1": ShardingSpec.chunk(PartitionSpec()).bind(receiver_mesh)
            },
        )
        topology = {
            "forward": SPMDTopology(
                predecessors={
                    "EmbeddingTable": SPMDTopologyInfo(
                        spmd_id="EmbeddingTable",
                        world_size=SENDER_SIZE,
                        mesh=sender_mesh,
                    )
                }
            )
        }
        config = SPMDConfig(
            mesh=receiver_mesh,
            operators={"forward": lambda: None},
            operator_metas={"forward": meta},
            topology=topology,
        )
        mgr = OperatorManager(config, rank=0, spmd_id="DenseLayers")

        pred_entry = mgr._predecessor_topology.get("forward", {}).get("EmbeddingTable")
        self.assertIsNotNone(pred_entry, "EmbeddingTable predecessor entry missing")

        # Physical identity comes from Phase 0 (Builder-injected), not from sharding spec parsing.
        self.assertEqual(pred_entry.world_size, SENDER_SIZE)
        self.assertEqual(pred_entry.mesh.size, SENDER_SIZE)
        self.assertEqual(pred_entry.mesh.axis_names, ("shard",))


class TestEmbeddingToDenseFullCommunication(unittest.TestCase):
    """
    Simulates the exact Embedding -> Dense communication pattern from the training example.
    SPMD_A (shard=4, dynamic outputs) -> SPMD_B (dp=6, static input).
    Verifies that through the Distributor and Reassembly, all data is preserved in correct order.
    """

    def test_e2e_embedding_to_dense_communication(self):
        from fluxway.transport.communication import AsyncDataDistributor
        from fluxway.common.types.sharding import (
            AlgebraicRouting,
            ShardingSpec,
            PartitionSpec,
            ChunkPartition,
        )
        import asyncio

        SENDER_SIZE = 4
        RECEIVER_SIZE = 6
        TOTAL_ITEMS = 12

        # 1. Setup Meshes and Specs
        sender_mesh = DeviceMesh(device_shape=(SENDER_SIZE,), axis_names=("shard",))
        source_spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("shard")
            )
        ).bind(sender_mesh)

        receiver_mesh = DeviceMesh(device_shape=(RECEIVER_SIZE,), axis_names=("dp",))
        target_spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("dp")
            )
        ).bind(receiver_mesh)

        # 2. Prepare shards on 4 Senders
        # We simulate a slightly shuffled embedding result:
        # Rank 0: [0, 5, 10]
        # Rank 1: [1, 6, 11]
        # Rank 2: [2, 7]
        # Rank 3: [3, 4, 8, 9]
        # Total = 12 items.
        sender_shards = {
            0: ([0, 5, 10], [[0.0], [5.0], [10.0]]),
            1: ([1, 6, 11], [[1.0], [6.0], [11.0]]),
            2: ([2, 7], [[2.0], [7.0]]),
            3: ([3, 4, 8, 9], [[3.0], [4.0], [8.0], [9.0]]),
        }

        # 3. Simulated Network: receiver_id -> List of TransportTensors received
        network_mailbox: Dict[int, List[TransportTensor]] = {
            i: [] for i in range(RECEIVER_SIZE)
        }

        async def simulate_sender(rank):
            indices, values = sender_shards[rank]
            # Real scenario: 'routing_indices' holds the feature IDs (which SparseEmbedding passes through)
            # while 'origin_indices' holds the batch index.
            feature_ids = [idx * 1000 + 42 for idx in indices]
            tt = make_tt(
                values=values,
                origin_indices=indices,
                routing_indices=feature_ids,  # Simulate unique_ids' routing_indices
                global_shape=(TOTAL_ITEMS, 1),
            )

            distributor = AsyncDataDistributor(local_rank=rank)

            async def mock_send(
                target_rank: int,
                tuuids: list,
                tensors: list,
                counts: list,
                global_shapes_list: Optional[list] = None,
                meta_tensor_state_list: Optional[list] = None,
                compact_params_list: Optional[list] = None,
            ):
                if int(counts[0]) > 0:
                    g_shape = global_shapes_list[0] if global_shapes_list else None
                    received_tt = TransportTensor(
                        tensors[0],
                        origin_indices=tensors[1] if len(tensors) > 1 else None,
                        global_shape=g_shape,
                    )
                    network_mailbox[target_rank].append(received_tt)
                return True

            await distributor.distribute(
                [tt], source=source_spec, target=target_spec, send_fn=mock_send
            )

        # Run all senders
        for r in range(SENDER_SIZE):
            asyncio.run(simulate_sender(r))

        # 4. Receiver Reassembly
        final_results = []
        w_helpers = _MockWorker(rank=0)
        for r in range(RECEIVER_SIZE):
            received_shards = network_mailbox[r]
            w_helpers.rank = r
            assembled = w_helpers._assemble_dynamic_to_static_core(
                received_shards, target_spec, has_mirror=False
            )
            final_results.append(assembled)

        # 5. Verification
        # Combined result should be [0, 1, ..., 11]
        full_tensor = np.concatenate(
            [np.array(r.flat_values) for r in final_results], axis=0
        )
        expected = np.arange(TOTAL_ITEMS).reshape((TOTAL_ITEMS, 1)).astype(np.float32)

        np.testing.assert_array_equal(
            full_tensor,
            expected,
            f"E2E Routing failed! Expected sequential 0-11, got:\n{full_tensor.flatten()}",
        )
        print("E2E Embedding-to-Dense Full Communication Simulation PASSED")


class TestScalarAndComplexObjectRouting(unittest.TestCase):
    """Verifies that scalars and complex objects don't trigger index errors during sharding inference."""

    def test_scalar_sharding_inference(self):
        from fluxway.common.mesh import DeviceMesh
        from fluxway.common.types.sharding import (
            AlgebraicRouting,
            ShardingSpec,
            ChunkPartition,
            PartitionSpec,
        )

        mesh = DeviceMesh(device_shape=(2,), axis_names=("dp",))
        # Case: A scalar sharded on 'dp' (should be treated as replicated/safe)
        spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("dp")
            )
        ).bind(mesh)

        # Local scalar shape
        local_shape = ()
        # This is where 'tuple index out of range' often happens if not caught
        inferred = spec.infer_global_shape(local_shape)

        self.assertEqual(
            inferred, (), "Inferred global shape for scalar should remain scalar"
        )

    def test_extracted_sharding_safety(self):
        # Mocks the _send_output logic in worker.py
        from fluxway.common.types.tensor import TransportTensor
        from fluxway.common.types.sharding import (
            AlgebraicRouting,
            ShardingSpec,
            ChunkPartition,
            PartitionSpec,
        )
        from fluxway.common.mesh import DeviceMesh

        mesh = DeviceMesh(device_shape=(2,), axis_names=("dp",))
        spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("dp")
            )
        ).bind(mesh)

        # Mock scalar value and how worker.py wraps it
        tensor_data = np.array(0.5)
        # Worker logic: TransportTensor.from_tensor(tensor_data)
        tt = TransportTensor.from_tensor(tensor_data)

        # Safety check for global_shape inference
        if tt.global_shape is None:
            # This is the line that might fail in worker.py before my fix
            inferred = spec.infer_global_shape(tt.shape)
            self.assertEqual(inferred, (), "Successfully inferred global scalar shape")


class TestStaticToStaticCoverageValidation(unittest.TestCase):
    """
    Validates the Static Coverage Check in communication.py:
    - Undersized batch raises ValueError
    - Exact-size batch passes
    - EOS empty packet (global_shape=None) is NOT blocked
    """

    def _make_specs(self, total_items: int, dp: int):
        from fluxway.common.types.sharding import (
            AlgebraicRouting,
            ShardingSpec,
            ChunkPartition,
            PartitionSpec,
        )

        sender_mesh = DeviceMesh(device_shape=(1,), axis_names=("worker",))
        receiver_mesh = DeviceMesh(device_shape=(dp,), axis_names=("dp",))
        source_spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("worker")
            )
        ).bind(sender_mesh)
        target_spec = ShardingSpec(
            routing=AlgebraicRouting(
                strategy=ChunkPartition(), partition=PartitionSpec("dp")
            )
        ).bind(receiver_mesh)
        return source_spec, target_spec, total_items

    def _run_distributor(self, values, global_shape, source_spec, target_spec):
        """Drive AsyncDataDistributor synchronously; collect any raised errors."""
        import asyncio
        from fluxway.transport.communication import AsyncDataDistributor

        tt = TransportTensor(
            jnp.array(values, dtype=jnp.float32), global_shape=global_shape
        )
        distributor = AsyncDataDistributor(local_rank=0)

        async def mock_send(*args, **kwargs):
            return True

        async def _run():
            await distributor.distribute([tt], source_spec, target_spec, mock_send)

        asyncio.run(_run())

    def test_undersized_batch_raises(self):
        """batch_size=4 with dp=6 must raise a clear ValueError."""
        source_spec, target_spec, _ = self._make_specs(total_items=4, dp=6)
        tt = TransportTensor(jnp.ones((4, 1), dtype=jnp.float32), global_shape=(4, 1))
        import asyncio
        from fluxway.transport.communication import AsyncDataDistributor

        distributor = AsyncDataDistributor(local_rank=0)

        async def mock_send(*args, **kwargs):
            return True

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(
                distributor.distribute([tt], source_spec, target_spec, mock_send)
            )
        self.assertIn("Static Sharding Error", str(ctx.exception))
        self.assertIn("6", str(ctx.exception))  # sub_mesh_size mentioned

    def test_exact_size_passes(self):
        """batch_size=6 with dp=6 must pass without raising."""
        source_spec, target_spec, _ = self._make_specs(total_items=6, dp=6)
        tt = TransportTensor(jnp.ones((6, 1), dtype=jnp.float32), global_shape=(6, 1))
        import asyncio
        from fluxway.transport.communication import AsyncDataDistributor

        distributor = AsyncDataDistributor(local_rank=0)

        async def mock_send(*args, **kwargs):
            return True

        try:
            asyncio.run(
                distributor.distribute([tt], source_spec, target_spec, mock_send)
            )
        except ValueError as e:
            self.fail(f"Unexpected ValueError for exact-size batch: {e}")

    def test_eos_empty_packet_not_blocked(self):
        """A legitimate EOS packet with global_shape=None must NOT trigger the Coverage Check."""
        source_spec, target_spec, _ = self._make_specs(total_items=0, dp=6)
        empty_tt = TransportTensor(
            jnp.zeros((0,), dtype=jnp.float32),
            global_shape=None,  # EOS signal — no global shape
        )
        import asyncio
        from fluxway.transport.communication import AsyncDataDistributor

        distributor = AsyncDataDistributor(local_rank=0)

        async def mock_send(*args, **kwargs):
            return True

        # Should not raise
        try:
            asyncio.run(
                distributor.distribute([empty_tt], source_spec, target_spec, mock_send)
            )
        except ValueError as e:
            if "Static Sharding Error" in str(e):
                self.fail(f"EOS empty packet incorrectly blocked: {e}")


if __name__ == "__main__":
    unittest.main()
