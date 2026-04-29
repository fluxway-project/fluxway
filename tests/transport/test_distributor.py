import pytest

from typing import Any

from fluxway.common.types.tensor import TransportTensor
from fluxway.common.types.sharding import ShardingSpec, P
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.communication import DataDistributor, TransportReceiver
from fluxway.common.backend import set_backend, get_backend
from fluxway.common.execution import shield, compile


# Force JAX backend
@pytest.fixture(autouse=True)
def use_jax_backend():
    """Forces the system to use the JAX engine backend for this test suite."""
    set_backend("jax")
    yield


class VirtualRank:
    """Mock node simulating a single physical rank in a distributed system pipeline."""

    def __init__(self, rank: int, mesh: DeviceMesh):
        self.rank = rank
        self.mesh = mesh
        self.distributor = DataDistributor(rank)
        # Using the new unified receiver
        self.receiver = TransportReceiver(wait_for_n_slices=0)
        self.data_store: dict[str, Any] = {}

    def send(
        self,
        target_ranks: list[int],
        tensors: list[Any],
        counts: list[int],
        cluster_receivers: dict[int, "VirtualRank"],
    ):
        """Simulates network payload transmission to target ranks via direct memory delivery."""
        for target_rank in target_ranks:
            if target_rank in cluster_receivers:
                # Direct memory transfer for simulation
                cluster_receivers[target_rank].receiver.receive_slices(
                    self.rank, tensors, counts
                )

    def process_incoming(self) -> list[Any]:
        """Extracts and consumes the buffered payloads received over the mock network."""
        return self.receiver.get_buffered_slices()


def test_embedding_lookup_flow():
    """Simulates a 3-step pipeline: Input (SPMD0) -> Embedding (SPMD1) -> Forward Output (SPMD2)."""

    # --- 1. Setup Global Data & Meshes ---
    bk = get_backend()

    # Global Data: 12 keys (data), 12 indices (coordinates)
    global_indices = bk.arange(0, 12, dtype=bk.int32())
    # Users input can be anything, here we use offset values to distinguish from indices
    global_keys = bk.arange(100, 112, dtype=bk.int32())

    vocab_size = 8
    emb_dim = 2
    # reshape
    _tmp = bk.arange(0, vocab_size * emb_dim, dtype=bk.float32())
    global_emb_table = bk.reshape(_tmp, (vocab_size, emb_dim))

    global_ids = global_keys % vocab_size
    expected_global_output = global_emb_table[global_ids]

    # Meshes
    mesh0 = DeviceMesh(("x",), (2,), [0, 1])
    mesh1 = DeviceMesh(("y",), (4,), [0, 1, 2, 3])
    mesh2 = DeviceMesh(("z",), (2,), [0, 1])
    from fluxway.common.index_descriptor import FunctionalIndex
    from fluxway.common.types.sharding import AlgebraicRouting

    # Sharding Specs
    spec0 = ShardingSpec.chunk(P("x")).bind(mesh0)

    def hash_fn(keys):
        return keys % vocab_size

    base_hash_strategy = ShardingSpec.hash(P("y")).strategy
    spec1 = ShardingSpec(
        routing=AlgebraicRouting(
            strategy=base_hash_strategy,
            partition=P("y"),
            index_descriptor=FunctionalIndex(hash_fn),
        )
    ).bind(mesh1)
    spec2 = ShardingSpec.chunk(P("z")).bind(mesh2)

    # --- 2. Simulation Environment ---

    # Create Virtual Ranks [0, 1, 2, 3] (Super-set of all ranks)
    cluster = {i: VirtualRank(i, mesh1) for i in range(4)}  # Max ranks = 4

    # --- 3. Forward Phase 1: Input (SPMD0) -> Embedding (SPMD1) ---
    # "Dispatch"

    # Step 3.1: Prepare Input Data on SPMD0 Ranks (0, 1)
    rank0_keys = global_keys[:6]
    rank1_keys = global_keys[6:]

    inputs = {
        0: TransportTensor(rank0_keys, origin_indices=global_indices[:6]),
        1: TransportTensor(rank1_keys, origin_indices=global_indices[6:]),
    }

    # Rank 0 sends
    dist0 = cluster[0].distributor

    def send_wrapper_0(target, tuuids, tensors, counts, *args, **kwargs):
        cluster[0].send([target], tensors, counts, cluster)
        return True

    # Check 1: Implicit Static Source + Content Addressable Target
    _res, original_sort_indices_0 = dist0.distribute(
        local_data=inputs[0],
        source=spec0,
        target=spec1,
        send_fn=send_wrapper_0,
        capacity=6,
    )

    # Rank 1 sends
    dist1 = cluster[1].distributor

    def send_wrapper_1(target, tuuids, tensors, counts, *args, **kwargs):
        cluster[1].send([target], tensors, counts, cluster)
        return True

    _res, original_sort_indices_1 = dist1.distribute(
        local_data=inputs[1],
        source=spec0,
        target=spec1,
        send_fn=send_wrapper_1,
        capacity=6,
    )

    # --- 4. Forward Phase 1 Verification: Input (SPMD0) -> Embedding (SPMD1) ---
    # Verify that keys were correctly routed to SPMD1 ranks based on hash_fn(key) = key % 8
    # Global keys: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
    # Hashes (k%8):[4,   5,   6,   7,   0,   1,   2,   3,   4,   5,   6,   7]
    # Mesh1 size 4, so Rank = Hash % 4.
    # Expected mapping:
    # Hash 0 -> Rank 0 (Key 104)
    # Hash 1 -> Rank 1 (Key 105)
    # Hash 2 -> Rank 2 (Key 106)
    # Hash 3 -> Rank 3 (Key 107)
    # Hash 4 -> Rank 0 (Key 100, 108)
    # Hash 5 -> Rank 1 (Key 101, 109)
    # Hash 6 -> Rank 2 (Key 102, 110)
    # Hash 7 -> Rank 3 (Key 103, 111)

    expected_rank0_keys = bk.tensor([104, 100, 108], dtype=bk.int32())
    expected_rank1_keys = bk.tensor([105, 101, 109], dtype=bk.int32())
    expected_rank2_keys = bk.tensor([106, 102, 110], dtype=bk.int32())
    expected_rank3_keys = bk.tensor([107, 103, 111], dtype=bk.int32())

    # --- 5. Forward Phase 2: Embedding Lookup (SPMD1) ---

    spmd1_results = {}

    for r in range(4):  # SPMD1 has 4 ranks
        # Receive dispatched keys
        shards = cluster[r].process_incoming()

        all_keys = []
        all_global_indices = []

        for tensors, counts in shards:
            if counts[0] > 0:
                # Valid unpadded data
                all_keys.append(tensors[0][: counts[0]])
                # origin_indices are packed into tensors[1] by the distributor
                all_global_indices.append(tensors[1][: counts[1]])

        if not all_keys:
            spmd1_results[r] = (
                bk.tensor([], dtype=bk.float32()),
                bk.tensor([], dtype=bk.int32()),
            )
            continue

        local_keys = bk.cat(all_keys)
        global_indices = bk.cat(all_global_indices)

        # Verification of received data on SPMD1
        if r == 0:
            assert bk.equal(bk.sort(local_keys)[0], bk.sort(expected_rank0_keys)[0])
        elif r == 1:
            assert bk.equal(bk.sort(local_keys)[0], bk.sort(expected_rank1_keys)[0])
        elif r == 2:
            assert bk.equal(bk.sort(local_keys)[0], bk.sort(expected_rank2_keys)[0])
        elif r == 3:
            assert bk.equal(bk.sort(local_keys)[0], bk.sort(expected_rank3_keys)[0])

        # Lookup
        local_emb = global_emb_table[local_keys % vocab_size]

        spmd1_results[r] = (local_emb, global_indices)

    # --- 5. Forward Phase 3: Combine (SPMD1 -> SPMD2) ---
    # "All-to-All Return"

    # optimization: Zero-Copy Receive
    # 1. Simulate "AllToAll" metadata exchange
    # We inspect spmd1_results to determine total size for destination ranks 0 and 1.

    # Destination mapping logic matches the plan:
    # Key -> Hash -> SPMD1 (0..3) -> Lookup -> SPMD2 (0..1)
    # The `spec2` (Target) is Chunk on SPMD2 ("z").
    # This means implicit static sharding of the GLOBAL domain.
    # Global indices 0..5 -> Rank 0.
    # Global indices 6..11 -> Rank 1.

    rank0_total_count = 0
    rank1_total_count = 0
    rank0_offsets = {}  # src_rank -> start_offset
    rank1_offsets = {}
    rank0_tags_list = []
    rank1_tags_list = []

    # We must iterate in the same order as distribute (0..3) to set offsets correctly
    # if we assume sequential arrival, OR we allocate specific slots per rank.
    # Since specific slots is cleaner:

    for r in range(4):
        _emb, _idx = spmd1_results[r]
        if len(_emb) == 0:
            continue

        # Coordinate-based routing: Use _idx (global_indices) to determine destination
        val_idx = bk.to_numpy(_idx)
        count0 = 0
        count1 = 0
        for v in val_idx:
            if v < 6:
                count0 += 1
                rank0_tags_list.append(r)
            else:
                count1 += 1
                rank1_tags_list.append(r)

        if count0 > 0:
            rank0_offsets[r] = rank0_total_count
            rank0_total_count += count0

        if count1 > 0:
            rank1_offsets[r] = rank1_total_count
            rank1_total_count += count1

    # 2. Pre-allocate buffers (Zero-Copy Target)
    # Rank 0
    buffer0 = bk.zeros((rank0_total_count, emb_dim), dtype=bk.float32())
    cluster[0].receiver.register_zero_copy_buffers([buffer0], rank0_offsets)

    # Rank 1
    buffer1 = bk.zeros((rank1_total_count, emb_dim), dtype=bk.float32())
    cluster[1].receiver.register_zero_copy_buffers([buffer1], rank1_offsets)

    # output_shards no longer needed for Zero-Copy flow
    # but we can keep it empty or refactor verify?
    # We will verify `buffer0` and `buffer1` directly.

    for r in range(4):
        local_emb, global_indices = spmd1_results[r]
        if len(local_emb) == 0:
            continue

        t_tensor = TransportTensor(
            local_emb, origin_indices=global_indices, global_shape=(12, emb_dim)
        )

        dist = cluster[r].distributor

        # Send wrapper calling receive_slices
        def send_wrapper_combine(
            target, tuuids, tensors, counts, *args, source_rank=r, **kwargs
        ):
            if target in cluster:
                cluster[target].receiver.receive_slices(source_rank, tensors, counts)
            return True

        # Check 2: Content Addressable Source (Hash) -> Implicit Static Target (Chunk)
        _res, _ = dist.distribute(
            local_data=t_tensor,
            source=spec1,
            target=spec2,
            send_fn=send_wrapper_combine,
            capacity=12,
        )

    # --- 6. Verify Forward Output ---

    # Verify Rank 0 (indices 0..5)
    # Zero-Copy: check the buffer directly
    assert cluster[0].receiver.target_buffers is not None, (
        "Rank 0 target_buffers should be registered"
    )
    rank0_data = cluster[0].receiver.target_buffers[0]

    expected_r0 = expected_global_output[:6]
    # Check shape
    assert rank0_data.shape[0] == 6, f"Rank 0 shape mismatch: {rank0_data.shape} vs 6"

    # Verify Values
    r0_sorted, _ = bk.sort(rank0_data.flatten())
    exp0_sorted, _ = bk.sort(expected_r0.flatten())
    assert bk.equal(r0_sorted, exp0_sorted), "Rank 0 Value Mismatch"

    # Verify Rank 1 (indices 6..11)
    rank1_data = cluster[1].receiver.target_buffers[0]

    assert rank1_data.shape[0] == 6, f"Rank 1 shape mismatch: {rank1_data.shape} vs 6"
    r1_sorted, _ = bk.sort(rank1_data.flatten())
    exp1_sorted, _ = bk.sort(expected_global_output[6:].flatten())
    assert bk.equal(r1_sorted, exp1_sorted), "Rank 1 Value Mismatch"

    # --- Verification: Reconstruct Ragged Tensor using meta ---
    # Rank 0 reconstructs its original ragged tensor embedding
    inv_sort_0 = bk.argsort(original_sort_indices_0, stable=False)
    flat_emb_ordered_0 = bk.take(rank0_data, inv_sort_0)

    _ = TransportTensor(flat_emb_ordered_0, inputs[0].nested_splits)

    # Verify exact match without sorting
    expected_ordered_0 = expected_global_output[:6]
    assert bk.equal(flat_emb_ordered_0, expected_ordered_0), (
        "Rank 0 Reconstructed Tensor Mismatch"
    )

    # --- 7. Backward Pass: Gradients (SPMD2 -> SPMD1) ---
    # "Dispatch Gradients"

    from fluxway.common.types.tensor import SymmetryReceipt

    grads0_tensor = cluster[0].receiver.target_buffers[0] * 2.0
    grads1_tensor = cluster[1].receiver.target_buffers[0] * 2.0

    assert grads0_tensor.shape[0] == 6
    assert grads1_tensor.shape[0] == 6

    rank0_tags = bk.tensor(rank0_tags_list, dtype=bk.int32())
    rank1_tags = bk.tensor(rank1_tags_list, dtype=bk.int32())

    grads0 = TransportTensor(
        grads0_tensor,
        receipt=SymmetryReceipt(world_size=4, sender_rank_tags=rank0_tags),
    )
    grads1 = TransportTensor(
        grads1_tensor,
        receipt=SymmetryReceipt(world_size=4, sender_rank_tags=rank1_tags),
    )

    # mirror routing: follow the forward input spec (spec1) back to its source (mesh1)
    spec_mirror = ShardingSpec.follow("fwd").bind(mesh1)

    # --- 8. Backward Pass Execution (SPMD2 -> SPMD1) ---
    # We use Zero-Copy on SPMD1 (Receivers) for the Backward Update to ensure dense reception.

    # Pre-calculate Backward Offsets for SPMD1 (Embedding Ranks 0..3)
    # The source for this is SPMD2 (Ranks 0..1).
    # Forward Phase 3 sent data from SPMD1 (r) to SPMD2 (target).
    # Symmetric routing means SPMD1 (r) will receive from SPMD2 (target) in reverse.

    for r in range(4):
        _emb, _idx = spmd1_results[r]
        if len(_emb) == 0:
            continue

        # Determine how many elements this rank sent to each SPMD2 rank
        # Rank 2 (SPMD2 R0): values with index < 6
        # Rank 0/1 (SPMD2 R0/R1) can be mapped from cluster dict.
        # We need to register buffers on cluster[r].receiver

        bw_offsets = {}
        total_bw_count = 0

        # Consistent with Forward Phase 3 routing:
        # Global indices 0..5 -> SPMD2 Rank 0
        # Global indices 6..11 -> SPMD2 Rank 1

        # We iterate over SPMD2 ranks (0, 1) to determine offsets for incoming gradients
        for spmd2_r in [0, 1]:
            count = 0
            val_idx = bk.to_numpy(_idx)
            for v in val_idx:
                if spmd2_r == 0 and v < 6:
                    count += 1
                elif spmd2_r == 1 and v >= 6:
                    count += 1

            if count > 0:
                # In backward pass, the source is spmd2_r
                bw_offsets[spmd2_r] = total_bw_count
                total_bw_count += count

        # Register Zero-Copy buffer for gradients on this rank
        # It must match the forward output shape exactly
        bw_grad_buffer = bk.zeros(_emb.shape, dtype=bk.float32())
        cluster[r].receiver.register_zero_copy_buffers([bw_grad_buffer], bw_offsets)

    # Send from SPMD2 Rank 0
    dist_bk_0 = cluster[0].distributor

    def send_wrapper_bk_0(target, tuuids, tensors, counts, *args, **kwargs):
        cluster[target].receiver.receive_slices(0, tensors, counts)
        return True

    _res, _ = dist_bk_0.distribute(
        local_data=grads0,
        source=spec2,
        target=spec_mirror,
        send_fn=send_wrapper_bk_0,
        capacity=6,
    )

    # Send from SPMD2 Rank 1
    dist_bk_1 = cluster[1].distributor

    def send_wrapper_bk_1(target, tuuids, tensors, counts, *args, **kwargs):
        cluster[target].receiver.receive_slices(1, tensors, counts)
        return True

    _res, _ = dist_bk_1.distribute(
        local_data=grads1,
        source=spec2,
        target=spec_mirror,
        send_fn=send_wrapper_bk_1,
        capacity=6,
    )

    # --- 9. Final Verification: Dense Gradients on SPMD1 ---

    for r in range(4):
        expected_grads = spmd1_results[r][0] * 2.0
        if len(expected_grads) == 0:
            continue

        # Verify dense buffer directly
        # The receiver should have written everything into target_buffers[0]
        actual_grads = cluster[r].receiver.target_buffers[0]

        assert actual_grads.shape == expected_grads.shape, (
            f"Rank {r} dense grad shape mismatch"
        )

        # Value check: MirrorPartition + SymmetryReceipt should preserve exact order
        assert bk.equal(actual_grads, expected_grads), (
            f"Rank {r} dense grad value/order mismatch"
        )

        # Confirm "Dense" expectation: local_keys length == local_grads length
        assert len(actual_grads) == len(spmd1_results[r][0]), (
            f"Rank {r} n-to-n consistency check failed"
        )

    # --- 10. Backward Return (SPMD1 -> SPMD0) ---
    # Final step: Gradients return from Embedding (SPMD1) to Input (SPMD0)

    # We need to capture results on Rank 0 and 1 (SPMD0)
    spmd0_grad_acc = {0: [], 1: []}

    # spec0 is the source for Phase 1 dispatch
    # we use speculative mirror to go back
    spec_mirror_0 = ShardingSpec.follow("fwd").bind(mesh0)

    for r in range(4):
        # The gradient on SPMD1 is the dense buffer we just verified
        dense_grads = cluster[r].receiver.target_buffers[0]
        if len(dense_grads) == 0:
            continue

        # We need the tags from Phase 1 to route back
        # The distributor collected these during forward dispatch
        dist = cluster[r].distributor

        # In a real system, the transport would handle the SymmetryReceipt.
        # Here we simulate by creating a TransportTensor with the correct metadata.
        # Since Forward Phase 1 used content-based routing, the receipt was implicit.
        # However, for the simulation to work with SymmetryReceipt, we need to know who sent what.

        # For simplicity in this test, we know the routing symmetry:
        # Global Keys 0..5 (SPMD0 Rank 0) -> SPMD1
        # Global Keys 6..11 (SPMD0 Rank 1) -> SPMD1

        # We simulate the receipt by looking at the origin_indices we sent in Forward Phase 3
        _, global_indices = spmd1_results[r]

        # Create a tag list: source_rank is SPMD0 rank (0 or 1)
        # Global index < 6 came from Rank 0, index >= 6 came from Rank 1
        tags = []
        for idx in bk.to_numpy(global_indices):
            tags.append(0 if idx < 6 else 1)

        grad_tensor_1_to_0 = TransportTensor(
            dense_grads,
            origin_indices=global_indices,
            receipt=SymmetryReceipt(
                world_size=2, sender_rank_tags=bk.tensor(tags, dtype=bk.int32())
            ),
        )

        def send_wrapper_1_to_0(target, tuuids, tensors, counts, *args, **kwargs):
            if target in spmd0_grad_acc:
                spmd0_grad_acc[target].append((tensors, counts))
            return True

        # Dispatch back to SPMD0
        dist.distribute(
            local_data=grad_tensor_1_to_0,
            source=spec1,
            target=spec_mirror_0,
            send_fn=send_wrapper_1_to_0,
            capacity=12,
        )

    # --- 11. Final Verification on SPMD0 ---
    for r in [0, 1]:
        shards = spmd0_grad_acc[r]
        all_g = []
        for tensors, counts in shards:
            if counts[0] > 0:
                all_g.append(tensors[0][: counts[0]])

        final_grads = bk.cat(all_g)

        # Expected: 2.0 * input embeddings
        # Rank 0: indices 0..5, Rank 1: indices 6..11
        if r == 0:
            expected = expected_global_output[:6] * 2.0
        else:
            expected = expected_global_output[6:] * 2.0

        assert len(final_grads) == 6
        assert bk.equal(
            bk.sort(final_grads.flatten())[0], bk.sort(expected.flatten())[0]
        )


def test_jit_shield_in_jit():
    """Verifies that transport can be shielded from JIT."""
    bk = get_backend()

    mesh = DeviceMesh(("x",), (1,), [0])
    spec = ShardingSpec.chunk(P("x")).bind(mesh)

    side_effects = []

    def side_effect_send(target, tuuids, tensors, counts, *args, **kwargs):
        count = counts[0]
        if count > 0:
            side_effects.append(tensors[0][:count])
        return True

    distributor = DataDistributor(0)

    @shield(result_shape=None)
    def shielded_distribute(data):
        _res, _ = distributor.distribute(
            local_data=data,
            source=spec,
            target=spec,
            send_fn=side_effect_send,
            capacity=3,
        )

    import jax

    @jax.jit
    def train_step(x):
        y = x * 2
        tt = TransportTensor(y)
        shielded_distribute(tt)
        return y

    input_data = bk.tensor([1, 2, 3], dtype=bk.float32())
    result = train_step(input_data)

    assert bk.equal(result, input_data * 2)
    assert len(side_effects) == 1

    import numpy as np

    assert np.allclose(side_effects[0], [2, 4, 6])


def test_jit_plan_and_shard():
    """Verify that plan_and_shard distributes data correctly across ranks."""
    bk = get_backend()

    mesh = DeviceMesh(("x",), (2,), [0, 1])
    spec = ShardingSpec.chunk(P("x")).bind(mesh)

    # Rank 0 holds the first half of a 4-element global tensor
    data = bk.tensor([1.0, 2.0, 3.0, 4.0], dtype=bk.float32())
    tt = TransportTensor(data[:2], global_shape=(4,))

    distributor = DataDistributor(0)
    received = []

    def mock_send(
        target_id,
        tuuids,
        tensors,
        counts,
        global_shapes=None,
        meta_state_list=None,
        compact_params=None,
    ):
        count = counts[0]
        if count > 0:
            received.append((target_id, int(count)))
        return True

    distributor.distribute(tt, source=spec, target=spec, send_fn=mock_send)

    # Rank 0 (chunk) owns indices [0,1] → should send 2 items to target rank 0 only
    assert len(received) == 1
    assert received[0] == (0, 2)


def test_unified_compile():
    """Verifies that unified compile correctly JITs functions."""
    bk = get_backend()

    @compile
    def simple_mul(x):
        return x * 2.0

    res = simple_mul(bk.tensor([1.0, 2.0]))
    assert bk.equal(res, bk.tensor([2.0, 4.0]))

    side_effects = []

    @compile
    def traced_fn(x):
        side_effects.append("tracing")
        return x + 1.0

    t1 = bk.tensor([1.0])
    traced_fn(t1)
    assert len(side_effects) == 1, "Should trace once"

    traced_fn(t1)
    assert len(side_effects) == 1, "Should not re-trace"


def test_colocation_dispatch():
    """Verify that distribute can take a List of TransportTensors and share the same plan."""
    bk = get_backend()

    mesh = DeviceMesh(("x",), (2,), [0, 1])
    spec_src = ShardingSpec.chunk(P("x")).bind(mesh)
    spec_dst = ShardingSpec.chunk(P("x")).bind(mesh)

    # 4 elements total
    keys = bk.tensor([10, 20, 30, 40], dtype=bk.int32())
    vals = bk.tensor([0.1, 0.2, 0.3, 0.4], dtype=bk.float32())

    # 4 elements total, distributed across 2 ranks (2 elements per rank)
    # Rank 0 (dist1) should have the first 2 elements locally
    tt_keys = TransportTensor(keys[:2], global_shape=(4,))
    tt_vals = TransportTensor(vals[:2], global_shape=(4,))

    dist1 = DataDistributor(0)  # local rank 0 (handles first 2 elements in chunk)

    received_payloads = []

    def mock_send(
        target: int,
        tuuids: list,
        tensors: list,
        counts: list,
        global_shapes_list: list | None = None,
        meta_tensor_state_list: list | None = None,
        compact_params_list: list | None = None,
    ):
        received_payloads.append((target, tensors, counts))
        return True

    _res, sort_idx = dist1.distribute(
        local_data=[tt_keys, tt_vals],
        source=spec_src,
        target=spec_dst,
        send_fn=mock_send,
        capacity=4,
    )

    # We expect 3 payload_tensors: [padded_keys, padded_vals, padded_routing]
    assert len(received_payloads) > 0
    _, tensors, counts = received_payloads[0]

    # Since dense and static, no lengths or extra routing. Just 2 data buffers.
    assert len(tensors) == 2, f"Expected 2 tensors (keys, vals) but got {len(tensors)}"

    buffer_keys = tensors[0][: counts[0]]
    buffer_vals = tensors[1][: counts[0]]

    assert bk.equal(buffer_keys, bk.tensor([10, 20], dtype=bk.int32()))
    assert bk.equal(buffer_vals, bk.tensor([0.1, 0.2], dtype=bk.float32()))


def test_reconstruct_splits_from_lengths():
    """
    Demonstrates how a receiver can reconstruct `nested_splits_flat` (and `segment_ids`)
    from the concisely transmitted `lengths` array for operations like `reduce_mean`.
    """
    bk = get_backend()

    # Assume the networking layer delivered these `lengths` to the receiver (instead of full nested_splits)
    # This corresponds to 3 ragged sequences with lengths: 2, 4, 1
    received_lengths = bk.tensor([2, 4, 1], dtype=bk.int32())

    # And the corresponding flat element buffer:
    received_data_flat = bk.tensor(
        [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0],  # seq 0  # seq 1  # seq 2
        dtype=bk.float32(),
    )

    # --- 1. Reconstruct nested_splits ---
    # nested_splits is simply the cumulative sum of lengths, prepended with 0.
    zero = bk.zeros((1,), dtype=received_lengths.dtype)
    lengths_with_zero = bk.cat([zero, received_lengths], dim=0)
    reconstructed_splits = bk.cumsum(lengths_with_zero, dim=0)

    # Verify the reconstructed splits
    expected_splits = bk.tensor([0, 2, 6, 7], dtype=bk.int32())
    assert bk.equal(reconstructed_splits, expected_splits), (
        "Failed to reconstruct splits!"
    )

    # --- 2. Reconstruct segment_ids (for jax.ops.segment_mean, etc.) ---
    # Often, frameworks prefer segment_ids (e.g., [0, 0, 1, 1, 1, 1, 2]) over splits.
    import numpy as np

    # Using numpy as stand-in for tracing logic (can be jnp.repeat or dynamic construction natively)
    lengths_np = bk.to_numpy(received_lengths)
    segment_ids = np.repeat(np.arange(len(lengths_np)), lengths_np)

    expected_segment_ids = np.array([0, 0, 1, 1, 1, 1, 2])
    assert np.array_equal(segment_ids, expected_segment_ids), (
        "Failed to reconstruct segment_ids!"
    )

    # --- 3. Example Use: jax.ops.segment_sum ---
    # Now the user can trivially do ops. We verify it with JAX.
    import jax
    import jax.numpy as jnp

    j_data = jnp.array(received_data_flat)
    j_ids = jnp.array(segment_ids)

    # 3 segments
    segment_sums = jax.ops.segment_sum(j_data, j_ids, num_segments=3)

    expected_sums = jnp.array(
        [
            1.0 + 1.0,  # seq 0: 2.0
            2.0 + 2.0 + 2.0 + 2.0,  # seq 1: 8.0
            3.0,  # seq 2: 3.0
        ],
        dtype=jnp.float32,
    )

    assert jnp.allclose(segment_sums, expected_sums), (
        f"Segment sum mismatch. Got {segment_sums}"
    )


def test_large_scale_sparse_moe():
    """Multi-bucket capacity distribute: all tokens must arrive across all ranks."""
    bk = get_backend()

    num_workers = 4
    mesh = DeviceMesh(("x",), (num_workers,), list(range(num_workers)))
    spec_src = ShardingSpec.chunk(P("x")).bind(mesh)
    spec_dst = ShardingSpec.hash(P("x")).bind(mesh)

    cluster = {i: VirtualRank(i, mesh) for i in range(num_workers)}

    import numpy as np

    np.random.seed(42)
    total_tokens = 100000
    emb_dim = 64

    tokens_np = np.random.randn(total_tokens, emb_dim).astype(np.float32)
    routing_np = np.random.randint(0, num_workers, size=total_tokens).astype(np.int32)
    tt = TransportTensor(bk.tensor(tokens_np), routing_indices=bk.tensor(routing_np))

    def send_fn(target, tuuids, tensors, counts, *args, **kwargs):
        if counts[0] > 0:
            cluster[target].receiver.receive_slices(0, tensors, counts)
        return True

    cluster[0].distributor.distribute(
        local_data=tt,
        source=spec_src,
        target=spec_dst,
        send_fn=send_fn,
        capacity=[10000, 20000, 30000, 40000],
    )

    total_received = sum(
        counts[0]
        for slices in (
            cluster[r].receiver.get_buffered_slices() for r in range(num_workers)
        )
        for _, counts in slices
        if counts[0] > 0
    )
    assert total_received == total_tokens, (
        f"Expected {total_tokens} tokens, got {total_received}"
    )
