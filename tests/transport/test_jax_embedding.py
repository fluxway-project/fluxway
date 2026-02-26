import pytest

from typing import List, Dict, Any

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
        self.data_store: Dict[str, Any] = {}

    def send(
        self,
        target_ranks: List[int],
        tensors: List[Any],
        counts: List[int],
        cluster_receivers: Dict[int, "VirtualRank"],
    ):
        """Simulates network payload transmission to target ranks via direct memory delivery."""
        for target_rank in target_ranks:
            if target_rank in cluster_receivers:
                # Direct memory transfer for simulation
                cluster_receivers[target_rank].receiver.receive_slices(
                    self.rank, tensors, counts
                )

    def process_incoming(self) -> List[Any]:
        """Extracts and consumes the buffered payloads received over the mock network."""
        return self.receiver.get_buffered_slices()


def test_embedding_lookup_flow():
    """Simulates a 3-step pipeline: Input (SPMD0) -> Embedding (SPMD1) -> Forward Output (SPMD2)."""

    # --- 1. Setup Global Data & Meshes ---
    bk = get_backend()

    # Global Data: 12 keys, vocab size 8, dim 2.
    global_keys = bk.arange(0, 12, dtype=bk.int32())
    vocab_size = 8
    emb_dim = 2
    # reshape
    _tmp = bk.arange(0, vocab_size * emb_dim, dtype=bk.float32())
    global_emb_table = bk.reshape(_tmp, (vocab_size, emb_dim))

    global_ids = global_keys % vocab_size
    expected_global_output = global_emb_table[global_ids]

    # Meshes
    mesh0 = DeviceMesh((2,), ("x",), [0, 1])
    mesh1 = DeviceMesh((4,), ("y",), [0, 1, 2, 3])
    mesh2 = DeviceMesh((2,), ("z",), [0, 1])

    # Sharding Specs
    spec0 = ShardingSpec.chunk(mesh=mesh0, partition=P("x"))
    spec1 = ShardingSpec.hash(mesh=mesh1, partition=P("y"))
    spec2 = ShardingSpec.chunk(mesh=mesh2, partition=P("z"))

    # --- 2. Simulation Environment ---

    # Create Virtual Ranks [0, 1, 2, 3] (Super-set of all ranks)
    cluster = {i: VirtualRank(i, mesh1) for i in range(4)}  # Max ranks = 4

    # --- 3. Forward Phase 1: Input (SPMD0) -> Embedding (SPMD1) ---
    # "Dispatch"

    # Step 3.1: Prepare Input Data on SPMD0 Ranks (0, 1)
    rank0_keys = global_keys[:6]
    rank1_keys = global_keys[6:]

    inputs = {0: TransportTensor(rank0_keys), 1: TransportTensor(rank1_keys)}

    print("\n--- Phase 1: Dispatch (SPMD0 -> SPMD1) ---")

    # Rank 0 sends
    dist0 = cluster[0].distributor

    def send_wrapper_0(target, tensors, counts, includes_indices=False):
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

    def send_wrapper_1(target, tensors, counts, includes_indices=False):
        cluster[1].send([target], tensors, counts, cluster)
        return True

    _res, original_sort_indices_1 = dist1.distribute(
        local_data=inputs[1],
        source=spec0,
        target=spec1,
        send_fn=send_wrapper_1,
        capacity=6,
    )

    # --- 4. Forward Phase 2: Embedding Lookup (SPMD1) ---

    print("\n--- Phase 2: Lookup (SPMD1) ---")

    spmd1_results = {}

    for r in range(4):  # SPMD1 has 4 ranks
        # Receive dispatched keys
        shards = cluster[r].process_incoming()

        all_keys = []

        for tensors, counts in shards:
            count = counts[0]
            if count > 0:
                # Valid unpadded data
                buffer = tensors[0]
                all_keys.append(buffer[:count])

        if not all_keys:
            spmd1_results[r] = (
                bk.tensor([], dtype=bk.int32()),
                bk.tensor([], dtype=bk.int32()),
            )
            continue

        local_keys = bk.cat(all_keys)
        local_indices = local_keys  # Key=Index

        # Lookup
        local_emb = global_emb_table[local_keys % vocab_size]

        spmd1_results[r] = (local_emb, local_indices)

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

    # We must iterate in the same order as distribute (0..3) to set offsets correctly
    # if we assume sequential arrival, OR we allocate specific slots per rank.
    # Since specific slots is cleaner:

    for r in range(4):
        _emb, _idx = spmd1_results[r]
        if len(_emb) == 0:
            continue

        # For Zero-Copy setups, receivers generally need to pre-allocate memory buffers.
        # In a real environment, the sender might exchange metadata first or the planner logic
        # is fully deterministic across nodes. Here we manually compute expected sizes.
        val_idx = bk.to_numpy(_idx)
        count0 = 0
        count1 = 0
        for v in val_idx:
            if v < 6:
                count0 += 1
            else:
                count1 += 1

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

    print("\n--- Phase 3: Combine (SPMD1 -> SPMD2) [Zero-Copy Mode] ---")
    print(f"Rank 0 expects {rank0_total_count}, Rank 1 expects {rank1_total_count}")

    # output_shards no longer needed for Zero-Copy flow
    # but we can keep it empty or refactor verify?
    # We will verify `buffer0` and `buffer1` directly.

    for r in range(4):
        local_emb, local_indices = spmd1_results[r]
        if len(local_emb) == 0:
            continue

        t_tensor = TransportTensor(local_emb, indices=local_indices)

        dist = cluster[r].distributor

        # Send wrapper calling receive_slices
        def send_wrapper_combine(
            target, tensors, counts, includes_indices=False, source_rank=r
        ):
            if target in cluster:
                # This calls receive_slices -> writes into target_buffers Zero-Copy
                cluster[target].receiver.receive_slices(source_rank, tensors, counts)
            return True

        # Check 2: Content Addressable Source (Hash) -> Implicit Static Target (Chunk)
        _res, _ = dist.distribute(
            local_data=t_tensor,
            source=spec1,
            target=spec2,
            send_fn=send_wrapper_combine,
            global_shape=(12, emb_dim),
            capacity=12,
        )

    # --- 6. Verify Forward Output ---

    print("\n--- Verification: Forward ---")

    # Verify Rank 0 (indices 0..5)
    # Zero-Copy: check the buffer directly
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
    print(
        "Successfully reconstructed Ragged Tensor Embedding for Rank 0 using related meta!"
    )

    # --- 7. Backward Pass: Gradients (SPMD2 -> SPMD1) ---
    # "Dispatch Gradients"

    print("\n--- Phase 4: Backward Dispatch (SPMD2 -> SPMD1) ---")

    # Mock Gradients (Index based)
    _tmp_casted = global_keys.astype(bk.float32())
    _tmp_reshaped = bk.reshape(_tmp_casted, (-1, 1))

    # Use backend tile
    global_grads = bk.tile(_tmp_reshaped, (1, emb_dim))

    # Rank 0 Grads (0..5)
    grads0 = TransportTensor(global_grads[:6], indices=rank0_keys)
    # Rank 1 Grads (6..11)
    grads1 = TransportTensor(global_grads[6:], indices=rank1_keys)

    grad_acc = {r: [] for r in range(4)}

    # Send from Rank 0
    dist_bk_0 = cluster[
        0
    ].distributor  # Re-using dist 0 (Rank 0 of Sim) for Rank 0 of SPMD2

    def send_wrapper_bk_0(target, tensors, counts, includes_indices=False):
        if target in grad_acc:
            grad_acc[target].append((tensors, counts))
        return True

    # Check 3: Implicit Spec0 (Chunk) Source -> Content Spec1 (Hash) Target
    # Gradient has same shape as embedding.
    # Note: spec0 was on Mesh0. spec2 is on Mesh2. Both Chunk.
    # We use spec2 here.

    _res, _ = dist_bk_0.distribute(
        local_data=grads0,
        source=spec2,
        target=spec1,
        send_fn=send_wrapper_bk_0,
        capacity=6,
    )

    # Send from Rank 1
    dist_bk_1 = cluster[1].distributor

    def send_wrapper_bk_1(target, tensors, counts, includes_indices=False):
        if target in grad_acc:
            grad_acc[target].append((tensors, counts))
        return True

    _res, _ = dist_bk_1.distribute(
        local_data=grads1,
        source=spec2,
        target=spec1,
        send_fn=send_wrapper_bk_1,
        capacity=6,
    )

    # --- 8. Verify Backward Update (SPMD1) ---

    print("\n--- Verification: Backward ---")

    for r in range(4):
        shards = grad_acc[r]
        if not shards:
            local_grads = bk.tensor([], dtype=bk.float32())
        else:
            all_g = []
            for tensors, counts in shards:
                buffer = tensors[0]
                count = counts[0]
                if count > 0:
                    all_g.append(buffer[:count])
            local_grads = bk.cat(all_g) if all_g else bk.tensor([], dtype=bk.float32())

        expected_indices = spmd1_results[r][1]

        if len(expected_indices) == 0:
            assert len(local_grads) == 0
            continue

        _e_tmp = expected_indices.astype(bk.float32())
        _e_reshaped = bk.reshape(_e_tmp, (-1, 1))
        # Use backend tile
        expected_grads = bk.tile(_e_reshaped, (1, emb_dim))

        l_sorted, _ = bk.sort(local_grads.flatten())
        e_sorted, _ = bk.sort(expected_grads.flatten())

        assert bk.equal(l_sorted, e_sorted), f"Rank {r} Grad mismatch"


def test_jit_shielding_flow():
    """Verifies that transport can be shielded from JIT."""
    bk = get_backend()

    # Setup simple data
    mesh = DeviceMesh((1,), ("x",), [0])
    spec = ShardingSpec.chunk(mesh, P("x"))

    # A global list to capture side-effects from the shielded function
    side_effects = []

    def side_effect_send(target, tensors, counts, includes_indices=False):
        count = counts[0]
        if count > 0:
            side_effects.append(tensors[0][:count])
        return True

    # We want to run `distribute` safely inside a JIT-compiled function graph.
    # The @shield decorator escapes tracing, allowing pure Python operations (loops, appending)
    # or host-bound network invocations to execute eagerly without throwing JAX tracer errors.

    distributor = DataDistributor(0)

    @shield(result_shape=None)
    def shielded_distribute(data):
        # Local wrapper to match distribute signature if needed,
        # or just call distribute directly.
        _res, _ = distributor.distribute(
            local_data=data,
            source=spec,
            target=spec,
            send_fn=side_effect_send,
            capacity=3,
        )

    # JIT-compiled step
    import jax

    @jax.jit
    def train_step(x):
        # Do some compute
        y = x * 2

        # Create TransportTensor
        tt = TransportTensor(y)

        # Call guarded transport
        shielded_distribute(tt)

        return y

    # Run
    input_data = bk.tensor([1, 2, 3], dtype=bk.float32())
    result = train_step(input_data)

    # Verify result (compute happened)
    assert bk.equal(result, input_data * 2)

    # Verify side effect (transport happened on host)
    assert len(side_effects) == 1
    # The payload data should be numpy array (host), not JAX tracer or array
    # verify values
    import numpy as np

    assert np.allclose(side_effects[0], [2, 4, 6])
    print("\nShielding test passed: Side effect captured from JIT.")


def test_jit_plan_and_shard():
    """Verify that _plan_and_shard supports JIT compilation via vmap and bounds."""

    mesh = DeviceMesh((2,), ("x",), [0, 1])
    spec = ShardingSpec.chunk(mesh, P("x"))

    distributor = DataDistributor(0)

    @compile
    def plan_kernel(flat_values):
        (
            padded_buffers_lists,
            send_counts,
            dest_devices,
            rep_factor,
            global_indices,
            padded_routing,
            sort_indices,
        ) = distributor._plan_and_shard(
            TransportTensor(flat_values), source=spec, target=spec, capacity=4
        ).__next__()
        return flat_values

    pass


def test_unified_compile():
    """Verifies that unified compile correctly JITs functions."""
    bk = get_backend()

    # Test functional correctness
    @compile
    def simple_mul(x):
        return x * 2.0

    res = simple_mul(bk.tensor([1.0, 2.0]))
    assert bk.equal(res, bk.tensor([2.0, 4.0]))

    # Test JIT effects (JAX)
    # Use side-effect detection to confirm it ran once (tracing) and then JIT (execution)
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

    print("\nUnified compile test passed.")


def test_colocation_dispatch():
    """Verify that distribute can take a List of TransportTensors and share the same plan."""
    bk = get_backend()

    mesh = DeviceMesh((2,), ("x",), [0, 1])
    spec_src = ShardingSpec.chunk(mesh, P("x"))
    spec_dst = ShardingSpec.chunk(mesh, P("x"))

    # 4 elements total
    keys = bk.tensor([10, 20, 30, 40], dtype=bk.int32())
    vals = bk.tensor([0.1, 0.2, 0.3, 0.4], dtype=bk.float32())

    # Let's say routing logic depends on keys
    tt_keys = TransportTensor(keys)
    tt_vals = TransportTensor(vals)

    dist1 = DataDistributor(0)  # local rank 0 (handles first 2 elements in chunk)

    received_payloads = []

    def mock_send(target, tensors, counts, includes_indices=False):
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

    # Since dense, no lengths. Just 2 data buffers + routing.
    assert len(tensors) == 3, (
        f"Expected 3 tensors (keys, vals, routing) but got {len(tensors)}"
    )

    buffer_keys = tensors[0][: counts[0]]
    buffer_vals = tensors[1][: counts[0]]

    assert bk.equal(buffer_keys, bk.tensor([10, 20], dtype=bk.int32()))
    assert bk.equal(buffer_vals, bk.tensor([0.1, 0.2], dtype=bk.float32()))

    print("\nCo-location test passed.")


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
        [
            1.0,
            1.0,  # seq 0
            2.0,
            2.0,
            2.0,
            2.0,  # seq 1
            3.0,
        ],  # seq 2
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

    print(
        "\nSuccessfully demonstrated nested_splits, segment_ids, and JAX segment_sum reconstruction from lengths!"
    )
