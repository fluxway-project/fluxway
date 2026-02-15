import pytest

from typing import List, Dict, Any

from fluxway.transport.tensor import TransportTensor
from fluxway.transport.sharding import ShardingSpec, P
from fluxway.transport.mesh import DeviceMesh
from fluxway.transport.communication import DataDistributor, DataReceiverMixin
from fluxway.transport.backend import set_backend, get_backend


# Force JAX backend
@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


class MockReceiver(DataReceiverMixin):
    def __init__(self):
        super().__init__()


class VirtualRank:
    """Simulates a single rank in a distributed system."""

    def __init__(self, rank: int, mesh: DeviceMesh):
        self.rank = rank
        self.mesh = mesh
        self.distributor = DataDistributor(rank)
        self.receiver = MockReceiver()
        self.data_store: Dict[str, Any] = {}

    def send(
        self,
        target_ranks: List[int],
        payload: Any,
        cluster_receivers: Dict[int, "VirtualRank"],
    ):
        """Simulates network send to other ranks."""
        for target_rank in target_ranks:
            if target_rank in cluster_receivers:
                # Direct memory transfer for simulation
                cluster_receivers[target_rank].receiver.receive_shard(
                    self.rank, payload
                )

    def process_incoming(self) -> List[Any]:
        """Process received payloads."""
        return self.receiver.get_received_shards()


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

    def send_wrapper_0(target, payload):
        cluster[0].send([target], payload, cluster)
        return True

    # Check 1: Implicit Static Source + Content Addressable Target
    dist0.distribute(
        local_data=inputs[0], source=spec0, target=spec1, send_fn=send_wrapper_0
    )

    # Rank 1 sends
    dist1 = cluster[1].distributor

    def send_wrapper_1(target, payload):
        cluster[1].send([target], payload, cluster)
        return True

    dist1.distribute(
        local_data=inputs[1], source=spec0, target=spec1, send_fn=send_wrapper_1
    )

    # --- 4. Forward Phase 2: Embedding Lookup (SPMD1) ---

    print("\n--- Phase 2: Lookup (SPMD1) ---")

    spmd1_results = {}

    for r in range(4):  # SPMD1 has 4 ranks
        # Receive dispatched keys
        shards = cluster[r].process_incoming()

        all_keys = []

        for shard in shards:
            all_keys.append(shard.data)

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

    print("\n--- Phase 3: Combine (SPMD1 -> SPMD2) ---")

    output_shards = {0: [], 1: []}

    for r in range(4):
        local_emb, local_indices = spmd1_results[r]
        if len(local_emb) == 0:
            continue

        t_tensor = TransportTensor(local_emb, global_indices=local_indices)

        dist = cluster[r].distributor

        # Send wrapper collecting into output_shards
        def send_wrapper_combine(target, payload):
            if target in output_shards:
                output_shards[target].append(payload)
            return True

        # Check 2: Content Addressable Source (Hash) -> Implicit Static Target (Chunk)
        dist.distribute(
            local_data=t_tensor,
            source=spec1,
            target=spec2,
            send_fn=send_wrapper_combine,
            global_shape=(12, emb_dim),
        )

    # --- 6. Verify Forward Output ---

    print("\n--- Verification: Forward ---")

    # Verify Rank 0 (indices 0..5)
    rank0_vals = [p.data for p in output_shards[0]]
    if rank0_vals:
        rank0_data = bk.cat(rank0_vals)
    else:
        rank0_data = bk.tensor([], dtype=bk.float32())

    expected_r0 = expected_global_output[:6]
    # Check shape
    assert rank0_data.shape[0] == 6, f"Rank 0 shape mismatch: {rank0_data.shape} vs 6"

    # Verify Values
    r0_sorted, _ = bk.sort(rank0_data.flatten())
    exp0_sorted, _ = bk.sort(expected_r0.flatten())
    assert bk.equal(r0_sorted, exp0_sorted), "Rank 0 Value Mismatch"

    # Verify Rank 1 (indices 6..11)
    rank1_vals = [p.data for p in output_shards[1]]
    if rank1_vals:
        rank1_data = bk.cat(rank1_vals)
    else:
        rank1_data = bk.tensor([], dtype=bk.float32())

    assert rank1_data.shape[0] == 6, f"Rank 1 shape mismatch: {rank1_data.shape} vs 6"
    r1_sorted, _ = bk.sort(rank1_data.flatten())
    exp1_sorted, _ = bk.sort(expected_global_output[6:].flatten())
    assert bk.equal(r1_sorted, exp1_sorted), "Rank 1 Value Mismatch"

    # --- 7. Backward Pass: Gradients (SPMD2 -> SPMD1) ---
    # "Dispatch Gradients"

    print("\n--- Phase 4: Backward Dispatch (SPMD2 -> SPMD1) ---")

    # Mock Gradients (Index based)
    _tmp_casted = global_keys.astype(bk.float32())
    _tmp_reshaped = bk.reshape(_tmp_casted, (-1, 1))

    # Use backend tile
    global_grads = bk.tile(_tmp_reshaped, (1, emb_dim))

    # Rank 0 Grads (0..5)
    grads0 = TransportTensor(global_grads[:6], global_indices=rank0_keys)
    # Rank 1 Grads (6..11)
    grads1 = TransportTensor(global_grads[6:], global_indices=rank1_keys)

    grad_acc = {r: [] for r in range(4)}

    # Send from Rank 0
    dist_bk_0 = cluster[
        0
    ].distributor  # Re-using dist 0 (Rank 0 of Sim) for Rank 0 of SPMD2

    def send_wrapper_bk_0(target, payload):
        if target in grad_acc:
            grad_acc[target].append(payload)
        return True

    # Check 3: Implicit Spec0 (Chunk) Source -> Content Spec1 (Hash) Target
    # Gradient has same shape as embedding.
    # Note: spec0 was on Mesh0. spec2 is on Mesh2. Both Chunk.
    # We use spec2 here.

    dist_bk_0.distribute(
        local_data=grads0, source=spec2, target=spec1, send_fn=send_wrapper_bk_0
    )

    # Send from Rank 1
    dist_bk_1 = cluster[1].distributor

    def send_wrapper_bk_1(target, payload):
        if target in grad_acc:
            grad_acc[target].append(payload)
        return True

    dist_bk_1.distribute(
        local_data=grads1, source=spec2, target=spec1, send_fn=send_wrapper_bk_1
    )

    # --- 8. Verify Backward Update (SPMD1) ---

    print("\n--- Verification: Backward ---")

    for r in range(4):
        shards = grad_acc[r]
        if not shards:
            local_grads = bk.tensor([], dtype=bk.float32())
        else:
            local_grads = bk.cat([p.data for p in shards])

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
