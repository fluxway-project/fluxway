import pytest
import time
from typing import List, Dict, Any

from fluxway.transport.tensor import TransportTensor
from fluxway.common.types.sharding import ShardingSpec, P
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.communication import DataDistributor, TransportReceiver
from fluxway.common.backend import set_backend, get_backend


@pytest.fixture(autouse=True)
def use_jax_backend():
    set_backend("jax")
    yield


class VirtualRank:
    def __init__(self, rank: int, mesh: DeviceMesh):
        self.rank = rank
        self.mesh = mesh
        self.distributor = DataDistributor(rank)
        self.receiver = TransportReceiver(wait_for_n_slices=0)

    def send(
        self,
        target_ranks: List[int],
        tensors: List[Any],
        counts: List[int],
        cluster_receivers: Dict[int, "VirtualRank"],
    ):
        for target_rank in target_ranks:
            if target_rank in cluster_receivers:
                cluster_receivers[target_rank].receiver.receive_slices(
                    self.rank, tensors, counts
                )


def test_large_scale_sparse_moe():
    """Test simulated large-scale sparse MoE using multi-bucket capacity and zero-copy."""
    bk = get_backend()

    # 4 Experts
    world_size = 4
    mesh = DeviceMesh((world_size,), ("x",), list(range(world_size)))
    spec = ShardingSpec.chunk(mesh, P("x"))

    cluster = {i: VirtualRank(i, mesh) for i in range(world_size)}

    # Large scale tokens (100k)
    import numpy as np

    np.random.seed(42)
    total_tokens = 100000
    emb_dim = 64

    # Random token embeddings
    tokens_np = np.random.randn(total_tokens, emb_dim).astype(np.float32)
    tokens = bk.tensor(tokens_np)

    # Random routing to 4 experts (sparse distribution) -> each expert ~ 25k
    routing_np = np.random.randint(0, world_size, size=total_tokens).astype(np.int32)
    routing = bk.tensor(routing_np)

    tt = TransportTensor(tokens, indices=routing)

    # Mock send wrapper
    def send_wrapper_moe_0(target, tensors, counts):
        count = counts[0]
        if count > 0:
            cluster[target].receiver.receive_slices(0, tensors, counts)
        return True

    capacity_buckets = [10000, 20000, 30000, 40000]

    # We set content addressable target to route based on `routing` array
    spec_target = ShardingSpec.hash(mesh, P("x"))

    t0 = time.time()

    # Using DataDistributor with capacity buckets
    _res, _ = cluster[0].distributor.distribute(
        local_data=tt,
        source=spec,
        target=spec_target,
        send_fn=send_wrapper_moe_0,
        capacity=capacity_buckets,
    )
    t1 = time.time()
    print(f"\nDistribute time: {(t1 - t0) * 1000:.2f} ms")

    # Check experts received their shards
    total_received = 0
    for r in range(world_size):
        slices = cluster[r].receiver.get_buffered_slices()
        for tensors, counts in slices:
            count = counts[0]
            if count > 0:
                total_received += count

    assert total_received == total_tokens, (
        f"Expected {total_tokens} tokens, got {total_received}."
    )
    print(f"\nMoE AllToAll routing test passed. Total tokens routed: {total_received}")
