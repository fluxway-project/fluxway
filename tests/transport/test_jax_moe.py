import pytest
from typing import List, Dict, Any

from fluxway.common.types.tensor import TransportTensor
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

    num_workers = 4
    mesh = DeviceMesh(("x",), (num_workers,), list(range(num_workers)))
    spec = ShardingSpec.chunk(P("x")).bind(mesh)

    cluster = {i: VirtualRank(i, mesh) for i in range(num_workers)}

    import numpy as np

    np.random.seed(42)
    total_tokens = 100000
    emb_dim = 64

    tokens_np = np.random.randn(total_tokens, emb_dim).astype(np.float32)
    tokens = bk.tensor(tokens_np)

    routing_np = np.random.randint(0, num_workers, size=total_tokens).astype(np.int32)
    routing = bk.tensor(routing_np)

    tt = TransportTensor(tokens, routing_indices=routing)

    def send_wrapper_moe_0(
        target,
        tuuids,
        tensors,
        counts,
        global_shapes=None,
        meta_state_list=None,
        compact_params=None,
    ):
        count = counts[0]
        if count > 0:
            cluster[target].receiver.receive_slices(0, tensors, counts)
        return True

    capacity_buckets = [10000, 20000, 30000, 40000]
    spec_target = ShardingSpec.hash(P("x")).bind(mesh)

    _res, _ = cluster[0].distributor.distribute(
        local_data=tt,
        source=spec,
        target=spec_target,
        send_fn=send_wrapper_moe_0,
        capacity=capacity_buckets,
    )

    total_received = 0
    for r in range(num_workers):
        slices = cluster[r].receiver.get_buffered_slices()
        for tensors, counts in slices:
            count = counts[0]
            if count > 0:
                total_received += count

    assert total_received == total_tokens, (
        f"Expected {total_tokens} tokens, got {total_received}."
    )
