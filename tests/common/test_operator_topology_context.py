from fluxway.common.decorators import (
    get_current_context,
    get_instance_storage,
    operator,
    operator_context,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import SPMDConfig
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.spmd.worker.worker import SPMDWorker
import pytest


class TopologyAwareOperator:
    seen_in_init = None

    def __init__(self):
        type(self).seen_in_init = (
            self.rank,
            self.world_size,
            getattr(self, "local_rank", None),
            getattr(self, "local_world_size", None),
            getattr(self, "node_rank", None),
            getattr(self, "num_nodes", None),
        )

    @operator(name="inspect")
    def inspect(self):
        return (
            self.rank,
            self.world_size,
            getattr(self, "local_rank", None),
            getattr(self, "local_world_size", None),
            getattr(self, "node_rank", None),
            getattr(self, "num_nodes", None),
        )


def _clear_operator_instance():
    storage, _ = get_instance_storage()
    storage.pop(id(TopologyAwareOperator), None)
    TopologyAwareOperator.seen_in_init = None


def test_operator_context_exposes_topology_fields():
    with operator_context(
        static_args={},
        rank=3,
        world_size=8,
        local_rank=1,
        local_world_size=4,
        node_rank=0,
        num_nodes=2,
    ):
        ctx = get_current_context()
        assert ctx.local_rank == 1
        assert ctx.local_world_size == 4
        assert ctx.node_rank == 0
        assert ctx.num_nodes == 2


def test_operator_wrapper_sets_topology_fields_on_instance():
    _clear_operator_instance()

    with operator_context(
        static_args={},
        rank=3,
        world_size=8,
        local_rank=1,
        local_world_size=4,
        node_rank=0,
        num_nodes=2,
    ):
        result = TopologyAwareOperator.inspect()

    assert result == (3, 8, 1, 4, 0, 2)


def test_preinstantiate_operator_sets_topology_fields_in_init():
    _clear_operator_instance()
    meta = TopologyAwareOperator.inspect._fluxway_operator_meta
    config = SPMDConfig(
        mesh=DeviceMesh(axis_names=("dp",), device_shape=(8,)),
        operators={"inspect": "inspect"},
        operator_metas={"inspect": meta},
    )

    SPMDWorker.__ray_metadata__.modified_class._pre_instantiate_operators(
        config, 3, local_rank=1, local_world_size=4, node_rank=0, num_nodes=2
    )

    storage, _ = get_instance_storage()
    instance = storage[id(TopologyAwareOperator)]

    assert TopologyAwareOperator.seen_in_init == (3, 8, 1, 4, 0, 2)

    # Outside context: fields should evaluate to default/None
    assert instance.local_rank is None
    assert instance.local_world_size is None
    assert instance.node_rank == 0
    assert instance.num_nodes == 1

    # Inside context: fields should evaluate to the current context values
    with operator_context(
        static_args={},
        rank=3,
        world_size=8,
        local_rank=1,
        local_world_size=4,
        node_rank=0,
        num_nodes=2,
    ):
        assert instance.local_rank == 1
        assert instance.local_world_size == 4
        assert instance.node_rank == 0
        assert instance.num_nodes == 2


@pytest.mark.asyncio
async def test_operator_manager_execute_preserves_topology_fields():
    _clear_operator_instance()
    meta = TopologyAwareOperator.inspect._fluxway_operator_meta
    config = SPMDConfig(
        mesh=DeviceMesh(axis_names=("dp",), device_shape=(8,)),
        operators={"inspect": "inspect"},
        operator_metas={"inspect": meta},
    )
    manager = OperatorManager(
        config,
        rank=3,
        spmd_id="spmd-a",
        local_rank=1,
        local_world_size=4,
        node_rank=0,
        num_nodes=2,
    )

    result = await manager.execute_async("inspect", step=0)

    assert result == (3, 8, 1, 4, 0, 2)


def test_operator_wrapper_resolves_local_classes():
    """
    Test that the _OperatorWrapper descriptor correctly resolves the owner class
    even if the class is defined inside a local scope (e.g. factory function),
    where importlib string-based reflection would normally fail.
    """

    def make_local_spmd():
        class LocalTopologyAwareOperator:
            @operator(name="inspect_local")
            def inspect_local(self):
                return getattr(self, "rank", -1)

        return LocalTopologyAwareOperator

    LocalClass = make_local_spmd()

    with operator_context(
        static_args={},
        rank=5,
        world_size=8,
    ):
        # Trigger __get__(None, LocalClass) on the descriptor.
        # This will resolve self._cls = LocalClass and successfully inject properties
        # without crashing due to '<locals>' in __qualname__.
        result = LocalClass.inspect_local()

    assert result == 5
