import sys
import os

sys.path.append(os.getcwd())

from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.common.types.configs import SPMDConfig
from fluxway.common.decorators import spmd, operator


@spmd()
class DummySPMD:
    """Mock SPMD for testing graph topology."""

    @operator
    def compute(self, *args, **kwargs):
        pass

    @operator
    def optimize(self, *args, **kwargs):
        pass

    @operator
    def step(self, *args, **kwargs):
        pass

    @operator
    def rollout(self, *args, **kwargs):
        pass

    @operator
    def compute_reward(self, *args, **kwargs):
        pass

    @operator
    def train_step(self, *args, **kwargs):
        pass


def _mock_metas():
    # Patch DummySPMD.get_operator_metas to return something dummy so Builder passes validation
    from fluxway.common.types.configs import OperatorMeta

    # Mock validation method
    OperatorMeta.validate_tensor_io_match = lambda self, dst, src_is_output, ctx: None

    metas = {}
    for op in [
        "compute",
        "optimize",
        "step",
        "rollout",
        "compute_reward",
        "train_step",
    ]:
        metas[op] = OperatorMeta(
            name=op,
            fn_name=op,
            data_inputs=[],
            control_inputs=[],
            sync_inputs=[],
            data_outputs=[],
            sync_outputs=[],
            input_arguments={},
            output_arguments={},
            input_sharding={},
            output_sharding={},
        )
    DummySPMD.get_operator_metas = classmethod(lambda cls: metas)

    # Builder extracts metas from spmd_config
    orig_get_config = DummySPMD.get_spmd_config

    def mock_get_config(cls):
        config = orig_get_config()
        if config is None:
            config = SPMDConfig()
        config.operator_metas = metas
        config.operators = {op: getattr(DummySPMD, op) for op in metas.keys()}
        return config

    DummySPMD.get_spmd_config = classmethod(mock_get_config)


_mock_metas()


def test_functional_graph_single_chain():
    """Verify simple A -> B data flow."""
    graph = FunctionalGraph()

    node_a = graph.node("NodeA", DummySPMD)
    node_b = graph.node("NodeB", DummySPMD)

    # NodeA.compute -> NodeB.compute
    h_a = node_a.compute()
    _ = node_b.compute(h_a)

    task_graph = graph.compile()

    assert len(task_graph.nodes) == 2
    assert "NodeA" in task_graph.nodes
    assert "NodeB" in task_graph.nodes

    # Check edges
    assert len(task_graph.edges) == 1
    edge = task_graph.edges[0]
    assert edge.source_node == "NodeA"
    assert edge.source_op == "compute"
    assert edge.target_node == "NodeB"
    assert edge.target_op == "compute"


def test_functional_graph_multi_input():
    """Verify multiple inputs (e.g. two embeddings into an FFN)."""
    graph = FunctionalGraph()

    emb1 = graph.node("Emb1", DummySPMD)
    emb2 = graph.node("Emb2", DummySPMD)
    ffn = graph.node("FFN", DummySPMD)

    h1 = emb1.compute()
    h2 = emb2.compute()

    # FFN takes both
    _ = ffn.compute(input1=h1, input2=h2)

    task_graph = graph.compile()

    assert len(task_graph.edges) == 2

    sources = {(e.source_node, e.source_op) for e in task_graph.edges}
    assert sources == {("Emb1", "compute"), ("Emb2", "compute")}

    targets = {(e.target_node, e.target_op) for e in task_graph.edges}
    assert targets == {("FFN", "compute")}


def test_functional_graph_implicit_self_loop():
    """Verify self-loops are naturally recorded for backward pass activation saving."""
    graph = FunctionalGraph()

    actor = graph.node("Actor", DummySPMD)

    h_fwd = actor.compute()
    _ = actor.optimize(h_fwd)

    task_graph = graph.compile()
    assert len(task_graph.edges) == 1

    edge = task_graph.edges[0]
    assert edge.source_node == "Actor"
    assert edge.source_op == "compute"
    assert edge.target_node == "Actor"
    assert edge.target_op == "optimize"


def test_functional_graph_agentic_rl():
    """Verify complex asynchronous Agentic RL topology."""
    graph = FunctionalGraph()

    env = graph.node("Env", DummySPMD)
    actor = graph.node("Actor", DummySPMD)
    _ = graph.node("Reward", DummySPMD)

    h_obs = env.step()
    h_action = actor.rollout(h_obs)
    h_reward = env.compute_reward(h_action)
    _ = actor.train_step(h_reward)

    task_graph = graph.compile()

    # Should have 3 edges:
    # 1. Env.step -> Actor.rollout
    # 2. Actor.rollout -> Env.compute_reward
    # 3. Env.compute_reward -> Actor.train_step
    assert len(task_graph.edges) == 3

    edge1 = task_graph.edges[0]
    assert edge1.source_node == "Env" and edge1.source_op == "step"
    assert edge1.target_node == "Actor" and edge1.target_op == "rollout"

    edge2 = task_graph.edges[1]
    assert edge2.source_node == "Actor" and edge2.source_op == "rollout"
    assert edge2.target_node == "Env" and edge2.target_op == "compute_reward"

    edge3 = task_graph.edges[2]
    assert edge3.source_node == "Env" and edge3.source_op == "compute_reward"
    assert edge3.target_node == "Actor" and edge3.target_op == "train_step"
