from typing import Any
import sys
import os

sys.path.append(os.getcwd())

from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.common.decorators import spmd, operator, data_type


@data_type()
class DummyData:
    h_obs: Any | None = None
    h_action: Any | None = None
    h_reward: Any | None = None
    h_train: Any | None = None
    dummy_tensor: Any | None = None
    input1: Any | None = None
    input2: Any | None = None
    h_fwd: Any | None = None
    dummy: Any | None = None


# Universal sharding config for replicated data
REP = {"type": "replicated", "axis": []}


@spmd(mesh_axis_names=("data",), mesh_axis_shapes=(1,))
class DummySPMD:
    """Mock SPMD for testing graph topology."""

    @operator(
        output_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
        input_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
    )
    def compute(self, data: DummyData) -> DummyData:
        return data

    @operator(
        output_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
        input_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
    )
    def optimize(self, data: DummyData) -> DummyData:
        return data

    @operator(
        output_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
        input_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
    )
    def step(self, data: DummyData) -> DummyData:
        return data

    @operator(
        output_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
        input_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
    )
    def rollout(self, data: DummyData) -> DummyData:
        return data

    @operator(
        output_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
        input_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
    )
    def compute_reward(self, data: DummyData) -> DummyData:
        return data

    @operator(
        output_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
        input_sharding={
            "h_obs": REP,
            "h_action": REP,
            "h_reward": REP,
            "h_train": REP,
            "dummy_tensor": REP,
            "input1": REP,
            "input2": REP,
            "h_fwd": REP,
            "dummy": REP,
        },
    )
    def train_step(self, data: DummyData) -> DummyData:
        return data


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
    """Verify self-loops are naturally recorded."""
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

    h_obs = env.step()
    h_action = actor.rollout(h_obs)
    h_reward = env.compute_reward(h_action)
    _ = actor.train_step(h_reward)

    task_graph = graph.compile()

    # Edges:
    # 1. Env.step -> Actor.rollout
    # 2. Actor.rollout -> Env.compute_reward
    # 3. Env.compute_reward -> Actor.train_step
    assert len(task_graph.edges) == 3


def test_functional_graph_arbitrary_rl_loop():
    """
    Verify arbitrary reinforcement learning closed-loop: Actor -> Reward -> Learner -> Actor.
    This loop is a typical scenario in offline RL or continuous learning.
    """
    graph = FunctionalGraph()

    actor = graph.node("Actor", DummySPMD)
    reward = graph.node("Reward", DummySPMD)
    learner = graph.node("Learner", DummySPMD)

    # 1. Actor generates behavior
    h_obs = actor.rollout()
    # 2. Reward calculates score
    h_reward = reward.compute_reward(h_obs)
    # 3. Learner updates model and generates new strategy/parameters
    h_train = learner.train_step(h_reward)
    # 4. Feedback to Actor (closed loop)
    _ = actor.rollout(h_train)

    task_graph = graph.compile()

    assert len(task_graph.nodes) == 3
    assert len(task_graph.edges) == 3

    # Use Builder for secondary verification to ensure metadata can be correctly
    # injected into OperatorMeta after compilation
    from fluxway.control_plane.graph.builder import Builder

    builder = Builder()
    for name in task_graph.nodes:
        builder.add_node(name, DummySPMD)

    for edge in task_graph.edges:
        builder.add_func_edge(
            edge.source_node, edge.source_op, edge.target_node, edge.target_op
        )

    linked_graph = builder.build("linked_graph")

    # Verify that the critical "feedback" edge is effective in operator metadata
    actor_meta = linked_graph.nodes["Actor"].spmd_config.operator_metas["rollout"]

    # The operator's predecessor (Input) should include Learner, and the successor (Output) should include Reward
    assert "Learner" in actor_meta.input_pre_sharding
    assert "Reward" in actor_meta.output_post_sharding
