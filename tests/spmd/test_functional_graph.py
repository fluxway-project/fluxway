import sys
import os
import copy

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
    """
    [Beginner's Guide: Why do we need to mock metadata?]

    In Fluxway's formal workflow, each operator needs to strictly define its input/output data types (Dataclass)
    and tensor sharding strategies. The Builder performs rigorous physical validation when connecting edges:
    1. Validate whether the output of A and the input of B are type-compatible.
    2. Validate whether data needs to be rearranged (Resharding) when transferred from A's memory distribution to B's.

    In "graph logic tests," we often only care about "whether Node A is connected to Node B," without worrying about specific physical implementations.

    The purpose of this function is:
    1. Dynamically inject (Patch) metadata so the Builder thinks these Dummy operators have valid inputs and outputs.
    2. Disable strict physical validation (e.g., validate_tensor_io_match), allowing us to quickly test various complex graph topologies (e.g., RL loops, multi-input merges) without defining complex data structures.
    3. Ensure each test node gets an independent metadata copy (Deepcopy) to prevent test interference.
    """
    from fluxway.common.types.configs import OperatorMeta

    # Shield physical validation logic, keep only topological logic
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
            # Inject virtual sharding info to trigger Builder's EdgeShardingInfo routing records
            input_sharding={"dummy_tensor": {"type": "replicated"}},
            output_sharding={"dummy_tensor": {"type": "replicated"}},
        )
    DummySPMD.get_operator_metas = classmethod(lambda cls: metas)

    # Intercept configuration acquisition to ensure Builder gets metas processed by Mock
    orig_get_config = DummySPMD.get_spmd_config

    def mock_get_config(cls):
        config = orig_get_config()
        if config is None:
            config = SPMDConfig()
        # Core: Must use deepcopy!
        # Otherwise, all nodes in the graph using DummySPMD will point to the same meta object,
        # causing edge info to overwrite each other.
        config.operator_metas = copy.deepcopy(metas)
        config.operators = {op: getattr(DummySPMD, op) for op in metas.keys()}
        return config

    DummySPMD.get_spmd_config = classmethod(mock_get_config)


# Execute Mock injection
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
