from typing import Any, Callable, Optional, Type
from ...common.types.configs import SPMDConfig
from .builder import Builder


class OpHandle:
    """
    Handle representing the output of an SPMD operator in the computational graph.

    Used to track dependencies between operators across (or within) SPMD nodes.
    An OpHandle is purely a placeholder for routing purposes during the graph
    compilation phase; it contains no actual tensor data.
    """

    def __init__(self, node_id: str, op_name: str, graph: "FunctionalGraph"):
        self.node_id = node_id
        self.op_name = op_name
        self._graph = graph

    def __repr__(self):
        return f"OpHandle({self.node_id}.{self.op_name})"


class NodeProxy:
    """
    Proxy object representing an SPMD node in the functional graph API.

    Any attribute access on this proxy returns a callable that, when invoked
    with OpHandles, registers a dependency edge in the underlying Graph Builder.
    """

    def __init__(self, node_id: str, graph: "FunctionalGraph"):
        self.node_id = node_id
        self._graph = graph

    def __getattr__(self, op_name: str) -> Callable[..., OpHandle]:
        """
        Intercepts method calls to dynamically construct graph edges.

        Example:
            actor = graph.node("Actor", ActorSPMD)
            h_rollout = actor.rollout()
            # actor.rollout -> returns _call_op closure
            # _call_op() -> logs op and returns new OpHandle
        """

        def _call_op(
            *args: OpHandle,
            condition_fn: Optional[Callable[..., bool]] = None,
            **kwargs,
        ) -> OpHandle:
            inputs = list(args) + list(kwargs.values())

            # Register edges for all inputs
            for in_handle in inputs:
                if not isinstance(in_handle, OpHandle):
                    raise TypeError(
                        f"Inputs must be OpHandle instances. Got {type(in_handle)} "
                        f"for arg to {self.node_id}.{op_name}"
                    )

                # Implicitly handles both inter-node edges and intra-node self-loops
                self._graph._builder.add_func_edge(
                    source_node=in_handle.node_id,
                    source_op=in_handle.op_name,
                    target_node=self.node_id,
                    target_op=op_name,
                    condition_fn=condition_fn,
                )

            return OpHandle(self.node_id, op_name, self._graph)

        return _call_op


class FunctionalGraph:
    """
    Keras-style functional API for constructing Fluxway's MPMD asynchronous execution graphs.

    This abstracts away the manual `builder.add_func_edge` calls, allowing users
    to define complex multi-stage asynchronous data flows simply by "calling"
    operators on node proxies and passing the resulting handles.

    Example:
        graph = FunctionalGraph()

        env = graph.node("Env", EnvSPMD, mesh=mesh0)
        actor = graph.node("Actor", ActorSPMD, mesh=mesh1)

        h_obs = env.step()
        h_action = actor.rollout(h_obs)

        h_reward = env.compute_reward(h_action)
        h_train = actor.train_step(h_reward)

        task_graph = graph.compile("RL_Task")
    """

    def __init__(self):
        self._builder = Builder()

    def node(
        self,
        name: str,
        spmd_class: Type,
        description: str = "",
        config: Optional[SPMDConfig] = None,
        *args,
        **kwargs,
    ) -> NodeProxy:
        """
        Registers an SPMD node and returns a proxy for functional graph construction.

        Args:
            name: Unique identifier for the node.
            spmd_class: The SPMDWorker subclass to instantiate.
            description: Human-readable description of the node.
            config: Optional SPMDConfig.
            *args, **kwargs: Additional arguments passed to builder.add_node.

        Returns:
            A NodeProxy object representing this SPMD.
        """
        self._builder.add_node(name, spmd_class, description=description)
        return NodeProxy(name, self)

    def reader(self, name: str, config: Any, *args, **kwargs) -> NodeProxy:
        """
        Registers a Reader node and returns a proxy.
        """
        self._builder.add_reader(name, config, *args, **kwargs)
        return NodeProxy(name, self)

    def compile(self, graph_name: str = "functional_graph") -> Any:
        """
        Compiles the defined functional flow into a concrete Fluxway Graph.

        Returns:
            The compiled Graph object ready for execution.
        """
        return self._builder.build(name=graph_name)
