from typing import Any, Callable, Dict, Optional, Type
from fluxway.control_plane.graph.builder import Builder


class OpHandle:
    """
    Handle representing the output of an SPMD operator in the computational graph.

    Used to track dependencies between operators across (or within) SPMD nodes.
    An OpHandle is purely a placeholder for routing purposes during the graph
    compilation phase; it contains no actual tensor data.
    """

    def __init__(
        self,
        node_id: str,
        op_name: str,
        graph: "FunctionalGraph",
        tuuids: Dict[str, int],
        output_name: Optional[str] = None,
    ):
        self.node_id = node_id
        self.op_name = op_name
        self._graph = graph
        self.tuuids = tuuids
        self.output_name = output_name

    def __getattr__(self, name: str) -> "OpHandle":
        """Allows selecting a specific output field, e.g. h_proc.unique_ids1"""
        return OpHandle(
            self.node_id,
            self.op_name,
            self._graph,
            tuuids=self.tuuids,
            output_name=name,
        )

    def __repr__(self):
        res = f"OpHandle({self.node_id}.{self.op_name})"
        if self.output_name:
            res += f".{self.output_name}"
        return res


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
        """

        def _call_op(
            *args: OpHandle,
            condition_fn: Optional[Callable[..., bool]] = None,
            **kwargs,
        ) -> OpHandle:
            # Fetch target arguments to map positional args
            target_node = self._graph._builder._nodes.get(self.node_id)
            all_target_args = []
            if target_node:
                target_meta = target_node.spmd_config.operator_metas.get(op_name)
                if target_meta:
                    from fluxway.common.types.configs import ArgumentType

                    data_args = target_meta.get_arguments(
                        is_output=False, arg_type=ArgumentType.DATA
                    )
                    control_args = target_meta.get_arguments(
                        is_output=False, arg_type=ArgumentType.CONTROL
                    )
                    all_target_args = data_args + control_args

            # Register edges for positional arguments
            for i, in_handle in enumerate(args):
                if not isinstance(in_handle, OpHandle):
                    raise TypeError(
                        f"Inputs must be OpHandle instances. Got {type(in_handle)} "
                        f"for arg to {self.node_id}.{op_name}"
                    )

                # Fetch the source argument names from the source operator
                src_node = self._graph._builder._nodes.get(in_handle.node_id)
                src_output_names = []
                if src_node:
                    src_meta = src_node.spmd_config.operator_metas.get(
                        in_handle.op_name
                    )
                    if src_meta:
                        from fluxway.common.types.configs import ArgumentType

                        src_output_names = src_meta.get_arguments(
                            is_output=True, arg_type=ArgumentType.DATA
                        )

                # Check if the handle already has UUIDs.
                # In the new TUUID-driven architecture, this is mandatory.
                if not in_handle.tuuids:
                    raise RuntimeError(
                        f"OpHandle from {in_handle.node_id}.{in_handle.op_name} "
                        f"is missing mandatory UUIDs. This should not happen."
                    )

                # If specific output field is not selected, and source has exactly one tensor output, infer it.
                inferred_source_output = in_handle.output_name
                if inferred_source_output is None and len(src_output_names) == 1:
                    inferred_source_output = src_output_names[0]

                # Map positional target arguments, but if the user passed a single
                # handle for an operator with multiple inputs, treat it as a bundle (intersection).
                if (
                    len(args) == 1
                    and len(all_target_args) > 1
                    and inferred_source_output is None
                ):
                    target_input = None
                else:
                    target_input = (
                        all_target_args[i] if i < len(all_target_args) else None
                    )

                self._graph._builder.add_func_edge(
                    source_node=in_handle.node_id,
                    source_op=in_handle.op_name,
                    target_node=self.node_id,
                    target_op=op_name,
                    condition_fn=condition_fn,
                    source_output=inferred_source_output,
                    target_input=target_input,
                )

            # Register edges for keyword arguments (with explicit target mapping)
            for target_arg, in_handle in kwargs.items():
                if not isinstance(in_handle, OpHandle):
                    # Allow non-OpHandle kwargs (passed as static args maybe? No, builder doesn't support it here)
                    continue

                # Fetch the source argument names from the source operator
                src_node = self._graph._builder._nodes.get(in_handle.node_id)
                src_output_names = []
                if src_node:
                    src_meta = src_node.spmd_config.operator_metas.get(
                        in_handle.op_name
                    )
                    if src_meta:
                        from fluxway.common.types.configs import ArgumentType

                        src_output_names = src_meta.get_arguments(
                            is_output=True, arg_type=ArgumentType.DATA
                        )

                # If specific output field is not selected, and source has exactly one tensor output, infer it.
                inferred_source_output = in_handle.output_name
                if inferred_source_output is None and len(src_output_names) == 1:
                    inferred_source_output = src_output_names[0]

                self._graph._builder.add_func_edge(
                    source_node=in_handle.node_id,
                    source_op=in_handle.op_name,
                    target_node=self.node_id,
                    target_op=op_name,
                    condition_fn=condition_fn,
                    source_output=inferred_source_output,
                    target_input=target_arg,
                )

            # 3. Handle Outputs: Retrieve pre-assigned UUIDs from the Builder
            # These were assigned during add_node() and are the canonical IDs.
            output_uuids = {}
            if target_node:
                target_meta = target_node.spmd_config.operator_metas.get(op_name)
                if target_meta:
                    from fluxway.common.types.configs import ArgumentType

                    out_args = target_meta.get_arguments(
                        is_output=True, arg_type=ArgumentType.DATA
                    )
                    for out_name in out_args:
                        u_key = (self.node_id, op_name, "out", out_name)
                        if u_key not in self._graph._builder._tuuid_slots:
                            # Safety check: ensure builder assigned it
                            raise RuntimeError(
                                f"TUUID for {u_key} not found in Builder registry."
                            )
                        output_uuids[out_name] = self._graph._builder._tuuid_slots[
                            u_key
                        ]

            return OpHandle(self.node_id, op_name, self._graph, tuuids=output_uuids)

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
        self, name: str, spmd_class: Type, description: str = "", *args, **kwargs
    ) -> NodeProxy:
        """
        Registers an SPMD node and returns a proxy for functional graph construction.

        Args:
            name: Unique identifier for the node.
            spmd_class: The SPMDWorker subclass to instantiate.
            description: Human-readable description of the node.
            *args, **kwargs: Additional arguments passed to builder.add_node (e.g., mesh).

        Returns:
            A NodeProxy object representing this SPMD.
        """
        self._builder.add_node(name, spmd_class, description, *args, **kwargs)
        return NodeProxy(name, self)

    def reader(
        self,
        name: str,
        reader_class: Type,
        mesh_axis_names: Optional[tuple] = None,
        mesh_shape: Optional[tuple] = None,
        num_workers: int = 1,
        resource: Optional[Dict[str, Any]] = None,
        description: str = "",
        **reader_kwargs: Any,
    ) -> NodeProxy:
        """
        Add a Reader node and return its proxy.

        Args:
            name: Unique identifier for the reader node.
            reader_class: The SPMD reader class (must subclass GrainReader with @spmd).
            mesh_axis_names: Axis names for the execution mesh.
            mesh_shape: Shape of the execution mesh.
            num_workers: Number of workers when no explicit mesh is given.
            resource: Optional resource requirements dict.
            description: Human-readable description.
            **reader_kwargs: Additional static args (data_source, shuffle, etc.)
                             forwarded to the reader via static_args at runtime.
        """
        self._builder.add_reader(
            name=name,
            reader_class=reader_class,
            mesh_axis_names=mesh_axis_names,
            mesh_shape=mesh_shape,
            num_workers=num_workers,
            resource=resource,
            description=description,
            **reader_kwargs,
        )
        return NodeProxy(name, self)

    def compile(self, graph_name: str = "functional_graph") -> Any:
        """
        Compiles the defined functional flow into a concrete Fluxway Graph.

        Returns:
            The compiled Graph object ready for execution.
        """
        return self._builder.build(name=graph_name)
