"""
Graph Builder for constructing task graphs programmatically.

Based on graph_example.py API design.

Usage:
    builder = Builder()
    builder.add_node("SPMD1", SPMD1, "description1")
    builder.add_node("SPMD2", SPMD2, "description2")
    builder.add_func_edge("SPMD1", "forward", "SPMD2", "forward")
    builder.add_func_edge("SPMD2", "backward", "SPMD1", "backward")
    graph = builder.build(name="my_graph")
"""

from typing import Any, Dict, List, Optional, Callable, Type, Tuple
import logging

from ...common.types import (
    Graph,
    Node,
    Edge,
    OperatorMeta,
    EdgeShardingInfo,
    ResourceSpec,
)


logger = logging.getLogger(__name__)


class Builder:
    """
    Builder class for constructing task graphs programmatically.

    Provides a fluent interface for adding nodes and edges to build
    a complete task graph for SPMD orchestration.
    """

    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._spmd_classes: Dict[str, Type] = {}

    def add_node(self, name: str, spmd_class: Type, description: str = "") -> "Builder":
        """
        Add a node to the graph.

        Args:
            name: Unique identifier for the node
            spmd_class: The SPMD class decorated with @fluxway.spmd
            description: Human-readable description of the node

        Returns:
            Self for method chaining

        Raises:
            ValueError: If node with same name already exists
            ValueError: If spmd_class is not decorated with @fluxway.spmd
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists in the graph")

        # Validate that spmd_class is decorated with @fluxway.spmd
        if not getattr(spmd_class, "is_fluxway_spmd", lambda: False)():
            raise ValueError(
                f"Class '{spmd_class.__name__}' is not a valid Fluxway SPMD. "
                f"Did you forget to add the @spmd decorator?"
            )

        # Get resource and config from the decorated class
        resource = spmd_class.get_resource()
        spmd_config = spmd_class.get_spmd_config()

        node = Node(
            name=name,
            description=description,
            resource=resource,
            spmd_config=spmd_config,
        )

        self._nodes[name] = node
        self._spmd_classes[name] = spmd_class

        return self

    # -------------------------------------------------------------------------
    # Reader Node API
    # -------------------------------------------------------------------------

    def add_reader(
        self,
        name: str,
        data_source: Any = None,
        reader_class: Optional[Type] = None,
        num_workers: int = 1,
        mesh_shape: Optional[Tuple[int, ...]] = None,
        mesh_axis_names: Optional[Tuple[str, ...]] = None,
        resource: Optional[Dict[str, Any]] = None,
        description: str = "",
        **reader_kwargs: Any,
    ) -> "Builder":
        """
        Add a Reader node to the graph with a generic data source.

        This is the modern, source-driven API for adding data input nodes.
        It defaults to GrainReader which uses Google Grain for distributed loading.

        Args:
            name: Unique name for the reader node
            data_source: The underlying data source object (e.g., Grain DataSource)
            reader_class: Custom reader class (defaults to GrainReader)
            num_workers: Number of parallel workers if no mesh is provided
            mesh_shape: Optional multi-dimensional mesh shape
            mesh_axis_names: Names for mesh axes
            resource: Resource requirements (num_cpus, num_gpus, etc.)
            description: Optional description
            **reader_kwargs: Additional configuration (batch_size, shuffle, etc.)
                             that will be passed to the reader as static_args.
        """
        from ...reader.grain_reader import GrainReader

        reader_cls = reader_class or GrainReader

        # Merge data_source into static_args for the reader instance
        all_static_args = {**reader_kwargs}
        if data_source is not None:
            all_static_args["data_source"] = data_source

        # Handle resource specification
        resource_spec = None
        if resource:
            resource_spec = ResourceSpec(
                num_cpus=resource.get("num_cpus", 1),
                num_gpus=resource.get("num_gpus", 0),
                memory_gb=resource.get("memory_gb", 0.0),
            )

        # Add as a generic SPMD node
        self.add_node(name, reader_cls, description)

        # Inject configuration
        node = self._nodes[name]
        node.static_args.update(all_static_args)
        if resource_spec:
            node.resource = resource_spec

        # Establish parallel execution mesh
        from ...common.mesh import DeviceMesh

        if mesh_shape and mesh_axis_names:
            node.spmd_config.mesh = DeviceMesh(
                device_shape=mesh_shape, axis_names=mesh_axis_names
            )
        else:
            # Default to flat 1D mesh for simple data parallelism
            node.spmd_config.mesh = DeviceMesh(
                device_shape=(num_workers,), axis_names=("worker",)
            )

        return self

    # -------------------------------------------------------------------------
    # Generic func edge (core API)
    # -------------------------------------------------------------------------

    def add_func_edge(
        self,
        source_node: str,
        source_op: str,
        target_node: str,
        target_op: str,
        condition_fn: Optional[Callable[..., bool]] = None,
        source_output: Optional[str] = None,
        target_input: Optional[str] = None,
    ) -> "Builder":
        """
        Add a generic edge specifying source and target op names.

        This is the core edge API. All other edge methods are syntactic sugar
        built on top of this method.

        Args:
            source_node: Source node name
            source_op: Source node's op name (e.g., "forward", "rollout")
            target_node: Target node name
            target_op: Target node's op name (e.g., "forward", "reward")
            condition_fn: Optional condition function
            source_output: Optional specific output field name from source op
            target_input: Optional specific input field name for target op

        Returns:
            Self for method chaining

        Examples:
            # forward -> forward
            builder.add_func_edge("A", "forward", "B", "forward")

            # forward -> backward self-loop (sink node triggers its own backward)
            builder.add_func_edge("Sink", "forward", "Sink", "backward")

            # rollout -> reward (arbitrary user-defined ops)
            builder.add_func_edge("Actor", "rollout", "RewardModel", "compute_reward")
        """
        if source_node not in self._nodes:
            raise ValueError(f"Source node '{source_node}' not found")
        if target_node not in self._nodes:
            raise ValueError(f"Target node '{target_node}' not found")

        edge = Edge(
            source_node=source_node,
            target_node=target_node,
            source_op=source_op,
            target_op=target_op,
            condition_fn=condition_fn,
        )

        # All edges go into _edges (no semantic distinction anymore)
        self._edges.append(edge)

        # Validate and link sharding between source and target
        self._validate_and_link_edge(
            source_node,
            source_op,
            target_node,
            target_op,
            condition_fn,
            source_output,
            target_input,
        )

        return self

    # -------------------------------------------------------------------------
    # Internal helpers for operator metadata & sharding linking
    # -------------------------------------------------------------------------

    def _ensure_meta_exists(
        self, node_name: str, method_name: str, meta: Optional[OperatorMeta], ctx: str
    ) -> OperatorMeta:
        """Ensure that an operator meta exists for the given node/method; otherwise raise."""
        if meta is None:
            spmd_class = self._spmd_classes[node_name]
            class_name = spmd_class.__name__

            op_name_by_fn = {}
            for attr_name in dir(spmd_class):
                # Use getattr on the class to avoid triggering property descriptors if any
                attr = getattr(spmd_class, attr_name, None)
                m = getattr(attr, "_flux_operator_meta", None)
                if m is not None:
                    op_name_by_fn[attr_name] = m.name

            if method_name in op_name_by_fn:
                actual_op_name = op_name_by_fn[method_name]
                raise ValueError(
                    f"\n[Fluxway Graph Error] In {ctx}:\n"
                    f"  You referred to operator '{method_name}', which is the name of a function in '{class_name}'.\n"
                    f"  However, this function is decorated as @operator(name='{actual_op_name}').\n"
                    f"  Fix: Change your code to use '{actual_op_name}' instead of '{method_name}' in the builder (add_func_edge)."
                )

            if getattr(spmd_class, method_name, None) is not None:
                raise ValueError(
                    f"\n[Fluxway Graph Error] In {ctx}:\n"
                    f"  Method '{method_name}' exists in class '{class_name}' but it is NOT decorated with @operator.\n"
                    f"  Every method used in the graph must have the @operator decorator to define its sharding and IO.\n"
                    f"  Fix: Add @operator to '{class_name}.{method_name}':\n\n"
                    f"       @operator(name='{method_name}', ...)\n"
                    f"       def {method_name}(self, ...):"
                )

            available_ops = list(
                self._nodes[node_name].spmd_config.operator_metas.keys()
            )
            raise ValueError(
                f"\n[Fluxway Graph Error] In {ctx}:\n"
                f"  SPMD '{node_name}' (class {class_name}) has no operator named '{method_name}'.\n"
                f"  Available operators for this node: {available_ops}\n"
                f"  Fix: Check for typos in your add_func_edge call, or ensure the method exists and is decorated with @operator."
            )

        return meta

    def _validate_and_link_edge(
        self,
        source_node: str,
        source_op: str,
        target_node: str,
        target_op: str,
        condition_fn: Optional[Callable[..., bool]],
        source_output: Optional[str] = None,
        target_input: Optional[str] = None,
    ) -> None:
        """
        Validate and link operator metadata between two nodes.

        Writes back pre/post sharding info into OperatorMeta so that
        worker processes can read routing info from SPMDConfig.
        """
        src_node = self._nodes[source_node]
        dst_node = self._nodes[target_node]

        src_cfg = src_node.spmd_config
        dst_cfg = dst_node.spmd_config

        # 1. Update Topology in SPMDConfig
        from ...common.types.configs import SPMDTopology, SPMDTopologyInfo

        # Ensure topology entry exists for source op
        if source_op not in src_cfg.topology:
            src_cfg.topology[source_op] = SPMDTopology()
        # Ensure topology entry exists for target op
        if target_op not in dst_cfg.topology:
            dst_cfg.topology[target_op] = SPMDTopology()

        # Build tensor map if specified
        tensor_map = {}
        if source_output and target_input:
            tensor_map[source_output] = target_input
        elif source_output:
            # If only source_output is specified, we assume it maps to its own name in target
            tensor_map[source_output] = source_output
        elif target_input:
            # If only target_input is specified, we assume source provides it with the same name
            tensor_map[target_input] = target_input

        # Add successor info to source node
        if target_node not in src_cfg.topology[source_op].successors:
            src_cfg.topology[source_op].successors[target_node] = SPMDTopologyInfo(
                spmd_id=target_node, condition_fn=condition_fn, tensor_map=tensor_map
            )
        else:
            # Merge tensor_map if edge already exists
            src_cfg.topology[source_op].successors[target_node].tensor_map.update(
                tensor_map
            )

        # Add predecessor info to target node
        if source_node not in dst_cfg.topology[target_op].predecessors:
            dst_cfg.topology[target_op].predecessors[source_node] = SPMDTopologyInfo(
                spmd_id=source_node, tensor_map=tensor_map
            )
        else:
            # Merge tensor_map if edge already exists
            dst_cfg.topology[target_op].predecessors[source_node].tensor_map.update(
                tensor_map
            )

        ctx = f"edge {source_node}.{source_op} -> {target_node}.{target_op}"
        src_meta = self._ensure_meta_exists(
            source_node, source_op, src_cfg.operator_metas.get(source_op), ctx
        )
        dst_meta = self._ensure_meta_exists(
            target_node, target_op, dst_cfg.operator_metas.get(target_op), ctx
        )

        # 1) Validate & update state: src output_state vs dst input_state
        src_cls = self._spmd_classes[source_node]
        src_meta.validate_and_update_state_output_match_dest_input(
            dst_meta=dst_meta,
            src_spmd_class=src_cls,
            src_method_name=source_op,
            ctx=ctx,
        )

        # 2) Write back pre/post sharding to OperatorMeta
        src_out = src_meta.output_sharding or {}
        dst_in = dst_meta.input_sharding or {}

        # If tensor_map is used, we should only link the specified tensors
        # If not, we link all overlapping names (default behavior)
        if tensor_map:
            effective_src_out = {
                tensor_map[src_name]: spec
                for src_name, spec in src_out.items()
                if src_name in tensor_map
            }
            effective_dst_in = {
                src_name: dst_in[dst_name]
                for src_name, dst_name in tensor_map.items()
                if dst_name in dst_in
            }
        else:
            effective_src_out = src_out
            effective_dst_in = dst_in

        dst_meta.input_pre_sharding.setdefault(source_node, []).append(
            EdgeShardingInfo(
                condition_fn=condition_fn,
                tensor_sharding=dict(effective_src_out) if effective_src_out else {},
                target_op=target_op,
                tensor_map=tensor_map,
            )
        )
        logger.debug(
            "Builder linked input_pre_sharding: %s.%s <- %s.%s, tensor_keys=%s",
            target_node,
            target_op,
            source_node,
            source_op,
            list(effective_src_out.keys()) if effective_src_out else [],
        )

        src_meta.output_post_sharding.setdefault(target_node, []).append(
            EdgeShardingInfo(
                condition_fn=condition_fn,
                tensor_sharding=dict(effective_dst_in) if effective_dst_in else {},
                target_op=target_op,
                tensor_map=tensor_map,
            )
        )
        logger.debug(
            "Builder linked output_post_sharding: %s.%s -> %s.%s, tensor_keys=%s",
            source_node,
            source_op,
            target_node,
            target_op,
            list(effective_dst_in.keys()) if effective_dst_in else [],
        )

    def build(self, name: str = "") -> Graph:
        """
        Build and return the complete graph.

        Args:
            name: Name for the graph

        Returns:
            Constructed Graph object

        Raises:
            ValueError: If graph validation fails
        """
        # Validate that all edge references exist
        for edge in self._edges:
            if edge.source_node not in self._nodes:
                raise ValueError(
                    f"Edge references non-existent source node: {edge.source_node}"
                )
            if edge.target_node not in self._nodes:
                raise ValueError(
                    f"Edge references non-existent target node: {edge.target_node}"
                )

        # Sync static_args to spmd_config
        for node in self._nodes.values():
            if node.static_args:
                node.spmd_config.static_args.update(node.static_args)

        graph = Graph(name=name, nodes=self._nodes.copy(), edges=self._edges.copy())

        return graph

    def reset(self) -> "Builder":
        """
        Reset the builder to empty state.

        Returns:
            Self for method chaining
        """
        self._nodes.clear()
        self._spmd_classes.clear()
        self._edges.clear()
        return self

    def get_spmd_class(self, node_name: str) -> Optional[Type]:
        """
        Get the SPMD class associated with a node.

        Args:
            node_name: Name of the node

        Returns:
            The SPMD class or None if not found
        """
        return self._spmd_classes.get(node_name)
