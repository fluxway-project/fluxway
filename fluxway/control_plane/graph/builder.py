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
    SPMDConfig,
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
        if (
            not hasattr(spmd_class, "is_fluxway_spmd")
            or not spmd_class.is_fluxway_spmd()
        ):
            raise ValueError(
                f"Class '{spmd_class.__name__}' must be decorated with @fluxway.spmd"
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
        src_dir: str,
        data_type: str = "json",
        batch_size: int = 1000,
        file_pattern: str = "*",
        num_workers: int = 1,
        resource: Optional[Dict[str, Any]] = None,
        description: str = "",
        reader_class: Optional[Type] = None,
        **static_args: Any,
    ) -> "Builder":
        """
        Add a Reader node to the graph.

        Reader nodes are now generic SPMD nodes that use a Reader class.
        """
        from ...reader.grain_reader import GrainReader

        reader_cls = reader_class or GrainReader

        # Combine specialized reader args with other static args
        all_static_args = {
            "src_dir": src_dir,
            "data_type": data_type,
            "batch_size": batch_size,
            "file_pattern": file_pattern,
            "num_workers": num_workers,
            **static_args,
        }

        # Handle resource
        resource_spec = None
        if resource:
            resource_spec = ResourceSpec(
                num_cpus=resource.get("nums_cpu", resource.get("num_cpus", 1)),
                num_gpus=resource.get("nums_gpu", resource.get("num_gpus", 0)),
                memory_gb=resource.get("memory", resource.get("memory_gb", 0.0)),
            )

        # Add generic node
        self.add_node(name, reader_cls, description)

        # Inject static args and resource if provided
        node = self._nodes[name]
        node.static_args.update(all_static_args)
        if resource_spec:
            node.resource = resource_spec

        # Reader nodes default to world_size == num_workers
        if node.spmd_config:
            # We don't have mesh yet in Node typically at this stage,
            # but we can store it in static_args for now or wait for mesh creation.
            # For now, just ensure static_args has it.
            pass

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
            source_node, source_op, target_node, target_op, condition_fn
        )

        return self

    # -------------------------------------------------------------------------
    # Internal helpers for operator metadata & sharding linking
    # -------------------------------------------------------------------------

    def _ensure_meta_exists(
        self, node_name: str, method_name: str, meta: Optional[OperatorMeta], ctx: str
    ) -> None:
        """Ensure that an operator meta exists for the given node/method; otherwise raise."""
        if meta is None:
            raise ValueError(
                f"{ctx}: SPMD '{node_name}' does not define operator '{method_name}' "
                f"or it is not decorated with @operator."
            )

    def _validate_and_link_edge(
        self,
        source_node: str,
        source_op: str,
        target_node: str,
        target_op: str,
        condition_fn: Optional[Callable[..., bool]],
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

        # Add successor info to source node
        src_cfg.topology[source_op].successors[target_node] = SPMDTopologyInfo(
            spmd_id=target_node,
            condition_fn=condition_fn,
            # tensor_map can be populated later if needed, but for now ID is enough
        )

        # Add predecessor info to target node
        dst_cfg.topology[target_op].predecessors[source_node] = SPMDTopologyInfo(
            spmd_id=source_node
            # tensor_map can be populated later if needed
        )

        def _get_fn_and_meta(
            cfg: SPMDConfig, op_name: str
        ) -> Tuple[Optional[Callable], Optional[OperatorMeta]]:
            fn = cfg.operators.get(op_name)
            meta = cfg.operator_metas.get(op_name)
            if fn is None:
                return None, None
            return fn, meta

        _, src_meta = _get_fn_and_meta(src_cfg, source_op)
        _, dst_meta = _get_fn_and_meta(dst_cfg, target_op)

        src_cls = self._spmd_classes[source_node]
        ctx = f"edge {source_node}.{source_op} -> {target_node}.{target_op}"
        self._ensure_meta_exists(source_node, source_op, src_meta, ctx)
        self._ensure_meta_exists(target_node, target_op, dst_meta, ctx)

        # 1) Validate & update state: src output_state vs dst input_state
        src_meta.validate_and_update_state_output_match_dest_input(
            dst_meta=dst_meta,
            src_spmd_class=src_cls,
            src_method_name=source_op,
            ctx=ctx,
        )

        # 2) Write back pre/post sharding to OperatorMeta
        src_out = src_meta.output_sharding or {}
        dst_in = dst_meta.input_sharding or {}

        dst_meta.input_pre_sharding.setdefault(source_node, []).append(
            EdgeShardingInfo(
                condition_fn=condition_fn,
                tensor_sharding=dict(src_out) if src_out else {},
                target_op=target_op,
            )
        )
        logger.debug(
            "Builder linked input_pre_sharding: %s.%s <- %s.%s, tensor_keys=%s",
            target_node,
            target_op,
            source_node,
            source_op,
            list(src_out.keys()) if src_out else [],
        )

        src_meta.output_post_sharding.setdefault(target_node, []).append(
            EdgeShardingInfo(
                condition_fn=condition_fn,
                tensor_sharding=dict(dst_in) if dst_in else {},
                target_op=target_op,
            )
        )
        logger.debug(
            "Builder linked output_post_sharding: %s.%s -> %s.%s, tensor_keys=%s",
            source_node,
            source_op,
            target_node,
            target_op,
            list(dst_in.keys()) if dst_in else [],
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
