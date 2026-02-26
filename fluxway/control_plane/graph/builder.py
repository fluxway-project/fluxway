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
    NodeType,
    ReaderConfig,
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
        self._reader_configs: Dict[str, ReaderConfig] = {}  # Reader configurations

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
        encoding: str = "utf-8",
        output_data_type: Optional[Type] = None,
    ) -> "Builder":
        """
        Add a Reader node to the graph.

        Reader nodes are special nodes that actively read data from external sources
        and push data to downstream SPMD nodes. Unlike normal SPMD nodes, Reader nodes
        don't wait for upstream triggers - they actively generate data.

        Args:
            name: Unique identifier for the node
            src_dir: Source data directory path
            data_type: Data type to read (json, parquet, csv, etc.)
            batch_size: Number of records per batch
            file_pattern: Glob pattern for matching files (e.g., "*.json")
            num_workers: Number of reader workers
            resource: Resource requirements dict (nums_cpu, nums_gpu, memory)
            description: Human-readable description of the node
            encoding: File encoding (default: utf-8)
            output_data_type: Output data type class (e.g., TensorOnly).
                             If provided, reader will convert batch data to this type.
                             The tensor field name from this type will be used when sending data.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If node with same name already exists

        Example:
            builder.add_reader(
                name="DataReader",
                src_dir="/data/input",
                data_type="json",
                batch_size=1000,
                num_workers=4,
                resource={"nums_cpu": 1, "nums_gpu": 0, "memory": 10},
                description="JSON data reader"
            )
        """
        if name in self._nodes:
            raise ValueError(f"Node '{name}' already exists in the graph")

        # Parse output_data_type to extract tensor field name and type
        tensor_name = None
        tensor_type = None
        if output_data_type:
            tensor_name, tensor_type = self._extract_tensor_field_info(output_data_type)
            logger.debug(
                f"Extracted tensor field name: {tensor_name}, type: {tensor_type}"
            )
            if not tensor_name:
                raise ValueError(
                    f"Cannot extract tensor field name from output_data_type {output_data_type.__name__}. "
                    f"For eager mode (reader's downstream), output_data_type must have exactly one tensor field (np.ndarray)."
                )

        # Create ReaderConfig
        # Note: output_data_type is only used for parsing, not stored in ReaderConfig
        reader_config = ReaderConfig(
            src_dir=src_dir,
            data_type=data_type,
            batch_size=batch_size,
            file_pattern=file_pattern,
            num_workers=num_workers,
            encoding=encoding,
            tensor_name=tensor_name,
            tensor_type=tensor_type,
        )

        # Parse resource specification
        resource_spec = None
        if resource:
            resource_spec = ResourceSpec(
                num_cpus=resource.get("nums_cpu", resource.get("num_cpus", 1)),
                num_gpus=resource.get("nums_gpu", resource.get("num_gpus", 0)),
                memory_gb=resource.get("memory", resource.get("memory_gb", 0.0)),
            )

        # Create SPMDConfig for Reader
        # Reader uses num_workers from ReaderConfig, not from mesh
        spmd_config = SPMDConfig(
            # num_workers=num_workers, # Property is derived from mesh
            max_async_steps=1  # Reader doesn't need async steps
        )

        # Create Node with READER type
        node = Node(
            name=name,
            description=description,
            resource=resource_spec,
            spmd_config=spmd_config,
            node_type=NodeType.READER,
            reader_config=reader_config,
        )

        self._nodes[name] = node
        self._reader_configs[name] = reader_config

        return self

    def _extract_tensor_field_info(
        self, output_data_type: Type
    ) -> Tuple[Optional[str], Optional[Type]]:
        """
        Extract tensor field name and type from output_data_type.

        For eager mode (reader's downstream), there should be exactly one tensor arg.
        A tensor field is identified by having type np.ndarray (or compatible).

        Args:
            output_data_type: The output data type class (e.g., TensorOnly)

        Returns:
            Tuple of (tensor_field_name, tensor_field_type), or (None, None) if not found
        """
        import numpy as np

        # Get field annotations
        annotations = getattr(output_data_type, "__annotations__", {})

        # Try dataclass_fields for more info
        if hasattr(output_data_type, "__dataclass_fields__"):
            fields = output_data_type.__dataclass_fields__
            # Check each field's type
            for field_name, field_info in fields.items():
                if field_name.startswith("_"):
                    continue
                field_type = annotations.get(field_name, None)
                # Check if type is np.ndarray or compatible
                if field_type is np.ndarray or (
                    hasattr(field_type, "__origin__")
                    and getattr(field_type, "__origin__", None) is np.ndarray
                ):
                    return field_name, field_type
                # Also check string representation
                type_str = str(field_type)
                if "ndarray" in type_str or "numpy" in type_str.lower():
                    return field_name, field_type

        # Fallback: check annotations directly
        for field_name, field_type in annotations.items():
            if field_name.startswith("_"):
                continue
            if field_type is np.ndarray:
                return field_name, field_type
            type_str = str(field_type)
            if "ndarray" in type_str or "numpy" in type_str.lower():
                return field_name, field_type

        # If no tensor field found, return first non-private field as fallback
        for field_name, field_type in annotations.items():
            if not field_name.startswith("_"):
                return field_name, field_type

        return None, None

    def _is_reader_node(self, name: str) -> bool:
        """
        Check if a node is a Reader node.

        Args:
            name: Node name to check

        Returns:
            True if the node is a Reader node
        """
        node = self._nodes.get(name)
        return node is not None and node.node_type == NodeType.READER

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

        # Skip sharding validation for Reader nodes (they don't have @spmd decorator)
        if self._is_reader_node(source_node):
            # Update Reader target_op (Reader only has one successor/operator target)
            src_node = self._nodes[source_node]
            if src_node.reader_config:
                src_node.reader_config.target_op = target_op
            return self

        if self._is_reader_node(target_node):
            return self

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

        # For self-loops (same node, different ops), skip tensor I/O validation
        # because the data flows internally. State inference must still run.
        is_self_loop = source_node == target_node

        if not is_self_loop:
            # 1) Validate tensor: src_op output vs dst_op input
            src_meta.validate_tensor_io_match(dst_meta, src_is_output=True, ctx=ctx)

        # 2) Validate & update state: src output_state vs dst input_state
        src_meta.validate_and_update_state_output_match_dest_input(
            dst_meta=dst_meta,
            src_spmd_class=src_cls,
            src_method_name=source_op,
            ctx=ctx,
        )

        # 3) Write back pre/post sharding to OperatorMeta
        src_out = src_meta.output_sharding or {}
        dst_in = dst_meta.input_sharding or {}

        if src_out or is_self_loop:
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
                list(src_out.keys()),
            )

        if dst_in or is_self_loop:
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
                list(dst_in.keys()),
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
        self._reader_configs.clear()
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
