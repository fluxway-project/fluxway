"""
Graph Builder for constructing task graphs programmatically.

Based on graph_example.py API design.

Usage:
    builder = Builder()
    builder.add_node("SPMD1", SPMD1, "description1")
    builder.add_node("SPMD2", SPMD2, "description2")
    builder.add_forward_conditional_edge("SPMD1", check_fn, {True: "SPMD2"})
    builder.add_backward_edge("SPMD2", "SPMD1")
    graph = builder.build(name="my_graph")
"""

from typing import Any, Dict, List, Optional, Callable, Union, Type, Tuple
import logging

from ...common.types import (
    Graph,
    Node,
    Edge,
    OperatorMeta,
    OpType,
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
        self._forward_edges: List[Edge] = []
        self._backward_edges: List[Edge] = []
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
        worker_count: int = 1,
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
            worker_count: Number of reader workers
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
                worker_count=4,
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
            worker_count=worker_count,
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
        # Reader uses worker_count from ReaderConfig, not from mesh
        spmd_config = SPMDConfig(
            worker_count=worker_count,
            max_async_steps=1,  # Reader doesn't need async steps
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
        source_op: OpType,
        target_node: str,
        target_op: OpType,
        condition_fn: Optional[Callable[..., bool]] = None,
    ) -> "Builder":
        """
        Add a generic edge specifying source and target op types.

        This is the core edge API. All other edge methods are syntactic sugar
        built on top of this method.

        Args:
            source_node: Source node name
            source_op: Source node's op type (FORWARD/BACKWARD/UPDATE)
            target_node: Target node name
            target_op: Target node's op type
            condition_fn: Optional condition function

        Returns:
            Self for method chaining

        Examples:
            # Forward -> Forward (equivalent to add_forward_edge)
            builder.add_func_edge("A", OpType.FORWARD, "B", OpType.FORWARD)

            # Forward -> Backward (sink node triggers its own backward)
            builder.add_func_edge("Sink", OpType.FORWARD, "Sink", OpType.BACKWARD)

            # Backward -> Backward
            builder.add_func_edge("B", OpType.BACKWARD, "A", OpType.BACKWARD)
        """
        if source_node not in self._nodes:
            raise ValueError(f"Source node {source_node} not found")
        if target_node not in self._nodes:
            raise ValueError(f"Target node {target_node} not found")

        edge = Edge(
            source_node=source_node,
            target_node=target_node,
            source_op=source_op,
            target_op=target_op,
            condition_fn=condition_fn,
        )

        # Store edge in appropriate list based on source_op
        if source_op == OpType.FORWARD:
            self._forward_edges.append(edge)
        elif source_op == OpType.BACKWARD:
            self._backward_edges.append(edge)
        else:
            # For UPDATE or other ops, store in forward_edges for now
            self._forward_edges.append(edge)

        # Skip sharding validation for Reader nodes (they don't have @spmd decorator)
        if self._is_reader_node(source_node) or self._is_reader_node(target_node):
            # Reader nodes don't have operator metadata, skip validation
            return self

        # Validate and link sharding between source and target
        self._validate_and_link_edge(
            source_node, source_op, target_node, target_op, condition_fn
        )

        return self

    def add_sink_backward_trigger(
        self, node_name: str, condition_fn: Optional[Callable[..., bool]] = None
    ) -> "Builder":
        """
        Add forward -> backward self-trigger edge for a sink node.

        This is syntactic sugar for:
            add_func_edge(node, OpType.FORWARD, node, OpType.BACKWARD, condition_fn)

        Args:
            node_name: Sink node name
            condition_fn: Optional condition function (e.g., only trigger in training mode)

        Returns:
            Self for method chaining
        """
        return self.add_func_edge(
            node_name, OpType.FORWARD, node_name, OpType.BACKWARD, condition_fn
        )

    # -------------------------------------------------------------------------
    # Forward edges
    # -------------------------------------------------------------------------

    def add_forward_edge(
        self,
        source_node: str,
        target_node: str,
        condition_fn: Optional[Callable[..., bool]] = None,
    ) -> "Builder":
        """
        Add a forward edge: source.forward -> target.forward

        This is syntactic sugar for:
            add_func_edge(source, OpType.FORWARD, target, OpType.FORWARD, condition_fn)

        Args:
            source_node: Source node name
            target_node: Target node name
            condition_fn: Condition function for the edge

        Returns:
            Self for method chaining
        """
        return self.add_func_edge(
            source_node, OpType.FORWARD, target_node, OpType.FORWARD, condition_fn
        )

    def add_forward_conditional_edge(
        self,
        source_node: str,
        condition_fn: Optional[Callable[..., bool]],
        target_mapping: Union[Dict[bool, Optional[str]], str],
    ) -> "Builder":
        """
        Add conditional forward edges based on a condition function.

        Supports multiple forms:
        1. add_forward_conditional_edge("SPMD1", check_fn, {True: "SPMD2", False: "SPMD3"})
        2. add_forward_conditional_edge("SPMD1", check_fn, "SPMD2")  # Equivalent to {True: "SPMD2"}
        3. add_forward_conditional_edge("SPMD1", None, "SPMD2")  # Unconditional edge

        Args:
            source_node: Source node name
            condition_fn: Condition function that returns bool, or None for unconditional
            target_mapping: Either a mapping from condition result to target node,
                          or a simple target node string

        Returns:
            Self for method chaining
        """
        # Simple string form
        if isinstance(target_mapping, str):
            if condition_fn is None:
                # Unconditional edge: add_forward_conditional_edge("SPMD1", None, "SPMD2")
                return self.add_forward_edge(source_node, target_mapping)
            else:
                # Conditional edge with True: add_forward_conditional_edge("SPMD1", check, "SPMD2")
                # Equivalent to {True: "SPMD2"}
                self.add_forward_edge(source_node, target_mapping, condition_fn)
                return self

        # Dict form with condition mapping
        for condition_result, target_node in target_mapping.items():
            if target_node is not None:
                # Create a wrapped condition function that checks for specific result
                if condition_fn is not None:

                    def make_condition(cond_fn: Callable, expected: bool) -> Callable:
                        def wrapped_fn(*args, **kwargs) -> bool:
                            return cond_fn(*args, **kwargs) == expected

                        return wrapped_fn

                    wrapped_condition = make_condition(condition_fn, condition_result)
                    self.add_forward_edge(source_node, target_node, wrapped_condition)
                else:
                    # No condition function means unconditional edge
                    if condition_result is True:
                        self.add_forward_edge(source_node, target_node)

        return self

    # -------------------------------------------------------------------------
    # Backward edges
    # -------------------------------------------------------------------------

    def add_backward_edge(
        self,
        source_node: str,
        target_node: str,
        condition_fn: Optional[Callable[..., bool]] = None,
    ) -> "Builder":
        """
        Add a backward edge: source.backward -> target.backward

        This is syntactic sugar for:
            add_func_edge(source, OpType.BACKWARD, target, OpType.BACKWARD, condition_fn)

        Args:
            source_node: Source node name (where gradient originates)
            target_node: Target node name (where gradient flows to)

        Returns:
            Self for method chaining
        """
        return self.add_func_edge(
            source_node, OpType.BACKWARD, target_node, OpType.BACKWARD, condition_fn
        )

    def add_backward_conditional_edge(
        self,
        source_node: str,
        condition_fn: Optional[Callable[..., bool]],
        target_mapping: Union[Dict[bool, Optional[str]], str],
    ) -> "Builder":
        """
        Add conditional backward edges.

        Supports multiple forms:
        1. add_backward_conditional_edge("SPMD2", check_fn, {True: "SPMD1", False: None})
        2. add_backward_conditional_edge("SPMD2", check_fn, "SPMD1")  # Equivalent to {True: "SPMD1"}
        3. add_backward_conditional_edge("SPMD2", None, "SPMD1")  # Unconditional edge

        Args:
            source_node: Source node name
            condition_fn: Condition function that returns bool, or None for unconditional
            target_mapping: Either a mapping from condition result to target node,
                          or a simple target node string

        Returns:
            Self for method chaining
        """
        # Simple string form
        if isinstance(target_mapping, str):
            if condition_fn is None:
                # Unconditional edge: add_backward_conditional_edge("SPMD2", None, "SPMD1")
                return self.add_backward_edge(source_node, target_mapping)
            else:
                # Conditional edge with True: add_backward_conditional_edge("SPMD2", check, "SPMD1")
                # Equivalent to {True: "SPMD1"}
                self.add_backward_edge(source_node, target_mapping, condition_fn)
                return self

        # Dict form with condition mapping
        for condition_result, target_node in target_mapping.items():
            if target_node is not None:
                if condition_fn is not None:

                    def make_condition(cond_fn: Callable, expected: bool) -> Callable:
                        def wrapped_fn(*args, **kwargs) -> bool:
                            return cond_fn(*args, **kwargs) == expected

                        return wrapped_fn

                    wrapped_condition = make_condition(condition_fn, condition_result)
                    self.add_backward_edge(source_node, target_node, wrapped_condition)
                else:
                    # No condition function means unconditional edge
                    if condition_result is True:
                        self.add_backward_edge(source_node, target_node)

        return self

    # -------------------------------------------------------------------------
    # Bidirectional edges
    # -------------------------------------------------------------------------

    def add_edge(self, source_node: str, target_node: str) -> "Builder":
        """
        Add a bidirectional edge between two nodes.

        This will:
        - Add a forward edge source_node -> target_node
        - Add a backward edge target_node -> source_node
        And validate both forward and backward operator compatibility.
        """
        # Forward direction
        self.add_forward_edge(source_node, target_node)
        # Backward direction
        self.add_backward_edge(target_node, source_node)
        return self

    def add_conditional_edge(
        self,
        source_node: str,
        condition_fn: Optional[Callable[..., bool]],
        target_mapping: Union[Dict[bool, Optional[str]], str],
    ) -> "Builder":
        """
        Add bidirectional conditional edges.

        Semantics:
        - Forward conditional edge: source_node -> targets (as in add_forward_conditional_edge)
        - Backward conditional edge: targets -> source_node (mirror)
        """
        # Forward conditional edge(s)
        self.add_forward_conditional_edge(source_node, condition_fn, target_mapping)

        # Backward conditional edge(s) are mirrored:
        if isinstance(target_mapping, str):
            # Single target: backward from target -> source
            # 反向的发送依赖于前向的实际来源，不再直接绑定 condition_fn
            self.add_backward_conditional_edge(target_mapping, None, source_node)
        else:
            # Dict mapping: create backward edges for non-None targets
            for _, tgt in target_mapping.items():
                if tgt is not None:
                    # 同样不使用 condition_fn，表示反向发送依赖于前向的来源
                    self.add_backward_conditional_edge(tgt, None, source_node)
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
        source_op: OpType,
        target_node: str,
        target_op: OpType,
        condition_fn: Optional[Callable[..., bool]],
    ) -> None:
        """
        Validate and link operators between two nodes with arbitrary op types.

        核心注意点：这里必须基于 Node.spmd_config 里的 user_forward_fn / user_backward_fn
        上挂载的 OperatorMeta 来写回 input_pre_sharding / output_post_sharding，
        这样修改才能随 SPMDConfig 一起传递到 worker 进程。
        """
        # 从 Node 的 spmd_config 中获取实际传给 worker 的函数对象
        # 同时仍然保留 spmd_class 引用，用于 state 校验时的反射
        src_node = self._nodes[source_node]
        dst_node = self._nodes[target_node]

        src_cfg = src_node.spmd_config
        dst_cfg = dst_node.spmd_config

        def _get_fn_and_meta(
            cfg: SPMDConfig, op: OpType
        ) -> Tuple[Optional[Callable], Optional[OperatorMeta]]:
            if op == OpType.FORWARD:
                fn = cfg.user_forward_fn
                meta = cfg.forward_meta
            elif op == OpType.BACKWARD:
                fn = cfg.user_backward_fn
                meta = cfg.backward_meta
            else:
                fn = cfg.user_update_fn
                meta = cfg.update_meta
            if fn is None:
                return None, None
            return fn, meta

        _, src_meta = _get_fn_and_meta(src_cfg, source_op)
        _, dst_meta = _get_fn_and_meta(dst_cfg, target_op)

        # 用于错误信息和 state 校验的上下文
        src_cls = self._spmd_classes[source_node]
        ctx = f"edge {source_node}.{source_op.name.lower()} -> {target_node}.{target_op.name.lower()}"
        self._ensure_meta_exists(source_node, source_op.name.lower(), src_meta, ctx)
        self._ensure_meta_exists(target_node, target_op.name.lower(), dst_meta, ctx)

        # For self-loops (same node, different ops), skip tensor validation
        # because the data flows internally without external sharding requirements.
        # However, state output inference must still run so that
        # _send_output knows which state fields to transmit via _self_receive_state.
        is_self_loop = source_node == target_node

        if not is_self_loop:
            # 1) Validate tensor: src_op output vs dst_op input
            src_meta.validate_tensor_io_match(dst_meta, src_is_output=True, ctx=ctx)

        # 2) Validate & update state: src_op output_state vs dst_op input_state
        #    This must run for ALL edges (including self-loops) so that
        #    output_arguments[STATE] is populated and _send_output can
        #    extract and transmit state fields.
        src_meta.validate_and_update_state_output_match_dest_input(
            dst_meta=dst_meta,
            src_spmd_class=src_cls,
            src_method_name=source_op.name.lower(),
            ctx=ctx,
        )

        # 3) Write back pre/post sharding to OperatorMeta (with target_op info)
        src_out = src_meta.output_sharding or {}
        dst_in = dst_meta.input_sharding or {}

        # Update target.target_op.input_pre_sharding:
        #   key: source_node
        #   value: edge sharding info including target_op
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
                target_op.name.lower(),
                source_node,
                source_op.name.lower(),
                list(src_out.keys()),
            )

        # Update source.source_op.output_post_sharding:
        #   key: target_node
        #   value: edge sharding info including target_op
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
                source_op.name.lower(),
                target_node,
                target_op.name.lower(),
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
        # Combine all edges
        all_edges = self._forward_edges + self._backward_edges

        # Validate that all edge references exist
        for edge in all_edges:
            if edge.source_node not in self._nodes:
                raise ValueError(
                    f"Edge references non-existent source node: {edge.source_node}"
                )
            if edge.target_node not in self._nodes:
                raise ValueError(
                    f"Edge references non-existent target node: {edge.target_node}"
                )

        graph = Graph(name=name, nodes=self._nodes.copy(), edges=all_edges)

        return graph

    def reset(self) -> "Builder":
        """
        Reset the builder to empty state.

        Returns:
            Self for method chaining
        """
        self._nodes.clear()
        self._forward_edges.clear()
        self._backward_edges.clear()
        self._spmd_classes.clear()
        self._reader_configs.clear()
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
