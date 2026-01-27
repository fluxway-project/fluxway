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

from typing import Dict, List, Optional, Callable, Any, Union, Type

from fluxway.common.types import Graph, Node, Edge, SPMDConfig, ResourceSpec


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
        if not hasattr(spmd_class, "is_fluxway_spmd") or not spmd_class.is_fluxway_spmd():
            raise ValueError(f"Class '{spmd_class.__name__}' must be decorated with @fluxway.spmd")

        # Get resource and config from the decorated class
        resource = spmd_class.get_resource()
        spmd_config = spmd_class.get_spmd_config()

        node = Node(name=name, description=description, resource=resource, spmd_config=spmd_config)

        self._nodes[name] = node
        self._spmd_classes[name] = spmd_class

        return self

    def add_forward_edge(self, source_node: str, target_node: str) -> "Builder":
        """
        Add an unconditional forward edge between two nodes.

        Args:
            source_node: Source node name
            target_node: Target node name

        Returns:
            Self for method chaining
        """
        if source_node not in self._nodes:
            raise ValueError(f"Source node {source_node} not found")
        if target_node not in self._nodes:
            raise ValueError(f"Target node {target_node} not found")
        edge = Edge(source_node=source_node, target_node=target_node, condition_fn=None)
        self._forward_edges.append(edge)

        # Update node's forward_edges
        self._nodes[source_node].forward_edges.append(edge)

        return self

    def _add_forward_edge_with_condition(
        self, source_node: str, target_node: str, condition_fn: Callable[..., bool]
    ) -> "Builder":
        """
        Internal method to add a forward edge with a condition function.

        Args:
            source_node: Source node name
            target_node: Target node name
            condition_fn: Condition function for the edge

        Returns:
            Self for method chaining
        """
        if source_node not in self._nodes:
            raise ValueError(f"Source node {source_node} not found")
        if target_node not in self._nodes:
            raise ValueError(f"Target node {target_node} not found")
        edge = Edge(source_node=source_node, target_node=target_node, condition_fn=condition_fn)
        self._forward_edges.append(edge)

        # Update node's forward_edges
        if source_node in self._nodes:
            self._nodes[source_node].forward_edges.append(edge)

        return self

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
                self._add_forward_edge_with_condition(source_node, target_mapping, condition_fn)
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
                    self._add_forward_edge_with_condition(
                        source_node, target_node, wrapped_condition
                    )
                else:
                    # No condition function means unconditional edge
                    if condition_result == True:
                        self.add_forward_edge(source_node, target_node)

        return self

    def add_backward_edge(self, source_node: str, target_node: str) -> "Builder":
        """
        Add an unconditional backward edge between two nodes.

        Args:
            source_node: Source node name (where gradient originates)
            target_node: Target node name (where gradient flows to)

        Returns:
            Self for method chaining
        """
        if source_node not in self._nodes:
            raise ValueError(f"Source node {source_node} not found")
        if target_node not in self._nodes:
            raise ValueError(f"Target node {target_node} not found")
        edge = Edge(source_node=source_node, target_node=target_node, condition_fn=None)
        self._backward_edges.append(edge)

        # Update node's backward_edges
        if source_node in self._nodes:
            self._nodes[source_node].backward_edges.append(edge)

        return self

    def _add_backward_edge_with_condition(
        self, source_node: str, target_node: str, condition_fn: Callable[..., bool]
    ) -> "Builder":
        """
        Internal method to add a backward edge with a condition function.

        Args:
            source_node: Source node name (where gradient originates)
            target_node: Target node name (where gradient flows to)
            condition_fn: Condition function for the edge

        Returns:
            Self for method chaining
        """
        if source_node not in self._nodes:
            raise ValueError(f"Source node {source_node} not found")
        if target_node not in self._nodes:
            raise ValueError(f"Target node {target_node} not found")
        edge = Edge(source_node=source_node, target_node=target_node, condition_fn=condition_fn)
        self._backward_edges.append(edge)

        # Update node's backward_edges
        if source_node in self._nodes:
            self._nodes[source_node].backward_edges.append(edge)

        return self

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
                self._add_backward_edge_with_condition(source_node, target_mapping, condition_fn)
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
                    self._add_backward_edge_with_condition(
                        source_node, target_node, wrapped_condition
                    )
                else:
                    # No condition function means unconditional edge
                    if condition_result == True:
                        self.add_backward_edge(source_node, target_node)

        return self

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
                raise ValueError(f"Edge references non-existent source node: {edge.source_node}")
            if edge.target_node not in self._nodes:
                raise ValueError(f"Edge references non-existent target node: {edge.target_node}")

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
