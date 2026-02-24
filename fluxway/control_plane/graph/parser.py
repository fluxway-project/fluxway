"""
Graph Parser for task orchestration.

Based on ARCHITECTURE.md Section 2.1.1

Responsibilities:
- Parse user-defined Graph task orchestration
- Build dependency graph between SPMD nodes
- Identify SPMD hierarchy and connections
- Validate graph structure (acyclicity, reachability)
- Compute topological ordering for execution
"""

from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path
from collections import deque

from ...common.types import Graph, Node, Edge, GraphValidationResult, SPMDConfig


class GraphParser:
    """
    Parser for user-defined Graph configurations.

    Supports JSON/YAML configuration files and Python API for graph definition.
    Uses data-driven execution model where nodes execute when all upstream
    dependencies have completed.
    """

    def __init__(self):
        self._graph: Optional[Graph] = None

    def parse_file(self, config_path: Union[str, Path]) -> Graph:
        """
        Parse Graph configuration from a file (JSON/YAML).

        Args:
            config_path: Path to the configuration file

        Returns:
            Parsed Graph object

        TODO: Implement JSON parsing
        TODO: Implement YAML parsing
        TODO: Validate configuration schema
        """
        pass

    def parse_dict(self, config: Dict[str, Any]) -> Graph:
        """
        Parse Graph configuration from a dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Parsed Graph object

        TODO: Implement dictionary parsing
        TODO: Build Node objects
        TODO: Build Edge objects
        TODO: Identify source and sink nodes
        """
        pass

    def build_node(
        self,
        node_id: str,
        config: SPMDConfig,
        upstream_nodes: Optional[List[str]] = None,
        downstream_nodes: Optional[List[str]] = None,
        forward_fn: Optional[callable] = None,
        backward_fn: Optional[callable] = None,
    ) -> Node:
        """
        Build a Node programmatically.

        Args:
            node_id: Unique identifier for the node
            config: SPMD configuration
            upstream_nodes: List of upstream dependency node IDs
            downstream_nodes: List of downstream dependent node IDs
            forward_fn: User-defined forward function
            backward_fn: User-defined backward function

        Returns:
            Constructed Node

        TODO: Implement node building
        TODO: Validate configuration
        """
        pass

    def add_edge(
        self,
        source_node: str,
        target_node: str,
        edge_name: str = "",
        is_gradient_edge: bool = False,
    ) -> Edge:
        """
        Add a directed edge to the Graph.

        Args:
            source_node: Upstream node ID
            target_node: Downstream node ID
            edge_name: Optional name for the edge
            is_gradient_edge: True if edge carries gradient data

        Returns:
            Constructed Edge

        TODO: Implement edge creation
        TODO: Update node upstream/downstream lists
        TODO: Validate node existence
        """
        pass

    def compute_topological_order(self) -> List[str]:
        """
        Compute topological ordering of Graph nodes.

        Uses Kahn's algorithm to find a valid execution order
        where each node comes after all its dependencies.

        Returns:
            List of node IDs in topological order

        Raises:
            ValueError: If cycle detected (not a valid DAG)
        """
        if self._graph is None:
            return []

        # Build adjacency info from edges
        in_degree: Dict[str, int] = dict.fromkeys(self._graph.nodes, 0)
        for node_name, node in self._graph.nodes.items():
            # Count incoming edges based on forward_edges of other nodes
            pass

        # Count in-degrees from edges
        for edge in self._graph.edges:
            if edge.target_node in in_degree:
                in_degree[edge.target_node] += 1

        # Start with nodes that have no dependencies
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node_id = queue.popleft()
            result.append(node_id)

            # Reduce in-degree of downstream nodes
            for edge in self._graph.edges:
                if edge.source_node == node_id and edge.target_node in in_degree:
                    in_degree[edge.target_node] -= 1
                    if in_degree[edge.target_node] == 0:
                        queue.append(edge.target_node)

        # Check for cycles
        if len(result) != len(self._graph.nodes):
            raise ValueError("Cycle detected in Graph - not a valid DAG")

        return result

    def identify_source_sink_nodes(self) -> tuple:
        """
        Identify source nodes (no upstream) and sink nodes (no downstream).

        Returns:
            (source_nodes, sink_nodes) tuple of lists
        """
        if self._graph is None:
            return [], []

        # Build sets of nodes with incoming/outgoing edges
        has_incoming = set()
        has_outgoing = set()

        for edge in self._graph.edges:
            has_incoming.add(edge.target_node)
            has_outgoing.add(edge.source_node)

        source_nodes = [name for name in self._graph.nodes if name not in has_incoming]
        sink_nodes = [name for name in self._graph.nodes if name not in has_outgoing]

        return source_nodes, sink_nodes

    def validate(self) -> GraphValidationResult:
        """
        Validate the Graph configuration.

        Checks:
        - Graph is acyclic (valid DAG)
        - All referenced nodes exist
        - No orphaned nodes (unreachable from sources)
        - Source and sink nodes correctly identified

        Returns:
            GraphValidationResult with validation details
        """
        result = GraphValidationResult()

        if self._graph is None:
            result.is_valid = False
            result.errors.append("Graph is not initialized")
            return result

        # Check for cycles using topological sort
        try:
            self.compute_topological_order()
            result.is_acyclic = True
        except ValueError as e:
            result.is_valid = False
            result.is_acyclic = False
            result.errors.append(str(e))

        # Check all referenced nodes in edges exist
        for edge in self._graph.edges:
            if edge.source_node not in self._graph.nodes:
                result.is_valid = False
                result.errors.append(
                    f"Edge references non-existent source node '{edge.source_node}'"
                )
            if edge.target_node not in self._graph.nodes:
                result.is_valid = False
                result.errors.append(
                    f"Edge references non-existent target node '{edge.target_node}'"
                )

        # Check for unreachable nodes
        source_nodes, _ = self.identify_source_sink_nodes()
        if source_nodes:
            reachable = self._find_reachable_nodes(set(source_nodes))
            for node_name in self._graph.nodes:
                if node_name not in reachable:
                    result.unreachable_nodes.append(node_name)
                    result.warnings.append(
                        f"Node '{node_name}' is unreachable from source nodes"
                    )

        return result

    def _find_reachable_nodes(self, start_nodes: Set[str]) -> Set[str]:
        """Find all nodes reachable from the given start nodes."""
        if self._graph is None:
            return set()

        reachable = set()
        queue = deque(start_nodes)

        while queue:
            node_id = queue.popleft()
            if node_id in reachable:
                continue
            reachable.add(node_id)

            # Find downstream nodes from edges
            for edge in self._graph.edges:
                if edge.source_node == node_id and edge.target_node not in reachable:
                    queue.append(edge.target_node)

        return reachable

    def get_spmd_topology(self) -> Dict[str, List[str]]:
        """
        Extract SPMD topology (hierarchy and connections).

        Returns:
            Mapping from node name to list of downstream node names
        """
        if self._graph is None:
            return {}

        topology: Dict[str, List[str]] = {name: [] for name in self._graph.nodes}
        for edge in self._graph.edges:
            if edge.source_node in topology:
                topology[edge.source_node].append(edge.target_node)

        return topology

    def get_execution_levels(self) -> List[List[str]]:
        """
        Group nodes into execution levels for parallel execution.

        Nodes in the same level have no dependencies on each other
        and can be executed in parallel.

        Returns:
            List of levels, each level is a list of node names
        """
        if self._graph is None:
            return []

        # Compute levels based on longest path from sources
        levels: Dict[str, int] = {}
        topo_order = self.compute_topological_order()

        for node_id in topo_order:
            # Find max level of upstream nodes
            max_upstream_level = -1
            for edge in self._graph.edges:
                if edge.target_node == node_id and edge.source_node in levels:
                    max_upstream_level = max(
                        max_upstream_level, levels[edge.source_node]
                    )
            levels[node_id] = max_upstream_level + 1

        # Group by level
        if not levels:
            return []

        max_level = max(levels.values())
        result = [[] for _ in range(max_level + 1)]
        for node_id, level in levels.items():
            result[level].append(node_id)

        return result

    @property
    def graph(self) -> Optional[Graph]:
        """Get the parsed Graph."""
        return self._graph
