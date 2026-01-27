"""
Scheduler for resource allocation based on Ray Placement Groups.

Based on ARCHITECTURE.md Section 2.1.2

Responsibilities:
- Resource allocation based on Ray Placement Group
- Schedule based on SPMD resource requirements (CPU, GPU, memory)
- Optimize resource utilization (affinity, data locality)
- Handle resource shortage and rescheduling
- Support preemption and eviction (via Ray API)
"""

from typing import List, Optional, Any

from fluxway.common.types import Graph, ResourceSpec, PlacementPlan


class Scheduler:
    """
    Resource scheduler using Ray Placement Groups.

    Scheduling strategies:
    - Resource-aware scheduling
    - Affinity scheduling (TODO: interface only)
    - Dynamic adjustment (TODO: interface only)
    - Conditional branch shared PG (TODO: interface only)
    """

    def __init__(self):
        self._placement_groups: dict = {}
        self._node_priorities: dict = {}

    def schedule(self, graph: Graph) -> PlacementPlan:
        """
        Generate resource allocation plan based on Graph.

        Args:
            graph: The task graph to schedule

        Returns:
            Placement plan for all SPMDs

        TODO: Analyze graph resource requirements
        TODO: Generate placement group configurations
        TODO: Handle conditional branch resource sharing
        """
        pass

    def allocate_resources(self, plan: PlacementPlan) -> Any:
        """
        Allocate resources based on Ray Placement Group.

        Args:
            plan: The placement plan to execute

        Returns:
            Ray PlacementGroup object

        TODO: Create Ray Placement Groups
        TODO: Handle allocation failures
        TODO: Implement retry logic
        """
        pass

    def create_shared_pg(self, node_ids: List[str], resources: ResourceSpec) -> Any:
        """
        Create shared Placement Group for conditional branch nodes.

        Args:
            node_ids: IDs of nodes sharing the PG
            resources: Resource specification

        Returns:
            Ray PlacementGroup object

        TODO: Implement shared PG creation
        TODO: Handle resource conflicts
        """
        pass

    def preempt_node(self, node_id: str, reason: str) -> bool:
        """
        Preempt node resources (based on Ray API).

        Args:
            node_id: ID of node to preempt
            reason: Reason for preemption

        Returns:
            True if preemption successful

        TODO: Implement preemption logic (interface only for now)
        TODO: Trigger state saving before preemption
        """
        pass

    def evict_node(self, node_id: str, checkpoint: bool = True) -> bool:
        """
        Evict node (save state before eviction).

        Args:
            node_id: ID of node to evict
            checkpoint: Whether to save checkpoint before eviction

        Returns:
            True if eviction successful

        TODO: Implement eviction logic (interface only for now)
        TODO: Coordinate with StateManager for checkpoint
        """
        pass

    def check_resource_availability(self, resources: ResourceSpec) -> bool:
        """
        Check if resources are available.

        Args:
            resources: Required resources

        Returns:
            True if resources are available

        TODO: Query Ray cluster for available resources
        TODO: Consider reserved resources
        """
        pass

    def get_node_priority(self, node_id: str) -> int:
        """
        Get priority of a node for preemption decisions.

        TODO: Implement priority calculation
        TODO: Consider execution phase, resource usage, etc.
        """
        pass

    def release_resources(self, node_id: str) -> None:
        """
        Release resources held by a node.

        TODO: Implement resource release
        TODO: Update placement group state
        """
        pass
