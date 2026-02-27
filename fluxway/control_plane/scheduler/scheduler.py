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

from typing import List, Optional, Any, Dict, TYPE_CHECKING
import asyncio
import logging

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from fluxway.common.types import (
    Graph,
    ResourceSpec,
    PlacementPlan,
    PlacementStrategy,
    ResourcePriority,
)
from .pg_manager import PGManager

if TYPE_CHECKING:
    from ..spmd_manager import SPMDManager

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Resource scheduler using Ray Placement Groups.

    Holds references to:
    - spmd_manager: For SPMD lifecycle operations
    - pg_pool_manager: For resource pool management

    Scheduling strategies:
    - Resource-aware scheduling
    - Affinity scheduling (TODO: interface only)
    - Dynamic adjustment (TODO: interface only)
    - Conditional branch shared PG (TODO: interface only)
    """

    def __init__(self, spmd_manager: Optional["SPMDManager"] = None):
        """
        Initialize Scheduler.

        Args:
            spmd_manager: Reference to SPMDManager for SPMD lifecycle operations
        """
        self._spmd_manager = spmd_manager
        self._pg_manager = PGManager()
        self._current_plan: Optional[PlacementPlan] = None

    @property
    def spmd_manager(self) -> Optional["SPMDManager"]:
        """Get the SPMD manager reference."""
        return self._spmd_manager

    @property
    def pg_manager(self) -> PGManager:
        """Get the PG manager."""
        return self._pg_manager

    async def schedule(self, graph: Graph) -> bool:
        """
        Schedule resources and initialize SPMDs for the given graph.

        This method performs the complete scheduling workflow:
        1. Generate placement plan from graph
        2. Allocate resources via pg_pool_manager
        3. Build and establish SPMD topology (data structure only)
        4. Initialize SPMDs via spmd_manager (fire-and-forget)

        Note: Topology registration to drivers is NOT done here.
        It is triggered later by Controller when all drivers report ready
        via heartbeat (event-driven).

        Args:
            graph: The task graph to schedule

        Returns:
            True if scheduling succeeded, False otherwise
        """
        try:
            # Step 1: Generate placement plan
            plan = self._generate_plan(graph)
            self._current_plan = plan

            # Step 2: Allocate resources via pg_pool_manager
            if not await self.allocate_resources(graph):
                logger.error("Failed to allocate resources")
                return False

            # Step 3: Build and establish topology (data structure only,
            #         no remote calls to drivers yet)
            self.build_and_establish_topology(graph)

            # Step 4: Initialize SPMDs via spmd_manager (fire-and-forget)
            if not await self.init_spmds(graph):
                logger.error("Failed to initialize SPMDs")
                return False

            logger.debug(f"Successfully scheduled {len(plan.spmd_placements)} SPMDs")
            return True

        except Exception as e:
            logger.error(f"Scheduling failed: {e}")
            return False

    async def allocate_resources(self, graph: Graph) -> bool:
        """
        Allocate resources for all SPMDs in the graph.

        Public interface for the resource allocation phase.
        Must be called before init_spmds().

        Args:
            graph: The task graph

        Returns:
            True if all resources allocated successfully
        """
        if self._current_plan is None:
            self._current_plan = self._generate_plan(graph)
        return await self._allocate_resources(graph, self._current_plan)

    async def init_spmds(self, graph: Graph) -> bool:
        """
        Initialize SPMDs (fire-and-forget).

        Public interface for the SPMD initialization phase.
        Must be called after allocate_resources().
        Drivers will report readiness via heartbeat.

        Args:
            graph: The task graph

        Returns:
            True if SPMD init tasks were created successfully
        """
        if self._current_plan is None:
            self._current_plan = self._generate_plan(graph)
        return await self._init_spmds(graph, self._current_plan)

    def build_and_establish_topology(self, graph: Graph) -> None:
        """
        Build SPMD topology from graph edges and establish it in SPMDManager.

        This only builds the data structure (adjacency lists).
        No remote calls to drivers are made.
        Should be called before init_spmds() so that SPMDManager
        can provide predecessor/successor info during driver creation.

        Args:
            graph: The task graph
        """
        if self._spmd_manager is not None:
            topology = self._build_topology(graph)
            self._spmd_manager.establish_topology(topology)
            logger.debug(f"Topology established for {len(topology)} SPMDs")

    def _generate_plan(self, graph: Graph) -> PlacementPlan:
        """
        Generate resource allocation plan based on Graph.

        Args:
            graph: The task graph to schedule

        Returns:
            Placement plan for all SPMDs
        """
        spmd_placements: Dict[str, ResourceSpec] = {}
        spmd_priorities: Dict[str, ResourcePriority] = {}
        shared_pgs: Dict[str, List[str]] = {}

        for node_name, node in graph.nodes.items():
            spmd_placements[node_name] = (
                node.resource if node.resource is not None else ResourceSpec(num_cpus=1)
            )

            # Use node.priority if specified, otherwise default to MID
            spmd_priorities[node_name] = (
                node.priority if node.priority is not None else ResourcePriority.MID
            )

        # TODO: Analyze conditional branches for shared PG opportunities
        # TODO: Implement affinity rules based on graph edges

        return PlacementPlan(
            spmd_placements=spmd_placements,
            spmd_priorities=spmd_priorities,
            shared_pgs=shared_pgs,
            strategy=PlacementStrategy.PACK,
        )

    async def _allocate_resources(self, graph: Graph, plan: PlacementPlan) -> bool:
        """
        Allocate resources via pg_pool_manager.

        Creates multi-bundle PG for each SPMD:
        - 1 driver bundle
        - N worker bundles (based on reader_config.num_workers for Reader nodes,
          or spmd_config.num_workers for SPMD nodes)

        Args:
            graph: The task graph (for num_workers info)
            plan: The placement plan to execute

        Returns:
            True if all resources allocated successfully
        """
        success = True

        for spmd_id, resource in plan.spmd_placements.items():
            priority = plan.spmd_priorities.get(spmd_id, ResourcePriority.MID)

            # Get worker count from graph node
            node = graph.nodes.get(spmd_id)
            num_workers = 1
            if node and node.spmd_config:
                num_workers = node.spmd_config.num_workers

            pg = await self._pg_manager.create_pg(
                spmd_id=spmd_id,
                resource=resource,
                priority=priority,
                strategy=plan.strategy,
                num_workers=num_workers,
            )

            if pg is None and RAY_AVAILABLE:
                logger.warning(f"Failed to create PG for SPMD {spmd_id}")
                success = False

        return success

    async def _init_spmds(self, graph: Graph, plan: PlacementPlan) -> bool:
        """
        Initialize SPMDs via spmd_manager.

        Passes PG info including worker bundle indices to SPMDManager.

        Args:
            graph: The task graph
            plan: The placement plan

        Returns:
            True if all SPMDs initialized successfully
        """
        if self._spmd_manager is None:
            logger.warning("No spmd_manager set, skipping SPMD initialization")
            return True

        # Start all SPMDs concurrently to reduce init latency.
        # Fire-and-forget: we don't wait for individual start_spmd() completions here.
        # Driver/worker readiness is tracked via their own heartbeats / FSM.
        for node_name, node in graph.nodes.items():
            spmd_config = node.spmd_config
            if spmd_config is None:
                continue

            # Get the allocated PG info for this SPMD
            pg_info = self._pg_manager.get_pg_info(node_name)

            # Start SPMD via spmd_manager, passing PG and bundle info
            # Registration happens inside start_spmd
            asyncio.create_task(
                self._spmd_manager.start_spmd(
                    spmd_id=node_name,
                    config=spmd_config,
                    placement_group=pg_info.get("pg") if pg_info else None,
                    worker_bundle_indices=pg_info.get("worker_bundle_indices")
                    if pg_info
                    else None,
                    driver_bundle_index=pg_info.get("driver_bundle_index", 0)
                    if pg_info
                    else 0,
                    checkpoint_id=None,
                )
            )

        return True

    def _build_topology(self, graph: Graph) -> Dict[str, List[str]]:
        """
        Build SPMD topology from graph edges.

        Args:
            graph: The task graph

        Returns:
            Mapping of node name to list of downstream node names
        """
        topology: Dict[str, List[str]] = {name: [] for name in graph.nodes}
        for edge in graph.edges:
            if edge.source_node in topology:
                if edge.target_node not in topology[edge.source_node]:
                    topology[edge.source_node].append(edge.target_node)
        return topology

    def create_shared_pg(
        self,
        spmd_ids: List[str],
        resources: ResourceSpec,
        priority: ResourcePriority = ResourcePriority.MID,
    ) -> Any:
        """
        Create shared Placement Group for conditional branch SPMDs.

        Args:
            spmd_ids: IDs of SPMDs sharing the PG
            resources: Resource specification (maximum required)
            priority: Resource priority

        Returns:
            PlacementGroup or None

        TODO: Implement shared PG creation
        """
        logger.debug(f"create_shared_pg called for SPMDs {spmd_ids} (not implemented)")
        return None

    def preempt_spmd(self, spmd_id: str, reason: str) -> bool:
        """
        Preempt SPMD resources.

        Args:
            spmd_id: ID of SPMD to preempt
            reason: Reason for preemption

        Returns:
            True if preemption successful
        """
        return self._pg_manager.preempt(spmd_id, reason)

    def evict_spmd(self, spmd_id: str, checkpoint: bool = True) -> bool:
        """
        Evict SPMD (save state before eviction).

        Args:
            spmd_id: ID of SPMD to evict
            checkpoint: Whether to save checkpoint before eviction

        Returns:
            True if eviction successful
        """
        if checkpoint and self._spmd_manager is not None:
            # TODO: Trigger checkpoint via spmd_manager
            pass

        return self._pg_manager.evict(spmd_id)

    def release_resources(self, spmd_id: str) -> None:
        """
        Release resources held by an SPMD.

        Args:
            spmd_id: ID of the SPMD
        """
        self._pg_manager.release_pg(spmd_id)

    def get_allocated_resources(self, spmd_id: str) -> Optional[ResourceSpec]:
        """
        Get the resources allocated to an SPMD.

        Args:
            spmd_id: ID of the SPMD

        Returns:
            ResourceSpec or None if not allocated
        """
        return self._pg_manager.get_resource(spmd_id)

    def get_spmd_priority(self, spmd_id: str) -> ResourcePriority:
        """
        Get priority of an SPMD for preemption decisions.

        Args:
            spmd_id: ID of the SPMD

        Returns:
            ResourcePriority value
        """
        return self._pg_manager.get_priority(spmd_id)

    def check_resource_availability(self, resources: ResourceSpec) -> bool:
        """
        Check if resources are available in the Ray cluster.

        Args:
            resources: Required resources

        Returns:
            True if resources are available
        """
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, assuming resources are available")
            return True

        try:
            available = ray.available_resources()

            if resources.num_gpus > 0:
                available_gpus = available.get("GPU", 0)
                if available_gpus < resources.num_gpus:
                    return False

            if resources.num_cpus > 0:
                available_cpus = available.get("CPU", 0)
                if available_cpus < resources.num_cpus:
                    return False

            if resources.memory_gb > 0:
                available_memory = available.get("memory", 0)
                required_memory = resources.memory_gb * 1024 * 1024 * 1024
                if available_memory < required_memory:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking resource availability: {e}")
            return False

    def get_preemptable_spmds(self, required_resource: ResourceSpec) -> List[str]:
        """
        Get SPMDs that could be preempted to free resources.

        Args:
            required_resource: Resource needed

        Returns:
            List of SPMD IDs sorted by priority (LOW first)
        """
        return self._pg_manager.get_preemptable_spmds(required_resource)
