"""
Placement Group Manager for resource mapping and lifecycle.

Based on ARCHITECTURE.md Section 2.1.2

Responsibilities:
- Maintain SPMD ID → Placement Group mapping
- Resource pool management
- Execute preemption and eviction operations
- Track resource availability
- Support multi-bundle PG for driver + workers
"""

from typing import Dict, List, Optional, Any, Tuple
import logging

try:
    import ray
    from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    PlacementGroup = Any  # type: ignore

from ...common.types import ResourceSpec, PlacementStrategy, ResourcePriority, DriverResourceSpec

logger = logging.getLogger(__name__)


class PGManager:
    """
    Placement Group Manager.
    
    Manages the pool of Ray Placement Groups and their mappings to SPMDs.
    Handles resource allocation, preemption, and eviction at the PG level.
    
    Each SPMD's PG contains:
    - 1 driver bundle (index 0)
    - N worker bundles (indices 1 to worker_count)
    """
    
    def __init__(self):
        # SPMD ID → PlacementGroup mapping
        self._spmd_to_pg: Dict[str, PlacementGroup] = {}
        # SPMD ID → ResourceSpec mapping (single worker resource)
        self._spmd_to_resource: Dict[str, ResourceSpec] = {}
        # SPMD ID → ResourcePriority mapping
        self._spmd_to_priority: Dict[str, ResourcePriority] = {}
        # SPMD ID → worker count mapping
        self._spmd_to_worker_count: Dict[str, int] = {}
        # SPMD ID → driver resource mapping
        self._spmd_to_driver_resource: Dict[str, DriverResourceSpec] = {}
    
    def create_pg(
        self,
        spmd_id: str,
        resource: ResourceSpec,
        priority: ResourcePriority = ResourcePriority.MID,
        strategy: PlacementStrategy = PlacementStrategy.PACK,
        worker_count: int = 1,
        driver_resource: Optional[DriverResourceSpec] = None,
    ) -> Optional[PlacementGroup]:
        """
        Create a Placement Group for an SPMD with driver and worker bundles.
        
        The PG contains:
        - Bundle 0: Driver bundle
        - Bundle 1 to worker_count: Worker bundles
        
        Args:
            spmd_id: Unique SPMD identifier
            resource: Resource requirements for each worker
            priority: Resource priority for preemption decisions
            strategy: Placement strategy
            worker_count: Number of workers
            driver_resource: Optional driver resource spec (uses default if None)
            
        Returns:
            Created PlacementGroup or None if failed
        """
        driver_res = driver_resource or DriverResourceSpec()
        
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, returning mock placement group")
            self._spmd_to_resource[spmd_id] = resource
            self._spmd_to_priority[spmd_id] = priority
            self._spmd_to_worker_count[spmd_id] = worker_count
            self._spmd_to_driver_resource[spmd_id] = driver_res
            return None
        
        try:
            # Build bundles: driver bundle + worker bundles
            bundles = []
            
            # Bundle 0: Driver bundle
            driver_bundle = self._driver_resource_to_bundle(driver_res)
            bundles.append(driver_bundle)
            
            # Bundles 1 to worker_count: Worker bundles
            worker_bundle = self._resource_to_bundle(resource)
            for _ in range(worker_count):
                bundles.append(worker_bundle.copy())
            
            pg = placement_group(
                bundles=bundles,
                strategy=strategy.value
            )
            
            # Wait for placement group to be ready
            ray.get(pg.ready())
            
            # Store mappings
            self._spmd_to_pg[spmd_id] = pg
            self._spmd_to_resource[spmd_id] = resource
            self._spmd_to_priority[spmd_id] = priority
            self._spmd_to_worker_count[spmd_id] = worker_count
            self._spmd_to_driver_resource[spmd_id] = driver_res
            
            logger.info(
                f"Created placement group for SPMD {spmd_id}: "
                f"driver={driver_res}, worker={resource}, count={worker_count}"
            )
            return pg
            
        except Exception as e:
            logger.error(f"Failed to create placement group for SPMD {spmd_id}: {e}")
            return None
    
    def get_driver_bundle_index(self, spmd_id: str) -> int:
        """Get the bundle index for the driver (always 0)."""
        return 0
    
    def get_worker_bundle_indices(self, spmd_id: str) -> List[int]:
        """
        Get the bundle indices for workers.
        
        Returns:
            List of bundle indices [1, 2, ..., worker_count]
        """
        worker_count = self._spmd_to_worker_count.get(spmd_id, 1)
        return list(range(1, worker_count + 1))
    
    def get_worker_count(self, spmd_id: str) -> int:
        """Get the worker count for an SPMD."""
        return self._spmd_to_worker_count.get(spmd_id, 1)
    
    def get_pg(self, spmd_id: str) -> Optional[PlacementGroup]:
        """Get the PlacementGroup for an SPMD."""
        return self._spmd_to_pg.get(spmd_id)
    
    def get_resource(self, spmd_id: str) -> Optional[ResourceSpec]:
        """Get the resource allocation for an SPMD."""
        return self._spmd_to_resource.get(spmd_id)
    
    def get_priority(self, spmd_id: str) -> ResourcePriority:
        """Get the priority for an SPMD."""
        return self._spmd_to_priority.get(spmd_id, ResourcePriority.MID)
    
    def release_pg(self, spmd_id: str) -> bool:
        """
        Release the Placement Group for an SPMD.
        
        Args:
            spmd_id: SPMD identifier
            
        Returns:
            True if released successfully
        """
        if spmd_id not in self._spmd_to_pg and spmd_id not in self._spmd_to_resource:
            return False
        
        pg = self._spmd_to_pg.get(spmd_id)
        if pg is not None and RAY_AVAILABLE:
            try:
                remove_placement_group(pg)
                logger.info(f"Released placement group for SPMD {spmd_id}")
            except Exception as e:
                logger.error(f"Error releasing placement group for {spmd_id}: {e}")
        
        # Clean up mappings
        self._spmd_to_pg.pop(spmd_id, None)
        self._spmd_to_resource.pop(spmd_id, None)
        self._spmd_to_priority.pop(spmd_id, None)
        
        return True
    
    def preempt(self, spmd_id: str, reason: str) -> bool:
        """
        Execute preemption on an SPMD's resources.
        
        Args:
            spmd_id: SPMD to preempt
            reason: Reason for preemption
            
        Returns:
            True if preemption executed
            
        TODO: Coordinate with spmd_manager for state saving
        TODO: Actually implement preemption logic
        """
        logger.warning(f"preempt called for {spmd_id} (reason: {reason}) - not implemented")
        return False
    
    def evict(self, spmd_id: str) -> bool:
        """
        Execute eviction on an SPMD's resources.
        
        Args:
            spmd_id: SPMD to evict
            
        Returns:
            True if eviction executed
            
        TODO: Coordinate with spmd_manager for checkpoint
        TODO: Actually implement eviction logic
        """
        logger.warning(f"evict called for {spmd_id} - not implemented")
        return False
    
    def get_all_spmd_ids(self) -> List[str]:
        """Get all SPMD IDs with allocated resources."""
        return list(self._spmd_to_resource.keys())
    
    def get_preemptable_spmds(self, required_resource: ResourceSpec) -> List[str]:
        """
        Get SPMDs that could be preempted to free resources.
        
        Returns SPMDs sorted by priority (LOW first).
        
        Args:
            required_resource: Resource needed
            
        Returns:
            List of SPMD IDs that could be preempted
        """
        # Sort by priority: LOW < MID < HIGH
        priority_order = {
            ResourcePriority.LOW: 0,
            ResourcePriority.MID: 1,
            ResourcePriority.HIGH: 2
        }
        
        candidates = [
            (spmd_id, priority_order[self._spmd_to_priority.get(spmd_id, ResourcePriority.MID)])
            for spmd_id in self._spmd_to_resource.keys()
        ]
        
        # Sort by priority (ascending) - lower priority first
        candidates.sort(key=lambda x: x[1])
        
        return [spmd_id for spmd_id, _ in candidates]
    
    def _resource_to_bundle(self, resource: ResourceSpec) -> Dict[str, float]:
        """Convert ResourceSpec to Ray bundle specification."""
        bundle: Dict[str, float] = {}
        
        if resource.num_cpus > 0:
            bundle["CPU"] = float(resource.num_cpus)
        
        if resource.num_gpus > 0:
            bundle["GPU"] = float(resource.num_gpus)
        
        if resource.memory_gb > 0:
            bundle["memory"] = resource.memory_gb * 1024 * 1024 * 1024
        
        return bundle
    
    def _driver_resource_to_bundle(self, resource: DriverResourceSpec) -> Dict[str, float]:
        """Convert DriverResourceSpec to Ray bundle specification."""
        bundle: Dict[str, float] = {}
        
        if resource.num_cpus > 0:
            bundle["CPU"] = float(resource.num_cpus)
        
        if resource.memory_gb > 0:
            bundle["memory"] = resource.memory_gb * 1024 * 1024 * 1024
        
        return bundle
    
    def get_pg_info(self, spmd_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed placement group info for an SPMD.
        
        Returns:
            Dict with pg, driver_bundle_index, worker_bundle_indices, worker_count
        """
        if spmd_id not in self._spmd_to_resource:
            return None
        
        return {
            "pg": self._spmd_to_pg.get(spmd_id),
            "driver_bundle_index": self.get_driver_bundle_index(spmd_id),
            "worker_bundle_indices": self.get_worker_bundle_indices(spmd_id),
            "worker_count": self.get_worker_count(spmd_id),
            "worker_resource": self._spmd_to_resource.get(spmd_id),
            "driver_resource": self._spmd_to_driver_resource.get(spmd_id),
        }