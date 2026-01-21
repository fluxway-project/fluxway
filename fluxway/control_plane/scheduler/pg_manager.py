"""
Placement Group Manager for resource mapping and lifecycle.

Based on ARCHITECTURE.md Section 2.1.2

Responsibilities:
- Maintain SPMD ID → Placement Group mapping
- Resource pool management
- Execute preemption and eviction operations
- Track resource availability
"""

from typing import Dict, List, Optional, Any
import logging

try:
    import ray
    from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    PlacementGroup = Any  # type: ignore

from ...common.types import ResourceSpec, PlacementStrategy, ResourcePriority

logger = logging.getLogger(__name__)


class PGManager:
    """
    Placement Group Manager.
    
    Manages the pool of Ray Placement Groups and their mappings to SPMDs.
    Handles resource allocation, preemption, and eviction at the PG level.
    """
    
    def __init__(self):
        # SPMD ID → PlacementGroup mapping
        self._spmd_to_pg: Dict[str, PlacementGroup] = {}
        # SPMD ID → ResourceSpec mapping
        self._spmd_to_resource: Dict[str, ResourceSpec] = {}
        # SPMD ID → ResourcePriority mapping
        self._spmd_to_priority: Dict[str, ResourcePriority] = {}
    
    def create_pg(
        self,
        spmd_id: str,
        resource: ResourceSpec,
        priority: ResourcePriority = ResourcePriority.MID,
        strategy: PlacementStrategy = PlacementStrategy.PACK
    ) -> Optional[PlacementGroup]:
        """
        Create a Placement Group for an SPMD.
        
        Args:
            spmd_id: Unique SPMD identifier
            resource: Resource requirements
            priority: Resource priority for preemption decisions
            strategy: Placement strategy
            
        Returns:
            Created PlacementGroup or None if failed
        """
        if not RAY_AVAILABLE:
            logger.warning("Ray not available, returning mock placement group")
            self._spmd_to_resource[spmd_id] = resource
            self._spmd_to_priority[spmd_id] = priority
            return None
        
        try:
            bundle = self._resource_to_bundle(resource)
            pg = placement_group(
                bundles=[bundle],
                strategy=strategy.value
            )
            
            # Wait for placement group to be ready
            ray.get(pg.ready())
            
            # Store mappings
            self._spmd_to_pg[spmd_id] = pg
            self._spmd_to_resource[spmd_id] = resource
            self._spmd_to_priority[spmd_id] = priority
            
            logger.info(f"Created placement group for SPMD {spmd_id}: {resource}")
            return pg
            
        except Exception as e:
            logger.error(f"Failed to create placement group for SPMD {spmd_id}: {e}")
            return None
    
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
