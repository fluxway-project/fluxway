"""
Controller - Central coordination node for the Control Plane.

Based on ARCHITECTURE.md Section 2.1

The Controller orchestrates the training job by coordinating:
- Monitor: Cluster state monitoring and telemetry
- Scheduler: Resource allocation via Ray Placement Groups
- SPMDManager: SPMD lifecycle management
- StateManager: Global state and checkpointing
"""

from typing import Dict, List, Optional, Any

from ..common.types import Graph
from .monitor import Monitor
from .scheduler import Scheduler
from .spmd_manager import SPMDManager, SPMDHandle
from .state_manager import StateManager


class Controller:
    """
    Central Controller for the Fluxway Control Plane.
    
    Coordinates all components for distributed training orchestration:
    - Initializes and manages the training graph
    - Allocates resources via Scheduler
    - Manages SPMD instances via SPMDManager
    - Tracks state via StateManager
    - Monitors cluster health via Monitor
    
    Note: Resource-related operations are proxied through Scheduler.
    Controller can get SPMD view from spmd_manager but without resource info.
    """
    
    def __init__(
        self,
        graph: Graph,
        monitor: Monitor,
        scheduler: Scheduler,
        spmd_manager: SPMDManager,
        state_manager: StateManager,
    ):
        """
        Initialize Controller with all components.
        
        The scheduler is initialized with spmd_manager reference for
        SPMD-resource binding operations.
        
        Use from_graph() class method for standard initialization.
        
        Args:
            graph: The task graph defining SPMD nodes and edges
            monitor: Cluster state monitor
            scheduler: Resource scheduler (holds spmd_manager reference)
            spmd_manager: SPMD lifecycle manager
            state_manager: State and checkpoint manager
        """
        self._graph = graph
        self._monitor = monitor
        self._scheduler = scheduler
        self._spmd_manager = spmd_manager
        self._state_manager = state_manager
    
    @classmethod
    def from_graph(cls, graph: Graph) -> "Controller":
        """
        Initialize Controller from a Graph.
        
        Initialization flow:
        1. Create all components (spmd_manager, state_manager, monitor)
        2. Create scheduler with spmd_manager reference
        3. Call scheduler.schedule() which handles:
           - Resource allocation
           - SPMD initialization
           - Topology establishment
        
        Args:
            graph: The task graph built using Builder
            
        Returns:
            Initialized Controller instance
            
        Raises:
            RuntimeError: If scheduling fails
        """
        # Step 1: Create all components
        spmd_manager = SPMDManager()
        state_manager = StateManager()
        monitor = Monitor()
        
        # Step 2: Create scheduler with spmd_manager reference
        scheduler = Scheduler(spmd_manager=spmd_manager)
        
        # Step 3: Schedule - handles resource allocation, SPMD init, and topology
        success = scheduler.schedule(graph)
        if not success:
            raise RuntimeError("Failed to schedule graph")
        
        return cls(
            graph=graph,
            monitor=monitor,
            scheduler=scheduler,
            spmd_manager=spmd_manager,
            state_manager=state_manager,
        )
    
    @property
    def graph(self) -> Graph:
        """Get the task graph."""
        return self._graph
    
    @property
    def monitor(self) -> Monitor:
        """Get the monitor component."""
        return self._monitor
    
    @property
    def scheduler(self) -> Scheduler:
        """Get the scheduler component."""
        return self._scheduler
    
    @property
    def spmd_manager(self) -> SPMDManager:
        """Get the SPMD manager component."""
        return self._spmd_manager
    
    @property
    def state_manager(self) -> StateManager:
        """Get the state manager component."""
        return self._state_manager
    
    def get_spmd_handle(self, spmd_id: str) -> Optional[SPMDHandle]:
        """
        Get handle to an SPMD instance from spmd_manager.
        
        Note: This returns the handle without resource info.
        Use scheduler for resource-related operations.
        
        Args:
            spmd_id: ID of the SPMD
            
        Returns:
            SPMDHandle or None if not found
        """
        return self._spmd_manager.get_spmd_handle(spmd_id)
    
    def start(self) -> None:
        """
        Start training execution.
        
        TODO: Implement execution loop
        TODO: Start all SPMDs
        TODO: Begin data flow
        """
        pass
    
    def stop(self, save_checkpoint: bool = True) -> None:
        """
        Stop training execution.
        
        Args:
            save_checkpoint: Whether to save checkpoint before stopping
            
        TODO: Stop all SPMDs gracefully
        TODO: Save final checkpoint
        """
        pass
    
    def get_cluster_status(self) -> Dict[str, str]:
        """
        Get status of all SPMD nodes.
        
        Returns:
            Mapping of node name to status
        """
        return self._monitor.get_cluster_health()
