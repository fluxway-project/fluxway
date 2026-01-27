"""
Controller - Central coordination node for the Control Plane.

Based on ARCHITECTURE.md Section 2.1

The Controller orchestrates the training job by coordinating:
- Monitor: Cluster state monitoring and telemetry
- Scheduler: Resource allocation via Ray Placement Groups
- SPMDManager: SPMD lifecycle management
- StateManager: Global state and checkpointing
"""

from typing import Dict, List, Optional, Any, Type

from ..common.types import Graph, Node, ResourceSpec, PlacementPlan, SPMDConfig
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
    """

    def __init__(
        self,
        graph: Graph,
        monitor: Monitor,
        scheduler: Scheduler,
        spmd_manager: SPMDManager,
        state_manager: StateManager,
        spmd_handles: Dict[str, SPMDHandle],
    ):
        """
        Initialize Controller with all components.

        Use from_graph() class method for standard initialization.

        Args:
            graph: The task graph defining SPMD nodes and edges
            monitor: Cluster state monitor
            scheduler: Resource scheduler
            spmd_manager: SPMD lifecycle manager
            state_manager: State and checkpoint manager
            spmd_handles: Handles to initialized SPMD instances
        """
        self._graph = graph
        self._monitor = monitor
        self._scheduler = scheduler
        self._spmd_manager = spmd_manager
        self._state_manager = state_manager
        self._spmd_handles = spmd_handles

    @classmethod
    def from_graph(cls, graph: Graph) -> "Controller":
        """
        Initialize Controller from a Graph.

        Initialization flow:
        1. Initialize scheduler
        2. Request placement groups for each Node based on resource requirements
        3. Initialize SPMDs (allocate resources to master and workers)
        4. Initialize spmd_manager, state_manager, and monitor

        Args:
            graph: The task graph built using Builder

        Returns:
            Initialized Controller instance
        """
        # Step 1: Initialize scheduler
        scheduler = Scheduler()

        # Step 2: Request placement groups for each Node
        # For now, allocate resources per node without sharing/eviction/preemption
        placement_groups: Dict[str, Any] = {}
        for node_name, node in graph.nodes.items():
            resource = node.resource
            if resource is not None:
                # Create placement group for this node
                pg = scheduler.allocate_resources(
                    PlacementPlan(node_placements={node_name: resource})
                )
                placement_groups[node_name] = pg

        # Step 3: Initialize SPMDs
        # For now, just create handles and assign allocated resources
        spmd_handles: Dict[str, SPMDHandle] = {}
        for node_name, node in graph.nodes.items():
            spmd_config = node.spmd_config
            if spmd_config is None:
                continue

            # Create SPMD handle
            handle = SPMDHandle(spmd_id=node_name)

            # Assign placement group resources (master and workers)
            # The actual SPMD initialization logic will be implemented later
            if node_name in placement_groups:
                handle._placement_group = placement_groups[node_name]

            spmd_handles[node_name] = handle

        # Step 4: Initialize remaining components
        spmd_manager = SPMDManager()
        for spmd_id, handle in spmd_handles.items():
            spmd_manager.register_spmd(spmd_id, handle)

        # Establish topology from graph edges
        topology = cls._build_topology(graph)
        spmd_manager.establish_topology(topology)

        state_manager = StateManager()
        # Initialize Monitor with backends (can be configured)
        monitor = Monitor(
            enable_tensorboard=True,
            enable_ray=True,
            enable_prometheus=False,  # Disabled by default
            enable_alert=True,
        )

        return cls(
            graph=graph,
            monitor=monitor,
            scheduler=scheduler,
            spmd_manager=spmd_manager,
            state_manager=state_manager,
            spmd_handles=spmd_handles,
        )

    @staticmethod
    def _build_topology(graph: Graph) -> Dict[str, List[str]]:
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

    def get_spmd_handle(self, node_name: str) -> Optional[SPMDHandle]:
        """
        Get handle to an SPMD instance.

        Args:
            node_name: Name of the node/SPMD

        Returns:
            SPMDHandle or None if not found
        """
        return self._spmd_handles.get(node_name)

    def start(self) -> None:
        """
        Start training execution.

        Starts monitoring daemon and begins execution loop.
        """
        # Register Monitor with MonitoringFacade
        try:
            from ..common.monitoring import MonitoringFacade

            MonitoringFacade.register_main_monitor(self._monitor)
        except Exception:
            pass

        # Register SPMD handles with Monitor
        for spmd_id, handle in self._spmd_handles.items():
            self._monitor.register_spmd_handle(spmd_id, handle)

        # Start monitoring daemon
        try:
            from .monitor.daemon import MonitorDaemon

            # Create daemon with failure callback
            def failure_handler(spmd_id: str, reason: str):
                """Handle SPMD failures detected by daemon."""
                # Log failure
                # In production, can trigger recovery actions or alerts
                print(f"SPMD {spmd_id} failed: {reason}")

            self._monitor_daemon = MonitorDaemon(
                monitor=self._monitor,
                probe_interval=30.0,  # Probe every 30 seconds
                health_check_interval=10.0,  # Health check every 10 seconds
                failure_callback=failure_handler,
            )

            # Start daemon in background
            self._monitor_daemon.start()
        except Exception:
            # If daemon fails to start, continue without it
            # In production, should log the error
            self._monitor_daemon = None

        # TODO: Implement execution loop
        # TODO: Start all SPMDs
        # TODO: Begin data flow

    def stop(self, save_checkpoint: bool = True) -> None:
        """
        Stop training execution.

        Args:
            save_checkpoint: Whether to save checkpoint before stopping

        Stops monitoring daemon and performs graceful shutdown.
        """
        # Stop monitoring daemon
        if hasattr(self, "_monitor_daemon") and self._monitor_daemon is not None:
            try:
                self._monitor_daemon.stop()
            except Exception:
                pass

        # Monitor final checkpoint save attempt
        if save_checkpoint:
            from fluxway.common.monitoring import MonitoringFacade
            import time

            start_time = time.time()

            MonitoringFacade.record_checkpoint_event(
                event_type="final_save_attempt",
                checkpoint_id="final",
                component_type="task",
                component_id="task",
                status="not_implemented",
            )

            # TODO: Stop all SPMDs gracefully
            # TODO: Save final checkpoint

            duration = time.time() - start_time
            MonitoringFacade.record_checkpoint_event(
                event_type="final_save_complete",
                checkpoint_id="final",
                component_type="task",
                component_id="task",
                status="not_implemented",
                duration_seconds=duration,
            )

    def get_cluster_status(self) -> Dict[str, str]:
        """
        Get status of all SPMD nodes.

        Returns:
            Mapping of node name to status
        """
        return self._monitor.get_cluster_health()
