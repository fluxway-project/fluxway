"""
SPMD Manager for global SPMD lifecycle management.

Based on ARCHITECTURE.md Section 2.1.3

Responsibilities:
- SPMD lifecycle: Start SPMD Driver based on scheduling results
- SPMD topology: Establish connections between SPMDs
- SPMD initialization: Initialize SPMD state and internal FSM
- State recovery: Direct or FSM-triggered checkpoint recovery
- Node changes: Manage node creation, eviction, and rebuilding

Note: Controller layer only starts drivers. Workers are started by drivers.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from ...common.types import SPMDConfig

logger = logging.getLogger(__name__)


class SPMDHandle:
    """
    Handle to an SPMD instance.

    Contains reference to the driver actor only.
    Worker handles are managed internally by the driver.
    """

    def __init__(self, spmd_id: str):
        self.spmd_id = spmd_id
        self._driver_handle: Optional[Any] = None  # ray.ActorHandle
        self._config: Optional[SPMDConfig] = None
        self._is_initialized: bool = False

    @property
    def driver_handle(self) -> Optional[Any]:
        """Get the driver actor handle."""
        return self._driver_handle

    @property
    def is_initialized(self) -> bool:
        """Check if the SPMD is initialized."""
        return self._is_initialized

    def set_driver_handle(self, handle: Any) -> None:
        """Set the driver actor handle."""
        self._driver_handle = handle

    def set_initialized(self, initialized: bool = True) -> None:
        """Mark the SPMD as initialized."""
        self._is_initialized = initialized

    def call_driver(self, method: str, *args, **kwargs) -> Any:
        """
        Call a method on the driver.

        Args:
            method: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Ray ObjectRef for the method call result
        """
        if self._driver_handle is None:
            raise RuntimeError(f"Driver handle not set for SPMD {self.spmd_id}")

        if not hasattr(self._driver_handle, method):
            raise AttributeError(f"Driver does not have method: {method}")

        method_fn = getattr(self._driver_handle, method)
        return method_fn.remote(*args, **kwargs)


class SPMDManager:
    """
    Global SPMD Manager (Orchestrator).

    Manages the lifecycle of all SPMD drivers in the training job.
    Workers are managed by their respective drivers.
    """

    def __init__(self, controller_handle: Optional[Any] = None):
        """
        Initialize SPMDManager.

        Args:
            controller_handle: Reference to the Controller (for driver callbacks)
        """
        self._controller_handle = controller_handle
        self._spmd_handles: Dict[str, SPMDHandle] = {}
        self._topology: Dict[str, List[str]] = {}  # spmd_id -> downstream spmd_ids
        self._reverse_topology: Dict[
            str, List[str]
        ] = {}  # spmd_id -> upstream spmd_ids

    def set_controller_handle(self, controller_handle: Any) -> None:
        """Set the controller handle for driver callbacks."""
        self._controller_handle = controller_handle

    def register_spmd(self, spmd_id: str, handle: SPMDHandle) -> None:
        """
        Register an SPMD instance.

        Args:
            spmd_id: SPMD identifier
            handle: SPMD handle
        """
        if spmd_id in self._spmd_handles:
            logger.warning(f"SPMD {spmd_id} already registered, overwriting")

        self._spmd_handles[spmd_id] = handle
        logger.info(f"Registered SPMD: {spmd_id}")

    async def start_spmd(
        self,
        spmd_id: str,
        config: SPMDConfig,
        placement_group: Optional[Any] = None,
        worker_bundle_indices: Optional[List[int]] = None,
        driver_bundle_index: int = 0,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[SPMDHandle]:
        """
        Start an SPMD instance by creating its driver.

        The driver will start workers internally using the provided
        placement group and bundle indices.

        Args:
            spmd_id: Unique identifier for the SPMD
            config: SPMD configuration
            placement_group: Ray PlacementGroup for resource allocation
            worker_bundle_indices: Bundle indices for workers [1, 2, ..., n]
            driver_bundle_index: Bundle index for driver (default 0)
            checkpoint_id: Optional checkpoint to restore from
            node_type: Type of node (SPMD or READER)
            reader_config: Reader configuration (required for READER nodes)

        Returns:
            Handle to the started SPMD, or None on failure
        """
        # Create handle
        handle = SPMDHandle(spmd_id)
        handle._config = config

        if not RAY_AVAILABLE:
            logger.warning(
                f"Ray not available, creating mock SPMD handle for {spmd_id}"
            )
            self._spmd_handles[spmd_id] = handle
            return handle

        try:
            # Import driver here to avoid circular imports
            from ...spmd.driver import SPMDDriver

            # Build graph topology for this SPMD (self-loops are already filtered)
            graph_topology = {
                "predecessors": self.get_predecessors(spmd_id),
                "successors": self.get_successors(spmd_id),
            }

            # Create driver actor with placement group
            driver_name = f"spmd_driver_{spmd_id}"

            if placement_group is not None:
                driver = SPMDDriver.options(
                    placement_group=placement_group,
                    placement_group_bundle_index=driver_bundle_index,
                    name=driver_name,
                ).remote(
                    spmd_id=spmd_id,
                    config=config,
                    placement_group=placement_group,
                    worker_bundle_indices=worker_bundle_indices
                    or list(range(1, config.num_workers + 1)),
                    controller_handle=self._controller_handle,
                    graph_topology=graph_topology,
                )
            else:
                # No placement group - for testing or mock scenarios
                driver = SPMDDriver.remote(
                    spmd_id=spmd_id,
                    config=config,
                    placement_group=None,
                    worker_bundle_indices=worker_bundle_indices
                    or list(range(1, config.num_workers + 1)),
                    controller_handle=self._controller_handle,
                    graph_topology=graph_topology,
                )

            handle.set_driver_handle(driver)

            driver.initialize.remote(checkpoint_id)

            # Register handle
            self._spmd_handles[spmd_id] = handle
            return handle

        except Exception as e:
            logger.error(f"Failed to start SPMD {spmd_id}: {e}")
            return None

    async def stop_spmd(self, spmd_id: str, save_checkpoint: bool = True) -> None:
        """
        Stop an SPMD instance.

        Args:
            spmd_id: ID of SPMD to stop
            save_checkpoint: Whether to save checkpoint before stopping
        """
        handle = self._spmd_handles.get(spmd_id)
        if handle is None:
            logger.warning(f"SPMD {spmd_id} not found")
            return

        if handle.driver_handle is None:
            logger.warning(f"SPMD {spmd_id} has no driver handle")
            return

        try:
            if RAY_AVAILABLE:
                # Call stop on driver
                await handle.call_driver("stop", save_checkpoint)

                # Kill the driver actor
                ray.kill(handle.driver_handle)

            handle.set_initialized(False)
            logger.info(f"Stopped SPMD {spmd_id}")

        except Exception as e:
            logger.error(f"Error stopping SPMD {spmd_id}: {e}")

    def get_spmd_handle(self, spmd_id: str) -> Optional[SPMDHandle]:
        """
        Get handle to an SPMD instance.

        Args:
            spmd_id: SPMD identifier

        Returns:
            SPMDHandle or None if not found
        """
        return self._spmd_handles.get(spmd_id)

    def get_driver_handle(self, spmd_id: str) -> Optional[Any]:
        """
        Get the driver actor handle for an SPMD.

        Args:
            spmd_id: SPMD identifier

        Returns:
            Driver ActorHandle or None
        """
        handle = self._spmd_handles.get(spmd_id)
        if handle is None:
            return None
        return handle.driver_handle

    def get_all_spmd_ids(self) -> List[str]:
        """Get all registered SPMD IDs."""
        return list(self._spmd_handles.keys())

    def evict_spmd(self, spmd_id: str) -> Optional[str]:
        """
        Evict an SPMD and return checkpoint ID.

        Args:
            spmd_id: ID of SPMD to evict

        Returns:
            Checkpoint ID for later restoration, or None on failure

        TODO: Implement checkpoint saving
        """
        logger.warning(f"evict_spmd called for {spmd_id} - checkpoint not implemented")
        self.stop_spmd(spmd_id, save_checkpoint=True)
        return None

    def restore_spmd(
        self,
        spmd_id: str,
        config: SPMDConfig,
        placement_group: Optional[Any] = None,
        worker_bundle_indices: Optional[List[int]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[SPMDHandle]:
        """
        Restore an SPMD from checkpoint.

        Args:
            spmd_id: ID of SPMD to restore
            config: SPMD configuration
            placement_group: Placement group for resources
            worker_bundle_indices: Worker bundle indices
            checkpoint_id: Checkpoint to restore from
            node_type: Type of node (SPMD or READER)
            reader_config: Reader configuration (for READER nodes)

        Returns:
            Handle to restored SPMD

        TODO: Implement checkpoint loading
        """
        logger.warning(
            f"restore_spmd called for {spmd_id} - checkpoint not implemented"
        )
        return self.start_spmd(
            spmd_id=spmd_id,
            config=config,
            placement_group=placement_group,
            worker_bundle_indices=worker_bundle_indices,
            checkpoint_id=checkpoint_id,
        )

    def establish_topology(self, topology: Dict[str, List[str]]) -> None:
        """
        Establish SPMD topology connections.

        Args:
            topology: Mapping of SPMD ID to downstream SPMD IDs
        """
        # Filter out self-loops when building topology
        self._topology = {}
        for source, targets in topology.items():
            # Filter out self-loops (where target == source)
            self._topology[source] = [target for target in targets if target != source]

        # Build reverse topology (also filtering self-loops)
        self._reverse_topology = {spmd_id: [] for spmd_id in topology}
        for source, targets in topology.items():
            for target in targets:
                # Skip self-loops
                if target == source:
                    continue
                if target not in self._reverse_topology:
                    self._reverse_topology[target] = []
                if source not in self._reverse_topology[target]:
                    self._reverse_topology[target].append(source)

        logger.info(f"Established topology with {len(topology)} SPMDs")

        # Notify existing drivers of topology update
        if RAY_AVAILABLE:
            for spmd_id, handle in self._spmd_handles.items():
                ## TODO(Chiave): Must ensure all drivers are is_initialized before registering predecessor/successor nodes
                if handle.driver_handle is not None and handle.is_initialized:
                    try:
                        predecessors = self._reverse_topology.get(spmd_id, [])
                        successors = self._topology.get(spmd_id, [])
                        handle.call_driver("update_topology", predecessors, successors)
                    except Exception as e:
                        logger.error(f"Failed to update topology for {spmd_id}: {e}")

    def get_predecessors(self, spmd_id: str) -> List[str]:
        """
        Get predecessor SPMD IDs.

        Returns list excluding self-loops (where predecessor == spmd_id).
        """
        predecessors = self._reverse_topology.get(spmd_id, [])
        # Filter out self-loops as a safety measure
        return [p for p in predecessors if p != spmd_id]

    def get_successors(self, spmd_id: str) -> List[str]:
        """
        Get successor SPMD IDs.

        Returns list excluding self-loops (where successor == spmd_id).
        """
        successors = self._topology.get(spmd_id, [])
        # Filter out self-loops as a safety measure
        return [s for s in successors if s != spmd_id]

    async def broadcast_to_all_drivers(
        self, method: str, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Broadcast a method call to all drivers.

        Args:
            method: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Dict mapping spmd_id to result (or exception)
        """
        results = {}
        refs = {}

        for spmd_id, handle in self._spmd_handles.items():
            if handle.driver_handle is not None and handle.is_initialized:
                try:
                    ref = handle.call_driver(method, *args, **kwargs)
                    refs[spmd_id] = ref
                except Exception as e:
                    results[spmd_id] = e

        if RAY_AVAILABLE and refs:
            spmd_ids = list(refs.keys())
            obj_refs = [refs[s] for s in spmd_ids]
            gathered = await asyncio.gather(*obj_refs, return_exceptions=True)
            for spmd_id, result in zip(spmd_ids, gathered, strict=False):
                results[spmd_id] = result

        return results
