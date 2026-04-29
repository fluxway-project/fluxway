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

import asyncio
from typing import Any

try:
    import ray  # noqa: F401

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from fluxway.common.types import SPMDConfig
from fluxway.common.utils.logging import get_logger

logger = get_logger(__name__)


class SPMDHandle:
    """
    Handle to an SPMD instance.

    Contains reference to the driver actor only.
    Worker handles are managed internally by the driver.
    """

    def __init__(self, spmd_id: str):
        self.spmd_id = spmd_id
        self._driver_handle: Any | None = None  # ray.ActorHandle
        self._config: SPMDConfig | None = None
        self._is_initialized: bool = False

    @property
    def driver_handle(self) -> Any | None:
        """Get the driver actor handle."""
        return self._driver_handle

    @property
    def is_initialized(self) -> bool:
        """Check if the SPMD is initialized."""
        return self._is_initialized

    def set_driver_handle(self, handle: Any | None) -> None:
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

        method_fn: Any = getattr(self._driver_handle, method, None)
        if method_fn is None:
            raise AttributeError(f"Driver does not have method: {method}")

        return method_fn.remote(*args, **kwargs)


class SPMDManager:
    """
    Global SPMD Manager (Orchestrator).

    Manages the lifecycle of all SPMD drivers in the training job.
    Workers are managed by their respective drivers.
    """

    def __init__(self, controller_handle: Any | None = None):
        """
        Initialize SPMDManager.

        Args:
            controller_handle: Reference to the Controller (for driver callbacks)
        """
        self._controller_handle = controller_handle
        self._spmd_handles: dict[str, SPMDHandle] = {}
        self._topology: dict[str, list[str]] = {}  # spmd_id -> downstream spmd_ids
        self._reverse_topology: dict[str, list[str]] = (
            {}
        )  # spmd_id -> upstream spmd_ids
        self._checkpoint_backend: Any | None = None

    def set_controller_handle(self, controller_handle: Any) -> None:
        """Set the controller handle for driver callbacks."""
        self._controller_handle = controller_handle

    def set_checkpoint_backend(self, backend: Any) -> None:
        """Set the per-worker checkpoint backend to forward to all Drivers."""
        self._checkpoint_backend = backend

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
        placement_group: Any | None = None,
        worker_bundle_indices: list[int] | None = None,
        driver_bundle_index: int = 0,
        checkpoint_id: str | None = None,
    ) -> SPMDHandle | None:
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
            from fluxway.spmd.driver import SPMDDriver

            # Case to Any to satisfy static type checkers for Ray-specific calls (.remote, .options)
            spmd_driver_cls: Any = SPMDDriver

            # Build graph topology for this SPMD (self-loops are already filtered)
            graph_topology = {
                "predecessors": self.get_predecessors(spmd_id),
                "successors": self.get_successors(spmd_id),
            }

            # Create driver actor with placement group
            driver_name = f"spmd_driver_{spmd_id}"
            driver_opts: dict[str, Any] = {"name": driver_name}
            if placement_group is not None:
                driver_opts["placement_group"] = placement_group
                driver_opts["placement_group_bundle_index"] = driver_bundle_index
            if config.runtime_env:
                driver_opts["runtime_env"] = config.runtime_env

            driver = spmd_driver_cls.options(**driver_opts).remote(
                spmd_id=spmd_id,
                config=config,
                placement_group=placement_group,
                worker_bundle_indices=worker_bundle_indices
                or list(range(1, config.num_workers + 1)),
                controller_handle=self._controller_handle,
                graph_topology=graph_topology,
                checkpoint_backend=self._checkpoint_backend,
            )

            handle.set_driver_handle(driver)

            driver.initialize.remote(checkpoint_id)

            # Register handle
            self._spmd_handles[spmd_id] = handle
            return handle

        except Exception as e:
            logger.error(f"Failed to start SPMD {spmd_id}: {e}")
            return None

    async def stop_spmd(
        self,
        spmd_id: str,
        save_checkpoint: bool = True,
        release_driver_handle: bool = True,
    ) -> None:
        """
        Stop an SPMD instance.

        Args:
            spmd_id: ID of SPMD to stop
            save_checkpoint: Whether to save checkpoint before stopping
            release_driver_handle: Whether to clear the controller-side driver
                reference immediately after stop returns. Terminal cleanup keeps
                handles until the controller reaches its terminal state so Ray
                bookkeeping can settle before actor references are dropped.
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
                # Stop the driver and wait for its worker shutdown path.
                await handle.call_driver("stop", save_checkpoint)
        except Exception as e:
            logger.error(f"Error stopping SPMD {spmd_id}: {e}")
        finally:
            if release_driver_handle:
                handle.set_driver_handle(None)
            handle.set_initialized(False)
            logger.info(f"Stopped SPMD {spmd_id}")

    def release_driver_handle(self, spmd_id: str) -> None:
        """Release the controller-side handle to a stopped driver actor."""
        handle = self._spmd_handles.get(spmd_id)
        if handle is None:
            return
        handle.set_driver_handle(None)

    def release_all_driver_handles(self) -> None:
        """Release controller-side handles for all registered SPMDs."""
        for handle in self._spmd_handles.values():
            handle.set_driver_handle(None)

    def get_spmd_handle(self, spmd_id: str) -> SPMDHandle | None:
        """
        Get handle to an SPMD instance.

        Args:
            spmd_id: SPMD identifier

        Returns:
            SPMDHandle or None if not found
        """
        return self._spmd_handles.get(spmd_id)

    def get_driver_handle(self, spmd_id: str) -> Any | None:
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

    def get_all_spmd_ids(self) -> list[str]:
        """Get all registered SPMD IDs."""
        return list(self._spmd_handles.keys())

    async def evict_spmd(self, spmd_id: str) -> str | None:
        """
        Evict an SPMD and return checkpoint ID.

        Args:
            spmd_id: ID of SPMD to evict

        Returns:
            Checkpoint ID for later restoration, or None on failure

        TODO: Implement checkpoint saving
        """
        logger.warning(f"evict_spmd called for {spmd_id} - checkpoint not implemented")
        await self.stop_spmd(spmd_id, save_checkpoint=True)
        return None

    async def restore_spmd(
        self,
        spmd_id: str,
        config: SPMDConfig,
        placement_group: Any | None = None,
        worker_bundle_indices: list[int] | None = None,
        checkpoint_id: str | None = None,
    ) -> SPMDHandle | None:
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
        return await self.start_spmd(
            spmd_id=spmd_id,
            config=config,
            placement_group=placement_group,
            worker_bundle_indices=worker_bundle_indices,
            checkpoint_id=checkpoint_id,
        )

    def establish_topology(self, topology: dict[str, list[str]]) -> None:
        """
        Establish SPMD topology connections.

        Args:
            topology: Mapping of SPMD ID to downstream SPMD IDs
        """
        # Establish topology connections
        self._topology = {source: list(targets) for source, targets in topology.items()}

        # Build reverse topology
        self._reverse_topology = {spmd_id: [] for spmd_id in topology}
        for source, targets in topology.items():
            for target in targets:
                if target not in self._reverse_topology:
                    self._reverse_topology[target] = []
                if source not in self._reverse_topology[target]:
                    self._reverse_topology[target].append(source)

        logger.debug(f"Established topology with {len(topology)} SPMDs")

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

    def get_predecessors(self, spmd_id: str) -> list[str]:
        """
        Get predecessor SPMD IDs.
        """
        return self._reverse_topology.get(spmd_id, [])

    def get_successors(self, spmd_id: str) -> list[str]:
        """
        Get successor SPMD IDs.
        """
        return self._topology.get(spmd_id, [])

    async def broadcast_to_all_drivers(
        self, method: str, *args, **kwargs
    ) -> dict[str, Any]:
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
            for spmd_id, result in zip(spmd_ids, gathered, strict=True):  # type: ignore
                results[spmd_id] = result

        return results
