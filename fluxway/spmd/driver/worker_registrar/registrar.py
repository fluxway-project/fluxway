"""
Worker Registrar for SPMD worker registration.

Based on ARCHITECTURE.md Section 3.1.2 (SPMD Worker Registrar)

Responsibilities:
- Register SPMD Worker Actor Handles
- Register Worker names and RPC Server UCP addresses
"""

from typing import Any, Dict, Optional


class WorkerRegistrar:
    """
    Worker Registrar.

    Manages registration of workers within an SPMD.

    State information:
    - Registration info Dict
    """

    def __init__(self, spmd_id: str):
        self.spmd_id = spmd_id
        self._registrations: Dict[str, dict] = {}  # worker_id -> registration info

    def register_worker(
        self, worker_id: str, actor_handle: Any, ucp_address: Optional[str] = None
    ) -> None:
        """
        Register a worker.

        Args:
            worker_id: Unique worker identifier
            actor_handle: Ray Actor handle
            ucp_address: Optional UCX RPC server address

        TODO: Store registration info
        TODO: Validate uniqueness
        """
        pass

    def unregister_worker(self, worker_id: str) -> None:
        """
        Unregister a worker.

        TODO: Remove registration
        TODO: Notify dependent components
        """
        pass

    def get_worker_handle(self, worker_id: str) -> Optional[Any]:
        """
        Get worker Actor handle.

        TODO: Return registered handle
        """
        pass

    def get_worker_ucp_address(self, worker_id: str) -> Optional[str]:
        """
        Get worker UCP address.

        TODO: Return registered UCP address
        """
        pass

    def list_workers(self) -> Dict[str, dict]:
        """
        List all registered workers.

        Returns:
            All registration info
        """
        return self._registrations.copy()

    def get_num_workers(self) -> int:
        """Get number of registered workers."""
        return len(self._registrations)
