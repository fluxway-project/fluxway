"""
Worker handle wrapper for remote calls.

Based on ARCHITECTURE.md Section 6.1.2
"""

from typing import Any
from .handle_mapping import HandleMapping


class WorkerHandle:
    """
    Worker handle wrapper providing stable interface for remote calls.

    Automatically handles node changes and recovery through HandleMapping.
    """

    def __init__(
        self,
        logical_id: str,
        spmd_id: str,
        worker_id: str,
        handle_mapping: HandleMapping,
    ):
        self.logical_id = logical_id
        self.spmd_id = spmd_id
        self.worker_id = worker_id
        self._handle_mapping = handle_mapping

    def receive_forward_data(self, step: int, data: Any, source_id: str) -> Any:
        """
        Remote call: receive forward data.

        TODO: Implement data reception with handle mapping
        """
        return self._handle_mapping.call_with_retry(
            self.logical_id, "receive_forward_data", step, data, source_id
        )

    def receive_backward_data(self, step: int, grad: Any, source_id: str) -> Any:
        """
        Remote call: receive backward gradient data.

        TODO: Implement gradient reception with handle mapping
        """
        return self._handle_mapping.call_with_retry(
            self.logical_id, "receive_backward_data", step, grad, source_id
        )

    def mark_forward_complete(self, step: int, source_id: str) -> Any:
        """
        Remote call: mark forward pass complete.

        TODO: Implement completion signaling
        """
        return self._handle_mapping.call_with_retry(
            self.logical_id, "mark_forward_complete", step, source_id
        )

    def mark_backward_complete(self, step: int, source_id: str) -> Any:
        """
        Remote call: mark backward pass complete.

        TODO: Implement completion signaling
        """
        return self._handle_mapping.call_with_retry(
            self.logical_id, "mark_backward_complete", step, source_id
        )
