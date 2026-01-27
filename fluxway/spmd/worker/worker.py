"""
SPMD Worker: Actual computation execution unit.

Based on ARCHITECTURE.md Section 3.2

Each SPMD Worker runs user-defined forward/backward logic
and implements the dual-stream convergence mechanism.
"""

from typing import Any, Dict, Optional
import asyncio

from .view import WorkerView
from .mailbox import ControlMailbox
from .wrapper import ForwardWrapper, BackwardWrapper


class SPMDWorker:
    """
    SPMD Worker.

    Event-driven event loop that implements dual-stream convergence:
    On(Data) AND On(Control)
    """

    def __init__(self, spmd_id: str, worker_id: str):
        self.spmd_id = spmd_id
        self.worker_id = worker_id
        self.view = WorkerView(spmd_id, worker_id)
        self.mailbox = ControlMailbox()
        self.forward_wrapper: Optional[ForwardWrapper] = None
        self.backward_wrapper: Optional[BackwardWrapper] = None
        self._running = False

    def initialize(self, user_module: Any) -> None:
        """
        Initialize worker with user module.

        Args:
            user_module: User's custom module with forward/backward

        TODO: Extract forward/backward functions
        TODO: Create wrappers
        TODO: Initialize view with topology
        """
        pass

    async def start(self) -> None:
        """
        Start the worker event loop.

        TODO: Start listening for data and control events
        TODO: Implement event-driven execution
        """
        pass

    async def stop(self) -> None:
        """
        Stop the worker.

        TODO: Graceful shutdown
        TODO: Save state if needed
        """
        pass

    async def set_control(self, batch_id: str, config: Dict[str, Any]) -> None:
        """
        Control Plane: Receive pre-pushed configuration.

        Args:
            batch_id: Batch identifier for alignment
            config: Control configuration

        TODO: Store in mailbox
        TODO: Signal waiting data handlers
        """
        pass

    async def receive_data(self, batch_id: str, data: Any) -> Any:
        """
        Data Plane: Receive data and wait for control info.

        Args:
            batch_id: Batch identifier for alignment
            data: Received tensor data

        Returns:
            Computation result

        TODO: Check mailbox for control config
        TODO: Wait if control not yet arrived
        TODO: Execute computation when both arrive
        """
        pass

    def receive_forward_data(self, step: int, data: Any, source_id: str) -> None:
        """
        Receive forward data from upstream SPMD.

        TODO: Store in data buffer
        TODO: Check for completion
        """
        pass

    def receive_backward_data(self, step: int, grad: Any, source_id: str) -> None:
        """
        Receive backward gradient from downstream SPMD.

        TODO: Store in gradient buffer
        TODO: Check for completion
        """
        pass

    def mark_forward_complete(self, step: int, source_id: str) -> None:
        """
        Mark forward complete from a source.

        TODO: Update completion tracking
        TODO: Trigger execution if all sources complete
        """
        pass

    def mark_backward_complete(self, step: int, source_id: str) -> None:
        """
        Mark backward complete from a source.

        TODO: Update completion tracking
        TODO: Trigger execution if all sources complete
        """
        pass

    def execute_forward(self, step: int) -> Any:
        """
        Execute forward pass.

        TODO: Gather all inputs
        TODO: Call user forward function via wrapper
        TODO: Send outputs to downstream
        """
        pass

    def execute_backward(self, step: int) -> Any:
        """
        Execute backward pass.

        TODO: Gather all gradients
        TODO: Call user backward function via wrapper
        TODO: Send gradients to upstream
        """
        pass

    def save_states(self) -> Dict[str, Any]:
        """
        Save worker state for checkpointing.

        TODO: Serialize model state
        TODO: Serialize optimizer state
        TODO: Save RNG seeds
        """
        # Monitor worker state save attempt
        import time

        start_time = time.time()

        from fluxway.common.monitoring import MonitoringFacade

        MonitoringFacade.record_checkpoint_event(
            event_type="save_attempt",
            checkpoint_id="pending",
            component_type="worker",
            component_id=self.worker_id,
            status="not_implemented",
        )

        # TODO: Serialize model state
        # TODO: Serialize optimizer state
        # TODO: Save RNG seeds

        duration = time.time() - start_time
        MonitoringFacade.record_checkpoint_event(
            event_type="save_complete",
            checkpoint_id=f"{self.worker_id}_checkpoint",
            component_type="worker",
            component_id=self.worker_id,
            status="not_implemented",
            duration_seconds=duration,
        )
        return {}

    def load_states(self, state: Dict[str, Any]) -> None:
        """
        Load worker state from checkpoint.

        TODO: Restore model state
        TODO: Restore optimizer state
        TODO: Restore RNG seeds
        """
        # Monitor worker state load attempt
        import time

        start_time = time.time()

        from fluxway.common.monitoring import MonitoringFacade

        MonitoringFacade.record_checkpoint_event(
            event_type="load_attempt",
            checkpoint_id="unknown",
            component_type="worker",
            component_id=self.worker_id,
            status="not_implemented",
        )

        # TODO: Restore model state
        # TODO: Restore optimizer state
        # TODO: Restore RNG seeds

        duration = time.time() - start_time
        MonitoringFacade.record_checkpoint_event(
            event_type="load_complete",
            checkpoint_id="unknown",
            component_type="worker",
            component_id=self.worker_id,
            status="not_implemented",
            duration_seconds=duration,
        )

    def send_heartbeat(self) -> None:
        """
        Send heartbeat to monitor.

        Reports telemetry and updates health status.
        Uses MonitoringFacade for minimal code intrusion.
        """
        try:
            from .fluxway.common.monitoring import MonitoringFacade

            # Check if metrics collector exists
            if not hasattr(self, "_metrics_collector") or self._metrics_collector is None:
                # Create metrics collector if not exists
                self._metrics_collector = MonitoringFacade.create_worker_collector(
                    self.worker_id, self.spmd_id
                )

            # Collect metrics
            try:
                import ray

                telemetry = ray.get(self._metrics_collector.collect.remote())
            except Exception:
                # If Ray is not available or collector failed, skip
                return

            # Send heartbeat via MonitoringFacade
            MonitoringFacade.send_heartbeat(self.worker_id, self.spmd_id, telemetry)
        except Exception as e:
            # Log error but don't fail the worker
            # In production, should use proper logging
            pass
