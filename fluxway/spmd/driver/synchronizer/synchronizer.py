"""
SPMD Synchronizer: Synchronization manager for SPMD sub-loops.

Based on ARCHITECTURE.md Section 3.1.1

Responsibilities:
- Execute SPMD internal sub-loops based on Main Orchestrator directives
- Block all Worker execution or call Worker Actor functions under specific conditions
- Implement synchronization points (Barrier)

Use cases:
- Parameter update sync: Before next step execution, wait for dependencies
- Checkpoint sync: All Workers must sync at checkpoint
- Debugging and monitoring: Pause all Workers when needed
"""

import logging
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class SPMDSynchronizer:
    """
    SPMD Synchronizer (Internal State Machine).

    Manages the internal training loop within an SPMD.
    Statefully executes SPMD via StatefulExecutor.
    """

    def __init__(self, spmd_id: str):
        self.spmd_id = spmd_id
        self._current_state: str = "idle"
        self._current_step: int = 0
        self._accumulation_steps: int = 1
        self._op_steps: Dict[str, int] = {}

    def set_loop_config(
        self,
        accumulation_steps: int = 1,
        sync_condition: Optional[Callable[[int], bool]] = None,
    ) -> None:
        """
        Configure the training loop.

        Args:
            accumulation_steps: Gradient accumulation steps
            sync_condition: Custom sync condition (e.g., step % K == 0)

        TODO: Store loop configuration
        TODO: Validate configuration
        """
        self._accumulation_steps = accumulation_steps
        logger.info(
            f"[spmd_id={self.spmd_id}] Loop config updated: accumulation_steps={accumulation_steps}"
        )

    def step(self) -> str:
        """
        Execute one state transition.

        Returns:
            Next state name

        TODO: Implement state transition logic
        TODO: Handle conditional triggers
        """
        return self._current_state

    def should_sync(self, step: int) -> bool:
        """
        Check if synchronization is needed at this step.

        Args:
            step: Current step number

        Returns:
            True if sync is required

        TODO: Check accumulation_steps condition
        TODO: Check custom sync conditions
        """
        return step % self._accumulation_steps == 0

    def wait_for_op_complete(self, op_name: str, step: int) -> None:
        """
        Wait for a specific operation to complete.

        Args:
            op_name: Operation name to wait for
            step: Step number to wait for

        TODO: Implement blocking wait
        TODO: Handle timeouts
        """
        pass

    def signal_op_complete(self, op_name: str, step: int) -> None:
        """
        Signal completion of an operation step.

        Args:
            op_name: Operation name
            step: Completed step number

        TODO: Notify waiting processes
        """
        self._op_steps[op_name] = max(self._op_steps.get(op_name, 0), step)

    def trigger_parameter_update(self) -> None:
        """
        Trigger parameter update via SPMD optimizer.

        Called by FSM when accumulation is complete.

        TODO: Call StatefulExecutor to trigger update
        """
        pass

    def get_state(self) -> str:
        """Get current FSM state."""
        return self._current_state

    def reset(self) -> None:
        """
        Reset FSM to initial state.

        TODO: Reset all state variables
        TODO: Clear pending signals
        """
        self._current_state = "idle"
        self._current_step = 0
        self._op_steps.clear()
        logger.info(f"[spmd_id={self.spmd_id}] Synchronizer reset")
