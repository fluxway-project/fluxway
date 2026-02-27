"""
State Manager for global state management and checkpointing.

Based on ARCHITECTURE.md Section 2.1.4

Responsibilities:
- Unified state management: Aggregate all SPMD state info, provide query interface
- Persistence strategy: Hold persistence strategy, trigger checkpoints via SPMD Manager
- Support node state recovery: Manage state save before eviction and load after recovery
"""

from typing import Dict, List, Optional

from fluxway.common.types import GlobalState, SPMDState


class StateManager:
    """
    Global State Manager.

    Manages training state aggregation, checkpointing, and recovery.

    State information:
    - Training Metric Dict (current step, loss, auc, etc.)
    - Profiler-level resource usage
    - Checkpoint version and location
    """

    def __init__(self):
        self._global_state: GlobalState = GlobalState()
        self._spmd_states: Dict[str, SPMDState] = {}
        self._checkpoints: Dict[str, List[str]] = {}  # spmd_id -> [checkpoint_ids]

    def save_checkpoint(self, checkpoint_id: str, state: GlobalState) -> None:
        """
        Save global checkpoint.

        Args:
            checkpoint_id: Unique checkpoint identifier
            state: Global state to save

        TODO: Implement checkpoint serialization
        TODO: Support incremental checkpoints
        TODO: Support full checkpoints at key points (epoch end)
        """
        pass

    def load_checkpoint(self, checkpoint_id: str) -> GlobalState:
        """
        Load checkpoint.

        Args:
            checkpoint_id: Checkpoint to load

        Returns:
            Loaded global state

        TODO: Implement checkpoint deserialization
        TODO: Validate checkpoint integrity
        """
        pass

    def get_latest_checkpoint(self, spmd_id: str) -> Optional[str]:
        """
        Get the latest checkpoint ID for an SPMD.

        Args:
            spmd_id: SPMD identifier

        Returns:
            Latest checkpoint ID or None

        TODO: Implement checkpoint tracking
        """
        pass

    def list_checkpoints(self, spmd_id: str) -> List[str]:
        """
        List all checkpoints for an SPMD.

        Args:
            spmd_id: SPMD identifier

        Returns:
            List of checkpoint IDs

        TODO: Implement checkpoint listing
        """
        pass

    def get_spmd_state(self, spmd_id: str) -> Optional[SPMDState]:
        """
        Get current state of an SPMD.

        TODO: Implement state lookup
        """
        pass

    def update_spmd_state(self, spmd_id: str, state: SPMDState) -> None:
        """
        Update SPMD state.

        Called by SPMD to report its current state.

        TODO: Implement state update
        TODO: Trigger metrics reporting (Tensorboard)
        """
        pass

    def get_global_state(self) -> GlobalState:
        """
        Get current global state.

        Aggregates state from all SPMDs.

        TODO: Implement state aggregation
        """
        pass

    def save_spmd_state(self, spmd_id: str, state: SPMDState) -> str:
        """
        Save SPMD state, return checkpoint ID.

        Used for pre-eviction checkpointing.

        Args:
            spmd_id: SPMD identifier
            state: State to save

        Returns:
            Generated checkpoint ID

        TODO: Implement state serialization
        TODO: Include parameters, optimizer state, RNG seeds
        TODO: Ensure consistency guarantees
        """
        pass

    def load_spmd_state(self, spmd_id: str, checkpoint_id: str) -> SPMDState:
        """
        Load SPMD state from checkpoint.

        Args:
            spmd_id: SPMD identifier
            checkpoint_id: Checkpoint to load from

        Returns:
            Loaded SPMD state

        TODO: Implement state deserialization
        TODO: Support incremental and full recovery
        """
        pass

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        TODO: Implement checkpoint deletion
        TODO: Handle cleanup of old checkpoints
        """
        pass
