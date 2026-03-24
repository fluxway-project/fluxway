"""
State Manager for global state management and snapshot persistence.

Responsibilities:
- Aggregate snapshot states received from all SPMDs
- Provide in-memory save/load interface (persistence backend is user responsibility)
- Track SPMD runtime states for monitoring
"""

from typing import Any, Dict, List, Optional

from fluxway.common.types import GlobalState, SPMDState


class StateManager:
    """
    Global State Manager.

    Stores snapshots keyed by snapshot_id.  The snapshot payload is a nested dict:
        {spmd_id: {rank: worker_state_dict}}

    Persistence (writing to disk, GCS, etc.) is the caller's responsibility;
    this class provides the in-memory aggregation layer only.
    """

    def __init__(self):
        self._global_state: GlobalState = GlobalState()
        self._spmd_states: Dict[str, SPMDState] = {}
        self._snapshot_store: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

    # =========================================================================
    # Snapshot API (framework-level, no persistence)
    # =========================================================================

    def save_snapshot(
        self, snapshot_id: str, spmd_states: Dict[str, Dict[int, Dict[str, Any]]]
    ) -> None:
        self._snapshot_store[snapshot_id] = spmd_states

    def load_snapshot(self, snapshot_id: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
        if snapshot_id not in self._snapshot_store:
            raise KeyError(f"Snapshot {snapshot_id!r} not found")
        return self._snapshot_store[snapshot_id]

    def list_snapshots(self) -> List[str]:
        return list(self._snapshot_store.keys())

    def delete_snapshot(self, snapshot_id: str) -> None:
        if snapshot_id not in self._snapshot_store:
            raise KeyError(f"Snapshot {snapshot_id!r} not found")
        del self._snapshot_store[snapshot_id]

    # =========================================================================
    # SPMD Runtime State
    # =========================================================================

    def update_spmd_state(self, spmd_id: str, state: SPMDState) -> None:
        self._spmd_states[spmd_id] = state

    def get_spmd_state(self, spmd_id: str) -> Optional[SPMDState]:
        return self._spmd_states.get(spmd_id)

    def get_global_state(self) -> GlobalState:
        return self._global_state
