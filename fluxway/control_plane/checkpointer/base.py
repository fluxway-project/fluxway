"""
FluxwayCheckpointer: Abstract base class for checkpoint managers.

Subclasses implement only the persistence layer (_persist / _load / list_steps).
All snapshot lifecycle logic (waiting for RUNNING, triggering, polling, restoring)
lives here and is shared across all backends.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from fluxway.common.types import DriverState

logger = logging.getLogger(__name__)


class FluxwayCheckpointer(ABC):
    """
    Abstract base class for Fluxway checkpoint managers.

    Handles the full snapshot lifecycle (coordination with the controller).
    Subclasses only implement persistence:
      - :meth:`list_steps`  — discover available steps on disk
      - :meth:`_persist`    — write a state dict to disk
      - :meth:`_load`       — read a state dict from disk

    Args:
        controller:     Ray remote handle returned by ``create_controller``.
        ckpt_dir:       Root directory for checkpoints.
        interval_steps: Steps between successive automatic checkpoints.
    """

    def __init__(self, controller, ckpt_dir: str, *, interval_steps: int) -> None:
        if interval_steps < 1:
            raise ValueError(f"interval_steps must be >= 1, got {interval_steps}")

        self._controller = controller
        self._ckpt_dir = os.path.abspath(ckpt_dir)
        self._interval_steps = interval_steps
        self._snapshot_counter = 0
        self._loop_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    # =========================================================================
    # Abstract persistence interface
    # =========================================================================

    @abstractmethod
    def list_steps(self) -> List[int]:
        """Return all checkpoint steps available on disk, in ascending order."""

    @abstractmethod
    def _persist(self, step: int, state: Dict[str, Any]) -> None:
        """Write ``state`` to disk under ``step``."""

    @abstractmethod
    def _load(self, step: int) -> Dict[str, Any]:
        """Read and return the state dict for ``step``."""

    # =========================================================================
    # Public API
    # =========================================================================

    def latest_step(self) -> Optional[int]:
        """Return the most recent checkpoint step, or None if none exist."""
        steps = self.list_steps()
        return steps[-1] if steps else None

    async def save(self, step: int) -> None:
        """
        Explicitly trigger a global snapshot at ``step`` and persist it.

        Blocks until the snapshot is fully collected and written to disk.
        Can be called at any point during training after the controller is RUNNING.

        Args:
            step: The training step at which to take the snapshot.
        """
        await self._take_snapshot(step)

    async def restore(self, step: Optional[int] = None) -> Optional[int]:
        """
        Restore model state from a checkpoint.

        Args:
            step: Step to restore from. If ``None``, restores from the latest
                  available checkpoint. If no checkpoint exists and ``step`` is
                  ``None``, returns ``None`` without error.

        Returns:
            The step number restored, or ``None`` if no checkpoint was found
            (only possible when ``step`` is ``None``).

        Raises:
            FileNotFoundError: If a specific ``step`` is requested but not found.
        """
        if step is None:
            step = self.latest_step()
            if step is None:
                logger.info("FluxwayCheckpointer: no existing checkpoint found.")
                return None

        available = self.list_steps()
        if step not in available:
            raise FileNotFoundError(
                f"FluxwayCheckpointer: checkpoint step {step} not found in "
                f"{self._ckpt_dir}. Available steps: {available}"
            )

        logger.info(f"FluxwayCheckpointer: restoring from step {step}…")
        state = self._load(step)
        if not state:
            raise RuntimeError(
                f"FluxwayCheckpointer: _load({step}) returned an empty dict. "
                f"Checkpoint in {self._ckpt_dir} may be corrupt."
            )

        snapshot_id = self._make_snapshot_id(step)
        await self._controller.restore_snapshot(snapshot_id, state)
        logger.info(f"FluxwayCheckpointer: successfully restored step {step}.")
        return step

    async def run_loop(self) -> None:
        """
        Background coroutine. Waits for all SPMDs to reach RUNNING state, then
        checkpoints every ``interval_steps`` steps until asked to stop.
        """
        logger.debug("FluxwayCheckpointer: run_loop started")
        stop_task = asyncio.create_task(self._stop_event.wait())
        wait_task = asyncio.create_task(self._wait_for_running())

        done, pending = await asyncio.wait(
            [wait_task, stop_task], return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done:
            wait_task.cancel()
            return  # Stopped before running

        next_step = await self._current_max_step() + self._interval_steps
        while not self._stop_event.is_set():
            snapshot_task = asyncio.create_task(self._take_snapshot(next_step))
            done, pending = await asyncio.wait(
                [snapshot_task, stop_task], return_when=asyncio.FIRST_COMPLETED
            )

            if stop_task in done:
                snapshot_task.cancel()
                try:
                    await snapshot_task
                except asyncio.CancelledError:
                    logger.warning(
                        f"FluxwayCheckpointer: snapshot task cancelled exception caught {snapshot_task.exception()}"
                    )
                break

            if snapshot_task.exception():
                raise snapshot_task.exception()

            next_step += self._interval_steps

    def start(self) -> None:
        """
        Start the background checkpointing loop.

        This method must be called from within a running asyncio event loop.
        """
        if self._loop_task is not None and not self._loop_task.done():
            logger.warning("FluxwayCheckpointer: background loop already running.")
            return

        self._stop_event.clear()
        self._loop_task = asyncio.create_task(self.run_loop())

    async def wait_until_finished(self) -> None:
        """Stop the background loop task and await its termination."""
        self._stop_event.set()
        if self._loop_task is not None and not self._loop_task.done():
            await self._loop_task

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _make_snapshot_id(self, step: int) -> str:
        self._snapshot_counter += 1
        return f"snapshot_{step}_{self._snapshot_counter}"

    async def _wait_for_running(self) -> None:
        """Block until every SPMD reports RUNNING state."""
        logger.debug(
            "FluxwayCheckpointer: waiting for all drivers to reach RUNNING state..."
        )
        while True:
            heartbeats: dict = await self._controller.get_all_heartbeats()
            if heartbeats:
                all_running = True
                for hb in heartbeats.values():
                    if hb.state != DriverState.RUNNING:
                        all_running = False
                        break

                if all_running:
                    logger.debug(
                        "FluxwayCheckpointer: all drivers RUNNING, backend loop active."
                    )
                    return
            await asyncio.sleep(0.5)

    async def _current_max_step(self) -> int:
        """Return the highest completed op-step across all SPMDs."""
        heartbeats: dict = await self._controller.get_all_heartbeats()
        max_step = -1
        for hb in heartbeats.values():
            for step in hb.op_steps.values():
                if step > max_step:
                    max_step = step
        if max_step < 0:
            raise RuntimeError(
                "FluxwayCheckpointer: could not determine current training step. "
                "Ensure heartbeats are reporting op_steps correctly."
            )
        return max_step

    async def _take_snapshot(self, target_step: int) -> None:
        """Trigger a global snapshot at ``target_step`` and persist to disk."""
        snapshot_id = self._make_snapshot_id(target_step)
        logger.info(
            f"FluxwayCheckpointer: requesting global snapshot '{snapshot_id}' "
            f"at step {target_step}…"
        )
        await self._controller.request_snapshot(snapshot_id, target_step)
        state = await self._poll_snapshot(snapshot_id)
        logger.info(
            f"FluxwayCheckpointer: saving step {target_step} "
            f"(SPMDs: {list(state.keys())})…"
        )
        self._persist(target_step, state)
        logger.info(f"FluxwayCheckpointer: step {target_step} saved.")

    async def _poll_snapshot(self, snapshot_id: str, timeout: float = 300.0) -> dict:
        """
        Poll controller until the snapshot is fully aggregated.

        ``controller.get_snapshot`` returns ``None`` while the snapshot is not
        yet in the store, and the state dict once it is ready.

        Raises:
            TimeoutError: If the snapshot is not ready within ``timeout`` seconds.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            state = await self._controller.get_snapshot(snapshot_id)
            if state is not None:
                return state
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(
                    f"FluxwayCheckpointer: snapshot '{snapshot_id}' was not "
                    f"ready within {timeout}s. Check controller logs."
                )
            await asyncio.sleep(0.1)
