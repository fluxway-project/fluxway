import itertools
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


class FluxwayCheckpointer:
    """
    Controller-side checkpoint orchestrator.

    Manages only the per-step manifest.json that maps each (spmd_id, rank) to
    the path metadata returned by the Worker's WorkerCheckpointBackend.  The
    actual tensor serialization is performed by each Worker in its own process.

    Lifecycle initiated by Controller.report_op_completed (event-driven):
      1. save(step)  →  request_snapshot  →  Workers write tensors  →  manifest written
      2. restore(step)  →  manifest read  →  Workers read tensors

    Args:
        controller:     Ray Actor handle of the owning Controller.
        ckpt_dir:       Root directory on a shared filesystem for all checkpoints.
        interval_steps: How many completed steps between automatic checkpoints.
        max_to_keep:    Maximum number of on-disk checkpoint directories to retain.
        validate_deps:  Optional callable invoked at construction to validate optional
                        dependencies (e.g. ``import safetensors``). Raises if missing.
    """

    def __init__(
        self,
        controller,
        ckpt_dir: str,
        *,
        interval_steps: int,
        force_write: bool = False,
        max_to_keep: int = 2,
        validate_deps: Callable | None = None,
    ) -> None:
        if interval_steps < 1:
            raise ValueError(f"interval_steps must be >= 1, got {interval_steps}")
        if max_to_keep <= 0:
            raise ValueError(f"max_to_keep must be positive, got {max_to_keep}")

        if validate_deps is not None:
            validate_deps()

        self._controller = controller
        self._ckpt_dir = os.path.abspath(ckpt_dir)
        self._interval_steps = interval_steps
        self._force_write = force_write
        self._max_to_keep = max_to_keep
        self._snapshot_counter = itertools.count()

    @property
    def interval_steps(self) -> int:
        return self._interval_steps

    # =========================================================================
    # Public API
    # =========================================================================

    def list_steps(self) -> list[int]:
        """Return all valid checkpoint steps on disk, ascending."""
        root = Path(self._ckpt_dir)
        if not root.exists():
            return []
        return sorted(
            int(d.name)
            for d in root.iterdir()
            if d.is_dir() and d.name.isdigit() and (d / "manifest.json").exists()
        )

    def latest_step(self) -> int | None:
        steps = self.list_steps()
        return steps[-1] if steps else None

    async def save(self, step: int) -> None:
        """Trigger a global snapshot at ``step`` and persist its manifest."""
        snapshot_id = f"ckpt_{step}_{next(self._snapshot_counter)}"
        logger.info(f"Requesting snapshot '{snapshot_id}' at step {step}")
        manifest = await self._controller.request_snapshot(
            snapshot_id, step, self._ckpt_dir
        )
        if not isinstance(manifest, dict):
            raise RuntimeError(
                f"request_snapshot returned {type(manifest).__name__} instead of dict; "
                "controller may not be fully initialized."
            )
        self._persist(step, manifest)
        logger.info(f"Checkpoint step {step} saved ({list(manifest.keys())})")

    async def restore(self, step: int | None = None) -> int | None:
        """
        Restore state from a checkpoint.

        Returns the step restored, or None if no checkpoint exists and step was None.
        Raises FileNotFoundError if a specific step is requested but absent.
        """
        if step is None:
            step = self.latest_step()
            if step is None:
                logger.info("No existing checkpoint found.")
                return None

        # Single filesystem read — avoids TOCTOU between list_steps() and _load().
        try:
            manifest = self._load(step)
        except FileNotFoundError as e:
            available = self.list_steps()
            raise FileNotFoundError(
                f"Checkpoint step {step} not found in {self._ckpt_dir}. Available: {available}"
            ) from e

        if not manifest:
            raise RuntimeError(
                f"_load({step}) returned empty — checkpoint may be corrupt."
            )
        snapshot_id = f"restore_{step}_{next(self._snapshot_counter)}"
        await self._controller.restore_snapshot(snapshot_id, step, manifest)
        logger.info(f"Checkpoint step {step} restored.")
        return step

    # =========================================================================
    # Internal Persistence (format-agnostic manifest JSON)
    # =========================================================================

    def _persist(self, step: int, manifest: dict[str, Any]) -> None:
        """Write the manifest that maps spmd_id/rank to Worker-written path metadata."""
        step_dir = Path(self._ckpt_dir) / str(step)
        manifest_path = step_dir / "manifest.json"

        # 1. Deduplicate and commit (rename) all orbax temporary directories safely
        renamed_paths = set()
        for spmd_id, ranks in manifest.items():
            for rank, meta in ranks.items():
                if "orbax_tmp_path" in meta and "orbax_final_path" in meta:
                    o_tmp = Path(meta["orbax_tmp_path"])
                    o_final = Path(meta["orbax_final_path"])

                    if str(o_final) not in renamed_paths:
                        if o_tmp.exists() and not o_final.exists():
                            o_tmp.rename(o_final)
                            renamed_paths.add(str(o_final))
                        elif o_tmp.exists() and o_final.exists():
                            if self._force_write:
                                shutil.rmtree(o_final)
                                o_tmp.rename(o_final)
                                renamed_paths.add(str(o_final))
                            else:
                                raise FileExistsError(
                                    f"Orbax checkpoint directory already exists at {o_final}. "
                                    "Set force_write=True to overwrite."
                                )

        if manifest_path.exists() and not self._force_write:
            raise FileExistsError(
                f"Checkpoint step {step} already exists at {manifest_path}. "
                "Set force_write=True in CheckpointConfig to overwrite."
            )
        step_dir.mkdir(parents=True, exist_ok=True)
        tmp_man = step_dir / "manifest.json.tmp"
        with open(tmp_man, "w") as f:
            json.dump(manifest, f)
        tmp_man.rename(manifest_path)
        self._evict_old_checkpoints()

    def _load(self, step: int) -> dict[str, Any]:
        """Read and return the manifest dict for ``step``."""
        manifest_path = Path(self._ckpt_dir) / str(step) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"FluxwayCheckpointer: manifest not found: {manifest_path}"
            )
        with open(manifest_path) as f:
            return json.load(f)

    def _evict_old_checkpoints(self) -> None:
        """Remove oldest checkpoint directories exceeding max_to_keep."""
        steps = self.list_steps()
        for old_step in steps[: -self._max_to_keep]:
            shutil.rmtree(Path(self._ckpt_dir) / str(old_step))
