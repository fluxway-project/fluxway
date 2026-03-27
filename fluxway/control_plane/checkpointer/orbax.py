"""
Per-Worker Orbax serialization utilities.

OrbaxBackend uses StandardCheckpointHandler directly with the async_save API,
so all I/O runs natively in the Ray worker's uvloop without any nested event loops.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict

from etils import epath

from fluxway.control_plane.checkpointer.backend import WorkerCheckpointBackend


class OrbaxBackend(WorkerCheckpointBackend, backend_name="orbax"):
    """Orbax StandardCheckpointHandler"""

    def __init__(self, force_write: bool = False):
        self._handler = None
        self._force_write = force_write

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handler = None

    def _get_handler(self):
        if self._handler is None:
            import orbax.checkpoint as ocp

            self._handler = ocp.StandardCheckpointHandler()
        return self._handler

    async def save(
        self, state: Dict[str, Any], ckpt_dir: str, spmd_id: str, rank: int, step: int
    ) -> Dict[str, Any]:
        import orbax.checkpoint as ocp

        final_path = Path(ckpt_dir) / str(step) / spmd_id
        tmp_path = final_path.parent / (final_path.name + ".orbax-checkpoint-tmp")

        await asyncio.to_thread(tmp_path.mkdir, parents=True, exist_ok=True)

        handler = self._get_handler()
        commit_futures = (
            await handler.async_save(
                epath.Path(tmp_path), args=ocp.args.StandardSave(state)
            )
            or []
        )

        await asyncio.gather(*[asyncio.to_thread(f.result) for f in commit_futures])

        if rank == 0:
            await asyncio.to_thread(handler.finalize, epath.Path(tmp_path))
            await asyncio.to_thread(tmp_path.rename, final_path)

        return {"ckpt_dir": str(final_path)}

    async def load(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        ckpt_path = epath.Path(metadata["ckpt_dir"])
        handler = self._get_handler()
        return await asyncio.to_thread(handler.restore, ckpt_path)

    def finalize(self) -> None:
        # All I/O in save() is fully awaited before returning; nothing pending.
        pass

    def close(self) -> None:
        if self._handler:
            self._handler.close()
            self._handler = None
