import asyncio
import jax.numpy as jnp
import shutil
import os
import pytest
from fluxway.control_plane.checkpointer import OrbaxBackend


@pytest.mark.asyncio
async def test_orbax_backend_loop_compatibility():
    ckpt_dir = "/tmp/test_orbax_backend_async"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    backend = OrbaxBackend()
    state = {"params": jnp.ones((5, 5))}

    # Mock Ray Worker logic: call backend.save within an event loop.
    metadata = await backend.save(state, ckpt_dir, "spmd_test", rank=0, step=1)
    backend.finalize()

    if "orbax_tmp_path" in metadata and "orbax_final_path" in metadata:
        os.rename(metadata["orbax_tmp_path"], metadata["orbax_final_path"])

    restored = await backend.load(metadata)
    assert jnp.allclose(state["params"], restored["params"])


if __name__ == "__main__":
    # Mock an already running loop.
    asyncio.run(test_orbax_backend_loop_compatibility())
