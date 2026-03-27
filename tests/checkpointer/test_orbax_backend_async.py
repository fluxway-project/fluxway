import asyncio
import jax.numpy as jnp
import shutil
import os
from fluxway.control_plane.checkpointer import OrbaxBackend


async def test_orbax_backend_loop_compatibility():
    ckpt_dir = "/tmp/test_orbax_backend_async"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    backend = OrbaxBackend()
    state = {"params": jnp.ones((5, 5))}

    # Mock Ray Worker logic: call backend.save within an event loop.
    metadata = await backend.save(state, ckpt_dir, "spmd_test", rank=0, step=1)
    backend.finalize()
    restored = await backend.load(metadata, target=state)
    assert jnp.allclose(state["params"], restored["params"])


if __name__ == "__main__":
    # Mock an already running loop.
    asyncio.run(test_orbax_backend_loop_compatibility())
