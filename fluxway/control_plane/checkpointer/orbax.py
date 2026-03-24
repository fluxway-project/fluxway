"""
OrbaxCheckpointer: Orbax-backed implementation of FluxwayCheckpointer.

Designed for JAX backends. Handles nnx.State pytrees natively via Orbax's
StandardSave/StandardRestore which understands JAX pytrees.

Integer rank keys are preserved by passing the original state structure as the
``target`` to StandardRestore, so keys are never coerced to strings.
"""

from typing import Any, Dict, List

from fluxway.control_plane.checkpointer.base import FluxwayCheckpointer


class OrbaxCheckpointer(FluxwayCheckpointer):
    """
    Saves and loads Fluxway snapshots using Orbax CheckpointManager.

    JAX arrays and nnx.State pytrees are serialized natively.
    Integer rank keys in the state dict are preserved correctly on restore.

    Args:
        controller:     Ray remote handle returned by ``create_controller``.
        ckpt_dir:       Root directory for checkpoints.
        interval_steps: Steps between successive automatic checkpoints.
        max_to_keep:    Maximum number of checkpoints to retain on disk.
    """

    def __init__(
        self, controller, ckpt_dir: str, *, interval_steps: int, max_to_keep: int = 2
    ) -> None:
        super().__init__(controller, ckpt_dir, interval_steps=interval_steps)

        import orbax.checkpoint as ocp

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep, create=True, enable_async_checkpointing=False
        )
        self._mngr = ocp.CheckpointManager(self._ckpt_dir, options=options)
        # Cache: step -> state pytree structure for use as restore target.
        # We keep this in memory so that after saving we can restore with the
        # correct structure (preserving integer rank keys and nnx.State types).
        self._saved_states: Dict[int, Any] = {}

    def list_steps(self) -> List[int]:
        return sorted(self._mngr.all_steps())

    def _persist(self, step: int, state: Dict[str, Any]) -> None:
        import orbax.checkpoint as ocp

        self._mngr.save(step, args=ocp.args.StandardSave(state))
        self._mngr.wait_until_finished()
        # Cache the structure for a correct restore target later
        self._saved_states[step] = state

    def _load(self, step: int) -> Dict[str, Any]:
        import orbax.checkpoint as ocp

        # Use the cached target structure if available so that integer rank
        # keys and nnx.State pytree types are preserved exactly.
        # If not in cache (e.g. process restart), restore without target and
        # let the worker's restore_state() handle any type conversions.
        target = self._saved_states.get(step)
        if target is not None:
            return self._mngr.restore(step, args=ocp.args.StandardRestore(target))
        return self._mngr.restore(step, args=ocp.args.StandardRestore())
