"""
Standardized Reader implementation leveraging Google Grain.
"""

import base64
from typing import Any

from fluxway.common.decorators import spmd, operator
from fluxway.common.utils.logging import get_logger
from fluxway.reader.base_reader import BaseReader

logger = get_logger(__name__)

try:
    import grain.python as grain
except ImportError:
    grain = None


@spmd()
class GrainReader(BaseReader):
    """
    Standard Reader implementation using Google Grain.

    The DataLoader and its iterator are created eagerly at instantiation time
    (not lazily on the first ``read`` call).  This guarantees that
    ``save_state`` / ``restore_state`` always operate on a live iterator, which is
    required for correct Grain-style checkpointing where state is a JSON-
    serialisable index dict (``{"next_index": N}``) managed by the iterator.

    Supports Elastic Sharding via Logical Shard Pools.
    """

    def __init__(self):
        super().__init__()
        self._dataloader = None
        self._iterator = None
        self._output_type = None

        # The framework sets rank / world_size / static_args on `self` before
        # calling __init__, so we can eagerly initialise the data pipeline here.
        static_args = getattr(self, "static_args", None)
        rank = getattr(self, "rank", None)
        world_size = getattr(self, "world_size", 1)
        if static_args is not None and rank is not None:
            self._create_data_loader(static_args, rank, world_size)

    def _create_data_loader(
        self,
        static_args: dict[str, Any],
        rank: int,
        world_size: int,
        shard_index: int | None = None,
        shard_count: int | None = None,
    ):
        """
        Construct the Grain DataLoader pipeline and create the iterator.

        Sub-classes that override this method must call ``super()`` **or**
        ensure ``self._iterator`` is set to a Grain ``DatasetIterator`` by the
        time the method returns, so that ``save_state`` / ``restore_state`` work.

        Args:
            shard_index: If provided, overrides rank for data sharding.
            shard_count: If provided, overrides world_size for data sharding.
        """
        if grain is None:
            raise ImportError("google-grain not found. Install it to use GrainReader.")

        data_source = static_args.get("data_source")
        if data_source is None:
            raise ValueError("GrainReader requires 'data_source' in static_args.")

        dataset = grain.MapDataset.source(data_source)
        for tx in static_args.get("transformations", []):
            dataset = dataset.map(tx)

        num_logical_shards = static_args.get("num_logical_shards")
        if shard_index is not None and shard_count is not None:
            s_idx, s_cnt = shard_index, shard_count
        elif num_logical_shards:
            s_cnt = num_logical_shards
            s_idx = rank % num_logical_shards
        else:
            s_idx, s_cnt = rank, world_size

        sampler = static_args.get("sampler") or grain.IndexSampler(
            num_records=len(data_source),
            shard_options=grain.ShardOptions(
                shard_index=s_idx,
                shard_count=s_cnt,
                drop_remainder=static_args.get("drop_remainder", False),
            ),
            shuffle=static_args.get("shuffle", True),
            seed=static_args.get("seed", 42),
            num_epochs=static_args.get("num_epochs"),
        )

        self._output_type = static_args.get("output_data_type")
        self._dataloader = grain.DataLoader(
            data_source=dataset,
            sampler=sampler,
            worker_count=static_args.get("num_threads", 0),
        )
        self._iterator = iter(self._dataloader)

        logger.info(
            f"[spmd_id={getattr(self, 'spmd_id', 'unknown')}] "
            f"Rank {rank} mapped to Logical Shard {s_idx}/{s_cnt}. "
            f"Serving {len(sampler)} records."
        )

    @operator("read", is_source=True)
    def read(self) -> Any:
        return self.next_item()

    def next_item(self) -> Any:
        if self._iterator is None:
            raise RuntimeError(
                f"GrainReader (rank={getattr(self, 'rank', '?')}): iterator is not "
                f"initialised.  This means _create_data_loader was not called during "
                f"__init__.  Check that 'static_args', 'rank', and 'world_size' are "
                f"injected by the framework before __init__ is called."
            )
        batch = next(self._iterator, None)
        if batch is None:
            return None
        if self._output_type and not isinstance(batch, self._output_type):
            if isinstance(batch, dict):
                return self._output_type(**batch)
            return self._output_type(batch)
        return batch

    _GRAIN_STATE_KEY = "__grain_iter_state__"

    def save_state(self) -> dict[str, Any]:
        """
        Return the iterator position as a framework-compatible dict.

        Grain's ``DatasetIterator.get_state()`` returns ``bytes`` (a JSON
        snapshot). We wrap it under ``_GRAIN_STATE_KEY`` so the dict-based
        state protocol used by the framework can transport and round-trip it
        without modification.
        """
        if hasattr(self._iterator, "get_state"):
            raw: bytes = self._iterator.get_state()
            return {self._GRAIN_STATE_KEY: base64.b64encode(raw).decode()}
        return {}

    def restore_state(self, state: dict[str, Any]) -> None:
        """
        Restore the iterator to a previously saved position.

        Raises ``TypeError`` when the iterator does not support ``set_state``
        so that sub-classes that use a non-Grain iterator get an explicit
        error instead of silently continuing from the wrong position.
        """
        if self._GRAIN_STATE_KEY not in state:
            return
        if not hasattr(self._iterator, "set_state"):
            raise TypeError(
                f"GrainReader (rank={getattr(self, 'rank', '?')}): the iterator "
                f"{type(self._iterator)!r} does not implement set_state(). "
                f"Override set_state() in your subclass to handle this iterator type."
            )
        raw = base64.b64decode(state[self._GRAIN_STATE_KEY].encode())
        self._iterator.set_state(raw)
