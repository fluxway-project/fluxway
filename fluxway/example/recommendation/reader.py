import jax.numpy as jnp
from typing import Any, Dict, Optional
from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from fluxway.reader import GrainReader
from .data_types import RaggedInput


@spmd()
class RecommendationReader(GrainReader):
    """
    Standard Reader SPMD for Recommendation Example.

    This class demonstrates how to integrate Google Grain with Fluxway:
    1. Inherit from GrainReader for distributed sharding and data loading.
    2. Use TransportTensor to wrap ragged (variable-length) data.
    3. Generate synthetic samples with different lengths to verify E2E training logic.
    """

    def _create_data_loader(
        self, static_args: Dict[str, Any], rank: int, world_size: int
    ):
        """
        Initializes the data loading pipeline.
        Provides synthetic ragged samples if no data_source is specified.
        """
        if "data_source" not in static_args:
            # Base samples with different lengths (Ragged structure)
            base_samples = [
                {"feature1": [10, 20, 10], "feature2": [30, 40]},
                {"feature1": [20, 40], "feature2": [100, 110, 120, 100, 110]},
            ]

            # Create a dataset for training demonstration
            data = [base_samples[i % 2].copy() for i in range(100)]
            static_args["data_source"] = data

            # Set default training configurations
            static_args.setdefault("num_epochs", 1)
            static_args.setdefault("shuffle", False)

            print(
                f"[RecommendationReader] Initialized synthetic dataset (Rank {rank})."
            )

        # Default batch size for the demo
        static_args.setdefault("batch_size", 2)

        # Build Grain DataLoader
        super()._create_data_loader(static_args, rank, world_size)

    @operator("read")
    def read(self) -> Optional[RaggedInput]:
        """
        Produces a batch of RaggedInput for downstream operators.
        """
        static_args = getattr(self, "static_args", {})
        batch_size = static_args.get("batch_size", 2)
        samples = []

        # Accumulate samples from Grain iterator
        for _ in range(batch_size):
            try:
                # Use next_item() from GrainReader to fetch the next record
                item = self.next_item()
                if item is None:
                    break
                samples.append(item)
            except Exception as e:
                print(f"[RecommendationReader] Error fetching item: {e}")
                break

        if not samples:
            return None

        # Wrap into TransportTensor with nested_splits for ragged data transport.
        # TransportTensor.from_lists automatically detects variable lengths.
        f1_list = [s["feature1"] for s in samples]
        f2_list = [s["feature2"] for s in samples]

        t1 = TransportTensor.from_lists(f1_list, dtype=jnp.int32)
        t2 = TransportTensor.from_lists(f2_list, dtype=jnp.int32)

        print(f"[RecommendationReader] Produced Batch Shard (Size {len(samples)})")
        return RaggedInput(feature1=t1, feature2=t2)
