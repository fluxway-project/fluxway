"""
Partition Manager for SPMD sharding information.

Based on ARCHITECTURE.md Section 3.1.3 (SPMD Sharded Partition Manager)

Responsibilities:
- Declare Shared Partition info for SPMD UCX RPC function inputs/outputs
- Extract functions from Flax nnx.Module, JIT based on SPMD resource Mesh,
  then extract Dense Shared info
"""

from typing import Any, Callable, Dict, Optional


class ShardingInfo:
    """Sharding information for a tensor."""

    def __init__(
        self, shape: tuple, partition_spec: Optional[tuple] = None, mesh: Optional[Any] = None
    ):
        self.shape = shape
        self.partition_spec = partition_spec
        self.mesh = mesh


class PartitionManager:
    """
    Sharded Partition Manager.

    Manages sharding information for SPMD functions.

    State information:
    - Function to Sharding info mapping
    """

    def __init__(self):
        self._function_sharding: Dict[str, Dict[str, ShardingInfo]] = (
            {}
        )  # func_name -> param_name -> sharding

    def register_function_sharding(
        self,
        func_name: str,
        input_sharding: Dict[str, ShardingInfo],
        output_sharding: Dict[str, ShardingInfo],
    ) -> None:
        """
        Register sharding info for a function.

        Args:
            func_name: Function name
            input_sharding: Sharding for inputs
            output_sharding: Sharding for outputs

        TODO: Store sharding information
        TODO: Validate against mesh
        """
        pass

    def extract_sharding_from_module(self, module: Any, mesh: Any) -> Dict[str, ShardingInfo]:
        """
        Extract sharding info from Flax nnx.Module.

        Args:
            module: Flax nnx.Module
            mesh: SPMD resource mesh

        Returns:
            Extracted sharding info

        TODO: Extract functions from module
        TODO: JIT with mesh
        TODO: Extract Dense Shared info
        """
        pass

    def get_input_sharding(self, func_name: str) -> Dict[str, ShardingInfo]:
        """
        Get input sharding for a function.

        TODO: Implement lookup
        """
        pass

    def get_output_sharding(self, func_name: str) -> Dict[str, ShardingInfo]:
        """
        Get output sharding for a function.

        TODO: Implement lookup
        """
        pass

    def compute_routing(
        self, func_name: str, param_name: str, source_mesh: Any, target_mesh: Any
    ) -> Any:
        """
        Compute routing between source and target meshes.

        TODO: Implement routing computation
        TODO: Return LayoutSpec
        """
        pass
