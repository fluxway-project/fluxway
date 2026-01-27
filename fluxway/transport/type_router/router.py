"""
Data Plane Router for transport layer.

Based on ARCHITECTURE.md Section 4.2.3

Implements type routing at the SPMD function call abstraction layer,
automatically routing input parameters to different transmission planes.

Routing rules:
1. Check input parameter type and Sharding attributes
2. Sharded Tensor -> Data Plane (ULP protocol)
3. Global Scalar / Config -> Control Plane (Orchestrator propagation)
4. Synchronized Result -> Control Plane (post-aggregation)
"""

from typing import Any, Dict, Tuple

from fluxway.common.types import LayoutSpec


class DataPlaneRouter:
    """
    Data Plane Router for automatic transmission plane selection.

    User interface:
    - Users only define unified function signatures
    - Framework auto-routes based on type annotations
    - No need for explicit different call methods
    """

    def __init__(self):
        self._layout_specs: Dict[str, LayoutSpec] = {}

    def register_layout(self, param_name: str, layout_spec: LayoutSpec) -> None:
        """
        Register layout specification for a parameter.

        Args:
            param_name: Parameter name
            layout_spec: Layout specification for routing

        TODO: Store layout spec
        TODO: Validate spec
        """
        pass

    def route_to_data_plane(self, param_name: str, data: Any) -> Tuple[bool, Any]:
        """
        Attempt to route data to Data Plane.

        Args:
            param_name: Parameter name
            data: Data to route

        Returns:
            (is_routed, processed_data)

        TODO: Check if param should go to Data Plane
        TODO: Prepare data for ULP transmission
        """
        pass

    def route_to_control_plane(self, param_name: str, data: Any) -> Tuple[bool, Any]:
        """
        Attempt to route data to Control Plane.

        Args:
            param_name: Parameter name
            data: Data to route

        Returns:
            (is_routed, processed_data)

        TODO: Check if param should go to Control Plane
        TODO: Prepare data for Orchestrator propagation
        """
        pass

    def is_sharded_tensor(self, data: Any) -> bool:
        """
        Check if data is a sharded tensor.

        TODO: Check tensor sharding attributes
        TODO: Support JAX/PyTorch/NumPy
        """
        pass

    def is_global_scalar(self, data: Any) -> bool:
        """
        Check if data is a global scalar/config.

        TODO: Check for scalar types
        TODO: Check for config objects
        """
        pass

    def get_layout_spec(self, param_name: str) -> LayoutSpec:
        """
        Get layout specification for a parameter.

        TODO: Implement lookup
        TODO: Handle missing specs
        """
        pass

    def infer_layout(self, data: Any) -> LayoutSpec:
        """
        Infer layout specification from data.

        TODO: Infer from tensor sharding
        TODO: Infer from data shape and type
        """
        pass
