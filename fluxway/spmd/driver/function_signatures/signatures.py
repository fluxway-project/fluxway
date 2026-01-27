"""
Function Signatures for SPMD worker functions.

Based on ARCHITECTURE.md Section 3.1.3 (SPMD Worker Function Signatures)

Responsibilities:
- Collect and save registered Worker Actor supported functions
- Collect and save registered Worker Actor UCX RPC supported functions
  (TODO: UCX RPC initially implemented via Ray)
"""

from typing import Callable, Dict, List, Optional


class FunctionSignature:
    """Signature of a worker function."""

    def __init__(
        self,
        name: str,
        input_types: Dict[str, type],
        output_types: Dict[str, type],
        is_rpc: bool = False,
    ):
        self.name = name
        self.input_types = input_types
        self.output_types = output_types
        self.is_rpc = is_rpc


class FunctionSignatures:
    """
    Function Signatures Registry.

    Stores function signature information for all registered workers.

    State information:
    - Function signature info Dict
    """

    def __init__(self):
        self._signatures: Dict[str, FunctionSignature] = {}
        self._rpc_signatures: Dict[str, FunctionSignature] = {}

    def register_function(
        self,
        name: str,
        func: Callable,
        input_types: Optional[Dict[str, type]] = None,
        output_types: Optional[Dict[str, type]] = None,
    ) -> None:
        """
        Register a worker function signature.

        Args:
            name: Function name
            func: The function object
            input_types: Input parameter types
            output_types: Output types

        TODO: Extract signature from function
        TODO: Validate signature
        """
        pass

    def register_rpc_function(
        self,
        name: str,
        func: Callable,
        input_types: Optional[Dict[str, type]] = None,
        output_types: Optional[Dict[str, type]] = None,
    ) -> None:
        """
        Register a UCX RPC function signature.

        TODO: Store RPC function signature
        TODO: Initially use Ray for RPC
        """
        pass

    def get_signature(self, name: str) -> Optional[FunctionSignature]:
        """
        Get function signature by name.

        TODO: Implement lookup
        """
        pass

    def get_rpc_signature(self, name: str) -> Optional[FunctionSignature]:
        """
        Get RPC function signature by name.

        TODO: Implement lookup
        """
        pass

    def list_functions(self) -> List[str]:
        """List all registered function names."""
        return list(self._signatures.keys())

    def list_rpc_functions(self) -> List[str]:
        """List all registered RPC function names."""
        return list(self._rpc_signatures.keys())
