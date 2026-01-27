"""
Control Mailbox for dual-stream convergence.

Based on ARCHITECTURE.md Section 3.2.2

Implements the control plane side of dual-stream convergence:
- Store pre-pushed control info
- Support put/get with alignment ID
- Implement backpressure mechanism

Key technical points:
1. Unique Alignment ID: Data and control converge via batch_id + global_step
2. Credit-based Flow Control: Prevent speed mismatch between Controller and data
3. Sidecar Agent: Host CPU receives Controller push, writes to shared memory
"""

from typing import Any, Dict, Optional
import asyncio

from .fluxway.common.types import ControlConfig


class ControlMailbox:
    """
    Control Mailbox for storing pre-pushed control configurations.

    Supports:
    - put(batch_id, config): Write config, wake up waiters
    - get(batch_id, timeout): Get config, throw on timeout
    - Backpressure: Trigger when exceeding max_cache_size
    """

    def __init__(self, max_cache_size: int = 100):
        self._configs: Dict[str, ControlConfig] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._max_cache_size = max_cache_size
        self._consumed_step: int = 0

    async def put(self, batch_id: str, config: ControlConfig) -> None:
        """
        Write config, wake up waiters.

        Args:
            batch_id: Unique batch identifier
            config: Control configuration

        TODO: Store config
        TODO: Signal waiting event if exists
        TODO: Check cache size for backpressure
        """
        pass

    async def get(self, batch_id: str, timeout: Optional[float] = None) -> ControlConfig:
        """
        Get config, wait if not available.

        Args:
            batch_id: Unique batch identifier
            timeout: Optional timeout in seconds

        Returns:
            Control configuration

        Raises:
            TimeoutError: If timeout exceeded

        TODO: Check if config exists
        TODO: If not, create event and wait
        TODO: Return config and clean up
        """
        pass

    def get_nowait(self, batch_id: str) -> Optional[ControlConfig]:
        """
        Get config without waiting.

        Returns None if not available.

        TODO: Check and return config
        """
        pass

    def has_config(self, batch_id: str) -> bool:
        """Check if config is available for batch_id."""
        return batch_id in self._configs

    def report_consumed(self, step: int) -> None:
        """
        Report consumed step for flow control.

        TODO: Update consumed_step
        TODO: Clear old configs
        """
        pass

    def get_pending_count(self) -> int:
        """Get number of pending configs."""
        return len(self._configs)

    def should_apply_backpressure(self) -> bool:
        """
        Check if backpressure should be applied.

        TODO: Check against max_cache_size
        """
        pass

    def clear(self) -> None:
        """Clear all cached configs."""
        self._configs.clear()
        self._events.clear()
