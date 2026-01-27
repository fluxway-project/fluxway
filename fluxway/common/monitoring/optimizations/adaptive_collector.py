"""
Adaptive metrics collector with dynamic collection frequency.

Adjusts collection frequency based on system load to balance
monitoring overhead and data freshness.
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AdaptiveMetricsCollector:
    """
    Adaptive metrics collector.

    Dynamically adjusts collection frequency based on CPU load.
    """

    def __init__(
        self,
        base_interval: float = 5.0,
        min_interval: float = 1.0,
        max_interval: float = 30.0,
        high_load_threshold: float = 0.8,
        low_load_threshold: float = 0.3,
        adjustment_factor: float = 1.5,
    ):
        """
        Initialize adaptive collector.

        Args:
            base_interval: Base collection interval in seconds (default: 5.0)
            min_interval: Minimum collection interval (default: 1.0)
            max_interval: Maximum collection interval (default: 30.0)
            high_load_threshold: CPU load threshold for high load (default: 0.8)
            low_load_threshold: CPU load threshold for low load (default: 0.3)
            adjustment_factor: Factor for adjusting interval (default: 1.5)
        """
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.high_load_threshold = high_load_threshold
        self.low_load_threshold = low_load_threshold
        self.adjustment_factor = adjustment_factor

        self.current_interval = base_interval
        self._last_cpu_load: Optional[float] = None
        self._last_adjustment_time = time.time()
        self._adjustment_cooldown = 10.0  # Don't adjust more than once per 10 seconds

    def adjust_interval(self, cpu_load: float) -> float:
        """
        Adjust collection interval based on CPU load.

        Args:
            cpu_load: Current CPU load (0.0 - 1.0)

        Returns:
            Adjusted interval in seconds
        """
        current_time = time.time()

        # Cooldown check
        if (current_time - self._last_adjustment_time) < self._adjustment_cooldown:
            return self.current_interval

        # Adjust based on load
        if cpu_load > self.high_load_threshold:
            # High load: increase interval (reduce frequency)
            new_interval = min(self.current_interval * self.adjustment_factor, self.max_interval)
        elif cpu_load < self.low_load_threshold:
            # Low load: decrease interval (increase frequency)
            new_interval = max(self.current_interval / self.adjustment_factor, self.min_interval)
        else:
            # Normal load: gradually return to base interval
            if self.current_interval > self.base_interval:
                new_interval = max(
                    self.current_interval / self.adjustment_factor, self.base_interval
                )
            elif self.current_interval < self.base_interval:
                new_interval = min(
                    self.current_interval * self.adjustment_factor, self.base_interval
                )
            else:
                new_interval = self.current_interval

        # Only update if changed significantly
        if abs(new_interval - self.current_interval) > 0.5:
            old_interval = self.current_interval
            self.current_interval = new_interval
            self._last_cpu_load = cpu_load
            self._last_adjustment_time = current_time

            logger.debug(
                f"Adjusted collection interval: {old_interval:.1f}s -> "
                f"{new_interval:.1f}s (CPU load: {cpu_load:.2f})"
            )

        return self.current_interval

    def get_current_interval(self) -> float:
        """Get current collection interval."""
        return self.current_interval

    def reset_to_base(self) -> None:
        """Reset interval to base value."""
        self.current_interval = self.base_interval
        self._last_adjustment_time = time.time()

    def get_stats(self) -> dict:
        """Get collector statistics."""
        return {
            "current_interval": self.current_interval,
            "base_interval": self.base_interval,
            "min_interval": self.min_interval,
            "max_interval": self.max_interval,
            "last_cpu_load": self._last_cpu_load,
            "last_adjustment_time": self._last_adjustment_time,
        }
