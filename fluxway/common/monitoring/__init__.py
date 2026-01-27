"""
Unified monitoring module for Fluxway.

Provides a facade interface for monitoring functionality to minimize code intrusion.
"""

from .facade import MonitoringFacade
from .aggregator import MetricsAggregator

__all__ = [
    "MonitoringFacade",
    "MetricsAggregator",
]
