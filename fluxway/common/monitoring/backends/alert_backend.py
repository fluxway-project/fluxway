"""
Alert backend for notification system.

Provides interface for alert SDK integration.
"""

from typing import Dict, Optional, Any, List
from enum import Enum
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertData:
    """
    Alert data structure for SDK integration.

    Fields:
        alert_id: Unique alert identifier
        level: Alert severity level (INFO, WARNING, ERROR, CRITICAL)
        title: Alert title/summary
        message: Detailed alert message
        component_type: Type of component (worker, spmd, task)
        component_id: Component identifier (worker_id, spmd_id, etc.)
        timestamp: Alert timestamp (Unix timestamp)
        metadata: Additional metadata dictionary
        metrics: Related metrics snapshot (optional)
    """

    alert_id: str
    level: AlertLevel
    title: str
    message: str
    component_type: str  # "worker", "spmd", "task"
    component_id: str
    timestamp: float
    metadata: Dict[str, Any]
    metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SDK."""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "component_type": self.component_type,
            "component_id": self.component_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "metrics": self.metrics,
        }


class AlertBackend:
    """
    Alert backend for notification system.

    Provides interface for alert SDK integration.
    Users can implement their own SDK integration by subclassing this class.
    """

    def __init__(self, sdk_client: Optional[Any] = None):
        """
        Initialize Alert backend.

        Args:
            sdk_client: User's alert SDK client instance (optional)
        """
        self.sdk_client = sdk_client
        self._alert_history: List[AlertData] = []
        self._max_history = 1000  # Keep last 1000 alerts

    def send_alert(self, alert_data: AlertData) -> bool:
        """
        Send alert via SDK.

        Args:
            alert_data: AlertData instance containing alert information

        Returns:
            True if alert sent successfully, False otherwise

        TODO: Implement SDK integration
        - If sdk_client is provided, call SDK method to send alert
        - SDK method signature should be: sdk_client.send_alert(alert_data.to_dict())
        - Handle SDK errors gracefully
        - Log alert for debugging
        """
        # Add to history
        self._alert_history.append(alert_data)
        if len(self._alert_history) > self._max_history:
            self._alert_history.pop(0)

        # Log alert
        logger.warning(
            f"Alert [{alert_data.level.value}]: {alert_data.title} - "
            f"{alert_data.component_type}:{alert_data.component_id}"
        )

        # TODO: Implement SDK integration
        # if self.sdk_client is not None:
        #     try:
        #         # Call user's SDK
        #         result = self.sdk_client.send_alert(alert_data.to_dict())
        #         return result is not None
        #     except Exception as e:
        #         logger.error(f"Failed to send alert via SDK: {e}")
        #         return False

        return False

    def send_worker_alert(
        self,
        spmd_id: str,
        worker_id: str,
        level: AlertLevel,
        title: str,
        message: str,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send worker-level alert.

        Args:
            spmd_id: SPMD identifier
            worker_id: Worker identifier
            level: Alert level
            title: Alert title
            message: Alert message
            metrics: Related metrics (optional)
            metadata: Additional metadata (optional)

        Returns:
            True if alert sent successfully
        """
        alert_data = AlertData(
            alert_id=f"worker_{spmd_id}_{worker_id}_{int(time.time())}",
            level=level,
            title=title,
            message=message,
            component_type="worker",
            component_id=f"{spmd_id}:{worker_id}",
            timestamp=time.time(),
            metadata=metadata or {},
            metrics=metrics,
        )
        return self.send_alert(alert_data)

    def send_spmd_alert(
        self,
        spmd_id: str,
        level: AlertLevel,
        title: str,
        message: str,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send SPMD-level alert.

        Args:
            spmd_id: SPMD identifier
            level: Alert level
            title: Alert title
            message: Alert message
            metrics: Related metrics (optional)
            metadata: Additional metadata (optional)

        Returns:
            True if alert sent successfully
        """
        alert_data = AlertData(
            alert_id=f"spmd_{spmd_id}_{int(time.time())}",
            level=level,
            title=title,
            message=message,
            component_type="spmd",
            component_id=spmd_id,
            timestamp=time.time(),
            metadata=metadata or {},
            metrics=metrics,
        )
        return self.send_alert(alert_data)

    def send_task_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send task-level alert.

        Args:
            level: Alert level
            title: Alert title
            message: Alert message
            metrics: Related metrics (optional)
            metadata: Additional metadata (optional)

        Returns:
            True if alert sent successfully
        """
        alert_data = AlertData(
            alert_id=f"task_{int(time.time())}",
            level=level,
            title=title,
            message=message,
            component_type="task",
            component_id="task",
            timestamp=time.time(),
            metadata=metadata or {},
            metrics=metrics,
        )
        return self.send_alert(alert_data)

    def get_alert_history(
        self,
        component_type: Optional[str] = None,
        level: Optional[AlertLevel] = None,
        limit: int = 100,
    ) -> List[AlertData]:
        """
        Get alert history.

        Args:
            component_type: Filter by component type (optional)
            level: Filter by alert level (optional)
            limit: Maximum number of alerts to return

        Returns:
            List of AlertData instances
        """
        filtered = self._alert_history

        if component_type:
            filtered = [a for a in filtered if a.component_type == component_type]

        if level:
            filtered = [a for a in filtered if a.level == level]

        return filtered[-limit:]
