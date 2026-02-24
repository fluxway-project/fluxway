"""Operator Manager module."""

from .operator_manager import OperatorManager
from ..mailbox import (
    OpType,
)  # Re-export if needed, or just let users import from mailbox

__all__ = [OpType, OperatorManager]
