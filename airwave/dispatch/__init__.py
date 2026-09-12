"""The dispatcher owns every side effect in Airwave."""

from .actions import ActionExecutor, ActionResult, Controls
from .dispatcher import DispatchRecord, Dispatcher, DispatchStats
from .platform import InputBackend, PlatformInfo, detect, describe

__all__ = [
    "ActionExecutor",
    "ActionResult",
    "Controls",
    "DispatchRecord",
    "DispatchStats",
    "Dispatcher",
    "InputBackend",
    "PlatformInfo",
    "describe",
    "detect",
]
