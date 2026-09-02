"""The v1 gesture vocabulary (PRD FR-1.3).

Kept in its own module with no dependencies so that the config validator can
name every legal gesture in an error message without importing OpenCV.
"""

from __future__ import annotations

from typing import Final

NEUTRAL: Final[str] = "fist"
"""Reserved neutral pose (FR-1.4). Returning here guarantees clean transitions
between two distinct commands; without it, going palm -> one looks like a
single noisy transition and the stabilizer can miss the boundary."""

NONE_LABEL: Final[str] = "none"
"""Emitted when no hand is in frame. Never bindable — binding it would fire an
action every time you lower your hand."""

BUILTIN_GESTURES: Final[tuple[str, ...]] = (
    "fist",
    "open_palm",
    "one",
    "two",
    "three",
    "four",
    "thumbs_up",
    "pinch",
    "ok",
    "spock",
    "l_shape",
    "rock",
)

RESERVED_LABELS: Final[frozenset[str]] = frozenset({NONE_LABEL})

#: Human-facing description used by `airwave gestures` and the config error text.
GESTURE_HELP: Final[dict[str, str]] = {
    "fist": "0 fingers extended (neutral pose)",
    "open_palm": "5 fingers extended",
    "one": "index only",
    "two": "index + middle",
    "three": "index + middle + ring",
    "four": "four fingers, thumb tucked",
    "thumbs_up": "thumb only, hand vertical",
    "pinch": "thumb tip touching index tip",
    "ok": "thumb+index pinched, other three extended",
    "spock": "index+middle together, ring+pinky together, split in the middle",
    "l_shape": "thumb + index extended at a right angle",
    "rock": "index + pinky extended",
}
