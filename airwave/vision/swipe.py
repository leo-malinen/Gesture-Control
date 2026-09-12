"""Swipe detection - the project's first *dynamic* gesture.

Everything else in `vision/` classifies a pose held still, and the stability
window exists to reject motion. A swipe is the exact inverse: it only exists
while the hand is travelling, and it ends when the hand stops. That is why it
is a separate trigger kind rather than another entry in the gesture vocabulary.

No temporal model is involved. A swipe has three measurable properties and all
three must hold, which is what keeps it from firing on ordinary hand movement:

* **Distance** - the palm crossed a real fraction of the frame, not a twitch.
* **Speed** - it crossed quickly. Slowly repositioning your hand is not a swipe.
* **Straightness** - horizontal travel dominated vertical, or the reverse.

The companion idea is the motion gate in the pipeline: while the hand is moving
fast enough to be swiping, pose classification is suppressed. Without it, a
swipe performed with an open palm would fire the swipe *and* whatever open_palm
is bound to, because the hand is briefly an open palm at both ends of the
motion. Suppressing poses during fast travel is also just correct on its own -
a hand flying across the frame is not somebody holding a pose.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .normalize import FINGER_MCPS, WRIST, to_array

SWIPE_LEFT = "swipe_left"
SWIPE_RIGHT = "swipe_right"
SWIPE_UP = "swipe_up"
SWIPE_DOWN = "swipe_down"

MOTION_VALUES: tuple[str, ...] = (SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN)

MOTION_HELP: dict[str, str] = {
    SWIPE_LEFT: "hand sweeps right-to-left across the frame",
    SWIPE_RIGHT: "hand sweeps left-to-right across the frame",
    SWIPE_UP: "hand sweeps upward",
    SWIPE_DOWN: "hand sweeps downward",
}


def palm_center(landmarks) -> tuple[float, float]:
    """Wrist plus the four knuckles, averaged.

    Steadier than a fingertip: the tip of an extended finger swings through a
    much larger arc than the palm when the wrist merely rotates, which reads as
    travel that never happened.
    """
    pts = to_array(landmarks)
    anchors = pts[[WRIST, *FINGER_MCPS[1:]], :2]
    center = anchors.mean(axis=0)
    return float(center[0]), float(center[1])


@dataclass(frozen=True, slots=True)
class Swipe:
    """A completed swipe, with the numbers that justified calling it one."""

    direction: str
    distance: float
    velocity: float
    started_at: float
    ended_at: float


class SwipeDetector:
    """Sliding-window motion detector over the palm centre."""

    def __init__(
        self,
        *,
        min_distance: float = 0.22,
        min_velocity: float = 0.9,
        axis_ratio: float = 1.6,
        refractory_ms: int = 700,
        window_ms: int = 400,
    ) -> None:
        self.min_distance = min_distance
        self.min_velocity = min_velocity
        self.axis_ratio = axis_ratio
        self.refractory_ms = refractory_ms
        self.window_ms = window_ms
        self._history: deque[tuple[float, float, float]] = deque()
        # -inf, not 0: a zero start would put the detector inside its own
        # refractory window for the first `refractory_ms` of the clock. With
        # time.monotonic() that is invisible; with any clock that starts near
        # zero the first swipe of the session is silently swallowed.
        self._last_fire = float("-inf")
        self.speed = 0.0

    def reset(self) -> None:
        """Called when the hand leaves the frame - a swipe cannot span a gap."""
        self._history.clear()
        self.speed = 0.0

    # ------------------------------------------------------------------ input

    def push(self, landmarks, *, now: float | None = None) -> Swipe | None:
        """Feed one frame's landmarks. Returns a swipe when one completes."""
        now = time.monotonic() if now is None else now
        if landmarks is None:
            self.reset()
            return None

        x, y = palm_center(landmarks)
        self._history.append((now, x, y))
        cutoff = now - self.window_ms / 1000.0
        while len(self._history) > 2 and self._history[0][0] < cutoff:
            self._history.popleft()

        self.speed = self._current_speed()
        if len(self._history) < 3:
            return None
        if (now - self._last_fire) * 1000.0 < self.refractory_ms:
            return None

        t0, x0, y0 = self._history[0]
        dt = now - t0
        if dt <= 0:
            return None
        dx, dy = x - x0, y - y0

        horizontal = abs(dx) >= abs(dy) * self.axis_ratio
        vertical = abs(dy) >= abs(dx) * self.axis_ratio
        if not (horizontal or vertical):
            # Diagonal drift: real swipes are committed along one axis.
            return None

        distance = abs(dx) if horizontal else abs(dy)
        if distance < self.min_distance or distance / dt < self.min_velocity:
            return None

        if horizontal:
            # The preview is mirrored, so a hand moving to the user's right
            # increases x - the on-screen direction and the felt direction agree.
            direction = SWIPE_RIGHT if dx > 0 else SWIPE_LEFT
        else:
            # Image Y grows downward.
            direction = SWIPE_DOWN if dy > 0 else SWIPE_UP

        self._last_fire = now
        swipe = Swipe(direction=direction, distance=distance,
                      velocity=distance / dt, started_at=t0, ended_at=now)
        # One motion, one event: keeping the history would let the tail of this
        # swipe seed the next one.
        self._history.clear()
        return swipe

    # ------------------------------------------------------------------ state

    def _current_speed(self) -> float:
        """Palm speed in frame-widths per second, over the last two samples."""
        if len(self._history) < 2:
            return 0.0
        (t0, x0, y0), (t1, x1, y1) = self._history[-2], self._history[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return math.hypot(x1 - x0, y1 - y0) / dt

    @property
    def travelling(self) -> bool:
        """Whether the hand is currently in motion. Drives the pipeline's gate."""
        return self.speed > 0.0

    def is_moving(self, threshold: float) -> bool:
        return self.speed >= threshold
