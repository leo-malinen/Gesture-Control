"""The debounce state machine (PRD 4.2) - the critical path.

A per-frame classifier that is 95% accurate is still wrong about ninety times
a minute at 30fps. Everything that makes Airwave usable rather than a demo
happens here and in the dispatcher's cooldown.

Three rules, in order:

* **Stability** (FR-2.1): a label counts only after N consecutive frames agree.
  Anything shorter than N frames - a hand passing through "two" on its way to
  "open_palm", a single misread frame - never becomes a candidate at all.
* **Transition-only emission** (FR-2.2): an event fires when the stable label
  *changes*. Holding a pose for a minute is one event, not eighteen hundred.
* **Neutral reset**: dropping the hand (``none``) or returning to the neutral
  fist clears the last-emitted label, so showing the same pose twice in a row
  is two deliberate commands rather than one command and one no-op.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from ..gestures import NEUTRAL, NONE_LABEL


@dataclass(frozen=True, slots=True)
class Transition:
    """A stable label change worth turning into an event."""

    label: str
    previous: str
    captured_at: float
    frames: int


class Stabilizer:
    """Turns a noisy per-frame label stream into discrete transitions."""

    def __init__(
        self,
        stable_frames: int = 5,
        *,
        neutral: str = NEUTRAL,
        emit_neutral: bool = False,
        clock=time.monotonic,
    ) -> None:
        self._n = max(1, int(stable_frames))
        self._window: deque[str] = deque(maxlen=self._n)
        self.neutral = neutral
        self.emit_neutral = emit_neutral
        self._clock = clock
        self.stable_label: str = NONE_LABEL
        self.last_emitted: str = NONE_LABEL
        self.stable_since: float = clock()
        self._raw: str = NONE_LABEL

    # ------------------------------------------------------------------ config

    @property
    def stable_frames(self) -> int:
        return self._n

    @stable_frames.setter
    def stable_frames(self, value: int) -> None:
        """Resize on hot reload without losing the frames already collected."""
        value = max(1, int(value))
        if value == self._n:
            return
        kept = list(self._window)[-value:]
        self._n = value
        self._window = deque(kept, maxlen=value)

    # ------------------------------------------------------------------ state

    @property
    def raw_label(self) -> str:
        """The most recent frame's label, stable or not. For the overlay."""
        return self._raw

    @property
    def progress(self) -> float:
        """0..1 - how far the current run is toward becoming stable.

        The overlay renders this as a bar. A user whose gesture never fires can
        see whether they are being classified as something else (bar keeps
        resetting) or simply not held long enough (bar climbing).
        """
        if not self._window:
            return 0.0
        last = self._window[-1]
        run = 0
        for label in reversed(self._window):
            if label != last:
                break
            run += 1
        return min(1.0, run / self._n)

    def held_ms(self, now: float | None = None) -> float:
        """How long the current stable label has been stable, in milliseconds."""
        now = self._clock() if now is None else now
        return max(0.0, (now - self.stable_since) * 1000.0)

    def is_held(self, label: str, ms: float, now: float | None = None) -> bool:
        """Used by arming: 'thumbs_up held for a full second' (FR-2.4)."""
        return self.stable_label == label and self.held_ms(now) >= ms

    # ------------------------------------------------------------------ input

    def push(self, label: str | None, *, captured_at: float | None = None) -> Transition | None:
        """Feed one frame. Returns a transition when one is worth emitting."""
        label = label or NONE_LABEL
        now = self._clock()
        captured_at = now if captured_at is None else captured_at
        self._raw = label
        self._window.append(label)

        if len(self._window) < self._n or any(x != label for x in self._window):
            return None

        if label == self.stable_label:
            return None

        previous = self.stable_label
        self.stable_label = label
        self.stable_since = now

        if label in (NONE_LABEL, self.neutral) and not self.emit_neutral:
            # Returning to neutral is not a command; it is what makes the *next*
            # command legible. Clearing last_emitted is the whole point.
            self.last_emitted = NONE_LABEL if label == NONE_LABEL else self.neutral
            return None

        if label == self.last_emitted:
            return None

        self.last_emitted = label
        return Transition(label=label, previous=previous, captured_at=captured_at, frames=self._n)

    def reset(self) -> None:
        """Full reset - used when the camera reopens or the config reloads."""
        self._window.clear()
        self.stable_label = NONE_LABEL
        self.last_emitted = NONE_LABEL
        self._raw = NONE_LABEL
        self.stable_since = self._clock()

    def feed(self, labels: Iterable[str]) -> list[Transition]:
        """Convenience for tests and the offline video runner."""
        return [t for t in (self.push(label) for label in labels) if t is not None]
