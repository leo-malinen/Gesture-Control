"""Transient detector for claps and snaps (FR-3.1 - FR-3.3).

A clap is a broadband spike an order of magnitude above the room. The naive
detector - a fixed RMS threshold - fails immediately in practice: it fires
constantly in a noisy room and never fires in a quiet one, and it needs
re-tuning every time the user moves the laptop. So the threshold rides an
adaptive baseline instead, and the baseline only tracks *quiet* frames, which
keeps a long clap from raising the bar enough to suppress its own detection.

The double-clap trade-off is stated plainly because it is a real cost: telling
a single clap from the first half of a double clap requires waiting out the
inter-onset window (up to 600ms). Airwave therefore only pays that latency
when a double-clap binding actually exists.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class Onset:
    """A detected transient, with the numbers that produced it."""

    at: float
    rms: float
    baseline: float
    threshold: float

    @property
    def ratio(self) -> float:
        return self.rms / self.baseline if self.baseline > 1e-9 else float("inf")


@dataclass(slots=True)
class OnsetDetector:
    """Adaptive RMS-spike detector over fixed-size audio blocks."""

    baseline_alpha: float = 0.995
    multiplier: float = 8.0
    floor: float = 0.06
    refractory_ms: int = 180
    baseline: float = 0.01
    last_onset_at: float = 0.0
    last_rms: float = 0.0
    peak_rms: float = 0.0
    history: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    @property
    def threshold(self) -> float:
        """``max(baseline x multiplier, floor)`` - the PRD's rule, verbatim.

        The floor matters in a silent room: an 8x multiple of near-zero noise
        would fire on a keyboard press.
        """
        return max(self.baseline * self.multiplier, self.floor)

    @staticmethod
    def rms(block: np.ndarray) -> float:
        if block.size == 0:
            return 0.0
        data = np.asarray(block, dtype=np.float32).reshape(-1)
        return float(math.sqrt(float(np.mean(np.square(data)))))

    def push(self, block: np.ndarray, *, now: float | None = None) -> Onset | None:
        """Feed one audio block. Returns an onset when one fired.

        Runs inside the sounddevice callback, so it is pure numpy and does no
        allocation beyond a deque append - a blocking callback drops audio.
        """
        now = time.monotonic() if now is None else now
        level = self.rms(block)
        self.last_rms = level
        self.peak_rms = max(self.peak_rms * 0.995, level)
        self.history.append(level)

        threshold = self.threshold
        fired = None
        if level > threshold and (now - self.last_onset_at) * 1000.0 >= self.refractory_ms:
            fired = Onset(at=now, rms=level, baseline=self.baseline, threshold=threshold)
            self.last_onset_at = now

        # Only quiet blocks update the baseline. Letting the transient itself
        # into the average is how adaptive detectors go deaf mid-clap.
        if level < threshold:
            self.baseline = self.baseline_alpha * self.baseline + (1.0 - self.baseline_alpha) * level
        return fired

    def reset(self) -> None:
        self.baseline = 0.01
        self.last_onset_at = 0.0
        self.peak_rms = 0.0
        self.history.clear()

    def snapshot(self) -> dict[str, float]:
        """Live numbers for calibration mode (FR-3.2)."""
        return {
            "rms": self.last_rms,
            "baseline": self.baseline,
            "threshold": self.threshold,
            "peak": self.peak_rms,
            "headroom": self.last_rms / self.threshold if self.threshold else 0.0,
        }


class ClapGrouper:
    """Turns raw onsets into ``clap`` / ``double_clap`` labels (FR-3.3)."""

    def __init__(
        self,
        *,
        min_gap_ms: int = 150,
        max_gap_ms: int = 600,
        detect_double: bool = True,
    ) -> None:
        self.min_gap_ms = min_gap_ms
        self.max_gap_ms = max_gap_ms
        self.detect_double = detect_double
        self._pending: Onset | None = None

    def feed(self, onset: Onset | None, *, now: float | None = None) -> list[tuple[str, float]]:
        """Feed one onset (or ``None`` to just advance time).

        Returns ``(label, captured_at)`` pairs. ``captured_at`` is the time of
        the *first* onset so latency is measured from when the user clapped,
        not from when we finished deciding what it was.
        """
        now = time.monotonic() if now is None else now
        out: list[tuple[str, float]] = []

        if self._pending is not None:
            gap_ms = (now - self._pending.at) * 1000.0
            if onset is not None and self.min_gap_ms <= gap_ms <= self.max_gap_ms:
                out.append(("double_clap", self._pending.at))
                self._pending = None
                return out
            if gap_ms > self.max_gap_ms:
                out.append(("clap", self._pending.at))
                self._pending = None

        if onset is not None:
            if not self.detect_double:
                # No double-clap binding: fire immediately and keep the sub-100ms
                # latency the PRD budgets for the fast audio channel.
                out.append(("clap", onset.at))
            else:
                self._pending = onset
        return out

    def reset(self) -> None:
        self._pending = None

    @property
    def waiting(self) -> bool:
        return self._pending is not None
