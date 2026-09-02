"""Fingertip-to-cursor mapping (PRD 4.4).

Raw landmark positions are unusable as a cursor: MediaPipe jitters a couple of
pixels frame to frame even on a perfectly still hand, and a cursor that
vibrates cannot be aimed. The fix is a One Euro filter, which is specifically
designed for this trade-off - heavy smoothing while the hand is nearly still
(kills jitter) and almost none while it moves fast (kills lag). A plain
exponential filter has to pick one or the other.

The active region is the second half of making this usable: the middle 50% of
the frame maps to the whole screen, so the corners are reachable with a wrist
movement instead of a full arm extension (FR-4.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..events import PointerSample


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez, Roussel & Vogel, 2012).

    ``min_cutoff`` sets how much jitter is removed when stationary; ``beta``
    sets how quickly smoothing is abandoned as speed rises.

    Units matter more than they look: the published beta values assume pixels,
    where a moving cursor has a velocity in the hundreds. Airwave filters in
    screen *fractions*, where a brisk hand sweep is under 2.0/second - so a
    pixel-tuned beta of 0.02 contributes ~0.02Hz to the cutoff, the adaptive
    term does nothing at all, and the cursor lags a quarter of the screen
    behind a fast movement. The default here is scaled for 0..1 input.
    """

    def __init__(self, *, min_cutoff: float = 1.0, beta: float = 3.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, t: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev, self._t_prev = x, t
            return x
        dt = t - self._t_prev
        if dt <= 0:
            return self._x_prev
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat


@dataclass(slots=True)
class _Dwell:
    """Click-by-hovering, for users who cannot hold a pinch (FR-4.4)."""

    radius: float
    duration_ms: float
    anchor: tuple[float, float] | None = None
    since: float = 0.0
    fired: bool = False

    def update(self, x: float, y: float, t: float) -> tuple[bool, float]:
        if self.anchor is None or math.hypot(x - self.anchor[0], y - self.anchor[1]) > self.radius:
            self.anchor = (x, y)
            self.since = t
            self.fired = False
            return False, 0.0
        elapsed_ms = (t - self.since) * 1000.0
        progress = min(1.0, elapsed_ms / self.duration_ms)
        if progress >= 1.0 and not self.fired:
            self.fired = True
            return True, 1.0
        return False, progress

    def reset(self) -> None:
        self.anchor = None
        self.fired = False
        self.since = 0.0


class PointerController:
    """Maps a fingertip in frame space to a smoothed screen position."""

    def __init__(
        self,
        *,
        active_region: tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75),
        smoothing: float = 0.6,
        min_cutoff: float = 1.0,
        beta: float = 3.0,
        dwell_click: bool = False,
        dwell_ms: int = 900,
        dwell_radius: float = 0.02,
        invert_x: bool = False,
    ) -> None:
        self.active_region = active_region
        self.invert_x = invert_x
        # `smoothing` is the user-facing dial; it scales the filter's resting
        # cutoff so one number in YAML means "steadier" without exposing Hz.
        effective_cutoff = max(0.05, min_cutoff * (1.0 - min(smoothing, 0.95)))
        self._fx = OneEuroFilter(min_cutoff=effective_cutoff, beta=beta)
        self._fy = OneEuroFilter(min_cutoff=effective_cutoff, beta=beta)
        self._dwell = _Dwell(radius=dwell_radius, duration_ms=float(dwell_ms)) if dwell_click else None

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()
        if self._dwell:
            self._dwell.reset()

    def map_region(self, x: float, y: float) -> tuple[float, float]:
        """Rescale the active sub-region of the frame onto the full screen."""
        x0, y0, x1, y1 = self.active_region
        nx = (x - x0) / max(x1 - x0, 1e-6)
        ny = (y - y0) / max(y1 - y0, 1e-6)
        nx = min(1.0, max(0.0, nx))
        ny = min(1.0, max(0.0, ny))
        if self.invert_x:
            nx = 1.0 - nx
        return nx, ny

    def update(self, x: float, y: float, t: float, *, captured_at: float | None = None) -> PointerSample:
        """One frame of pointer state. ``x``/``y`` are frame-normalized 0..1."""
        nx, ny = self.map_region(x, y)
        sx = self._fx(nx, t)
        sy = self._fy(ny, t)
        click, progress = (False, 0.0)
        if self._dwell is not None:
            click, progress = self._dwell.update(sx, sy, t)
        return PointerSample(
            x=min(1.0, max(0.0, sx)),
            y=min(1.0, max(0.0, sy)),
            captured_at=captured_at if captured_at is not None else t,
            click=click,
            dwell_progress=progress,
        )
