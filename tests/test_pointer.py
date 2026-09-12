"""Cursor smoothing and mapping (PRD 4.4).

The claim being tested is specific: the One Euro filter must remove jitter
while a hand is still *and* not add meaningful lag while it moves. A test that
only checked smoothing would pass for a filter that makes the cursor unusable.
"""

from __future__ import annotations

import numpy as np
import pytest

from airwave.vision.pointer import OneEuroFilter, PointerController


def test_filter_passes_the_first_sample_through():
    assert OneEuroFilter()(0.5, 0.0) == 0.5


def test_filter_removes_jitter_from_a_stationary_signal():
    rng = np.random.default_rng(0)
    filt = OneEuroFilter(min_cutoff=0.4, beta=3.0)
    raw, smoothed = [], []
    t = 0.0
    for _ in range(120):
        t += 1 / 30
        sample = 0.5 + rng.normal(0, 0.01)
        raw.append(sample)
        smoothed.append(filt(sample, t))
    raw_jitter = float(np.std(np.diff(raw[30:])))
    smooth_jitter = float(np.std(np.diff(smoothed[30:])))
    assert smooth_jitter < raw_jitter / 4


def test_filter_keeps_up_with_fast_motion():
    """The adaptive part: heavy smoothing when still, light when moving.

    Regression: with a pixel-scale beta this lagged a quarter of the screen
    behind the hand, because the velocity term is tiny in 0..1 units.
    """
    filt = OneEuroFilter(min_cutoff=0.4, beta=3.0)
    t, position = 0.0, 0.0
    for _ in range(30):
        t += 1 / 30
        position += 0.03
        out = filt(position, t)
    assert out == pytest.approx(position, abs=0.05)


def test_beta_controls_the_lag_speed_tradeoff():
    def travel(beta: float) -> float:
        filt = OneEuroFilter(min_cutoff=0.4, beta=beta)
        t, position, out = 0.0, 0.0, 0.0
        for _ in range(15):
            t += 1 / 30
            position += 0.03
            out = filt(position, t)
        return out

    assert travel(6.0) > travel(0.0)


def test_zero_or_negative_dt_does_not_explode():
    filt = OneEuroFilter()
    filt(0.5, 1.0)
    assert filt(0.9, 1.0) == 0.5


def test_reset_forgets_history():
    filt = OneEuroFilter()
    filt(0.1, 0.0)
    filt(0.1, 0.1)
    filt.reset()
    assert filt(0.9, 0.2) == 0.9


# ------------------------------------------------------------- region mapping


def test_active_region_maps_to_the_full_screen():
    """FR-4.3: the middle of the frame reaches every screen corner."""
    controller = PointerController(active_region=(0.25, 0.25, 0.75, 0.75))
    assert controller.map_region(0.25, 0.25) == (0.0, 0.0)
    assert controller.map_region(0.75, 0.75) == (1.0, 1.0)
    assert controller.map_region(0.5, 0.5) == (0.5, 0.5)


def test_outside_the_region_clamps_rather_than_overshooting():
    controller = PointerController(active_region=(0.25, 0.25, 0.75, 0.75))
    assert controller.map_region(0.0, 0.0) == (0.0, 0.0)
    assert controller.map_region(1.0, 1.0) == (1.0, 1.0)


def test_invert_x_flips_the_axis():
    controller = PointerController(active_region=(0.0, 0.0, 1.0, 1.0), invert_x=True)
    assert controller.map_region(0.2, 0.5)[0] == pytest.approx(0.8)


def test_update_returns_values_inside_the_screen():
    controller = PointerController(smoothing=0.6)
    sample = controller.update(0.5, 0.5, 0.0)
    assert 0.0 <= sample.x <= 1.0 and 0.0 <= sample.y <= 1.0


def test_smoothing_dial_makes_the_cursor_steadier():
    def spread(smoothing: float) -> float:
        rng = np.random.default_rng(1)
        controller = PointerController(smoothing=smoothing, active_region=(0.0, 0.0, 1.0, 1.0))
        out, t = [], 0.0
        for _ in range(90):
            t += 1 / 30
            out.append(controller.update(0.5 + rng.normal(0, 0.01), 0.5, t).x)
        return float(np.std(out[30:]))

    assert spread(0.9) < spread(0.0)


# --------------------------------------------------------------------- dwell


def test_dwell_click_fires_once_after_the_hold():
    controller = PointerController(dwell_click=True, dwell_ms=500, dwell_radius=0.02,
                                   active_region=(0.0, 0.0, 1.0, 1.0), smoothing=0.0)
    clicks, t = 0, 0.0
    for _ in range(45):
        t += 1 / 30
        if controller.update(0.5, 0.5, t).click:
            clicks += 1
    assert clicks == 1


def test_dwell_resets_when_the_cursor_moves_away():
    controller = PointerController(dwell_click=True, dwell_ms=500, dwell_radius=0.02,
                                   active_region=(0.0, 0.0, 1.0, 1.0), smoothing=0.0)
    t = 0.0
    for i in range(45):
        t += 1 / 30
        x = 0.5 + (0.3 if i % 5 == 0 else 0.0)
        assert controller.update(x, 0.5, t).click is False


def test_dwell_progress_is_reported_for_the_overlay():
    controller = PointerController(dwell_click=True, dwell_ms=1000, dwell_radius=0.02,
                                   active_region=(0.0, 0.0, 1.0, 1.0), smoothing=0.0)
    t = 0.0
    progress = 0.0
    for _ in range(15):
        t += 1 / 30
        progress = controller.update(0.5, 0.5, t).dwell_progress
    assert 0.0 < progress < 1.0


def test_dwell_is_off_unless_enabled():
    controller = PointerController(dwell_click=False, active_region=(0.0, 0.0, 1.0, 1.0))
    t = 0.0
    for _ in range(60):
        t += 1 / 30
        assert controller.update(0.5, 0.5, t).click is False
