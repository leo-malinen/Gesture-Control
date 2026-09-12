"""Overlay rendering.

Drawing code is easy to leave untested and easy to break - an IndexError in a
draw call takes down the vision thread, which is the main thread. These render
onto a numpy array and never call ``imshow``, so they run headless on CI.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeClock, hand_for

from airwave.config import parse_config
from airwave.dispatch.actions import ActionExecutor
from airwave.dispatch.dispatcher import Dispatcher
from airwave.dispatch.platform import InputBackend
from airwave.events import Event, EventKind
from airwave.ui.overlay import HAND_CONNECTIONS, Overlay
from airwave.vision.landmarks import HandFrame
from airwave.vision.pipeline import VisionPipeline


def build(settings=None):
    cfg, issues = parse_config({
        "settings": {"audio": {"enabled": False}, **(settings or {})},
        "bindings": [{"name": "Play", "trigger": {"type": "gesture", "value": "open_palm"},
                      "action": {"type": "key", "keys": ["playpause"]}}],
    })
    assert not issues, issues
    clock = FakeClock()
    dispatcher = Dispatcher(cfg, executor=ActionExecutor(InputBackend(dry_run=True)), clock=clock)
    return cfg, dispatcher, VisionPipeline(cfg, clock=clock)


def blank(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize("settings", [
    {},
    {"arming": {"enabled": True}},
    {"mouse": {"enabled": True}},
    {"speech": {"enabled": True}},
])
def test_render_draws_without_error_in_each_mode(settings):
    cfg, dispatcher, pipeline = build(settings)
    result = pipeline.process(HandFrame(hand_for("open_palm"), handedness="Right", score=0.9))
    frame = blank()
    out = Overlay().render(frame, result, dispatcher,
                           extra={"p95_ms": 12.0, "mouse_active": True,
                                  "audio": {"rms": 0.01, "threshold": 0.06}})
    assert out.shape == frame.shape
    assert out.any(), "the overlay should have drawn something"


def test_render_handles_no_hand():
    cfg, dispatcher, pipeline = build()
    result = pipeline.process(HandFrame(None))
    assert Overlay().render(blank(), result, dispatcher, extra=None) is not None


def test_render_includes_the_event_log():
    cfg, dispatcher, pipeline = build()
    dispatcher.handle(Event(kind=EventKind.GESTURE, value="open_palm"))
    result = pipeline.process(HandFrame(hand_for("open_palm"), handedness="Right"))
    before = blank()
    after = Overlay().render(before.copy(), result, dispatcher, extra=None)
    assert not np.array_equal(before, after)


def test_render_survives_a_small_frame():
    """A camera negotiating down to 320x240 must not crash the draw path."""
    cfg, dispatcher, pipeline = build()
    result = pipeline.process(HandFrame(hand_for("one"), handedness="Right"))
    assert Overlay().render(blank(320, 240), result, dispatcher, extra=None) is not None


def test_help_overlay_renders():
    cfg, dispatcher, pipeline = build()
    overlay = Overlay()
    overlay.state.show_help = True
    result = pipeline.process(HandFrame(hand_for("fist"), handedness="Right"))
    assert overlay.render(blank(), result, dispatcher, extra=None) is not None


def test_landmarks_can_be_hidden():
    cfg, dispatcher, pipeline = build()
    overlay = Overlay(show_landmarks=False)
    result = pipeline.process(HandFrame(hand_for("open_palm"), handedness="Right"))
    assert overlay.render(blank(), result, dispatcher, extra=None) is not None


def test_hand_topology_is_complete():
    """21 landmarks, 20 bones plus the palm closure."""
    assert len(HAND_CONNECTIONS) == 21
    referenced = {i for pair in HAND_CONNECTIONS for i in pair}
    assert referenced == set(range(21))
