"""End-to-end: landmarks in, actions out.

This is the regression suite the PRD asks for. Recorded clips are the eventual
input, but landmark sequences are the same thing with the MediaPipe step
already paid for - the suite runs in milliseconds, on CI, with no webcam and
no model download, and every fixed false positive can be added as a sequence.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FakeClock, hand_for

from airwave.config import parse_config
from airwave.dispatch.actions import ActionExecutor
from airwave.dispatch.dispatcher import OUTCOME_FIRED, Dispatcher
from airwave.dispatch.platform import InputBackend
from airwave.events import EventKind
from airwave.vision.landmarks import HandFrame
from airwave.vision.pipeline import VisionPipeline


def config(settings=None, bindings=None):
    cfg, issues = parse_config({
        "settings": {"stable_frames": 5, "cooldown_ms": 800, "audio": {"enabled": False}, **(settings or {})},
        "bindings": bindings if bindings is not None else [
            {"name": "Play", "trigger": {"type": "gesture", "value": "open_palm"},
             "action": {"type": "key", "keys": ["playpause"]}},
            {"name": "Vol up", "trigger": {"type": "gesture", "value": "one"},
             "action": {"type": "key", "keys": ["volumeup"], "repeat": 3}},
        ],
    })
    assert not issues, issues
    return cfg


def frames(*spec):
    """``frames(("open_palm", 10), (None, 3))`` -> a list of HandFrames."""
    out = []
    for label, count in spec:
        for _ in range(count):
            out.append(HandFrame(None) if label is None else
                       HandFrame(hand_for(label), handedness="Right", score=0.95))
    return out


def run(cfg, sequence, clock=None):
    pipeline = VisionPipeline(cfg, clock=clock or FakeClock())
    events = []
    for hand in sequence:
        events.extend(pipeline.process(hand).events)
    return pipeline, events


def gestures(events):
    return [e.value for e in events if e.kind is EventKind.GESTURE]


# ------------------------------------------------------------------ vision


def test_a_held_pose_produces_exactly_one_event():
    _, events = run(config(), frames((None, 3), ("open_palm", 20), (None, 3)))
    assert gestures(events) == ["open_palm"]


def test_a_hand_waving_through_poses_emits_only_stable_ones():
    """Fingers passing through 'one' and 'two' on the way to a palm are not
    commands - this is the ordinary motion that a naive classifier acts on."""
    _, events = run(config(), frames(
        (None, 4), ("one", 2), ("two", 2), ("three", 2), ("open_palm", 12), ("fist", 8),
    ))
    assert gestures(events) == ["open_palm"]


def test_returning_to_neutral_lets_the_same_gesture_fire_again():
    _, events = run(config(), frames(
        ("open_palm", 10), ("fist", 10), ("open_palm", 10),
    ))
    assert gestures(events) == ["open_palm", "open_palm"]


def test_an_empty_frame_run_emits_nothing():
    _, events = run(config(), frames((None, 60)))
    assert events == []


def test_jitter_between_two_labels_never_fires():
    cfg = config()
    pipeline = VisionPipeline(cfg, clock=FakeClock())
    events = []
    for i in range(60):
        hand = HandFrame(hand_for("one" if i % 2 else "two"), handedness="Right")
        events.extend(pipeline.process(hand).events)
    assert events == []


def test_classification_survives_a_moving_hand():
    """Same pose, drifting across the frame - normalization should absorb it."""
    cfg = config()
    pipeline = VisionPipeline(cfg, clock=FakeClock())
    events = []
    for i in range(20):
        hand = hand_for("open_palm", position=(0.25 + i * 0.02, 0.4 + i * 0.01))
        events.extend(pipeline.process(HandFrame(hand, handedness="Right")).events)
    assert gestures(events) == ["open_palm"]


def test_stability_progress_is_reported_for_the_overlay():
    cfg = config()
    pipeline = VisionPipeline(cfg, clock=FakeClock())
    progress = [pipeline.process(HandFrame(hand_for("one"), handedness="Right")).stability
                for _ in range(5)]
    assert progress == [0.2, 0.4, 0.6, 0.8, 1.0]


# ------------------------------------------------------------------ arming


def test_arming_emits_a_system_event_after_the_hold(clock):
    cfg = config({"arming": {"enabled": True, "gesture": "thumbs_up", "hold_ms": 1000}})
    pipeline = VisionPipeline(cfg, clock=clock)
    events = []
    for _ in range(6):
        events.extend(pipeline.process(HandFrame(hand_for("thumbs_up"), handedness="Right")).events)
        clock.advance(0.033)
    assert not [e for e in events if e.kind is EventKind.SYSTEM]

    clock.advance(1.1)
    events.extend(pipeline.process(HandFrame(hand_for("thumbs_up"), handedness="Right")).events)
    assert [e.value for e in events if e.kind is EventKind.SYSTEM] == ["arm"]


def test_arming_fires_once_per_hold_not_every_frame(clock):
    cfg = config({"arming": {"enabled": True, "gesture": "thumbs_up", "hold_ms": 200}})
    pipeline = VisionPipeline(cfg, clock=clock)
    events = []
    for _ in range(40):
        events.extend(pipeline.process(HandFrame(hand_for("thumbs_up"), handedness="Right")).events)
        clock.advance(0.033)
    assert len([e for e in events if e.kind is EventKind.SYSTEM]) == 1


# ------------------------------------------------------------------ pointer


def test_mouse_mode_turns_a_pinch_into_a_click_not_a_gesture():
    cfg = config({"mouse": {"enabled": True, "pinch_click": True}})
    pipeline = VisionPipeline(cfg, clock=FakeClock())
    pipeline.set_mouse_active(True)
    events = []
    for _ in range(8):
        events.extend(pipeline.process(HandFrame(hand_for("pinch"), handedness="Right")).events)
    clicks = [e for e in events if e.kind is EventKind.POINTER and e.value == "click"]
    assert len(clicks) == 1
    assert gestures(events) == []


def test_mouse_mode_emits_pointer_moves_while_a_hand_is_visible():
    cfg = config({"mouse": {"enabled": True}})
    pipeline = VisionPipeline(cfg, clock=FakeClock())
    pipeline.set_mouse_active(True)
    result = pipeline.process(HandFrame(hand_for("one"), handedness="Right"))
    moves = [e for e in result.events if e.kind is EventKind.POINTER and e.value == "move"]
    assert len(moves) == 1
    assert 0.0 <= moves[0].payload["x"] <= 1.0


def test_no_pointer_events_when_mouse_mode_is_off():
    _, events = run(config(), frames(("one", 10)))
    assert [e for e in events if e.kind is EventKind.POINTER] == []


# ------------------------------------------------- full pipeline + dispatcher


def test_landmarks_to_keypress(clock):
    """The whole path: hand -> classify -> stabilize -> dispatch -> keypress."""
    cfg = config()
    backend = InputBackend(dry_run=True)
    dispatcher = Dispatcher(cfg, executor=ActionExecutor(backend), clock=clock)
    pipeline = VisionPipeline(cfg, clock=clock)

    for hand in frames((None, 3), ("open_palm", 10), ("fist", 6), ("one", 10)):
        for event in pipeline.process(hand).events:
            dispatcher.handle(event)
        clock.advance(0.033)

    assert backend.log == ["press playpause x1", "press volumeup x3"]


def test_cooldown_stops_a_rapid_repeat_end_to_end(clock):
    cfg = config({"stable_frames": 2, "cooldown_ms": 2000})
    backend = InputBackend(dry_run=True)
    dispatcher = Dispatcher(cfg, executor=ActionExecutor(backend), clock=clock)
    pipeline = VisionPipeline(cfg, clock=clock)

    for hand in frames(("open_palm", 3), ("fist", 3), ("open_palm", 3)):
        for event in pipeline.process(hand).events:
            dispatcher.handle(event)
        clock.advance(0.033)

    assert backend.log == ["press playpause x1"], "second palm was inside the cooldown"


def test_arming_blocks_the_action_until_armed(clock):
    cfg = config({"arming": {"enabled": True, "gesture": "thumbs_up", "hold_ms": 300}})
    backend = InputBackend(dry_run=True)
    dispatcher = Dispatcher(cfg, executor=ActionExecutor(backend), clock=clock)
    pipeline = VisionPipeline(cfg, clock=clock)

    def feed(sequence, step=0.033):
        for hand in sequence:
            for event in pipeline.process(hand).events:
                dispatcher.handle(event)
            clock.advance(step)

    feed(frames(("open_palm", 10)))
    assert backend.log == [], "unarmed gestures must not act"

    feed(frames(("thumbs_up", 20)))
    feed(frames(("fist", 6), ("open_palm", 10)))
    assert backend.log == ["press playpause x1"]


def test_hot_reload_changes_behaviour_mid_stream(clock):
    """FR-5.4: a new binding takes effect without restarting the camera."""
    cfg = config()
    backend = InputBackend(dry_run=True)
    dispatcher = Dispatcher(cfg, executor=ActionExecutor(backend), clock=clock)
    pipeline = VisionPipeline(cfg, clock=clock)

    def feed(sequence):
        for hand in sequence:
            for event in pipeline.process(hand).events:
                dispatcher.handle(event)
            clock.advance(0.033)

    feed(frames(("two", 10)))
    assert backend.log == []

    updated = config(bindings=[
        {"name": "Mute", "trigger": {"type": "gesture", "value": "two"},
         "action": {"type": "key", "keys": ["volumemute"]}},
    ])
    dispatcher.apply_config(updated)
    pipeline.apply_config(updated)

    feed(frames(("fist", 8), ("two", 10)))
    assert backend.log == ["press volumemute x1"]


def test_reload_that_tightens_stability_takes_effect(clock):
    cfg = config({"stable_frames": 2})
    pipeline = VisionPipeline(cfg, clock=clock)
    pipeline.apply_config(config({"stable_frames": 10}))
    _, events = [], []
    for hand in frames(("one", 5)):
        events.extend(pipeline.process(hand).events)
    assert events == []


# -------------------------------------------------------------- performance


def test_frame_budget_is_met_without_mediapipe():
    """The PRD budgets p95 < 50ms for the whole frame. MediaPipe dominates
    that; this asserts that everything Airwave adds stays negligible."""
    cfg = config()
    pipeline = VisionPipeline(cfg, clock=None or __import__("time").monotonic)
    hand = HandFrame(hand_for("open_palm"), handedness="Right")
    for _ in range(300):
        pipeline.process(hand)
    assert pipeline.p95_frame_ms() < 5.0, f"p95 {pipeline.p95_frame_ms():.2f}ms"


def test_fps_is_reported():
    cfg = config()
    pipeline = VisionPipeline(cfg)
    hand = HandFrame(hand_for("fist"), handedness="Right")
    for _ in range(10):
        pipeline.process(hand)
    assert pipeline.fps > 0


def test_enabling_mouse_via_hot_reload_takes_effect():
    """Regression: the reload path only ever turned mouse mode *off*, so
    flipping `mouse.enabled` to true in the file did nothing until restart."""
    pipeline = VisionPipeline(config(), clock=FakeClock())
    assert pipeline.mouse_active is False

    pipeline.apply_config(config({"mouse": {"enabled": True}}))
    assert pipeline.mouse_active is True

    result = pipeline.process(HandFrame(hand_for("one"), handedness="Right"))
    assert [e for e in result.events if e.kind is EventKind.POINTER]


def test_disabling_mouse_via_hot_reload_takes_effect():
    pipeline = VisionPipeline(config({"mouse": {"enabled": True}}), clock=FakeClock())
    assert pipeline.mouse_active is True
    pipeline.apply_config(config({"mouse": {"enabled": False}}))
    assert pipeline.mouse_active is False


def test_an_unrelated_reload_does_not_undo_a_runtime_toggle():
    """Pressing 'm' then saving an unrelated config change must not turn the
    cursor back off - that reads as the app randomly forgetting."""
    pipeline = VisionPipeline(config(), clock=FakeClock())
    pipeline.set_mouse_active(True)
    pipeline.apply_config(config({"cooldown_ms": 1234}))
    assert pipeline.mouse_active is True
