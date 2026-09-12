"""Swipe detection and the motion gate.

A swipe detector's job is mostly refusal: hands move across the camera all day
without meaning anything by it. These tests spend most of their effort on what
must *not* fire.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import hand_for

from airwave.config import parse_config
from airwave.events import EventKind
from airwave.vision.landmarks import HandFrame
from airwave.vision.pipeline import VisionPipeline
from airwave.vision.swipe import SwipeDetector, palm_center

FPS = 30.0


def glide(detector, *, dx=0.0, dy=0.0, frames=8, start=(0.5, 0.5), t0=0.0, pose="open_palm"):
    """Move a hand in a straight line, one landmark set per frame."""
    out = []
    for i in range(frames):
        fraction = i / max(frames - 1, 1)
        hand = hand_for(pose, position=(start[0] + dx * fraction, start[1] + dy * fraction))
        out.append(detector.push(hand, now=t0 + i / FPS))
    return [s for s in out if s is not None]


def test_a_fast_horizontal_sweep_is_a_swipe():
    swipes = glide(SwipeDetector(), dx=0.45, frames=8)
    assert [s.direction for s in swipes] == ["swipe_right"]


def test_direction_follows_the_hand():
    assert glide(SwipeDetector(), dx=-0.45)[0].direction == "swipe_left"
    assert glide(SwipeDetector(), dy=-0.45)[0].direction == "swipe_up"
    assert glide(SwipeDetector(), dy=0.45)[0].direction == "swipe_down"


def test_a_short_movement_is_not_a_swipe():
    """Repositioning your hand a few centimetres is not a command."""
    assert glide(SwipeDetector(), dx=0.10, frames=8) == []


def test_a_slow_movement_is_not_a_swipe():
    """Same distance, drifted over two seconds instead of a quarter of one."""
    assert glide(SwipeDetector(), dx=0.45, frames=60) == []


def test_a_diagonal_movement_is_not_a_swipe():
    """Real swipes commit to an axis; a diagonal is usually just reaching."""
    assert glide(SwipeDetector(), dx=0.35, dy=0.35, frames=8) == []


def test_a_stationary_hand_never_swipes():
    assert glide(SwipeDetector(), dx=0.0, dy=0.0, frames=60) == []


def test_the_refractory_period_stops_a_double_fire():
    detector = SwipeDetector(refractory_ms=700)
    assert len(glide(detector, dx=0.45, t0=0.0)) == 1
    assert glide(detector, dx=0.45, t0=0.4) == []
    assert len(glide(detector, dx=0.45, t0=1.6)) == 1


def test_history_is_cleared_at_the_moment_a_swipe_fires():
    """Without this the tail of one swipe seeds the next."""
    detector = SwipeDetector()
    for i in range(8):
        hand = hand_for("open_palm", position=(0.5 + 0.45 * i / 7, 0.5))
        if detector.push(hand, now=i / FPS) is not None:
            assert len(detector._history) == 0
            return
    pytest.fail("no swipe fired")


def test_losing_the_hand_resets_the_detector():
    """A swipe cannot span a gap where the hand was not visible."""
    detector = SwipeDetector()
    for i in range(4):
        detector.push(hand_for("open_palm", position=(0.3 + i * 0.05, 0.5)), now=i / FPS)
    detector.push(None, now=5 / FPS)
    assert detector.speed == 0.0
    assert detector.push(hand_for("open_palm", position=(0.75, 0.5)), now=6 / FPS) is None


def test_thresholds_are_configurable():
    lenient = SwipeDetector(min_distance=0.05, min_velocity=0.1)
    assert len(glide(lenient, dx=0.10, frames=8)) == 1
    strict = SwipeDetector(min_distance=0.6)
    assert glide(strict, dx=0.45, frames=8) == []


def test_swipe_reports_distance_and_velocity():
    swipe = glide(SwipeDetector(), dx=0.45, frames=8)[0]
    assert swipe.distance > 0.2
    assert swipe.velocity > 0.9
    assert swipe.ended_at > swipe.started_at


def test_palm_centre_is_steadier_than_a_fingertip():
    """The anchor must not move just because the fingers changed shape."""
    fist = palm_center(hand_for("fist"))
    palm = palm_center(hand_for("open_palm"))
    assert abs(fist[0] - palm[0]) < 0.05
    assert abs(fist[1] - palm[1]) < 0.05


# --------------------------------------------------------------- integration


def config(**settings):
    cfg, issues = parse_config({
        "settings": {"stable_frames": 3, "audio": {"enabled": False}, **settings},
        "bindings": [
            {"name": "Play", "trigger": {"type": "gesture", "value": "open_palm"},
             "action": {"type": "key", "keys": ["playpause"]}},
            {"name": "Next slide", "trigger": {"type": "motion", "value": "swipe_left"},
             "action": {"type": "key", "keys": ["right"]}},
        ],
    })
    assert not issues, issues
    return cfg


def run(cfg, positions, pose="open_palm", clock_start=100.0):
    """Drive the pipeline with a hand at a series of positions."""
    times = iter(clock_start + i / FPS for i in range(len(positions) + 5))
    current = {"t": clock_start}

    def clock():
        return current["t"]

    pipeline = VisionPipeline(cfg, clock=clock)
    events = []
    for i, pos in enumerate(positions):
        current["t"] = clock_start + i / FPS
        hand = HandFrame(hand_for(pose, position=pos), handedness="Right")
        events.extend(pipeline.process(hand).events)
    return events


def straight_line(dx, frames):
    return [(0.5 + dx * i / max(frames - 1, 1), 0.5) for i in range(frames)]


def test_pipeline_emits_a_motion_event():
    events = run(config(), straight_line(-0.45, 8))
    motions = [e for e in events if e.kind is EventKind.MOTION]
    assert [e.value for e in motions] == ["swipe_left"]


def test_the_motion_gate_suppresses_the_pose_during_a_swipe():
    """Regression: swiping with an open palm used to fire Play/pause as well,
    because the hand is briefly a stable open palm at both ends of the sweep."""
    events = run(config(), straight_line(-0.45, 10))
    gestures = [e for e in events if e.kind is EventKind.GESTURE]
    assert gestures == [], f"pose leaked through the motion gate: {[e.value for e in gestures]}"


def test_a_pose_still_fires_once_the_hand_settles():
    """The gate must suppress during motion, not permanently."""
    positions = straight_line(-0.45, 8) + [(0.05, 0.5)] * 10
    events = run(config(), positions)
    assert [e.value for e in events if e.kind is EventKind.MOTION] == ["swipe_left"]
    assert [e.value for e in events if e.kind is EventKind.GESTURE] == ["open_palm"]


def test_swipes_can_be_disabled():
    events = run(config(swipe={"enabled": False}), straight_line(-0.45, 8))
    assert [e for e in events if e.kind is EventKind.MOTION] == []


def test_disabling_swipes_also_disables_the_gate():
    """With no swipe detector there is no speed to gate on, so a moving palm
    is classified normally - which is the pre-swipe behaviour."""
    positions = straight_line(-0.45, 10)
    events = run(config(swipe={"enabled": False}), positions)
    assert [e.value for e in events if e.kind is EventKind.GESTURE] == ["open_palm"]


def test_motion_gate_threshold_is_configurable():
    """A very high gate lets the pose through even while the hand travels."""
    events = run(config(swipe={"motion_gate": 9.0}), straight_line(-0.45, 10))
    assert [e.value for e in events if e.kind is EventKind.GESTURE] == ["open_palm"]
