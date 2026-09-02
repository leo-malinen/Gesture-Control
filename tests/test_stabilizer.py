"""Stabilizer semantics (PRD 4.2) - the single most important behaviour here.

"Fires exactly once, when the user meant it, and never otherwise" is the
product. These tests are the executable version of that sentence.
"""

from __future__ import annotations

import pytest
from conftest import FakeClock

from airwave.vision.stabilizer import Stabilizer


def labels(transitions):
    return [t.label for t in transitions]


def test_prd_fixture_sequence_emits_exactly_one_event():
    """The sequence named in the PRD's testing section: none x3, palm x10, none x3."""
    stabilizer = Stabilizer(5)
    events = stabilizer.feed(["none"] * 3 + ["open_palm"] * 10 + ["none"] * 3)
    assert labels(events) == ["open_palm"]


def test_holding_a_pose_forever_is_still_one_event():
    stabilizer = Stabilizer(5)
    assert labels(stabilizer.feed(["open_palm"] * 300)) == ["open_palm"]


def test_a_flicker_shorter_than_the_window_never_fires():
    """A hand passing through 'two' on the way to 'open_palm' is not a command."""
    stabilizer = Stabilizer(5)
    events = stabilizer.feed(["fist"] * 5 + ["two"] * 3 + ["open_palm"] * 8)
    assert labels(events) == ["open_palm"]


def test_distinct_gestures_each_fire():
    stabilizer = Stabilizer(3)
    events = stabilizer.feed(["one"] * 4 + ["two"] * 4 + ["three"] * 4)
    assert labels(events) == ["one", "two", "three"]


def test_neutral_pose_rearms_the_same_gesture():
    """Palm, fist, palm is two deliberate commands - the fist is what makes the
    second one legible (FR-1.4)."""
    stabilizer = Stabilizer(3)
    events = stabilizer.feed(["open_palm"] * 4 + ["fist"] * 4 + ["open_palm"] * 4)
    assert labels(events) == ["open_palm", "open_palm"]


def test_dropping_the_hand_also_rearms():
    stabilizer = Stabilizer(3)
    events = stabilizer.feed(["one"] * 4 + ["none"] * 4 + ["one"] * 4)
    assert labels(events) == ["one", "one"]


def test_neutral_itself_emits_nothing_by_default():
    stabilizer = Stabilizer(3)
    assert stabilizer.feed(["fist"] * 10) == []


def test_neutral_can_be_emitted_when_explicitly_enabled():
    """FR-1.4 allows overriding the neutral pose, with a documented warning."""
    stabilizer = Stabilizer(3, emit_neutral=True)
    assert labels(stabilizer.feed(["fist"] * 10)) == ["fist"]


def test_alternating_noise_never_stabilizes():
    """The worst realistic input: a classifier oscillating between two labels."""
    stabilizer = Stabilizer(5)
    assert stabilizer.feed(["one", "two"] * 50) == []


def test_progress_climbs_then_completes():
    stabilizer = Stabilizer(4)
    seen = []
    for label in ["one"] * 4:
        stabilizer.push(label)
        seen.append(stabilizer.progress)
    assert seen == [0.25, 0.5, 0.75, 1.0]


def test_progress_resets_when_the_label_changes():
    stabilizer = Stabilizer(4)
    stabilizer.feed(["one"] * 3)
    stabilizer.push("two")
    assert stabilizer.progress == 0.25


def test_held_ms_and_is_held_track_the_clock():
    clock = FakeClock()
    stabilizer = Stabilizer(2, clock=clock)
    stabilizer.feed(["thumbs_up"] * 2)
    assert stabilizer.held_ms() == 0
    clock.advance(1.2)
    assert stabilizer.held_ms() == pytest.approx(1200)
    assert stabilizer.is_held("thumbs_up", 1000)
    assert not stabilizer.is_held("open_palm", 1000)


def test_raw_label_reports_the_latest_frame_even_when_unstable():
    stabilizer = Stabilizer(5)
    stabilizer.feed(["fist"] * 5 + ["two"])
    assert stabilizer.raw_label == "two"
    assert stabilizer.stable_label == "fist"


def test_stable_frames_can_be_retuned_at_runtime():
    """Hot reload changes this while the camera runs; it must not need a reset."""
    stabilizer = Stabilizer(5)
    stabilizer.feed(["one"] * 3)
    stabilizer.stable_frames = 3
    assert labels(stabilizer.feed(["one"])) == ["one"]


def test_shrinking_the_window_keeps_recent_frames_only():
    stabilizer = Stabilizer(6)
    stabilizer.feed(["two"] * 3 + ["one"] * 2)
    stabilizer.stable_frames = 2
    assert stabilizer.progress == 1.0


def test_reset_clears_everything():
    stabilizer = Stabilizer(3)
    stabilizer.feed(["one"] * 5)
    stabilizer.reset()
    assert stabilizer.stable_label == "none"
    assert labels(stabilizer.feed(["one"] * 3)) == ["one"]


def test_stable_frames_of_one_fires_immediately():
    """Degenerate but legal config - it must not crash or double-fire."""
    stabilizer = Stabilizer(1)
    assert labels(stabilizer.feed(["one", "one", "one"])) == ["one"]


def test_transition_records_the_previous_label():
    stabilizer = Stabilizer(2)
    stabilizer.feed(["one"] * 2)
    transition = stabilizer.feed(["two"] * 2)[0]
    assert transition.previous == "one"
    assert transition.frames == 2


def test_captured_at_is_carried_through_for_latency_measurement():
    stabilizer = Stabilizer(2, clock=FakeClock())
    stabilizer.push("one", captured_at=5.0)
    transition = stabilizer.push("one", captured_at=5.5)
    assert transition is not None and transition.captured_at == 5.5
