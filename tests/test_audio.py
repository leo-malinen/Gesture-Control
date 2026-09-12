"""Clap detection, double-clap grouping and fuzzy phrase matching (PRD 4.3)."""

from __future__ import annotations

import numpy as np
import pytest

from airwave.audio.matching import best_match, contains_wake_word, normalize_phrase, similarity
from airwave.audio.onset import ClapGrouper, OnsetDetector


def noise(level: float, n: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * level).astype(np.float32)


def settle(detector: OnsetDetector, level: float = 0.004, blocks: int = 400, t0: float = 0.0) -> float:
    """Run room tone through the detector so its baseline converges."""
    now = t0
    for i in range(blocks):
        now = t0 + i * 0.032
        detector.push(noise(level, seed=i), now=now)
    return now


# ------------------------------------------------------------------- onsets


def test_room_tone_alone_never_fires():
    detector = OnsetDetector(floor=0.06)
    fired = [detector.push(noise(0.01, seed=i), now=i * 0.032) is not None for i in range(300)]
    assert not any(fired)


def test_a_clap_fires():
    detector = OnsetDetector(floor=0.06)
    now = settle(detector)
    assert detector.push(noise(0.5, seed=999), now=now + 0.032) is not None


def test_the_floor_stops_a_silent_room_from_firing_on_a_whisper():
    """8x almost-nothing is still almost nothing; without a floor, typing fires."""
    detector = OnsetDetector(floor=0.06, multiplier=8.0)
    now = settle(detector, level=0.0005)
    assert detector.push(noise(0.02, seed=7), now=now + 0.032) is None


def test_the_baseline_adapts_to_a_louder_room():
    """A noisy cafe must not fire continuously - the bar rises with the room."""
    quiet = OnsetDetector(floor=0.0)
    settle(quiet, level=0.005)
    loud = OnsetDetector(floor=0.0)
    settle(loud, level=0.05)
    assert loud.threshold > quiet.threshold * 5


def test_the_transient_itself_does_not_raise_the_baseline():
    """Regression: letting the clap into the average makes the detector deaf
    to the second clap in a double-clap."""
    detector = OnsetDetector(floor=0.05)
    now = settle(detector)
    before = detector.baseline
    detector.push(noise(0.8, seed=1), now=now + 0.032)
    assert detector.baseline == pytest.approx(before, rel=0.01)


def test_refractory_period_blocks_the_echo_of_one_clap():
    detector = OnsetDetector(floor=0.06, refractory_ms=180)
    now = settle(detector)
    assert detector.push(noise(0.5, seed=2), now=now + 0.03) is not None
    assert detector.push(noise(0.5, seed=3), now=now + 0.08) is None
    assert detector.push(noise(0.5, seed=4), now=now + 0.30) is not None


def test_snapshot_exposes_what_calibration_mode_prints():
    detector = OnsetDetector()
    settle(detector)
    snap = detector.snapshot()
    assert {"rms", "baseline", "threshold", "peak", "headroom"} <= snap.keys()
    assert snap["threshold"] > 0


def test_rms_of_silence_is_zero():
    assert OnsetDetector.rms(np.zeros(512, dtype=np.float32)) == 0.0


# --------------------------------------------------------------- clap grouping


def make_onset(at: float):
    from airwave.audio.onset import Onset

    return Onset(at=at, rms=0.5, baseline=0.01, threshold=0.08)


def test_single_clap_is_immediate_when_double_clap_is_not_bound():
    """The fast path: no double-clap binding means no waiting (FR-3.6)."""
    grouper = ClapGrouper(detect_double=False)
    assert grouper.feed(make_onset(1.0), now=1.0) == [("clap", 1.0)]


def test_single_clap_waits_out_the_window_when_double_clap_is_bound():
    grouper = ClapGrouper(detect_double=True, max_gap_ms=600)
    assert grouper.feed(make_onset(1.0), now=1.0) == []
    assert grouper.feed(None, now=1.3) == []
    assert grouper.feed(None, now=1.7) == [("clap", 1.0)]


def test_two_quick_claps_become_one_double_clap():
    grouper = ClapGrouper(detect_double=True, min_gap_ms=150, max_gap_ms=600)
    grouper.feed(make_onset(1.0), now=1.0)
    assert grouper.feed(make_onset(1.3), now=1.3) == [("double_clap", 1.0)]


def test_two_slow_claps_are_two_single_claps():
    grouper = ClapGrouper(detect_double=True, min_gap_ms=150, max_gap_ms=600)
    grouper.feed(make_onset(1.0), now=1.0)
    out = grouper.feed(make_onset(2.0), now=2.0)
    assert out == [("clap", 1.0)]
    assert grouper.feed(None, now=2.7) == [("clap", 2.0)]


def test_double_clap_timestamp_is_the_first_clap():
    """Latency must be measured from when the user clapped, not from when we
    finished deciding what it was."""
    grouper = ClapGrouper(detect_double=True)
    grouper.feed(make_onset(5.0), now=5.0)
    label, at = grouper.feed(make_onset(5.25), now=5.25)[0]
    assert (label, at) == ("double_clap", 5.0)


# ------------------------------------------------------------- phrase matching


@pytest.mark.parametrize("heard,phrase", [
    ("volume up", "volume up"),
    ("volume app", "volume up"),
    ("next rack", "next track"),
    ("Switch, window!", "switch window"),
    ("uh switch window please", "switch window"),
])
def test_near_misses_still_match(heard, phrase):
    assert similarity(heard, phrase) >= 0.72, similarity(heard, phrase)


@pytest.mark.parametrize("heard,phrase", [
    ("what time is it", "volume up"),
    ("open the door", "next track"),
    ("volume", "volume up down mute"),
])
def test_unrelated_phrases_do_not_match(heard, phrase):
    assert similarity(heard, phrase) < 0.72


def test_best_match_picks_the_closest_of_several():
    phrases = ["volume up", "volume down", "next track"]
    match = best_match("volume down please", phrases)
    assert match is not None and match[0] == "volume down"


def test_best_match_returns_none_below_threshold():
    assert best_match("completely different", ["volume up"]) is None


def test_normalize_phrase_strips_case_punctuation_and_accents():
    assert normalize_phrase("  Volume, Up!  ") == "volume up"
    assert normalize_phrase("café") == "cafe"


def test_wake_word_is_found_anywhere_in_the_transcript():
    assert contains_wake_word("hey airwave switch window", "hey airwave")
    assert contains_wake_word("ok so hey airwave next track", "hey airwave")


def test_wake_word_tolerates_a_mishearing():
    assert contains_wake_word("hey air wave switch window", "hey airwave") or \
           contains_wake_word("hey airwaive", "hey airwave")


def test_wake_word_absent_means_absent():
    assert not contains_wake_word("switch window", "hey airwave")


def test_empty_inputs_are_safe():
    assert similarity("", "volume up") == 0.0
    assert not contains_wake_word("", "hey airwave")
    assert best_match("", ["volume up"]) is None
