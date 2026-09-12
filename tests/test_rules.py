"""Rule classifier behaviour, including the failure modes it was hardened against."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import POSES, hand_for

from airwave.gestures import NONE_LABEL
from airwave.vision.rules import RuleClassifier


@pytest.mark.parametrize("label", sorted(POSES))
def test_every_vocabulary_pose_is_recognized(label):
    assert RuleClassifier().classify(hand_for(label)).label == label


@pytest.mark.parametrize("label", sorted(POSES))
def test_recognition_survives_position_and_scale(label):
    """A gesture that only works dead-centre at one distance is not a feature."""
    for position in ((0.2, 0.3), (0.75, 0.7)):
        for scale in (0.6, 1.3):
            result = RuleClassifier().classify(hand_for(label, position=position, scale=scale))
            assert result.label == label, f"{label} failed at {position} scale {scale}"


@pytest.mark.parametrize("label", ["one", "two", "open_palm", "fist"])
def test_recognition_survives_a_tilted_hand(label):
    """Nobody holds their hand perfectly vertical."""
    for angle in (-20, 20):
        assert RuleClassifier().classify(hand_for(label, rotation_deg=angle)).label == label


def test_no_hand_returns_none():
    assert RuleClassifier().classify(None).label == NONE_LABEL


def test_fist_is_not_read_as_pinch():
    """Regression: a curled fist puts the thumb tip within the pinch distance
    of the index tip. Without the forward-of-the-knuckles check it fired a
    click every time the user returned to neutral."""
    result = RuleClassifier().classify(hand_for("fist"))
    assert result.label == "fist"
    assert result.details["pinch_dist"] < 0.04, "the fixture should still be inside the distance threshold"
    assert result.details["pinch_forward"] < 1.15


def test_thumbs_up_requires_a_vertical_thumb():
    """A relaxed sideways thumb must not fire a confirm/Enter binding."""
    sideways = hand_for("thumbs_up", rotation_deg=75)
    assert RuleClassifier().classify(sideways).label != "thumbs_up"


def test_finger_count_matches_the_pose():
    assert RuleClassifier().classify(hand_for("three")).finger_count == 3
    assert RuleClassifier().classify(hand_for("fist")).finger_count == 0


def test_confidence_is_higher_for_unambiguous_poses():
    clear = RuleClassifier().classify(hand_for("open_palm"))
    assert 0.0 <= clear.confidence <= 1.0
    assert clear.confidence > 0.5


def _borderline_index_hand():
    """A hand whose index sits between the on (1.10) and off (0.95) ratios.

    Found by search rather than a magic constant, so the fixture stays valid
    if the thresholds are retuned.
    """
    for blend in np.linspace(0.0, 1.0, 101):
        hand = hand_for("one")
        hand[8] = hand[8] * (1 - blend) + hand[6] * blend
        ratio = RuleClassifier().classify(hand).details["ratios"][1]
        if 0.98 < ratio < 1.08:
            return hand, ratio
    raise AssertionError("no blend lands in the hysteresis band")


def test_hysteresis_holds_a_borderline_finger_steady():
    """A finger parked at the boundary must not flicker; every flicker resets
    the stability window, which is what makes a pose 'never fire'."""
    borderline, ratio = _borderline_index_hand()
    assert 0.95 < ratio < 1.10

    warmed = RuleClassifier()
    warmed.classify(hand_for("one"))  # index becomes extended first
    assert warmed.classify(borderline).fingers[1] is True

    fresh = RuleClassifier()
    assert fresh.classify(borderline).fingers[1] is False


def test_hysteresis_can_be_disabled():
    classifier = RuleClassifier(hysteresis=False)
    classifier.classify(hand_for("open_palm"))
    assert classifier.classify(hand_for("fist")).label == "fist"


def test_reset_clears_finger_state():
    classifier = RuleClassifier()
    classifier.classify(hand_for("open_palm"))
    classifier.reset()
    assert classifier._state == [False] * 5


def test_pinch_threshold_is_configurable():
    """A tighter threshold should stop recognizing a loose pinch."""
    loose = hand_for("pinch")
    loose[4] = loose[8] + np.array([0.015, 0.0, 0.0], dtype=np.float32)
    assert RuleClassifier(pinch_threshold=0.04).classify(loose).label == "pinch"
    assert RuleClassifier(pinch_threshold=0.01).classify(loose).label != "pinch"


def test_left_hand_is_classified_the_same_as_right():
    """Handedness-free geometry is the point; a left-handed user is not a bug."""
    right = hand_for("two")
    left = right.copy()
    left[:, 0] = 1.0 - left[:, 0]
    assert RuleClassifier().classify(left, handedness="Left").label == "two"


def test_spock_needs_a_real_split():
    """Four fingers evenly spread is a palm, not a spock."""
    assert RuleClassifier().classify(hand_for("four")).label == "four"
    assert RuleClassifier().classify(hand_for("spock")).label == "spock"
