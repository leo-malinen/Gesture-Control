"""Normalization is the load-bearing assumption of the whole vision path.

If the same pose at two screen positions produces two different vectors, then
every classifier downstream is learning position, and the accuracy numbers in
the PRD are unreachable no matter how good the model is (FR-1.2).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import hand_for, synth_hand

from airwave.vision.normalize import (
    FEATURE_DIM,
    WRIST,
    feature_vector,
    finger_angles,
    hand_scale,
    normalize,
    pinch_distance,
    to_array,
)


def test_wrist_moves_to_origin():
    pts = normalize(hand_for("open_palm"))
    assert np.allclose(pts[WRIST], 0.0, atol=1e-6)


def test_same_pose_different_position_is_near_identical():
    left = normalize(hand_for("open_palm", position=(0.15, 0.2)))
    right = normalize(hand_for("open_palm", position=(0.8, 0.75)))
    assert np.max(np.abs(left - right)) < 1e-5


def test_same_pose_different_distance_is_near_identical():
    near = normalize(hand_for("two", scale=1.6))
    far = normalize(hand_for("two", scale=0.5))
    assert np.max(np.abs(near - far)) < 1e-4


def test_different_poses_are_far_apart():
    """The flip side: normalization must not erase the signal it is protecting."""
    palm = normalize(hand_for("open_palm"))
    fist = normalize(hand_for("fist"))
    assert np.max(np.abs(palm - fist)) > 0.2


def test_rotation_invariance_only_when_asked():
    upright = hand_for("two")
    tilted = hand_for("two", rotation_deg=35)
    assert np.max(np.abs(normalize(upright) - normalize(tilted))) > 0.05
    rotated = np.max(np.abs(normalize(upright, rotate=True) - normalize(tilted, rotate=True)))
    assert rotated < 1e-4


def test_left_hand_is_mirrored_onto_right():
    """One model should cover both hands; mirroring is how that happens."""
    right = normalize(hand_for("one"), handedness="Right")
    left_raw = hand_for("one")
    left_raw[:, 0] = 1.0 - left_raw[:, 0]
    left = normalize(left_raw, handedness="Left")
    assert np.max(np.abs(right - left)) < 1e-5


def test_mirroring_can_be_disabled():
    mirrored = normalize(hand_for("one"), handedness="Left")
    plain = normalize(hand_for("one"), handedness="Left", mirror_left=False)
    assert not np.allclose(mirrored, plain)


def test_hand_scale_is_positive_and_scales_linearly():
    small = hand_scale(to_array(hand_for("open_palm", scale=0.5)))
    large = hand_scale(to_array(hand_for("open_palm", scale=1.0)))
    assert small > 0
    assert large == pytest.approx(small * 2, rel=0.01)


def test_feature_vector_shape_and_finiteness():
    features = feature_vector(hand_for("open_palm"))
    assert features.shape == (FEATURE_DIM,)
    assert np.all(np.isfinite(features))


def test_feature_vector_is_position_invariant():
    a = feature_vector(hand_for("three", position=(0.2, 0.3)))
    b = feature_vector(hand_for("three", position=(0.7, 0.8)))
    # 1e-3 rather than 1e-5: the joint-angle features use acos, which is
    # numerically steep near pi (a perfectly straight finger), so float32
    # differences of a few 1e-4 are precision, not position leaking through.
    assert np.max(np.abs(a - b)) < 1e-3


def test_finger_angles_separate_curled_from_extended():
    extended = finger_angles(normalize(hand_for("open_palm")))
    curled = finger_angles(normalize(hand_for("fist")))
    # An extended finger is close to a straight line at the PIP (angle ~ pi).
    assert extended[1] > curled[1]
    assert extended[1] > 2.5


def test_pinch_distance_shrinks_when_pinching():
    assert pinch_distance(hand_for("pinch")) < pinch_distance(hand_for("open_palm"))


def test_to_array_accepts_mediapipe_like_objects():
    class _LM:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    raw = hand_for("one")
    landmarks = [_LM(*row) for row in raw]
    assert np.allclose(to_array(landmarks), raw, atol=1e-6)


def test_to_array_rejects_wrong_shape():
    with pytest.raises(ValueError):
        to_array(np.zeros((5, 3)))


def test_degenerate_hand_does_not_divide_by_zero():
    """All landmarks stacked on one point: real MediaPipe output on a bad frame."""
    pts = normalize(np.zeros((21, 3), dtype=np.float32))
    assert np.all(np.isfinite(pts))
