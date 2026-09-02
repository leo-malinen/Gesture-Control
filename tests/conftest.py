"""Shared fixtures: synthetic hands, so the suite needs no camera.

Every test that involves a hand pose builds one here rather than loading a
recorded fixture. The point is not to fake MediaPipe's output exactly - it is
to produce landmark geometry with the *relationships* the classifier reasons
about (tip further from the wrist than the joint below it, thumb clear of the
palm), then verify the classifier reads them the way a person would.

That makes the whole suite run in under a second on CI with no webcam, and it
makes a failure legible: if ``two`` starts reading as ``three``, the geometry
that broke it is right here in this file.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Canonical right hand, wrist at the origin, fingers pointing "up" (-Y, since
# image coordinates grow downward). Units are fractions of hand size.
_MCP = {"index": (-0.09, -0.32), "middle": (-0.02, -0.34), "ring": (0.05, -0.32), "pinky": (0.12, -0.29)}

_EXTENDED_OFFSETS = ((0.0, -0.11), (0.0, -0.19), (0.0, -0.27))
"""PIP, DIP, TIP offsets from the MCP for an extended finger."""

_CURLED_OFFSETS = ((0.0, -0.09), (0.02, -0.04), (0.02, 0.03))
"""A curled finger: PIP still forward, then the tip folds back toward the palm,
which is exactly the tip-closer-than-PIP relationship the rules test for."""

_THUMB_TUCKED = ((-0.05, -0.05), (-0.08, -0.10), (-0.10, -0.16), (-0.04, -0.24))
_THUMB_OUT = ((-0.07, -0.04), (-0.14, -0.08), (-0.22, -0.14), (-0.34, -0.12))
_THUMB_UP = ((-0.05, -0.08), (-0.07, -0.18), (-0.08, -0.30), (-0.10, -0.44))


def synth_hand(
    *,
    thumb: str = "tucked",
    index: bool = False,
    middle: bool = False,
    ring: bool = False,
    pinky: bool = False,
    spread: float = 0.0,
    pinch: bool = False,
    position: tuple[float, float] = (0.5, 0.55),
    scale: float = 0.9,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Build a (21, 3) landmark array in frame-normalized coordinates.

    ``scale`` is relative to a hand that spans roughly a third of the frame;
    ``position`` places the wrist. Both exist so tests can assert that
    normalization actually removes them.
    """
    points = np.zeros((21, 3), dtype=np.float32)

    thumb_chain = {"tucked": _THUMB_TUCKED, "out": _THUMB_OUT, "up": _THUMB_UP}[thumb]
    for i, (x, y) in enumerate(thumb_chain, start=1):
        points[i] = (x, y, 0.0)

    fingers = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    # Spock: the gap opens between middle and ring, not evenly across the hand.
    spread_shift = {"index": -spread, "middle": -spread, "ring": spread, "pinky": spread}
    for slot, (name, extended) in enumerate(fingers.items()):
        base_index = 5 + slot * 4
        mcp = np.array(_MCP[name], dtype=np.float32)
        points[base_index] = (*mcp, 0.0)
        offsets = _EXTENDED_OFFSETS if extended else _CURLED_OFFSETS
        for joint, (dx, dy) in enumerate(offsets, start=1):
            shift = spread_shift[name] * (joint / len(offsets)) if extended else 0.0
            points[base_index + joint] = (mcp[0] + dx + shift, mcp[1] + dy, 0.0)

    if pinch:
        # Thumb tip meets index tip. Distance in *frame* units is what the
        # pinch threshold measures, so this must happen before scaling.
        points[4] = points[8] + np.array([0.004, 0.004, 0.0], dtype=np.float32)

    if rotation_deg:
        theta = math.radians(rotation_deg)
        cos, sin = math.cos(theta), math.sin(theta)
        rot = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
        points[:, :2] = points[:, :2] @ rot.T

    points[:, :2] *= scale * 0.42
    points[:, 0] += position[0]
    points[:, 1] += position[1]
    return points


#: Each vocabulary entry as a set of ``synth_hand`` arguments.
POSES: dict[str, dict] = {
    "fist": {},
    "one": {"index": True},
    "two": {"index": True, "middle": True},
    "three": {"index": True, "middle": True, "ring": True},
    "four": {"index": True, "middle": True, "ring": True, "pinky": True},
    "open_palm": {"thumb": "out", "index": True, "middle": True, "ring": True, "pinky": True},
    "thumbs_up": {"thumb": "up"},
    "l_shape": {"thumb": "out", "index": True},
    "rock": {"index": True, "pinky": True},
    "spock": {"index": True, "middle": True, "ring": True, "pinky": True, "spread": 0.11},
    "pinch": {"index": True, "pinch": True},
    "ok": {"index": True, "middle": True, "ring": True, "pinky": True, "pinch": True},
}


def hand_for(label: str, **overrides) -> np.ndarray:
    """Landmarks for a named gesture, with per-test tweaks."""
    return synth_hand(**{**POSES[label], **overrides})


@pytest.fixture
def hands():
    """All poses, built fresh per test."""
    return {label: hand_for(label) for label in POSES}


@pytest.fixture
def classifier():
    from airwave.vision.rules import RuleClassifier

    return RuleClassifier(pinch_threshold=0.04)


@pytest.fixture
def base_config():
    """A minimal valid config: one gesture binding, no side effects."""
    from airwave.config import parse_config

    cfg, issues = parse_config(
        {
            "settings": {"stable_frames": 3, "cooldown_ms": 500, "audio": {"enabled": False}},
            "bindings": [
                {"name": "Play", "trigger": {"type": "gesture", "value": "open_palm"},
                 "action": {"type": "key", "keys": ["playpause"]}},
            ],
        },
        source="<fixture>",
    )
    assert not issues, issues
    return cfg


@pytest.fixture
def dry_executor():
    """An executor that records what it would have done."""
    from airwave.dispatch.actions import ActionExecutor
    from airwave.dispatch.platform import InputBackend

    return ActionExecutor(InputBackend(dry_run=True))


class FakeClock:
    """Deterministic time, so cooldown and dwell tests are not flaky."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock():
    return FakeClock()
