"""Hand-written geometric classifier - the zero-setup default (PRD v1).

Two ideas carry most of the reliability here:

**Handedness-free tests.** Extension is measured as "is the tip further from
the wrist than the joint below it", not "is the tip above the joint". The
former survives a rotated hand, a left hand and a mirrored preview; the latter
is the reason most webcam gesture demos only work for right-handed users
holding their hand perfectly upright.

**Hysteresis.** A finger sitting exactly at the extension boundary flickers
between states at 30fps, and every flicker resets the stability window - the
user holds a pose and nothing ever fires. Each finger therefore has separate
"becomes extended" and "becomes curled" thresholds, so a borderline finger
keeps whatever state it already had (PRD risk table: ring/pinky reliability).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..gestures import NONE_LABEL
from .classifier import Classification
from .normalize import (
    FINGER_PIPS,
    FINGER_TIPS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    THUMB_IP,
    THUMB_TIP,
    WRIST,
    normalize,
    to_array,
)

EXTEND_ON = 1.10
"""Tip must be 10% further from the wrist than the PIP to count as extended."""

EXTEND_OFF = 0.95
"""...and must fall back below 95% to count as curled. The gap is the hysteresis."""

THUMB_ON = 1.18
THUMB_OFF = 1.02

PINCH_FORWARD = 1.15
"""How far in front of the knuckles the thumb/index contact point must sit,
as a multiple of the wrist-to-palm distance. Measured values: a closed fist
sits at ~0.85, a deliberate pinch at ~1.9."""

PINCH_RELATIVE = 0.10
"""Thumb-to-index gap as a fraction of hand size. The config threshold is in
frame units because that is what the PRD specifies and what a user can reason
about - but frame units shrink with distance from the camera, so a hand far
enough away has *every* fingertip within 4% of frame width. This second,
scale-free test is what keeps a distant thumbs-up from reading as a pinch."""


class RuleClassifier:
    """Geometric classifier over normalized landmarks."""

    name = "rules"

    def __init__(self, *, pinch_threshold: float = 0.04, hysteresis: bool = True) -> None:
        self.pinch_threshold = pinch_threshold
        self.hysteresis = hysteresis
        self._state: list[bool] = [False] * 5

    def reset(self) -> None:
        self._state = [False] * 5

    # ------------------------------------------------------------------ parts

    def _extension_ratios(self, pts: np.ndarray) -> np.ndarray:
        """Per-finger tip/joint distance ratio, measured from the wrist.

        The thumb uses the pinky MCP as its reference instead of the wrist: a
        tucked thumb still sits far from the wrist, but it collapses onto the
        palm, so distance-from-the-far-side-of-the-palm separates the two
        cases cleanly and without needing to know which hand this is.
        """
        wrist = pts[WRIST, :2]
        ratios = np.zeros(5, dtype=np.float32)

        pinky_mcp = pts[PINKY_MCP, :2]
        thumb_tip_d = float(np.linalg.norm(pts[THUMB_TIP, :2] - pinky_mcp))
        thumb_ip_d = float(np.linalg.norm(pts[THUMB_IP, :2] - pinky_mcp))
        ratios[0] = thumb_tip_d / max(thumb_ip_d, 1e-6)

        for i in range(1, 5):
            tip_d = float(np.linalg.norm(pts[FINGER_TIPS[i], :2] - wrist))
            pip_d = float(np.linalg.norm(pts[FINGER_PIPS[i], :2] - wrist))
            ratios[i] = tip_d / max(pip_d, 1e-6)
        return ratios

    def _extended(self, ratios: np.ndarray) -> tuple[list[bool], float]:
        """Apply hysteresis and report the weakest decision margin."""
        out: list[bool] = []
        margins: list[float] = []
        for i, ratio in enumerate(ratios):
            on, off = (THUMB_ON, THUMB_OFF) if i == 0 else (EXTEND_ON, EXTEND_OFF)
            previous = self._state[i] if self.hysteresis else False
            if previous:
                state = ratio > off
                margins.append(abs(ratio - off))
            else:
                state = ratio > on
                margins.append(abs(ratio - on))
            out.append(bool(state))
        weakest = min(margins) if margins else 0.0
        # Squash the margin into 0..1; 0.15 of ratio is a comfortably clear call.
        return out, float(min(1.0, weakest / 0.15))

    def _spread(self, pts: np.ndarray) -> dict[str, float]:
        """Gaps between adjacent fingertips - what separates spock from palm."""
        tips = pts[[INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP], :2]
        gaps = np.linalg.norm(np.diff(tips, axis=0), axis=1)
        return {"index_middle": float(gaps[0]), "middle_ring": float(gaps[1]), "ring_pinky": float(gaps[2])}

    # ------------------------------------------------------------------ entry

    def classify(self, landmarks: Any, *, handedness: str | None = None, aspect: float = 0.75) -> Classification:
        """Label one frame.

        ``aspect`` is frame height / width. MediaPipe normalizes x by width and
        y by height, so a raw distance in that space is stretched on non-square
        frames; scaling y by the aspect makes the pinch threshold mean what the
        PRD says it means - a fraction of frame *width*.
        """
        if landmarks is None:
            self.reset()
            return Classification(label=NONE_LABEL)

        raw = to_array(landmarks)
        pts = normalize(raw, handedness=handedness, rotate=False)

        ratios = self._extension_ratios(pts)
        extended, margin = self._extended(ratios)
        self._state = list(extended)
        thumb, index, middle, ring, pinky = extended

        delta = raw[THUMB_TIP, :2] - raw[INDEX_TIP, :2]
        pinch_dist = float(math.hypot(delta[0], delta[1] * aspect))
        forward = self._pinch_is_forward(pts)
        relative_gap = float(np.linalg.norm(pts[THUMB_TIP, :2] - pts[INDEX_TIP, :2]))
        # Three conditions, each covering a failure the others miss:
        #   frame distance  - the user-facing threshold from the PRD
        #   relative gap    - survives the user sitting further from the camera
        #   forward of palm - a closed fist also brings thumb and index together
        pinching = (
            pinch_dist < self.pinch_threshold
            and relative_gap < PINCH_RELATIVE
            and forward >= PINCH_FORWARD
        )

        spread = self._spread(pts)
        hand_span = float(np.linalg.norm(pts[INDEX_MCP, :2] - pts[PINKY_MCP, :2])) or 1e-6

        details = {
            "ratios": [round(float(r), 3) for r in ratios],
            "pinch_dist": round(pinch_dist, 4),
            "pinch_forward": round(forward, 3),
            "pinch_relative": round(relative_gap, 3),
            "spread": {k: round(v, 3) for k, v in spread.items()},
        }

        label = self._label(
            thumb=thumb, index=index, middle=middle, ring=ring, pinky=pinky,
            pinching=pinching, pts=pts, spread=spread, hand_span=hand_span,
        )
        confidence = margin
        if label in ("pinch", "ok"):
            # A pinch that is well inside the threshold is a confident pinch.
            confidence = float(min(1.0, max(0.0, 1.0 - pinch_dist / max(self.pinch_threshold, 1e-6))))
        return Classification(
            label=label,
            confidence=confidence,
            fingers=(thumb, index, middle, ring, pinky),
            details=details,
        )

    def _label(self, *, thumb, index, middle, ring, pinky, pinching, pts, spread, hand_span) -> str:
        """The decision tree, ordered most-specific first.

        Order matters: 'ok' is a pinch with three fingers out, so pinch must be
        tested before finger counting, and 'ok' before bare pinch.
        """
        if pinching:
            return "ok" if (middle and ring and pinky) else "pinch"

        count = sum((thumb, index, middle, ring, pinky))

        if count == 0:
            return "fist"

        if count == 1:
            if thumb:
                return "thumbs_up" if self._thumb_is_up(pts) else "fist"
            if index:
                return "one"
            return "fist" if not (middle or ring or pinky) else "fist"

        if thumb and index and count == 2:
            return "l_shape" if self._thumb_index_angle(pts) > math.radians(50) else "two"

        if index and pinky and not middle and not ring:
            return "rock"

        if index and middle and not ring and not pinky:
            return "two"

        if index and middle and ring and not pinky:
            return "three"

        if index and middle and ring and pinky:
            # Spock splits between middle and ring; a flat palm spreads evenly.
            middle_gap = spread["middle_ring"]
            neighbours = max(spread["index_middle"], spread["ring_pinky"], 1e-6)
            if middle_gap > 1.7 * neighbours and middle_gap > 0.35 * hand_span:
                return "spock"
            return "open_palm" if thumb else "four"

        return "fist"

    @staticmethod
    def _pinch_is_forward(pts: np.ndarray) -> float:
        """Distance of the thumb/index contact point from the wrist, relative
        to the knuckle line. Above 1 means the contact is in front of the palm."""
        palm = pts[[INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2].mean(axis=0)
        contact = (pts[THUMB_TIP, :2] + pts[INDEX_TIP, :2]) / 2.0
        palm_d = float(np.linalg.norm(palm))
        if palm_d < 1e-6:
            return 0.0
        return float(np.linalg.norm(contact) / palm_d)

    @staticmethod
    def _thumb_is_up(pts: np.ndarray) -> bool:
        """Vertical enough to be a deliberate thumbs-up (FR-1.3).

        Without this, a hand resting sideways with a relaxed thumb reads as a
        confirm gesture, which is the worst possible false positive to have on
        an Enter key.
        """
        v = pts[THUMB_TIP, :2] - pts[WRIST, :2]
        norm = float(np.linalg.norm(v))
        if norm < 1e-6:
            return False
        # Image Y grows downward, so "up" is negative Y.
        return (-v[1] / norm) > 0.6

    @staticmethod
    def _thumb_index_angle(pts: np.ndarray) -> float:
        a = pts[THUMB_TIP, :2] - pts[WRIST, :2]
        b = pts[INDEX_TIP, :2] - pts[WRIST, :2]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
