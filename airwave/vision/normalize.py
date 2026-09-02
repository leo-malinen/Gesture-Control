"""Landmark normalization - the step that decides what a classifier can learn.

MediaPipe hands out coordinates in frame space. Feed those to a classifier and
it learns "index finger up *in the top-left corner*", because position and
distance dominate the variance. Every downstream component in Airwave consumes
normalized landmarks instead (FR-1.2):

1. **Translate** the wrist to the origin - removes where the hand is.
2. **Scale** by the bounding-box diagonal - removes how far away it is.
3. **Mirror** left hands onto right - one model covers both hands instead of
   needing twice the training data.
4. **Rotate** (opt-in) so the wrist->middle-MCP axis points "up" - removes hand
   roll. Left off for the rule classifier, which *needs* absolute orientation
   to tell a thumbs-up from a thumbs-sideways.

Pure numpy on purpose: this module is the one the unit tests hammer, and it
must import on a machine with no camera.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

N_LANDMARKS = 21

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
FINGER_MCPS = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

_EPS = 1e-8


def to_array(landmarks: Any) -> np.ndarray:
    """Coerce whatever the caller has into a (21, 3) float32 array.

    Accepts a MediaPipe ``NormalizedLandmarkList``, a list of objects with
    ``.x/.y/.z``, or anything numpy can read. Keeping this tolerant means the
    tests can hand in plain arrays and never import MediaPipe.
    """
    if landmarks is None:
        raise ValueError("landmarks is None")
    inner = getattr(landmarks, "landmark", landmarks)
    if hasattr(inner, "__len__") and len(inner) and hasattr(inner[0], "x"):
        pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in inner], dtype=np.float32)
    else:
        pts = np.asarray(inner, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] != N_LANDMARKS:
        raise ValueError(f"expected {N_LANDMARKS} landmarks, got shape {pts.shape}")
    if pts.shape[1] == 2:
        pts = np.hstack([pts, np.zeros((N_LANDMARKS, 1), dtype=np.float32)])
    return pts[:, :3].astype(np.float32, copy=False)


def hand_scale(pts: np.ndarray) -> float:
    """Bounding-box diagonal in the XY plane.

    Preferred over "wrist to middle-MCP" because a closed fist barely changes
    the box while it dramatically shortens any single bone length - scaling by
    a bone would make a fist look like a giant hand.
    """
    xy = pts[:, :2]
    span = xy.max(axis=0) - xy.min(axis=0)
    return float(math.hypot(span[0], span[1]))


def mirror_x(pts: np.ndarray) -> np.ndarray:
    """Flip across the X axis so a left hand looks like a right hand."""
    out = pts.copy()
    out[:, 0] = -out[:, 0]
    return out


def _rotation_to_up(pts: np.ndarray) -> np.ndarray:
    """2x2 rotation taking wrist->middle-MCP onto the -Y axis (screen "up")."""
    v = pts[MIDDLE_MCP, :2] - pts[WRIST, :2]
    norm = float(np.linalg.norm(v))
    if norm < _EPS:
        return np.eye(2, dtype=np.float32)
    v = v / norm
    # Angle from -Y (up in image coordinates, where Y grows downward).
    angle = math.atan2(v[0], -v[1])
    c, s = math.cos(-angle), math.sin(-angle)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def normalize(
    landmarks: Any,
    *,
    handedness: str | None = None,
    rotate: bool = False,
    mirror_left: bool = True,
) -> np.ndarray:
    """Return pose-only landmarks: translated, scaled, optionally derotated.

    The same pose held in the top-left corner and the bottom-right corner
    produces near-identical output, which is the property the test suite
    asserts and the reason a classifier trained on twenty samples generalizes.
    """
    pts = to_array(landmarks).copy()
    pts -= pts[WRIST]
    if mirror_left and handedness and handedness.lower().startswith("l"):
        pts = mirror_x(pts)
    if rotate:
        rot = _rotation_to_up(pts)
        pts[:, :2] = pts[:, :2] @ rot.T
    # Scale last, and only after any rotation: the bounding box of a tilted
    # hand is larger than the same hand upright, so scaling first would leave
    # a residual that scales with tilt - the exact thing rotation is meant to
    # remove.
    scale = hand_scale(pts)
    if scale > _EPS:
        pts /= scale
    return pts.astype(np.float32, copy=False)


def finger_angles(pts: np.ndarray) -> np.ndarray:
    """Interior angle at each PIP joint, in radians.

    A curl measure that survives scaling and rotation, and the single most
    informative feature for "is this finger extended" beyond a tip/PIP test.
    """
    out = np.zeros(5, dtype=np.float32)
    for i, (mcp, pip, tip) in enumerate(zip(FINGER_MCPS, FINGER_PIPS, FINGER_TIPS)):
        a = pts[mcp, :2] - pts[pip, :2]
        b = pts[tip, :2] - pts[pip, :2]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < _EPS or nb < _EPS:
            out[i] = math.pi
            continue
        cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        out[i] = math.acos(cos)
    return out


def feature_vector(landmarks: Any, *, handedness: str | None = None) -> np.ndarray:
    """The trained classifier's input (M5).

    63 rotated coordinates plus 19 derived scalars. The derived features are
    redundant in principle - a big enough net would learn them - but with a
    few hundred hand-recorded samples they are the difference between a model
    that generalizes and one that memorizes.
    """
    pts = normalize(landmarks, handedness=handedness, rotate=True)
    tips = pts[list(FINGER_TIPS), :2]
    wrist_dists = np.linalg.norm(tips, axis=1)                       # 5
    adjacent = np.linalg.norm(np.diff(tips, axis=0), axis=1)         # 4
    palm = pts[list(FINGER_MCPS), :2]
    tip_to_mcp = np.linalg.norm(tips - palm, axis=1)                 # 5
    angles = finger_angles(pts)                                      # 5
    return np.concatenate([
        pts.reshape(-1),
        wrist_dists,
        adjacent,
        tip_to_mcp,
        angles,
    ]).astype(np.float32)


FEATURE_DIM = N_LANDMARKS * 3 + 5 + 4 + 5 + 5


def landmark_span(landmarks: Any) -> tuple[float, float, float, float]:
    """Raw-space bounding box (x0, y0, x1, y1), used by the overlay."""
    pts = to_array(landmarks)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def pinch_distance(landmarks: Any) -> float:
    """Thumb-tip to index-tip distance in *frame* units (FR-1.3 uses % of frame).

    Deliberately not normalized by hand size: the PRD defines the pinch
    threshold as a fraction of frame width, which is what the user sees and
    what stays stable as they move closer to or further from the camera than
    the threshold was tuned for.
    """
    pts = to_array(landmarks)
    return float(np.linalg.norm(pts[THUMB_TIP, :2] - pts[INDEX_TIP, :2]))


def stack_features(samples: Sequence[Any], handedness: Sequence[str | None] | None = None) -> np.ndarray:
    """Feature matrix for a dataset. Shape (n, FEATURE_DIM)."""
    hands = handedness or [None] * len(samples)
    return np.stack([feature_vector(s, handedness=h) for s, h in zip(samples, hands)])
