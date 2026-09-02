"""Golden-clip regression harness.

The PRD's testing strategy says every fixed false positive gets a clip added to
the suite. This is the machinery for that: a stored landmark sequence plus the
event list it must produce. Anything in ``tests/fixtures/*.npz`` is picked up
automatically, so adding a regression is a data change, not a code change.

Landmark sequences rather than MP4s, because they are what the pipeline
actually consumes: a clip stored this way has the MediaPipe step already paid
for, so the suite stays under a second and needs no model download on CI.
``airwave.app.replay_video`` runs the same assertions against a real video when
one is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import hand_for

from airwave.config import parse_config
from airwave.events import EventKind
from airwave.vision.landmarks import HandFrame
from airwave.vision.pipeline import VisionPipeline

FIXTURES = Path(__file__).parent / "fixtures"


def save_clip(name: str, sequence, expected, *, settings=None, note: str = "",
              overwrite: bool = False) -> Path:
    """Write a clip fixture. ``sequence`` is a list of labels or ``None``."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / f"{name}.npz"
    if path.exists() and not overwrite:
        return path
    frames = np.stack([
        np.full((21, 3), np.nan, dtype=np.float32) if label is None else hand_for(label)
        for label in sequence
    ])
    np.savez_compressed(
        path,
        frames=frames,
        meta=json.dumps({"expected": expected, "settings": settings or {}, "note": note}),
    )
    return path


def load_clip(path: Path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return data["frames"], meta


def run_clip(frames: np.ndarray, settings: dict) -> list[str]:
    cfg, issues = parse_config({"settings": {"stable_frames": 5, "audio": {"enabled": False}, **settings}})
    assert not issues, issues
    pipeline = VisionPipeline(cfg)
    out: list[str] = []
    for landmarks in frames:
        hand = HandFrame(None) if np.isnan(landmarks).any() else HandFrame(landmarks, handedness="Right")
        for event in pipeline.process(hand).events:
            if event.kind is EventKind.GESTURE:
                out.append(event.value)
    return out


def seed_clips() -> None:
    """Ship a few clips so the harness is exercised even before real footage.

    Each one is a failure mode the design is specifically built to prevent.
    Run at import, not in a fixture: the parametrization below reads the
    directory at collection time, which happens first.

    Existing files are never overwritten, so a real recorded clip committed
    under the same name wins over the synthetic one.
    """
    save_clip(
        "hold_palm_then_drop",
        [None] * 3 + ["open_palm"] * 20 + [None] * 5,
        ["open_palm"],
        note="A held pose is one command, not twenty.",
    )
    save_clip(
        "reach_through_poses",
        [None] * 3 + ["one", "two", "three"] * 2 + ["open_palm"] * 12 + ["fist"] * 6,
        ["open_palm"],
        note="Fingers unfurling pass through other poses; none of them are commands.",
    )
    save_clip(
        "fist_between_two_palms",
        ["open_palm"] * 8 + ["fist"] * 8 + ["open_palm"] * 8,
        ["open_palm", "open_palm"],
        note="The neutral pose is what makes a repeated command legible.",
    )
    save_clip(
        "hand_leaves_and_returns",
        ["one"] * 8 + [None] * 8 + ["one"] * 8,
        ["one", "one"],
        note="Lowering the hand rearms; showing it again is a second command.",
    )
    save_clip(
        "flicker_between_two_labels",
        ["one", "two"] * 40,
        [],
        note="A classifier oscillating between two labels must never stabilize.",
    )
    save_clip(
        "neutral_fist_only",
        ["fist"] * 40,
        [],
        note="Resting in the neutral pose is not a command.",
    )
    save_clip(
        "brief_glimpse",
        [None] * 5 + ["open_palm"] * 3 + [None] * 20,
        [],
        note="A hand appearing for three frames is someone walking past the camera.",
    )


seed_clips()


def clip_paths():
    return sorted(FIXTURES.glob("*.npz"))


def test_fixtures_exist():
    assert clip_paths(), "the seeding fixture should have written clips"


@pytest.mark.parametrize("path", clip_paths(), ids=lambda p: p.stem)
def test_clip_produces_its_expected_events(path):
    frames, meta = load_clip(path)
    assert run_clip(frames, meta.get("settings", {})) == meta["expected"], meta.get("note", "")


def test_every_clip_carries_a_note():
    """A regression clip without an explanation is unmaintainable."""
    for path in clip_paths():
        _, meta = load_clip(path)
        assert meta.get("note"), f"{path.name} has no note"
