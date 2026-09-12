"""Recording mode and dataset format (FR-1.5, M5).

The dataset is one ``.npz`` per gesture holding raw landmarks, not feature
vectors. That ordering matters: features are a function of code, and code
changes. Storing raw landmarks means an improvement to ``normalize.py`` is a
retrain away instead of a re-record - and re-recording is the expensive part,
because it needs the user's hands.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..vision.normalize import feature_vector

log = logging.getLogger("airwave.training")

DEFAULT_DATA_DIR = Path("data") / "gestures"


@dataclass(slots=True)
class GestureSamples:
    label: str
    landmarks: np.ndarray  # (n, 21, 3)
    handedness: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.landmarks.shape[0])


class GestureRecorder:
    """Accumulates labelled landmark samples and appends them to disk."""

    def __init__(self, label: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
        self.label = label
        self.data_dir = Path(data_dir)
        self.samples: list[np.ndarray] = []
        self.handedness: list[str] = []

    @property
    def path(self) -> Path:
        return self.data_dir / f"{self.label}.npz"

    def add(self, landmarks: np.ndarray, handedness: str | None = None) -> int:
        self.samples.append(np.asarray(landmarks, dtype=np.float32).reshape(21, 3))
        self.handedness.append(handedness or "Unknown")
        return len(self.samples)

    def save(self, *, append: bool = True) -> Path:
        """Append to any existing samples for this label.

        Recording is done in short bursts - twenty samples, rest the hand,
        twenty more - and each burst should add to the set rather than replace
        it, or the third session silently deletes the first two.
        """
        if not self.samples:
            raise ValueError(f"no samples recorded for {self.label!r}")
        landmarks = np.stack(self.samples)
        handedness = list(self.handedness)
        if append and self.path.exists():
            existing = load_samples(self.path)
            landmarks = np.concatenate([existing.landmarks, landmarks])
            handedness = existing.handedness + handedness
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            landmarks=landmarks,
            handedness=np.array(handedness),
            label=self.label,
            recorded_at=time.time(),
        )
        log.info("saved %d samples (%d total) to %s", len(self.samples), len(landmarks), self.path)
        return self.path


def load_samples(path: str | Path) -> GestureSamples:
    data = np.load(path, allow_pickle=False)
    return GestureSamples(
        label=str(data["label"]),
        landmarks=data["landmarks"],
        handedness=[str(h) for h in data["handedness"]],
    )


def load_dataset(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Build the feature matrix from every recorded gesture on disk."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no recordings in {data_dir}. Record some first:\n"
            f"  python -m airwave record open_palm --samples 60"
        )
    features: list[np.ndarray] = []
    labels: list[str] = []
    counts: dict[str, int] = {}
    for file in files:
        samples = load_samples(file)
        for landmarks, hand in zip(samples.landmarks, samples.handedness):
            features.append(feature_vector(landmarks, handedness=hand))
            labels.append(samples.label)
        counts[samples.label] = len(samples)
    return np.stack(features), labels, counts


def dataset_summary(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict:
    data_dir = Path(data_dir)
    out: dict[str, int] = {}
    for file in sorted(data_dir.glob("*.npz")):
        try:
            out[file.stem] = len(load_samples(file))
        except Exception:  # noqa: BLE001 - a corrupt file should not hide the rest
            out[file.stem] = -1
    return out


def record_session(
    cfg: Config,
    label: str,
    *,
    samples: int = 60,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    countdown_s: float = 3.0,
    interval_ms: int = 80,
) -> Path:
    """Interactive capture loop with a live preview.

    Deliberately samples on an interval rather than every frame: consecutive
    frames of a held pose are near-duplicates, and a dataset of 300 duplicates
    trains a model that is confident and wrong on the 301st. Spacing the
    captures forces the small natural variation that makes the model general.
    """
    import cv2  # noqa: PLC0415

    from ..capture.camera import Camera  # noqa: PLC0415
    from ..ui.overlay import Overlay  # noqa: PLC0415
    from ..vision.landmarks import HandTracker  # noqa: PLC0415

    recorder = GestureRecorder(label, data_dir)
    overlay = Overlay(show_landmarks=True)
    settings = cfg.settings
    camera = Camera(settings.camera_index, width=settings.camera_width,
                    height=settings.camera_height, mirror=settings.mirror).open()
    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=settings.min_detection_confidence,
        min_tracking_confidence=settings.min_tracking_confidence,
    )

    recording = False
    started_at = 0.0
    last_capture = 0.0
    print(
        f"\nRecording '{label}'. Hold the pose, vary distance and angle slightly.\n"
        f"  space  start / pause      q  save and quit\n"
        f"  target: {samples} samples\n"
    )
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            hand = tracker.process(frame.image, timestamp_ms=int(frame.captured_at * 1000))
            now = time.monotonic()

            if hand.found and overlay.state.show_landmarks:
                overlay.draw_landmarks(frame.image, hand.landmarks)

            captured = len(recorder.samples)
            if recording and started_at and now >= started_at:
                if hand.found and (now - last_capture) * 1000.0 >= interval_ms:
                    recorder.add(hand.landmarks, hand.handedness)
                    last_capture = now
                    captured += 1

            h, w = frame.image.shape[:2]
            overlay._text(frame.image, f"RECORD {label}", (14, 34), scale=0.8,
                          color=(90, 90, 240) if recording else (236, 240, 241), thickness=2)
            overlay._text(frame.image, f"{captured}/{samples} samples", (14, 62), scale=0.55)
            overlay._bar(frame.image, 14, 74, 200, 8, captured / max(samples, 1), (120, 220, 120))
            if recording and started_at and now < started_at:
                overlay._text(frame.image, f"starting in {started_at - now:.1f}s", (14, 104), scale=0.6,
                              color=(250, 190, 80))
            if not hand.found:
                overlay._text(frame.image, "no hand detected", (14, 104), scale=0.5, color=(90, 90, 240))
            overlay._text(frame.image, "space start/pause   q save+quit", (14, h - 16), scale=0.45,
                          color=(150, 150, 150))

            overlay.show(frame.image)
            key = overlay.poll_key()
            if key in (27, ord("q")) or overlay.window_closed():
                break
            if key == ord(" "):
                recording = not recording
                started_at = now + countdown_s if recording else 0.0
                last_capture = 0.0
            if captured >= samples:
                break
    finally:
        camera.close()
        tracker.close()
        overlay.close()
        cv2.destroyAllWindows()

    if not recorder.samples:
        raise ValueError("no samples captured - nothing was saved")
    path = recorder.save()
    print(f"saved {len(recorder.samples)} samples to {path}")
    return path


def write_manifest(data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """A human-readable index of the dataset, useful in a portfolio repo."""
    data_dir = Path(data_dir)
    summary = dataset_summary(data_dir)
    path = data_dir / "manifest.json"
    path.write_text(json.dumps({"counts": summary, "written_at": time.time()}, indent=2), encoding="utf-8")
    return path
