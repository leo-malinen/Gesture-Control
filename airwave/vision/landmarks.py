"""MediaPipe wrapper. Returns a normalized 21x3 array or ``None`` (FR-1.1).

MediaPipe shipped two incompatible Python APIs for hand tracking:

* the legacy ``mediapipe.solutions.hands`` module, which bundles its model
  inside the wheel and needs no setup, and
* the Tasks API (``mediapipe.tasks.python.vision.HandLandmarker``), which
  replaced it in newer releases and loads an external ``.task`` bundle.

Which one you get depends purely on the version pip resolved, so this module
detects and adapts rather than pinning users to one. The rest of Airwave sees
one interface either way: give it a BGR frame, get back landmarks or ``None``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("airwave.vision")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = Path("models") / "hand_landmarker.task"


@dataclass(frozen=True, slots=True)
class HandFrame:
    """One frame's tracking result.

    ``landmarks`` is ``None`` when no hand was found - explicitly modelled
    rather than returned as an empty array, so callers cannot accidentally
    classify a hand that is not there.
    """

    landmarks: np.ndarray | None
    handedness: str | None = None
    score: float = 0.0

    @property
    def found(self) -> bool:
        return self.landmarks is not None


class ModelMissingError(RuntimeError):
    """Raised with the exact command that fixes it."""


def resolve_model_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Find the ``.task`` bundle: argument > env > ./models > package dir."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("AIRWAVE_HAND_MODEL")
    if env:
        return Path(env).expanduser()
    candidates = [
        Path.cwd() / DEFAULT_MODEL_PATH,
        Path(__file__).resolve().parent.parent.parent / DEFAULT_MODEL_PATH,
        Path.home() / ".cache" / "airwave" / "hand_landmarker.task",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def download_model(dest: str | os.PathLike[str] | None = None, *, url: str = MODEL_URL) -> Path:
    """Fetch the hand landmarker bundle (~7.5 MB). Explicit, one-time, opt-in.

    Airwave makes no network calls while running (G5). This is a setup step the
    user runs knowingly, and only when their MediaPipe build lacks the bundled
    legacy model.
    """
    import urllib.request

    path = Path(dest) if dest else resolve_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    log.info("downloading hand landmarker model to %s", path)
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as fh:  # noqa: S310
        fh.write(response.read())
    tmp.replace(path)
    return path


def _legacy_available() -> bool:
    try:
        import mediapipe as mp  # noqa: PLC0415

        return hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
    except Exception:  # pragma: no cover - mediapipe missing entirely
        return False


class HandTracker:
    """Frame in, landmarks out. One hand (FR-1.1)."""

    def __init__(
        self,
        *,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0,
        model_path: str | os.PathLike[str] | None = None,
        backend: str = "auto",
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self._impl: Any = None
        self._frame_index = 0

        if backend == "auto":
            backend = "solutions" if _legacy_available() else "tasks"
        self.backend = backend
        if backend == "solutions":
            self._init_solutions()
        else:
            self._init_tasks(model_path)

    # ------------------------------------------------------------- backends

    def _init_solutions(self) -> None:
        import mediapipe as mp  # noqa: PLC0415

        self._mp = mp
        self._impl = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        log.debug("hand tracker: legacy solutions backend")

    def _init_tasks(self, model_path: str | os.PathLike[str] | None) -> None:
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks.python import BaseOptions  # noqa: PLC0415
        from mediapipe.tasks.python import vision  # noqa: PLC0415

        path = resolve_model_path(model_path)
        if not path.exists():
            raise ModelMissingError(
                f"MediaPipe {getattr(mp, '__version__', '?')} needs an external hand model and "
                f"none was found at {path}.\n"
                f"Fix it once with:  python -m airwave fetch-model\n"
                f"(downloads {MODEL_URL.rsplit('/', 1)[-1]}, ~7.5 MB; Airwave makes no other network calls)"
            )
        self._mp = mp
        self._vision = vision
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._impl = vision.HandLandmarker.create_from_options(options)
        log.debug("hand tracker: tasks backend, model=%s", path)

    # ---------------------------------------------------------------- input

    def process(self, frame_bgr: np.ndarray, *, timestamp_ms: int | None = None) -> HandFrame:
        """Run inference on one BGR frame (OpenCV's native order)."""
        import cv2  # noqa: PLC0415

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            return self._process_solutions(rgb)
        return self._process_tasks(rgb, timestamp_ms)

    def _process_solutions(self, rgb: np.ndarray) -> HandFrame:
        rgb.flags.writeable = False
        result = self._impl.process(rgb)
        if not result.multi_hand_landmarks:
            return HandFrame(None)
        landmarks = result.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32)
        handedness, score = None, 0.0
        if getattr(result, "multi_handedness", None):
            top = result.multi_handedness[0].classification[0]
            handedness, score = top.label, float(top.score)
        return HandFrame(pts, handedness, score)

    def _process_tasks(self, rgb: np.ndarray, timestamp_ms: int | None) -> HandFrame:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if timestamp_ms is None:
            # detect_for_video demands strictly increasing timestamps; a frame
            # counter is monotonic by construction, unlike a wall clock.
            self._frame_index += 1
            timestamp_ms = self._frame_index * 33
        result = self._impl.detect_for_video(mp_image, int(timestamp_ms))
        if not result.hand_landmarks:
            return HandFrame(None)
        landmarks = result.hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        handedness, score = None, 0.0
        if result.handedness:
            top = result.handedness[0][0]
            handedness, score = top.category_name, float(top.score)
        return HandFrame(pts, handedness, score)

    # --------------------------------------------------------------- teardown

    def close(self) -> None:
        if self._impl is not None:
            try:
                self._impl.close()
            except Exception:  # pragma: no cover - backend-specific teardown
                pass
            self._impl = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
