"""Camera abstraction over ``cv2.VideoCapture``.

Beyond opening a device, this handles the two things that make webcam code
annoying in the field: cameras that report a resolution they do not deliver,
and cameras that vanish mid-session (a USB hub sleeps, another app steals the
device, a laptop lid closes). A dropped device reconnects with backoff instead
of taking down the process.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("airwave.capture.camera")


class CameraError(RuntimeError):
    """Raised with a message aimed at the person, not the stack trace."""


@dataclass(frozen=True, slots=True)
class Frame:
    image: np.ndarray
    captured_at: float
    index: int

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.image.shape[1]), int(self.image.shape[0])

    @property
    def aspect(self) -> float:
        """Height / width - what the pinch threshold needs to stay isotropic."""
        return float(self.image.shape[0]) / float(self.image.shape[1])


def _backend():
    """Pick the fastest-opening backend per OS.

    On Windows the default MSMF backend can take several seconds to open a
    webcam; DirectShow opens in a few hundred milliseconds. That difference is
    most of the PRD's "time to first success" budget.
    """
    import cv2  # noqa: PLC0415

    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


class Camera:
    """A webcam that reopens itself when the device misbehaves."""

    def __init__(
        self,
        index: int = 0,
        *,
        width: int = 640,
        height: int = 480,
        mirror: bool = True,
        max_reopen_attempts: int = 5,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.mirror = mirror
        self.max_reopen_attempts = max_reopen_attempts
        self._cap = None
        self._frames = 0
        self._failures = 0
        self.actual_size: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------- lifecycle

    def open(self) -> "Camera":
        import cv2  # noqa: PLC0415

        cap = cv2.VideoCapture(self.index, _backend())
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"could not open camera {self.index}.\n"
                f"  - is another app (Zoom, Teams, a browser tab) holding it?\n"
                f"  - list what is available with: python -m airwave devices\n"
                + ("  - on macOS, grant Camera permission to your terminal\n" if sys.platform == "darwin" else "")
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # A 1-frame buffer keeps latency honest; without it OpenCV hands back
        # frames that are several hundred milliseconds old under load.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        ok, frame = cap.read()
        if not ok or frame is None:
            self.close()
            raise CameraError(f"camera {self.index} opened but returned no frames")
        self.actual_size = (int(frame.shape[1]), int(frame.shape[0]))
        if self.actual_size != (self.width, self.height):
            log.info("camera negotiated %dx%d (asked for %dx%d)", *self.actual_size, self.width, self.height)
        return self

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ----------------------------------------------------------------- input

    def read(self) -> Frame | None:
        """Grab one frame, timestamped at capture. ``None`` means try again."""
        import cv2  # noqa: PLC0415

        if self._cap is None:
            raise CameraError("camera is not open")
        ok, image = self._cap.read()
        captured_at = time.monotonic()
        if not ok or image is None:
            self._failures += 1
            if self._failures <= 3:
                return None
            if not self._reopen():
                raise CameraError(
                    f"camera {self.index} stopped delivering frames and could not be reopened "
                    f"after {self.max_reopen_attempts} attempts"
                )
            return None
        self._failures = 0
        self._frames += 1
        if self.mirror:
            # Selfie view: users move their hand right and expect the on-screen
            # hand to move right. Un-mirrored preview is disorienting enough
            # that people misjudge every gesture position.
            image = cv2.flip(image, 1)
        return Frame(image=image, captured_at=captured_at, index=self._frames)

    def _reopen(self) -> bool:
        log.warning("camera %s dropped, reconnecting", self.index)
        self.close()
        for attempt in range(self.max_reopen_attempts):
            time.sleep(min(0.25 * (2**attempt), 2.0))
            try:
                self.open()
                log.info("camera %s recovered", self.index)
                self._failures = 0
                return True
            except CameraError:
                continue
        return False


def list_cameras(limit: int = 6) -> list[dict]:
    """Probe indices so ``airwave devices`` can tell the user what exists."""
    import cv2  # noqa: PLC0415

    found = []
    for index in range(limit):
        cap = cv2.VideoCapture(index, _backend())
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append({"index": index, "width": int(frame.shape[1]), "height": int(frame.shape[0])})
        cap.release()
    return found
