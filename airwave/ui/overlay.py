"""Debug overlay (FR-2.5): the app's answer to "why did nothing happen?".

Every gate that can swallow a gesture is drawn: how close the current pose is
to being stable, whether the system is armed and for how much longer, and what
the dispatcher did with the last few events. A user who can see the stability
bar resetting knows their pose is being misread; one who sees it fill and then
"cooldown" knows to wait. Without that, both failures look identical - nothing
happens - and the tool feels broken rather than strict.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..dispatch.dispatcher import (
    OUTCOME_ARMED,
    OUTCOME_COOLDOWN,
    OUTCOME_ERROR,
    OUTCOME_FIRED,
    OUTCOME_UNARMED,
    Dispatcher,
)
from ..gestures import NONE_LABEL
from ..vision.pipeline import FrameResult

#: 21-point hand topology. Defined here rather than imported from MediaPipe,
#: whose newer releases moved (and in some builds dropped) the drawing utils.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

# BGR, because OpenCV.
INK = (236, 240, 241)
MUTED = (150, 150, 150)
ACCENT = (250, 190, 80)
GOOD = (120, 220, 120)
WARN = (80, 200, 250)
BAD = (90, 90, 240)
PANEL = (28, 28, 32)

OUTCOME_COLORS = {
    OUTCOME_FIRED: GOOD,
    OUTCOME_ARMED: ACCENT,
    OUTCOME_COOLDOWN: WARN,
    OUTCOME_UNARMED: WARN,
    OUTCOME_ERROR: BAD,
}

HELP_LINES = (
    "q / esc  quit",
    "a        arm now",
    "m        toggle mouse control",
    "p        pause / resume actions",
    "l        toggle landmarks",
    "h        toggle this help",
)


@dataclass(slots=True)
class OverlayState:
    show_help: bool = False
    show_landmarks: bool = True
    fps_history: deque = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.fps_history is None:
            self.fps_history = deque(maxlen=120)


class Overlay:
    """Draws onto the BGR frame in place and owns the preview window."""

    WINDOW = "Airwave"

    def __init__(self, *, show_landmarks: bool = True, window: str | None = None) -> None:
        self.state = OverlayState(show_landmarks=show_landmarks)
        self.window = window or self.WINDOW
        self._created = False

    # ------------------------------------------------------------------ chrome

    def _ensure_window(self) -> None:
        import cv2  # noqa: PLC0415

        if not self._created:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, 960, 720)
            self._created = True

    def show(self, image: np.ndarray) -> None:
        import cv2  # noqa: PLC0415

        self._ensure_window()
        cv2.imshow(self.window, image)

    def poll_key(self) -> int:
        import cv2  # noqa: PLC0415

        return cv2.waitKey(1) & 0xFF

    def close(self) -> None:
        import cv2  # noqa: PLC0415

        if self._created:
            cv2.destroyWindow(self.window)
            self._created = False

    def window_closed(self) -> bool:
        """True once the user clicks the window's X, so the app can exit."""
        import cv2  # noqa: PLC0415

        if not self._created:
            return False
        try:
            return cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    # ------------------------------------------------------------------ parts

    @staticmethod
    def _text(image, text, org, *, scale=0.5, color=INK, thickness=1) -> None:
        import cv2  # noqa: PLC0415

        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _bar(image, x, y, width, height, fraction, color) -> None:
        import cv2  # noqa: PLC0415

        cv2.rectangle(image, (x, y), (x + width, y + height), (60, 60, 66), -1)
        filled = int(width * max(0.0, min(1.0, fraction)))
        if filled:
            cv2.rectangle(image, (x, y), (x + filled, y + height), color, -1)

    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> None:
        import cv2  # noqa: PLC0415

        h, w = image.shape[:2]
        points = [(int(pt[0] * w), int(pt[1] * h)) for pt in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, points[a], points[b], (200, 200, 200), 2, cv2.LINE_AA)
        for i, point in enumerate(points):
            color = ACCENT if i in (4, 8, 12, 16, 20) else GOOD
            cv2.circle(image, point, 4, color, -1, cv2.LINE_AA)

    def draw_active_region(self, image: np.ndarray, region) -> None:
        """Show the sub-region that maps to the screen, so the user can see
        where the cursor stops responding before they think it froze."""
        import cv2  # noqa: PLC0415

        h, w = image.shape[:2]
        x0, y0, x1, y1 = region
        cv2.rectangle(image, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), ACCENT, 1)

    # ------------------------------------------------------------------ panel

    def render(self, frame: np.ndarray, result: FrameResult, dispatcher: Dispatcher, *, extra: dict | None = None) -> np.ndarray:
        import cv2  # noqa: PLC0415

        image = frame
        cfg = dispatcher.cfg
        self.state.fps_history.append(result.fps)

        if self.state.show_landmarks and result.hand is not None and result.hand.found:
            self.draw_landmarks(image, result.hand.landmarks)
        if cfg.settings.mouse.enabled:
            self.draw_active_region(image, cfg.settings.mouse.active_region)

        h, w = image.shape[:2]
        panel_h = 132
        panel = image[0:panel_h, 0:w]
        cv2.rectangle(image, (0, 0), (w, panel_h), PANEL, -1)
        cv2.addWeighted(panel, 0.35, image[0:panel_h, 0:w], 0.65, 0, image[0:panel_h, 0:w])

        label = result.classification.label
        label_color = MUTED if label == NONE_LABEL else (GOOD if result.stability >= 1.0 else INK)
        self._text(image, label.upper(), (14, 34), scale=0.9, color=label_color, thickness=2)
        confidence = result.classification.confidence
        self._text(image, f"conf {confidence:.2f}", (14, 56), scale=0.45, color=MUTED)

        # Stability: the single most useful number when a gesture will not fire.
        self._text(image, f"stable {int(result.stability * 100):3d}%", (150, 34), scale=0.5)
        self._bar(image, 150, 42, 130, 8, result.stability, GOOD if result.stability >= 1.0 else ACCENT)

        # Arming.
        if cfg.settings.arming.enabled:
            remaining = dispatcher.arm_remaining()
            armed = remaining > 0
            self._text(image, "ARMED" if armed else "not armed", (150, 74), scale=0.5,
                       color=GOOD if armed else BAD)
            fraction = remaining / max(cfg.settings.arming.timeout_s, 1e-6) if armed else result.arm_progress
            self._bar(image, 150, 82, 130, 8, fraction, GOOD if armed else ACCENT)
            if not armed:
                self._text(image, f"hold {cfg.settings.arming.gesture} to arm", (150, 104), scale=0.4, color=MUTED)
        else:
            self._text(image, "arming off", (150, 74), scale=0.45, color=MUTED)

        # Cooldown on whatever is currently showing.
        cooldown = dispatcher.cooldown_remaining(("gesture", label)) if label != NONE_LABEL else 0.0
        if cooldown > 0:
            self._text(image, f"cooldown {cooldown * 1000:.0f}ms", (300, 34), scale=0.5, color=WARN)
            self._bar(image, 300, 42, 120, 8,
                      cooldown / max(cfg.settings.cooldown_ms / 1000.0, 1e-6), WARN)

        fps = sum(self.state.fps_history) / len(self.state.fps_history) if self.state.fps_history else 0.0
        fps_color = GOOD if fps >= 20 else (WARN if fps >= 12 else BAD)
        self._text(image, f"{fps:4.1f} fps", (w - 110, 26), scale=0.55, color=fps_color)
        self._text(image, f"p95 {extra.get('p95_ms', 0):.0f}ms" if extra else "", (w - 110, 46), scale=0.42, color=MUTED)

        flags = []
        if dispatcher.paused:
            flags.append("PAUSED")
        if extra and extra.get("mouse_active"):
            flags.append("MOUSE")
        if cfg.settings.speech.enabled:
            flags.append("VOICE")
        if flags:
            self._text(image, " ".join(flags), (w - 110, 68), scale=0.45, color=ACCENT)

        stats = dispatcher.stats
        self._text(image,
                   f"fired {stats.fired}  cooldown {stats.cooldown}  unarmed {stats.unarmed}  unbound {stats.unbound}",
                   (14, 104), scale=0.42, color=MUTED)
        if extra and extra.get("audio"):
            audio = extra["audio"]
            self._text(image, f"rms {audio['rms']:.3f} / thr {audio['threshold']:.3f}", (14, 122),
                       scale=0.42, color=MUTED)

        self._event_log(image, dispatcher, w, h)
        if self.state.show_help:
            self._help(image, w, h)
        else:
            self._text(image, "h for keys", (w - 110, h - 14), scale=0.4, color=MUTED)
        return image

    def _event_log(self, image, dispatcher: Dispatcher, w: int, h: int) -> None:
        now = time.monotonic()
        y = h - 16
        for record in list(dispatcher.records)[-6:][::-1]:
            age = now - record.at
            if age > 12:
                continue
            color = OUTCOME_COLORS.get(record.outcome, MUTED)
            line = f"{age:4.1f}s  {record.event.describe():<24} {record.outcome}"
            if record.binding:
                line += f"  [{record.binding}]"
            if record.outcome == OUTCOME_FIRED and record.latency_ms:
                line += f"  {record.latency_ms:.0f}ms"
            self._text(image, line, (14, y), scale=0.42, color=color)
            y -= 18

    def _help(self, image, w: int, h: int) -> None:
        import cv2  # noqa: PLC0415

        box_w, box_h = 260, 20 * len(HELP_LINES) + 20
        x0, y0 = w - box_w - 14, h - box_h - 14
        cv2.rectangle(image, (x0, y0), (x0 + box_w, y0 + box_h), PANEL, -1)
        for i, line in enumerate(HELP_LINES):
            self._text(image, line, (x0 + 12, y0 + 24 + i * 20), scale=0.45)
