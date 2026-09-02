"""Vision producer: landmarks in, typed events out.

This is the "producers never act" boundary from the PRD. Nothing in this file
presses a key or moves a cursor; it returns events and lets the dispatcher
decide. That is what makes the whole vision path testable by feeding it a list
of landmark arrays - or a recorded MP4 - and asserting the event sequence.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..events import Event, EventKind, PointerSample
from ..gestures import NONE_LABEL
from .classifier import NO_HAND, Classification, Classifier
from .landmarks import HandFrame
from .normalize import INDEX_TIP
from .pointer import PointerController
from .rules import RuleClassifier
from .stabilizer import Stabilizer

log = logging.getLogger("airwave.vision.pipeline")


@dataclass(slots=True)
class FrameResult:
    """Everything one frame produced - events for the queue, state for the UI."""

    classification: Classification = NO_HAND
    events: list[Event] = field(default_factory=list)
    pointer: PointerSample | None = None
    hand: HandFrame | None = None
    stability: float = 0.0
    arm_progress: float = 0.0
    fps: float = 0.0
    frame_ms: float = 0.0
    captured_at: float = 0.0


def build_classifier(cfg: Config) -> Classifier:
    """Pick the classifier the config asks for, falling back loudly.

    A missing model file must not be a crash on a machine where the rules path
    would have worked fine; the user gets a warning and a running app.
    """
    if cfg.settings.classifier == "model":
        from .model import ModelClassifier  # noqa: PLC0415 - optional path

        try:
            return ModelClassifier.from_path(cfg.settings.model_path)
        except FileNotFoundError as exc:
            log.warning("%s\nfalling back to the rule classifier", exc)
    return RuleClassifier(pinch_threshold=cfg.settings.pinch_threshold)


class VisionPipeline:
    """Classify, stabilize, and emit - the whole camera-side state machine."""

    def __init__(self, cfg: Config, *, classifier: Classifier | None = None, clock=time.monotonic) -> None:
        self._clock = clock
        self.classifier = classifier or build_classifier(cfg)
        self.stabilizer = Stabilizer(cfg.settings.stable_frames, clock=clock)
        self.pointer: PointerController | None = None
        self._frame_times: deque[float] = deque(maxlen=60)
        self._last_frame_at: float | None = None
        self._arm_emitted_at = 0.0
        self.mouse_active = False
        self.cfg = cfg
        self.apply_config(cfg)

    # ------------------------------------------------------------ hot reload

    def apply_config(self, cfg: Config) -> None:
        """Adopt a new config without dropping the camera (FR-5.4).

        Only the classifier is rebuilt, and only when its identity changed -
        re-creating it every reload would throw away hysteresis state and make
        a config save look like a glitch.
        """
        old = self.cfg
        self.cfg = cfg
        self.stabilizer.stable_frames = cfg.settings.stable_frames
        if old is not cfg and (
            old.settings.classifier != cfg.settings.classifier
            or old.settings.model_path != cfg.settings.model_path
        ):
            self.classifier = build_classifier(cfg)
        elif isinstance(self.classifier, RuleClassifier):
            self.classifier.pinch_threshold = cfg.settings.pinch_threshold

        mouse = cfg.settings.mouse
        # Follow the config only when the *setting* changes. Following it
        # unconditionally would undo a runtime toggle (the 'm' key or a `mode`
        # binding) on the next unrelated config save; not following it at all
        # would mean enabling mouse control in the file did nothing until
        # restart. Comparing against the previous value is what distinguishes
        # "the user just asked for this" from "this value has not moved".
        if self.pointer is None or old.settings.mouse.enabled != mouse.enabled:
            self.mouse_active = mouse.enabled
        self.pointer = PointerController(
            active_region=mouse.active_region,
            smoothing=mouse.smoothing,
            min_cutoff=mouse.min_cutoff,
            beta=mouse.beta,
            dwell_click=mouse.dwell_click,
            dwell_ms=mouse.dwell_ms,
            dwell_radius=mouse.dwell_radius,
            invert_x=mouse.invert_x,
        )

    def set_mouse_active(self, active: bool) -> None:
        """Toggled by a ``mode`` binding at runtime."""
        self.mouse_active = active
        if self.pointer:
            self.pointer.reset()

    # ---------------------------------------------------------------- frames

    def process(self, hand: HandFrame, *, captured_at: float | None = None, aspect: float = 0.75) -> FrameResult:
        now = self._clock()
        captured_at = now if captured_at is None else captured_at
        if self._last_frame_at is not None:
            self._frame_times.append(now - self._last_frame_at)
        self._last_frame_at = now

        classification = self.classifier.classify(
            hand.landmarks, handedness=hand.handedness, aspect=aspect
        ) if hand.found else NO_HAND
        if not hand.found:
            self.classifier.reset()

        transition = self.stabilizer.push(classification.label, captured_at=captured_at)
        events: list[Event] = []
        pointer_sample = None

        pinch_is_click = (
            self.mouse_active
            and self.cfg.settings.mouse.pinch_click
            and classification.label == "pinch"
        )

        if transition is not None:
            if transition.label == "pinch" and pinch_is_click:
                # In mouse mode a pinch is a click, not a bindable gesture.
                events.append(self._pointer_event("click", captured_at, {"source": "pinch"}))
            else:
                events.append(
                    Event(
                        kind=EventKind.GESTURE,
                        value=transition.label,
                        captured_at=transition.captured_at,
                        emitted_at=self._clock(),
                        payload={
                            "confidence": round(classification.confidence, 3),
                            "previous": transition.previous,
                            "fingers": classification.finger_count,
                        },
                    )
                )

        arm_progress = self._arming(classification, events, captured_at, now)

        if self.mouse_active and hand.found and self.pointer is not None:
            tip = np.asarray(hand.landmarks)[INDEX_TIP]
            pointer_sample = self.pointer.update(float(tip[0]), float(tip[1]), now, captured_at=captured_at)
            events.append(
                self._pointer_event("move", captured_at, {"x": pointer_sample.x, "y": pointer_sample.y})
            )
            if pointer_sample.click:
                events.append(self._pointer_event("click", captured_at, {"source": "dwell"}))
        elif self.pointer is not None and not hand.found:
            self.pointer.reset()

        return FrameResult(
            classification=classification,
            events=events,
            pointer=pointer_sample,
            hand=hand,
            stability=self.stabilizer.progress,
            arm_progress=arm_progress,
            fps=self.fps,
            frame_ms=(self._clock() - now) * 1000.0,
            captured_at=captured_at,
        )

    def _pointer_event(self, value: str, captured_at: float, payload: dict) -> Event:
        return Event(
            kind=EventKind.POINTER,
            value=value,
            captured_at=captured_at,
            emitted_at=self._clock(),
            payload=payload,
        )

    def _arming(self, classification: Classification, events: list[Event], captured_at: float, now: float) -> float:
        """Emit an arm event once the arming pose has been held long enough.

        Progress is returned even when arming is disabled so the overlay can
        stay silent rather than showing a stale bar.
        """
        arming = self.cfg.settings.arming
        if not arming.enabled:
            return 0.0
        if self.stabilizer.stable_label != arming.gesture:
            return 0.0
        held = self.stabilizer.held_ms(now)
        progress = min(1.0, held / max(arming.hold_ms, 1))
        if progress >= 1.0 and self.stabilizer.stable_since > self._arm_emitted_at:
            self._arm_emitted_at = self.stabilizer.stable_since
            events.append(
                Event(
                    kind=EventKind.SYSTEM,
                    value="arm",
                    captured_at=captured_at,
                    emitted_at=self._clock(),
                    payload={"source": "gesture", "gesture": arming.gesture},
                )
            )
        return progress

    # ------------------------------------------------------------------ misc

    @property
    def fps(self) -> float:
        if not self._frame_times:
            return 0.0
        mean = sum(self._frame_times) / len(self._frame_times)
        return 1.0 / mean if mean > 0 else 0.0

    @property
    def frame_time_ms(self) -> list[float]:
        return [t * 1000.0 for t in self._frame_times]

    def p95_frame_ms(self) -> float:
        """Performance target from the PRD testing section (p95 < 50ms)."""
        if not self._frame_times:
            return 0.0
        ordered = sorted(t * 1000.0 for t in self._frame_times)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    def reset(self) -> None:
        self.stabilizer.reset()
        self.classifier.reset()
        if self.pointer:
            self.pointer.reset()

    def run_labels(self, labels) -> list[Event]:
        """Drive the stabilizer directly from labels. Used by the test suite."""
        events: list[Event] = []
        for label in labels:
            transition = self.stabilizer.push(label)
            if transition is not None and transition.label != NONE_LABEL:
                events.append(Event(kind=EventKind.GESTURE, value=transition.label))
        return events
