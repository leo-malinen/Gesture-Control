"""Shared state between the pipeline and the web hub.

The hub is a *consumer*, exactly like the OpenCV overlay: the vision loop hands
it frames and the dispatcher hands it decisions, and it renders them. It cannot
press a key or reach back into the pipeline, which keeps the producer boundary
the rest of the project depends on intact.

Two details carry most of the design:

**JPEG encoding happens on the HTTP thread, not the vision thread.** Encoding a
640x480 frame costs a few milliseconds; paying that inside the capture loop
would come straight out of the frame budget, and it would be paid even with no
browser open. The vision loop only stores a reference; the stream handler
encodes at its own rate, and the result is cached so ten open tabs cost one
encode.

**Log timestamps are stamped here, in UTC wall-clock.** ``DispatchRecord.at``
is ``time.monotonic()`` - correct for measuring intervals, meaningless as a
time of day. The conversion happens at observation, microseconds after the
decision, which is accurate enough to read as the moment it happened.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

MAX_EVENTS = 500
"""Ring size for the log. The browser keeps its own window; this is the backlog
a page reload or a second tab gets to catch up on."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HubState:
    """Thread-safe handoff point. Written by the pipeline, read by HTTP threads."""

    def __init__(self, *, max_events: int = MAX_EVENTS) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_seq = 0
        self._jpeg: bytes | None = None
        self._jpeg_seq = -1
        self._jpeg_quality = 0
        self._status: dict[str, Any] = {
            "camera": "starting",
            "camera_detail": "",
            "running": True,
        }
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._event_seq = 0
        self._new_event = threading.Condition(self._lock)
        self.started_at = utc_now()

    # ------------------------------------------------------------------ video

    def publish_frame(self, frame: np.ndarray) -> None:
        """Store the latest annotated frame. Called from the vision thread.

        The caller must not mutate the array afterwards - the vision loop hands
        over a frame it has already finished drawing on.
        """
        with self._lock:
            self._frame = frame
            self._frame_seq += 1

    def latest_jpeg(self, quality: int = 72) -> tuple[bytes | None, int]:
        """Encode-on-demand with a one-slot cache shared by all stream clients."""
        with self._lock:
            frame = self._frame
            seq = self._frame_seq
            if frame is None:
                return None, seq
            if self._jpeg is not None and self._jpeg_seq == seq and self._jpeg_quality == quality:
                return self._jpeg, seq
        import cv2  # noqa: PLC0415 - kept out of module import for headless use

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None, seq
        data = buffer.tobytes()
        with self._lock:
            self._jpeg, self._jpeg_seq, self._jpeg_quality = data, seq, quality
        return data, seq

    @property
    def frame_seq(self) -> int:
        with self._lock:
            return self._frame_seq

    @property
    def has_video(self) -> bool:
        with self._lock:
            return self._frame is not None

    # ----------------------------------------------------------------- status

    def publish_status(self, **fields: Any) -> None:
        with self._lock:
            self._status.update(fields)

    def status(self) -> dict[str, Any]:
        with self._lock:
            out = dict(self._status)
        out["server_time"] = utc_now().isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return out

    # ------------------------------------------------------------------ log

    def add_record(self, record: Any) -> dict[str, Any] | None:
        """Turn a :class:`DispatchRecord` into a log row.

        Pointer *moves* are dropped: they arrive thirty times a second and would
        bury every real decision within a second. Clicks are kept - those are
        discrete actions the user took.
        """
        event = getattr(record, "event", None)
        kind = getattr(getattr(event, "kind", None), "value", "unknown")
        value = getattr(event, "value", "")
        if kind == "pointer" and value == "move":
            return None

        now = utc_now()
        with self._lock:
            self._event_seq += 1
            row = {
                "seq": self._event_seq,
                "ts": now.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "time": now.strftime("%H:%M:%S"),
                "millis": f"{now.microsecond // 1000:03d}",
                "kind": kind,
                "value": str(value),
                "outcome": getattr(record, "outcome", "unknown"),
                "binding": getattr(record, "binding", None),
                "detail": getattr(record, "detail", "") or "",
                "latency_ms": round(float(getattr(record, "latency_ms", 0.0) or 0.0), 1),
                "confidence": _confidence(event),
            }
            self._events.append(row)
            self._new_event.notify_all()
        return row

    def events_since(self, cursor: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            rows = [row for row in self._events if row["seq"] > cursor]
            return rows, self._event_seq

    def wait_for_event(self, cursor: int, timeout: float) -> tuple[list[dict[str, Any]], int]:
        """Block until there is something newer than ``cursor``, or time out.

        The stream handler uses this instead of polling so an idle hub costs
        nothing - which matters, because an idle hub is the normal state.
        """
        with self._lock:
            if self._event_seq <= cursor:
                self._new_event.wait(timeout)
            rows = [row for row in self._events if row["seq"] > cursor]
            return rows, self._event_seq

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)[-limit:]

    def wake_all(self) -> None:
        """Release every waiting stream handler so shutdown is not delayed."""
        with self._lock:
            self._status["running"] = False
            self._new_event.notify_all()


def _confidence(event: Any) -> float | None:
    payload = getattr(event, "payload", None)
    if isinstance(payload, dict) and "confidence" in payload:
        try:
            return round(float(payload["confidence"]), 2)
        except (TypeError, ValueError):
            return None
    return None
