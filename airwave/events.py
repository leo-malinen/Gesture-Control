"""The one currency the pipeline trades in.

Producers (vision, audio) build events. The dispatcher consumes them. Nothing
else crosses the thread boundary, which is what makes a producer testable by
feeding it a recorded clip and asserting the event sequence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    """Every event kind the dispatcher knows how to route.

    The string values double as the ``trigger.type`` names in YAML, so
    ``EventKind("gesture")`` is the whole config-to-runtime mapping.
    """

    GESTURE = "gesture"
    MOTION = "motion"
    SOUND = "sound"
    SPEECH = "speech"
    POINTER = "pointer"
    SYSTEM = "system"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


TRIGGER_KINDS = frozenset(
    k.value for k in (EventKind.GESTURE, EventKind.MOTION, EventKind.SOUND, EventKind.SPEECH)
)
"""Kinds a user may bind an action to. POINTER and SYSTEM are internal.

MOTION is deliberately separate from GESTURE. A gesture is a *pose held still*
and is debounced by the stability window; a motion is the opposite - it only
exists while the hand is travelling, and holding still is how it ends. Giving
them one trigger type would mean one of the two had to lie about what it is."""


@dataclass(frozen=True, slots=True)
class Event:
    """A discrete, intentional input.

    ``kind`` + ``value`` form the lookup key into the binding table; together
    they are also the cooldown key, so two bindings on the same trigger share
    one cooldown rather than firing twice.

    ``captured_at`` is stamped at *frame capture* (not at emission) so the
    end-to-end latency budget in the PRD can be measured honestly.
    """

    kind: EventKind
    value: str
    captured_at: float = field(default_factory=time.monotonic)
    emitted_at: float = field(default_factory=time.monotonic)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def trigger_key(self) -> tuple[str, str]:
        return (self.kind.value, self.value)

    @property
    def pipeline_latency_ms(self) -> float:
        """Milliseconds burned between capture and this event existing."""
        return (self.emitted_at - self.captured_at) * 1000.0

    def describe(self) -> str:
        return f"{self.kind.value}:{self.value}"


@dataclass(frozen=True, slots=True)
class PointerSample:
    """A cursor target in normalized screen space (0..1), already smoothed."""

    x: float
    y: float
    captured_at: float = field(default_factory=time.monotonic)
    click: bool = False
    dwell_progress: float = 0.0
