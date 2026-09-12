"""The interface every gesture classifier implements.

``rules.RuleClassifier`` and ``model.ModelClassifier`` are interchangeable
because both speak this protocol - swapping them is a one-line config change
(``settings.classifier``), which is exactly the seam the PRD asks for in M5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..gestures import NONE_LABEL


@dataclass(frozen=True, slots=True)
class Classification:
    """One frame's verdict.

    ``confidence`` is not a probability from a calibrated model; it is a
    margin - how far this frame was from being labelled something else. The
    overlay shows it so a user can see *why* a gesture is not registering
    (usually: it is hovering at the boundary, not being missed entirely).
    """

    label: str = NONE_LABEL
    confidence: float = 0.0
    fingers: tuple[bool, bool, bool, bool, bool] = (False, False, False, False, False)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def finger_count(self) -> int:
        return sum(self.fingers)

    @property
    def is_none(self) -> bool:
        return self.label == NONE_LABEL


NO_HAND = Classification()


@runtime_checkable
class Classifier(Protocol):
    """Stateless-per-call contract, though implementations may keep hysteresis."""

    name: str

    def classify(self, landmarks: Any, *, handedness: str | None = ..., aspect: float = ...) -> Classification:
        ...

    def reset(self) -> None:
        """Drop any per-hand state. Called when the hand leaves the frame."""
