"""Queue consumer: cooldown -> arming -> config lookup -> action (PRD 3.1).

The dispatcher is the only component that decides whether something happens.
Producers push events; this decides. That single-owner rule is what makes the
false-positive budget enforceable - there is exactly one place where an
unintended action can escape, and three gates in front of it:

1. **Binding lookup.** No binding, nothing happens. An unbound gesture is not
   an error, it is the normal case for most of the vocabulary.
2. **Arming** (FR-2.4). When enabled, an action fires only inside a few
   seconds of a deliberate arm. This is the highest-leverage defense in the
   product, because it shrinks the always-listening surface to almost nothing.
3. **Cooldown** (FR-2.3). Per-trigger, so a repeated gesture cannot double-fire
   while the user is still holding the pose that caused the first one.

Every decision is recorded, including the suppressions, because "why did
nothing happen?" is the question the overlay has to answer.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from ..audio.matching import best_match
from ..config import Binding, Config
from ..events import Event, EventKind
from .actions import ActionExecutor, ActionResult

log = logging.getLogger("airwave.dispatch")

OUTCOME_FIRED = "fired"
OUTCOME_COOLDOWN = "cooldown"
OUTCOME_UNARMED = "unarmed"
OUTCOME_UNBOUND = "unbound"
OUTCOME_PAUSED = "paused"
OUTCOME_ARMED = "armed"
OUTCOME_ERROR = "error"


@dataclass(slots=True)
class DispatchRecord:
    """One decision, kept for the overlay's event log and for the tests."""

    event: Event
    outcome: str
    binding: str | None = None
    detail: str = ""
    at: float = field(default_factory=time.monotonic)
    latency_ms: float = 0.0

    @property
    def fired(self) -> bool:
        return self.outcome == OUTCOME_FIRED

    def describe(self) -> str:
        head = f"{self.event.describe()} -> {self.outcome}"
        return f"{head} ({self.binding})" if self.binding else head


@dataclass(slots=True)
class DispatchStats:
    fired: int = 0
    cooldown: int = 0
    unarmed: int = 0
    unbound: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / self.fired if self.fired else 0.0


class Dispatcher:
    """Consumes events, applies the gates, and runs the surviving actions."""

    def __init__(
        self,
        cfg: Config,
        *,
        executor: ActionExecutor | None = None,
        clock=time.monotonic,
        log_size: int = 40,
    ) -> None:
        self.cfg = cfg
        self.executor = executor or ActionExecutor(allow_shell=cfg.settings.allow_shell)
        self._clock = clock
        self.queue: queue.Queue[Event] = queue.Queue(maxsize=256)
        self.records: deque[DispatchRecord] = deque(maxlen=log_size)
        self.stats = DispatchStats()
        self.armed_until: float = 0.0
        self.paused: bool = False
        self._cooldowns: dict[tuple[str, str], float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------- config/state

    def apply_config(self, cfg: Config) -> None:
        """Hot reload. Cooldowns survive: a reload should not re-open a window
        the user just consumed, or saving the file becomes a way to double-fire."""
        with self._lock:
            self.cfg = cfg
            self.executor.allow_shell = cfg.settings.allow_shell

    @property
    def arming_enabled(self) -> bool:
        return self.cfg.settings.arming.enabled

    def is_armed(self, now: float | None = None) -> bool:
        if not self.arming_enabled:
            return True
        return (self._clock() if now is None else now) < self.armed_until

    def arm(self, now: float | None = None) -> None:
        now = self._clock() if now is None else now
        self.armed_until = now + self.cfg.settings.arming.timeout_s

    def disarm(self) -> None:
        self.armed_until = 0.0

    def arm_remaining(self, now: float | None = None) -> float:
        now = self._clock() if now is None else now
        return max(0.0, self.armed_until - now)

    def cooldown_remaining(self, key: tuple[str, str], now: float | None = None) -> float:
        now = self._clock() if now is None else now
        return max(0.0, self._cooldowns.get(key, 0.0) - now)

    # ------------------------------------------------------------------ input

    def submit(self, event: Event) -> None:
        """Called from producer threads. Never blocks them.

        Pointer moves are the one stream fast enough to overrun the queue; when
        it is full they are dropped rather than queued, because a stale cursor
        position is worse than no cursor update.
        """
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            if event.kind is not EventKind.POINTER:
                log.warning("event queue full, dropped %s", event.describe())

    def submit_all(self, events) -> None:
        for event in events:
            self.submit(event)

    # ------------------------------------------------------------- resolution

    def _resolve(self, event: Event) -> Binding | None:
        """Find the binding for an event, fuzzily for speech (FR-3.5).

        ASR routinely returns "volume app" for "volume up"; an exact-match
        table would make voice control feel broken for reasons the user cannot
        see or fix.
        """
        if event.kind is EventKind.SPEECH:
            phrases = [b.trigger.value for b in self.cfg.bindings if b.enabled and b.trigger.type == "speech"]
            match = best_match(event.value, phrases, threshold=self.cfg.settings.speech.similarity)
            if match is None:
                return None
            phrase, score = match
            event.payload.setdefault("matched", phrase)
            event.payload.setdefault("similarity", round(score, 3))
            candidates = self.cfg.bindings_for("speech", phrase)
        else:
            candidates = self.cfg.bindings_for(event.kind.value, event.value)
        return candidates[0] if candidates else None

    def _requires_arm(self, binding: Binding) -> bool:
        if not self.arming_enabled:
            return False
        return True if binding.requires_arm is None else binding.requires_arm

    def _cooldown_ms(self, binding: Binding) -> int:
        return binding.cooldown_ms if binding.cooldown_ms is not None else self.cfg.settings.cooldown_ms

    # -------------------------------------------------------------- handling

    def handle(self, event: Event) -> DispatchRecord:
        """Process one event synchronously. The whole decision path lives here,
        which is what lets the test suite assert behaviour without threads."""
        now = self._clock()

        if event.kind is EventKind.SYSTEM:
            return self._handle_system(event, now)

        if self.paused and not (event.kind is EventKind.GESTURE and self._is_unpause(event)):
            return self._record(event, OUTCOME_PAUSED, detail="dispatcher paused")

        if event.kind is EventKind.POINTER:
            return self._handle_pointer(event, now)

        binding = self._resolve(event)
        if binding is None:
            self.stats.unbound += 1
            return self._record(event, OUTCOME_UNBOUND)

        if self._requires_arm(binding) and not self.is_armed(now):
            self.stats.unarmed += 1
            return self._record(event, OUTCOME_UNARMED, binding.name,
                                detail=f"arm with {self.cfg.settings.arming.gesture} first")

        key = event.trigger_key
        remaining = self.cooldown_remaining(key, now)
        if remaining > 0:
            self.stats.cooldown += 1
            return self._record(event, OUTCOME_COOLDOWN, binding.name, detail=f"{remaining * 1000:.0f}ms left")

        self._cooldowns[key] = now + self._cooldown_ms(binding) / 1000.0
        result = self.executor.execute(binding.action, payload=event.payload)
        return self._finish(event, binding, result, now)

    def _handle_system(self, event: Event, now: float) -> DispatchRecord:
        if event.value == "arm":
            self.arm(now)
            return self._record(event, OUTCOME_ARMED,
                                detail=f"armed for {self.cfg.settings.arming.timeout_s:g}s")
        if event.value == "disarm":
            self.disarm()
            return self._record(event, OUTCOME_ARMED, detail="disarmed")
        return self._record(event, OUTCOME_UNBOUND)

    def _handle_pointer(self, event: Event, now: float) -> DispatchRecord:
        """Cursor motion bypasses bindings and cooldown - it *is* the action.

        Clicks do not: they keep a short cooldown so one long pinch cannot
        stutter into a double-click.
        """
        from ..config import Action  # noqa: PLC0415 - avoids a cycle at import time

        if event.value == "move":
            result = self.executor.execute(Action("mouse_move"), payload=event.payload)
            return self._record(event, OUTCOME_FIRED if result.ok else OUTCOME_ERROR, "pointer", result.detail)
        if event.value == "click":
            key = ("pointer", "click")
            if self.cooldown_remaining(key, now) > 0:
                self.stats.cooldown += 1
                return self._record(event, OUTCOME_COOLDOWN, "pointer click")
            self._cooldowns[key] = now + self.cfg.settings.mouse.click_cooldown_ms / 1000.0
            result = self.executor.execute(Action("mouse_click", button="left", clicks=1))
            return self._finish(event, None, result, now, name="pointer click")
        return self._record(event, OUTCOME_UNBOUND)

    def _is_unpause(self, event: Event) -> bool:
        """While paused, only a binding that unpauses is allowed through."""
        for binding in self.cfg.bindings_for(event.kind.value, event.value):
            if binding.action.type == "mode" and binding.action.target == "pause":
                return True
        return False

    def _finish(self, event: Event, binding: Binding | None, result: ActionResult, now: float,
                name: str | None = None) -> DispatchRecord:
        label = binding.name if binding else name
        latency = (self._clock() - event.captured_at) * 1000.0
        if result.ok:
            self.stats.fired += 1
            self.stats.total_latency_ms += latency
            if binding is not None and self.cfg.settings.arming.disarm_on_fire:
                self.disarm()
            outcome = OUTCOME_FIRED
        else:
            self.stats.errors += 1
            outcome = OUTCOME_ERROR
        return self._record(event, outcome, label, result.detail, latency)

    def _record(self, event: Event, outcome: str, binding: str | None = None, detail: str = "",
                latency_ms: float = 0.0) -> DispatchRecord:
        record = DispatchRecord(event=event, outcome=outcome, binding=binding, detail=detail,
                                at=self._clock(), latency_ms=latency_ms)
        self.records.append(record)
        level = logging.INFO if outcome in (OUTCOME_FIRED, OUTCOME_ARMED) else logging.DEBUG
        log.log(level, "%s %s", record.describe(), f"[{detail}]" if detail else "")
        return record

    # ---------------------------------------------------------------- thread

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="airwave-dispatch", daemon=True)
        self._thread.start()

    def run(self) -> None:
        """Blocking loop with a timeout, so the stop event is observed promptly."""
        while not self._stop.is_set():
            try:
                event = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.handle(event)
            except Exception:  # pragma: no cover - the loop must outlive any handler
                log.exception("dispatch failed for %s", event.describe())
            finally:
                self.queue.task_done()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def drain(self) -> list[DispatchRecord]:
        """Process everything queued right now. Used by tests and offline runs."""
        out = []
        while True:
            try:
                event = self.queue.get_nowait()
            except queue.Empty:
                return out
            out.append(self.handle(event))
            self.queue.task_done()
