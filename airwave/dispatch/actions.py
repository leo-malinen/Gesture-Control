"""Action executors - the only code in Airwave that touches the outside world.

Every executor is small and total: it either performs the effect or records
why it could not. Nothing here raises into the dispatcher thread, because a
dispatcher that dies on one bad binding takes the whole session with it.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

from ..config import Action
from .platform import InputBackend

log = logging.getLogger("airwave.actions")

MAX_REPEAT_MS = 400.0
"""Ceiling on a repeated key action. ``repeat: 3`` with a long interval would
otherwise block the dispatcher thread and delay every queued event behind it."""


@dataclass(slots=True)
class Controls:
    """Runtime switches a ``mode`` action can flip.

    Passed in rather than reached for, so the dispatcher can be tested with
    plain lambdas and no app object.
    """

    set_mouse: Callable[[bool], None] = lambda _: None
    set_armed: Callable[[bool], None] = lambda _: None
    set_paused: Callable[[bool], None] = lambda _: None
    get_mouse: Callable[[], bool] = lambda: False
    get_armed: Callable[[], bool] = lambda: False
    get_paused: Callable[[], bool] = lambda: False


@dataclass(slots=True)
class ActionResult:
    ok: bool
    detail: str
    duration_ms: float = 0.0


class ActionExecutor:
    """Maps an :class:`Action` onto the input backend."""

    def __init__(
        self,
        backend: InputBackend | None = None,
        *,
        allow_shell: bool = False,
        controls: Controls | None = None,
        dry_run: bool = False,
    ) -> None:
        self.backend = backend or InputBackend(dry_run=dry_run)
        self.allow_shell = allow_shell
        self.controls = controls or Controls()

    def execute(self, action: Action, *, payload: dict | None = None) -> ActionResult:
        started = time.perf_counter()
        try:
            detail = self._dispatch(action, payload or {})
            ok = True
        except Exception as exc:  # noqa: BLE001 - one bad binding must not kill the thread
            log.exception("action %s failed", action.type)
            detail, ok = f"{type(exc).__name__}: {exc}", False
        return ActionResult(ok=ok, detail=detail, duration_ms=(time.perf_counter() - started) * 1000.0)

    # ------------------------------------------------------------- executors

    def _dispatch(self, action: Action, payload: dict) -> str:
        handler = getattr(self, f"_do_{action.type}", None)
        if handler is None:
            return f"no executor for action type {action.type!r}"
        return handler(action, payload)

    def _do_noop(self, action: Action, payload: dict) -> str:
        return "noop"

    def _do_key(self, action: Action, payload: dict) -> str:
        """Press keys in sequence. ``repeat`` covers 'volume up by three steps'."""
        interval = min(action.interval_ms / 1000.0, MAX_REPEAT_MS / 1000.0 / max(action.repeat, 1))
        for key in action.keys:
            self.backend.press(key, presses=action.repeat, interval=interval)
        return f"pressed {' '.join(action.keys)} x{action.repeat}"

    def _do_hotkey(self, action: Action, payload: dict) -> str:
        self.backend.hotkey(*action.keys)
        return "hotkey " + "+".join(action.keys)

    def _do_scroll(self, action: Action, payload: dict) -> str:
        for i in range(action.repeat):
            self.backend.scroll(action.amount)
            if i + 1 < action.repeat and action.interval_ms:
                time.sleep(min(action.interval_ms / 1000.0, 0.05))
        return f"scrolled {action.amount * action.repeat:+d}"

    def _do_mouse_move(self, action: Action, payload: dict) -> str:
        """Coordinates come from the pointer pipeline, or are fixed in config."""
        x = payload.get("x", action.x)
        y = payload.get("y", action.y)
        if x is None or y is None:
            return "mouse_move with no coordinates"
        self.backend.move_to(float(x), float(y))
        return f"moved to {float(x):.3f},{float(y):.3f}"

    def _do_mouse_click(self, action: Action, payload: dict) -> str:
        self.backend.click(button=action.button, clicks=action.clicks)
        return f"{action.button} click x{action.clicks}"

    def _do_mode(self, action: Action, payload: dict) -> str:
        """Runtime toggles: mouse control, arming, and a global pause."""
        getter = {"mouse": self.controls.get_mouse, "arming": self.controls.get_armed,
                  "pause": self.controls.get_paused}[action.target or "mouse"]
        setter = {"mouse": self.controls.set_mouse, "arming": self.controls.set_armed,
                  "pause": self.controls.set_paused}[action.target or "mouse"]
        value = {"on": True, "off": False}.get(action.verb, not getter())
        setter(value)
        return f"{action.target} = {value}"

    def _do_shell(self, action: Action, payload: dict) -> str:
        """Run a command without waiting for it.

        Gated twice - the config validator refuses shell bindings unless
        ``allow_shell`` is set, and this checks again at execution, because a
        hot reload could in principle deliver a config the validator never saw.

        The threat model is explicit: a config file *is* code once this flag is
        on, so treat a shared airwave.yaml the way you would a shell script.
        """
        if not self.allow_shell:
            return "shell action blocked (settings.allow_shell is false)"
        if not action.command:
            return "shell action with no command"
        if sys.platform == "win32":
            subprocess.Popen(action.command, shell=True)  # noqa: S602
        else:
            subprocess.Popen(shlex.split(action.command))  # noqa: S603
        return f"ran {action.command!r}"
