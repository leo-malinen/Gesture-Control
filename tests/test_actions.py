"""Action executors and platform shims (PRD 4.6).

The executors are the only code that touches the outside world, so the tests
run them against a dry-run backend and assert on what *would* have happened.
"""

from __future__ import annotations

import pytest

from airwave.config import Action
from airwave.dispatch.actions import ActionExecutor, Controls
from airwave.dispatch.platform import KEY_ALIASES, InputBackend, detect


@pytest.fixture
def backend():
    return InputBackend(dry_run=True)


@pytest.fixture
def executor(backend):
    return ActionExecutor(backend)


def test_key_action_presses_once(executor, backend):
    executor.execute(Action("key", keys=("space",)))
    assert backend.log == ["press space x1"]


def test_repeat_presses_the_key_n_times(executor, backend):
    executor.execute(Action("key", keys=("volumeup",), repeat=3))
    assert backend.log == ["press volumeup x3"]


def test_hotkey_presses_keys_together(executor, backend):
    executor.execute(Action("hotkey", keys=("alt", "tab")))
    assert backend.log == ["hotkey alt+tab"]


def test_scroll_and_click(executor, backend):
    executor.execute(Action("scroll", amount=-3))
    executor.execute(Action("mouse_click", button="right", clicks=2))
    assert backend.log == ["scroll -3", "click right x2"]


def test_mouse_move_uses_payload_coordinates(executor, backend):
    executor.execute(Action("mouse_move"), payload={"x": 0.25, "y": 0.75})
    assert backend.log == ["move 0.250,0.750"]


def test_mouse_move_without_coordinates_is_a_no_op(executor, backend):
    result = executor.execute(Action("mouse_move"))
    assert result.ok and backend.log == []


def test_noop_does_nothing(executor, backend):
    assert executor.execute(Action("noop")).ok
    assert backend.log == []


def test_shell_is_blocked_unless_allowed(backend):
    """Second gate: the validator refuses these too, but a hot-reloaded config
    must not be able to slip one past."""
    executor = ActionExecutor(backend, allow_shell=False)
    result = executor.execute(Action("shell", command="echo hi"))
    assert result.ok
    assert "blocked" in result.detail


def test_a_failing_action_is_reported_not_raised(backend):
    class Exploding(InputBackend):
        def press(self, *a, **k):
            raise RuntimeError("no display")

    result = ActionExecutor(Exploding(dry_run=True)).execute(Action("key", keys=("a",)))
    assert result.ok is False
    assert "no display" in result.detail


def test_unknown_action_type_is_reported_not_raised(executor):
    result = executor.execute(Action("teleport"))
    assert result.ok and "no executor" in result.detail


def test_mode_action_flips_and_reads_state(backend):
    state = {"paused": False}
    controls = Controls(
        set_paused=lambda v: state.__setitem__("paused", v),
        get_paused=lambda: state["paused"],
    )
    executor = ActionExecutor(backend, controls=controls)
    executor.execute(Action("mode", target="pause", verb="toggle"))
    assert state["paused"] is True
    executor.execute(Action("mode", target="pause", verb="off"))
    assert state["paused"] is False


def test_repeat_is_bounded_so_the_dispatcher_thread_is_not_blocked(executor):
    """A long repeat interval must not stall every event queued behind it."""
    result = executor.execute(Action("key", keys=("volumeup",), repeat=5, interval_ms=10_000))
    assert result.duration_ms < 500


# ---------------------------------------------------------------- key naming


@pytest.mark.parametrize("alias,canonical", sorted(KEY_ALIASES.items())[:6])
def test_key_aliases_resolve(alias, canonical):
    assert InputBackend.canonical(alias) == canonical


def test_canonical_is_case_and_space_insensitive():
    assert InputBackend.canonical("  Volume_Up ") == "volumeup"


def test_media_key_names_pass_through():
    assert InputBackend.canonical("playpause") == "playpause"


# ----------------------------------------------------------------- platform


def test_detect_describes_this_machine():
    info = detect()
    assert info.system
    assert isinstance(info.warnings, list)
    assert isinstance(info.blocking, list)


def test_wayland_is_reported_as_blocking(monkeypatch):
    """FR-6.2 - a silent no-op is the worst possible behaviour here."""
    import platform as stdlib_platform

    monkeypatch.setattr(stdlib_platform, "system", lambda: "Linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    info = detect()
    assert info.is_wayland
    assert any("Wayland" in message for message in info.blocking)
    assert any("ydotool" in message for message in info.blocking)


def test_x11_without_display_is_reported(monkeypatch):
    import platform as stdlib_platform

    monkeypatch.setattr(stdlib_platform, "system", lambda: "Linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    assert any("DISPLAY" in message for message in detect().blocking)


def test_macos_missing_accessibility_is_reported(monkeypatch):
    """FR-6.1 - the single most common first-run failure on macOS."""
    import platform as stdlib_platform

    from airwave.dispatch import platform as airwave_platform

    monkeypatch.setattr(stdlib_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(airwave_platform, "macos_accessibility_trusted", lambda: False)
    info = airwave_platform.detect()
    assert any("Accessibility" in message for message in info.blocking)
    assert any("System Settings" in message for message in info.blocking)


def test_dry_run_backend_never_touches_pyautogui(backend):
    """The whole test suite depends on this being true."""
    backend.press("a")
    backend.hotkey("ctrl", "c")
    backend.move_to(0.5, 0.5)
    backend.click()
    backend.scroll(1)
    assert backend._gui is None
