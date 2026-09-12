"""Every OS-specific fact in Airwave lives in this file (FR-6.1 - FR-6.3).

The dispatcher owns all side effects; this module owns all the reasons those
side effects differ per machine. Three problems it exists to solve:

* **macOS Accessibility.** Without the permission, every ``pyautogui`` call
  succeeds and does nothing. The app looks broken rather than unpermitted, and
  the PRD calls this out as the most common first-run failure - so we detect it
  and say exactly which settings pane to open.
* **Wayland.** ``pyautogui``'s X11 backend cannot synthesize input under
  Wayland, by design. Detecting it at startup beats a silent no-op.
* **Media keys.** ``playpause`` is a real key name on Windows and X11 and does
  not exist at all on macOS, where media control goes through a private
  NSEvent type. One alias table hides the difference.
"""

from __future__ import annotations

import logging
import os
import platform as _platform
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field

log = logging.getLogger("airwave.platform")

MEDIA_KEYS = frozenset({"playpause", "nexttrack", "prevtrack", "volumeup", "volumedown", "volumemute", "stop"})

#: Spellings people reasonably try, mapped onto pyautogui's canonical names.
KEY_ALIASES: dict[str, str] = {
    "play": "playpause",
    "pause": "playpause",
    "play_pause": "playpause",
    "media_play_pause": "playpause",
    "next": "nexttrack",
    "next_track": "nexttrack",
    "media_next": "nexttrack",
    "prev": "prevtrack",
    "previous": "prevtrack",
    "prev_track": "prevtrack",
    "media_prev": "prevtrack",
    "volume_up": "volumeup",
    "vol_up": "volumeup",
    "volume_down": "volumedown",
    "vol_down": "volumedown",
    "mute": "volumemute",
    "volume_mute": "volumemute",
    "cmd": "command",
    "win": "win",
    "meta": "winleft" if sys.platform == "win32" else "command",
    "return": "enter",
    "esc": "escape",
    "del": "delete",
    "pgup": "pageup",
    "pgdn": "pagedown",
}

#: macOS NX_KEYTYPE codes for the private media-key NSEvent.
_MAC_MEDIA_CODES = {"playpause": 16, "nexttrack": 17, "prevtrack": 18, "volumeup": 0, "volumedown": 1, "volumemute": 7}


@dataclass(slots=True)
class PlatformInfo:
    system: str
    session_type: str
    is_wayland: bool = False
    warnings: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blocking


def detect() -> PlatformInfo:
    """Describe the machine and everything likely to stop input synthesis."""
    system = _platform.system().lower()
    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    info = PlatformInfo(system=system, session_type=session or system)

    if system == "linux":
        wayland = session == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))
        info.is_wayland = wayland
        if wayland:
            has_ydotool = shutil.which("ydotool") is not None
            message = (
                "Wayland detected. pyautogui cannot synthesize input under Wayland, so no "
                "action will reach your desktop.\n"
                "  Workarounds: log into an X11/Xorg session, or install ydotool "
                "(https://github.com/ReimuNotMoe/ydotool) and use shell actions that call it."
            )
            if has_ydotool:
                message += "\n  ydotool is installed - shell actions like 'ydotool key 57' will work."
            info.blocking.append(message)
        elif not os.environ.get("DISPLAY"):
            info.blocking.append("No DISPLAY set - X11 input synthesis will fail. Are you in a TTY or over SSH?")

    elif system == "darwin":
        trusted = macos_accessibility_trusted()
        if trusted is False:
            info.blocking.append(
                "macOS Accessibility permission is not granted. Every keystroke Airwave sends "
                "will be silently discarded.\n"
                "  Fix: System Settings > Privacy & Security > Accessibility > enable your "
                "terminal (or Python).\n"
                "  Open it with:  open 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'"
            )
        elif trusted is None:
            info.warnings.append(
                "Could not verify macOS Accessibility permission (pyobjc not installed). "
                "If nothing happens when a gesture fires, grant it in "
                "System Settings > Privacy & Security > Accessibility."
            )
        if not os.environ.get("AIRWAVE_SKIP_CAMERA_HINT"):
            info.warnings.append(
                "macOS also prompts for Camera and Microphone permission on first run - "
                "accept both or the capture threads will see empty devices."
            )

    return info


def macos_accessibility_trusted() -> bool | None:
    """True / False / None when it cannot be determined."""
    if _platform.system() != "Darwin":
        return True
    try:
        from ApplicationServices import AXIsProcessTrusted  # type: ignore  # noqa: PLC0415

        return bool(AXIsProcessTrusted())
    except Exception:
        return None


class InputBackend:
    """Thin, lazy wrapper over pyautogui plus the macOS media-key detour.

    pyautogui is imported on first use rather than at module import: it grabs a
    display connection and, on macOS, can trigger a permission prompt. Neither
    belongs in ``import airwave``.
    """

    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.system = _platform.system().lower()
        self._gui = None
        self.log: list[str] = []

    @property
    def gui(self):
        if self._gui is None:
            import pyautogui  # noqa: PLC0415

            # A gesture that flings the cursor to a corner must not raise
            # FailSafeException and kill the dispatcher thread mid-session.
            pyautogui.FAILSAFE = False
            pyautogui.PAUSE = 0.0
            self._gui = pyautogui
        return self._gui

    # ------------------------------------------------------------ key naming

    @staticmethod
    def canonical(key: str) -> str:
        key = key.strip().lower().replace(" ", "")
        return KEY_ALIASES.get(key, key)

    def unknown_keys(self, keys) -> list[str]:
        """Report key names pyautogui will not recognize, without raising."""
        try:
            valid = set(self.gui.KEYBOARD_KEYS)
        except Exception:
            return []
        return [k for k in (self.canonical(k) for k in keys) if k not in valid and k not in MEDIA_KEYS]

    # --------------------------------------------------------------- actions

    def press(self, key: str, *, presses: int = 1, interval: float = 0.0) -> None:
        key = self.canonical(key)
        if self.dry_run:
            self.log.append(f"press {key} x{presses}")
            return
        if self.system == "darwin" and key in _MAC_MEDIA_CODES:
            for _ in range(presses):
                self._mac_media(key)
            return
        self.gui.press(key, presses=presses, interval=interval)

    def hotkey(self, *keys: str) -> None:
        resolved = [self.canonical(k) for k in keys]
        if self.dry_run:
            self.log.append("hotkey " + "+".join(resolved))
            return
        self.gui.hotkey(*resolved)

    def scroll(self, amount: int) -> None:
        if self.dry_run:
            self.log.append(f"scroll {amount}")
            return
        self.gui.scroll(amount)

    def move_to(self, x: float, y: float) -> None:
        """``x``/``y`` are 0..1 screen fractions - resolution stays down here."""
        if self.dry_run:
            self.log.append(f"move {x:.3f},{y:.3f}")
            return
        width, height = self.gui.size()
        self.gui.moveTo(int(x * width), int(y * height), _pause=False)

    def type_text(self, text: str) -> None:
        """Type a string at the focus point.

        ``write`` rather than key-by-key: pyautogui handles the shift states and
        layout mapping, and dictation output is arbitrary text with punctuation
        that hand-rolled keying gets wrong.
        """
        if self.dry_run:
            self.log.append(f"type {text!r}")
            return
        self.gui.write(text, interval=0.0)

    def sleep_machine(self) -> str:
        """Suspend the computer. One bounded capability, not a shell escape.

        Runs detached: on Windows ``SetSuspendState`` does not return until the
        machine wakes, and blocking the dispatcher thread for the length of a
        night's sleep would mean every event queued in between fires at once on
        resume. The short delay gives the log and the hub time to show what
        happened before the screen goes dark.
        """
        if self.dry_run:
            self.log.append("system sleep")
            return "sleep (dry run)"

        def suspend() -> None:
            import time as _time  # noqa: PLC0415

            _time.sleep(0.4)
            try:
                if self.system == "windows":
                    import ctypes  # noqa: PLC0415

                    # (hibernate=0 -> sleep, force=1, wake_events_disabled=0).
                    # Note: on a machine with hibernation enabled Windows may
                    # still choose hibernate; `powercfg -h off` makes it sleep.
                    ctypes.windll.powrprof.SetSuspendState(0, 1, 0)
                elif self.system == "darwin":
                    subprocess.run(["pmset", "sleepnow"], check=False, capture_output=True)
                else:
                    subprocess.run(["systemctl", "suspend"], check=False, capture_output=True)
            except Exception:  # noqa: BLE001
                log.exception("could not suspend the machine")

        threading.Thread(target=suspend, name="airwave-suspend", daemon=True).start()
        return "suspending"

    def click(self, button: str = "left", clicks: int = 1) -> None:
        if self.dry_run:
            self.log.append(f"click {button} x{clicks}")
            return
        self.gui.click(button=button, clicks=clicks)

    def screen_size(self) -> tuple[int, int]:
        if self.dry_run:
            return (1920, 1080)
        return tuple(self.gui.size())  # type: ignore[return-value]

    # ------------------------------------------------------------ mac detour

    def _mac_media(self, key: str) -> None:
        """Media keys on macOS are an NSEvent subtype, not a keycode.

        Quartz is the correct path; the AppleScript fallback covers volume only,
        which is still better than a silent no-op for the most common bindings.
        """
        code = _MAC_MEDIA_CODES[key]
        try:
            import Quartz  # type: ignore  # noqa: PLC0415
            from AppKit import NSEvent  # type: ignore  # noqa: PLC0415

            for down in (True, False):
                data = (code << 16) | ((0xA if down else 0xB) << 8)
                event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                    14, (0, 0), 0xA00 if down else 0xB00, 0, 0, None, 8, data, -1
                )
                Quartz.CGEventPost(0, event.CGEvent())
            return
        except Exception:
            pass
        script = {
            "volumeup": "set volume output volume (output volume of (get volume settings) + 6)",
            "volumedown": "set volume output volume (output volume of (get volume settings) - 6)",
            "volumemute": "set volume with output muted",
        }.get(key)
        if script is None:
            log.warning(
                "media key %r needs pyobjc on macOS: pip install pyobjc-framework-Quartz pyobjc-framework-Cocoa",
                key,
            )
            return
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True)


def describe() -> str:
    """One-line environment summary for the overlay and ``airwave doctor``."""
    info = detect()
    bits = [f"{_platform.system()} {_platform.release()}", f"python {_platform.python_version()}"]
    if info.is_wayland:
        bits.append("wayland")
    elif info.session_type:
        bits.append(info.session_type)
    return " | ".join(bits)
