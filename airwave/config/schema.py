"""Typed config schema plus the validator that produces the error messages.

Everything is plain dataclasses with a ``_parse`` classmethod. No pydantic:
the dependency budget in the PRD is "one pip install", and hand-written
validation is what lets errors carry line numbers and spelling suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from ..events import TRIGGER_KINDS, EventKind
from ..gestures import BUILTIN_GESTURES, RESERVED_LABELS
from .errors import ConfigIssue, suggest

ACTION_TYPES = ("key", "hotkey", "mouse_move", "mouse_click", "scroll", "shell", "mode", "noop")
SOUND_VALUES = ("clap", "double_clap")
MODE_TARGETS = ("mouse", "arming", "pause")
MODE_VERBS = ("toggle", "on", "off")
CLASSIFIERS = ("rules", "model")
SPEECH_ENGINES = ("vosk", "faster-whisper", "none")


class _Ctx:
    """Collects issues while walking the parsed YAML.

    A parser that raises on the first mistake can only ever report one; this
    keeps walking with a safe default so the user gets the full list.
    """

    def __init__(self) -> None:
        self.issues: list[ConfigIssue] = []

    def add(self, path: str, message: str, line: int | None = None, hint: str | None = None) -> None:
        self.issues.append(ConfigIssue(path=path, message=message, line=line, hint=hint))


def _line_of(node: Any, key: str | None = None) -> int | None:
    """Pull the line number the loader stapled onto a mapping, if present."""
    if key is not None:
        lines = getattr(node, "key_lines", None)
        if lines and key in lines:
            return lines[key]
    return getattr(node, "line", None)


def _get(ctx: _Ctx, node: Any, path: str, key: str, default, kind, *, choices: Iterable | None = None):
    """Read one key with type coercion, range checks and a helpful message."""
    if not isinstance(node, dict) or key not in node:
        return default
    value = node[key]
    line = _line_of(node, key)
    where = f"{path}.{key}" if path else key
    if kind is bool:
        if not isinstance(value, bool):
            ctx.add(where, f"expected true or false, got {value!r}", line)
            return default
        return value
    if kind in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            ctx.add(where, f"expected a number, got {value!r}", line)
            return default
        return kind(value)
    if kind is str:
        if not isinstance(value, str):
            ctx.add(where, f"expected a string, got {value!r}", line)
            return default
        if choices is not None and value not in choices:
            ctx.add(where, f"unknown value {value!r}", line, suggest(value, choices))
            return default
        return value
    if kind is list:
        if not isinstance(value, list):
            ctx.add(where, f"expected a list, got {value!r}", line)
            return default
        return value
    return value


def _bounded(ctx: _Ctx, value, path: str, line: int | None, *, minimum=0.0, maximum=None, name="value"):
    if value is None:
        return value
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        ctx.add(path, f"{name} must be {bound}, got {value}", line)
    return value


def _unknown_keys(ctx: _Ctx, node: Any, path: str, allowed: Iterable[str]) -> None:
    """Silent typos in setting names are the worst failure mode: the app runs,
    the setting does nothing, and the user concludes the feature is broken."""
    if not isinstance(node, dict):
        return
    allowed_set = set(allowed)
    for key in node:
        if key not in allowed_set:
            ctx.add(
                f"{path}.{key}" if path else str(key),
                f"unknown setting {key!r}",
                _line_of(node, key),
                suggest(key, allowed_set, "known keys"),
            )


# --------------------------------------------------------------------------- settings


@dataclass(slots=True)
class ArmingSettings:
    """Arming collapses the always-listening surface to a few seconds (FR-2.4)."""

    enabled: bool = False
    gesture: str = "thumbs_up"
    hold_ms: int = 1000
    timeout_s: float = 5.0
    wake_word: bool = True
    disarm_on_fire: bool = False

    _KEYS = ("enabled", "gesture", "hold_ms", "timeout_s", "wake_word", "disarm_on_fire")

    @classmethod
    def _parse(cls, ctx: _Ctx, node: Any, path: str, known_gestures) -> "ArmingSettings":
        _unknown_keys(ctx, node, path, cls._KEYS)
        out = cls(
            enabled=_get(ctx, node, path, "enabled", False, bool),
            gesture=_get(ctx, node, path, "gesture", "thumbs_up", str),
            hold_ms=int(_get(ctx, node, path, "hold_ms", 1000, int)),
            timeout_s=_get(ctx, node, path, "timeout_s", 5.0, float),
            wake_word=_get(ctx, node, path, "wake_word", True, bool),
            disarm_on_fire=_get(ctx, node, path, "disarm_on_fire", False, bool),
        )
        if out.gesture not in known_gestures:
            ctx.add(
                f"{path}.gesture",
                f"unknown gesture {out.gesture!r}",
                _line_of(node, "gesture"),
                suggest(out.gesture, known_gestures, "known gestures"),
            )
        _bounded(ctx, out.timeout_s, f"{path}.timeout_s", _line_of(node, "timeout_s"), minimum=0.5, maximum=120)
        _bounded(ctx, out.hold_ms, f"{path}.hold_ms", _line_of(node, "hold_ms"), minimum=0, maximum=10000)
        return out


@dataclass(slots=True)
class MouseSettings:
    """FR-4.x. ``active_region`` is the fraction of the frame that maps to the
    whole screen, so reaching a corner is a wrist move, not a shoulder move."""

    enabled: bool = False
    smoothing: float = 0.6
    active_region: tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75)
    min_cutoff: float = 1.0
    beta: float = 3.0
    pinch_click: bool = True
    dwell_click: bool = False
    dwell_ms: int = 900
    dwell_radius: float = 0.02
    click_cooldown_ms: int = 400
    invert_x: bool = False

    _KEYS = (
        "enabled", "smoothing", "active_region", "min_cutoff", "beta", "pinch_click",
        "dwell_click", "dwell_ms", "dwell_radius", "click_cooldown_ms", "invert_x",
    )

    @classmethod
    def _parse(cls, ctx: _Ctx, node: Any, path: str) -> "MouseSettings":
        _unknown_keys(ctx, node, path, cls._KEYS)
        region = _get(ctx, node, path, "active_region", [0.25, 0.25, 0.75, 0.75], list)
        line = _line_of(node, "active_region")
        parsed: tuple[float, float, float, float] = (0.25, 0.25, 0.75, 0.75)
        if isinstance(region, list):
            numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in region)
            if len(region) != 4 or not numeric:
                ctx.add(f"{path}.active_region", "expected [x0, y0, x1, y1] as four numbers in 0..1", line)
            elif not (0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1):
                ctx.add(
                    f"{path}.active_region",
                    f"region {list(region)} is not a valid box; needs 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1",
                    line,
                )
            else:
                parsed = (float(region[0]), float(region[1]), float(region[2]), float(region[3]))
        out = cls(
            enabled=_get(ctx, node, path, "enabled", False, bool),
            smoothing=_get(ctx, node, path, "smoothing", 0.6, float),
            active_region=parsed,
            min_cutoff=_get(ctx, node, path, "min_cutoff", 1.0, float),
            beta=_get(ctx, node, path, "beta", 3.0, float),
            pinch_click=_get(ctx, node, path, "pinch_click", True, bool),
            dwell_click=_get(ctx, node, path, "dwell_click", False, bool),
            dwell_ms=int(_get(ctx, node, path, "dwell_ms", 900, int)),
            dwell_radius=_get(ctx, node, path, "dwell_radius", 0.02, float),
            click_cooldown_ms=int(_get(ctx, node, path, "click_cooldown_ms", 400, int)),
            invert_x=_get(ctx, node, path, "invert_x", False, bool),
        )
        _bounded(ctx, out.smoothing, f"{path}.smoothing", _line_of(node, "smoothing"), minimum=0.0, maximum=0.99)
        _bounded(ctx, out.dwell_ms, f"{path}.dwell_ms", _line_of(node, "dwell_ms"), minimum=100, maximum=10000)
        _bounded(ctx, out.min_cutoff, f"{path}.min_cutoff", _line_of(node, "min_cutoff"), minimum=0.01, maximum=30.0)
        return out


@dataclass(slots=True)
class AudioSettings:
    """FR-3.x. Thresholds are per-room, which is why calibrate mode exists."""

    enabled: bool = True
    device: int | str | None = None
    samplerate: int = 16000
    blocksize: int = 512
    channels: int = 1
    baseline_alpha: float = 0.995
    multiplier: float = 8.0
    floor: float = 0.06
    refractory_ms: int = 180
    double_clap_min_ms: int = 150
    double_clap_max_ms: int = 600

    _KEYS = (
        "enabled", "device", "samplerate", "blocksize", "channels", "baseline_alpha",
        "multiplier", "floor", "refractory_ms", "double_clap_min_ms", "double_clap_max_ms",
    )

    @classmethod
    def _parse(cls, ctx: _Ctx, node: Any, path: str) -> "AudioSettings":
        _unknown_keys(ctx, node, path, cls._KEYS)
        device = node.get("device") if isinstance(node, dict) else None
        if device is not None and not isinstance(device, (int, str)):
            ctx.add(f"{path}.device", f"expected a device index or name, got {device!r}", _line_of(node, "device"))
            device = None
        out = cls(
            enabled=_get(ctx, node, path, "enabled", True, bool),
            device=device,
            samplerate=int(_get(ctx, node, path, "samplerate", 16000, int)),
            blocksize=int(_get(ctx, node, path, "blocksize", 512, int)),
            channels=int(_get(ctx, node, path, "channels", 1, int)),
            baseline_alpha=_get(ctx, node, path, "baseline_alpha", 0.995, float),
            multiplier=_get(ctx, node, path, "multiplier", 8.0, float),
            floor=_get(ctx, node, path, "floor", 0.06, float),
            refractory_ms=int(_get(ctx, node, path, "refractory_ms", 180, int)),
            double_clap_min_ms=int(_get(ctx, node, path, "double_clap_min_ms", 150, int)),
            double_clap_max_ms=int(_get(ctx, node, path, "double_clap_max_ms", 600, int)),
        )
        _bounded(ctx, out.baseline_alpha, f"{path}.baseline_alpha", _line_of(node, "baseline_alpha"),
                 minimum=0.5, maximum=0.99999)
        _bounded(ctx, out.multiplier, f"{path}.multiplier", _line_of(node, "multiplier"), minimum=1.0, maximum=100.0)
        _bounded(ctx, out.floor, f"{path}.floor", _line_of(node, "floor"), minimum=0.0, maximum=1.0)
        if out.double_clap_min_ms >= out.double_clap_max_ms:
            ctx.add(
                f"{path}.double_clap_max_ms",
                f"double_clap_max_ms ({out.double_clap_max_ms}) must be greater than "
                f"double_clap_min_ms ({out.double_clap_min_ms})",
                _line_of(node, "double_clap_max_ms"),
            )
        return out


@dataclass(slots=True)
class SpeechSettings:
    """FR-3.4/3.5. ASR is gated behind a wake word and never runs continuously."""

    enabled: bool = False
    engine: str = "vosk"
    model_path: str | None = None
    wake_word: str = "hey airwave"
    wake_engine: str = "keyword"
    listen_window_s: float = 4.0
    similarity: float = 0.72

    _KEYS = ("enabled", "engine", "model_path", "wake_word", "wake_engine", "listen_window_s", "similarity")

    @classmethod
    def _parse(cls, ctx: _Ctx, node: Any, path: str) -> "SpeechSettings":
        _unknown_keys(ctx, node, path, cls._KEYS)
        model_path = node.get("model_path") if isinstance(node, dict) else None
        if model_path is not None and not isinstance(model_path, str):
            ctx.add(f"{path}.model_path", f"expected a path, got {model_path!r}", _line_of(node, "model_path"))
            model_path = None
        out = cls(
            enabled=_get(ctx, node, path, "enabled", False, bool),
            engine=_get(ctx, node, path, "engine", "vosk", str, choices=SPEECH_ENGINES),
            model_path=model_path,
            wake_word=_get(ctx, node, path, "wake_word", "hey airwave", str),
            wake_engine=_get(ctx, node, path, "wake_engine", "keyword", str, choices=("keyword", "openwakeword")),
            listen_window_s=_get(ctx, node, path, "listen_window_s", 4.0, float),
            similarity=_get(ctx, node, path, "similarity", 0.72, float),
        )
        _bounded(ctx, out.similarity, f"{path}.similarity", _line_of(node, "similarity"), minimum=0.3, maximum=1.0)
        _bounded(ctx, out.listen_window_s, f"{path}.listen_window_s", _line_of(node, "listen_window_s"),
                 minimum=0.5, maximum=30.0)
        return out


@dataclass(slots=True)
class Settings:
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    mirror: bool = True
    stable_frames: int = 5
    cooldown_ms: int = 800
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    classifier: str = "rules"
    model_path: str = "data/model.npz"
    pinch_threshold: float = 0.04
    allow_shell: bool = False
    overlay: bool = True
    show_landmarks: bool = True
    log_level: str = "info"
    hot_reload: bool = True
    arming: ArmingSettings = field(default_factory=ArmingSettings)
    mouse: MouseSettings = field(default_factory=MouseSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)

    _SCALARS = (
        "camera_index", "camera_width", "camera_height", "mirror", "stable_frames", "cooldown_ms",
        "min_detection_confidence", "min_tracking_confidence", "classifier", "model_path",
        "pinch_threshold", "allow_shell", "overlay", "show_landmarks", "log_level", "hot_reload",
    )

    @classmethod
    def _parse(cls, ctx: _Ctx, node: Any, known_gestures) -> "Settings":
        path = "settings"
        if node is None:
            node = {}
        if not isinstance(node, dict):
            ctx.add(path, f"expected a mapping, got {type(node).__name__}", _line_of(node))
            node = {}
        _unknown_keys(ctx, node, path, list(cls._SCALARS) + ["arming", "mouse", "audio", "speech"])
        out = cls(
            camera_index=int(_get(ctx, node, path, "camera_index", 0, int)),
            camera_width=int(_get(ctx, node, path, "camera_width", 640, int)),
            camera_height=int(_get(ctx, node, path, "camera_height", 480, int)),
            mirror=_get(ctx, node, path, "mirror", True, bool),
            stable_frames=int(_get(ctx, node, path, "stable_frames", 5, int)),
            cooldown_ms=int(_get(ctx, node, path, "cooldown_ms", 800, int)),
            min_detection_confidence=_get(ctx, node, path, "min_detection_confidence", 0.6, float),
            min_tracking_confidence=_get(ctx, node, path, "min_tracking_confidence", 0.5, float),
            classifier=_get(ctx, node, path, "classifier", "rules", str, choices=CLASSIFIERS),
            model_path=_get(ctx, node, path, "model_path", "data/model.npz", str),
            pinch_threshold=_get(ctx, node, path, "pinch_threshold", 0.04, float),
            allow_shell=_get(ctx, node, path, "allow_shell", False, bool),
            overlay=_get(ctx, node, path, "overlay", True, bool),
            show_landmarks=_get(ctx, node, path, "show_landmarks", True, bool),
            log_level=_get(ctx, node, path, "log_level", "info", str, choices=("debug", "info", "warning", "error")),
            hot_reload=_get(ctx, node, path, "hot_reload", True, bool),
            arming=ArmingSettings._parse(ctx, node.get("arming") or {}, f"{path}.arming", known_gestures),
            mouse=MouseSettings._parse(ctx, node.get("mouse") or {}, f"{path}.mouse"),
            audio=AudioSettings._parse(ctx, node.get("audio") or {}, f"{path}.audio"),
            speech=SpeechSettings._parse(ctx, node.get("speech") or {}, f"{path}.speech"),
        )
        _bounded(ctx, out.stable_frames, f"{path}.stable_frames", _line_of(node, "stable_frames"),
                 minimum=1, maximum=60, name="stable_frames")
        _bounded(ctx, out.cooldown_ms, f"{path}.cooldown_ms", _line_of(node, "cooldown_ms"),
                 minimum=0, maximum=60000, name="cooldown_ms")
        _bounded(ctx, out.camera_index, f"{path}.camera_index", _line_of(node, "camera_index"),
                 minimum=0, maximum=32, name="camera_index")
        _bounded(ctx, out.pinch_threshold, f"{path}.pinch_threshold", _line_of(node, "pinch_threshold"),
                 minimum=0.005, maximum=0.5, name="pinch_threshold")
        for name in ("min_detection_confidence", "min_tracking_confidence"):
            _bounded(ctx, getattr(out, name), f"{path}.{name}", _line_of(node, name),
                     minimum=0.0, maximum=1.0, name=name)
        return out


# --------------------------------------------------------------------------- bindings


@dataclass(frozen=True, slots=True)
class Trigger:
    type: str
    value: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.type, self.value)


@dataclass(frozen=True, slots=True)
class Action:
    type: str
    keys: tuple[str, ...] = ()
    command: str | None = None
    repeat: int = 1
    interval_ms: int = 0
    amount: int = 0
    button: str = "left"
    clicks: int = 1
    target: str | None = None
    verb: str = "toggle"
    x: float | None = None
    y: float | None = None

    def describe(self) -> str:
        if self.type in ("key", "hotkey"):
            joined = "+".join(self.keys) if self.type == "hotkey" else " ".join(self.keys)
            return f"{self.type} {joined}" + (f" x{self.repeat}" if self.repeat > 1 else "")
        if self.type == "shell":
            return f"shell {self.command!r}"
        if self.type == "scroll":
            return f"scroll {self.amount:+d}"
        if self.type == "mouse_click":
            return f"{self.button} click x{self.clicks}"
        if self.type == "mode":
            return f"mode {self.verb} {self.target}"
        return self.type


@dataclass(frozen=True, slots=True)
class Binding:
    name: str
    trigger: Trigger
    action: Action
    enabled: bool = True
    cooldown_ms: int | None = None
    requires_arm: bool | None = None
    line: int | None = None

    @property
    def key(self) -> tuple[str, str]:
        return self.trigger.key


def _parse_trigger(ctx: _Ctx, node: Any, path: str, known_gestures, speech_phrases: list[str]) -> Trigger:
    if not isinstance(node, dict):
        ctx.add(path, f"expected a mapping like {{ type: gesture, value: open_palm }}, got {node!r}", _line_of(node))
        return Trigger("gesture", "__invalid__")
    _unknown_keys(ctx, node, path, ("type", "value"))
    ttype = _get(ctx, node, path, "type", None, str, choices=TRIGGER_KINDS)
    if ttype is None:
        ctx.add(f"{path}.type", "missing required key 'type'", _line_of(node),
                f"one of: {', '.join(sorted(TRIGGER_KINDS))}")
        ttype = "gesture"
    raw_value = node.get("value")
    line = _line_of(node, "value")
    if raw_value is None:
        ctx.add(f"{path}.value", "missing required key 'value'", _line_of(node))
        return Trigger(ttype, "__invalid__")
    value = str(raw_value)
    if ttype == EventKind.GESTURE.value:
        if value in RESERVED_LABELS:
            ctx.add(
                f"{path}.value",
                f"{value!r} is reserved and cannot be bound",
                line,
                "'none' means 'no hand in frame'; binding it would fire every time you lower your hand",
            )
        elif value not in known_gestures:
            ctx.add(f"{path}.value", f"unknown gesture {value!r}", line,
                    suggest(value, known_gestures, "known gestures"))
    elif ttype == EventKind.SOUND.value:
        if value not in SOUND_VALUES:
            ctx.add(f"{path}.value", f"unknown sound {value!r}", line, suggest(value, SOUND_VALUES, "known sounds"))
    elif ttype == EventKind.SPEECH.value:
        if not value.strip():
            ctx.add(f"{path}.value", "speech phrase is empty", line)
        speech_phrases.append(value)
    return Trigger(ttype, value)


def _parse_action(ctx: _Ctx, node: Any, path: str, allow_shell: bool) -> Action:
    if not isinstance(node, dict):
        ctx.add(path, f"expected a mapping like {{ type: key, keys: [space] }}, got {node!r}", _line_of(node))
        return Action("noop")
    atype = _get(ctx, node, path, "type", None, str, choices=ACTION_TYPES)
    if atype is None:
        ctx.add(f"{path}.type", "missing required key 'type'", _line_of(node), f"one of: {', '.join(ACTION_TYPES)}")
        return Action("noop")

    allowed = {"type"}
    if atype in ("key", "hotkey"):
        allowed |= {"keys", "repeat", "interval_ms"}
    elif atype == "shell":
        allowed |= {"command"}
    elif atype == "scroll":
        allowed |= {"amount", "repeat", "interval_ms"}
    elif atype == "mouse_click":
        allowed |= {"button", "clicks"}
    elif atype == "mouse_move":
        allowed |= {"x", "y"}
    elif atype == "mode":
        allowed |= {"target", "verb"}
    _unknown_keys(ctx, node, path, allowed)

    keys: tuple[str, ...] = ()
    if atype in ("key", "hotkey"):
        raw = node.get("keys")
        line = _line_of(node, "keys")
        if raw is None:
            ctx.add(f"{path}.keys", f"'{atype}' action requires 'keys'", _line_of(node), "example: keys: [ctrl, c]")
        elif isinstance(raw, str):
            keys = (raw,)
        elif isinstance(raw, list) and all(isinstance(k, str) for k in raw):
            keys = tuple(raw)
        else:
            ctx.add(f"{path}.keys", f"expected a string or list of strings, got {raw!r}", line)
        if atype == "hotkey" and len(keys) == 1:
            ctx.add(
                f"{path}.keys",
                "'hotkey' presses keys simultaneously and needs at least two",
                line,
                "for a single key use type: key",
            )

    command = None
    if atype == "shell":
        raw = node.get("command")
        line = _line_of(node, "command")
        if not isinstance(raw, str) or not raw.strip():
            ctx.add(f"{path}.command", "'shell' action requires a non-empty 'command' string", line)
        else:
            command = raw
        if not allow_shell:
            ctx.add(
                f"{path}.command",
                "shell actions are disabled",
                line,
                "set 'settings.allow_shell: true' to enable them - a shared config that runs "
                "arbitrary commands is a real attack surface, so this is opt-in (FR-5.5)",
            )

    amount = 0
    if atype == "scroll":
        amount = int(_get(ctx, node, path, "amount", 0, int))
        if amount == 0:
            ctx.add(f"{path}.amount", "scroll amount of 0 does nothing", _line_of(node, "amount"),
                    "positive scrolls up, negative scrolls down")

    target, verb = None, "toggle"
    if atype == "mode":
        target = _get(ctx, node, path, "target", None, str, choices=MODE_TARGETS)
        verb = _get(ctx, node, path, "verb", "toggle", str, choices=MODE_VERBS)
        if target is None:
            ctx.add(f"{path}.target", "'mode' action requires 'target'", _line_of(node),
                    f"one of: {', '.join(MODE_TARGETS)}")

    button = _get(ctx, node, path, "button", "left", str, choices=("left", "right", "middle"))
    repeat = int(_get(ctx, node, path, "repeat", 1, int))
    _bounded(ctx, repeat, f"{path}.repeat", _line_of(node, "repeat"), minimum=1, maximum=50, name="repeat")
    clicks = int(_get(ctx, node, path, "clicks", 1, int))
    _bounded(ctx, clicks, f"{path}.clicks", _line_of(node, "clicks"), minimum=1, maximum=5, name="clicks")

    x = node.get("x")
    y = node.get("y")
    return Action(
        type=atype,
        keys=keys,
        command=command,
        repeat=repeat,
        interval_ms=int(_get(ctx, node, path, "interval_ms", 0, int)),
        amount=amount,
        button=button,
        clicks=clicks,
        target=target,
        verb=verb,
        x=float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None,
        y=float(y) if isinstance(y, (int, float)) and not isinstance(y, bool) else None,
    )


@dataclass(slots=True)
class Config:
    settings: Settings = field(default_factory=Settings)
    bindings: tuple[Binding, ...] = ()
    custom_gestures: tuple[str, ...] = ()
    source: str | None = None

    def bindings_for(self, kind: str, value: str) -> list[Binding]:
        return [b for b in self.bindings if b.enabled and b.trigger.type == kind and b.trigger.value == value]

    @property
    def speech_phrases(self) -> tuple[str, ...]:
        return tuple(b.trigger.value for b in self.bindings if b.enabled and b.trigger.type == "speech")

    @property
    def gesture_labels(self) -> tuple[str, ...]:
        return tuple(BUILTIN_GESTURES) + tuple(self.custom_gestures)

    def with_source(self, source: str) -> "Config":
        return replace(self, source=source)


def parse_config(
    data: Any,
    *,
    custom_gestures: Iterable[str] = (),
    source: str | None = None,
) -> tuple[Config, list[ConfigIssue]]:
    """Validate a parsed YAML document into a :class:`Config` plus every issue.

    Returns issues instead of raising so callers can choose: the CLI raises,
    the hot-reloader logs and keeps the last good config running.
    """
    ctx = _Ctx()
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return Config(source=source), [ConfigIssue("<root>", f"expected a YAML mapping, got {type(data).__name__}")]

    _unknown_keys(ctx, data, "", ("settings", "bindings", "custom_gestures"))

    declared_custom = data.get("custom_gestures") or []
    customs: list[str] = []
    if isinstance(declared_custom, list):
        for i, name in enumerate(declared_custom):
            if isinstance(name, str) and name:
                customs.append(name)
            else:
                ctx.add(f"custom_gestures[{i}]", f"expected a gesture name string, got {name!r}",
                        _line_of(data, "custom_gestures"))
    elif declared_custom:
        ctx.add("custom_gestures", "expected a list of gesture names", _line_of(data, "custom_gestures"))

    known_gestures = tuple(BUILTIN_GESTURES) + tuple(customs) + tuple(custom_gestures)
    settings = Settings._parse(ctx, data.get("settings"), known_gestures)

    raw_bindings = data.get("bindings")
    if raw_bindings is None:
        raw_bindings = []
    if not isinstance(raw_bindings, list):
        ctx.add("bindings", f"expected a list of bindings, got {type(raw_bindings).__name__}",
                _line_of(data, "bindings"))
        raw_bindings = []

    bindings: list[Binding] = []
    speech_phrases: list[str] = []
    seen: dict[tuple[str, str], str] = {}
    for i, raw in enumerate(raw_bindings):
        path = f"bindings[{i}]"
        if not isinstance(raw, dict):
            ctx.add(path, f"expected a mapping, got {raw!r}", _line_of(raw))
            continue
        _unknown_keys(ctx, raw, path, ("name", "trigger", "action", "enabled", "cooldown_ms", "requires_arm"))
        name = _get(ctx, raw, path, "name", f"binding {i}", str)
        if "trigger" not in raw:
            ctx.add(f"{path}.trigger", "missing required key 'trigger'", _line_of(raw))
            continue
        if "action" not in raw:
            ctx.add(f"{path}.action", "missing required key 'action'", _line_of(raw))
            continue
        trigger = _parse_trigger(ctx, raw["trigger"], f"{path}.trigger", known_gestures, speech_phrases)
        action = _parse_action(ctx, raw["action"], f"{path}.action", settings.allow_shell)

        cooldown = raw.get("cooldown_ms")
        if cooldown is not None and (isinstance(cooldown, bool) or not isinstance(cooldown, (int, float))):
            ctx.add(f"{path}.cooldown_ms", f"expected a number, got {cooldown!r}", _line_of(raw, "cooldown_ms"))
            cooldown = None
        requires_arm = raw.get("requires_arm")
        if requires_arm is not None and not isinstance(requires_arm, bool):
            ctx.add(f"{path}.requires_arm", f"expected true or false, got {requires_arm!r}",
                    _line_of(raw, "requires_arm"))
            requires_arm = None

        binding = Binding(
            name=name,
            trigger=trigger,
            action=action,
            enabled=_get(ctx, raw, path, "enabled", True, bool),
            cooldown_ms=int(cooldown) if cooldown is not None else None,
            requires_arm=requires_arm,
            line=_line_of(raw),
        )
        if binding.enabled and trigger.value != "__invalid__":
            previous = seen.get(trigger.key)
            if previous is not None:
                ctx.add(
                    path,
                    f"duplicate trigger {trigger.type}:{trigger.value} - already bound by {previous!r}",
                    binding.line,
                    "one trigger fires one action; disable or remove one of them",
                )
            else:
                seen[trigger.key] = name
        bindings.append(binding)

    settings_node = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    if settings.arming.enabled:
        armed_key = ("gesture", settings.arming.gesture)
        if armed_key in seen:
            arming_node = settings_node.get("arming") if isinstance(settings_node.get("arming"), dict) else {}
            ctx.add(
                "settings.arming.gesture",
                f"{settings.arming.gesture!r} is both the arming gesture and bound to {seen[armed_key]!r}",
                _line_of(arming_node, "gesture"),
                "arming consumes the gesture, so that binding would never fire",
            )

    if not settings.speech.enabled and speech_phrases:
        speech_node = settings_node.get("speech") if isinstance(settings_node.get("speech"), dict) else {}
        ctx.add(
            "settings.speech.enabled",
            f"{len(speech_phrases)} speech binding(s) declared but speech is disabled",
            _line_of(speech_node, "enabled") or _line_of(settings_node, "speech"),
            "set settings.speech.enabled: true, or remove the speech bindings",
        )

    cfg = Config(settings=settings, bindings=tuple(bindings), custom_gestures=tuple(customs), source=source)
    return cfg, ctx.issues
