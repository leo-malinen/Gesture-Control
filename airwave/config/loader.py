"""YAML loading with line tracking, plus hot reload (FR-5.3, FR-5.4).

Two things live here that a plain ``yaml.safe_load`` does not give you:

1. Every mapping remembers the line it came from, so "unknown gesture
   'thumbs_upp'" can point at *that* line instead of at the file.
2. A watcher that polls the file rather than pulling in ``watchdog``. Reading
   a few kilobytes twice a second costs nothing and keeps the install to one
   requirements file, which is a stated product goal.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

from .errors import ConfigError, ConfigIssue
from .schema import Config, parse_config

log = logging.getLogger("airwave.config")

DEFAULT_CONFIG_NAME = "airwave.yaml"


class LinedDict(dict):
    """A dict that remembers where it and each of its keys were written."""

    __slots__ = ("line", "key_lines")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.line: int | None = None
        self.key_lines: dict[str, int] = {}


class LineLoader(yaml.SafeLoader):
    """SafeLoader (never FullLoader - a config file is untrusted input)."""


def _construct_mapping(loader: LineLoader, node: yaml.MappingNode) -> LinedDict:
    loader.flatten_mapping(node)
    out = LinedDict()
    out.line = node.start_mark.line + 1
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        value = loader.construct_object(value_node, deep=True)
        out[key] = value
        if isinstance(key, str):
            out.key_lines[key] = key_node.start_mark.line + 1
    return out


LineLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(text: str, source: str | None = None) -> Any:
    """Parse YAML, converting parser errors into our own issue format."""
    try:
        return yaml.load(text, Loader=LineLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        line = mark.line + 1 if mark is not None else None
        problem = getattr(exc, "problem", None) or str(exc)
        hint = None
        if "found character '\\t'" in str(exc):
            hint = "YAML forbids tabs for indentation - use spaces"
        elif "mapping values are not allowed" in str(exc):
            hint = "a colon inside a value needs quotes, e.g. command: \"open -a Notes\""
        raise ConfigError([ConfigIssue("<syntax>", f"YAML syntax error: {problem}", line, hint)], source) from exc


def load_config(path: str | os.PathLike[str], *, custom_gestures=()) -> Config:
    """Load and validate a config file, raising :class:`ConfigError` on problems."""
    p = Path(path)
    if not p.exists():
        raise ConfigError([ConfigIssue("<file>", f"config file not found: {p}",
                                       hint="run 'airwave init' to write a starter config")], str(p))
    text = p.read_text(encoding="utf-8")
    data = load_yaml(text, str(p))
    cfg, issues = parse_config(data, custom_gestures=custom_gestures, source=str(p))
    if issues:
        raise ConfigError(issues, str(p))
    return cfg


def load_config_lenient(path: str | os.PathLike[str], *, custom_gestures=()) -> tuple[Config | None, list[ConfigIssue]]:
    """Load without raising. Used by hot reload, which must survive a bad save.

    An editor writing a half-finished file should not take down a running
    session; the caller keeps the previous config and shows the errors.
    """
    try:
        return load_config(path, custom_gestures=custom_gestures), []
    except ConfigError as exc:
        return None, exc.issues


def find_config(explicit: str | None = None, start: str | os.PathLike[str] | None = None) -> Path:
    """Resolve which config file to use.

    Order: explicit ``--config`` > ``AIRWAVE_CONFIG`` > ``./airwave.yaml`` >
    ``~/.config/airwave/airwave.yaml``. The last one is what makes an installed
    copy usable from any directory.
    """
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("AIRWAVE_CONFIG")
    if env:
        return Path(env).expanduser()
    here = Path(start or Path.cwd())
    local = here / DEFAULT_CONFIG_NAME
    if local.exists():
        return local
    user = Path.home() / ".config" / "airwave" / DEFAULT_CONFIG_NAME
    if user.exists():
        return user
    return local


class ConfigWatcher:
    """Polls a file's contents and calls back with a freshly parsed config.

    The signature is a hash of the bytes, not mtime and size. Those are the
    obvious choice and they are wrong here: the single most common edit is
    changing one digit (``cooldown_ms: 800`` -> ``400``), which leaves the size
    identical, and filesystem timestamp granularity can leave the mtime
    identical too. Hashing a few kilobytes twice a second costs nothing and
    turns "sometimes reloads" into "always reloads".
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        on_reload: Callable[[Config], None],
        *,
        on_error: Callable[[list[ConfigIssue]], None] | None = None,
        interval_s: float = 0.5,
        custom_gestures=(),
    ) -> None:
        self.path = Path(path)
        self.on_reload = on_reload
        self.on_error = on_error
        self.interval_s = interval_s
        self.custom_gestures = tuple(custom_gestures)
        self._stamp = self._signature()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _signature(self) -> str | None:
        """Hash of the file's bytes, or ``None`` if it is unreadable right now
        (mid-save, or temporarily replaced by an editor's atomic rename)."""
        try:
            return hashlib.sha1(self.path.read_bytes()).hexdigest()
        except OSError:
            return None

    def poll(self) -> bool:
        """Check once. Returns True if a reload was applied.

        Split out from the thread so tests can drive it deterministically.
        """
        signature = self._signature()
        if signature is None or signature == self._stamp:
            return False
        self._stamp = signature
        cfg, issues = load_config_lenient(self.path, custom_gestures=self.custom_gestures)
        if cfg is None:
            log.warning("config reload rejected, keeping previous config")
            if self.on_error:
                self.on_error(issues)
            return False
        self.on_reload(cfg)
        return True

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="airwave-config-watch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.poll()
            except Exception:  # pragma: no cover - a watcher must never kill the app
                log.exception("config watcher error")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
