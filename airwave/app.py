"""The runtime: three threads, one process (PRD 3.3).

* **Vision** runs on the main thread. Not a preference - ``cv2.imshow`` must be
  called from the main thread on macOS, and moving it later would be a rewrite.
* **Audio** is sounddevice's own callback thread; it only ever appends to a
  queue.
* **Dispatch** blocks on the queue with a timeout so it notices the stop event.

The GIL is not the bottleneck it looks like: MediaPipe and OpenCV release it
during inference, and the audio path is pure numpy. If profiling ever says
otherwise, the vision stage moves to a ``multiprocessing.Process`` and the
queue becomes a ``multiprocessing.Queue`` - nothing else changes, which is the
whole reason producers only ever emit events.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .audio.producer import AudioProducer
from .config import Config, ConfigWatcher
from .dispatch.actions import ActionExecutor, Controls
from .dispatch.dispatcher import Dispatcher
from .dispatch.platform import InputBackend, detect
from .events import Event, EventKind
from .vision.landmarks import HandFrame, HandTracker
from .vision.pipeline import VisionPipeline

log = logging.getLogger("airwave.app")


@dataclass(slots=True)
class RunStats:
    """What ``airwave run`` prints on exit, and what the tests assert on."""

    frames: int = 0
    started_at: float = field(default_factory=time.monotonic)
    fps: float = 0.0
    p95_frame_ms: float = 0.0

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at


class AirwaveApp:
    """Owns every thread and the shutdown path."""

    def __init__(
        self,
        cfg: Config,
        *,
        config_path: str | Path | None = None,
        dry_run: bool = False,
        headless: bool = False,
        audio: bool | None = None,
    ) -> None:
        self.cfg = cfg
        self.config_path = Path(config_path) if config_path else None
        self.dry_run = dry_run
        self.headless = headless
        self.audio_enabled = cfg.settings.audio.enabled if audio is None else audio

        self.backend = InputBackend(dry_run=dry_run)
        self.controls = Controls(
            set_mouse=self.set_mouse,
            set_armed=self.set_armed,
            set_paused=self.set_paused,
            get_mouse=lambda: self.pipeline.mouse_active,
            get_armed=lambda: self.dispatcher.is_armed(),
            get_paused=lambda: self.dispatcher.paused,
        )
        self.executor = ActionExecutor(self.backend, allow_shell=cfg.settings.allow_shell, controls=self.controls)
        self.dispatcher = Dispatcher(cfg, executor=self.executor)
        self.pipeline = VisionPipeline(cfg)
        self.overlay = None
        self.tracker: HandTracker | None = None
        self.camera = None
        self.audio_producer: AudioProducer | None = None
        self.watcher: ConfigWatcher | None = None
        self.stats = RunStats()

        self._pending_config: Config | None = None
        self._config_lock = threading.Lock()
        self._stop = threading.Event()

    # -------------------------------------------------------------- controls

    def set_mouse(self, active: bool) -> None:
        self.pipeline.set_mouse_active(active)
        log.info("mouse control %s", "on" if active else "off")

    def set_armed(self, armed: bool) -> None:
        self.dispatcher.arm() if armed else self.dispatcher.disarm()

    def set_paused(self, paused: bool) -> None:
        self.dispatcher.paused = paused
        log.info("dispatcher %s", "paused" if paused else "resumed")

    def stop(self) -> None:
        self._stop.set()

    # ---------------------------------------------------------- config reload

    def _on_reload(self, cfg: Config) -> None:
        """Called from the watcher thread. Only stages the config.

        Applying it here would mutate the classifier and stabilizer out from
        under the vision thread mid-frame; the main loop picks it up between
        frames instead.
        """
        with self._config_lock:
            self._pending_config = cfg
        log.info("config reloaded from %s", cfg.source)

    def _apply_pending(self) -> None:
        with self._config_lock:
            cfg, self._pending_config = self._pending_config, None
        if cfg is None:
            return
        self.cfg = cfg
        self.dispatcher.apply_config(cfg)
        self.pipeline.apply_config(cfg)
        if self.audio_producer is not None:
            self.audio_producer.apply_config(cfg)

    @staticmethod
    def _on_reload_error(issues) -> None:
        from .config import ConfigError  # noqa: PLC0415

        print("\n" + ConfigError(issues).render() + "\nkeeping the previous config\n")

    # -------------------------------------------------------------- lifecycle

    def startup_checks(self) -> None:
        """Say what will not work *before* the user concludes it is broken."""
        info = detect()
        for warning in info.warnings:
            log.warning(warning)
        for blocker in info.blocking:
            print(f"\n!! {blocker}\n")
        if self.cfg.settings.allow_shell:
            shell_bindings = [b.name for b in self.cfg.bindings if b.action.type == "shell"]
            if shell_bindings:
                log.warning("shell actions are enabled for: %s", ", ".join(shell_bindings))

    def start(self) -> None:
        from .capture.camera import Camera  # noqa: PLC0415

        self.startup_checks()
        settings = self.cfg.settings
        self.camera = Camera(
            settings.camera_index,
            width=settings.camera_width,
            height=settings.camera_height,
            mirror=settings.mirror,
        ).open()
        self.tracker = HandTracker(
            max_num_hands=1,
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        self.dispatcher.start()

        if self.audio_enabled:
            self.audio_producer = AudioProducer(self.cfg, self.dispatcher.submit)
            self.audio_producer.start()

        if settings.hot_reload and self.config_path is not None:
            self.watcher = ConfigWatcher(
                self.config_path,
                self._on_reload,
                on_error=self._on_reload_error,
                custom_gestures=self.cfg.custom_gestures,
            )
            self.watcher.start()

        if not self.headless and settings.overlay:
            from .ui.overlay import Overlay  # noqa: PLC0415

            self.overlay = Overlay(show_landmarks=settings.show_landmarks)

        if settings.mouse.enabled:
            self.pipeline.set_mouse_active(True)

    def shutdown(self) -> None:
        for closer in (
            lambda: self.watcher and self.watcher.stop(),
            lambda: self.audio_producer and self.audio_producer.stop(),
            lambda: self.dispatcher.stop(),
            lambda: self.tracker and self.tracker.close(),
            lambda: self.camera and self.camera.close(),
            lambda: self.overlay and self.overlay.close(),
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001 - shutdown must complete
                log.exception("error during shutdown")

    # -------------------------------------------------------------- main loop

    def run(self, *, max_frames: int | None = None) -> RunStats:
        self.start()
        try:
            self._loop(max_frames)
        except KeyboardInterrupt:
            print()
        finally:
            self.stats.fps = self.pipeline.fps
            self.stats.p95_frame_ms = self.pipeline.p95_frame_ms()
            self.shutdown()
        return self.stats

    def _loop(self, max_frames: int | None) -> None:
        assert self.camera is not None and self.tracker is not None
        while not self._stop.is_set():
            if max_frames is not None and self.stats.frames >= max_frames:
                return
            self._apply_pending()

            frame = self.camera.read()
            if frame is None:
                continue
            self.stats.frames += 1

            hand = self.tracker.process(frame.image, timestamp_ms=int(frame.captured_at * 1000))
            result = self.pipeline.process(hand, captured_at=frame.captured_at, aspect=frame.aspect)
            self.dispatcher.submit_all(result.events)

            if self.overlay is not None:
                extra = {
                    "p95_ms": self.pipeline.p95_frame_ms(),
                    "mouse_active": self.pipeline.mouse_active,
                    "audio": self.audio_producer.snapshot if self.audio_producer else None,
                }
                self.overlay.render(frame.image, result, self.dispatcher, extra=extra)
                self.overlay.show(frame.image)
                if self._handle_key(self.overlay.poll_key()) or self.overlay.window_closed():
                    return

    def _handle_key(self, key: int) -> bool:
        """Returns True to quit. Keyboard shortcuts mirror the overlay's help."""
        if key in (27, ord("q")):
            return True
        if key == ord("a"):
            self.dispatcher.arm()
        elif key == ord("m"):
            self.pipeline.set_mouse_active(not self.pipeline.mouse_active)
        elif key == ord("p"):
            self.set_paused(not self.dispatcher.paused)
        elif key == ord("l") and self.overlay is not None:
            self.overlay.state.show_landmarks = not self.overlay.state.show_landmarks
        elif key == ord("h") and self.overlay is not None:
            self.overlay.state.show_help = not self.overlay.state.show_help
        return False


def replay_video(
    path: str | Path,
    cfg: Config,
    *,
    dry_run: bool = True,
    max_frames: int | None = None,
) -> list[Event]:
    """Run a recorded clip through the real vision pipeline.

    This is the regression harness the PRD asks for: every false positive that
    gets fixed leaves behind a clip and an expected event list, and the whole
    suite runs on CI with no camera attached.
    """
    import cv2  # noqa: PLC0415

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")

    tracker = HandTracker(
        max_num_hands=1,
        min_detection_confidence=cfg.settings.min_detection_confidence,
        min_tracking_confidence=cfg.settings.min_tracking_confidence,
    )
    pipeline = VisionPipeline(cfg)
    events: list[Event] = []
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok or image is None:
                break
            index += 1
            if max_frames is not None and index > max_frames:
                break
            if cfg.settings.mirror:
                image = cv2.flip(image, 1)
            hand = tracker.process(image, timestamp_ms=index * 33)
            result = pipeline.process(hand, aspect=image.shape[0] / image.shape[1])
            events.extend(e for e in result.events if e.kind is not EventKind.POINTER)
    finally:
        capture.release()
        tracker.close()
    return events


def replay_labels(labels, cfg: Config) -> list:
    """Drive the dispatcher from a label sequence, no camera or model needed."""
    dispatcher = Dispatcher(cfg, executor=ActionExecutor(InputBackend(dry_run=True),
                                                         allow_shell=cfg.settings.allow_shell))
    pipeline = VisionPipeline(cfg, classifier=None)
    records = []
    for event in pipeline.run_labels(labels):
        records.append(dispatcher.handle(event))
    return records


def replay_hand_frames(frames, cfg: Config, *, clock=None) -> list[Event]:
    """Feed pre-extracted landmark arrays through classify + stabilize.

    Used by the integration tests: a stored ``.npz`` of landmarks is a video
    clip with the MediaPipe step already paid for, so the regression suite runs
    in milliseconds instead of minutes.
    """
    pipeline = VisionPipeline(cfg, clock=clock or time.monotonic)
    events: list[Event] = []
    for item in frames:
        hand = item if isinstance(item, HandFrame) else HandFrame(item)
        events.extend(pipeline.process(hand).events)
    return events
