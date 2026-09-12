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
        hub: bool = False,
        hub_port: int = 8760,
    ) -> None:
        self.cfg = cfg
        self.config_path = Path(config_path) if config_path else None
        self.dry_run = dry_run
        self.headless = headless
        self.audio_enabled = cfg.settings.audio.enabled if audio is None else audio
        self.hub_enabled = hub
        self.hub_port = hub_port
        self.hub_state = None
        self.hub_server = None
        self._hub_painter = None
        self.dictation_active = False

        self.backend = InputBackend(dry_run=dry_run)
        self.controls = Controls(
            set_mouse=self.set_mouse,
            set_armed=self.set_armed,
            set_paused=self.set_paused,
            set_dictation=self.set_dictation,
            get_mouse=lambda: self.pipeline.mouse_active,
            get_armed=lambda: self.dispatcher.is_armed(),
            get_paused=lambda: self.dispatcher.paused,
            get_dictation=lambda: self.dictation_active,
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

    def set_dictation(self, active: bool) -> None:
        """Start or stop typing what the user says.

        Says so out loud when it cannot start, because a dictation mode that
        silently produces no text is indistinguishable from a broken one - and
        the two causes (speech disabled, no ASR engine installed) both have a
        one-line fix the user needs to hear.
        """
        self.dictation_active = active
        started = self.audio_producer.set_dictation(active) if self.audio_producer else False
        if active and not started:
            self.dictation_active = False
            reason = ("settings.speech.enabled is false" if not self.cfg.settings.speech.enabled
                      else "no speech listener is running")
            print(
                f"\n!! dictation cannot start: {reason}\n"
                f"   enable speech in airwave.yaml, then: pip install vosk\n"
            )
            log.warning("dictation requested but %s", reason)
            return
        log.info("dictation %s", "started" if active else "stopped")

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

        # The hub comes up first, deliberately. If the camera or the hand model
        # is missing, the browser should show a page explaining that - which it
        # cannot do if the process died opening the camera three lines earlier.
        if self.hub_enabled:
            self._start_hub()

        try:
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
            self._publish_status(camera="live", camera_detail="")
        except Exception as exc:  # noqa: BLE001
            if not self.hub_enabled:
                raise
            # Degrade instead of exiting: audio triggers and the log still work,
            # and the hub explains what is missing.
            self._report_capture_failure(exc)

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

    # ------------------------------------------------------------------- hub

    def _start_hub(self) -> None:
        from .ui.hub import HubServer, HubState  # noqa: PLC0415

        self.hub_state = HubState()
        self.hub_server = HubServer(self.hub_state, port=self.hub_port).start()
        # The hub is a read-only tap on dispatch, not a second control path.
        self.dispatcher.observers.append(self.hub_state.add_record)
        self._publish_status(camera="starting", camera_detail="Opening the capture device.")
        print(f"hub:  {self.hub_server.url}")

    def _publish_status(self, **fields) -> None:
        if self.hub_state is not None:
            self.hub_state.publish_status(**fields)

    def _report_capture_failure(self, exc: Exception) -> None:
        """Turn a startup exception into something the page can render.

        The exception messages already carry the fix (that is the house style);
        this splits the first line off as a title so the UI has a heading and a
        copy-pasteable remedy rather than one long red paragraph.
        """
        from .vision.landmarks import ModelMissingError  # noqa: PLC0415

        text = str(exc).strip()
        lines = [line for line in text.splitlines() if line.strip()]
        title = "Hand model missing" if isinstance(exc, ModelMissingError) else "No camera"
        detail = lines[0] if lines else text
        fix = "\n".join(line.strip() for line in lines[1:]) or None

        log.warning("%s: %s", title, text)
        print(f"\n!! {text}\n")
        self._publish_status(camera="error", camera_title=title, camera_detail=detail, camera_fix=fix)
        self.camera = None
        self.tracker = None

    def _publish_hub_frame(self, frame, result) -> None:
        """Draw landmarks onto a copy and hand it to the hub.

        A copy because the overlay draws its own chrome onto the same array
        immediately afterwards, and the hub's HTML supplies that chrome itself -
        the browser should get the feed plus landmarks, not a second HUD.
        """
        image = frame.image.copy()
        if result.hand is not None and result.hand.found and self.cfg.settings.show_landmarks:
            self._hub_overlay().draw_landmarks(image, result.hand.landmarks)
        if self.pipeline.mouse_active:
            self._hub_overlay().draw_active_region(image, self.cfg.settings.mouse.active_region)
        self.hub_state.publish_frame(image)
        self._publish_hub_status(result)

    def _hub_overlay(self):
        """Reuse the overlay's drawing primitives without its window."""
        if self._hub_painter is None:
            from .ui.overlay import Overlay  # noqa: PLC0415

            self._hub_painter = Overlay()
        return self._hub_painter

    def _publish_hub_status(self, result=None) -> None:
        if self.hub_state is None:
            return
        stats = self.dispatcher.stats
        arming = self.cfg.settings.arming
        status = {
            "camera": "live" if self.camera is not None else self.hub_state.status().get("camera", "error"),
            "fps": round(self.pipeline.fps, 1),
            "p95_ms": round(self.pipeline.p95_frame_ms(), 1),
            "arming_enabled": arming.enabled,
            "arming_gesture": arming.gesture,
            "armed": self.dispatcher.is_armed(),
            "arm_remaining": round(self.dispatcher.arm_remaining(), 1),
            "paused": self.dispatcher.paused,
            "mouse_active": self.pipeline.mouse_active,
            "dictation": self.dictation_active,
            "audio_active": self.audio_producer is not None and self.audio_producer.microphone is not None,
            "fired": stats.fired,
            "cooldown": stats.cooldown,
            "unarmed": stats.unarmed,
            "unbound": stats.unbound,
            "errors": stats.errors,
            "mean_latency_ms": round(stats.mean_latency_ms, 1),
        }
        if result is not None:
            status.update({
                "gesture": result.classification.label,
                "confidence": round(result.classification.confidence, 2),
                "stability": round(result.stability, 3),
                "hold_progress": round(result.hold_progress, 3),
                "moving": result.moving,
                "hand_speed": result.hand_speed,
            })
        self.hub_state.publish_status(**status)

    def shutdown(self) -> None:
        for closer in (
            lambda: self.hub_server and self.hub_server.stop(),
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
        if self.camera is None or self.tracker is None:
            # Hub-only mode: no video, but audio triggers, the dispatcher and
            # the log all still work, and the page explains what is missing.
            return self._loop_without_camera()

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

            if self.hub_state is not None:
                self._publish_hub_frame(frame, result)

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

    def _loop_without_camera(self) -> None:
        """Keep the process (and therefore the hub and the audio thread) alive."""
        while not self._stop.is_set():
            self._apply_pending()
            self._publish_hub_status()
            if self._stop.wait(0.5):
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
