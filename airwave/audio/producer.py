"""Audio producer: microphone blocks in, typed events out.

Mirrors the vision pipeline's contract - this thread never acts, it only
emits. The audio callback path here is deliberately tiny: RMS, threshold,
maybe a buffer append. Everything expensive (ASR) happens on a worker thread
owned by :class:`~airwave.audio.speech.SpeechListener`.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from ..config import Config
from ..events import Event, EventKind
from .onset import ClapGrouper, OnsetDetector
from .speech import SpeechListener, build_engine

log = logging.getLogger("airwave.audio")

Emit = Callable[[Event], None]


class AudioProducer:
    """Clap/snap detection plus optional wake-word speech, in one handler."""

    def __init__(self, cfg: Config, emit: Emit, *, clock=time.monotonic) -> None:
        self.emit = emit
        self._clock = clock
        self.cfg = cfg
        settings = cfg.settings.audio
        self.detector = OnsetDetector(
            baseline_alpha=settings.baseline_alpha,
            multiplier=settings.multiplier,
            floor=settings.floor,
            refractory_ms=settings.refractory_ms,
        )
        self.grouper = ClapGrouper(
            min_gap_ms=settings.double_clap_min_ms,
            max_gap_ms=settings.double_clap_max_ms,
            detect_double=self._double_is_bound(cfg),
        )
        self.listener: SpeechListener | None = None
        self.microphone = None
        self._dictation = False
        if cfg.settings.speech.enabled:
            self._build_listener(cfg)

    @staticmethod
    def _double_is_bound(cfg: Config) -> bool:
        """Only pay the double-clap latency when something listens for it."""
        return any(b.enabled and b.trigger.type == "sound" and b.trigger.value == "double_clap"
                   for b in cfg.bindings)

    def _build_listener(self, cfg: Config) -> None:
        speech = cfg.settings.speech
        engine = build_engine(speech.engine, speech.model_path, cfg.settings.audio.samplerate)
        log.info("speech engine: %s", engine.name)
        self.listener = SpeechListener(
            engine=engine,
            wake_word=speech.wake_word,
            listen_window_s=speech.listen_window_s,
            samplerate=cfg.settings.audio.samplerate,
            on_phrase=self._on_phrase,
            on_wake=self._on_wake,
            on_dictation=self._on_dictation,
        )
        self.listener.dictation = self._dictation

    # ------------------------------------------------------------ hot reload

    def apply_config(self, cfg: Config) -> None:
        """Thresholds retune live; the ASR engine is only rebuilt if it changed
        (reloading a Whisper model on every file save would stall for seconds)."""
        old, self.cfg = self.cfg, cfg
        settings = cfg.settings.audio
        self.detector.baseline_alpha = settings.baseline_alpha
        self.detector.multiplier = settings.multiplier
        self.detector.floor = settings.floor
        self.detector.refractory_ms = settings.refractory_ms
        self.grouper.min_gap_ms = settings.double_clap_min_ms
        self.grouper.max_gap_ms = settings.double_clap_max_ms
        self.grouper.detect_double = self._double_is_bound(cfg)

        speech_changed = (
            old.settings.speech.engine != cfg.settings.speech.engine
            or old.settings.speech.model_path != cfg.settings.speech.model_path
            or old.settings.speech.enabled != cfg.settings.speech.enabled
        )
        if speech_changed:
            if self.listener is not None:
                self.listener.stop()
                self.listener = None
            if cfg.settings.speech.enabled:
                self._build_listener(cfg)
                self.listener.start()  # type: ignore[union-attr]
        elif self.listener is not None:
            self.listener.wake_word = cfg.settings.speech.wake_word
            self.listener.listen_window_s = cfg.settings.speech.listen_window_s

    # ----------------------------------------------------------- audio thread

    def handle_block(self, block: np.ndarray, now: float | None = None) -> None:
        """The microphone callback. Must stay allocation-light and non-blocking."""
        now = self._clock() if now is None else now
        onset = self.detector.push(block, now=now)
        speaking = self._suppress_sounds()
        for label, captured_at in self.grouper.feed(onset, now=now):
            if speaking:
                # Plosives are broadband transients: "t", "p" and "d" clear the
                # clap threshold comfortably. Measured on real speech, a single
                # dictated sentence produced six "claps" - which, with clap
                # bound to Next track, means talking skips your music. While
                # someone is speaking, a spike is a syllable, not a clap.
                log.debug("suppressed %s during speech", label)
                continue
            self.emit(Event(kind=EventKind.SOUND, value=label, captured_at=captured_at,
                            emitted_at=self._clock(),
                            payload={"rms": round(self.detector.last_rms, 4),
                                     "threshold": round(self.detector.threshold, 4)}))
        if self.listener is not None:
            self.listener.push(block, self.detector.last_rms, now)

    def _suppress_sounds(self) -> bool:
        """Whether a detected transient should be discarded rather than emitted.

        Two cases, both "the user is talking": dictation is on, or the voice
        activity detector currently has an utterance open. Missing a clap while
        someone speaks is the right trade - the PRD budgets misses far more
        cheaply than false fires.
        """
        if self.listener is None:
            return False
        return bool(self._dictation or self.listener.vad.speaking)

    # -------------------------------------------------------------- callbacks

    def _on_phrase(self, phrase: str, captured_at: float) -> None:
        """Raw transcript goes on the queue; the dispatcher does the matching.

        Keeping fuzzy matching in the dispatcher means the audio thread never
        needs to know the binding table, and a config reload changes what a
        phrase means without touching this thread at all.
        """
        self.emit(Event(kind=EventKind.SPEECH, value=phrase, captured_at=captured_at,
                        emitted_at=self._clock(), payload={"raw": phrase}))

    def set_dictation(self, active: bool) -> bool:
        """Turn free transcription on or off. Returns whether it took effect.

        Returns False when there is no listener - dictation with speech
        disabled, or with no ASR engine installed, would silently do nothing,
        and the caller needs to be able to say so.
        """
        self._dictation = active
        if self.listener is None:
            return False
        self.listener.dictation = active
        log.info("dictation %s", "on" if active else "off")
        return True

    @property
    def dictation(self) -> bool:
        return self._dictation

    def _on_dictation(self, text: str, captured_at: float) -> None:
        """Transcribed text on its way to the keyboard.

        Marked in the payload rather than given its own EventKind: it is still
        speech, and the dispatcher is the right place to decide that a speech
        event in dictation mode gets typed instead of matched.
        """
        self.emit(Event(kind=EventKind.SPEECH, value=text, captured_at=captured_at,
                        emitted_at=self._clock(), payload={"dictation": True, "raw": text}))

    def _on_wake(self, captured_at: float) -> None:
        if self.cfg.settings.arming.enabled and self.cfg.settings.arming.wake_word:
            self.emit(Event(kind=EventKind.SYSTEM, value="arm", captured_at=captured_at,
                            emitted_at=self._clock(), payload={"source": "wake_word"}))

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Open the microphone and the ASR worker. Failure here is not fatal:
        gestures must keep working on a machine with no usable input device."""
        from ..capture.microphone import Microphone, MicrophoneError  # noqa: PLC0415

        if self.listener is not None:
            self.listener.start()
        settings = self.cfg.settings.audio
        try:
            self.microphone = Microphone(
                lambda block, now: self.handle_block(block, now),
                device=settings.device,
                samplerate=settings.samplerate,
                blocksize=settings.blocksize,
                channels=settings.channels,
            ).start()
        except MicrophoneError as exc:
            log.warning("%s\naudio triggers are disabled for this session", exc)
            self.microphone = None

    def stop(self) -> None:
        if self.microphone is not None:
            self.microphone.stop()
            self.microphone = None
        if self.listener is not None:
            self.listener.stop()

    @property
    def snapshot(self) -> dict[str, float]:
        return self.detector.snapshot()
