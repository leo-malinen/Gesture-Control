"""Wake-word-gated speech commands (FR-3.4 - FR-3.6).

The shape of this module is dictated by two hard constraints:

**The ASR must not run continuously.** Continuous recognition burns a core,
heats the laptop, and turns every overheard sentence into a candidate command.
So an energy VAD gates it: audio is only buffered while someone is speaking,
and only completed utterances are transcribed.

**The audio callback must never block.** sounddevice calls it on a real-time
thread; a 400ms transcription there drops audio frames and corrupts the very
utterance being recognized. So ``push`` does nothing but slice and buffer, and
a worker thread does the recognition.

Engine choice stays open (vosk for latency, faster-whisper for accuracy) - the
PRD leaves it as an open question, so both are supported behind one interface
and neither is a hard dependency.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .matching import contains_wake_word, normalize_phrase

log = logging.getLogger("airwave.audio.speech")


class SpeechEngine(Protocol):
    name: str

    def transcribe(self, pcm: np.ndarray, samplerate: int) -> str:
        ...


class NullEngine:
    """Used when no ASR is installed: warns once, transcribes nothing."""

    name = "none"

    def __init__(self, reason: str = "no ASR engine installed") -> None:
        self.reason = reason
        self._warned = False

    def transcribe(self, pcm: np.ndarray, samplerate: int) -> str:
        if not self._warned:
            log.warning("speech is enabled but %s - install vosk or faster-whisper", self.reason)
            self._warned = True
        return ""


class VoskEngine:
    """Small, fast, CPU-only. ~50MB model, sub-200ms on short utterances."""

    name = "vosk"

    def __init__(self, model_path: str | None = None, samplerate: int = 16000) -> None:
        from vosk import KaldiRecognizer, Model, SetLogLevel  # noqa: PLC0415

        SetLogLevel(-1)
        self._model = Model(model_path) if model_path else Model(lang="en-us")
        self._recognizer_cls = KaldiRecognizer
        self.samplerate = samplerate

    def transcribe(self, pcm: np.ndarray, samplerate: int) -> str:
        import json  # noqa: PLC0415

        recognizer = self._recognizer_cls(self._model, samplerate)
        pcm16 = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
        recognizer.AcceptWaveform((pcm16 * 32767).astype(np.int16).tobytes())
        return str(json.loads(recognizer.FinalResult()).get("text", ""))


class FasterWhisperEngine:
    """More accurate, ~150MB download, noticeably slower on CPU."""

    name = "faster-whisper"

    def __init__(self, model_path: str | None = None) -> None:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self._model = WhisperModel(model_path or "base.en", device="cpu", compute_type="int8")

    def transcribe(self, pcm: np.ndarray, samplerate: int) -> str:
        segments, _ = self._model.transcribe(np.asarray(pcm, dtype=np.float32), language="en", beam_size=1)
        return " ".join(segment.text.strip() for segment in segments).strip()


def build_engine(engine: str, model_path: str | None = None, samplerate: int = 16000) -> SpeechEngine:
    """Never raises: a missing ASR must not stop the gesture pipeline."""
    try:
        if engine == "vosk":
            return VoskEngine(model_path, samplerate)
        if engine == "faster-whisper":
            return FasterWhisperEngine(model_path)
    except Exception as exc:  # noqa: BLE001
        return NullEngine(f"{engine} unavailable ({type(exc).__name__}: {exc})")
    return NullEngine()


@dataclass(slots=True)
class Utterance:
    pcm: np.ndarray
    started_at: float
    ended_at: float


class VoiceActivity:
    """Energy VAD with hangover.

    The hangover is what stops "next ... track" from being cut into two
    utterances: a short pause mid-phrase is normal speech, and ending the
    utterance there guarantees a mis-transcription.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.02,
        start_ms: int = 120,
        hangover_ms: int = 500,
        max_ms: int = 4000,
        samplerate: int = 16000,
    ) -> None:
        self.threshold = threshold
        self.start_ms = start_ms
        self.hangover_ms = hangover_ms
        self.max_ms = max_ms
        self.samplerate = samplerate
        self._buffer: list[np.ndarray] = []
        self._speaking = False
        self._loud_since: float | None = None
        self._quiet_since: float | None = None
        self._started_at = 0.0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def push(self, block: np.ndarray, level: float, now: float) -> Utterance | None:
        loud = level >= self.threshold
        if not self._speaking:
            if loud:
                self._loud_since = self._loud_since or now
                self._buffer.append(np.asarray(block, dtype=np.float32).reshape(-1))
                if (now - self._loud_since) * 1000.0 >= self.start_ms:
                    self._speaking = True
                    self._started_at = self._loud_since
                    self._quiet_since = None
            else:
                self._loud_since = None
                # Keep a short pre-roll so the first phoneme is not clipped.
                self._buffer = self._buffer[-4:]
            return None

        self._buffer.append(np.asarray(block, dtype=np.float32).reshape(-1))
        if loud:
            self._quiet_since = None
        else:
            self._quiet_since = self._quiet_since or now
            if (now - self._quiet_since) * 1000.0 >= self.hangover_ms:
                return self._finish(now)
        if (now - self._started_at) * 1000.0 >= self.max_ms:
            return self._finish(now)
        return None

    def _finish(self, now: float) -> Utterance:
        pcm = np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.float32)
        utterance = Utterance(pcm=pcm, started_at=self._started_at, ended_at=now)
        self.reset()
        return utterance

    def reset(self) -> None:
        self._buffer.clear()
        self._speaking = False
        self._loud_since = None
        self._quiet_since = None


class SpeechListener:
    """VAD -> worker-thread ASR -> wake-word gate -> phrase callback."""

    def __init__(
        self,
        *,
        engine: SpeechEngine,
        wake_word: str = "hey airwave",
        listen_window_s: float = 4.0,
        samplerate: int = 16000,
        vad_threshold: float = 0.02,
        on_phrase: Callable[[str, float], None] | None = None,
        on_wake: Callable[[float], None] | None = None,
    ) -> None:
        self.engine = engine
        self.wake_word = wake_word
        self.listen_window_s = listen_window_s
        self.samplerate = samplerate
        self.on_phrase = on_phrase
        self.on_wake = on_wake
        self.vad = VoiceActivity(threshold=vad_threshold, samplerate=samplerate,
                                 max_ms=int(listen_window_s * 1000))
        self._queue: queue.Queue[Utterance] = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.awake_until = 0.0
        self.last_transcript = ""

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="airwave-asr", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ----------------------------------------------------------------- input

    def push(self, block: np.ndarray, level: float, now: float | None = None) -> None:
        """Called from the audio callback. Buffers only - never transcribes."""
        now = time.monotonic() if now is None else now
        utterance = self.vad.push(block, level, now)
        if utterance is None or utterance.pcm.size == 0:
            return
        try:
            self._queue.put_nowait(utterance)
        except queue.Full:
            log.debug("ASR backlog, dropping an utterance")

    # ---------------------------------------------------------------- worker

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                utterance = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._handle(utterance)
            except Exception:  # pragma: no cover - engine-specific failures
                log.exception("transcription failed")

    def _handle(self, utterance: Utterance) -> None:
        text = self.engine.transcribe(utterance.pcm, self.samplerate).strip()
        self.last_transcript = text
        if not text:
            return
        now = time.monotonic()
        log.debug("heard %r", text)

        if now < self.awake_until:
            # Already woken by a previous utterance: this one is the command.
            self._emit(text, utterance.started_at)
            return

        if not contains_wake_word(text, self.wake_word):
            return

        remainder = self._strip_wake(text)
        if self.on_wake:
            self.on_wake(utterance.started_at)
        if remainder:
            self._emit(remainder, utterance.started_at)
        else:
            # Bare wake word: open a window and treat the next utterance as
            # the command, so "hey airwave" / "switch window" also works.
            self.awake_until = now + self.listen_window_s

    def _strip_wake(self, text: str) -> str:
        normalized = normalize_phrase(text)
        wake = normalize_phrase(self.wake_word)
        if wake and wake in normalized:
            return normalized.split(wake, 1)[1].strip()
        words = normalized.split()
        return " ".join(words[len(wake.split()):]).strip() if len(words) > len(wake.split()) else ""

    def _emit(self, phrase: str, captured_at: float) -> None:
        self.awake_until = 0.0
        if self.on_phrase:
            self.on_phrase(phrase, captured_at)
