"""Microphone abstraction over ``sounddevice.InputStream``.

sounddevice already runs the callback on its own high-priority thread, so
Airwave adds no thread here - it adds a guarantee: whatever the callback is
given must not block. Any handler that raises is caught and counted rather
than allowed to kill the stream, because a dead audio stream is silent in
every sense.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger("airwave.capture.mic")

BlockHandler = Callable[[np.ndarray, float], None]


class MicrophoneError(RuntimeError):
    pass


@dataclass(slots=True)
class MicrophoneStats:
    blocks: int = 0
    overflows: int = 0
    handler_errors: int = 0


class Microphone:
    """Mono float32 blocks delivered to a handler on the audio thread."""

    def __init__(
        self,
        handler: BlockHandler,
        *,
        device: int | str | None = None,
        samplerate: int = 16000,
        blocksize: int = 512,
        channels: int = 1,
    ) -> None:
        self.handler = handler
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.stats = MicrophoneStats()
        self._stream = None
        self._lock = threading.Lock()

    def start(self) -> "Microphone":
        import sounddevice as sd  # noqa: PLC0415

        try:
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise MicrophoneError(
                f"could not open microphone {self.device if self.device is not None else '(default)'}: {exc}\n"
                f"  - list inputs with: python -m airwave devices\n"
                f"  - on macOS, grant Microphone permission to your terminal"
            ) from exc
        return self

    def _callback(self, indata, frames, time_info, status) -> None:
        """Real-time thread. Slice, hand off, return. Nothing else."""
        if status:
            self.stats.overflows += 1
        self.stats.blocks += 1
        try:
            import time as _time  # noqa: PLC0415

            self.handler(np.asarray(indata[:, 0], dtype=np.float32), _time.monotonic())
        except Exception:  # noqa: BLE001 - never let an exception reach PortAudio
            self.stats.handler_errors += 1
            if self.stats.handler_errors <= 3:
                log.exception("audio handler raised")

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                finally:
                    self._stream = None

    def __enter__(self) -> "Microphone":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._stream is not None and self._stream.active


def list_microphones() -> list[dict]:
    """Input devices, for ``airwave devices`` and for config error messages."""
    import sounddevice as sd  # noqa: PLC0415

    out = []
    try:
        default_in = sd.default.device[0]
    except Exception:  # pragma: no cover - depends on host audio config
        default_in = None
    for index, device in enumerate(sd.query_devices()):
        if device.get("max_input_channels", 0) > 0:
            out.append({
                "index": index,
                "name": device.get("name", "?"),
                "channels": device.get("max_input_channels"),
                "samplerate": int(device.get("default_samplerate", 0)),
                "default": index == default_in,
            })
    return out
