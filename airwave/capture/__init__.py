"""Device abstraction. Imports stay lazy so `import airwave` needs no hardware."""

from .camera import Camera, CameraError, Frame, list_cameras
from .microphone import Microphone, MicrophoneError, list_microphones

__all__ = [
    "Camera",
    "CameraError",
    "Frame",
    "Microphone",
    "MicrophoneError",
    "list_cameras",
    "list_microphones",
]
