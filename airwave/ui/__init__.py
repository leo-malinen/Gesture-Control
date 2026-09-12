"""Debug overlay. Imported lazily - it needs OpenCV and a display."""

__all__ = ["Overlay"]


def __getattr__(name):
    if name == "Overlay":
        from .overlay import Overlay

        return Overlay
    raise AttributeError(name)
