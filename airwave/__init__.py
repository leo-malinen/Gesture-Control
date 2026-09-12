"""Airwave — local computer vision gesture & sound control.

The package is import-light on purpose: nothing here pulls in OpenCV,
MediaPipe, sounddevice or pyautogui. Heavy dependencies are imported inside
the components that need them, so config validation, the stabilizer, the
dispatcher and the whole test suite run on a headless machine.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
