"""Vision pipeline: landmarks -> normalize -> classify -> stabilize.

``landmarks`` and ``pipeline`` are *not* imported here: they pull in MediaPipe
and OpenCV, and the config layer and test suite must stay importable on a
machine with neither.
"""

from .classifier import Classification, Classifier
from .normalize import feature_vector, normalize, to_array
from .rules import RuleClassifier
from .stabilizer import Stabilizer, Transition

__all__ = [
    "Classification",
    "Classifier",
    "RuleClassifier",
    "Stabilizer",
    "Transition",
    "feature_vector",
    "normalize",
    "to_array",
]
