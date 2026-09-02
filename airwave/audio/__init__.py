"""Audio input: transient (clap) detection and wake-word-gated speech."""

from .matching import best_match, contains_wake_word, normalize_phrase, similarity
from .onset import ClapGrouper, Onset, OnsetDetector

__all__ = [
    "ClapGrouper",
    "Onset",
    "OnsetDetector",
    "best_match",
    "contains_wake_word",
    "normalize_phrase",
    "similarity",
]
