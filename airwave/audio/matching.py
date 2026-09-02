"""Fuzzy phrase matching for speech triggers (FR-3.5).

Small, local ASR models mishear short commands constantly: "volume up" comes
back as "volume app", "next track" as "next rack". Exact matching would make
voice control feel broken for a reason the user cannot see, so the config
declares phrases and this decides how close is close enough.

Two scores are combined because they fail differently: character similarity
catches "rack"/"track", while token overlap catches dropped filler words that
wreck the character ratio ("uh switch window").
"""

from __future__ import annotations

import difflib
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]+")
_SPACE = re.compile(r"\s+")

#: ASR output is spoken-form; config is usually written-form.
_NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

_FILLER = frozenset({"uh", "um", "er", "the", "a", "please", "hey"})


def normalize_phrase(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text.lower())
    return _SPACE.sub(" ", text).strip()


def _tokens(text: str) -> list[str]:
    return [_NUMBERS.get(t, t) for t in normalize_phrase(text).split() if t not in _FILLER]


def similarity(a: str, b: str) -> float:
    """0..1 similarity between a heard phrase and a configured one."""
    na, nb = normalize_phrase(a), normalize_phrase(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    char_score = difflib.SequenceMatcher(None, na, nb).ratio()

    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return char_score
    matched = 0.0
    remaining = list(ta)
    for token in tb:
        best, best_i = 0.0, -1
        for i, candidate in enumerate(remaining):
            score = 1.0 if candidate == token else difflib.SequenceMatcher(None, candidate, token).ratio()
            if score > best:
                best, best_i = score, i
        # A token only counts as present if it is genuinely close; without the
        # floor, two unrelated words half-match and every phrase matches.
        if best >= 0.7 and best_i >= 0:
            matched += best
            remaining.pop(best_i)
    token_score = matched / len(tb)

    # A heard phrase that contains the whole command plus extra words is a
    # match; the reverse (half the command) is not.
    containment = 1.0 if normalize_phrase(b) in normalize_phrase(a) else 0.0
    return max(char_score, token_score, containment)


def best_match(heard: str, phrases, *, threshold: float = 0.72) -> tuple[str, float] | None:
    """Return the closest configured phrase and its score, or ``None``."""
    best: tuple[str, float] | None = None
    for phrase in phrases:
        score = similarity(heard, phrase)
        if score >= threshold and (best is None or score > best[1]):
            best = (phrase, score)
    return best


def contains_wake_word(heard: str, wake_word: str, *, threshold: float = 0.8) -> bool:
    """Wake-word check over a sliding window of the transcript.

    The wake word can land anywhere in a block of speech, so comparing whole
    strings is wrong - "hey airwave switch window" barely resembles "hey
    airwave" as a whole, but contains it exactly.
    """
    heard_n = normalize_phrase(heard)
    wake_n = normalize_phrase(wake_word)
    if not heard_n or not wake_n:
        return False
    if wake_n in heard_n:
        return True
    words = heard_n.split()
    span = len(wake_n.split())
    for i in range(len(words) - span + 1):
        window = " ".join(words[i : i + span])
        if difflib.SequenceMatcher(None, window, wake_n).ratio() >= threshold:
            return True
    return False
