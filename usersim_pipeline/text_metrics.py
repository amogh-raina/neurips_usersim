"""Deterministic local text measurements."""

import re


_WORD_PATTERN = re.compile(r"\b\w+(?:[-'’]\w+)*\b", flags=re.UNICODE)
MIN_SCENARIO_WORDS = 200
MAX_SCENARIO_WORDS = 350


def count_words(text: str) -> int:
    """Count Unicode words, treating hyphenated words and contractions as one word."""
    return len(_WORD_PATTERN.findall(text))


def scenario_word_count_is_valid(word_count: int) -> bool:
    """Return whether a deterministic word count is inside the required inclusive range."""
    return MIN_SCENARIO_WORDS <= word_count <= MAX_SCENARIO_WORDS
