"""semfuse.language subpackage."""

from __future__ import annotations

from semfuse.language.banglish import BanglishNormalizer, banglish_marker_count, looks_banglish
from semfuse.language.base import LanguageDetector, TextNormalizer
from semfuse.language.detector import HeuristicLanguageDetector, detect_language
from semfuse.language.normalizer import normalize_for_search, normalize_text

__all__ = [
    "BanglishNormalizer",
    "HeuristicLanguageDetector",
    "LanguageDetector",
    "TextNormalizer",
    "banglish_marker_count",
    "detect_language",
    "looks_banglish",
    "normalize_for_search",
    "normalize_text",
]
