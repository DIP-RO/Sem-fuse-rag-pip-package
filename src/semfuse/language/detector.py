"""Language detection.

A lightweight heuristic detector covering English, Bangla, Banglish (romanized
Bangla), and Mixed text. The public ``detect_language`` interface is stable;
Phase 2 added Banglish/Mixed refinement on top of the Phase 1 script heuristic.
"""

from __future__ import annotations

import re

from semfuse.core.enums import Language
from semfuse.language.banglish import looks_banglish

_BANGLA_RANGE = re.compile(r"[ঀ-৿]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def detect_language(text: str) -> Language:
    """Detect the language category of ``text``.

    Heuristic:
      * Bangla script + Latin words          -> ``mixed``
      * Bangla script only                   -> ``bn``
      * Latin only, Banglish marker words    -> ``banglish``
      * Latin only otherwise                 -> ``en``
      * empty / no recognizable script       -> ``unknown``
    """
    if not text or not text.strip():
        return Language.UNKNOWN
    has_bangla = bool(_BANGLA_RANGE.search(text))
    latin_words = _LATIN_WORD.findall(text)
    has_latin = len(latin_words) > 0
    if has_bangla and has_latin:
        return Language.MIXED
    if has_bangla:
        return Language.BN
    if has_latin:
        return Language.BANGLISH if looks_banglish(text) else Language.EN
    return Language.UNKNOWN


class HeuristicLanguageDetector:
    """Protocol-compatible wrapper around :func:`detect_language`."""

    def detect(self, text: str) -> Language:
        return detect_language(text)
