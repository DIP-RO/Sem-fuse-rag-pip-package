"""Language detection.

Phase 1 implements a lightweight heuristic detector covering English, Bangla,
and Unknown. Banglish / Mixed detection is expanded in Phase 2, but the public
``detect_language`` interface is stable from the start.
"""

from __future__ import annotations

import re

from semfuse.core.enums import Language

_BANGLA_RANGE = re.compile(r"[\u0980-\u09FF]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")


def detect_language(text: str) -> Language:
    """Detect the language category of ``text``.

    Heuristic (Phase 1):
      * contains Bangla script chars  -> ``bn`` (unless substantial Latin too -> mixed)
      * only Latin script              -> ``en`` (Banglish refinement in Phase 2)
      * empty / no recognizable script -> ``unknown``
    """
    if not text or not text.strip():
        return Language.UNKNOWN
    has_bangla = bool(_BANGLA_RANGE.search(text))
    latin_words = _LATIN_WORD.findall(text)
    has_latin = len(latin_words) > 0
    if has_bangla and has_latin:
        # Refined mixed/banglish classification arrives in Phase 2.
        return Language.MIXED
    if has_bangla:
        return Language.BN
    if has_latin:
        return Language.EN
    return Language.UNKNOWN
