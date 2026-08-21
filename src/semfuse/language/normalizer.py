"""Text normalization.

Non-destructive: every function returns a normalized copy, never mutates input,
and the original text is always stored alongside the normalized form
(ADR-0005). ``normalize_text`` is the generic pass; ``normalize_for_search``
adds language-aware handling — most importantly Banglish canonicalization and
transliteration so romanized-Bangla queries land near Bangla documents.
"""

from __future__ import annotations

import re
import unicodedata

from semfuse.core.enums import Language
from semfuse.language.banglish import BanglishNormalizer

_WS_RE = re.compile(r"\s+")
# Zero-width joiner/non-joiner and BOM occur frequently in Bangla web text and
# break both hashing dedup and tokenization if left in place.
_ZERO_WIDTH_RE = re.compile("[\\u200b\\u200c\\u200d\\ufeff]")

_banglish = BanglishNormalizer()


def normalize_text(text: str) -> str:
    """Normalize whitespace and Unicode form, preserving meaningful content."""
    # NFKC normalizes compatibility characters and composes combining marks.
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_for_search(text: str, language: Language = Language.UNKNOWN) -> str:
    """Language-aware normalization used for embedding and keyword matching.

    Banglish text is spelling-canonicalized and known tokens are
    transliterated to Bangla script; all other languages get the generic
    :func:`normalize_text` pass. The original text is never modified.
    """
    normalized = normalize_text(text)
    if language == Language.BANGLISH:
        normalized = _banglish.transliterate(normalized)
    return normalized
