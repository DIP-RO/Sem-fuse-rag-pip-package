"""Text normalization (Phase 1 minimal implementation).

Non-destructive: returns a normalized copy, never mutates input. Phase 2 expands
this with Bangla-aware and Banglish-aware normalization.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace and Unicode form, preserving meaningful content."""
    if not text:
        return text
    # NFKC normalizes compatibility characters and composes combining marks.
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text).strip()
    return text
