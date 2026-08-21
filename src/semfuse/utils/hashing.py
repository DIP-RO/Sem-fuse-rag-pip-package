"""Content hashing utilities for deduplication."""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """Stable SHA-256 hash of text content (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str) -> str:
    """A shorter, URL-safe-ish hash suitable for ids."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
