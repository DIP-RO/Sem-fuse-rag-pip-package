"""Language layer protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from semfuse.core.enums import Language


@runtime_checkable
class LanguageDetector(Protocol):
    """Interface for language detection strategies."""

    def detect(self, text: str) -> Language:
        """Classify ``text`` into a :class:`Language` category."""
        ...


@runtime_checkable
class TextNormalizer(Protocol):
    """Interface for non-destructive text normalization strategies."""

    def normalize(self, text: str) -> str:
        """Return a normalized copy of ``text`` (never mutates the input)."""
        ...
