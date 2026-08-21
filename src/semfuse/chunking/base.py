"""Text chunker protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TextChunker(Protocol):
    """Interface for splitting a document's text into embeddable chunks."""

    def split(self, text: str) -> list[str]:
        """Split ``text`` into chunks. Never returns empty strings."""
        ...
