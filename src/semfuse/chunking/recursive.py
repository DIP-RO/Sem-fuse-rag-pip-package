"""Recursive character chunker.

Splits text along a hierarchy of separators — paragraphs first, then lines,
then sentences (including the Bangla dari ``।``), then words — merging pieces
back together up to ``chunk_size`` characters with ``chunk_overlap`` characters
of trailing context carried into the next chunk. Deterministic: the same input
always produces the same chunks.
"""

from __future__ import annotations

from semfuse.core.exceptions import ConfigurationError

# Ordered from coarsest to finest. Sentence-ending separators keep the
# terminator attached to the preceding piece so chunks end naturally.
_DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", "।", ".", "?", "!", ";", " ")


class RecursiveCharacterChunker:
    """Splits text recursively by separators, respecting size and overlap."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: tuple[str, ...] = _DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ConfigurationError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ConfigurationError("chunk_overlap must be in [0, chunk_size)")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self._chunk_size:
            return [text]
        pieces = self._split_recursive(text, 0)
        return self._merge(pieces)

    # ------------------------------------------------------------------ internal
    def _split_recursive(self, text: str, sep_index: int) -> list[str]:
        """Break ``text`` into pieces no longer than chunk_size."""
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []
        if sep_index >= len(self._separators):
            # No separator left: hard-cut at chunk_size boundaries.
            step = self._chunk_size
            return [text[i : i + step] for i in range(0, len(text), step)]
        sep = self._separators[sep_index]
        parts = self._split_keep_sep(text, sep)
        if len(parts) == 1:
            return self._split_recursive(text, sep_index + 1)
        pieces: list[str] = []
        for part in parts:
            if not part.strip():
                continue
            if len(part) <= self._chunk_size:
                pieces.append(part)
            else:
                pieces.extend(self._split_recursive(part, sep_index + 1))
        return pieces

    @staticmethod
    def _split_keep_sep(text: str, sep: str) -> list[str]:
        """Split on ``sep``, keeping sentence terminators attached."""
        if sep in ("\n\n", "\n", " "):
            return [p for p in text.split(sep) if p]
        raw = text.split(sep)
        if len(raw) == 1:
            return raw
        parts = [p + sep for p in raw[:-1]]
        if raw[-1]:
            parts.append(raw[-1])
        return parts

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedily pack pieces into chunks, carrying overlap between them."""
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = (current + " " + piece).strip() if current else piece.strip()
            if current and len(candidate) > self._chunk_size:
                chunks.append(current)
                tail = current[-self._chunk_overlap :] if self._chunk_overlap else ""
                # Start the next chunk at a word boundary within the overlap tail.
                if tail and " " in tail:
                    tail = tail[tail.index(" ") + 1 :]
                current = (tail + " " + piece).strip() if tail else piece.strip()
                # A pathological piece can still exceed chunk_size; keep it whole
                # rather than dropping content.
            else:
                current = candidate
        if current:
            chunks.append(current)
        return [c for c in chunks if c.strip()]
