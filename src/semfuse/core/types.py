"""Typed document and result objects for SemFuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from semfuse.core.enums import Language


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Document:
    """A unit of content to be indexed.

    Attributes:
        text: Original, unmodified text.
        source: Origin of the document (file path, URL, "user", etc.).
        title: Optional human-readable title.
        page: Optional page number (for paginated sources).
        language: Detected language category.
        metadata: Arbitrary user-supplied metadata.
        created_at: Creation timestamp.
        document_id: Stable identifier (set by the client).
    """

    text: str
    source: str = "user"
    title: str | None = None
    page: int | None = None
    language: Language = Language.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    document_id: str | None = None


@dataclass(frozen=True)
class DocumentChunk:
    """A chunk of a document that gets embedded and stored.

    Attributes:
        chunk_id: Stable unique identifier for the chunk.
        document_id: Parent document identifier.
        text: Original chunk text.
        normalized_text: Normalized representation used for embedding/search.
        language: Detected language of the chunk.
        source: Inherited from the parent document.
        title: Inherited from the parent document.
        page: Inherited from the parent document.
        metadata: Merged document + chunk metadata.
        chunk_index: Position of this chunk within the parent document.
        content_hash: Hash of the chunk content (for dedup).
    """

    chunk_id: str
    document_id: str
    text: str
    normalized_text: str
    language: Language
    source: str = "user"
    title: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    content_hash: str = ""


@dataclass(frozen=True)
class SearchResult:
    """A single retrieval result.

    Attributes:
        text: The retrieved chunk text (original).
        score: Similarity/fusion score (higher is better).
        document_id: Parent document id.
        chunk_id: Chunk id.
        metadata: Chunk metadata.
        language: Detected language of the chunk.
        source: Source of the chunk.
        page: Page number if available.
    """

    text: str
    score: float
    document_id: str | None = None
    chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    language: Language = Language.UNKNOWN
    source: str = "user"
    page: int | None = None

    def __repr__(self) -> str:
        preview = self.text if len(self.text) <= 60 else self.text[:57] + "..."
        return f"SearchResult(score={self.score:.4f}, text={preview!r})"


@dataclass(frozen=True)
class IndexInfo:
    """Persisted index metadata used for compatibility checks."""

    embedding_model: str
    embedding_dimension: int
    embedding_version: str
    index_version: str
    metric: str
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class CollectionInfo:
    """Information about a collection."""

    name: str
    document_count: int
    chunk_count: int
    language_distribution: dict[str, int] = field(default_factory=dict)
