"""Vector store protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from semfuse.core.types import DocumentChunk, IndexInfo, SearchResult


@runtime_checkable
class VectorStore(Protocol):
    """Interface for vector storage and search backends."""

    def add(self, chunk: DocumentChunk, vector: np.ndarray) -> bool:
        """Add a single chunk + vector. Returns False if deduplicated."""
        ...

    def add_many(self, chunks: list[DocumentChunk], vectors: np.ndarray) -> int:
        """Add many chunks + vectors (shape (n, dim)). Returns count added."""
        ...

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        """Return the top-k results for a query vector, optionally filtered."""
        ...

    def delete(self, chunk_id: str) -> None:
        """Delete a chunk by id."""
        ...

    def clear(self) -> None:
        """Remove all chunks."""
        ...

    def count(self) -> int:
        """Number of stored chunks."""
        ...

    def persist(self) -> None:
        """Persist the store to disk."""
        ...

    def load(self) -> None:
        """Load the store from disk."""
        ...

    def index_info(self) -> IndexInfo:
        """Return the persisted index metadata."""
        ...
