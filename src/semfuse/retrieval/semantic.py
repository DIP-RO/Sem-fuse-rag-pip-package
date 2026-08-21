"""Semantic retriever: embed query -> vector store search."""

from __future__ import annotations

from semfuse.core.exceptions import RetrievalError
from semfuse.core.types import SearchResult
from semfuse.embeddings.base import EmbeddingProvider
from semfuse.vectorstores.base import VectorStore


class SemanticRetriever:
    """Retrieves chunks by embedding similarity."""

    def __init__(self, embeddings: EmbeddingProvider, store: VectorStore) -> None:
        self._embeddings = embeddings
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        try:
            query_vec = self._embeddings.embed_query(query)
        except Exception as exc:
            if isinstance(exc, RetrievalError):
                raise
            raise RetrievalError(f"Failed to embed query: {exc}") from exc
        return self._store.search(query_vec, top_k=top_k, filter=filter)
