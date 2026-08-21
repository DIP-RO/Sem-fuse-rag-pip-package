"""Reranker protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from semfuse.core.types import SearchResult


@runtime_checkable
class Reranker(Protocol):
    """Interface for reordering retrieval candidates by query relevance."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return ``results`` reordered (and rescored), truncated to ``top_k``."""
        ...
