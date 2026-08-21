"""Retriever protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from semfuse.core.types import SearchResult


@runtime_checkable
class Retriever(Protocol):
    """Interface for retrieval strategies."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        ...
