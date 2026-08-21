"""Hybrid retriever: semantic + keyword, fused."""

from __future__ import annotations

from semfuse.core.enums import FusionMethod
from semfuse.core.types import SearchResult
from semfuse.retrieval.fusion import fuse_results
from semfuse.retrieval.keyword import KeywordRetriever
from semfuse.retrieval.semantic import SemanticRetriever


class HybridRetriever:
    """Runs semantic and keyword retrieval and fuses the two rankings."""

    def __init__(
        self,
        semantic: SemanticRetriever,
        keyword: KeywordRetriever,
        method: FusionMethod = FusionMethod.WEIGHTED,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> None:
        self._semantic = semantic
        self._keyword = keyword
        self._method = method
        self._weights = (semantic_weight, keyword_weight)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        # Over-fetch per retriever so fusion has candidates beyond the cut.
        candidate_k = max(top_k * 3, 10)
        semantic_results = self._semantic.retrieve(query, top_k=candidate_k, filter=filter)
        keyword_results = self._keyword.retrieve(query, top_k=candidate_k, filter=filter)
        return fuse_results(
            [semantic_results, keyword_results],
            weights=self._weights,
            method=self._method,
            top_k=top_k,
        )
