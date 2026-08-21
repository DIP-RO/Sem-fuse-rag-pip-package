"""Deterministic lexical reranker.

Blends the retriever's score with a Dice token-overlap coefficient between the
query and each candidate. Offline and dependency-free, so it works everywhere
(including unit tests) and provides a modest precision bump for keyword-ish
queries without a model download.
"""

from __future__ import annotations

from semfuse.core.types import SearchResult
from semfuse.retrieval.keyword import tokenize


class LexicalReranker:
    """Reranks by ``alpha * original_score + (1 - alpha) * dice_overlap``."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self._alpha = alpha

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        query_tokens = set(tokenize(query))
        rescored: list[SearchResult] = []
        for result in results:
            doc_tokens = set(tokenize(result.text))
            if query_tokens and doc_tokens:
                overlap = 2.0 * len(query_tokens & doc_tokens) / (len(query_tokens) + len(doc_tokens))
            else:
                overlap = 0.0
            score = self._alpha * max(result.score, 0.0) + (1.0 - self._alpha) * overlap
            rescored.append(
                SearchResult(
                    text=result.text,
                    score=score,
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    metadata=result.metadata,
                    language=result.language,
                    source=result.source,
                    page=result.page,
                )
            )
        rescored.sort(key=lambda r: -r.score)
        return rescored[:top_k] if top_k is not None else rescored
