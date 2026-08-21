"""Score fusion for hybrid retrieval.

Two methods, keyed by :class:`FusionMethod`:

* ``rrf`` — reciprocal rank fusion: robust to incomparable score scales,
  rewards agreement between retrievers. Fused scores are rescaled to [0, 1].
* ``weighted`` — weighted sum of per-retriever scores.  Each retriever's
  scores are min-max normalized to [0, 1] *before* weighting, so retrievers
  with inherently different score ranges (e.g. cosine 0.5–1.0 vs BM25 0–1.0)
  contribute proportionally rather than one dominating the other.  A chunk
  missing from one list contributes 0 from that retriever.

Results are merged by ``chunk_id``; the representative ``SearchResult`` for a
chunk comes from the first list that contains it.
"""

from __future__ import annotations

from collections.abc import Sequence

from semfuse.core.enums import FusionMethod
from semfuse.core.exceptions import RetrievalError
from semfuse.core.types import SearchResult

_RRF_K = 60.0


def _with_score(result: SearchResult, score: float) -> SearchResult:
    return SearchResult(
        text=result.text,
        score=score,
        document_id=result.document_id,
        chunk_id=result.chunk_id,
        metadata=result.metadata,
        language=result.language,
        source=result.source,
        page=result.page,
    )


def _minmax_normalize(results: list[SearchResult]) -> list[tuple[SearchResult, float]]:
    """Return (result, normalized_score) pairs with scores scaled to [0, 1]."""
    if not results:
        return []
    raw_scores = [max(r.score, 0.0) for r in results]
    lo = min(raw_scores)
    hi = max(raw_scores)
    if hi - lo < 1e-12:
        # All scores equal — give them all 1.0 so they contribute fully.
        return [(r, 1.0) for r in results]
    span = hi - lo
    return [(r, (s - lo) / span) for r, s in zip(results, raw_scores, strict=True)]


def fuse_results(
    result_lists: Sequence[list[SearchResult]],
    weights: Sequence[float] | None = None,
    method: FusionMethod = FusionMethod.RRF,
    top_k: int = 5,
) -> list[SearchResult]:
    """Fuse ranked result lists into one list of at most ``top_k`` results."""
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise RetrievalError("fusion weights length must match result_lists length")

    fused: dict[str, float] = {}
    representative: dict[str, SearchResult] = {}

    for results, weight in zip(result_lists, weights, strict=True):
        if not results:
            continue
        if method == FusionMethod.RRF:
            for rank, result in enumerate(results):
                key = result.chunk_id or result.text
                representative.setdefault(key, result)
                contribution = weight / (_RRF_K + rank + 1.0)
                fused[key] = fused.get(key, 0.0) + contribution
        elif method == FusionMethod.WEIGHTED:
            # Min-max normalize this retriever's scores before weighting.
            normalized = _minmax_normalize(results)
            for result, norm_score in normalized:
                key = result.chunk_id or result.text
                representative.setdefault(key, result)
                contribution = weight * norm_score
                fused[key] = fused.get(key, 0.0) + contribution
        else:
            raise RetrievalError(f"Unsupported fusion method: {method}")

    if not fused:
        return []
    max_score = max(fused.values())
    # RRF scores are rescaled to [0, 1]; weighted scores are already bounded.
    scale = 1.0 / max_score if method == FusionMethod.RRF and max_score > 0 else 1.0
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return [_with_score(representative[key], score * scale) for key, score in ranked]
