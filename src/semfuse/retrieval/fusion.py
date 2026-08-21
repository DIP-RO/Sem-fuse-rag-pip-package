"""Score fusion for hybrid retrieval.

Two methods, keyed by :class:`FusionMethod`:

* ``rrf`` — reciprocal rank fusion: robust to incomparable score scales,
  rewards agreement between retrievers. Fused scores are rescaled to [0, 1].
* ``weighted`` — weighted sum of per-retriever scores (assumed roughly in
  [0, 1]; negatives are clamped to 0). A chunk missing from one list simply
  contributes 0 from that retriever.

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
        for rank, result in enumerate(results):
            key = result.chunk_id or result.text
            representative.setdefault(key, result)
            if method == FusionMethod.RRF:
                contribution = weight / (_RRF_K + rank + 1.0)
            elif method == FusionMethod.WEIGHTED:
                contribution = weight * max(result.score, 0.0)
            else:
                raise RetrievalError(f"Unsupported fusion method: {method}")
            fused[key] = fused.get(key, 0.0) + contribution

    if not fused:
        return []
    max_score = max(fused.values())
    scale = 1.0 / max_score if method == FusionMethod.RRF and max_score > 0 else 1.0
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return [_with_score(representative[key], score * scale) for key, score in ranked]
