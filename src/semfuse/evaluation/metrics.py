"""Ranking metrics for retrieval evaluation.

All metrics take ``relevant`` (the set of relevant ids for a query) and
``ranked`` (retrieved ids, best first) and return a float in [0, 1].
Binary relevance is assumed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def hit_at_k(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    """1.0 if any relevant id appears in the top ``k``, else 0.0."""
    if not relevant:
        return 0.0
    return 1.0 if any(r in relevant for r in ranked[:k]) else 0.0


def recall_at_k(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    """Fraction of relevant ids retrieved within the top ``k``."""
    if not relevant:
        return 0.0
    found = sum(1 for r in dict.fromkeys(ranked[:k]) if r in relevant)
    return found / len(relevant)


def mrr(relevant: set[str], ranked: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant id (0.0 if none retrieved)."""
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: set[str], ranked: Sequence[str], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` (binary relevance)."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, r in enumerate(dict.fromkeys(ranked[:k]), start=1)
        if r in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
