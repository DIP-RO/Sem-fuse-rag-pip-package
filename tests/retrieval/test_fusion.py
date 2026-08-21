"""Phase 4: score fusion."""

from __future__ import annotations

import pytest

from semfuse.core.enums import FusionMethod
from semfuse.core.exceptions import RetrievalError
from semfuse.core.types import SearchResult
from semfuse.retrieval.fusion import fuse_results


def _r(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(text=f"text-{chunk_id}", score=score, chunk_id=chunk_id, document_id=chunk_id)


def test_weighted_fusion_sums_scores() -> None:
    # With min-max normalization, each retriever's scores are scaled to [0, 1]
    # before weighting.  List a: [0.8, 0.4] -> [1.0, 0.0].  List b: [1.0, 0.5]
    # -> [1.0, 0.0].  So:
    #   x: 0.5 * 1.0 = 0.5  (only in a, normalized to max)
    #   y: 0.5 * 0.0 + 0.5 * 1.0 = 0.5  (min in a, max in b)
    #   z: 0.5 * 0.0 = 0.0  (min in b)
    a = [_r("x", 0.8), _r("y", 0.4)]
    b = [_r("y", 1.0), _r("z", 0.5)]
    fused = fuse_results([a, b], weights=[0.5, 0.5], method=FusionMethod.WEIGHTED, top_k=3)
    scores = {r.chunk_id: r.score for r in fused}
    assert scores["x"] == pytest.approx(0.5)
    assert scores["y"] == pytest.approx(0.5)
    assert scores["z"] == pytest.approx(0.0)
    # x and y tie at 0.5; z is last.
    assert scores["z"] < scores["x"]


def test_weighted_fusion_normalizes_per_retriever() -> None:
    # Verify that min-max normalization makes different-scale retrievers
    # contribute proportionally.  Retriever A has scores [0.9, 0.1] (wide
    # spread), B has [0.51, 0.49] (narrow spread).  Without normalization B's
    # scores would be dwarfed by A's; with normalization both contribute
    # equally per rank position.
    a = [_r("x", 0.9), _r("y", 0.1)]
    b = [_r("x", 0.51), _r("y", 0.49)]
    fused = fuse_results([a, b], weights=[0.5, 0.5], method=FusionMethod.WEIGHTED, top_k=2)
    scores = {r.chunk_id: r.score for r in fused}
    # Both retrievers rank x first (normalized to 1.0) and y second (0.0).
    assert scores["x"] == pytest.approx(1.0)
    assert scores["y"] == pytest.approx(0.0)


def test_weighted_fusion_clamps_negative_scores() -> None:
    # A single negative score: min-max normalization maps it to 1.0 (it's the
    # only score, so it's both min and max — all-equal case gives 1.0).
    fused = fuse_results([[_r("x", -0.5)]], method=FusionMethod.WEIGHTED, top_k=1)
    assert fused[0].score == pytest.approx(1.0)


def test_rrf_rewards_consensus() -> None:
    a = [_r("x", 0.9), _r("y", 0.8)]
    b = [_r("y", 0.7), _r("z", 0.6)]
    fused = fuse_results([a, b], method=FusionMethod.RRF, top_k=3)
    assert fused[0].chunk_id == "y"  # appears in both lists
    assert fused[0].score == pytest.approx(1.0)  # RRF rescaled to [0, 1]
    assert all(0.0 <= r.score <= 1.0 for r in fused)


def test_fusion_respects_top_k() -> None:
    a = [_r(str(i), 1.0 - i * 0.1) for i in range(10)]
    assert len(fuse_results([a], top_k=3)) == 3


def test_fusion_empty_lists() -> None:
    assert fuse_results([[], []], top_k=5) == []


def test_fusion_weight_length_mismatch() -> None:
    with pytest.raises(RetrievalError):
        fuse_results([[_r("x", 1.0)]], weights=[1.0, 2.0])
