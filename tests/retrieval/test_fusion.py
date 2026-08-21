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
    a = [_r("x", 0.8), _r("y", 0.4)]
    b = [_r("y", 1.0), _r("z", 0.5)]
    fused = fuse_results([a, b], weights=[0.5, 0.5], method=FusionMethod.WEIGHTED, top_k=3)
    scores = {r.chunk_id: r.score for r in fused}
    assert scores["y"] == pytest.approx(0.5 * 0.4 + 0.5 * 1.0)
    assert scores["x"] == pytest.approx(0.4)
    assert scores["z"] == pytest.approx(0.25)
    assert fused[0].chunk_id == "y"


def test_weighted_fusion_clamps_negative_scores() -> None:
    fused = fuse_results([[_r("x", -0.5)]], method=FusionMethod.WEIGHTED, top_k=1)
    assert fused[0].score == 0.0


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
