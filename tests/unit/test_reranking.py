"""Phase 5: reranking."""

from __future__ import annotations

from pathlib import Path

import pytest

from semfuse import SemFuse
from semfuse.core.config import SemFuseConfig
from semfuse.core.exceptions import ConfigurationError
from semfuse.core.types import SearchResult
from semfuse.reranking.factory import create_reranker
from semfuse.reranking.lexical import LexicalReranker


def _r(chunk_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(text=text, score=score, chunk_id=chunk_id, document_id=chunk_id)


def test_lexical_reranker_promotes_overlap() -> None:
    reranker = LexicalReranker(alpha=0.3)
    results = [
        _r("weak", "completely unrelated content", 0.6),
        _r("strong", "the exact query words appear here", 0.5),
    ]
    reranked = reranker.rerank("exact query words", results)
    assert reranked[0].chunk_id == "strong"


def test_lexical_reranker_top_k_and_order() -> None:
    reranker = LexicalReranker()
    results = [_r(str(i), f"text {i}", 0.5) for i in range(5)]
    reranked = reranker.rerank("text 3", results, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].score >= reranked[1].score


def test_lexical_reranker_empty() -> None:
    assert LexicalReranker().rerank("query", []) == []


def test_lexical_reranker_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        LexicalReranker(alpha=1.5)


def test_factory_keys(tmp_path: Path) -> None:
    cfg = SemFuseConfig(storage_path=tmp_path)
    assert create_reranker(cfg) is None
    cfg.reranker = "lexical"
    assert isinstance(create_reranker(cfg), LexicalReranker)
    cfg.reranker = "bogus"
    with pytest.raises(ConfigurationError):
        create_reranker(cfg)


def test_search_rerank_flag_uses_lexical_fallback(db: SemFuse) -> None:
    db.add("the exact query words appear here", document_id="strong")
    db.add("completely unrelated content", document_id="weak")
    results = db.search("exact query words", rerank=True, top_k=2)
    assert results
    assert results[0].document_id == "strong"


def test_configured_reranker_applied_by_default(tmp_storage: Path) -> None:
    db = SemFuse(
        config=SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=tmp_storage,
            reranker="lexical",
        )
    )
    db.add("alpha beta gamma", document_id="abc")
    db.add("delta epsilon zeta", document_id="dez")
    results = db.search("alpha beta")
    assert results[0].document_id == "abc"
    # rerank=False disables the configured reranker for one call.
    assert db.search("alpha beta", rerank=False)


def test_cross_encoder_requires_model_only_on_use() -> None:
    from semfuse.reranking.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="definitely/not-a-real-model")
    assert reranker.rerank("q", []) == []  # no model load for empty input
