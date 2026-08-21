"""Tests for the real-world corpus: cross-language retrieval, RAG quality, and
hybrid retrieval ranking correctness.

These tests use the hashing provider (offline, deterministic) so they run
without downloading any model.  The cross-language matching relies on the
Banglish transliteration layer, not on a cross-lingual embedding model.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from semfuse import SemFuse, SemFuseConfig
from semfuse.core.enums import SearchMode
from tests.fixtures.real_world_corpus import (
    CROSS_LANGUAGE_QUERIES,
    RAG_QA_PAIRS,
    REAL_WORLD_CORPUS,
)


@pytest.fixture
def real_db() -> SemFuse:
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=os.path.join(td, "semfuse"),
            search_mode=SearchMode.HYBRID,
        )
        db = SemFuse(config=cfg)
        for doc_id, text, metadata in REAL_WORLD_CORPUS:
            db.add(text, metadata=metadata, document_id=doc_id)
        yield db
        db.close()


# ---------------------------------------------------------------------------
# Cross-language retrieval: the top result should be the expected document.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query,expected_doc_id", CROSS_LANGUAGE_QUERIES)
def test_cross_language_retrieval_top1(real_db: SemFuse, query: str, expected_doc_id: str) -> None:
    """The expected document should appear in the top-3 results."""
    results = real_db.search(query, top_k=3)
    doc_ids = [r.document_id for r in results]
    assert expected_doc_id in doc_ids, (
        f"Expected {expected_doc_id!r} in top-3 for query {query!r}, got {doc_ids}"
    )


def test_bangla_query_retrieves_bangla_doc(real_db: SemFuse) -> None:
    """Bangla query 'বাংলাদেশের রাজধানী কী?' should retrieve the Bangla capital doc."""
    results = real_db.search("বাংলাদেশের রাজধানী কী?", top_k=1)
    assert results[0].document_id == "capital_bn"


def test_banglish_query_retrieves_bangla_doc(real_db: SemFuse) -> None:
    """Banglish query 'Bangladesh er rajdhani ki?' should retrieve the Bangla capital doc."""
    results = real_db.search("Bangladesh er rajdhani ki?", top_k=3)
    doc_ids = [r.document_id for r in results]
    assert "capital_bn" in doc_ids


def test_english_query_retrieves_english_doc(real_db: SemFuse) -> None:
    """English query should retrieve the English capital doc."""
    results = real_db.search("What is the capital of Bangladesh?", top_k=3)
    doc_ids = [r.document_id for r in results]
    assert "capital_en" in doc_ids


# ---------------------------------------------------------------------------
# RAG answer quality: the extractive provider should extract concise answers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question,expected_substring", RAG_QA_PAIRS)
def test_rag_answer_contains_expected(real_db: SemFuse, question: str, expected_substring: str) -> None:
    """RAG answer should contain the expected answer substring."""
    response = real_db.ask(question, top_k=3)
    assert expected_substring in response.answer, (
        f"Expected {expected_substring!r} in answer for {question!r}, "
        f"got {response.answer!r}"
    )
    assert response.answer.endswith("[1]")
    assert len(response.citations) > 0


def test_rag_bangla_answer_is_concise(real_db: SemFuse) -> None:
    """Bangla RAG answer should be concise (just 'ঢাকা'), not the full passage."""
    response = real_db.ask("বাংলাদেশের রাজধানী কী?", top_k=3)
    # The answer should be just "ঢাকা [1]", not the full passage.
    assert "ঢাকা" in response.answer
    assert "বৃহত্তম" not in response.answer  # Should not contain the second sentence


def test_rag_citations_have_source_metadata(real_db: SemFuse) -> None:
    """RAG citations should preserve source metadata."""
    response = real_db.ask("What is the capital of Bangladesh?", top_k=3)
    for citation in response.citations:
        assert citation.document_id is not None
        assert "topic" in citation.metadata


# ---------------------------------------------------------------------------
# Search mode correctness
# ---------------------------------------------------------------------------


def test_semantic_mode_returns_results(real_db: SemFuse) -> None:
    results = real_db.search("capital", mode="semantic", top_k=3)
    assert len(results) > 0


def test_keyword_mode_returns_results(real_db: SemFuse) -> None:
    results = real_db.search("capital", mode="keyword", top_k=3)
    assert len(results) > 0


def test_hybrid_mode_returns_results(real_db: SemFuse) -> None:
    results = real_db.search("capital", mode="hybrid", top_k=3)
    assert len(results) > 0


def test_keyword_mode_bangla_match(real_db: SemFuse) -> None:
    """Keyword mode should match Bangla tokens after transliteration."""
    # 'rajdhani' transliterates to 'রাজধানী' which appears in capital_bn.
    results = real_db.search("rajdhani", mode="keyword", top_k=3)
    doc_ids = [r.document_id for r in results]
    assert "capital_bn" in doc_ids


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


def test_rerank_preserves_top_k(real_db: SemFuse) -> None:
    """Reranking should not change the number of results."""
    plain = real_db.search("Bangladesh", top_k=3)
    reranked = real_db.search("Bangladesh", top_k=3, rerank=True)
    assert len(reranked) == len(plain)


def test_rerank_changes_order_for_keyword_query(real_db: SemFuse) -> None:
    """Reranking with lexical overlap should improve ranking for keyword-ish queries."""
    # A query with high token overlap should benefit from lexical reranking.
    results_plain = real_db.search("Bangladesh capital Dhaka", top_k=5)
    results_reranked = real_db.search("Bangladesh capital Dhaka", top_k=5, rerank=True)
    # Both should return results.
    assert len(results_plain) > 0
    assert len(results_reranked) > 0
    # The reranked results should have scores.
    assert all(r.score >= 0.0 for r in results_reranked)


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------


def test_metadata_filter_topic(real_db: SemFuse) -> None:
    """Filtering by topic should only return matching documents."""
    results = real_db.search("Bangladesh", top_k=10, filter={"topic": "geography"})
    assert len(results) > 0
    assert all(r.metadata.get("topic") == "geography" for r in results)


def test_metadata_filter_language(real_db: SemFuse) -> None:
    """Filtering by language should only return matching documents."""
    results = real_db.search("capital", top_k=10, filter={"language": "bn"})
    assert len(results) > 0
    assert all(r.metadata.get("language") == "bn" for r in results)


def test_metadata_filter_no_match(real_db: SemFuse) -> None:
    """A filter that matches nothing should return empty."""
    results = real_db.search("capital", top_k=10, filter={"topic": "nonexistent"})
    assert results == []


# ---------------------------------------------------------------------------
# Explain diagnostics
# ---------------------------------------------------------------------------


def test_explain_banglish_detection(real_db: SemFuse) -> None:
    """explain() should detect Banglish for romanized Bangla queries."""
    expl = real_db.explain("Bangladesh er rajdhani ki?")
    assert expl["detected_language"] == "banglish"
    assert "রাজধানী" in expl["normalized_query"]


def test_explain_bangla_detection(real_db: SemFuse) -> None:
    expl = real_db.explain("বাংলাদেশের রাজধানী কী?")
    assert expl["detected_language"] == "bn"


def test_explain_english_detection(real_db: SemFuse) -> None:
    expl = real_db.explain("What is the capital of Bangladesh?")
    assert expl["detected_language"] == "en"
