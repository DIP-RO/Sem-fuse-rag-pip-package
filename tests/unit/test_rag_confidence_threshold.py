"""Tests for RAG confidence threshold (refusal on weak retrieval matches)."""

from __future__ import annotations

import tempfile

from semfuse import SemFuse, SemFuseConfig


def test_rag_refuses_when_score_below_threshold() -> None:
    """With a high confidence threshold, weak matches should refuse."""
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=td,
            rag_confidence_threshold=0.99,  # Very strict.
        )
        db = SemFuse(config=cfg)
        db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")
        db.add("পদ্মা একটি বড় নদী।", document_id="river")

        # Irrelevant query — gets a weak fuzzy match (score ~0.7).
        response = db.ask("What is the distance to Mars?")
        assert response.citations == []
        assert "could not find" in response.answer.lower()


def test_rag_answers_when_score_above_threshold() -> None:
    """With a low threshold, relevant queries should answer normally."""
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=td,
            rag_confidence_threshold=0.0,  # No threshold — always answer.
        )
        db = SemFuse(config=cfg)
        db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")

        response = db.ask("বাংলাদেশের রাজধানী কী?")
        assert len(response.citations) > 0
        assert "ঢাকা" in response.answer


def test_rag_refuses_irrelevant_query_with_moderate_threshold() -> None:
    """An irrelevant query should refuse with a moderate threshold."""
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=td,
            rag_confidence_threshold=0.75,  # Above the 0.7 weak-match score.
        )
        db = SemFuse(config=cfg)
        db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")
        db.add("পদ্মা একটি বড় নদী।", document_id="river")

        # Irrelevant query about Mars — gets score ~0.7, below 0.75.
        response = db.ask("What is the distance to Mars?")
        assert response.citations == []
        assert "could not find" in response.answer.lower()


def test_rag_relevant_query_passes_moderate_threshold() -> None:
    """A relevant query should still answer with a moderate threshold."""
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=td,
            rag_confidence_threshold=0.75,  # Moderate.
        )
        db = SemFuse(config=cfg)
        db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")

        # Relevant query — gets score 1.0, above 0.75.
        response = db.ask("বাংলাদেশের রাজধানী কী?")
        assert len(response.citations) > 0
        assert "ঢাকা" in response.answer


def test_rag_refusal_bangla_language() -> None:
    """Refusal message should be in Bangla for Bangla questions."""
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=td,
            rag_confidence_threshold=0.99,
        )
        db = SemFuse(config=cfg)
        db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")
        db.add("পদ্মা একটি বড় নদী।", document_id="river")

        # Bangla irrelevant query — should refuse in Bangla.
        response = db.ask("মঙ্গল গ্রহের দূরত্ব কত?")
        assert "পাওয়া যায়নি" in response.answer


def test_rag_refusal_english_language() -> None:
    """Refusal message should be in English for English questions."""
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=td,
            rag_confidence_threshold=0.99,
        )
        db = SemFuse(config=cfg)
        db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")
        db.add("পদ্মা একটি বড় নদী।", document_id="river")

        response = db.ask("What is the distance to Mars?")
        assert "could not find" in response.answer.lower()


def test_rag_confidence_threshold_default_is_zero() -> None:
    """Default threshold should be 0.0 (backward compatible — never refuse on score)."""
    cfg = SemFuseConfig()
    assert cfg.rag_confidence_threshold == 0.0


def test_rag_confidence_threshold_validation() -> None:
    """Threshold must be between 0.0 and 1.0."""
    import pytest

    with pytest.raises(ValueError, match="rag_confidence_threshold"):
        SemFuseConfig(rag_confidence_threshold=-0.1)
    with pytest.raises(ValueError, match="rag_confidence_threshold"):
        SemFuseConfig(rag_confidence_threshold=1.5)


def test_rag_threshold_zero_never_refuses_on_score() -> None:
    """With threshold=0.0, even weak matches should be answered (backward compat)."""
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=td,
            rag_confidence_threshold=0.0,
        )
        db = SemFuse(config=cfg)
        db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")
        db.add("পদ্মা একটি বড় নদী।", document_id="river")

        # Irrelevant query — with threshold=0, should still answer (old behavior).
        response = db.ask("What is the distance to Mars?")
        assert len(response.citations) > 0  # Old behavior: answers from weak match.
