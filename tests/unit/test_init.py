"""Unit tests for SemFuse initialization and core API."""

from __future__ import annotations

from pathlib import Path

import pytest

from semfuse import SemFuse
from semfuse.core.config import SemFuseConfig
from semfuse.core.exceptions import ConfigurationError


def test_zero_config_initialization(tmp_path: Path) -> None:
    db = SemFuse(config=SemFuseConfig(embedding_provider="hashing", storage_path=tmp_path / "s"))
    assert db.count() == 0
    info = db.info()
    assert info["package_version"]
    assert info["embedding_provider"] == "hashing"
    assert info["chunk_count"] == 0


def test_add_returns_chunk_count(db: SemFuse) -> None:
    added = db.add("Dhaka is the capital of Bangladesh.")
    assert added == 1
    assert db.count() == 1


def test_add_many(db: SemFuse) -> None:
    added = db.add_many(
        [
            "Dhaka is the capital of Bangladesh.",
            "ঢাকা বাংলাদেশের রাজধানী।",
            "The Eiffel Tower is in Paris.",
        ]
    )
    assert added == 3
    assert db.count() == 3


def test_add_empty_text_raises(db: SemFuse) -> None:
    with pytest.raises(ConfigurationError):
        db.add("   ")


def test_add_many_metadata_length_mismatch(db: SemFuse) -> None:
    with pytest.raises(ConfigurationError):
        db.add_many(["a", "b"], metadata=[{}])


def test_search_returns_results(db: SemFuse) -> None:
    db.add("Dhaka is the capital of Bangladesh.")
    db.add("The Eiffel Tower is in Paris.")
    results = db.search("capital of Bangladesh")
    assert len(results) > 0
    top = results[0]
    assert "Dhaka" in top.text or "Bangladesh" in top.text
    assert top.score > 0.0
    assert top.chunk_id is not None


def test_search_empty_query_raises(db: SemFuse) -> None:
    with pytest.raises(ConfigurationError):
        db.search("")


def test_search_top_k(db: SemFuse) -> None:
    db.add_many(
        [
            "Dhaka is the capital of Bangladesh.",
            "The Eiffel Tower is in Paris.",
            "Tokyo is the capital of Japan.",
        ]
    )
    results = db.search("capital", top_k=2)
    assert len(results) <= 2


def test_search_score_threshold(db: SemFuse) -> None:
    db.add("Dhaka is the capital of Bangladesh.")
    db.add("The Eiffel Tower is in Paris.")
    # Threshold of 1.0 should filter everything (hashing scores are < 1.0 after norm).
    results = db.search("capital", score_threshold=1.0)
    assert results == []


def test_info_has_required_fields(db: SemFuse) -> None:
    db.add("Dhaka is the capital of Bangladesh.")
    info = db.info()
    for key in (
        "package_version",
        "embedding_provider",
        "embedding_model",
        "embedding_dimension",
        "index_version",
        "vector_backend",
        "metric",
        "storage_path",
        "collection",
        "document_count",
        "chunk_count",
        "language_distribution",
    ):
        assert key in info, f"missing key: {key}"
    assert info["chunk_count"] == 1
    assert info["document_count"] == 1


def test_explain(db: SemFuse) -> None:
    db.add("Dhaka is the capital of Bangladesh.")
    expl = db.explain("capital of Bangladesh")
    assert expl["query"] == "capital of Bangladesh"
    assert "detected_language" in expl
    assert expl["candidate_count"] >= 1
    assert expl["top_score"] > 0.0


def test_clear(db: SemFuse) -> None:
    db.add("some text")
    assert db.count() == 1
    db.clear()
    assert db.count() == 0
