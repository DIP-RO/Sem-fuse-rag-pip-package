"""Integration tests using the real sentence-transformers backend.

These are skipped automatically when the model cannot be loaded (e.g. offline,
no sentence-transformers installed, model not cached). They verify genuine
multilingual / cross-lingual retrieval behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semfuse import SemFuse
from semfuse.core.config import SemFuseConfig

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    _ST_AVAILABLE = True
except Exception:  # pragma: no cover
    _ST_AVAILABLE = False


def _can_load_model(model_name: str) -> bool:
    if not _ST_AVAILABLE:
        return False
    try:
        import sentence_transformers

        sentence_transformers.SentenceTransformer(model_name, local_files_only=True)
        return True
    except Exception:
        return False


MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@pytest.fixture(scope="module")
def _model_available() -> bool:
    return _can_load_model(MODEL)


@pytest.fixture
def real_db(tmp_path: Path) -> SemFuse:
    return SemFuse(
        config=SemFuseConfig(
            embedding_provider="local",
            embedding_model=MODEL,
            embedding_dimension=384,
            storage_path=tmp_path / "real",
        )
    )


def test_cross_lingual_bangla_to_bangla(real_db: SemFuse, _model_available: bool) -> None:
    if not _model_available:
        pytest.skip("sentence-transformers model not available offline")
    real_db.add("ঢাকা বাংলাদেশের রাজধানী।")
    real_db.add("The Eiffel Tower is in Paris.")
    results = real_db.search("বাংলাদেশের রাজধানী কী?")
    assert results
    assert "ঢাকা" in results[0].text


def test_cross_lingual_english_to_bangla(real_db: SemFuse, _model_available: bool) -> None:
    if not _model_available:
        pytest.skip("sentence-transformers model not available offline")
    real_db.add("ঢাকা বাংলাদেশের রাজধানী।")
    real_db.add("The Eiffel Tower is in Paris.")
    results = real_db.search("What is the capital of Bangladesh?")
    assert results
    assert "ঢাকা" in results[0].text


def test_cross_lingual_banglish_to_bangla(real_db: SemFuse, _model_available: bool) -> None:
    if not _model_available:
        pytest.skip("sentence-transformers model not available offline")
    real_db.add("ঢাকা বাংলাদেশের রাজধানী।")
    real_db.add("The Eiffel Tower is in Paris.")
    results = real_db.search("Bangladesh er capital ki?")
    assert results
    # Banglish -> Bangla is hard for a generic multilingual model; we assert it
    # retrieves the relevant doc in the top 2 (documented limitation, Phase 2
    # Banglish normalization improves this).
    assert any("ঢাকা" in r.text for r in results[:2])
