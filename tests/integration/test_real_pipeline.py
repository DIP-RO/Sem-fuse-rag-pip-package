"""Integration tests for the full pipeline with the real embedding backend.

Skipped automatically when the model is not available offline (same policy as
test_real_embeddings.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from semfuse import SemFuse
from semfuse.core.config import SemFuseConfig
from semfuse.evaluation import RetrievalEvaluator, banglish_benchmark

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _can_load_model(model_name: str) -> bool:
    try:
        import sentence_transformers

        sentence_transformers.SentenceTransformer(model_name, local_files_only=True)
        return True
    except Exception:  # pragma: no cover - environment-dependent
        return False


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


def test_hybrid_banglish_to_bangla(real_db: SemFuse, _model_available: bool) -> None:
    if not _model_available:
        pytest.skip("sentence-transformers model not available offline")
    real_db.add("ঢাকা বাংলাদেশের রাজধানী।")
    real_db.add("The Eiffel Tower is in Paris.")
    real_db.add("Tokyo is the capital of Japan.")
    results = real_db.search("Bangladesh er rajdhani kothay?", mode="hybrid")
    assert results
    assert "ঢাকা" in results[0].text


def test_banglish_benchmark_real_model(real_db: SemFuse, _model_available: bool) -> None:
    if not _model_available:
        pytest.skip("sentence-transformers model not available offline")
    docs, samples = banglish_benchmark()
    for doc_id, text in docs:
        real_db.add(text, document_id=doc_id)
    report = RetrievalEvaluator(real_db).evaluate(samples, k_values=(1, 3))
    assert report.metrics["hit@3"] >= 0.8
    assert report.metrics["mrr"] >= 0.5


def test_ask_with_real_model(real_db: SemFuse, _model_available: bool) -> None:
    if not _model_available:
        pytest.skip("sentence-transformers model not available offline")
    real_db.add("ঢাকা বাংলাদেশের রাজধানী।")
    real_db.add("The Eiffel Tower is in Paris.")
    response = real_db.ask("What is the capital of Bangladesh?")
    assert response.citations
    assert "ঢাকা" in response.answer
