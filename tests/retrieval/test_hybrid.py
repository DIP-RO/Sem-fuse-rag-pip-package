"""Phase 4: hybrid retrieval, mode routing, and collections."""

from __future__ import annotations

from pathlib import Path

import pytest

from semfuse import SemFuse
from semfuse.core.config import SemFuseConfig
from semfuse.core.enums import SearchMode


def _corpus(db: SemFuse) -> None:
    db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")
    db.add("The Eiffel Tower is in Paris.", document_id="eiffel")
    db.add("Photosynthesis converts light into energy.", document_id="bio")


def test_auto_resolves_to_hybrid(db: SemFuse) -> None:
    assert db._resolve_mode(None) == SearchMode.HYBRID
    assert db._resolve_mode("auto") == SearchMode.HYBRID
    assert db._resolve_mode("semantic") == SearchMode.SEMANTIC
    assert db._resolve_mode(SearchMode.KEYWORD) == SearchMode.KEYWORD


def test_hybrid_combines_retrievers(db: SemFuse) -> None:
    _corpus(db)
    results = db.search("Eiffel Tower Paris", mode="hybrid")
    assert results
    assert results[0].document_id == "eiffel"
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_hybrid_degrades_to_semantic_when_no_keyword_hits(db: SemFuse) -> None:
    _corpus(db)
    # No shared tokens with the corpus -> keyword contributes nothing, but
    # hybrid must still return semantic candidates.
    results = db.search("photosynthesys enrgy", mode="hybrid", top_k=2)
    assert results


def test_all_modes_return_results(db: SemFuse) -> None:
    _corpus(db)
    for mode in ("semantic", "keyword", "hybrid", "auto"):
        assert db.search("Eiffel Tower", mode=mode), f"no results for mode={mode}"


def test_hybrid_respects_filter(db: SemFuse) -> None:
    db.add("CSE admission notice.", metadata={"department": "CSE"})
    db.add("EEE admission notice.", metadata={"department": "EEE"})
    results = db.search("admission", mode="hybrid", filter={"department": "CSE"})
    assert results
    assert all(r.metadata["department"] == "CSE" for r in results)


def test_rrf_fusion_configurable(tmp_storage: Path) -> None:
    db = SemFuse(
        config=SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=tmp_storage,
        ),
        fusion_method="rrf",
    )
    _corpus(db)
    results = db.search("Eiffel Tower Paris", mode="hybrid")
    assert results
    assert results[0].document_id == "eiffel"
    assert results[0].score == pytest.approx(1.0)


def test_invalid_weights_rejected(tmp_storage: Path) -> None:
    with pytest.raises(ValueError):
        SemFuseConfig(semantic_weight=-1.0, storage_path=tmp_storage)
    with pytest.raises(ValueError):
        SemFuseConfig(semantic_weight=0.0, keyword_weight=0.0, storage_path=tmp_storage)


def test_list_collections(tmp_storage: Path) -> None:
    for name in ("alpha", "beta"):
        db = SemFuse(
            config=SemFuseConfig(
                embedding_provider="hashing",
                embedding_dimension=256,
                storage_path=tmp_storage,
                collection=name,
            )
        )
        db.add(f"document in {name}")
    db = SemFuse(
        config=SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=256,
            storage_path=tmp_storage,
            collection="alpha",
        )
    )
    assert db.list_collections() == ["alpha", "beta"]
    info = db.collection_info()
    assert info.name == "alpha"
    assert info.chunk_count == 1
