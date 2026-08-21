"""Persistence tests: create -> close -> reopen -> search."""

from __future__ import annotations

from pathlib import Path

from semfuse import SemFuse
from semfuse.core.config import SemFuseConfig


def _cfg(storage: Path) -> SemFuseConfig:
    return SemFuseConfig(
        embedding_provider="hashing",
        embedding_model="hashing-ngram",
        embedding_dimension=256,
        storage_path=storage,
    )


def test_persistence_roundtrip(tmp_path: Path) -> None:
    storage = tmp_path / "persist"
    db1 = SemFuse(config=_cfg(storage))
    db1.add("Dhaka is the capital of Bangladesh.")
    db1.add("ঢাকা বাংলাদেশের রাজধানী।")
    assert db1.count() == 2
    db1.close()

    # Reopen without re-indexing.
    db2 = SemFuse(config=_cfg(storage))
    assert db2.count() == 2
    results = db2.search("capital of Bangladesh")
    assert len(results) > 0
    assert any("Dhaka" in r.text or "Bangladesh" in r.text for r in results)


def test_persistence_files_created(tmp_path: Path) -> None:
    storage = tmp_path / "persist_files"
    db = SemFuse(config=_cfg(storage))
    db.add("a document about cats")
    db.close()

    coll_dir = storage / "default"
    assert (coll_dir / "vectors.npy").exists()
    assert (coll_dir / "chunks.json").exists()
    assert (coll_dir / "index_info.json").exists()


def test_persistence_empty_storage_loads_cleanly(tmp_path: Path) -> None:
    storage = tmp_path / "empty_persist"
    db = SemFuse(config=_cfg(storage))
    assert db.count() == 0
    db.close()
