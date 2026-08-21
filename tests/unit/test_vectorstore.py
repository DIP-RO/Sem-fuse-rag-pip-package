"""LocalVectorStore unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from semfuse.core.enums import Language, SimilarityMetric
from semfuse.core.exceptions import IndexVersionError
from semfuse.core.types import DocumentChunk
from semfuse.vectorstores.local import LocalVectorStore


def _chunk(text: str, cid: str, chash: str = "") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        document_id="doc1",
        text=text,
        normalized_text=text,
        language=Language.EN,
        content_hash=chash or f"hash-{cid}",
        metadata={"department": "CSE"},
    )


def test_add_and_count(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path, "m", 8, SimilarityMetric.COSINE)
    store.add(_chunk("a", "c1"), np.ones(8, dtype=np.float32))
    store.add(_chunk("b", "c2"), np.ones(8, dtype=np.float32))
    assert store.count() == 2


def test_dedup_by_content_hash(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path, "m", 8)
    c = _chunk("a", "c1", chash="same")
    assert store.add(c, np.ones(8, dtype=np.float32)) is True
    assert store.add(c, np.ones(8, dtype=np.float32)) is False
    assert store.count() == 1


def test_search_returns_sorted_by_score(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path, "m", 4, SimilarityMetric.COSINE)
    store.add(_chunk("cat", "c1"), np.array([1, 0, 0, 0], dtype=np.float32))
    store.add(_chunk("dog", "c2"), np.array([0, 1, 0, 0], dtype=np.float32))
    store.add(_chunk("car", "c3"), np.array([1, 1, 0, 0], dtype=np.float32))
    results = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=3)
    assert len(results) == 3
    assert results[0].score >= results[1].score >= results[2].score


def test_search_with_filter(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path, "m", 4)
    store.add(_chunk("a", "c1"), np.zeros(4, dtype=np.float32))
    # c2 has different metadata.
    c2 = DocumentChunk(
        chunk_id="c2", document_id="d", text="b", normalized_text="b",
        language=Language.EN, content_hash="h2", metadata={"department": "EEE"},
    )
    store.add(c2, np.zeros(4, dtype=np.float32))
    results = store.search(np.zeros(4, dtype=np.float32), top_k=10, filter={"department": "EEE"})
    assert len(results) == 1
    assert results[0].chunk_id == "c2"


def test_persist_and_load(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path, "m", 4)
    store.add(_chunk("a", "c1"), np.array([1, 0, 0, 0], dtype=np.float32))
    store.persist()

    store2 = LocalVectorStore(tmp_path, "m", 4)
    store2.load()
    assert store2.count() == 1
    results = store2.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=1)
    assert results[0].text == "a"


def test_load_mismatch_raises(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path, "m", 4)
    store.add(_chunk("a", "c1"), np.ones(4, dtype=np.float32))
    store.persist()
    wrong = LocalVectorStore(tmp_path, "m", 8)
    import pytest

    with pytest.raises(IndexVersionError):
        wrong.load()


def test_delete(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path, "m", 4)
    store.add(_chunk("a", "c1"), np.ones(4, dtype=np.float32))
    store.add(_chunk("b", "c2"), np.ones(4, dtype=np.float32))
    store.delete("c1")
    assert store.count() == 1
    assert store._chunks[0].chunk_id == "c2"


def test_clear(tmp_path: Path) -> None:
    store = LocalVectorStore(tmp_path, "m", 4)
    store.add(_chunk("a", "c1"), np.ones(4, dtype=np.float32))
    store.clear()
    assert store.count() == 0
