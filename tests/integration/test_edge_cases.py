"""Edge case tests: empty index, long documents, Unicode, and boundary conditions."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from semfuse import SemFuse, SemFuseConfig
from semfuse.core.enums import Language
from semfuse.core.exceptions import ConfigurationError
from semfuse.core.types import DocumentChunk
from semfuse.vectorstores.local import LocalVectorStore


@pytest.fixture
def empty_db() -> SemFuse:
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=128,
            storage_path=os.path.join(td, "semfuse"),
        )
        yield SemFuse(config=cfg)


@pytest.fixture
def db() -> SemFuse:
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=128,
            storage_path=os.path.join(td, "semfuse"),
        )
        yield SemFuse(config=cfg)


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------


def test_search_empty_index_returns_empty(empty_db: SemFuse) -> None:
    assert empty_db.search("anything") == []
    assert empty_db.search("anything", mode="semantic") == []
    assert empty_db.search("anything", mode="keyword") == []
    assert empty_db.search("anything", mode="hybrid") == []


def test_ask_empty_index_returns_no_context(empty_db: SemFuse) -> None:
    response = empty_db.ask("anything?")
    assert "could not find" in response.answer
    assert response.citations == []


def test_count_empty_index(empty_db: SemFuse) -> None:
    assert empty_db.count() == 0


def test_info_empty_index(empty_db: SemFuse) -> None:
    info = empty_db.info()
    assert info["chunk_count"] == 0
    assert info["document_count"] == 0


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_add_empty_string_raises(db: SemFuse) -> None:
    with pytest.raises(ConfigurationError):
        db.add("")
    with pytest.raises(ConfigurationError):
        db.add("   ")


def test_search_empty_string_raises(db: SemFuse) -> None:
    with pytest.raises(ConfigurationError):
        db.search("")
    with pytest.raises(ConfigurationError):
        db.search("   ")


def test_add_many_mismatched_metadata(db: SemFuse) -> None:
    with pytest.raises(ConfigurationError):
        db.add_many(["a", "b"], metadata=[{"k": "v"}])


def test_add_many_empty_list(db: SemFuse) -> None:
    assert db.add_many([]) == 0


# ---------------------------------------------------------------------------
# Unicode edge cases
# ---------------------------------------------------------------------------


def test_zero_width_joiner_stripped(db: SemFuse) -> None:
    """Zero-width characters should be stripped during normalization."""
    text = "বাং\u200cলা ভাষা"  # ZWNJ in the middle
    assert db.add(text) == 1
    results = db.search("বাংলা ভাষা")
    assert len(results) > 0


def test_bom_stripped(db: SemFuse) -> None:
    """BOM should be stripped during normalization."""
    text = "\ufeffঢাকা বাংলাদেশের রাজধানী।"
    assert db.add(text) == 1


def test_multiple_spaces_normalized(db: SemFuse) -> None:
    """Multiple spaces should be collapsed."""
    assert db.add("Dhaka   is   the   capital.") == 1
    # Same text with single spaces should dedup.
    assert db.add("Dhaka is the capital.") == 0


def test_bangla_dari_sentence_split(db: SemFuse) -> None:
    """The Bangla dari । should be treated as a sentence terminator in chunking."""
    text = "ঢাকা বাংলাদেশের রাজধানী। এটি বৃহত্তম শহর। জনসংখ্যা অনেক।"
    # With a small chunk_size, this should produce multiple chunks.
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=128,
            storage_path=os.path.join(td, "semfuse"),
            chunk_size=25,
            chunk_overlap=5,
        )
        small_db = SemFuse(config=cfg)
        added = small_db.add(text)
        assert added > 1  # Should split into multiple chunks


# ---------------------------------------------------------------------------
# Long documents
# ---------------------------------------------------------------------------


def test_long_document_chunked(db: SemFuse) -> None:
    """A long document should be split into multiple chunks."""
    # Create a document longer than the default chunk_size (500 chars).
    long_text = "ঢাকা বাংলাদেশের রাজধানী। " * 100  # ~2300 chars
    added = db.add(long_text)
    assert added > 1
    assert db.count() == added


def test_very_long_document_performance(db: SemFuse) -> None:
    """Indexing a very long document should complete in reasonable time."""
    # Use varied text so chunks don't all dedup to the same hash.
    long_text = " ".join(
        f"Sentence number {i} discusses topic {i} in Bangladesh context."
        for i in range(500)
    )
    added = db.add(long_text)
    assert added > 10
    # Search should still work.
    results = db.search("Bangladesh", top_k=5)
    assert len(results) > 0


# ---------------------------------------------------------------------------
# Vector store edge cases
# ---------------------------------------------------------------------------


def test_vector_store_growable_buffer() -> None:
    """The vector store buffer should grow beyond the initial capacity."""
    with tempfile.TemporaryDirectory() as td:
        store = LocalVectorStore(
            storage_path=os.path.join(td, "vs"),
            embedding_model="test",
            embedding_dimension=64,
        )
        # Add more than the initial capacity (64).
        dim = 64
        for i in range(200):
            chunk = DocumentChunk(
                chunk_id=f"c{i}",
                document_id=f"d{i}",
                text=f"text {i}",
                normalized_text=f"text {i}",
                language=Language.EN,
                content_hash=f"h{i}",
            )
            vec = np.random.randn(dim).astype(np.float32)
            assert store.add(chunk, vec)
        assert store.count() == 200
        # Search should work.
        q = np.random.randn(dim).astype(np.float32)
        results = store.search(q, top_k=10)
        assert len(results) == 10


def test_vector_store_delete_compacts_buffer() -> None:
    """Deleting a chunk should properly compact the buffer."""
    with tempfile.TemporaryDirectory() as td:
        store = LocalVectorStore(
            storage_path=os.path.join(td, "vs"),
            embedding_model="test",
            embedding_dimension=64,
        )
        dim = 64
        for i in range(10):
            chunk = DocumentChunk(
                chunk_id=f"c{i}",
                document_id=f"d{i}",
                text=f"text {i}",
                normalized_text=f"text {i}",
                language=Language.EN,
                content_hash=f"h{i}",
            )
            vec = np.random.randn(dim).astype(np.float32)
            store.add(chunk, vec)
        assert store.count() == 10
        store.delete("c5")
        assert store.count() == 9
        # The remaining chunks should still be searchable.
        q = np.random.randn(dim).astype(np.float32)
        results = store.search(q, top_k=10)
        assert len(results) == 9
        # The deleted chunk should not appear.
        assert all(r.chunk_id != "c5" for r in results)


def test_vector_store_persist_and_reload() -> None:
    """Persisted data should survive a reload."""
    with tempfile.TemporaryDirectory() as td:
        store1 = LocalVectorStore(
            storage_path=os.path.join(td, "vs"),
            embedding_model="test",
            embedding_dimension=64,
        )
        dim = 64
        for i in range(50):
            chunk = DocumentChunk(
                chunk_id=f"c{i}",
                document_id=f"d{i}",
                text=f"text {i}",
                normalized_text=f"text {i}",
                language=Language.EN,
                content_hash=f"h{i}",
            )
            vec = np.random.randn(dim).astype(np.float32)
            store1.add(chunk, vec)
        store1.persist()

        store2 = LocalVectorStore(
            storage_path=os.path.join(td, "vs"),
            embedding_model="test",
            embedding_dimension=64,
        )
        store2.load()
        assert store2.count() == 50
        # Search should return the same results.
        q = np.zeros(dim, dtype=np.float32)
        q[0] = 1.0
        r1 = store1.search(q, top_k=5)
        r2 = store2.search(q, top_k=5)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]


# ---------------------------------------------------------------------------
# Persistence edge cases
# ---------------------------------------------------------------------------


def test_persist_and_reopen(db: SemFuse) -> None:
    """Data should survive close and reopen."""
    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(
            embedding_provider="hashing",
            embedding_dimension=128,
            storage_path=os.path.join(td, "semfuse"),
        )
        db1 = SemFuse(config=cfg)
        db1.add("Dhaka is the capital of Bangladesh.")
        db1.add("ঢাকা বাংলাদেশের রাজধানী।")
        db1.close()

        db2 = SemFuse(config=cfg)
        assert db2.count() == 2
        results = db2.search("capital")
        assert len(results) > 0


def test_clear_removes_all(db: SemFuse) -> None:
    db.add("some text")
    db.add("more text")
    assert db.count() == 2
    db.clear()
    assert db.count() == 0
    assert db.search("text") == []


# ---------------------------------------------------------------------------
# Score properties
# ---------------------------------------------------------------------------


def test_scores_are_in_range(db: SemFuse) -> None:
    """All search scores should be in [0, 1] for cosine metric."""
    db.add("Dhaka is the capital of Bangladesh.")
    db.add("The Eiffel Tower is in Paris.")
    results = db.search("capital", top_k=5)
    for r in results:
        assert 0.0 <= r.score <= 1.0


def test_results_are_sorted_by_score(db: SemFuse) -> None:
    """Results should be sorted by score descending."""
    db.add("Dhaka is the capital of Bangladesh.")
    db.add("The Eiffel Tower is in Paris.")
    db.add("Tokyo is the capital of Japan.")
    results = db.search("capital", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limit(db: SemFuse) -> None:
    """top_k should limit the number of results."""
    for i in range(10):
        db.add(f"Document number {i} about topic {i}.")
    results = db.search("document", top_k=3)
    assert len(results) <= 3
