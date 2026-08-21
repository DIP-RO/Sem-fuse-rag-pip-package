"""Phase 3: recursive chunking."""

from __future__ import annotations

import pytest

from semfuse.chunking.recursive import RecursiveCharacterChunker
from semfuse.core.exceptions import ConfigurationError


def test_short_text_single_chunk() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("short text") == ["short text"]


def test_empty_text_no_chunks() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
    assert chunker.split("   ") == []


def test_respects_chunk_size() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=80, chunk_overlap=10)
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunker.split(text)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_paragraphs_preferred() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=0)
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunker.split(text)
    assert any("First paragraph" in c for c in chunks)
    assert any("Third paragraph" in c for c in chunks)


def test_bangla_dari_sentence_split() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=0)
    text = "ঢাকা বাংলাদেশের রাজধানী। পদ্মা একটি বড় নদী। বাংলা আমাদের ভাষা।"
    chunks = chunker.split(text)
    assert len(chunks) >= 2
    # The dari terminator stays attached to its sentence.
    assert any(c.endswith("।") for c in chunks)


def test_overlap_carries_context() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=20)
    text = " ".join(f"tok{i}" for i in range(40))
    chunks = chunker.split(text)
    assert len(chunks) >= 2
    # Some trailing content of chunk N reappears at the start of chunk N+1.
    tail_words = chunks[0].split()[-2:]
    assert any(w in chunks[1].split()[:6] for w in tail_words)


def test_no_content_lost() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=0)
    words = [f"unique{i}" for i in range(50)]
    chunks = chunker.split(" ".join(words))
    joined = " ".join(chunks)
    for w in words:
        assert w in joined


def test_deterministic() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=70, chunk_overlap=15)
    text = "One sentence. Another sentence! A third one? " * 10
    assert chunker.split(text) == chunker.split(text)


def test_hard_cut_without_separators() -> None:
    chunker = RecursiveCharacterChunker(chunk_size=10, chunk_overlap=0)
    chunks = chunker.split("a" * 35)
    assert all(len(c) <= 10 for c in chunks)
    assert sum(len(c) for c in chunks) == 35


def test_invalid_params_raise() -> None:
    with pytest.raises(ConfigurationError):
        RecursiveCharacterChunker(chunk_size=0)
    with pytest.raises(ConfigurationError):
        RecursiveCharacterChunker(chunk_size=10, chunk_overlap=10)


def test_client_chunks_long_document(db) -> None:
    long_text = "\n\n".join(f"Paragraph {i} about topic number {i}." for i in range(60))
    added = db.add(long_text)
    assert added > 1
    assert db.count() == added
    # All chunks share one parent document.
    assert db.info()["document_count"] == 1
