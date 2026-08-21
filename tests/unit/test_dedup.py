"""Duplicate detection tests."""

from __future__ import annotations

from semfuse import SemFuse


def test_duplicate_text_not_double_indexed(db: SemFuse) -> None:
    text = "Dhaka is the capital of Bangladesh."
    assert db.add(text) == 1
    # Same text again -> dedup, no new chunk.
    assert db.add(text) == 0
    assert db.count() == 1


def test_duplicate_in_add_many(db: SemFuse) -> None:
    texts = [
        "Dhaka is the capital of Bangladesh.",
        "Dhaka is the capital of Bangladesh.",
        "Tokyo is the capital of Japan.",
    ]
    added = db.add_many(texts)
    assert added == 2
    assert db.count() == 2


def test_whitespace_normalized_text_dedups(db: SemFuse) -> None:
    # Normalization collapses whitespace, so these two dedup.
    assert db.add("Dhaka   is   the   capital.") == 1
    assert db.add("Dhaka is the capital.") == 0
    assert db.count() == 1
