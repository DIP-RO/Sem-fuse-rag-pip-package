"""Metadata filtering tests."""

from __future__ import annotations

from semfuse import SemFuse


def test_metadata_stored_on_results(db: SemFuse) -> None:
    db.add("CSE admission notice.", metadata={"department": "CSE"})
    db.add("EEE admission notice.", metadata={"department": "EEE"})
    results = db.search("admission", filter={"department": "CSE"})
    assert len(results) > 0
    assert all(r.metadata.get("department") == "CSE" for r in results)


def test_metadata_filter_excludes_non_matching(db: SemFuse) -> None:
    db.add("CSE admission notice.", metadata={"department": "CSE"})
    db.add("EEE admission notice.", metadata={"department": "EEE"})
    results = db.search("admission", filter={"department": "CSE"}, top_k=10)
    assert all("EEE" not in r.text for r in results)


def test_include_metadata_false(db: SemFuse) -> None:
    db.add("a doc", metadata={"k": "v"})
    results = db.search("doc", include_metadata=False)
    assert results
    assert results[0].metadata == {}


def test_language_recorded_in_metadata(db: SemFuse) -> None:
    db.add("Dhaka is the capital.")
    db.add("ঢাকা বাংলাদেশের রাজধানী।")
    results = db.search("capital", top_k=10)
    langs = {r.metadata.get("language") for r in results}
    assert "en" in langs or "bn" in langs
