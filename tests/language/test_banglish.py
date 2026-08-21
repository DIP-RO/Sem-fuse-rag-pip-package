"""Phase 2: Banglish detection, normalization, and transliteration."""

from __future__ import annotations

import pytest

from semfuse.core.enums import Language
from semfuse.language.banglish import BanglishNormalizer, banglish_marker_count, looks_banglish
from semfuse.language.detector import detect_language
from semfuse.language.normalizer import normalize_for_search, normalize_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Bangladesh er capital ki?", Language.BANGLISH),
        ("ami tomake bhalobashi", Language.BANGLISH),
        ("bhorti porikkha kokhon hobe?", Language.BANGLISH),
        ("tumi kemon acho? ami bhalo achi", Language.BANGLISH),
        ("Hello world", Language.EN),
        ("The capital of Bangladesh is Dhaka.", Language.EN),
        ("Machine learning models require training data.", Language.EN),
        ("ঢাকা বাংলাদেশের রাজধানী।", Language.BN),
        ("বাংলাদেশের admission process কী?", Language.MIXED),
    ],
)
def test_detect_language_with_banglish(text: str, expected: Language) -> None:
    assert detect_language(text) == expected


def test_looks_banglish_empty() -> None:
    assert not looks_banglish("")
    assert not looks_banglish("12345")


def test_marker_count() -> None:
    assert banglish_marker_count("Bangladesh er capital ki?") >= 2
    assert banglish_marker_count("plain english sentence") == 0


def test_variant_folding() -> None:
    norm = BanglishNormalizer()
    # Alternate romanizations fold to one canonical form.
    assert norm.normalize("achhe") == norm.normalize("ache")
    assert norm.normalize("valo") == norm.normalize("bhalo")
    assert norm.normalize("kothai") == norm.normalize("kothay")


def test_elongation_collapse() -> None:
    norm = BanglishNormalizer()
    assert norm.normalize("kiii") == "ki"


def test_transliterate_known_tokens() -> None:
    norm = BanglishNormalizer()
    result = norm.transliterate("Bangladesh er rajdhani kothay?")
    assert "এর" in result
    assert "রাজধানী" in result
    assert "কোথায়" in result
    # Unknown tokens (English/proper nouns) pass through.
    assert "bangladesh" in result.lower()


def test_normalize_for_search_banglish_only() -> None:
    text = "er rajdhani ki"
    banglish = normalize_for_search(text, Language.BANGLISH)
    english = normalize_for_search(text, Language.EN)
    assert "রাজধানী" in banglish
    assert english == text  # non-Banglish text is left in its script


def test_normalize_text_strips_zero_width() -> None:
    assert normalize_text("বাং‌লা") == "বাংলা"
    assert normalize_text("﻿hello  world ") == "hello world"


def test_banglish_query_retrieves_bangla_doc_offline(db) -> None:
    """The core Phase 2 promise: Banglish -> Bangla retrieval works even with
    the hashing provider (no cross-script model), because transliteration
    bridges the scripts."""
    db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")
    db.add("The Eiffel Tower is in Paris.", document_id="eiffel")
    results = db.search("desh er rajdhani kothay?", top_k=1)
    assert results
    assert results[0].document_id == "capital"
