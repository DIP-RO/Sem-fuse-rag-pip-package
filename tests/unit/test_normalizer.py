"""Text normalizer tests."""

from __future__ import annotations

from semfuse.language.normalizer import normalize_text


def test_normalize_whitespace() -> None:
    assert normalize_text("  Dhaka   is   the   capital.  ") == "Dhaka is the capital."


def test_normalize_empty() -> None:
    assert normalize_text("") == ""


def test_normalize_preserves_bangla() -> None:
    text = "ঢাকা   বাংলাদেশের   রাজধানী।"
    assert normalize_text(text) == "ঢাকা বাংলাদেশের রাজধানী।"


def test_normalize_is_non_destructive_to_input() -> None:
    original = "  Dhaka   is   the   capital.  "
    _ = normalize_text(original)
    assert original == "  Dhaka   is   the   capital.  "
