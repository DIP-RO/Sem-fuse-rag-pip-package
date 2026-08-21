"""Language detector tests (Phase 1 heuristic)."""

from __future__ import annotations

import pytest

from semfuse.core.enums import Language
from semfuse.language.detector import detect_language


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello world", Language.EN),
        ("The capital of Bangladesh is Dhaka.", Language.EN),
        ("ঢাকা বাংলাদেশের রাজধানী।", Language.BN),
        ("বাংলাদেশের রাজধানী কী?", Language.BN),
        ("বাংলাদেশের admission process কী?", Language.MIXED),
        ("", Language.UNKNOWN),
        ("   ", Language.UNKNOWN),
        ("12345", Language.UNKNOWN),
    ],
)
def test_detect_language(text: str, expected: Language) -> None:
    assert detect_language(text) == expected
