"""RAG-specific evaluation metrics.

These metrics evaluate the *answer* quality, not just retrieval ranking.
They complement the retrieval metrics in ``metrics.py`` (Hit@K, NDCG@K, MRR)
by measuring:

* **Answer accuracy** — does the generated answer contain the expected
  answer? Supports both exact-match and substring containment, with
  Bangla-aware tokenization.
* **Faithfulness** — is every claim in the answer supported by the
  retrieved evidence? Uses token overlap with stopword removal (same
  algorithm as the SLM grounding check).
* **Citation accuracy** — are the citation markers ``[n]`` in the answer
  pointing to passages that actually support the claims?
* **Refusal accuracy** — when the answer *should* be "I don't know", does
  the system correctly refuse? And when it *should* answer, does it avoid
  false refusals?

All metrics return floats in ``[0.0, 1.0]`` so they can be aggregated
across a dataset and reported alongside retrieval metrics.
"""

from __future__ import annotations

import re

# Citation marker pattern — [1], [2], etc.
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Refusal patterns — honest "I don't know" in English and Bangla.
_REFUSAL_PATTERNS = [
    r"could not find",
    r"cannot find",
    r"don't know",
    r"do not know",
    r"no (?:relevant )?(?:context|information|evidence|passage)",
    r"not (?:enough )?(?:information|context|evidence)",
    r"unable to (?:answer|find)",
    r"প্রাসঙ্গিক.*তথ্য.*নেই",
    r"তথ্য.*পাওয়া.*যায়নি",
    r"জানি না",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# Broad Unicode word regex — handles Bangla (U+0980–U+09FF) + English.
_WORD_RE = re.compile(r"[\wঀ-৿]+")

# Stopwords that don't carry answer meaning (English + Bangla).
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "in", "on",
    "at", "to", "of", "and", "or", "not", "no", "it", "this", "that",
    "i", "you", "he", "she", "we", "they", "for", "with", "by",
    "from", "as", "its", "his", "her", "our", "their",
    "একটি", "একটা", "এই", "সেই", "তার", "যা", "এবং", "বা", "না",
    "মধ্যে", "জন্য", "থেকে", "সাথে", "করে", "হয়", "আছে", "নেই",
    "কি", "কী", "কোথায়", "কখন", "কে", "কেন", "কিভাবে", "কীভাবে",
    "what", "where", "when", "who", "why", "how", "which", "whose",
})


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase Unicode word tokens, excluding digits."""
    tokens = set(_WORD_RE.findall(text.lower()))
    return {t for t in tokens if not t.isdigit()}


def _content_tokens(text: str) -> set[str]:
    """Tokenize and remove stopwords — the meaningful content tokens."""
    return _tokenize(text) - _STOPWORDS


# ---------------------------------------------------------------------------
# Answer accuracy
# ---------------------------------------------------------------------------


def answer_accuracy(
    answer: str,
    expected: str,
    *,
    mode: str = "substring",
) -> float:
    """Check if the answer contains the expected answer.

    Args:
        answer: The generated answer (may include citation markers).
        expected: The ground-truth answer string (e.g. "ঢাকা" or "Dhaka").
        mode: Matching mode:
            - ``"substring"`` — 1.0 if expected is a substring of answer
              (case-insensitive), 0.0 otherwise. Default.
            - ``"token"`` — fraction of expected content tokens found in
              the answer. More forgiving for multi-word answers.
            - ``"exact"`` — 1.0 if the answer (stripped of citations)
              exactly equals the expected, 0.0 otherwise.

    Returns:
        Float in [0.0, 1.0].
    """
    if not expected or not answer:
        return 0.0

    # Strip citation markers from the answer for comparison.
    answer_clean = _CITATION_RE.sub("", answer).strip().lower()
    expected_clean = expected.strip().lower()

    if mode == "exact":
        return 1.0 if answer_clean == expected_clean else 0.0

    if mode == "token":
        expected_tokens = _content_tokens(expected_clean)
        if not expected_tokens:
            return 1.0 if expected_clean in answer_clean else 0.0
        answer_tokens = _tokenize(answer_clean)
        overlap = expected_tokens & answer_tokens
        return len(overlap) / len(expected_tokens)

    # Default: substring
    return 1.0 if expected_clean in answer_clean else 0.0


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def faithfulness(answer: str, evidence_passages: list[str]) -> float:
    """Check if the answer is supported by the evidence passages.

    Uses token overlap with stopword removal: the answer is "faithful" if
    its content tokens overlap with at least one evidence passage. This is
    the same algorithm as the SLM grounding check, exposed as a metric.

    Args:
        answer: The generated answer (may include citation markers).
        evidence_passages: The retrieved context passages.

    Returns:
        1.0 if the answer is grounded in evidence, 0.0 if it appears
        hallucinated (no token overlap with any passage).
    """
    if not answer or not evidence_passages:
        return 0.0

    # Strip citations before checking.
    answer_clean = _CITATION_RE.sub("", answer)
    answer_content = _content_tokens(answer_clean)

    if not answer_content:
        # All stopwords — can't determine, assume faithful.
        return 1.0

    for passage in evidence_passages:
        passage_tokens = _tokenize(passage)
        overlap = answer_content & passage_tokens
        # Need at least 2 content tokens overlapping, or 1 if the answer
        # is very short (single content token).
        if len(overlap) >= 2:
            return 1.0
        if len(answer_content) == 1 and len(overlap) >= 1:
            return 1.0

    return 0.0


# ---------------------------------------------------------------------------
# Citation accuracy
# ---------------------------------------------------------------------------


def citation_accuracy(
    answer: str,
    evidence_passages: list[str],
) -> float:
    """Check if citations in the answer point to supporting passages.

    For each citation ``[n]`` in the answer:
    - If ``n`` is out of range (no passage n), that's an incorrect citation.
    - If passage n exists, check if the answer content overlaps with it.

    Args:
        answer: The generated answer with citation markers.
        evidence_passages: The retrieved context passages (1-indexed).

    Returns:
        Fraction of citations that are correct (point to a passage that
        supports the answer content). 0.0 if no citations present.
    """
    if not answer or not evidence_passages:
        return 0.0

    citations = _CITATION_RE.findall(answer)
    if not citations:
        return 0.0

    answer_clean = _CITATION_RE.sub("", answer)
    answer_content = _content_tokens(answer_clean)

    correct = 0
    for cite_str in citations:
        n = int(cite_str)
        if n < 1 or n > len(evidence_passages):
            continue  # Out of range — incorrect.
        passage = evidence_passages[n - 1]
        passage_tokens = _tokenize(passage)
        overlap = answer_content & passage_tokens
        if len(overlap) >= 1:
            correct += 1

    return correct / len(citations)


# ---------------------------------------------------------------------------
# Refusal accuracy
# ---------------------------------------------------------------------------


def is_refusal(answer: str) -> bool:
    """Check if the answer is an honest refusal ("I don't know")."""
    return bool(_REFUSAL_RE.search(answer))


def refusal_accuracy(answer: str, should_refuse: bool) -> float:
    """Check if the refusal behavior is correct.

    Args:
        answer: The generated answer.
        should_refuse: Whether the system *should* refuse (no relevant
            context available).

    Returns:
        1.0 if the refusal behavior is correct:
        - If should_refuse=True and the answer is a refusal → correct.
        - If should_refuse=False and the answer is NOT a refusal → correct.
        0.0 otherwise (false refusal or false answer).
    """
    actual_refusal = is_refusal(answer)
    if should_refuse:
        return 1.0 if actual_refusal else 0.0
    return 1.0 if not actual_refusal else 0.0


# ---------------------------------------------------------------------------
# Combined RAG score
# ---------------------------------------------------------------------------


def rag_score(
    answer: str,
    expected: str | None,
    evidence_passages: list[str],
    should_refuse: bool = False,
    *,
    accuracy_mode: str = "substring",
) -> dict[str, float]:
    """Compute all RAG metrics for a single answer.

    Args:
        answer: The generated answer.
        expected: The ground-truth answer, or None if should_refuse=True.
        evidence_passages: The retrieved context passages.
        should_refuse: Whether the system should have refused.
        accuracy_mode: Matching mode for answer_accuracy.

    Returns:
        Dict with keys: answer_accuracy, faithfulness, citation_accuracy,
        refusal_accuracy.
    """
    if should_refuse:
        return {
            "answer_accuracy": 1.0 if is_refusal(answer) else 0.0,
            "faithfulness": 1.0,  # No claim to verify.
            "citation_accuracy": 1.0,  # No citations expected.
            "refusal_accuracy": refusal_accuracy(answer, should_refuse=True),
        }

    return {
        "answer_accuracy": answer_accuracy(answer, expected or "", mode=accuracy_mode)
        if expected
        else 0.0,
        "faithfulness": faithfulness(answer, evidence_passages),
        "citation_accuracy": citation_accuracy(answer, evidence_passages),
        "refusal_accuracy": refusal_accuracy(answer, should_refuse=False),
    }
