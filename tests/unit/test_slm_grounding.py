"""Tests for the SLM provider's evidence-grounding post-processing.

These tests verify the citation enforcement, grounding validation, refusal
detection, and verbose trimming logic — all without loading the actual model
(which requires transformers + torch + a 1 GB download).
"""

from __future__ import annotations

from semfuse.rag.slm_provider import LocalSLMProvider

# ---------------------------------------------------------------------------
# Citation enforcement
# ---------------------------------------------------------------------------


def test_ensure_citation_adds_missing() -> None:
    assert LocalSLMProvider._ensure_citation("Dhaka is the capital") == "Dhaka is the capital [1]"


def test_ensure_citation_preserves_existing() -> None:
    assert LocalSLMProvider._ensure_citation("Dhaka [1]") == "Dhaka [1]"


def test_ensure_citation_preserves_multi_citation() -> None:
    assert LocalSLMProvider._ensure_citation("Dhaka [1] is large [2]") == "Dhaka [1] is large [2]"


def test_has_citation_true() -> None:
    assert LocalSLMProvider._has_citation("ঢাকা [1]") is True


def test_has_citation_false() -> None:
    assert LocalSLMProvider._has_citation("ঢাকা") is False


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------


def test_is_refusal_english() -> None:
    assert LocalSLMProvider._is_refusal("I could not find relevant context.") is True
    assert LocalSLMProvider._is_refusal("I don't know the answer.") is True
    assert LocalSLMProvider._is_refusal("No relevant information found.") is True


def test_is_refusal_bangla() -> None:
    assert LocalSLMProvider._is_refusal("প্রাসঙ্গিক তথ্য নেই।") is True
    assert LocalSLMProvider._is_refusal("আমি জানি না।") is True


def test_is_not_refusal() -> None:
    assert LocalSLMProvider._is_refusal("ঢাকা বাংলাদেশের রাজধানী [1]") is False
    assert LocalSLMProvider._is_refusal("Dhaka is the capital [1]") is False


# ---------------------------------------------------------------------------
# Grounding validation
# ---------------------------------------------------------------------------


def test_is_grounded_english() -> None:
    evidence = ["Dhaka is the capital of Bangladesh."]
    assert LocalSLMProvider._is_grounded("Dhaka is the capital [1]", evidence) is True


def test_is_grounded_bangla() -> None:
    evidence = ["ঢাকা বাংলাদেশের রাজধানী।"]
    assert LocalSLMProvider._is_grounded("ঢাকা বাংলাদেশের রাজধানী [1]", evidence) is True


def test_is_grounded_partial_overlap() -> None:
    evidence = ["The Padma is a major river of Bangladesh."]
    assert LocalSLMProvider._is_grounded("Padma is a major river [1]", evidence) is True


def test_is_not_grounded_hallucination() -> None:
    evidence = ["Dhaka is the capital of Bangladesh."]
    assert LocalSLMProvider._is_grounded(
        "The moon is made of cheese [1]", evidence
    ) is False


def test_is_grounded_short_answer() -> None:
    evidence = ["ঢাকা বাংলাদেশের রাজধানী।"]
    # Single-word answer with 1 content token should still be grounded.
    assert LocalSLMProvider._is_grounded("ঢাকা [1]", evidence) is True


def test_is_grounded_empty_evidence() -> None:
    assert LocalSLMProvider._is_grounded("anything", []) is False


# ---------------------------------------------------------------------------
# Verbose trimming
# ---------------------------------------------------------------------------


def test_trim_verbose_short_text_preserved() -> None:
    text = "ঢাকা [1]"
    assert LocalSLMProvider._trim_verbose(text) == text


def test_trim_verbose_long_text_truncated() -> None:
    # Build a long repetitive text that a small model might produce.
    text = "ঢাকা is the capital. " * 50 + "[1]"
    trimmed = LocalSLMProvider._trim_verbose(text)
    assert len(trimmed) < len(text)
    assert "[1]" in trimmed


def test_trim_verbose_preserves_citation() -> None:
    text = "Dhaka is the capital of Bangladesh. It is a big city. The population is large. More info here. Even more. [1]"
    trimmed = LocalSLMProvider._trim_verbose(text)
    assert "[1]" in trimmed


def test_trim_verbose_keeps_up_to_3_sentences() -> None:
    text = (
        "Dhaka is the capital of Bangladesh and it is a very large city. "
        "It is the largest city in the country with a huge population. "
        "The population is over 20 million people living in the metropolitan area. "
        "It has many rivers flowing through it including the Buriganga river. "
        "The food is great and the culture is very rich and diverse in many ways. "
        "[1]"
    )
    trimmed = LocalSLMProvider._trim_verbose(text)
    # Should be shorter than original (text is > 300 chars).
    assert len(trimmed) < len(text)
    # Should contain the citation.
    assert "[1]" in trimmed


# ---------------------------------------------------------------------------
# Evidence passage extraction from prompt
# ---------------------------------------------------------------------------


def test_extract_evidence_passages() -> None:
    prompt = """\
Answer the question using only the evidence below.

Evidence:
[1] (corpus) ঢাকা বাংলাদেশের রাজধানী।
[2] (corpus) পদ্মা একটি বড় নদী।

Question: বাংলাদেশের রাজধানী কী?

Answer:"""
    passages = LocalSLMProvider._extract_evidence_passages(prompt)
    assert len(passages) == 2
    assert "ঢাকা" in passages[0]
    assert "পদ্মা" in passages[1]


def test_extract_question() -> None:
    prompt = """\
Evidence:
[1] (corpus) Dhaka is the capital.

Question: What is the capital?

Answer:"""
    assert LocalSLMProvider._extract_question(prompt) == "What is the capital?"


# ---------------------------------------------------------------------------
# SLM provider initialization (no model download)
# ---------------------------------------------------------------------------


def test_slm_init_does_not_load_model() -> None:
    provider = LocalSLMProvider()
    assert provider._llm is None
    assert provider._tokenizer is None
    assert provider._backend is None
    assert provider.model_name == "Qwen/Qwen2.5-0.5B-Instruct-GGUF"


def test_slm_default_params_optimized_for_factual() -> None:
    provider = LocalSLMProvider()
    assert provider._max_new_tokens == 128  # Short answers
    assert provider._temperature == 0.1  # Low for factual
    assert provider._repetition_penalty == 1.1  # Prevent looping


# ---------------------------------------------------------------------------
# Prompt construction with structured evidence
# ---------------------------------------------------------------------------


def test_prompt_has_structured_instructions() -> None:
    from semfuse.core.types import SearchResult
    from semfuse.rag.prompt import build_rag_prompt

    results = [SearchResult(text="ঢাকা বাংলাদেশের রাজধানী।", score=0.9, chunk_id="c1", document_id="d1", source="corpus")]
    prompt = build_rag_prompt("বাংলাদেশের রাজধানী কী?", results)
    assert "Evidence:" in prompt
    assert "[1]" in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt
    assert "Cite" in prompt or "cite" in prompt
    assert "same language" in prompt


def test_system_instruction_is_concise() -> None:
    from semfuse.rag.prompt import build_system_instruction

    si = build_system_instruction()
    assert len(si) < 300  # Short for small models
    assert "ONLY" in si or "only" in si
    assert "Cite" in si or "cite" in si
