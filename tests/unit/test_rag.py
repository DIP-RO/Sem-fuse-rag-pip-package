"""Phase 6: RAG pipeline, providers, and citations."""

from __future__ import annotations

from pathlib import Path

import pytest

from semfuse import RAGResponse, SemFuse
from semfuse.core.config import SemFuseConfig
from semfuse.core.exceptions import ConfigurationError, RAGError
from semfuse.core.types import SearchResult
from semfuse.rag.factory import create_llm_provider
from semfuse.rag.pipeline import RAGPipeline
from semfuse.rag.prompt import build_rag_prompt, format_context
from semfuse.rag.template import TemplateLLMProvider


def _r(doc_id: str, text: str, score: float = 0.9, page: int | None = None) -> SearchResult:
    return SearchResult(
        text=text, score=score, chunk_id=doc_id, document_id=doc_id, source="corpus", page=page
    )


def test_format_context_numbering_and_pages() -> None:
    ctx = format_context([_r("a", "First passage."), _r("b", "Second passage.", page=4)])
    assert "[1] (corpus) First passage." in ctx
    assert "[2] (corpus, page 4) Second passage." in ctx


def test_build_prompt_contains_question_and_rules() -> None:
    prompt = build_rag_prompt("What is X?", [_r("a", "X is Y.")])
    assert "Question: What is X?" in prompt
    assert "[1] (corpus) X is Y." in prompt
    assert "Cite" in prompt


def test_template_provider_extracts_top_passage() -> None:
    provider = TemplateLLMProvider()
    # "What is X?" with passage "X is Y." -> extracts predicate "Y"
    prompt = build_rag_prompt("What is X?", [_r("a", "X is Y."), _r("b", "Other.")])
    result = provider.generate(prompt)
    assert result.endswith("[1]")
    assert "Y" in result
    assert provider.model_name == "template-extractive"


def test_template_provider_extracts_subject_for_capital_question() -> None:
    provider = TemplateLLMProvider()
    # "What is the capital?" with "Dhaka is the capital of Bangladesh."
    # -> extracts subject "Dhaka" (the thing that IS the capital)
    prompt = build_rag_prompt(
        "What is the capital of Bangladesh?",
        [_r("a", "Dhaka is the capital of Bangladesh.")],
    )
    assert provider.generate(prompt) == "Dhaka [1]"


def test_template_provider_bangla_extraction() -> None:
    provider = TemplateLLMProvider()
    prompt = build_rag_prompt(
        "বাংলাদেশের রাজধানী কী?",
        [_r("a", "ঢাকা বাংলাদেশের রাজধানী।")],
    )
    assert provider.generate(prompt) == "ঢাকা [1]"


def test_template_provider_banglish_extraction() -> None:
    provider = TemplateLLMProvider()
    prompt = build_rag_prompt(
        "Bangladesh er capital ki?",
        [_r("a", "ঢাকা বাংলাদেশের রাজধানী।")],
    )
    assert provider.generate(prompt) == "ঢাকা [1]"


def test_template_provider_where_extraction() -> None:
    provider = TemplateLLMProvider()
    prompt = build_rag_prompt(
        "Where is the Eiffel Tower?",
        [_r("a", "The Eiffel Tower is in Paris.")],
    )
    assert provider.generate(prompt) == "Paris [1]"


def test_template_provider_when_extraction_bangla() -> None:
    provider = TemplateLLMProvider()
    prompt = build_rag_prompt(
        "bhorti porikkha kokhon hobe?",
        [_r("a", "বিশ্ববিদ্যালয়ে ভর্তি পরীক্ষা ডিসেম্বরে অনুষ্ঠিত হবে।")],
    )
    result = provider.generate(prompt)
    assert "ডিসেম্বরে" in result
    assert result.endswith("[1]")


def test_pipeline_returns_citations() -> None:
    results = [_r("a", "Dhaka is the capital."), _r("b", "Padma is a river.")]
    pipeline = RAGPipeline(retrieve=lambda q: results, llm=TemplateLLMProvider())
    response = pipeline.ask("capital?")
    assert isinstance(response, RAGResponse)
    assert response.answer.endswith("[1]")
    assert [c.document_id for c in response.citations] == ["a", "b"]
    assert "Dhaka is the capital." in response.prompt
    assert response.model == "template-extractive"


def test_pipeline_no_context() -> None:
    pipeline = RAGPipeline(retrieve=lambda q: [], llm=TemplateLLMProvider())
    response = pipeline.ask("anything?")
    assert "could not find" in response.answer
    assert response.citations == []
    assert response.prompt == ""


def test_pipeline_rejects_empty_question() -> None:
    pipeline = RAGPipeline(retrieve=lambda q: [], llm=TemplateLLMProvider())
    with pytest.raises(RAGError):
        pipeline.ask("   ")


def test_client_ask_end_to_end(db: SemFuse) -> None:
    db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")
    db.add("The Eiffel Tower is in Paris.", document_id="eiffel")
    response = db.ask("Bangladesh er rajdhani kothay?")
    assert "ঢাকা" in response.answer
    assert response.answer.endswith("[1]")
    assert response.citations[0].document_id == "capital"


def test_client_ask_empty_index(db: SemFuse) -> None:
    response = db.ask("anything?")
    assert response.citations == []


def test_factory_template_and_unknown(tmp_path: Path) -> None:
    cfg = SemFuseConfig(storage_path=tmp_path)
    assert isinstance(create_llm_provider(cfg), TemplateLLMProvider)
    cfg.llm_provider = "bogus"
    with pytest.raises(ConfigurationError):
        create_llm_provider(cfg)


def test_openai_provider_lazy_no_key_needed_at_init(tmp_path: Path) -> None:
    from semfuse.rag.openai_provider import OpenAILLMProvider

    provider = OpenAILLMProvider(model="gpt-4o-mini")
    assert provider.model_name == "gpt-4o-mini"  # no client/key needed until generate()
