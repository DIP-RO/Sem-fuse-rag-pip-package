"""RAG pipeline: retrieve -> prompt with citations -> generate.

Includes robust error handling so that any failure during generation
falls back to the extractive template provider — the user always gets
a grounded, cited answer, even if the SLM or OpenAI provider crashes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from semfuse.core.exceptions import RAGError
from semfuse.core.types import RAGResponse, SearchResult
from semfuse.rag.base import LLMProvider
from semfuse.rag.prompt import build_rag_prompt

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str], list[SearchResult]]

# Fallback answer when no context is found — in both English and Bangla.
_NO_CONTEXT_EN = "I could not find relevant context to answer this question."
_NO_CONTEXT_BN = "এই প্রশ্নের উত্তর দেওয়ার জন্য প্রাসঙ্গিক তথ্য পাওয়া যায়নি।"


def _detect_bangla(text: str) -> bool:
    """Quick check: does the text contain Bangla Unicode characters?"""
    return any("\u0980" <= ch <= "\u09FF" for ch in text)


class RAGPipeline:
    """Composes a retrieval callable with an LLM provider.

    Decoupled from the client via ``retrieve``: any callable mapping a query
    to ranked :class:`SearchResult` objects works, so the pipeline is reusable
    with custom retrievers.

    Error handling: if the primary LLM provider raises an exception during
    generation, the pipeline falls back to the extractive template provider
    so the user always gets a grounded, cited answer.
    """

    def __init__(self, retrieve: RetrieveFn, llm: LLMProvider) -> None:
        self._retrieve = retrieve
        self._llm = llm

    def ask(self, question: str) -> RAGResponse:
        if not isinstance(question, str) or not question.strip():
            raise RAGError("question must be a non-empty string")

        citations = self._retrieve(question)
        if not citations:
            # Return "no context" in the same language as the question.
            answer = _NO_CONTEXT_BN if _detect_bangla(question) else _NO_CONTEXT_EN
            return RAGResponse(
                answer=answer,
                model=self._llm.model_name,
                citations=[],
                prompt="",
            )

        prompt = build_rag_prompt(question, citations)

        # Try the primary LLM provider.
        try:
            answer = self._llm.generate(prompt)
            if answer and answer.strip():
                return RAGResponse(
                    answer=answer,
                    model=self._llm.model_name,
                    citations=citations,
                    prompt=prompt,
                )
            # Empty answer — fall through to fallback.
            logger.warning("LLM provider returned empty answer, falling back to extractive")
        except Exception as exc:
            logger.warning("LLM provider failed (%s), falling back to extractive", exc)

        # Fallback: extractive template provider (always works, offline).
        from semfuse.rag.template import TemplateLLMProvider

        fallback = TemplateLLMProvider()
        answer = fallback.generate(prompt)
        return RAGResponse(
            answer=answer,
            model=fallback.model_name,
            citations=citations,
            prompt=prompt,
        )
