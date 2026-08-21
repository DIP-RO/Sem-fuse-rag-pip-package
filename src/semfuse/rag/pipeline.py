"""RAG pipeline: retrieve -> prompt with citations -> generate."""

from __future__ import annotations

from collections.abc import Callable

from semfuse.core.exceptions import RAGError
from semfuse.core.types import RAGResponse, SearchResult
from semfuse.rag.base import LLMProvider
from semfuse.rag.prompt import build_rag_prompt

RetrieveFn = Callable[[str], list[SearchResult]]


class RAGPipeline:
    """Composes a retrieval callable with an LLM provider.

    Decoupled from the client via ``retrieve``: any callable mapping a query
    to ranked :class:`SearchResult` objects works, so the pipeline is reusable
    with custom retrievers.
    """

    def __init__(self, retrieve: RetrieveFn, llm: LLMProvider) -> None:
        self._retrieve = retrieve
        self._llm = llm

    def ask(self, question: str) -> RAGResponse:
        if not isinstance(question, str) or not question.strip():
            raise RAGError("question must be a non-empty string")
        citations = self._retrieve(question)
        if not citations:
            return RAGResponse(
                answer="I could not find relevant context to answer this question.",
                model=self._llm.model_name,
                citations=[],
                prompt="",
            )
        prompt = build_rag_prompt(question, citations)
        answer = self._llm.generate(prompt)
        return RAGResponse(
            answer=answer,
            model=self._llm.model_name,
            citations=citations,
            prompt=prompt,
        )
