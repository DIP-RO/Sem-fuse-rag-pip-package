"""RAG prompt construction with numbered citations."""

from __future__ import annotations

from semfuse.core.types import SearchResult

_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the question using ONLY the numbered
context passages below. Cite every claim with the passage number in square
brackets, e.g. [1]. Answer in the same language as the question. If the
context does not contain the answer, say so plainly.

Context:
{context}

Question: {question}

Answer:"""


def format_context(results: list[SearchResult]) -> str:
    """Render results as numbered context passages ([1], [2], ...)."""
    lines = []
    for i, result in enumerate(results, start=1):
        origin = result.source or "unknown"
        if result.page is not None:
            origin += f", page {result.page}"
        lines.append(f"[{i}] ({origin}) {result.text}")
    return "\n".join(lines)


def build_rag_prompt(question: str, results: list[SearchResult]) -> str:
    """Build the generation prompt for ``question`` over retrieved ``results``."""
    return _PROMPT_TEMPLATE.format(context=format_context(results), question=question)
