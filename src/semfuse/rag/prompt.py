"""RAG prompt construction with numbered citations and evidence grounding.

The prompt is structured to work well with **small models** (0.5B–3B params):
- Clear role instruction (short, imperative)
- Numbered evidence passages with explicit source labels
- A structured answer format that forces citation before generation
- Language-matching instruction (Bangla question → Bangla answer)
- An explicit "if not found" guard to reduce hallucination
"""

from __future__ import annotations

from semfuse.core.types import SearchResult

# System instruction for the SLM — kept short and imperative for small models.
_SYSTEM_INSTRUCTION = (
    "You are a factual answer engine. "
    "Answer using ONLY the evidence passages below. "
    "Cite every fact with [n] markers. "
    "Answer in the same language as the question. "
    "If the evidence does not contain the answer, say: "
    "I could not find relevant context to answer this question."
)

# Main prompt template — structured for small-model comprehension.
_PROMPT_TEMPLATE = """\
Answer the question using only the evidence below.

Evidence:
{context}

Question: {question}

Instructions:
1. Find the most relevant evidence passage(s) for the question.
2. Extract the answer from that passage — do not add outside knowledge.
3. Cite the passage number in square brackets, e.g. [1].
4. Answer in the same language as the question (Bangla, English, or mixed).
5. Keep the answer concise — one sentence or a short phrase.
6. If no passage contains the answer, say: I could not find relevant context to answer this question.

Answer:"""


def format_context(results: list[SearchResult]) -> str:
    """Render results as numbered context passages ([1], [2], ...).

    Each passage includes the source label and the text, formatted so that
    a small model can easily parse the boundary between passages.
    """
    lines = []
    for i, result in enumerate(results, start=1):
        origin = result.source or "unknown"
        if result.page is not None:
            origin += f", page {result.page}"
        # Use a clear delimiter that small models can parse.
        lines.append(f"[{i}] ({origin}) {result.text}")
    return "\n".join(lines)


def build_rag_prompt(question: str, results: list[SearchResult]) -> str:
    """Build the generation prompt for ``question`` over retrieved ``results``."""
    return _PROMPT_TEMPLATE.format(context=format_context(results), question=question)


def build_system_instruction() -> str:
    """Return the system instruction for chat-template-based generation."""
    return _SYSTEM_INSTRUCTION
