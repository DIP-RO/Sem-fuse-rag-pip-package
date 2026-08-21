"""Deterministic extractive "LLM" provider.

Keeps ``SemFuse().ask(...)`` zero-config and offline: instead of calling a
generative model it returns the top-ranked context passage verbatim with its
citation marker. Useful as a default, in tests, and anywhere an API key is
unavailable. Real generation comes from the ``openai`` provider
(``semfuse[rag]``) or a custom :class:`~semfuse.rag.base.LLMProvider`.
"""

from __future__ import annotations

import re

_CONTEXT_LINE_RE = re.compile(r"^\[1\] \([^)]*\) (?P<text>.+)$", re.MULTILINE)
_NO_CONTEXT_ANSWER = "I could not find relevant context to answer this question."


class TemplateLLMProvider:
    """Extractive provider: answers with the first context passage, cited."""

    @property
    def model_name(self) -> str:
        return "template-extractive"

    def generate(self, prompt: str) -> str:
        match = _CONTEXT_LINE_RE.search(prompt)
        if not match:
            return _NO_CONTEXT_ANSWER
        return f"{match.group('text').strip()} [1]"
