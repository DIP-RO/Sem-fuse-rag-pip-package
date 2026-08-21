"""LLM provider protocol for the RAG layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Interface for text-generation backends used by the RAG pipeline."""

    @property
    def model_name(self) -> str:
        """Stable identifier of the generation model."""
        ...

    def generate(self, prompt: str) -> str:
        """Generate a completion for ``prompt``."""
        ...
