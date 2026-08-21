"""semfuse.rag subpackage."""

from __future__ import annotations

from semfuse.rag.base import LLMProvider
from semfuse.rag.factory import create_llm_provider
from semfuse.rag.openai_provider import OpenAILLMProvider
from semfuse.rag.pipeline import RAGPipeline
from semfuse.rag.prompt import build_rag_prompt, format_context
from semfuse.rag.template import TemplateLLMProvider

__all__ = [
    "LLMProvider",
    "OpenAILLMProvider",
    "RAGPipeline",
    "TemplateLLMProvider",
    "build_rag_prompt",
    "create_llm_provider",
    "format_context",
]
