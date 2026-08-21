"""LLM provider factory."""

from __future__ import annotations

from semfuse.core.config import SemFuseConfig
from semfuse.core.exceptions import ConfigurationError
from semfuse.rag.base import LLMProvider
from semfuse.rag.openai_provider import OpenAILLMProvider
from semfuse.rag.template import TemplateLLMProvider


def create_llm_provider(config: SemFuseConfig) -> LLMProvider:
    """Build an LLM provider from configuration.

    Supported keys:
      * ``template`` -> deterministic extractive provider (offline, default)
      * ``slm``      -> local small language model (requires ``semfuse[slm]``)
      * ``openai``   -> OpenAI chat completions (requires ``semfuse[rag]``)
    """
    key = config.llm_provider
    if key == "template":
        return TemplateLLMProvider()
    if key == "slm":
        from semfuse.rag.slm_provider import LocalSLMProvider

        options = dict(config.llm_options)
        device = options.pop("device", config.device)
        max_new_tokens = options.pop("max_new_tokens", 256)
        return LocalSLMProvider(
            model=config.llm_model,
            device=device if isinstance(device, str) else None,
            max_new_tokens=max_new_tokens if isinstance(max_new_tokens, int) else 256,
            **options,
        )
    if key == "openai":
        options = dict(config.llm_options)
        api_key = options.pop("api_key", None)
        return OpenAILLMProvider(
            model=config.llm_model,
            api_key=api_key if isinstance(api_key, str) else None,
            **options,
        )
    raise ConfigurationError(
        f"Unknown llm_provider {key!r}. Supported: 'template', 'slm', 'openai'."
    )
