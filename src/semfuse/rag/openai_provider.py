"""OpenAI-backed LLM provider (optional extra: ``semfuse[rag]``)."""

from __future__ import annotations

from typing import Any

from semfuse.core.exceptions import RAGError


class OpenAILLMProvider:
    """Chat-completions provider. Requires ``openai`` and an API key.

    The client is created lazily so importing SemFuse never requires
    credentials. ``api_key`` falls back to the ``OPENAI_API_KEY`` environment
    variable (handled by the openai SDK itself).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        **client_options: Any,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client_options = client_options
        self._client: Any = None

    @property
    def model_name(self) -> str:
        return self._model

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise RAGError(
                "The openai package is required for llm_provider='openai'. "
                "Install it with `pip install semfuse[rag]`."
            ) from exc
        try:
            self._client = OpenAI(api_key=self._api_key, **self._client_options)
        except Exception as exc:
            raise RAGError(
                f"Failed to create OpenAI client: {exc}. Set OPENAI_API_KEY or "
                "pass api_key explicitly."
            ) from exc

    def generate(self, prompt: str) -> str:
        self._ensure_client()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise RAGError(f"OpenAI generation failed: {exc}") from exc
        content = response.choices[0].message.content
        return content.strip() if content else ""
