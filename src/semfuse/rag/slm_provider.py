"""Local Small Language Model (SLM) provider — lightweight, offline, no API key.

Uses a tiny multilingual model via ``transformers`` (optional extra:
``semfuse[slm]``).  The default model is ``Qwen2.5-0.5B-Instruct`` (500M
parameters, ~1 GB in FP16), which:

  * Runs on CPU (no GPU required)
  * Supports Bangla, English, and mixed-language text
  * Downloads once and is cached locally
  * Is lazy-loaded (import is fast; model loads on first ``generate()`` call)

This makes ``SemFuse(llm_provider="slm")`` fully self-contained — no OpenAI
key, no network calls after the initial model download, no large orchestration
framework.  The model is small enough to run on a laptop.

For zero-dependency offline use, the default ``template`` (extractive)
provider remains the zero-config default.  The SLM provider is for developers
who want generative answers without an external API.
"""

from __future__ import annotations

from typing import Any

from semfuse.core.exceptions import RAGError

_DEFAULT_SLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class LocalSLMProvider:
    """Local SLM provider using ``transformers`` for lightweight generation.

    The model is loaded lazily on the first ``generate()`` call so that
    importing ``semfuse`` never triggers a download.  After the first call,
    the model and tokenizer are reused for all subsequent generations.
    """

    def __init__(
        self,
        model: str = _DEFAULT_SLM_MODEL,
        device: str | None = None,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> None:
        self._model_name = model
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._kwargs = kwargs
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RAGError(
                "The transformers package is required for llm_provider='slm'. "
                "Install it with `pip install semfuse[slm]`."
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype="auto",
                device_map=self._device or "auto",
                trust_remote_code=True,
                **self._kwargs,
            )
            # Ensure pad token exists for generation.
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        except Exception as exc:
            raise RAGError(
                f"Failed to load SLM model {self._model_name!r}: {exc}. "
                "The model will be downloaded on first use (~1 GB for the "
                "default Qwen2.5-0.5B-Instruct). Ensure you have an internet "
                "connection for the initial download."
            ) from exc

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        try:
            # Use chat template if available (Qwen2.5 supports it).
            if hasattr(self._tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer concisely in the same language as the question. Cite sources with [n] markers."},
                    {"role": "user", "content": prompt},
                ]
                text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = prompt

            inputs = self._tokenizer(text, return_tensors="pt")
            if self._device and self._device != "auto":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            elif hasattr(self._model, "device"):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,  # Deterministic for reproducibility.
                pad_token_id=self._tokenizer.pad_token_id,
            )
            # Decode only the new tokens (skip the prompt).
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][prompt_len:]
            result = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            return result.strip()
        except Exception as exc:
            raise RAGError(f"SLM generation failed: {exc}") from exc
