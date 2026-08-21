"""Local Small Language Model (SLM) provider — lightweight, offline, no API key.

Uses a tiny multilingual model via ``transformers`` (optional extra:
``semfuse[slm]``).  The default model is ``Qwen2.5-0.5B-Instruct`` (500M
parameters, ~1 GB in FP16), which:

  * Runs on CPU (no GPU required)
  * Supports Bangla, English, and mixed-language text
  * Downloads once and is cached locally
  * Is lazy-loaded (import is fast; model loads on first ``generate()`` call)

**Evidence-grounded generation**: The SLM provider doesn't just forward the
prompt to the model — it post-processes the output to:

  1. **Enforce citations** — if the model forgot to cite, append [1].
  2. **Validate grounding** — if the answer doesn't overlap with any evidence
     passage, fall back to the extractive template provider (no hallucination).
  3. **Trim verbosity** — small models often repeat or ramble; the output is
     truncated to the first sentence(s) that contain a citation.
  4. **Detect refusals** — if the model says "I don't know" or similar, the
     answer is passed through as-is (honest refusal is better than fabrication).

This makes ``SemFuse(llm_provider="slm")`` fully self-contained — no OpenAI
key, no network calls after the initial model download, and answers are
grounded in retrieved evidence with citations.
"""

from __future__ import annotations

import re
from typing import Any

from semfuse.core.exceptions import RAGError
from semfuse.rag.prompt import build_system_instruction

_DEFAULT_SLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Patterns that indicate the model is honestly refusing (not hallucinating).
_REFUSAL_PATTERNS = [
    r"could not find",
    r"cannot find",
    r"don't know",
    r"do not know",
    r"no (?:relevant )?(?:context|information|evidence|passage)",
    r"not (?:enough )?(?:information|context|evidence)",
    r"unable to (?:answer|find)",
    r"প্রাসঙ্গিক.*তথ্য.*নেই",  # Bangla: relevant info not found
    r"তথ্য.*পাওয়া.*যায়নি",  # Bangla: info could not be found
    r"জানি না",  # Bangla: I don't know
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

# Citation marker pattern — [1], [2], etc.
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Context line pattern from the prompt.
_CONTEXT_LINE_RE = re.compile(r"^\[(\d+)\] \([^)]*\) (?P<text>.+)$", re.MULTILINE)
_QUESTION_RE = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)


class LocalSLMProvider:
    """Local SLM provider using ``transformers`` for lightweight generation.

    The model is loaded lazily on the first ``generate()`` call so that
    importing ``semfuse`` never triggers a download.  After the first call,
    the model and tokenizer are reused for all subsequent generations.

    Post-processing ensures answers are grounded in evidence with citations:
    - If the model output lacks citations, [1] is appended.
    - If the output doesn't overlap with any evidence passage, the extractive
      fallback is used (no hallucination).
    - Verbose or repetitive output is trimmed.
    """

    def __init__(
        self,
        model: str = _DEFAULT_SLM_MODEL,
        device: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.1,
        repetition_penalty: float = 1.1,
        **kwargs: Any,
    ) -> None:
        self._model_name = model
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
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
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        except Exception as exc:
            raise RAGError(
                f"Failed to load SLM model {self._model_name!r}: {exc}. "
                "The model will be downloaded on first use (~1 GB for the "
                "default Qwen2.5-0.5B-Instruct). Ensure you have an internet "
                "connection for the initial download."
            ) from exc

    def _generate_raw(self, prompt: str) -> str:
        """Run the model and return the raw decoded output."""
        self._ensure_loaded()
        try:
            # Use chat template if available (Qwen2.5 supports it).
            if hasattr(self._tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": build_system_instruction()},
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

            # Use low temperature for factual answers, with repetition penalty
            # to prevent small-model looping.
            do_sample = self._temperature > 0.0
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=do_sample,
                temperature=self._temperature if do_sample else 1.0,
                repetition_penalty=self._repetition_penalty,
                pad_token_id=self._tokenizer.pad_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][prompt_len:]
            result = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            return result.strip()
        except Exception as exc:
            raise RAGError(f"SLM generation failed: {exc}") from exc

    @staticmethod
    def _extract_evidence_passages(prompt: str) -> list[str]:
        """Extract the numbered evidence passage texts from the prompt."""
        return [m.group("text").strip() for m in _CONTEXT_LINE_RE.finditer(prompt)]

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """Extract the question from the prompt."""
        m = _QUESTION_RE.search(prompt)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _is_refusal(text: str) -> bool:
        """Check if the model output is an honest refusal (not hallucination)."""
        return bool(_REFUSAL_RE.search(text))

    @staticmethod
    def _has_citation(text: str) -> bool:
        """Check if the text contains at least one [n] citation marker."""
        return bool(_CITATION_RE.search(text))

    @staticmethod
    def _is_grounded(answer: str, evidence: list[str]) -> bool:
        """Check if the answer overlaps with at least one evidence passage.

        We check for significant token overlap (at least 2 non-stopword tokens)
        between the answer and any evidence passage. This detects hallucination
        where the model generates text not supported by any passage.
        """
        if not evidence:
            return False
        # Strip citation markers before tokenizing.
        answer_clean = _CITATION_RE.sub("", answer)
        # Simple word-level overlap check.
        # Use a broad Unicode word regex to handle Bangla + English.
        answer_tokens = set(re.findall(r"[\wঀ-৿]+", answer_clean.lower()))
        # Remove pure digits (from citation numbers, dates without context).
        answer_tokens = {t for t in answer_tokens if not t.isdigit()}
        if len(answer_tokens) < 2:
            # For very short answers (1 content token), accept 1-token overlap.
            if not answer_tokens:
                return True  # Empty or all-stopword answer — let it pass.
            for passage in evidence:
                passage_tokens = set(re.findall(r"[\wঀ-৿]+", passage.lower()))
                if answer_tokens & passage_tokens:
                    return True
            return False
        # Remove common stopwords/punctuation that don't carry meaning.
        _STOP = {
            "the", "a", "an", "is", "are", "was", "were", "be", "in", "on",
            "at", "to", "of", "and", "or", "not", "no", "it", "this", "that",
            "i", "you", "he", "she", "we", "they", "for", "with", "by",
            "from", "as", "its", "his", "her", "our", "their",
            "একটি", "একটা", "এই", "সেই", "তার", "যা", "এবং", "বা", "না",
            "মধ্যে", "জন্য", "থেকে", "সাথে", "করে", "হয়", "আছে", "নেই",
        }
        answer_content = answer_tokens - _STOP
        if not answer_content:
            # If the answer is all stopwords, consider it grounded (short answer).
            return True
        for passage in evidence:
            passage_tokens = set(re.findall(r"[\wঀ-৿]+", passage.lower()))
            overlap = answer_content & passage_tokens
            if len(overlap) >= 2:
                return True
        # For very short answers (1 content token), accept 1-token overlap.
        if len(answer_content) == 1:
            for passage in evidence:
                passage_tokens = set(re.findall(r"[\wঀ-৿]+", passage.lower()))
                if answer_content & passage_tokens:
                    return True
        return False

    @staticmethod
    def _trim_verbose(text: str) -> str:
        """Trim verbose/repetitive output to the first meaningful sentence(s).

        Small models (0.5B) often repeat themselves or generate run-on text.
        We keep up to the first 3 sentences, or up to the first citation,
        whichever comes first. We always preserve the citation.
        """
        # If the text is short enough, keep it as-is.
        if len(text) <= 300:
            return text

        # Split into sentences (Bangla + English terminators).
        sentences = re.split(r"(?<=[।.!?])\s+", text)
        if len(sentences) <= 1:
            return text

        # Keep up to 3 sentences, but always include any sentence with a citation.
        kept: list[str] = []
        for s in sentences[:5]:
            kept.append(s)
            if _CITATION_RE.search(s):
                break
            if len(kept) >= 3:
                break

        result = " ".join(kept).strip()
        # Ensure we didn't lose the citation.
        if not _CITATION_RE.search(result) and _CITATION_RE.search(text):
            # Find the first citation in the original text and append it.
            m = _CITATION_RE.search(text)
            if m:
                result = f"{result} {m.group(0)}"
        return result

    @staticmethod
    def _ensure_citation(text: str) -> str:
        """Ensure the answer ends with a citation marker."""
        if _CITATION_RE.search(text):
            return text
        return f"{text} [1]"

    def _extractive_fallback(self, prompt: str) -> str:
        """Fall back to the extractive template provider for grounded answers."""
        from semfuse.rag.template import TemplateLLMProvider

        return TemplateLLMProvider().generate(prompt)

    def generate(self, prompt: str) -> str:
        """Generate an answer with evidence grounding and citation enforcement.

        Pipeline:
        1. Run the SLM to get raw output.
        2. If the model refuses (honest "I don't know"), pass through.
        3. Trim verbose/repetitive output.
        4. Enforce citation — append [1] if missing.
        5. Validate grounding — if the answer doesn't overlap with any
           evidence passage, fall back to the extractive provider.
        """
        raw = self._generate_raw(prompt)

        # Step 2: Honest refusal passes through.
        if self._is_refusal(raw):
            return raw

        # Step 3: Trim verbose output.
        trimmed = self._trim_verbose(raw)

        # Step 4: Enforce citation.
        cited = self._ensure_citation(trimmed)

        # Step 5: Validate grounding against evidence passages.
        evidence = self._extract_evidence_passages(prompt)
        if evidence and not self._is_grounded(cited, evidence):
            # The model hallucinated — fall back to extractive.
            return self._extractive_fallback(prompt)

        return cited
