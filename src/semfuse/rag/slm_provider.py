"""Local Small Language Model (SLM) provider — lightweight, offline, no API key.

Uses ``llama-cpp-python`` with a quantized GGUF model (optional extra:
``semfuse[slm]``).  The default model is ``Qwen2.5-0.5B-Instruct`` in Q4_K_M
quantization (~400 MB on disk), which:

  * Runs on CPU (no GPU required, no CUDA toolkit)
  * ~50 MB library + ~400 MB model = ~450 MB total (vs ~2.5 GB for torch)
  * Supports Bangla, English, and mixed-language text
  * Downloads once and is cached locally
  * Is lazy-loaded (import is fast; model loads on first ``generate()`` call)
  * Works on x86_64 and ARM64 (Apple Silicon, Graviton, RPi)

**Backend auto-detection**: The provider tries ``llama-cpp-python`` first
(lightweight).  If ``transformers`` + ``torch`` are installed instead, it
falls back to the HuggingFace backend.  This lets users choose their
preferred backend without changing code.

**Evidence-grounded generation**: The SLM provider post-processes the output
to enforce citations, validate grounding, and fall back to extractive answers
if the model hallucinates.
"""

from __future__ import annotations

import re
from typing import Any

from semfuse.core.exceptions import RAGError
from semfuse.rag.prompt import build_system_instruction

# Default GGUF model — Q4_K_M quantization (~400 MB).
# Hosted on HuggingFace as Qwen/Qwen2.5-0.5B-Instruct-GGUF.
_DEFAULT_GGUF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
_DEFAULT_GGUF_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Fallback HuggingFace model (if transformers+torch installed).
_DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Patterns that indicate the model is honestly refusing (not hallucinating).
_REFUSAL_PATTERNS = [
    r"could not find",
    r"cannot find",
    r"don't know",
    r"do not know",
    r"no (?:relevant )?(?:context|information|evidence|passage)",
    r"not (?:enough )?(?:information|context|evidence)",
    r"unable to (?:answer|find)",
    r"প্রাসঙ্গিক.*তথ্য.*নেই",
    r"তথ্য.*পাওয়া.*যায়নি",
    r"জানি না",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_CONTEXT_LINE_RE = re.compile(r"^\[(\d+)\] \([^)]*\) (?P<text>.+)$", re.MULTILINE)
_QUESTION_RE = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)


class LocalSLMProvider:
    """Local SLM provider using ``llama-cpp-python`` (preferred) or ``transformers``.

    The model is loaded lazily on the first ``generate()`` call so that
    importing ``semfuse`` never triggers a download.  After the first call,
    the model is reused for all subsequent generations.

    Backend selection:
    1. If ``llama-cpp-python`` is installed → use GGUF model (lightweight, ~450 MB)
    2. Elif ``transformers`` + ``torch`` are installed → use HF model (~2.5 GB)
    3. Else → raise RAGError with install instructions

    Post-processing ensures answers are grounded in evidence with citations.
    """

    def __init__(
        self,
        model: str = _DEFAULT_GGUF_MODEL,
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
        self._backend: str | None = None  # "llama_cpp" or "transformers"
        self._llm: Any = None  # llama_cpp.Llama or transformers model
        self._tokenizer: Any = None  # transformers tokenizer (if applicable)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        """Load the model lazily, auto-detecting the available backend."""
        if self._llm is not None:
            return

        # Try llama-cpp-python first (lightweight ~50 MB library).
        try:
            self._load_llama_cpp()
            return
        except ImportError:
            pass
        except Exception as exc:
            raise RAGError(f"Failed to load SLM via llama-cpp-python: {exc}") from exc

        # Fall back to transformers + torch (heavier ~2.5 GB).
        try:
            self._load_transformers()
            return
        except ImportError:
            pass
        except Exception as exc:
            raise RAGError(f"Failed to load SLM via transformers: {exc}") from exc

        raise RAGError(
            "No SLM backend available. Install one of:\n"
            "  pip install semfuse[slm]       # llama-cpp-python (~50 MB, recommended)\n"
            "  pip install semfuse[slm-torch]  # transformers + torch (~2.5 GB)"
        )

    def _load_llama_cpp(self) -> None:
        """Load model via llama-cpp-python (lightweight GGUF backend)."""
        from llama_cpp import Llama

        # If the user passed a HuggingFace model ID, convert to GGUF equivalent.
        model_id = self._model_name
        gguf_file = self._kwargs.get("gguf_file", _DEFAULT_GGUF_FILE)

        # If user specified a plain HF model name, use the GGUF repo.
        if "gguf" not in model_id.lower() and "Qwen2.5-0.5B" in model_id:
            model_id = _DEFAULT_GGUF_MODEL

        self._llm = Llama.from_pretrained(
            repo_id=model_id,
            filename=gguf_file,
            n_ctx=2048,  # Context window
            n_threads=self._kwargs.get("n_threads", 4),
            verbose=False,
            **{k: v for k, v in self._kwargs.items() if k != "gguf_file"},
        )
        self._backend = "llama_cpp"
        self._model_name = f"{model_id}/{gguf_file}"

    def _load_transformers(self) -> None:
        """Load model via transformers + torch (heavier backend)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self._model_name
        # If user has the GGUF model name, switch to HF equivalent.
        if "gguf" in model_id.lower():
            model_id = _DEFAULT_HF_MODEL

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self._llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=self._device or "auto",
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._backend = "transformers"
        self._model_name = model_id

    def _generate_raw(self, prompt: str) -> str:
        """Run the model and return the raw decoded output."""
        self._ensure_loaded()
        try:
            if self._backend == "llama_cpp":
                return self._generate_llama_cpp(prompt)
            return self._generate_transformers(prompt)
        except Exception as exc:
            raise RAGError(f"SLM generation failed: {exc}") from exc

    def _generate_llama_cpp(self, prompt: str) -> str:
        """Generate using llama-cpp-python."""
        messages = [
            {"role": "system", "content": build_system_instruction()},
            {"role": "user", "content": prompt},
        ]
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self._max_new_tokens,
            temperature=self._temperature,
            repeat_penalty=self._repetition_penalty,
            stream=False,
        )
        return response["choices"][0]["message"]["content"].strip()

    def _generate_transformers(self, prompt: str) -> str:
        """Generate using transformers + torch."""
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
        elif hasattr(self._llm, "device"):
            inputs = {k: v.to(self._llm.device) for k, v in inputs.items()}

        do_sample = self._temperature > 0.0
        output_ids = self._llm.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens,
            do_sample=do_sample,
            temperature=self._temperature if do_sample else 1.0,
            repetition_penalty=self._repetition_penalty,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Post-processing: citation enforcement, grounding, trimming
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_evidence_passages(prompt: str) -> list[str]:
        return [m.group("text").strip() for m in _CONTEXT_LINE_RE.finditer(prompt)]

    @staticmethod
    def _extract_question(prompt: str) -> str:
        m = _QUESTION_RE.search(prompt)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _is_refusal(text: str) -> bool:
        return bool(_REFUSAL_RE.search(text))

    @staticmethod
    def _has_citation(text: str) -> bool:
        return bool(_CITATION_RE.search(text))

    @staticmethod
    def _is_grounded(answer: str, evidence: list[str]) -> bool:
        if not evidence:
            return False
        answer_clean = _CITATION_RE.sub("", answer)
        answer_tokens = set(re.findall(r"[\wঀ-৿]+", answer_clean.lower()))
        answer_tokens = {t for t in answer_tokens if not t.isdigit()}
        if len(answer_tokens) < 2:
            if not answer_tokens:
                return True
            for passage in evidence:
                passage_tokens = set(re.findall(r"[\wঀ-৿]+", passage.lower()))
                if answer_tokens & passage_tokens:
                    return True
            return False
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
            return True
        for passage in evidence:
            passage_tokens = set(re.findall(r"[\wঀ-৿]+", passage.lower()))
            overlap = answer_content & passage_tokens
            if len(overlap) >= 2:
                return True
        if len(answer_content) == 1:
            for passage in evidence:
                passage_tokens = set(re.findall(r"[\wঀ-৿]+", passage.lower()))
                if answer_content & passage_tokens:
                    return True
        return False

    @staticmethod
    def _trim_verbose(text: str) -> str:
        if len(text) <= 300:
            return text
        sentences = re.split(r"(?<=[।.!?])\s+", text)
        if len(sentences) <= 1:
            return text
        kept: list[str] = []
        for s in sentences[:5]:
            kept.append(s)
            if _CITATION_RE.search(s):
                break
            if len(kept) >= 3:
                break
        result = " ".join(kept).strip()
        if not _CITATION_RE.search(result) and _CITATION_RE.search(text):
            m = _CITATION_RE.search(text)
            if m:
                result = f"{result} {m.group(0)}"
        return result

    @staticmethod
    def _ensure_citation(text: str) -> str:
        if _CITATION_RE.search(text):
            return text
        return f"{text} [1]"

    def _extractive_fallback(self, prompt: str) -> str:
        from semfuse.rag.template import TemplateLLMProvider

        return TemplateLLMProvider().generate(prompt)

    def generate(self, prompt: str) -> str:
        """Generate an answer with evidence grounding and citation enforcement."""
        raw = self._generate_raw(prompt)

        if self._is_refusal(raw):
            return raw

        trimmed = self._trim_verbose(raw)
        cited = self._ensure_citation(trimmed)

        evidence = self._extract_evidence_passages(prompt)
        if evidence and not self._is_grounded(cited, evidence):
            return self._extractive_fallback(prompt)

        return cited
