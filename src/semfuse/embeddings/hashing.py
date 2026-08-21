"""Deterministic hashing embedding provider.

This provider is NOT a true semantic embedder. It maps text to a fixed-size
vector using character n-gram feature hashing. It is:

* deterministic (same input -> same output, always)
* dependency-free (numpy only)
* offline (no model download)
* fast

It exists primarily so that unit tests can exercise the full retrieval,
persistence, and deduplication pipeline without internet access or a model
download. It also serves as a fallback when ``sentence-transformers`` is not
available.

Because it shares character n-grams, queries that share substrings with stored
documents (e.g. "capital" in both) will score higher — enough to make simple
unit tests meaningful — but it cannot bridge scripts (Bangla <-> English).
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from semfuse.core.exceptions import EmbeddingError

_UNIGRAM_RE = re.compile(r"\w+", re.UNICODE)


class HashingEmbeddingProvider:
    """Deterministic character n-gram hashing embedder."""

    def __init__(
        self,
        dimension: int = 384,
        ngram_sizes: tuple[int, ...] = (2, 3, 4),
        model_name: str = "hashing-ngram",
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._ngram_sizes = ngram_sizes
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        if not text:
            return vec
        # Tokenize into unicode word tokens, then build char n-grams within tokens.
        tokens = _UNIGRAM_RE.findall(text.lower())
        for token in tokens:
            for n in self._ngram_sizes:
                for i in range(max(1, len(token) - n + 1)):
                    gram = token[i : i + n]
                    h = hashlib.md5(gram.encode("utf-8")).digest()
                    idx = int.from_bytes(h[:4], "little") % self._dimension
                    sign = 1.0 if (h[4] & 1) else -1.0
                    vec[idx] += sign
        # L2 normalize so cosine/dot behave sensibly.
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        try:
            return np.vstack([self._embed_one(t) for t in texts])
        except Exception as exc:  # pragma: no cover - defensive
            raise EmbeddingError(f"Hashing embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)
