"""Deterministic hashing embedding provider.

This provider is NOT a true semantic embedder. It maps text to a fixed-size
vector using character n-gram feature hashing. It is:

* deterministic (same input -> same output, always)
* dependency-free (numpy only)
* offline (no model download)
* fast — uses zlib.crc32 for hashing (10x faster than hashlib.md5)

It exists primarily so that unit tests can exercise the full retrieval,
persistence, and deduplication pipeline without internet access or a model
download. It also serves as a fallback when ``sentence-transformers`` is not
available.

Because it shares character n-grams, queries that share substrings with stored
documents (e.g. "capital" in both) will score higher — enough to make simple
unit tests meaningful — but it cannot bridge scripts (Bangla <-> English).
"""

from __future__ import annotations

import re
import zlib

import numpy as np

from semfuse.core.exceptions import EmbeddingError

_UNIGRAM_RE = re.compile(r"\w+", re.UNICODE)


class HashingEmbeddingProvider:
    """Deterministic character n-gram hashing embedder.

    Uses zlib.crc32 for fast hashing (~10x faster than hashlib.md5) with a
    fixed seed for determinism. Vectors are preallocated and filled in-place
    to avoid the overhead of np.vstack in a loop.
    """

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

    def _embed_one_into(self, text: str, vec: np.ndarray) -> None:
        """Embed text into a preallocated vector (in-place, no allocation)."""
        vec[:] = 0.0
        if not text:
            return
        dim = self._dimension
        tokens = _UNIGRAM_RE.findall(text.lower())
        for token in tokens:
            tlen = len(token)
            for n in self._ngram_sizes:
                if tlen < n:
                    # For tokens shorter than n, hash the whole token.
                    gram = token
                    h = zlib.crc32(gram.encode("utf-8")) & 0xFFFFFFFF
                    idx = h % dim
                    vec[idx] += 1.0 if (h & 1) else -1.0
                    continue
                for i in range(tlen - n + 1):
                    gram = token[i : i + n]
                    h = zlib.crc32(gram.encode("utf-8")) & 0xFFFFFFFF
                    idx = h % dim
                    vec[idx] += 1.0 if (h & 1) else -1.0
        # L2 normalize so cosine/dot behave sensibly.
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype=np.float32)
        self._embed_one_into(text, vec)
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        try:
            # Preallocate the full matrix and fill in-place — avoids np.vstack
            # overhead which allocates a new array each iteration.
            result = np.zeros((len(texts), self._dimension), dtype=np.float32)
            for i, text in enumerate(texts):
                self._embed_one_into(text, result[i])
            return result
        except Exception as exc:  # pragma: no cover - defensive
            raise EmbeddingError(f"Hashing embedding failed: {exc}") from exc

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)
