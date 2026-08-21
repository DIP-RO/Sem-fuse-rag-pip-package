"""Local embedding provider backed by ``sentence-transformers``.

The model is loaded lazily on first use and reused for all subsequent calls.
CPU is used by default; GPU is used when available and supported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from semfuse.core.exceptions import ModelLoadError
from semfuse.utils.logging import get_logger

logger = get_logger(__name__)


class LocalEmbeddingProvider:
    """Sentence-transformers backed embedding provider with lazy loading."""

    def __init__(
        self,
        model_name: str,
        dimension: int,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._device = device
        self._kwargs = kwargs
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model %s ...", self._model_name)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - import guard
            raise ModelLoadError(
                "sentence-transformers is required for the local embedding "
                "provider. Install it with `pip install sentence-transformers` "
                "or use embedding_provider='hashing'."
            ) from exc

        try:
            self._model = SentenceTransformer(self._model_name, device=self._device)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load embedding model {self._model_name!r}: {exc}. "
                "If this is a network issue, ensure the model can be downloaded "
                "from HuggingFace, or pre-cache it, or use "
                "embedding_provider='hashing' for offline use."
            ) from exc

        # Reconcile the true dimension if the model exposes it.
        # Newer sentence-transformers renamed the method; support both.
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension", None
        )
        actual = dim_fn() if dim_fn is not None else None
        if actual is not None and actual != self._dimension:
            logger.warning(
                "Configured dimension %d differs from model dimension %d; "
                "using model dimension.",
                self._dimension,
                actual,
            )
            self._dimension = int(actual)
        logger.info("Embedding model loaded (dim=%d).", self._dimension)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        try:
            vecs = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:
            raise ModelLoadError(f"Embedding generation failed: {exc}") from exc
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        try:
            vec = self._model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:
            raise ModelLoadError(f"Query embedding failed: {exc}") from exc
        return np.asarray(vec[0], dtype=np.float32)
