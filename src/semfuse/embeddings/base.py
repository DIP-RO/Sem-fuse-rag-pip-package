"""Embedding provider protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for embedding generation backends.

    Implementations must be reusable: the model is loaded once and shared across
    calls. Implementations should support batching and CPU by default, using GPU
    when available and supported.
    """

    @property
    def model_name(self) -> str:
        """Stable identifier of the model (used for index compatibility checks)."""
        ...

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced embedding vectors."""
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of documents. Returns shape (len(texts), dimension)."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns shape (dimension,)."""
        ...
