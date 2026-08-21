"""Embedding provider factory."""

from __future__ import annotations

from typing import Any

from semfuse.core.config import SemFuseConfig
from semfuse.core.exceptions import ConfigurationError
from semfuse.embeddings.base import EmbeddingProvider
from semfuse.embeddings.hashing import HashingEmbeddingProvider
from semfuse.embeddings.local import LocalEmbeddingProvider


def create_embedding_provider(config: SemFuseConfig, **overrides: Any) -> EmbeddingProvider:
    """Build an embedding provider from configuration.

    Supported provider keys:
      * ``local``    -> sentence-transformers (default, lazy-loaded)
      * ``hashing``  -> deterministic offline hashing embedder
    """
    key = config.embedding_provider
    if key == "local":
        return LocalEmbeddingProvider(
            model_name=config.embedding_model,
            dimension=config.embedding_dimension,
            device=config.device,
            **{**config.embedding_options, **overrides},
        )
    if key == "hashing":
        return HashingEmbeddingProvider(
            dimension=config.embedding_dimension,
            model_name=config.embedding_model or "hashing-ngram",
            **{**config.embedding_options, **overrides},
        )
    raise ConfigurationError(
        f"Unknown embedding_provider {key!r}. Supported: 'local', 'hashing'."
    )
