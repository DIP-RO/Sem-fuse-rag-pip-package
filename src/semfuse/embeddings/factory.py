"""Embedding provider factory."""

from __future__ import annotations

from typing import Any

from semfuse.core.config import SemFuseConfig
from semfuse.core.exceptions import ConfigurationError
from semfuse.embeddings.base import EmbeddingProvider


def create_embedding_provider(config: SemFuseConfig, **overrides: Any) -> EmbeddingProvider:
    """Build an embedding provider from configuration.

    Supported provider keys:
      * ``local``    -> sentence-transformers (semfuse[embeddings], lazy-loaded)
      * ``hashing``  -> deterministic offline hashing embedder (zero-dep)
    """
    key = config.embedding_provider
    if key == "local":
        from semfuse.embeddings.local import LocalEmbeddingProvider

        return LocalEmbeddingProvider(
            model_name=config.embedding_model,
            dimension=config.embedding_dimension,
            device=config.device,
            **{**config.embedding_options, **overrides},
        )
    if key == "hashing":
        from semfuse.embeddings.hashing import HashingEmbeddingProvider

        return HashingEmbeddingProvider(
            dimension=config.embedding_dimension,
            model_name=config.embedding_model or "hashing-ngram",
            **{**config.embedding_options, **overrides},
        )
    raise ConfigurationError(
        f"Unknown embedding_provider {key!r}. Supported: 'local', 'hashing'."
    )
