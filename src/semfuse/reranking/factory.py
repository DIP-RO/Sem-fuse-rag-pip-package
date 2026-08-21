"""Reranker factory."""

from __future__ import annotations

from semfuse.core.config import SemFuseConfig
from semfuse.core.exceptions import ConfigurationError
from semfuse.reranking.base import Reranker
from semfuse.reranking.cross_encoder import CrossEncoderReranker
from semfuse.reranking.lexical import LexicalReranker


def create_reranker(config: SemFuseConfig) -> Reranker | None:
    """Build a reranker from configuration.

    Supported keys:
      * ``None``           -> no reranking
      * ``"lexical"``      -> deterministic offline token-overlap reranker
      * ``"cross-encoder"``-> multilingual cross-encoder (lazy-loaded)
    """
    key = config.reranker
    if key is None:
        return None
    if key == "lexical":
        return LexicalReranker()
    if key == "cross-encoder":
        return CrossEncoderReranker(
            model_name=config.reranker_model,
            device=config.device,
        )
    raise ConfigurationError(
        f"Unknown reranker {key!r}. Supported: None, 'lexical', 'cross-encoder'."
    )
