"""Configuration for SemFuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from semfuse.core.enums import SearchMode, SimilarityMetric

# Centralized defaults — change here, not scattered through the codebase.
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIMENSION = 384
EMBEDDING_VERSION = "1"
INDEX_VERSION = "1"
DEFAULT_STORAGE_DIR = ".semfuse"
DEFAULT_COLLECTION = "default"


@dataclass
class SemFuseConfig:
    """User-facing configuration.

    Keep the default path zero-config: ``SemFuse()`` uses these defaults.
    """

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    # Provider key: "local" (sentence-transformers), "hashing" (deterministic, offline).
    embedding_provider: str = "local"
    vector_store: str = "local"
    storage_path: str | Path = DEFAULT_STORAGE_DIR
    collection: str = DEFAULT_COLLECTION
    metric: SimilarityMetric = SimilarityMetric.COSINE
    search_mode: SearchMode = SearchMode.AUTO
    top_k: int = 5
    score_threshold: float = 0.0
    chunk_size: int = 500
    chunk_overlap: int = 50
    # When True, the embedding model is loaded on first use (not at construction).
    lazy: bool = True
    # Device hint for backends that support it: "cpu", "cuda", "mps", or None for auto.
    device: str | None = None
    # Extra provider-specific options forwarded to the embedding provider.
    embedding_options: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.storage_path = Path(self.storage_path)
