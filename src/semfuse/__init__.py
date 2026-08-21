"""SemFuse — lightweight multilingual semantic retrieval with optional RAG."""

from __future__ import annotations

from semfuse.core.client import SemFuse, __version__
from semfuse.core.config import SemFuseConfig
from semfuse.core.enums import FusionMethod, Language, SearchMode, SimilarityMetric
from semfuse.core.exceptions import (
    ConfigurationError,
    DocumentLoadError,
    EmbeddingError,
    IndexVersionError,
    ModelLoadError,
    RAGError,
    RerankingError,
    RetrievalError,
    SemFuseError,
    UnsupportedLanguageError,
    VectorStoreError,
)
from semfuse.core.types import (
    CollectionInfo,
    Document,
    DocumentChunk,
    RAGResponse,
    SearchResult,
)

__all__ = [
    "SemFuse",
    "SemFuseConfig",
    "__version__",
    "Language",
    "SearchMode",
    "SimilarityMetric",
    "FusionMethod",
    "SemFuseError",
    "ConfigurationError",
    "ModelLoadError",
    "UnsupportedLanguageError",
    "VectorStoreError",
    "IndexVersionError",
    "DocumentLoadError",
    "EmbeddingError",
    "RetrievalError",
    "RerankingError",
    "RAGError",
    "Document",
    "DocumentChunk",
    "SearchResult",
    "CollectionInfo",
    "RAGResponse",
]
