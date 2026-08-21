"""Custom exceptions for SemFuse.

Every exception carries an actionable message explaining what failed, why (when
known), and how the developer can fix it.
"""

from __future__ import annotations


class SemFuseError(Exception):
    """Base class for all SemFuse errors."""


class ConfigurationError(SemFuseError):
    """Raised when the provided configuration is invalid or incomplete."""


class ModelLoadError(SemFuseError):
    """Raised when an embedding or reranker model fails to load."""


class UnsupportedLanguageError(SemFuseError):
    """Raised when a language is not supported by the active pipeline."""


class VectorStoreError(SemFuseError):
    """Raised when a vector store operation fails."""


class IndexVersionError(SemFuseError):
    """Raised when an existing index is incompatible with the active embedding provider.

    The message explains how to rebuild/reindex.
    """


class DocumentLoadError(SemFuseError):
    """Raised when a document cannot be loaded or parsed."""


class EmbeddingError(SemFuseError):
    """Raised when embedding generation fails."""


class RetrievalError(SemFuseError):
    """Raised when the retrieval pipeline fails."""


class RerankingError(SemFuseError):
    """Raised when the reranker fails."""


class RAGError(SemFuseError):
    """Raised when the RAG layer fails."""
