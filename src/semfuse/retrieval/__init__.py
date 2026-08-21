"""semfuse.retrieval subpackage."""

from __future__ import annotations

from semfuse.retrieval.base import Retriever
from semfuse.retrieval.fusion import fuse_results
from semfuse.retrieval.hybrid import HybridRetriever
from semfuse.retrieval.keyword import KeywordRetriever
from semfuse.retrieval.semantic import SemanticRetriever

__all__ = [
    "HybridRetriever",
    "KeywordRetriever",
    "Retriever",
    "SemanticRetriever",
    "fuse_results",
]
