"""Enumerations for SemFuse."""

from __future__ import annotations

from enum import Enum


class Language(str, Enum):
    """Detected language category for a piece of text."""

    EN = "en"
    BN = "bn"
    BANGLISH = "banglish"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SearchMode(str, Enum):
    """Retrieval strategy."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    AUTO = "auto"


class SimilarityMetric(str, Enum):
    """Vector similarity metric."""

    COSINE = "cosine"
    DOT = "dot"
    EUCLIDEAN = "euclidean"


class FusionMethod(str, Enum):
    """Score fusion method for hybrid retrieval."""

    WEIGHTED = "weighted"
    RRF = "rrf"
