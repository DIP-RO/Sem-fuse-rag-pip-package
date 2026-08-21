"""Configuration for SemFuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from semfuse.core.enums import FusionMethod, SearchMode, SimilarityMetric

# Centralized defaults — change here, not scattered through the codebase.
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_SLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
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
    # Hybrid retrieval: fusion method and per-retriever weights (semantic, keyword).
    # WEIGHTED keeps fused scores comparable to score_threshold; RRF is rank-based.
    fusion_method: FusionMethod = FusionMethod.WEIGHTED
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    # Reranker key: None (off), "lexical" (offline), "cross-encoder" (model-based).
    reranker: str | None = None
    reranker_model: str = DEFAULT_RERANKER_MODEL
    # Candidates fetched for reranking before cutting back to top_k.
    rerank_candidates: int = 25
    # RAG: "template" (extractive, offline, default), "slm" (local SLM,
    # semfuse[slm]), or "openai" (semfuse[rag]).
    llm_provider: str = "template"
    llm_model: str = DEFAULT_LLM_MODEL
    # Extra provider-specific options forwarded to the LLM provider.
    llm_options: dict[str, object] = field(default_factory=dict)
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
        if self.semantic_weight < 0 or self.keyword_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        if self.semantic_weight + self.keyword_weight <= 0:
            raise ValueError("at least one retrieval weight must be positive")
        if self.rerank_candidates <= 0:
            raise ValueError("rerank_candidates must be positive")
        if isinstance(self.fusion_method, str):
            self.fusion_method = FusionMethod(self.fusion_method)
        self.storage_path = Path(self.storage_path)
