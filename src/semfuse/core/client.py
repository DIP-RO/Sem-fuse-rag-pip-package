"""Public SemFuse client.

This is the single entry point users interact with::

    from semfuse import SemFuse

    db = SemFuse()
    db.add("ঢাকা বাংলাদেশের রাজধানী।")
    db.search("Bangladesh er capital ki?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from semfuse.core.config import (
    SemFuseConfig,
)
from semfuse.core.enums import SearchMode, SimilarityMetric
from semfuse.core.exceptions import ConfigurationError
from semfuse.core.types import (
    CollectionInfo,
    Document,
    DocumentChunk,
    SearchResult,
)
from semfuse.embeddings.base import EmbeddingProvider
from semfuse.embeddings.factory import create_embedding_provider
from semfuse.language.detector import detect_language
from semfuse.language.normalizer import normalize_text
from semfuse.retrieval.semantic import SemanticRetriever
from semfuse.utils.hashing import content_hash, short_hash
from semfuse.utils.logging import get_logger
from semfuse.vectorstores.local import LocalVectorStore

logger = get_logger(__name__)

__version__ = "0.1.0"


class SemFuse:
    """The public SemFuse retrieval client."""

    def __init__(
        self,
        *,
        embedding_model: str | None = None,
        embedding_provider: str | None = None,
        vector_store: str | None = None,
        storage_path: str | Path | None = None,
        collection: str | None = None,
        metric: str | SimilarityMetric | None = None,
        search_mode: str | SearchMode | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        device: str | None = None,
        embedding_options: dict[str, object] | None = None,
        config: SemFuseConfig | None = None,
    ) -> None:
        if config is not None:
            cfg = config
        else:
            cfg = SemFuseConfig()

        # Apply overrides without mutating the default dataclass in surprising ways.
        if embedding_model is not None:
            cfg.embedding_model = embedding_model
        if embedding_provider is not None:
            cfg.embedding_provider = embedding_provider
        if vector_store is not None:
            cfg.vector_store = vector_store
        if storage_path is not None:
            cfg.storage_path = Path(storage_path)
        if collection is not None:
            cfg.collection = collection
        if metric is not None:
            cfg.metric = SimilarityMetric(metric) if isinstance(metric, str) else metric
        if search_mode is not None:
            cfg.search_mode = SearchMode(search_mode) if isinstance(search_mode, str) else search_mode
        if top_k is not None:
            cfg.top_k = top_k
        if score_threshold is not None:
            cfg.score_threshold = score_threshold
        if chunk_size is not None:
            cfg.chunk_size = chunk_size
        if chunk_overlap is not None:
            cfg.chunk_overlap = chunk_overlap
        if device is not None:
            cfg.device = device
        if embedding_options is not None:
            cfg.embedding_options = dict(embedding_options)

        # Re-validate after overrides.
        cfg.__post_init__()
        self._config = cfg

        if cfg.vector_store != "local":
            raise ConfigurationError(
                f"vector_store={cfg.vector_store!r} is not implemented in Phase 1. "
                "Use 'local'."
            )

        self._embeddings: EmbeddingProvider = create_embedding_provider(cfg)
        self._store = LocalVectorStore(
            storage_path=cfg.storage_path,
            embedding_model=self._embeddings.model_name,
            embedding_dimension=self._embeddings.dimension,
            metric=cfg.metric,
            collection=cfg.collection,
        )
        self._retriever = SemanticRetriever(self._embeddings, self._store)

        # Load any existing persisted index.
        self._store.load()
        logger.info(
            "SemFuse initialized (provider=%s, model=%s, dim=%d, collection=%s, chunks=%d).",
            cfg.embedding_provider,
            self._embeddings.model_name,
            self._embeddings.dimension,
            cfg.collection,
            self._store.count(),
        )

    # ------------------------------------------------------------------ props
    @property
    def config(self) -> SemFuseConfig:
        return self._config

    @property
    def version(self) -> str:
        return __version__

    # ------------------------------------------------------------------ ingest
    def _make_document(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        source: str = "user",
        title: str | None = None,
        page: int | None = None,
        document_id: str | None = None,
    ) -> Document:
        return Document(
            text=text,
            source=source,
            title=title,
            page=page,
            language=detect_language(text),
            metadata=dict(metadata or {}),
            document_id=document_id,
        )

    def _chunk_document(self, doc: Document) -> list[DocumentChunk]:
        # Phase 1: a document is a single chunk. Recursive chunking arrives in Phase 3.
        normalized = normalize_text(doc.text)
        doc_id = doc.document_id or short_hash(doc.text + "|" + doc.source)
        chunk_id = short_hash(doc_id + "|0")
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            text=doc.text,
            normalized_text=normalized,
            language=doc.language,
            source=doc.source,
            title=doc.title,
            page=doc.page,
            metadata={**doc.metadata, "language": doc.language.value},
            chunk_index=0,
            content_hash=content_hash(normalized),
        )
        return [chunk]

    def add(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        source: str = "user",
        title: str | None = None,
        page: int | None = None,
        document_id: str | None = None,
        persist: bool = True,
    ) -> int:
        """Add a single text document. Returns the number of chunks added."""
        if not isinstance(text, str) or not text.strip():
            raise ConfigurationError("text must be a non-empty string")
        doc = self._make_document(
            text, metadata=metadata, source=source, title=title, page=page, document_id=document_id
        )
        return self._index_documents([doc], persist=persist)

    def add_many(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
        source: str = "user",
        persist: bool = True,
    ) -> int:
        """Add many texts. Returns the number of chunks added (post-dedup)."""
        if not texts:
            return 0
        metas = metadata or [{} for _ in texts]
        if len(metas) != len(texts):
            raise ConfigurationError("metadata length must match texts length")
        docs = [self._make_document(t, metadata=m, source=source) for t, m in zip(texts, metas, strict=True)]
        return self._index_documents(docs, persist=persist)

    def _index_documents(self, docs: list[Document], persist: bool) -> int:
        chunks: list[DocumentChunk] = []
        for doc in docs:
            chunks.extend(self._chunk_document(doc))
        if not chunks:
            return 0
        # Embed normalized text (fall back to original if normalization emptied it).
        texts_to_embed = [c.normalized_text or c.text for c in chunks]
        vectors = self._embeddings.embed_documents(texts_to_embed)
        added = self._store.add_many(chunks, vectors)
        if persist:
            self._store.persist()
        logger.info("Indexed %d/%d chunks (dedup skipped %d).", added, len(chunks), len(chunks) - added)
        return added

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter: dict[str, object] | None = None,
        mode: str | SearchMode | None = None,
        include_metadata: bool = True,
    ) -> list[SearchResult]:
        """Search the index.

        Args:
            query: Query text (any supported language).
            top_k: Number of results (defaults to config).
            score_threshold: Minimum score (defaults to config).
            filter: Metadata equality filter, e.g. ``{"department": "CSE"}``.
            mode: Search mode. Phase 1 supports ``semantic``; ``auto``/``hybrid``/
                ``keyword`` resolve to semantic in Phase 1 and are expanded later.
            include_metadata: Whether to populate metadata on results.
        """
        if not isinstance(query, str) or not query.strip():
            raise ConfigurationError("query must be a non-empty string")
        k = top_k if top_k is not None else self._config.top_k
        threshold = score_threshold if score_threshold is not None else self._config.score_threshold
        chosen_mode = self._resolve_mode(mode)
        # Phase 1: all modes route to semantic retrieval.
        _ = chosen_mode  # mode selection layer expands in Phase 4
        results = self._retriever.retrieve(query, top_k=k, filter=filter)
        if threshold > 0:
            results = [r for r in results if r.score >= threshold]
        if not include_metadata:
            results = [
                SearchResult(
                    text=r.text,
                    score=r.score,
                    document_id=r.document_id,
                    chunk_id=r.chunk_id,
                    metadata={},
                    language=r.language,
                    source=r.source,
                    page=r.page,
                )
                for r in results
            ]
        return results

    def _resolve_mode(self, mode: str | SearchMode | None) -> SearchMode:
        if mode is None:
            return self._config.search_mode
        return SearchMode(mode) if isinstance(mode, str) else mode

    # ------------------------------------------------------------------ ops
    def delete(self, chunk_id: str, persist: bool = True) -> None:
        self._store.delete(chunk_id)
        if persist:
            self._store.persist()

    def clear(self) -> None:
        self._store.clear()
        self._store.persist()

    def count(self) -> int:
        return self._store.count()

    def persist(self) -> None:
        self._store.persist()

    # ------------------------------------------------------------------ info
    def info(self) -> dict[str, Any]:
        lang_dist: dict[str, int] = {}
        for chunk in getattr(self._store, "_chunks", []):
            key = chunk.language.value
            lang_dist[key] = lang_dist.get(key, 0) + 1
        info = self._store.index_info()
        return {
            "package_version": __version__,
            "embedding_provider": self._config.embedding_provider,
            "embedding_model": info.embedding_model,
            "embedding_dimension": info.embedding_dimension,
            "embedding_version": info.embedding_version,
            "index_version": info.index_version,
            "vector_backend": self._config.vector_store,
            "metric": info.metric,
            "storage_path": str(self._config.storage_path),
            "collection": self._config.collection,
            "document_count": self._unique_document_count(),
            "chunk_count": self._store.count(),
            "language_distribution": lang_dist,
        }

    def _unique_document_count(self) -> int:
        ids = {c.document_id for c in getattr(self._store, "_chunks", [])}
        return len(ids)

    def collection_info(self) -> CollectionInfo:
        lang_dist: dict[str, int] = {}
        for chunk in getattr(self._store, "_chunks", []):
            key = chunk.language.value
            lang_dist[key] = lang_dist.get(key, 0) + 1
        return CollectionInfo(
            name=self._config.collection,
            document_count=self._unique_document_count(),
            chunk_count=self._store.count(),
            language_distribution=lang_dist,
        )

    def explain(self, query: str) -> dict[str, Any]:
        """Diagnostic view of how a query would be processed."""
        from semfuse.language.normalizer import normalize_text as _norm

        lang = detect_language(query)
        mode = self._resolve_mode(None)
        results = self.search(query, top_k=self._config.top_k)
        return {
            "query": query,
            "detected_language": lang.value,
            "normalized_query": _norm(query),
            "search_mode": mode.value,
            "embedding_provider": self._config.embedding_provider,
            "embedding_model": self._embeddings.model_name,
            "candidate_count": len(results),
            "top_score": results[0].score if results else 0.0,
            "results": [
                {"score": r.score, "language": r.language.value, "text": r.text[:80]} for r in results
            ],
        }

    def close(self) -> None:
        """Persist and release resources."""
        self._store.persist()
