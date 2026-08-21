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

from semfuse.chunking.recursive import RecursiveCharacterChunker
from semfuse.core.config import (
    SemFuseConfig,
)
from semfuse.core.enums import FusionMethod, Language, SearchMode, SimilarityMetric
from semfuse.core.exceptions import ConfigurationError, DocumentLoadError
from semfuse.core.types import (
    CollectionInfo,
    Document,
    DocumentChunk,
    RAGResponse,
    SearchResult,
)
from semfuse.embeddings.base import EmbeddingProvider
from semfuse.embeddings.factory import create_embedding_provider
from semfuse.language.detector import detect_language
from semfuse.language.normalizer import normalize_for_search
from semfuse.loaders.factory import SUPPORTED_EXTENSIONS, load_document
from semfuse.rag.factory import create_llm_provider
from semfuse.rag.pipeline import RAGPipeline
from semfuse.reranking.base import Reranker
from semfuse.reranking.factory import create_reranker
from semfuse.reranking.lexical import LexicalReranker
from semfuse.retrieval.hybrid import HybridRetriever
from semfuse.retrieval.keyword import KeywordRetriever
from semfuse.retrieval.semantic import SemanticRetriever
from semfuse.utils.hashing import content_hash, short_hash
from semfuse.utils.logging import get_logger
from semfuse.vectorstores.local import LocalVectorStore

logger = get_logger(__name__)

__version__ = "0.2.0"

_INDEX_INFO_FILE = "index_info.json"


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
        fusion_method: str | FusionMethod | None = None,
        reranker: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
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
        if fusion_method is not None:
            cfg.fusion_method = (
                FusionMethod(fusion_method) if isinstance(fusion_method, str) else fusion_method
            )
        if reranker is not None:
            cfg.reranker = reranker
        if llm_provider is not None:
            cfg.llm_provider = llm_provider
        if llm_model is not None:
            cfg.llm_model = llm_model
        if device is not None:
            cfg.device = device
        if embedding_options is not None:
            cfg.embedding_options = dict(embedding_options)

        # Re-validate after overrides.
        cfg.__post_init__()
        self._config = cfg

        if cfg.vector_store != "local":
            raise ConfigurationError(
                f"vector_store={cfg.vector_store!r} is not implemented yet. "
                "Use 'local'. (FAISS/Qdrant backends are planned extras.)"
            )

        self._embeddings: EmbeddingProvider = create_embedding_provider(cfg)
        self._store = LocalVectorStore(
            storage_path=cfg.storage_path,
            embedding_model=self._embeddings.model_name,
            embedding_dimension=self._embeddings.dimension,
            metric=cfg.metric,
            collection=cfg.collection,
        )
        self._chunker = RecursiveCharacterChunker(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
        self._retriever = SemanticRetriever(self._embeddings, self._store)
        self._keyword_retriever = KeywordRetriever(self._store)
        self._hybrid_retriever = HybridRetriever(
            self._retriever,
            self._keyword_retriever,
            method=cfg.fusion_method,
            semantic_weight=cfg.semantic_weight,
            keyword_weight=cfg.keyword_weight,
        )
        self._reranker: Reranker | None = create_reranker(cfg)
        self._llm = create_llm_provider(cfg)

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
        doc_id = doc.document_id or short_hash(doc.text + "|" + doc.source)
        chunks: list[DocumentChunk] = []
        for index, piece in enumerate(self._chunker.split(doc.text)):
            language = detect_language(piece)
            normalized = normalize_for_search(piece, language)
            chunks.append(
                DocumentChunk(
                    chunk_id=short_hash(doc_id + "|" + str(index)),
                    document_id=doc_id,
                    text=piece,
                    normalized_text=normalized,
                    language=language,
                    source=doc.source,
                    title=doc.title,
                    page=doc.page,
                    metadata={**doc.metadata, "language": language.value},
                    chunk_index=index,
                    content_hash=content_hash(normalized),
                )
            )
        return chunks

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

    def add_file(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> int:
        """Load and index a file (TXT/MD/PDF/DOCX). Returns chunks added."""
        docs = load_document(path)
        if not docs:
            logger.warning("No content extracted from %s.", path)
            return 0
        if metadata:
            docs = [
                Document(
                    text=d.text,
                    source=d.source,
                    title=d.title,
                    page=d.page,
                    language=d.language,
                    metadata={**d.metadata, **metadata},
                    document_id=d.document_id,
                )
                for d in docs
            ]
        return self._index_documents(docs, persist=persist)

    def add_directory(
        self,
        path: str | Path,
        recursive: bool = True,
        extensions: tuple[str, ...] | None = None,
        metadata: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> int:
        """Index every supported file under ``path``. Returns chunks added."""
        root = Path(path)
        if not root.is_dir():
            raise DocumentLoadError(f"Not a directory: {root}")
        allowed = tuple(e.lower() for e in (extensions or SUPPORTED_EXTENSIONS))
        pattern = "**/*" if recursive else "*"
        files = sorted(
            p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in allowed
        )
        total = 0
        for file_path in files:
            total += self.add_file(file_path, metadata=metadata, persist=False)
        if persist:
            self._store.persist()
        logger.info("Indexed %d chunks from %d files under %s.", total, len(files), root)
        return total

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
    def _prepare_query(self, query: str) -> tuple[Language, str]:
        """Detect the query language and produce the search-normalized form."""
        language = detect_language(query)
        return language, normalize_for_search(query, language)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter: dict[str, object] | None = None,
        mode: str | SearchMode | None = None,
        rerank: bool | None = None,
        include_metadata: bool = True,
    ) -> list[SearchResult]:
        """Search the index.

        Args:
            query: Query text (any supported language, including Banglish).
            top_k: Number of results (defaults to config).
            score_threshold: Minimum score (defaults to config).
            filter: Metadata equality filter, e.g. ``{"department": "CSE"}``.
            mode: Search mode: ``semantic``, ``keyword``, ``hybrid``, or
                ``auto`` (resolves to hybrid).
            rerank: Force reranking on/off for this call. Defaults to "on if a
                reranker is configured". ``rerank=True`` without a configured
                reranker uses the offline lexical reranker.
            include_metadata: Whether to populate metadata on results.
        """
        if not isinstance(query, str) or not query.strip():
            raise ConfigurationError("query must be a non-empty string")
        k = top_k if top_k is not None else self._config.top_k
        threshold = score_threshold if score_threshold is not None else self._config.score_threshold
        chosen_mode = self._resolve_mode(mode)
        _, normalized_query = self._prepare_query(query)

        reranker = self._reranker
        if rerank is True and reranker is None:
            reranker = LexicalReranker()
        rerank_active = reranker is not None and rerank is not False

        fetch_k = max(k, self._config.rerank_candidates) if rerank_active else k
        results = self._retrieve(normalized_query, chosen_mode, fetch_k, filter)
        if rerank_active and reranker is not None:
            results = reranker.rerank(normalized_query, results, top_k=k)
        results = results[:k]
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

    def _retrieve(
        self,
        normalized_query: str,
        mode: SearchMode,
        top_k: int,
        filter: dict[str, object] | None,
    ) -> list[SearchResult]:
        if mode == SearchMode.SEMANTIC:
            return self._retriever.retrieve(normalized_query, top_k=top_k, filter=filter)
        if mode == SearchMode.KEYWORD:
            return self._keyword_retriever.retrieve(normalized_query, top_k=top_k, filter=filter)
        # HYBRID (and AUTO, resolved earlier) fuse both retrievers.
        return self._hybrid_retriever.retrieve(normalized_query, top_k=top_k, filter=filter)

    def _resolve_mode(self, mode: str | SearchMode | None) -> SearchMode:
        if mode is None:
            resolved = self._config.search_mode
        else:
            resolved = SearchMode(mode) if isinstance(mode, str) else mode
        # AUTO fuses semantic and keyword evidence; on corpora where keyword
        # matching finds nothing, fusion degrades gracefully to semantic-only.
        return SearchMode.HYBRID if resolved == SearchMode.AUTO else resolved

    # ------------------------------------------------------------------ rag
    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filter: dict[str, object] | None = None,
        mode: str | SearchMode | None = None,
    ) -> RAGResponse:
        """Answer ``question`` from the index with numbered citations.

        Uses the configured ``llm_provider`` — the default ``template``
        provider is extractive (offline, no API key); configure
        ``llm_provider="openai"`` for generative answers.
        """
        pipeline = RAGPipeline(
            retrieve=lambda q: self.search(q, top_k=top_k, filter=filter, mode=mode),
            llm=self._llm,
        )
        return pipeline.ask(question)

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
    def _language_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for chunk in self._store.chunks():
            key = chunk.language.value
            dist[key] = dist.get(key, 0) + 1
        return dist

    def info(self) -> dict[str, Any]:
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
            "search_mode": self._config.search_mode.value,
            "fusion_method": self._config.fusion_method.value,
            "reranker": self._config.reranker,
            "llm_provider": self._config.llm_provider,
            "storage_path": str(self._config.storage_path),
            "collection": self._config.collection,
            "document_count": self._unique_document_count(),
            "chunk_count": self._store.count(),
            "language_distribution": self._language_distribution(),
        }

    def _unique_document_count(self) -> int:
        return len({c.document_id for c in self._store.chunks()})

    def collection_info(self) -> CollectionInfo:
        return CollectionInfo(
            name=self._config.collection,
            document_count=self._unique_document_count(),
            chunk_count=self._store.count(),
            language_distribution=self._language_distribution(),
        )

    def list_collections(self) -> list[str]:
        """Names of collections persisted under this client's storage path."""
        root = Path(self._config.storage_path)
        if not root.is_dir():
            return []
        return sorted(
            p.name for p in root.iterdir() if p.is_dir() and (p / _INDEX_INFO_FILE).exists()
        )

    def explain(self, query: str) -> dict[str, Any]:
        """Diagnostic view of how a query would be processed."""
        language, normalized_query = self._prepare_query(query)
        mode = self._resolve_mode(None)
        results = self.search(query, top_k=self._config.top_k)
        return {
            "query": query,
            "detected_language": language.value,
            "normalized_query": normalized_query,
            "search_mode": mode.value,
            "fusion_method": self._config.fusion_method.value,
            "reranker": self._config.reranker,
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
