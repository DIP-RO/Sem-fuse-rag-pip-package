"""Local persistent vector store backed by numpy.

Persistence layout under ``storage_path``::

    <storage_path>/
        vectors.npy            # float32 array (n, dim)
        chunks.json            # list of chunk records
        index_info.json        # embedding model/dim/version + index version + metric

The store deduplicates chunks by ``content_hash``: adding a chunk whose hash
already exists is a no-op.

Performance notes:
  * Vectors are stored in a pre-allocated growable buffer (capacity doubles
    when exhausted), so ``add`` / ``add_many`` are amortized O(1) per chunk
    instead of O(n) per ``np.vstack``.
  * Search uses ``np.argpartition`` (O(n) average) to isolate the top-k
    candidates, then sorts only those k (O(k log k)) — not a full sort.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from semfuse.core.config import EMBEDDING_VERSION, INDEX_VERSION
from semfuse.core.enums import Language, SimilarityMetric
from semfuse.core.exceptions import IndexVersionError, VectorStoreError
from semfuse.core.types import DocumentChunk, IndexInfo, SearchResult
from semfuse.utils.logging import get_logger
from semfuse.utils.serialization import dump_json, load_json, parse_datetime

logger = get_logger(__name__)

_VECTORS_FILE = "vectors.npy"
_CHUNKS_FILE = "chunks.json"
_INDEX_INFO_FILE = "index_info.json"

# Initial buffer capacity and growth factor for the vector matrix.
_INITIAL_CAPACITY = 64
_GROWTH_FACTOR = 2


def _chunk_to_record(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "normalized_text": chunk.normalized_text,
        "language": chunk.language.value,
        "source": chunk.source,
        "title": chunk.title,
        "page": chunk.page,
        "metadata": chunk.metadata,
        "chunk_index": chunk.chunk_index,
        "content_hash": chunk.content_hash,
    }


def _record_to_chunk(rec: dict[str, Any]) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=rec["chunk_id"],
        document_id=rec["document_id"],
        text=rec["text"],
        normalized_text=rec.get("normalized_text", rec["text"]),
        language=Language(rec.get("language", "unknown")),
        source=rec.get("source", "user"),
        title=rec.get("title"),
        page=rec.get("page"),
        metadata=rec.get("metadata", {}),
        chunk_index=rec.get("chunk_index", 0),
        content_hash=rec.get("content_hash", ""),
    )


class LocalVectorStore:
    """In-memory numpy vector store with file persistence."""

    def __init__(
        self,
        storage_path: str | Path,
        embedding_model: str,
        embedding_dimension: int,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        collection: str = "default",
    ) -> None:
        self._storage_path = Path(storage_path) / collection
        self._embedding_model = embedding_model
        self._embedding_dimension = embedding_dimension
        self._metric = metric
        self._collection = collection

        # Growable buffer: _buf has capacity >= _count; only [:_count] is valid.
        self._capacity = _INITIAL_CAPACITY
        self._buf = np.zeros((_INITIAL_CAPACITY, embedding_dimension), dtype=np.float32)
        self._count = 0
        self._chunks: list[DocumentChunk] = []
        self._hash_to_idx: dict[str, int] = {}
        # Cached norms for cosine similarity — invalidated on add/load.
        self._norms_cache: np.ndarray | None = None
        self._index_info: IndexInfo = IndexInfo(
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            embedding_version=EMBEDDING_VERSION,
            index_version=INDEX_VERSION,
            metric=metric.value,
        )

    # ------------------------------------------------------------------ helpers
    @property
    def storage_path(self) -> Path:
        return self._storage_path

    @property
    def _vectors(self) -> np.ndarray:
        """View of the valid portion of the buffer (shape (count, dim))."""
        return self._buf[: self._count]

    def _ensure_capacity(self, needed: int) -> None:
        """Grow the buffer if ``needed`` rows won't fit."""
        if needed <= self._capacity:
            return
        new_cap = self._capacity
        while new_cap < needed:
            new_cap *= _GROWTH_FACTOR
        new_buf = np.zeros((new_cap, self._embedding_dimension), dtype=np.float32)
        new_buf[: self._count] = self._buf[: self._count]
        self._buf = new_buf
        self._capacity = new_cap
        # Norms cache is invalid after buffer reallocation.
        self._norms_cache = None

    def _score(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        if self._metric == SimilarityMetric.COSINE:
            q = query / (np.linalg.norm(query) + 1e-12)
            # Use cached norms if available — avoids recomputing on every search.
            if self._norms_cache is None or self._norms_cache.shape[0] != matrix.shape[0]:
                self._norms_cache = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
            m = matrix / self._norms_cache
            return m @ q
        if self._metric == SimilarityMetric.DOT:
            return matrix @ query
        if self._metric == SimilarityMetric.EUCLIDEAN:
            # Convert distance to similarity (higher is better): 1 / (1 + dist)
            diff = matrix - query
            dist = np.linalg.norm(diff, axis=1)
            return 1.0 / (1.0 + dist)
        raise VectorStoreError(f"Unsupported metric: {self._metric}")

    def _matches_filter(self, metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
        for key, value in filter.items():
            if metadata.get(key) != value:
                return False
        return True

    # ------------------------------------------------------------------ writes
    def add(self, chunk: DocumentChunk, vector: np.ndarray) -> bool:
        """Add a chunk. Returns False (no-op) if the content hash already exists."""
        if chunk.content_hash and chunk.content_hash in self._hash_to_idx:
            logger.debug("Skipping duplicate chunk (hash=%s).", chunk.content_hash[:12])
            return False
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self._embedding_dimension:
            raise VectorStoreError(
                f"Vector dimension {vec.shape[0]} != store dimension "
                f"{self._embedding_dimension}."
            )
        self._ensure_capacity(self._count + 1)
        self._buf[self._count] = vec
        idx = self._count
        self._count += 1
        self._chunks.append(chunk)
        if chunk.content_hash:
            self._hash_to_idx[chunk.content_hash] = idx
        self._norms_cache = None
        return True

    def add_many(self, chunks: list[DocumentChunk], vectors: np.ndarray) -> int:
        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError("chunks and vectors length mismatch")
        # First pass: filter out duplicates (both existing and within-batch).
        new_chunks: list[DocumentChunk] = []
        new_vecs: list[np.ndarray] = []
        seen_hashes: set[str] = set()
        for chunk, vec in zip(chunks, vectors, strict=True):
            if chunk.content_hash:
                if chunk.content_hash in self._hash_to_idx or chunk.content_hash in seen_hashes:
                    logger.debug("Skipping duplicate chunk (hash=%s).", chunk.content_hash[:12])
                    continue
                seen_hashes.add(chunk.content_hash)
            v = np.asarray(vec, dtype=np.float32).reshape(-1)
            if v.shape[0] != self._embedding_dimension:
                raise VectorStoreError(
                    f"Vector dimension {v.shape[0]} != store dimension "
                    f"{self._embedding_dimension}."
                )
            new_chunks.append(chunk)
            new_vecs.append(v)
        if not new_chunks:
            return 0
        # Batch-allocate and copy in one shot.
        self._ensure_capacity(self._count + len(new_chunks))
        for i, v in enumerate(new_vecs):
            self._buf[self._count + i] = v
        for i, chunk in enumerate(new_chunks):
            idx = self._count + i
            self._chunks.append(chunk)
            if chunk.content_hash:
                self._hash_to_idx[chunk.content_hash] = idx
        self._count += len(new_chunks)
        self._norms_cache = None
        return len(new_chunks)

    def delete(self, chunk_id: str) -> None:
        # Find and remove the chunk, then compact the buffer.
        target_idx = None
        for i, c in enumerate(self._chunks):
            if c.chunk_id == chunk_id:
                target_idx = i
                break
        if target_idx is None:
            return
        # Shift remaining vectors down by one.
        if target_idx < self._count - 1:
            self._buf[target_idx : self._count - 1] = self._buf[target_idx + 1 : self._count]
        self._count -= 1
        self._chunks.pop(target_idx)
        # Rebuild hash index (indices shifted).
        self._hash_to_idx = {
            c.content_hash: i for i, c in enumerate(self._chunks) if c.content_hash
        }

    def clear(self) -> None:
        self._capacity = _INITIAL_CAPACITY
        self._buf = np.zeros((_INITIAL_CAPACITY, self._embedding_dimension), dtype=np.float32)
        self._count = 0
        self._chunks = []
        self._hash_to_idx = {}
        self._norms_cache = None

    def count(self) -> int:
        return self._count

    def chunks(self) -> list[DocumentChunk]:
        """All stored chunks, in insertion order (copy of the internal list)."""
        return list(self._chunks)

    # ------------------------------------------------------------------ search
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        if self._count == 0:
            return []
        matrix = self._vectors
        scores = self._score(np.asarray(query_vector, dtype=np.float32), matrix)

        if filter:
            # With a filter we can't use argpartition directly — we may need to
            # scan past non-matching candidates.  Over-fetch partitions and then
            # walk sorted order until we have enough matching results.
            return self._search_filtered(scores, top_k, dict(filter))

        # No filter: use argpartition for O(n) top-k selection.
        k = min(top_k, self._count)
        if k <= 0:
            return []
        if k == 1:
            best = int(np.argmax(scores))
            top_indices: list[int] = [best]
        else:
            # argpartition puts the k largest at the front (unordered).
            part = np.argpartition(-scores, k - 1)[:k]
            # Sort just the k candidates by score descending.
            sorted_part = part[np.argsort(-scores[part])]
            top_indices = [int(i) for i in sorted_part]

        return [
            self._make_result(idx, float(scores[idx]))
            for idx in top_indices
        ]

    def _search_filtered(
        self, scores: np.ndarray, top_k: int, filter: dict[str, Any]
    ) -> list[SearchResult]:
        """Search with metadata filter — full sort then walk until enough match."""
        order = np.argsort(-scores)
        results: list[SearchResult] = []
        for idx in order:
            chunk = self._chunks[idx]
            if not self._matches_filter(chunk.metadata, filter):
                continue
            results.append(self._make_result(int(idx), float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def _make_result(self, idx: int, score: float) -> SearchResult:
        chunk = self._chunks[idx]
        return SearchResult(
            text=chunk.text,
            score=score,
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            metadata=dict(chunk.metadata),
            language=chunk.language,
            source=chunk.source,
            page=chunk.page,
        )

    # ------------------------------------------------------------------ persist
    def index_info(self) -> IndexInfo:
        return self._index_info

    def _index_info_path(self) -> Path:
        return self._storage_path / _INDEX_INFO_FILE

    def persist(self) -> None:
        self._storage_path.mkdir(parents=True, exist_ok=True)
        # Save only the valid portion, not the full buffer.
        np.save(self._storage_path / _VECTORS_FILE, self._vectors)
        dump_json([_chunk_to_record(c) for c in self._chunks], self._storage_path / _CHUNKS_FILE)
        dump_json(
            {
                "embedding_model": self._index_info.embedding_model,
                "embedding_dimension": self._index_info.embedding_dimension,
                "embedding_version": self._index_info.embedding_version,
                "index_version": self._index_info.index_version,
                "metric": self._index_info.metric,
                "created_at": self._index_info.created_at,
            },
            self._index_info_path(),
        )
        logger.debug("Persisted store (%d chunks) to %s.", self._count, self._storage_path)

    def load(self) -> None:
        info_path = self._index_info_path()
        if not info_path.exists():
            # Empty store: nothing to load. Keep configured index info.
            return
        raw = load_json(info_path)
        loaded_info = IndexInfo(
            embedding_model=raw["embedding_model"],
            embedding_dimension=raw["embedding_dimension"],
            embedding_version=raw.get("embedding_version", "0"),
            index_version=raw.get("index_version", "0"),
            metric=raw.get("metric", self._metric.value),
            created_at=parse_datetime(raw.get("created_at")) or self._index_info.created_at,
        )
        self._check_compatible(loaded_info)
        self._index_info = loaded_info

        vectors_path = self._storage_path / _VECTORS_FILE
        if vectors_path.exists():
            loaded = np.load(vectors_path).astype(np.float32)
            self._count = loaded.shape[0]
            self._capacity = max(_INITIAL_CAPACITY, self._count)
            self._buf = np.zeros((self._capacity, self._embedding_dimension), dtype=np.float32)
            if self._count > 0:
                self._buf[: self._count] = loaded
        else:
            self._count = 0
            self._capacity = _INITIAL_CAPACITY
            self._buf = np.zeros((_INITIAL_CAPACITY, self._embedding_dimension), dtype=np.float32)

        chunks_raw = load_json(self._storage_path / _CHUNKS_FILE)
        self._chunks = [_record_to_chunk(r) for r in chunks_raw]
        self._hash_to_idx = {c.content_hash: i for i, c in enumerate(self._chunks) if c.content_hash}
        self._norms_cache = None
        logger.info("Loaded store (%d chunks) from %s.", self._count, self._storage_path)

    def _check_compatible(self, loaded: IndexInfo) -> None:
        if (
            loaded.embedding_model != self._embedding_model
            or loaded.embedding_dimension != self._embedding_dimension
        ):
            raise IndexVersionError(
                "Existing index is incompatible with the configured embedding "
                f"provider.\n"
                f"  Index on disk: model={loaded.embedding_model!r}, "
                f"dim={loaded.embedding_dimension}\n"
                f"  Active config: model={self._embedding_model!r}, "
                f"dim={self._embedding_dimension}\n"
                f"To fix: use the same embedding model/dimension, or delete "
                f"{self._storage_path} and reindex."
            )
