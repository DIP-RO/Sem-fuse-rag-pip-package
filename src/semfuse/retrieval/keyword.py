"""BM25 keyword retriever over the vector store's chunk corpus.

Operates on ``normalized_text`` (lowercased unicode word tokens), so Banglish
queries that were transliterated at index/query time match Bangla documents
lexically as well. Scores are scaled to [0, 1] against the query's *ideal*
BM25 score (every term matched at saturation), so a document that matches only
low-IDF stopwords scores near zero rather than being inflated to the top —
this keeps keyword scores composable with semantic scores in hybrid fusion and
meaningful against ``score_threshold``.

The BM25 index is rebuilt lazily whenever the store's chunk count changes;
for the local numpy store this covers every mutation (add/delete/clear/load)
since deduplication prevents same-count replacement.

Performance: an inverted index (``term -> list[(doc_idx, tf)]``) is built at
index time, so scoring only touches documents that contain at least one query
term — O(sum of postings list lengths) instead of O(n * |query_terms|).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from semfuse.core.types import DocumentChunk, SearchResult
from semfuse.vectorstores.base import VectorStore

# \w alone drops combining marks (Bangla vowel signs are category Mn), which
# would shred "ঢাকা" into single consonants — include the Bangla block whole.
_TOKEN_RE = re.compile(r"[\wঀ-৿]+", re.UNICODE)

_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercased unicode word tokens."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class KeywordRetriever:
    """BM25 (Okapi) retrieval over stored chunks, backed by an inverted index."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._indexed_count = -1
        self._chunks: list[DocumentChunk] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        # Inverted index: term -> list of (doc_idx, term_freq).
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._avg_len = 0.0
        self._n = 0

    # ------------------------------------------------------------------ index
    def _ensure_index(self) -> None:
        count = self._store.count()
        if count == self._indexed_count:
            return
        self._chunks = self._store.chunks()
        self._doc_lens = []
        self._df = Counter()
        self._postings = {}
        for i, chunk in enumerate(self._chunks):
            tokens = tokenize(chunk.normalized_text or chunk.text)
            freqs = Counter(tokens)
            self._doc_lens.append(len(tokens))
            for term, tf in freqs.items():
                self._postings.setdefault(term, []).append((i, tf))
            self._df.update(freqs.keys())
        self._n = len(self._chunks)
        self._avg_len = (sum(self._doc_lens) / self._n) if self._n else 0.0
        self._indexed_count = count

    def _bm25_scores(self, query_tokens: list[str]) -> tuple[dict[int, float], float]:
        """Per-doc BM25 scores (only docs that matched) plus the ideal score.

        Returns a sparse dict ``{doc_idx: score}`` instead of a full list so
        the caller only iterates over matching documents.
        """
        scores: dict[int, float] = {}
        ideal = 0.0
        avg_len = self._avg_len or 1.0
        for term in query_tokens:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
            ideal += idf * (_K1 + 1.0)
            for doc_idx, tf in self._postings.get(term, ()):
                denom = tf + _K1 * (1.0 - _B + _B * self._doc_lens[doc_idx] / avg_len)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * (tf * (_K1 + 1.0)) / denom
        return scores, ideal

    # ------------------------------------------------------------------ search
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        self._ensure_index()
        if not self._chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores, ideal = self._bm25_scores(query_tokens)
        if ideal <= 0.0 or not scores:
            return []
        # Sort matched docs by score descending; only iterate matched docs.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        results: list[SearchResult] = []
        for idx, raw_score in ranked:
            if raw_score <= 0.0:
                break
            chunk = self._chunks[idx]
            if filter and any(chunk.metadata.get(k) != v for k, v in filter.items()):
                continue
            results.append(
                SearchResult(
                    text=chunk.text,
                    score=raw_score / ideal,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    metadata=dict(chunk.metadata),
                    language=chunk.language,
                    source=chunk.source,
                    page=chunk.page,
                )
            )
            if len(results) >= top_k:
                break
        return results
