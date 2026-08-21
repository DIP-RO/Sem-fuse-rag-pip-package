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
    """BM25 (Okapi) retrieval over stored chunks."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._indexed_count = -1
        self._chunks: list[DocumentChunk] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0

    # ------------------------------------------------------------------ index
    def _ensure_index(self) -> None:
        count = self._store.count()
        if count == self._indexed_count:
            return
        self._chunks = self._store.chunks()
        self._doc_freqs = []
        self._doc_lens = []
        self._df = Counter()
        for chunk in self._chunks:
            tokens = tokenize(chunk.normalized_text or chunk.text)
            freqs = Counter(tokens)
            self._doc_freqs.append(freqs)
            self._doc_lens.append(len(tokens))
            self._df.update(freqs.keys())
        self._avg_len = (sum(self._doc_lens) / len(self._doc_lens)) if self._doc_lens else 0.0
        self._indexed_count = count

    def _bm25_scores(self, query_tokens: list[str]) -> tuple[list[float], float]:
        """Per-chunk BM25 scores plus the query's ideal (saturation) score."""
        n = len(self._chunks)
        scores = [0.0] * n
        ideal = 0.0
        for term in query_tokens:
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            # Per-term contribution is bounded by idf * (k1 + 1) as tf -> inf.
            ideal += idf * (_K1 + 1.0)
            for i, freqs in enumerate(self._doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + _K1 * (1.0 - _B + _B * self._doc_lens[i] / (self._avg_len or 1.0))
                scores[i] += idf * (tf * (_K1 + 1.0)) / denom
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
        if ideal <= 0.0 or max(scores) <= 0.0:
            return []
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        results: list[SearchResult] = []
        for idx in order:
            if scores[idx] <= 0.0:
                break
            chunk = self._chunks[idx]
            if filter and any(chunk.metadata.get(k) != v for k, v in filter.items()):
                continue
            results.append(
                SearchResult(
                    text=chunk.text,
                    score=scores[idx] / ideal,
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
