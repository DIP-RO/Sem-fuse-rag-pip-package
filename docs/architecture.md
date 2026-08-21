# SemFuse Architecture

> Lightweight, model-agnostic multilingual semantic retrieval engine with
> first-class Bangla, English, Banglish, and mixed-language support, plus an
> optional RAG layer.

## Design principle

**Complex internally, simple externally.**

A beginner writes:

```python
from semfuse import SemFuse

db = SemFuse()
db.add("ঢাকা বাংলাদেশের রাজধানী।")
db.search("Bangladesh er capital ki?")
```

An advanced developer can swap every layer via protocols without touching the
simple path.

## Logical layers

```
                 SemFuse  (public client)
                    |
       +------------+-------------+
       |            |             |
       v            v             v
   Language      Embeddings    Ingestion
       |            |             |
       +------------+-------------+
                    |
                    v
               Indexing
                    |
                    v
             Retrieval Engine
             /      |       \
     Semantic    Keyword    Hybrid
                    |
                    v
                Reranker (optional)
                    |
                    v
                 Results
                    |
                    v
             Optional RAG  ->  LLM
```

## Module map (Phase 1 scope highlighted)

| Layer | Module | Phase |
|-------|--------|-------|
| Public API | `core.client.SemFuse` | 1 |
| Config | `core.config.SemFuseConfig` | 1 |
| Types | `core.types` | 1 |
| Enums | `core.enums` | 1 |
| Exceptions | `core.exceptions` | 1 |
| Embeddings | `embeddings.base`, `embeddings.local`, `embeddings.hashing`, `embeddings.factory` | 1 |
| Vector stores | `vectorstores.base`, `vectorstores.local` | 1 |
| Retrieval | `retrieval.base`, `retrieval.semantic` | 1 |
| Language | `language.*` | 2 |
| Chunking | `chunking.*` | 3 |
| Loaders | `loaders.*` | 3 |
| Keyword/Hybrid | `retrieval.keyword`, `retrieval.hybrid`, `retrieval.fusion` | 4 |
| Reranking | `reranking.*` | 5 |
| RAG | `rag.*` | 6 |
| Evaluation | `evaluation.*` | 7 |
| CLI | `cli.main` | 8 |

## Key abstractions (Protocols)

- `EmbeddingProvider` — `embed_documents`, `embed_query`, `dimension`, `model_name`
- `VectorStore` — `add`, `add_many`, `search`, `delete`, `clear`, `count`, `persist`, `load`
- `Retriever` — `retrieve(query, top_k, filter) -> list[SearchResult]`
- (later) `TextChunker`, `DocumentLoader`, `LanguageDetector`, `TextNormalizer`,
  `BanglishNormalizer`, `Reranker`, `LLMProvider`

## Dependency strategy

- **Core runtime deps:** `numpy` only. Kept intentionally light.
- **Default embedding backend:** `sentence-transformers` (lazy-loaded). Pulled in
  via the default install because semantic retrieval *is* the product. Loaded once
  and reused; never instantiated per query.
- **Optional extras:** `semfuse[pdf]`, `semfuse[docx]`, `semfuse[qdrant]`,
  `semfuse[faiss]`, `semfuse[rag]`, `semfuse[dev]`, `semfuse[all]`.
- No LangChain / LlamaIndex orchestration frameworks in core.

## Persistence

The local vector store serializes embeddings (`.npy`), document/chunk records
(JSON), and index metadata (JSON) under a configurable `storage_path`. Reopening
a `SemFuse(storage_path=...)` skips re-indexing. Index metadata records the
embedding model name + dimension + version; a mismatch raises
`IndexVersionError` with remediation guidance.

## Test strategy

- **Unit tests** use a deterministic `HashingEmbeddingProvider` (offline, fast,
  no model download) so they run anywhere without internet.
- **Integration / retrieval tests** use the real `sentence-transformers` backend
  and are skipped automatically when the model is unavailable or offline.
