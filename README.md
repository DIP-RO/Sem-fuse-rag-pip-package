# SemFuse

> Lightweight multilingual semantic retrieval with first-class **Bangla**,
> **English**, **Banglish**, and **mixed-language** support, plus an optional
> RAG layer.

SemFuse hides embedding models, vector stores, normalization, and retrieval
behind a tiny developer API. The default path needs zero configuration.

```python
from semfuse import SemFuse

db = SemFuse()

db.add("ঢাকা বাংলাদেশের রাজধানী।")

print(db.search("Bangladesh er capital ki?"))
```

## Why SemFuse?

Most retrieval frameworks are English-first and treat non-English scripts as an
afterthought. SemFuse is built around a different premise: **Bangla, English,
Banglish (Bengali written in Latin script), and mixed-language text are
first-class concerns**, not edge cases.

A query like `Bangladesh er capital ki?` should retrieve `ঢাকা বাংলাদেশের
রাজধানী।` without the developer wiring up transliteration, model selection, or
a vector database by hand.

## Features

- Zero-config initialization: `SemFuse()` just works
- Multilingual semantic retrieval (Bangla, English, cross-lingual)
- Banglish / mixed-language handling (Phase 2 expands this)
- Local persistent vector store (numpy-based, no external services)
- Deterministic offline embedding provider for testing
- Configurable embedding providers, metrics, and search modes
- Metadata filtering
- Content-hash deduplication
- Index version guards (clear errors on model/dimension mismatch)
- Typed results with citations-ready metadata
- Optional RAG layer (later phases)
- Lightweight core: numpy + sentence-transformers only by default

## Quickstart

```bash
pip install semfuse
```

```python
from semfuse import SemFuse

db = SemFuse()

db.add("Dhaka is the capital of Bangladesh.")
db.add("ঢাকা বাংলাদেশের রাজধানী।")

results = db.search("বাংলাদেশের রাজধানী কী?")
print(results)
# [SearchResult(score=0.89, text='ঢাকা বাংলাদেশের রাজধানী।')]
```

### English example

```python
db.add("The Eiffel Tower is in Paris.")
db.add("Tokyo is the capital of Japan.")
print(db.search("capital of Japan"))
```

### Bangla example

```python
db.add("ঢাকা বাংলাদেশের রাজধানী।")
print(db.search("বাংলাদেশের রাজধানী কী?"))
```

### Banglish example

```python
db.add("ঢাকা বাংলাদেশের রাজধানী।")
print(db.search("Bangladesh er capital ki?"))
```

### Cross-lingual example

```python
db.add("ঢাকা বাংলাদেশের রাজধানী।")
db.add("The Eiffel Tower is in Paris.")
# English query retrieves the Bangla document
print(db.search("What is the capital of Bangladesh?"))
```

### Persistence

```python
db = SemFuse(storage_path="./.semfuse")
db.add("some document")
db.close()

# Later — no re-indexing needed
db = SemFuse(storage_path="./.semfuse")
print(db.search("some document"))
```

### Metadata filtering

```python
db.add("CSE admission notice.", metadata={"department": "CSE"})
db.add("EEE admission notice.", metadata={"department": "EEE"})
print(db.search("admission", filter={"department": "CSE"}))
```

### Advanced configuration

```python
db = SemFuse(
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    metric="cosine",
    top_k=10,
    score_threshold=0.5,
    storage_path="./.semfuse",
    collection="admission",
)
```

### Diagnostics

```python
print(db.info())     # index metadata, counts, language distribution
print(db.explain("Bangladesh er capital ki?"))  # query processing breakdown
```

## Architecture

SemFuse is layered: **Language → Embeddings → Ingestion → Indexing → Retrieval
→ (Reranker) → (RAG)**. Every pluggable layer is a `Protocol`:

- `EmbeddingProvider` — `embed_documents`, `embed_query`, `dimension`, `model_name`
- `VectorStore` — `add`, `add_many`, `search`, `delete`, `clear`, `count`, `persist`, `load`
- `Retriever` — `retrieve(query, top_k, filter)`

See [docs/architecture.md](docs/architecture.md) and
[docs/architecture-decisions.md](docs/architecture-decisions.md) for the full
design and decision log.

### Embedding providers

| Key | Backend | Notes |
|-----|---------|-------|
| `local` (default) | `sentence-transformers` | Lazy-loaded, reused, multilingual |
| `hashing` | character n-gram hashing | Deterministic, offline, for tests |

The default model is `paraphrase-multilingual-MiniLM-L12-v2` (384-dim), selected
in `SemFuseConfig` — change it without code edits.

### Vector stores

| Key | Backend | Notes |
|-----|---------|-------|
| `local` (default) | numpy + JSON files | No external services, persistent |
| `faiss` | FAISS | Optional extra (`semfuse[faiss]`) — later phase |
| `qdrant` | Qdrant | Optional extra (`semfuse[qdrant]`) — later phase |

## Banglish support

Banglish (Bengali written in Latin script) is a core feature. The architecture
provides `detect_language`, `normalize_text`, and a `BanglishNormalizer`
abstraction (Phase 2) so Banglish processing can evolve independently of the
embedding model. The original text is always preserved.

See [docs/banglish.md](docs/banglish.md).

## Installation extras

```bash
pip install semfuse            # core (numpy + sentence-transformers)
pip install semfuse[pdf]       # PDF loader
pip install semfuse[docx]      # DOCX loader
pip install semfuse[faiss]     # FAISS vector store
pip install semfuse[qdrant]    # Qdrant vector store
pip install semfuse[rag]       # RAG / LLM providers
pip install semfuse[dev]       # pytest, ruff, mypy
pip install semfuse[all]       # everything
```

## CLI

```bash
semfuse info
semfuse --storage ./.semfuse info
```

(`index` and `search` subcommands arrive with later phases.)

## Evaluation

SemFuse includes an evaluation subsystem (later phases) with Recall@K, MRR,
NDCG, and Hit@K, plus a Banglish benchmark fixture. We do not publish benchmark
numbers that are not backed by runnable evaluations.

## Performance

- Lazy model loading (import is fast; model loads on first use)
- Model reuse (one instance shared across all queries)
- Batched embedding generation
- Content-hash deduplication (no duplicate chunks)
- Persistent index (reopen without re-indexing)
- CPU by default; GPU used when available and supported

## Limitations

- Phase 1 ships semantic retrieval only; keyword/hybrid retrieval, reranking,
  and RAG arrive in later phases.
- Banglish retrieval relies on the multilingual model's cross-script ability in
  Phase 1; dedicated Banglish normalization lands in Phase 2 and measurably
  improves Banglish→Bangla retrieval.
- The default local vector store is in-memory with file persistence; it is not
  optimized for very large corpora (FAISS/Qdrant extras address this later).
- A single document is a single chunk in Phase 1; recursive chunking lands in
  Phase 3.

## Roadmap

- [x] Phase 1 — Core foundation (embeddings, local store, semantic retrieval, persistence)
- [ ] Phase 2 — Language & Banglish normalization
- [ ] Phase 3 — Document ingestion (TXT/PDF/DOCX, chunking, dedup)
- [ ] Phase 4 — Keyword & hybrid retrieval, fusion, collections
- [ ] Phase 5 — Reranking
- [ ] Phase 6 — RAG (LLM providers, citations)
- [ ] Phase 7 — Evaluation (Recall@K, MRR, NDCG, Banglish benchmark)
- [ ] Phase 8 — Production quality (CLI, CI, docs, examples)

## Contributing

Contributions are welcome. Please run `ruff check`, `mypy`, and `pytest` before
submitting changes.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
