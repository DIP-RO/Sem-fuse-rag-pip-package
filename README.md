# SemFuse

[![PyPI version](https://img.shields.io/pypi/v/semfuse)](https://pypi.org/project/semfuse/)
[![Python versions](https://img.shields.io/pypi/pyversions/semfuse)](https://pypi.org/project/semfuse/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Tests](https://github.com/DIP-RO/Sem-fuse-rag-pip-package/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DIP-RO/Sem-fuse-rag-pip-package/actions/workflows/ci.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/semfuse)](https://pypistats.org/packages/semfuse)
[![GitHub stars](https://img.shields.io/github/stars/DIP-RO/Sem-fuse-rag-pip-package?style=flat)](https://github.com/DIP-RO/Sem-fuse-rag-pip-package/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/DIP-RO/Sem-fuse-rag-pip-package?style=flat)](https://github.com/DIP-RO/Sem-fuse-rag-pip-package/forks)

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
- Banglish detection, normalization, and transliteration ([docs/banglish.md](docs/banglish.md))
- Document ingestion: TXT/MD, PDF, DOCX loaders with recursive chunking
- Keyword (BM25) and hybrid retrieval with weighted / RRF score fusion
- Reranking: offline lexical reranker or a multilingual cross-encoder
- RAG with numbered citations: offline extractive default, OpenAI optional
- Evaluation subsystem (Recall@K, MRR, NDCG, Hit@K) + built-in Banglish benchmark
- Local persistent vector store (numpy-based, no external services)
- Deterministic offline embedding provider for testing
- Configurable embedding providers, metrics, and search modes
- Metadata filtering, collections, content-hash deduplication
- Index version guards (clear errors on model/dimension mismatch)
- Typed results with citations-ready metadata
- CLI: `semfuse info | index | search | ask`
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

### File & directory ingestion

```python
db.add_file("notes.txt")                  # TXT/MD; PDF/DOCX via extras
db.add_file("report.pdf")                 # one document per page
db.add_directory("./corpus", recursive=True)
```

Long documents are chunked recursively (paragraphs → sentences, including the
Bangla dari `।`) with configurable `chunk_size` / `chunk_overlap`; duplicate
chunks are skipped by content hash.

### Search modes & reranking

```python
db.search("query")                        # auto = hybrid (semantic + BM25, fused)
db.search("query", mode="semantic")       # embeddings only
db.search("query", mode="keyword")        # BM25 only
db.search("query", rerank=True)           # offline lexical reranker

db = SemFuse(reranker="cross-encoder")    # multilingual model-based reranking
db = SemFuse(fusion_method="rrf")         # rank-based fusion instead of weighted
```

### RAG with citations

```python
response = db.ask("Bangladesh er rajdhani kothay?")
print(response.answer)                    # "ঢাকা বাংলাদেশের রাজধানী। [1]"
print(response.citations[0].text)         # the passage behind [1]
```

The default provider is extractive (offline, no API key). For generative
answers:

```python
db = SemFuse(llm_provider="openai", llm_model="gpt-4o-mini")  # semfuse[rag]
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
- `VectorStore` — `add`, `add_many`, `search`, `delete`, `clear`, `count`, `chunks`, `persist`, `load`
- `Retriever` — `retrieve(query, top_k, filter)`
- `TextChunker`, `DocumentLoader`, `LanguageDetector`, `TextNormalizer`,
  `Reranker`, `LLMProvider`

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
| `faiss` | FAISS | Optional extra (`semfuse[faiss]`) — planned |
| `qdrant` | Qdrant | Optional extra (`semfuse[qdrant]`) — planned |

## Banglish support

Banglish (Bengali written in Latin script) is a core feature. Queries and
documents flow through `detect_language` → `BanglishNormalizer` (spelling
canonicalization + token transliteration to Bangla), so `Bangladesh er
rajdhani kothay?` matches `ঢাকা বাংলাদেশের রাজধানী।` in both embedding and
keyword space. The original text is always preserved.

See [docs/banglish.md](docs/banglish.md) for the pipeline, the lexicons, and
the runnable benchmark behind these claims.

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
semfuse index docs/ notes.txt --text "inline document"
semfuse search "Bangladesh er capital ki?" --top-k 3 --mode hybrid --json
semfuse ask "bhorti porikkha kokhon hobe?"
semfuse --storage ./.semfuse --collection admission info
```

## Docker

```bash
docker build -t semfuse .

# One named volume persists both the index and the embedding-model cache,
# so the model downloads once and is reused across containers.
docker run --rm -v semfuse-data:/data semfuse index --text "ঢাকা বাংলাদেশের রাজধানী।"
docker run --rm -v semfuse-data:/data semfuse search "Bangladesh er capital ki?"
docker run --rm -v semfuse-data:/data semfuse ask "desh er rajdhani kothay?"

# Fully offline (no model download): use the deterministic hashing provider
docker run --rm -v semfuse-data:/data semfuse --provider hashing info
```

The image uses CPU-only PyTorch (≈1.8 GB instead of the multi-GB CUDA
default). To index files from the host, mount them read-only:
`docker run --rm -v semfuse-data:/data -v "$PWD/docs:/docs:ro" semfuse index /docs`.

Tagged releases publish a prebuilt multi-arch image (amd64/arm64) to GHCR:
`docker pull ghcr.io/dip-ro/semfuse:latest`.

## Evaluation

SemFuse includes an evaluation subsystem with Recall@K, MRR, NDCG, and Hit@K,
plus a built-in Banglish benchmark. We do not publish benchmark numbers that
are not backed by runnable evaluations.

```python
from semfuse import SemFuse
from semfuse.evaluation import RetrievalEvaluator, EvalSample, banglish_benchmark

db = SemFuse(storage_path="./.semfuse-bench")
docs, samples = banglish_benchmark()
for doc_id, text in docs:
    db.add(text, document_id=doc_id)
print(RetrievalEvaluator(db).evaluate(samples, k_values=(1, 3)))
# EvaluationReport(samples=6, hit@1=..., hit@3=..., mrr=..., ...)
```

## Performance

- Lazy model loading (import is fast; model loads on first use)
- Model reuse (one instance shared across all queries)
- Batched embedding generation
- Content-hash deduplication (no duplicate chunks)
- Persistent index (reopen without re-indexing)
- CPU by default; GPU used when available and supported

## Limitations

- Banglish transliteration is lexicon-based: high-frequency words are covered,
  long-tail romanizations fall back to the multilingual model's cross-script
  ability. Growing the lexicons (with benchmark evidence) is welcome.
- The default local vector store is in-memory with file persistence; it is not
  optimized for very large corpora (FAISS/Qdrant extras will address this).
- The default RAG provider is extractive, not generative — it returns the
  best-matching passage with a citation. Configure `llm_provider="openai"` for
  generated answers.
- The BM25 index is rebuilt in memory when the corpus changes; this is fine for
  the corpus sizes the local store targets.

## Roadmap

- [x] Phase 1 — Core foundation (embeddings, local store, semantic retrieval, persistence)
- [x] Phase 2 — Language & Banglish normalization
- [x] Phase 3 — Document ingestion (TXT/PDF/DOCX, chunking, dedup)
- [x] Phase 4 — Keyword & hybrid retrieval, fusion, collections
- [x] Phase 5 — Reranking
- [x] Phase 6 — RAG (LLM providers, citations)
- [x] Phase 7 — Evaluation (Recall@K, MRR, NDCG, Banglish benchmark)
- [x] Phase 8 — Production quality (CLI, CI, docs, examples)
- [ ] FAISS / Qdrant vector store backends (optional extras)

## Contributing

Contributions are welcome. Please run `ruff check`, `mypy`, and `pytest` before
submitting changes.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
