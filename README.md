# SemFuse

[![PyPI version](https://img.shields.io/pypi/v/semfuse)](https://pypi.org/project/semfuse/)
[![Python versions](https://img.shields.io/pypi/pyversions/semfuse)](https://pypi.org/project/semfuse/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Tests](https://github.com/DIP-RO/Sem-fuse-rag-pip-package/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DIP-RO/Sem-fuse-rag-pip-package/actions/workflows/ci.yml)
[![PyPI downloads](https://img.shields.io/pypi/dm/semfuse)](https://pypistats.org/packages/semfuse)
[![GitHub stars](https://img.shields.io/github/stars/DIP-RO/Sem-fuse-rag-pip-package?style=flat)](https://github.com/DIP-RO/Sem-fuse-rag-pip-package/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/DIP-RO/Sem-fuse-rag-pip-package?style=flat)](https://github.com/DIP-RO/Sem-fuse-rag-pip-package/forks)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/DIP-RO/Sem-fuse-rag-pip-package/pkgs/container/semfuse)

> Lightweight multilingual semantic retrieval with first-class **Bangla**,
> **English**, **Banglish**, and **mixed-language** support, plus an optional
> RAG layer with citations.

SemFuse hides embedding models, vector stores, text normalization, Banglish
processing, chunking, hybrid retrieval, reranking, and RAG behind a tiny
developer API. The default path needs zero configuration.

```python
from semfuse import SemFuse

db = SemFuse()

db.add("ঢাকা বাংলাদেশের রাজধানী।")

print(db.search("Bangladesh er capital ki?"))
```

---

## Table of Contents

- [Why SemFuse?](#why-semfuse)
- [Features](#features)
- [Quickstart](#quickstart)
- [English Example](#english-example)
- [Bangla Example](#bangla-example)
- [Banglish Example](#banglish-example)
- [Cross-Lingual Example](#cross-lingual-example)
- [Document Indexing](#document-indexing)
- [Metadata Filtering](#metadata-filtering)
- [Search Modes](#search-modes)
- [Reranking](#reranking)
- [RAG with Citations](#rag-with-citations)
- [Collections](#collections)
- [Diagnostics](#diagnostics)
- [CLI](#cli)
- [Docker](#docker)
- [Architecture](#architecture)
- [Embedding Providers](#embedding-providers)
- [Vector Stores](#vector-stores)
- [Banglish Support](#banglish-support)
- [Evaluation](#evaluation)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Installation Extras](#installation-extras)
- [Performance](#performance)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Why SemFuse?

Most retrieval frameworks are English-first and treat non-English scripts as an
afterthought. SemFuse is built around a different premise: **Bangla, English,
Banglish (Bengali written in Latin script), and mixed-language text are
first-class concerns**, not edge cases.

A query like `Bangladesh er capital ki?` should retrieve `ঢাকা বাংলাদেশের
রাজধানী।` without the developer wiring up transliteration, model selection, or
a vector database by hand.

All four of these should retrieve the same document:

```
বাংলাদেশের রাজধানী কী?              ← Bangla
What is the capital of Bangladesh?   ← English
Bangladesh er capital ki?            ← Banglish
Bangladesher rajdhani ki?            ← Banglish (variant)
```

---

## Features

- **Zero-config initialization**: `SemFuse()` just works
- **Multilingual semantic retrieval**: Bangla, English, cross-lingual
- **Banglish detection, normalization, and transliteration** ([docs/banglish.md](docs/banglish.md))
- **Document ingestion**: TXT/MD, PDF, DOCX loaders with recursive chunking
- **Keyword (BM25) and hybrid retrieval** with weighted / RRF score fusion
- **Reranking**: offline lexical reranker or a multilingual cross-encoder
- **RAG with numbered citations**: offline extractive default, local SLM for generative answers, OpenAI optional
- **Evaluation subsystem**: Recall@K, MRR, NDCG, Hit@K + built-in Banglish benchmark
- **Local persistent vector store** (numpy-based, no external services)
- **Deterministic offline embedding provider** for testing
- **Configurable** embedding providers, metrics, search modes, fusion, rerankers, LLMs
- **Metadata filtering, collections, content-hash deduplication**
- **Index version guards** (clear errors on model/dimension mismatch)
- **Typed results** with citations-ready metadata
- **CLI**: `semfuse info | index | search | ask`
- **Docker image**: multi-arch (amd64/arm64) on GHCR
- **Lightweight core**: numpy + sentence-transformers only by default
- **153 tests** (offline unit + integration + language + retrieval + persistence)

---

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
# [SearchResult(score=0.96, text='ঢাকা বাংলাদেশের রাজধানী।')]
```

---

## English Example

```python
db.add("The Eiffel Tower is in Paris.")
db.add("Tokyo is the capital of Japan.")
print(db.search("capital of Japan"))
# [SearchResult(score=0.71, text='Tokyo is the capital of Japan.')]
```

## Bangla Example

```python
db.add("ঢাকা বাংলাদেশের রাজধানী।")
print(db.search("বাংলাদেশের রাজধানী কী?"))
# [SearchResult(score=0.96, text='ঢাকা বাংলাদেশের রাজধানী।')]
```

## Banglish Example

```python
db.add("ঢাকা বাংলাদেশের রাজধানী।")
print(db.search("Bangladesh er capital ki?"))
```

Banglish queries flow through `detect_language` → `BanglishNormalizer`
(spelling canonicalization + token transliteration to Bangla) so they match
Bangla documents in both embedding and keyword space.

## Cross-Lingual Example

```python
db.add("ঢাকা বাংলাদেশের রাজধানী।")
db.add("The Eiffel Tower is in Paris.")
# English query retrieves the Bangla document
print(db.search("What is the capital of Bangladesh?"))
# [SearchResult(score=0.82, text='ঢাকা বাংলাদেশের রাজধানী।')]
```

---

## Document Indexing

### Inline text

```python
db.add("some text")
db.add("text with metadata", metadata={"department": "CSE", "year": 2026})
db.add_many(["text one", "text two"], metadata=[{"k": "v"}, {"k": "w"}])
```

### Files (TXT/MD/PDF/DOCX)

```python
db.add_file("notes.txt")                    # TXT/MD built-in
db.add_file("report.pdf")                   # PDF via semfuse[pdf]
db.add_file("doc.docx")                     # DOCX via semfuse[docx]
db.add_directory("./corpus", recursive=True)
db.add_directory("./corpus", extensions=(".md", ".txt"))
```

Long documents are chunked recursively (paragraphs → sentences, including the
Bangla dari `।`) with configurable `chunk_size` / `chunk_overlap`. Duplicate
chunks are skipped by content hash — adding the same file twice is a no-op.

### Persistence

```python
db = SemFuse(storage_path="./.semfuse")
db.add("some document")
db.close()

# Later — no re-indexing needed
db = SemFuse(storage_path="./.semfuse")
print(db.search("some document"))
```

---

## Metadata Filtering

```python
db.add("CSE admission notice.", metadata={"department": "CSE"})
db.add("EEE admission notice.", metadata={"department": "EEE"})

# Only retrieve chunks where metadata["department"] == "CSE"
print(db.search("admission", filter={"department": "CSE"}))
```

Filters are exact-match equality on metadata keys. Multiple keys are ANDed.

---

## Search Modes

```python
db.search("query")                        # auto = hybrid (semantic + BM25, fused)
db.search("query", mode="semantic")       # embeddings only
db.search("query", mode="keyword")        # BM25 only
db.search("query", mode="hybrid")         # semantic + keyword, fused
```

The `auto` mode (default) resolves to `hybrid`, which runs semantic and keyword
retrieval in parallel and fuses the results. Fusion methods:

```python
db = SemFuse(fusion_method="weighted")    # weighted sum (default)
db = SemFuse(fusion_method="rrf")         # reciprocal rank fusion
```

Weights are configurable:

```python
db = SemFuse(semantic_weight=0.7, keyword_weight=0.3)  # default
```

---

## Reranking

```python
# On-demand reranking with the offline lexical reranker (no model download)
results = db.search("query", rerank=True)

# Configure a model-based cross-encoder reranker
db = SemFuse(reranker="cross-encoder")
# Uses cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 by default

# Turn reranking off for a specific call
results = db.search("query", rerank=False)
```

The reranker fetches `rerank_candidates` (default 25) candidates, reranks them,
then cuts to `top_k`. The offline `LexicalReranker` blends the retriever's score
with a Dice token-overlap coefficient — no model download required.

---

## RAG with Citations

```python
db.add("ঢাকা বাংলাদেশের রাজধানী।", source="geography.txt")
db.add("বিশ্ববিদ্যালয়ে ভর্তি পরীক্ষা ডিসেম্বরে অনুষ্ঠিত হবে।", source="admission.txt")

response = db.ask("Bangladesh er rajdhani kothay?")
print(response.answer)
# "ঢাকা বাংলাদেশের রাজধানী। [1]"

print(response.citations[0].text)     # the passage behind [1]
print(response.citations[0].source)   # "geography.txt"
print(response.citations[0].score)    # retrieval score
```

The `RAGResponse` object exposes:

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `str` | Answer text with `[n]` citation markers |
| `model` | `str` | LLM identifier |
| `citations` | `list[SearchResult]` | Retrieved chunks; `[n]` refers to `citations[n-1]` |
| `prompt` | `str` | The exact prompt sent to the LLM (for audit) |

The default provider is **extractive** (`llm_provider="template"`): offline, no
API key, returns a concise extracted answer with a citation. For generative
answers, use the **local SLM** provider (no API key, runs on CPU):

```python
db = SemFuse(llm_provider="slm")  # requires: pip install semfuse[slm]
```

The SLM provider uses [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
(500M params) in Q4_K_M GGUF quantization (~400 MB on disk) via `llama-cpp-python`
(~50 MB library). Total footprint: **~450 MB** — no torch, no CUDA, no GPU required.
Runs on CPU on x86_64 and ARM64 (Apple Silicon, Graviton, Raspberry Pi).

The provider auto-detects the backend:
- If `llama-cpp-python` is installed → uses GGUF (lightweight, recommended)
- Elif `transformers` + `torch` are installed → uses HuggingFace backend
- Else → raises with install instructions

The model downloads once and is cached locally. Post-processing ensures
evidence grounding: citation enforcement, hallucination detection with
extractive fallback, and verbose output trimming.

For OpenAI (optional, requires API key):

```python
db = SemFuse(llm_provider="openai", llm_model="gpt-4o-mini")  # requires semfuse[rag]
```

Citations only use retrieved metadata — the package never invents sources.

---

## Collections

Collections are lightweight namespaces within a storage path. Each collection
has its own vector index, chunk store, and index metadata.

```python
db_admission = SemFuse(storage_path="./.semfuse", collection="admission")
db_research  = SemFuse(storage_path="./.semfuse", collection="research")

db_admission.add("Admission requirements...")
db_research.add("Research paper on NLP...")

# Search within a collection
db_admission.search("admission requirements")

# List all collections in a storage path
print(db_admission.list_collections())  # ['admission', 'research']
```

---

## Diagnostics

```python
print(db.info())
# {
#   "package_version": "0.2.0",
#   "embedding_provider": "local",
#   "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
#   "embedding_dimension": 384,
#   "vector_backend": "local",
#   "metric": "cosine",
#   "storage_path": ".semfuse",
#   "collection": "default",
#   "document_count": 3,
#   "chunk_count": 3,
#   "language_distribution": {"en": 2, "bn": 1},
#   ...
# }

print(db.explain("Bangladesh er capital ki?"))
# {
#   "query": "Bangladesh er capital ki?",
#   "detected_language": "banglish",
#   "normalized_query": "Bangladesh er capital ki?",
#   "search_mode": "hybrid",
#   "embedding_provider": "local",
#   "candidate_count": 3,
#   "top_score": 0.81,
#   "results": [...]
# }
```

---

## CLI

```bash
# Show index information
semfuse info

# Index files, directories, or inline text
semfuse index docs/ notes.txt --text "inline document"

# Search the index
semfuse search "Bangladesh er capital ki?" --top-k 3 --mode hybrid --json

# Ask a question (RAG with citations)
semfuse ask "bhorti porikkha kokhon hobe?"

# Use a different storage path / collection / provider
semfuse --storage ./.semfuse --collection admission info
semfuse --provider hashing info   # offline deterministic provider
```

---

## Docker

```bash
# Pull the prebuilt multi-arch image (amd64/arm64)
docker pull ghcr.io/dip-ro/semfuse:latest

# Or build locally
docker build -t semfuse .

# One named volume persists both the index and the embedding-model cache,
# so the model downloads once and is reused across containers.
docker run --rm -v semfuse-data:/data semfuse index --text "ঢাকা বাংলাদেশের রাজধানী।"
docker run --rm -v semfuse-data:/data semfuse search "Bangladesh er capital ki?"
docker run --rm -v semfuse-data:/data semfuse ask "desh er rajdhani kothay?"

# Fully offline (no model download): use the deterministic hashing provider
docker run --rm -v semfuse-data:/data semfuse --provider hashing info

# Index files from the host (mount read-only)
docker run --rm -v semfuse-data:/data -v "$PWD/docs:/docs:ro" semfuse index /docs
```

The image uses `llama-cpp-python` (no torch/CUDA) and supports both `linux/amd64`
and `linux/arm64`. Tagged releases publish to GHCR automatically via CI.

---

## Architecture

SemFuse is layered: **Language → Embeddings → Ingestion → Indexing → Retrieval
→ (Reranker) → (RAG)**. Every pluggable layer is a `Protocol` so backends can
be swapped without touching the public API.

```mermaid
graph TD
    Client["SemFuse (public client)"]

    subgraph "Language Layer"
        Detect["detect_language"]
        Norm["normalize_for_search"]
        Banglish["BanglishNormalizer<br/>(dict + phonetic engine)"]
    end

    subgraph "Embedding Layer"
        Local["LocalEmbeddingProvider<br/>(sentence-transformers)"]
        Hashing["HashingEmbeddingProvider<br/>(deterministic, zero-dep)"]
    end

    subgraph "Ingestion Layer"
        Loaders["Loaders<br/>(TXT/MD/PDF/DOCX)"]
        Chunker["RecursiveCharacterChunker<br/>(Bangla dari aware)"]
        Dedup["Content-hash dedup"]
    end

    subgraph "Vector Store"
        Store["LocalVectorStore<br/>(growable buffer, argpartition)"]
    end

    subgraph "Retrieval Engine"
        Semantic["SemanticRetriever<br/>(cosine/dot/euclidean)"]
        Keyword["KeywordRetriever<br/>(BM25 inverted index)"]
        Hybrid["HybridRetriever<br/>(weighted / RRF fusion)"]
    end

    subgraph "Reranking (optional)"
        Lexical["LexicalReranker<br/>(offline, Dice overlap)"]
        CrossEncoder["CrossEncoderReranker<br/>(multilingual)"]
    end

    subgraph "RAG (optional)"
        Template["TemplateLLMProvider<br/>(extractive, offline)"]
        SLM["LocalSLMProvider<br/>(Qwen2.5-0.5B, CPU)"]
        OpenAI["OpenAILLMProvider<br/>(optional, API key)"]
    end

    Client --> Detect
    Client --> Local
    Client --> Hashing
    Client --> Loaders
    Detect --> Norm
    Norm --> Banglish
    Loaders --> Chunker
    Chunker --> Dedup
    Dedup --> Store
    Local --> Store
    Hashing --> Store
    Store --> Semantic
    Store --> Keyword
    Semantic --> Hybrid
    Keyword --> Hybrid
    Hybrid --> Lexical
    Hybrid --> CrossEncoder
    Lexical --> Results["Results"]
    CrossEncoder --> Results
    Results --> Template
    Results --> SLM
    Results --> OpenAI
```

### Banglish Processing Pipeline

```mermaid
graph LR
    Input["Input text<br/>(Latin script)"] --> Detect{"detect_language"}
    Detect -->|Bangla| PassBN["Pass through<br/>(already Bangla)"]
    Detect -->|English| PassEN["Pass through<br/>(English)"]
    Detect -->|Banglish| Dict["Dictionary lookup<br/>~250 curated entries"]
    Dict --> Phonetic["Phonetic fallback<br/>(rule-based, any token)"]
    Phonetic --> Output["Bangla script output"]
    PassBN --> Output
    PassEN --> Output
```

### RAG Answer Generation Pipeline

```mermaid
graph TD
    Q["Question"] --> Retrieve["Retrieve top-k passages"]
    Retrieve --> Prompt["Build evidence-grounded prompt"]
    Prompt --> Gen{"Generate answer"}
    Gen -->|Template| Extract["Extractive answer<br/>(question-aware span)"]
    Gen -->|SLM| SLMGen["SLM generation<br/>(Qwen2.5-0.5B)"]
    Gen -->|OpenAI| OAIGen["OpenAI generation"]
    SLMGen --> Ground{"Grounding check"}
    Ground -->|Grounded| Cite["Enforce citation [n]"]
    Ground -->|Hallucinated| Fallback["Extractive fallback"]
    Extract --> Answer["Cited answer"]
    Cite --> Answer
    Fallback --> Answer
    OAIGen --> Answer
```

See [docs/architecture.md](docs/architecture.md) and
[docs/architecture-decisions.md](docs/architecture-decisions.md) for the full
design and decision log.

### Embedding Providers

| Key | Backend | Notes |
|-----|---------|-------|
| `local` (default) | `sentence-transformers` | Lazy-loaded, reused, multilingual (incl. Bangla) |
| `hashing` | character n-gram hashing | Deterministic, offline, for tests |

The default model is `paraphrase-multilingual-MiniLM-L12-v2` (384-dim), selected
in `SemFuseConfig` — change it without code edits.

### Vector Stores

| Key | Backend | Notes |
|-----|---------|-------|
| `local` (default) | numpy + JSON files | No external services, persistent, cosine/dot/euclidean |
| `faiss` | FAISS | Optional extra (`semfuse[faiss]`) — planned |
| `qdrant` | Qdrant | Optional extra (`semfuse[qdrant]`) — planned |

---

## Banglish Support

Banglish (Bengali written in Latin script) is a core feature, not an
afterthought. Queries and documents flow through:

```
text → detect_language → BanglishNormalizer → canonical spelling + transliteration
```

The `BanglishNormalizer`:
- folds spelling variants (`achhe`/`ache` → `ache`, `valo`/`vhalo` → `bhalo`)
- collapses elongations (`kiii` → `ki`, `bhaloooo` → `bhalo`)
- transliterates known romanized tokens to Bangla script so Banglish queries
  land closer to Bangla documents in both embedding and keyword space
- preserves the original text (non-destructive)

The detector uses a curated marker lexicon of high-frequency romanized Bangla
function words to distinguish Banglish from plain English, reducing false
positives.

See [docs/banglish.md](docs/banglish.md) for the full pipeline, lexicons, and
the runnable benchmark behind these claims.

---

## Evaluation

SemFuse includes an evaluation subsystem with Recall@K, MRR, NDCG, and Hit@K,
plus a built-in Banglish benchmark fixture. We do not publish benchmark numbers
that are not backed by runnable evaluations.

```python
from semfuse import SemFuse
from semfuse.evaluation import RetrievalEvaluator, EvalSample, banglish_benchmark

db = SemFuse(storage_path="./.semfuse-bench")
docs, samples = banglish_benchmark()
for doc_id, text in docs:
    db.add(text, document_id=doc_id)

report = RetrievalEvaluator(db).evaluate(samples, k_values=(1, 3, 5))
print(report)
# EvaluationReport(samples=6, hit@1=..., hit@3=..., hit@5=..., mrr=..., ndcg@1=..., ...)
```

The benchmark includes cross-language pairs:

| Query | Relevant document |
|-------|-------------------|
| `Bangladesh er capital ki?` | `ঢাকা বাংলাদেশের রাজধানী।` |
| `Admission er jonno ki ki document lagbe?` | `ভর্তির জন্য প্রয়োজনীয় কাগজপত্র জমা দিতে হবে।` |
| `Bangladesher rajdhani ki?` | `Dhaka is the capital of Bangladesh.` |

---

## Configuration Reference

All options can be passed to `SemFuse(...)` or set via `SemFuseConfig`.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `embedding_provider` | `str` | `"local"` | `"local"` (sentence-transformers) or `"hashing"` (offline) |
| `embedding_model` | `str` | `"paraphrase-multilingual-MiniLM-L12-v2"` | Model name |
| `embedding_dimension` | `int` | `384` | Vector dimension |
| `vector_store` | `str` | `"local"` | Vector store backend |
| `storage_path` | `str/Path` | `".semfuse"` | Persistence directory |
| `collection` | `str` | `"default"` | Collection name |
| `metric` | `str` | `"cosine"` | `"cosine"`, `"dot"`, or `"euclidean"` |
| `search_mode` | `str` | `"auto"` | `"semantic"`, `"keyword"`, `"hybrid"`, `"auto"` |
| `top_k` | `int` | `5` | Default number of results |
| `score_threshold` | `float` | `0.0` | Minimum score (0.0–1.0) |
| `chunk_size` | `int` | `500` | Chunk size in characters |
| `chunk_overlap` | `int` | `50` | Overlap between chunks |
| `fusion_method` | `str` | `"weighted"` | `"weighted"` or `"rrf"` |
| `semantic_weight` | `float` | `0.7` | Semantic retriever weight (weighted fusion) |
| `keyword_weight` | `float` | `0.3` | Keyword retriever weight (weighted fusion) |
| `reranker` | `str/None` | `None` | `None`, `"lexical"`, or `"cross-encoder"` |
| `reranker_model` | `str` | `"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"` | Cross-encoder model |
| `rerank_candidates` | `int` | `25` | Candidates fetched before reranking |
| `llm_provider` | `str` | `"template"` | `"template"` (extractive), `"slm"` (local SLM), or `"openai"` |
| `llm_model` | `str` | `"gpt-4o-mini"` | LLM/SLM model name (use `Qwen/Qwen2.5-0.5B-Instruct` for SLM) |
| `device` | `str/None` | `None` | `"cpu"`, `"cuda"`, `"mps"`, or `None` (auto) |
| `lazy` | `bool` | `True` | Lazy-load the embedding model on first use |

---

## API Reference

### `SemFuse` — public client

| Method | Returns | Description |
|--------|---------|-------------|
| `add(text, metadata?, source?, title?, page?, document_id?)` | `int` | Add a text document; returns chunks added |
| `add_many(texts, metadata?)` | `int` | Add multiple texts; returns chunks added |
| `add_file(path, metadata?)` | `int` | Load and index a file (TXT/MD/PDF/DOCX) |
| `add_directory(path, recursive?, extensions?, metadata?)` | `int` | Index all supported files in a directory |
| `search(query, top_k?, score_threshold?, filter?, mode?, rerank?)` | `list[SearchResult]` | Search the index |
| `ask(question, top_k?, filter?, mode?)` | `RAGResponse` | Answer a question with citations |
| `delete(chunk_id)` | `None` | Delete a chunk by id |
| `clear()` | `None` | Remove all chunks |
| `count()` | `int` | Number of stored chunks |
| `persist()` | `None` | Persist to disk |
| `info()` | `dict` | Index metadata, counts, language distribution |
| `explain(query)` | `dict` | Query processing breakdown (language, mode, scores) |
| `list_collections()` | `list[str]` | Collections in the storage path |
| `collection_info()` | `CollectionInfo` | Current collection stats |
| `close()` | `None` | Persist and release resources |

### Typed objects

| Class | Description |
|-------|-------------|
| `SearchResult` | `text`, `score`, `document_id`, `chunk_id`, `metadata`, `language`, `source`, `page` |
| `RAGResponse` | `answer`, `model`, `citations` (list[SearchResult]), `prompt` |
| `Document` | `text`, `source`, `title`, `page`, `language`, `metadata`, `document_id` |
| `DocumentChunk` | `chunk_id`, `document_id`, `text`, `normalized_text`, `language`, `metadata`, `content_hash` |
| `CollectionInfo` | `name`, `document_count`, `chunk_count`, `language_distribution` |
| `SemFuseConfig` | Full configuration dataclass |

### Enums

| Enum | Values |
|------|--------|
| `Language` | `EN`, `BN`, `BANGLISH`, `MIXED`, `UNKNOWN` |
| `SearchMode` | `SEMANTIC`, `KEYWORD`, `HYBRID`, `AUTO` |
| `SimilarityMetric` | `COSINE`, `DOT`, `EUCLIDEAN` |
| `FusionMethod` | `WEIGHTED`, `RRF` |

### Exceptions

All inherit from `SemFuseError`:

`ConfigurationError`, `ModelLoadError`, `UnsupportedLanguageError`,
`VectorStoreError`, `IndexVersionError`, `DocumentLoadError`,
`EmbeddingError`, `RetrievalError`, `RerankingError`, `RAGError`

---

## Installation Extras

```bash
pip install semfuse            # core only (~83 KB wheel, numpy only)
pip install semfuse[embeddings] # sentence-transformers for real embeddings
pip install semfuse[pdf]       # PDF loader (pypdf)
pip install semfuse[docx]      # DOCX loader (python-docx)
pip install semfuse[faiss]     # FAISS vector store (planned)
pip install semfuse[qdrant]    # Qdrant vector store (planned)
pip install semfuse[slm]       # Local SLM RAG provider (llama-cpp-python, ~450 MB total)
pip install semfuse[slm-torch] # Alternative: transformers + torch backend (~2.5 GB)
pip install semfuse[rag]       # OpenAI RAG provider (optional, requires API key)
pip install semfuse[dev]       # pytest, ruff, mypy
pip install semfuse[all]       # everything
```

> **Lightweight by default**: The core `pip install semfuse` pulls only `numpy`
> (~50 MB). The package auto-detects whether `sentence-transformers` is
> installed and falls back to the zero-dependency hashing provider if not.
> Import time is ~40 ms with no heavy modules loaded until you create a
> `SemFuse` instance.

**Docker:**

```bash
docker pull ghcr.io/dip-ro/semfuse:latest
```

---

## Performance

- **Package size**: 83 KB wheel, 423 KB installed — truly lightweight
- **Import time**: ~40 ms (numpy deferred to first `SemFuse()` call)
- **Zero heavy deps at import**: `sentence-transformers`, `torch`, and
  `transformers` are NOT loaded until explicitly used
- **Lazy model loading** — embedding model loads on first use
- **Model reuse** — one instance shared across all queries
- **Batched embedding generation** — documents embedded in batches
- **Content-hash deduplication** — no duplicate chunks stored or re-embedded
- **Growable vector buffer** — amortized O(1) insertion, O(n) top-k search
- **BM25 inverted index** — O(matched docs) instead of O(n × query terms)
- **Persistent index** — reopen without re-indexing
- **CPU by default** — GPU used when available and supported
- **Multi-arch Docker image** — linux/amd64 + linux/arm64, no torch/CUDA

---

## Limitations

- Banglish transliteration uses a two-layer approach (dictionary + phonetic
  engine): common words are covered by ~250 curated entries, and any unknown
  token is handled by the rule-based phonetic fallback. The phonetic engine
  may not produce perfect Bangla for every romanization variant, but it
  ensures consistent query-document matching.
- The default local vector store is in-memory with file persistence; it is not
  optimized for very large corpora (FAISS/Qdrant extras will address this).
- The default RAG provider is extractive, not generative — it returns a
  concise extracted answer with a citation. For generative answers without
  an API key, use `llm_provider="slm"` (requires `semfuse[slm]`). For OpenAI,
  use `llm_provider="openai"` (requires `semfuse[rag]` + API key).
- The BM25 index is rebuilt in memory when the corpus changes; this is fine for
  the corpus sizes the local store targets.

---

## Roadmap

- [x] Phase 1 — Core foundation (embeddings, local store, semantic retrieval, persistence)
- [x] Phase 2 — Language & Banglish normalization
- [x] Phase 3 — Document ingestion (TXT/PDF/DOCX, chunking, dedup)
- [x] Phase 4 — Keyword & hybrid retrieval, fusion, collections
- [x] Phase 5 — Reranking (lexical + cross-encoder)
- [x] Phase 6 — RAG (LLM providers, citations)
- [x] Phase 7 — Evaluation (Recall@K, MRR, NDCG, Banglish benchmark)
- [x] Phase 8 — Production quality (CLI, CI, Docker, docs, examples)
- [ ] FAISS / Qdrant vector store backends (optional extras)
- [ ] Anthropic / Gemini / Ollama LLM providers
- [ ] Semantic chunking (in addition to recursive)
- [ ] Larger Banglish benchmark dataset

---

## Project Structure

```
semfuse/
├── pyproject.toml
├── README.md
├── LICENSE                          # Apache 2.0
├── CHANGELOG.md
├── Dockerfile                       # multi-stage, CPU-only torch
├── .github/workflows/
│   ├── ci.yml                       # ruff + mypy + pytest
│   └── release.yml                  # build + GitHub release + PyPI + Docker
│
├── src/semfuse/
│   ├── __init__.py                  # public API
│   ├── core/                        # client, config, types, enums, exceptions
│   ├── embeddings/                  # protocol, local ST, hashing, factory
│   ├── language/                    # detector, normalizer, banglish
│   ├── chunking/                    # base, recursive
│   ├── loaders/                     # text, pdf, docx, directory, factory
│   ├── vectorstores/                # base, local (numpy + JSON)
│   ├── retrieval/                   # semantic, keyword, hybrid, fusion
│   ├── reranking/                   # base, lexical, cross-encoder, factory
│   ├── rag/                         # pipeline, providers, prompts, template
│   ├── evaluation/                  # metrics, runner, banglish benchmark
│   ├── cli/                         # main (info/index/search/ask)
│   └── utils/                       # hashing, logging, paths, serialization
│
├── tests/                           # 153 tests
│   ├── unit/                        # init, metadata, dedup, vectorstore, etc.
│   ├── integration/                 # real sentence-transformers cross-lingual
│   ├── language/                    # detector, banglish
│   ├── retrieval/                   # keyword, hybrid, fusion
│   └── persistence/                 # roundtrip, files, mismatch
│
├── examples/
│   ├── basic.py                     # smallest example
│   ├── hybrid_and_rerank.py         # search modes + reranking
│   ├── ingestion.py                 # files, directories, chunking
│   └── rag.py                       # RAG with citations
│
└── docs/
    ├── architecture.md              # layer design + module map
    ├── architecture-decisions.md    # ADR log
    └── banglish.md                  # Banglish pipeline + lexicons + benchmark
```

---

## Contributing

Contributions are welcome. Please run `ruff check`, `mypy`, and `pytest` before
submitting changes.

```bash
pip install -e .[dev]
ruff check src/semfuse tests
mypy --python-version 3.12 src/semfuse
pytest
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
