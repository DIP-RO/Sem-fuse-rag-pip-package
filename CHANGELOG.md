# Changelog

All notable changes to SemFuse are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-08-21

### Added — Phase 2 (Language & Banglish normalization)

- Banglish detection: Latin-only text with romanized-Bangla marker words is
  classified as `Language.BANGLISH` (curated marker lexicon; proper nouns and
  English-common words excluded to avoid misclassifying English).
- `BanglishNormalizer`: spelling-variant folding (`achhe → ache`,
  `valo → bhalo`), elongation collapse (`kiii → ki`), and token
  transliteration to Bangla script (`rajdhani → রাজধানী`).
- Language-aware `normalize_for_search`: Banglish queries and chunks are
  transliterated before embedding/keyword matching, bridging Banglish→Bangla
  retrieval in both semantic and lexical space.
- Zero-width character stripping (ZWJ/ZWNJ/BOM) in `normalize_text`.
- `LanguageDetector` / `TextNormalizer` protocols; `docs/banglish.md`.

### Added — Phase 3 (Document ingestion)

- `RecursiveCharacterChunker`: paragraph → line → sentence (incl. Bangla dari
  `।`) → word splitting with configurable size/overlap; `TextChunker` protocol.
- Loaders: `TextLoader` (.txt/.md), `PdfLoader` (pypdf, one document per
  page), `DocxLoader` (python-docx); `DocumentLoader` protocol and
  extension-dispatch factory with actionable install hints.
- Client APIs: `add_file(...)` and `add_directory(...)`; long texts now chunk
  recursively with per-chunk language detection and content-hash dedup.

### Added — Phase 4 (Keyword & hybrid retrieval)

- `KeywordRetriever`: BM25 (Okapi) over normalized chunk text with a
  Bangla-aware tokenizer; scores max-normalized to [0, 1].
- `fuse_results`: weighted score fusion and reciprocal rank fusion (RRF).
- `HybridRetriever`: semantic + keyword, fused (default weights 0.7/0.3).
- Search modes are now real: `semantic`, `keyword`, `hybrid`; `auto` resolves
  to hybrid. Config: `fusion_method`, `semantic_weight`, `keyword_weight`.
- Collections: `list_collections()`; `VectorStore.chunks()` accessor.

### Added — Phase 5 (Reranking)

- `Reranker` protocol; `LexicalReranker` (deterministic, offline, Dice
  overlap blended with retrieval score) and `CrossEncoderReranker`
  (multilingual mMARCO cross-encoder, lazy-loaded).
- Config: `reranker`, `reranker_model`, `rerank_candidates`; per-call
  `search(..., rerank=True/False)` override.

### Added — Phase 6 (RAG)

- `LLMProvider` protocol; `TemplateLLMProvider` (extractive, offline,
  zero-config default) and `OpenAILLMProvider` (`semfuse[rag]`).
- `RAGPipeline` and `db.ask(question)`: retrieve → numbered-citation prompt →
  generate; returns typed `RAGResponse` (answer, model, citations, prompt).
- Config: `llm_provider`, `llm_model`, `llm_options`.

### Added — Phase 7 (Evaluation)

- Metrics: `recall_at_k`, `hit_at_k`, `mrr`, `ndcg_at_k`.
- `RetrievalEvaluator` + `EvalSample` → `EvaluationReport` with aggregate and
  per-query numbers.
- Built-in Banglish benchmark (`banglish_benchmark()`): Bangla corpus with
  labeled Banglish queries; scores hit@3 ≥ 0.8 even under the offline hashing
  provider, demonstrating the normalization gain.

### Added — Phase 8 (Production quality)

- CLI subcommands: `index` (files/dirs/inline text), `search`
  (`--mode/--top-k/--json`), `ask`; provider/model/storage/collection flags.
- GitHub Actions CI: ruff + mypy + offline pytest matrix (3.10–3.13).
- Docker support: multi-stage `Dockerfile` (CPU-only torch, ≈1.8 GB) with a
  single `/data` volume for both the index and the model cache; `.dockerignore`.
- Release pipeline (`release.yml`): pushing a `v*` tag builds distributions,
  creates a GitHub Release with artifacts and notes, pushes the image to GHCR,
  and publishes to PyPI (via Trusted Publishing, once configured).
- Examples: `ingestion.py`, `hybrid_and_rerank.py`, `rag.py`.
- Docs: `docs/banglish.md`, refreshed architecture docs, ADR-0007..0009.

### Changed

- `db.info()` / `db.explain()` now report search mode, fusion method,
  reranker, and LLM provider; `explain` shows the transliterated query.
- BM25/keyword tokenizer keeps Bangla words whole (Python's `\w` drops
  combining vowel signs).

## [0.1.0]

### Added — Phase 1 (Core foundation)

- `SemFuse` public client with zero-config initialization.
- `SemFuseConfig` with validated, overridable configuration.
- `EmbeddingProvider` protocol and two implementations:
  - `LocalEmbeddingProvider` (sentence-transformers, lazy-loaded, reused).
  - `HashingEmbeddingProvider` (deterministic, offline, for tests).
- `VectorStore` protocol and `LocalVectorStore` (numpy + JSON persistence).
- `SemanticRetriever` and `Retriever` protocol.
- Typed document model: `Document`, `DocumentChunk`, `SearchResult`,
  `IndexInfo`, `CollectionInfo`.
- Enums: `Language`, `SearchMode`, `SimilarityMetric`, `FusionMethod`.
- Custom exceptions with actionable messages, including `IndexVersionError`
  for embedding model/dimension mismatches.
- Language detection (`detect_language`) and non-destructive text
  normalization (`normalize_text`).
- Content-hash deduplication of chunks.
- Metadata filtering on search.
- `db.info()` and `db.explain()` diagnostic APIs.
- Minimal CLI (`semfuse info`).
- Architecture documentation and decision log.
- Unit, persistence, language, and integration test suites.
