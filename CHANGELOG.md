# Changelog

All notable changes to SemFuse are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-08-22

### Added — Local SLM provider (no API key needed)

- `LocalSLMProvider` (`llm_provider="slm"`): uses
  [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
  (500M params, ~1 GB) for generative RAG answers on CPU — no OpenAI key,
  no network after initial download. Lazy-loaded, model cached locally.
- New `semfuse[slm]` extra: `transformers>=4.40` + `torch>=2.0`.
- Factory now supports three providers: `template` (extractive, default),
  `slm` (local generative), `openai` (optional, external API).
- OpenAI is no longer the primary recommended path — the SLM provider
  makes SemFuse fully self-contained for generative RAG.

### Added — Phonetic transliteration engine

- `semfuse.language.phonetic`: rule-based phonetic Banglish→Bangla
  transliteration that covers **any** romanized token, not just dictionary
  entries. Uses greedy longest-match over consonant/vowel mapping tables
  (Avro Phonetic-style).
- Two-layer transliteration in `BanglishNormalizer`:
  1. **Dictionary layer** — high-confidence curated mappings for common
     words (~250 entries).
  2. **Phonetic fallback** — any unknown token is transliterated by rule,
     giving effectively unlimited coverage without a 1M-entry dictionary.
- English words embedded in Banglish pass through unchanged at both layers.

### Improved — Expanded dictionary

- Curated transliteration dictionary expanded from ~150 to ~250 entries:
  government/politics, education, geography/nature, food/agriculture,
  body/health, time/calendar, social/family, commerce, emotions, common
  verbs, question/function words, and place names.

### Changed

- README updated to feature the SLM provider as the recommended generative
  path; OpenAI reframed as optional.
- `semfuse[all]` extra now includes `transformers` and `torch`.
- Factory error message lists all three supported providers.

## [0.3.0] — 2026-08-22

## [0.3.0] — 2026-08-22

### Fixed — Bangla RAG answer quality

- The extractive `TemplateLLMProvider` now produces **concise extracted
  answers** instead of echoing the full passage. For example,
  `ask("বাংলাদেশের রাজধানী কী?")` returns `"ঢাকা [1]"` instead of the full
  `"ঢাকা বাংলাদেশের রাজধানী। [1]"`.
- Question-aware extraction handles Bangla (`কী`, `কি`, `কোথায়`, `কখন`),
  English (`what`, `where`, `when`, `who`, `how`, `why`), and Banglish
  (`ki`, `kothay`, `kokhon`, `keno`, `kivabe`) question markers.
- Pattern-based answer span extraction: "X is Y" → extract subject or
  predicate based on question type; Bangla genitive `ের` suffix handling;
  date/time extraction for "when" questions.

### Optimized — Vector store

- Replaced O(n²) `np.vstack`-per-add with a **pre-allocated growable buffer**
  (capacity doubles when exhausted). `add` / `add_many` are now amortized
  O(1) per chunk.
- Search uses `np.argpartition` (O(n) average) for top-k selection instead
  of full `np.argsort` (O(n log n)). Only the k candidates are sorted.
- `add_many` now does within-batch deduplication (not just against existing
  chunks), so adding duplicate texts in a single batch is correctly deduped.
- `delete` uses index shifting instead of rebuilding the entire matrix.

### Optimized — BM25 keyword retrieval

- Replaced full-corpus scan with an **inverted index** (`term → list of
  (doc_idx, tf)` postings). Scoring now only touches documents that contain
  at least one query term — O(sum of postings list lengths) instead of
  O(n × |query_terms|).

### Fixed — Hybrid fusion score normalization

- Weighted fusion now **min-max normalizes each retriever's scores** before
  applying weights. This fixes the issue where semantic scores (cosine,
  typically 0.5–1.0) and keyword scores (BM25 normalized, 0.0–1.0) had
  different distributions, causing one retriever to dominate the fusion.

### Improved — Banglish lexicon

- Expanded the marker lexicon from ~100 to ~180 words: added common verbs
  (`dekbo`, `pabo`, `bujhi`, `parbo`, `thakbe`), nouns (`shikkha`,
  `bidyaloy`, `sorkar`, `shastho`, `krishi`), places (`dhaka`,
  `chittagong`, `sylhet`, `coxsbazar`), numbers (`ek` through `dosh`),
  and more function words.
- Expanded transliteration dictionary from ~60 to ~150 entries, including
  all new markers plus place names, common nouns, and question words.
- Added Banglish question markers to the RAG template provider for
  Banglish-aware answer extraction.

### Added — Real-world test suite

- 15-document real-world corpus spanning Bangla, English, and mixed text
  across geography, education, history, culture, economy, and health topics.
- 10 cross-language retrieval tests (Bangla→Bangla, English→English,
  Banglish→Bangla, mixed).
- 6 RAG answer quality tests verifying concise extracted answers.
- Search mode correctness tests (semantic, keyword, hybrid).
- Reranking, metadata filtering, and explain diagnostics tests.
- Edge case tests: empty index, Unicode (ZWJ/ZWNJ/BOM), long documents,
  vector store buffer growth, delete compaction, persist/reload, score
  properties, top_k limits.
- Total: **215 tests** (up from 153), all passing.

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
