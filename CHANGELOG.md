# Changelog

All notable changes to SemFuse are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
