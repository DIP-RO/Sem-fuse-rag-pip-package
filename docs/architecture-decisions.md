# Architecture Decision Log

Each entry follows: **Decision · Context · Choice · Alternatives · Reason**.

---

## ADR-0001 — Use `src/` layout

- **Decision:** Package source lives under `src/semfuse/`.
- **Context:** Need a professional, installable, import-safe package.
- **Choice:** `src/` layout with `pyproject.toml`.
- **Alternatives:** Flat layout.
- **Reason:** `src/` layout prevents accidental imports from the working directory
  and is the recommended modern Python packaging convention.

## ADR-0002 — Default embedding backend is `sentence-transformers`, lazy-loaded

- **Decision:** The default `EmbeddingProvider` wraps `sentence-transformers`,
  loaded lazily on first use and reused thereafter.
- **Context:** The product's core value is real multilingual semantic retrieval
  (Bangla, English, Banglish, cross-lingual). A hash-based embedding cannot bridge
  scripts.
- **Choice:** `LocalEmbeddingProvider` backed by `sentence-transformers`, default
  model `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, multilingual incl. Bangla).
- **Alternatives:** (a) Hashing-only default — cannot do cross-script retrieval.
  (b) API-backed default — requires keys/network, violates zero-config. (c) LaBSE —
  better cross-lingual but ~1.8GB; too heavy for a default.
- **Reason:** Best balance of quality, size, and zero-config usability. Model name
  is centralized in config, so it can be changed without code edits. Lazy loading
  keeps import fast and defers the model download to first use.

## ADR-0003 — Deterministic `HashingEmbeddingProvider` for offline unit tests

- **Decision:** Provide a `HashingEmbeddingProvider` (character n-gram hashing)
  used by unit tests.
- **Context:** Tests must be deterministic and must not require internet. The real
  model download is non-deterministic and network-dependent.
- **Choice:** A deterministic, dependency-free hashing embedder injected via config
  in tests.
- **Alternatives:** Mock the protocol with random vectors — non-deterministic and
  not a real provider implementation.
- **Reason:** Keeps unit tests fast/offline/deterministic while still exercising the
  real retrieval + persistence + dedup pipeline. Integration tests cover the real
  model with auto-skip when unavailable.

## ADR-0004 — Local vector store on numpy + JSON, no FAISS by default

- **Decision:** The default `VectorStore` is a pure-numpy in-memory store with
  file persistence (`.npy` + JSON sidecars).
- **Context:** Core must stay lightweight and work without external services.
- **Choice:** `LocalVectorStore` with cosine/dot/euclidean metrics, persisted to
  `storage_path`.
- **Alternatives:** FAISS as default — adds a heavy native dependency to core.
- **Reason:** FAISS/Qdrant are optional extras. The numpy backend is enough for the
  default use case and keeps the core dependency-light.

## ADR-0005 — Non-destructive text handling

- **Decision:** Original text is always preserved; normalization produces separate
  fields.
- **Context:** Destructive rewriting of user data is unacceptable.
- **Choice:** `Document`/`DocumentChunk` carry both `text` (original) and a
  `normalized_text` field; language is stored as metadata.
- **Alternatives:** In-place normalization.
- **Reason:** Reproducibility, auditability, and future re-embedding with different
  strategies.

## ADR-0006 — Index metadata guards against embedding mismatch

- **Decision:** Persist `embedding_model`, `embedding_dimension`,
  `embedding_version`. On load, compare against the active provider; raise
  `IndexVersionError` on mismatch.
- **Context:** Silently mixing incompatible embeddings produces wrong results.
- **Choice:** Explicit version check with a clear remediation message.
- **Alternatives:** Silent re-embed — slow and surprising.
- **Reason:** Correctness over convenience; clear errors over silent corruption.
