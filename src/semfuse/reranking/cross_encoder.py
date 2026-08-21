"""Cross-encoder reranker backed by ``sentence-transformers`` (lazy-loaded).

The default model is multilingual (mMARCO-trained) so Bangla/English/Banglish
query–document pairs are scored meaningfully. Logits are squashed through a
sigmoid so reranked scores stay in [0, 1].
"""

from __future__ import annotations

import math
from typing import Any

from semfuse.core.exceptions import RerankingError
from semfuse.core.types import SearchResult
from semfuse.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderReranker:
    """Scores (query, document) pairs with a cross-encoder model."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading cross-encoder model %s ...", self._model_name)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - import guard
            raise RerankingError(
                "sentence-transformers is required for the cross-encoder "
                "reranker. Install it with `pip install sentence-transformers` "
                "or use reranker='lexical'."
            ) from exc
        try:
            self._model = CrossEncoder(self._model_name, device=self._device)
        except Exception as exc:
            raise RerankingError(
                f"Failed to load cross-encoder model {self._model_name!r}: {exc}. "
                "If this is a network issue, pre-cache the model or use "
                "reranker='lexical' for offline reranking."
            ) from exc

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if not results:
            return []
        self._ensure_loaded()
        pairs = [(query, r.text) for r in results]
        try:
            logits = self._model.predict(pairs, show_progress_bar=False)
        except Exception as exc:
            raise RerankingError(f"Cross-encoder scoring failed: {exc}") from exc
        rescored = [
            SearchResult(
                text=r.text,
                score=1.0 / (1.0 + math.exp(-float(logit))),
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                metadata=r.metadata,
                language=r.language,
                source=r.source,
                page=r.page,
            )
            for r, logit in zip(results, logits, strict=True)
        ]
        rescored.sort(key=lambda r: -r.score)
        return rescored[:top_k] if top_k is not None else rescored
