"""Retrieval evaluation runner.

Evaluates a SemFuse-compatible search callable against a labeled dataset of
queries and relevant document ids, reporting Recall@K, Hit@K, NDCG@K, and MRR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from semfuse.core.types import SearchResult
from semfuse.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k


@dataclass(frozen=True)
class EvalSample:
    """One labeled evaluation query."""

    query: str
    relevant_document_ids: frozenset[str]


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated metrics plus the per-query breakdown."""

    metrics: dict[str, float]
    sample_count: int
    per_query: list[dict[str, float | str]] = field(default_factory=list)

    def __repr__(self) -> str:
        summary = ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.metrics.items()))
        return f"EvaluationReport(samples={self.sample_count}, {summary})"


class _Searchable(Protocol):
    def search(self, query: str, *, top_k: int) -> list[SearchResult]: ...


class RetrievalEvaluator:
    """Runs a labeled dataset against a client (or anything with ``search``)."""

    def __init__(self, client: _Searchable) -> None:
        self._client = client

    def evaluate(
        self,
        samples: list[EvalSample],
        k_values: tuple[int, ...] = (1, 5),
    ) -> EvaluationReport:
        if not samples:
            return EvaluationReport(metrics={}, sample_count=0)
        max_k = max(k_values)
        totals: dict[str, float] = {"mrr": 0.0}
        for k in k_values:
            totals[f"recall@{k}"] = 0.0
            totals[f"hit@{k}"] = 0.0
            totals[f"ndcg@{k}"] = 0.0
        per_query: list[dict[str, float | str]] = []
        for sample in samples:
            results = self._client.search(sample.query, top_k=max_k)
            ranked = [r.document_id or "" for r in results]
            relevant = set(sample.relevant_document_ids)
            row: dict[str, float | str] = {"query": sample.query, "mrr": mrr(relevant, ranked)}
            totals["mrr"] += float(row["mrr"])
            for k in k_values:
                row[f"recall@{k}"] = recall_at_k(relevant, ranked, k)
                row[f"hit@{k}"] = hit_at_k(relevant, ranked, k)
                row[f"ndcg@{k}"] = ndcg_at_k(relevant, ranked, k)
                totals[f"recall@{k}"] += float(row[f"recall@{k}"])
                totals[f"hit@{k}"] += float(row[f"hit@{k}"])
                totals[f"ndcg@{k}"] += float(row[f"ndcg@{k}"])
            per_query.append(row)
        n = len(samples)
        return EvaluationReport(
            metrics={name: value / n for name, value in totals.items()},
            sample_count=n,
            per_query=per_query,
        )
