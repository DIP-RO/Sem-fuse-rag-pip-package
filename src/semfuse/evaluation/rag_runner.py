"""RAG evaluation runner — evaluates answer quality, not just retrieval.

Complements :class:`~semfuse.evaluation.runner.RetrievalEvaluator` by
measuring the *answer* produced by ``db.ask()`` against ground-truth
answers, with metrics for accuracy, faithfulness, citation correctness,
and refusal behavior.

Usage::

    from semfuse import SemFuse
    from semfuse.evaluation.rag_runner import RAGEvalSample, RAGEvaluator

    db = SemFuse()
    db.add_many(["ঢাকা বাংলাদেশের রাজধানী।", ...])

    samples = [
        RAGEvalSample(
            question="বাংলাদেশের রাজধানী কী?",
            expected_answer="ঢাকা",
            relevant_document_ids=frozenset({"capital_bn"}),
        ),
        RAGEvalSample(
            question="What is the population of Mars?",
            expected_answer=None,
            relevant_document_ids=frozenset(),
            should_refuse=True,
        ),
    ]

    report = RAGEvaluator(db).evaluate(samples, k_values=(1, 3, 5))
    print(report)
    # RAGEvaluationReport(samples=2, answer_accuracy=1.0000, faithfulness=1.0000,
    #   citation_accuracy=1.0000, refusal_accuracy=1.0000, hit@1=1.0000, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from semfuse.core.types import SearchResult
from semfuse.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from semfuse.evaluation.rag_metrics import rag_score


@dataclass(frozen=True)
class RAGEvalSample:
    """One labeled RAG evaluation question.

    Attributes:
        question: The question to ask.
        expected_answer: The ground-truth short answer (e.g. "ঢাকা"), or
            None if the system should refuse.
        relevant_document_ids: Document IDs that are relevant for retrieval
            scoring. Empty set if should_refuse=True.
        should_refuse: True if the system should respond with "I don't know"
            (no relevant context in the corpus).
    """

    question: str
    expected_answer: str | None = None
    relevant_document_ids: frozenset[str] = frozenset()
    should_refuse: bool = False


@dataclass(frozen=True)
class RAGEvaluationReport:
    """Aggregated RAG metrics plus per-query breakdown."""

    metrics: dict[str, float]
    sample_count: int
    per_query: list[dict[str, float | str | bool]] = field(default_factory=list)

    def __repr__(self) -> str:
        summary = ", ".join(f"{k}={v:.4f}" for k, v in sorted(self.metrics.items()))
        return f"RAGEvaluationReport(samples={self.sample_count}, {summary})"


class _Askable(Protocol):
    """Protocol for anything with an ``ask`` and ``search`` method."""

    def ask(self, question: str, *, top_k: int) -> object: ...

    def search(self, query: str, *, top_k: int) -> list[SearchResult]: ...


class RAGEvaluator:
    """Evaluates a SemFuse client's RAG output against labeled samples.

    Runs both retrieval metrics (Hit@K, NDCG@K, MRR, Recall@K) and RAG
    metrics (answer accuracy, faithfulness, citation accuracy, refusal
    accuracy) for each sample, then aggregates.
    """

    def __init__(self, client: _Askable) -> None:
        self._client = client

    def evaluate(
        self,
        samples: list[RAGEvalSample],
        k_values: tuple[int, ...] = (1, 3, 5),
        accuracy_mode: str = "substring",
    ) -> RAGEvaluationReport:
        """Run all samples and return an aggregated report.

        Args:
            samples: The labeled evaluation questions.
            k_values: K values for retrieval metrics.
            accuracy_mode: Matching mode for answer accuracy
                ("substring", "token", or "exact").

        Returns:
            RAGEvaluationReport with aggregated metrics + per-query breakdown.
        """
        if not samples:
            return RAGEvaluationReport(metrics={}, sample_count=0)

        max_k = max(k_values)
        # Initialize totals.
        totals: dict[str, float] = {
            "answer_accuracy": 0.0,
            "faithfulness": 0.0,
            "citation_accuracy": 0.0,
            "refusal_accuracy": 0.0,
            "mrr": 0.0,
        }
        for k in k_values:
            totals[f"recall@{k}"] = 0.0
            totals[f"hit@{k}"] = 0.0
            totals[f"ndcg@{k}"] = 0.0

        per_query: list[dict[str, float | str | bool]] = []

        for sample in samples:
            # Run retrieval for retrieval metrics.
            results = self._client.search(sample.question, top_k=max_k)
            ranked = [r.document_id or "" for r in results]
            relevant = set(sample.relevant_document_ids)

            # Run RAG for answer metrics.
            response = self._client.ask(sample.question, top_k=max_k)
            answer = getattr(response, "answer", str(response))
            citations = getattr(response, "citations", [])
            evidence_passages = [c.text for c in citations] if citations else [
                r.text for r in results
            ]

            # RAG metrics.
            rag = rag_score(
                answer=answer,
                expected=sample.expected_answer,
                evidence_passages=evidence_passages,
                should_refuse=sample.should_refuse,
                accuracy_mode=accuracy_mode,
            )

            # Retrieval metrics.
            row: dict[str, float | str | bool] = {
                "question": sample.question,
                "answer": answer[:100],  # Truncate for readability.
                "should_refuse": sample.should_refuse,
                "answer_accuracy": rag["answer_accuracy"],
                "faithfulness": rag["faithfulness"],
                "citation_accuracy": rag["citation_accuracy"],
                "refusal_accuracy": rag["refusal_accuracy"],
                "mrr": mrr(relevant, ranked),
            }
            for k in k_values:
                row[f"recall@{k}"] = recall_at_k(relevant, ranked, k)
                row[f"hit@{k}"] = hit_at_k(relevant, ranked, k)
                row[f"ndcg@{k}"] = ndcg_at_k(relevant, ranked, k)

            # Accumulate totals.
            for key in ("answer_accuracy", "faithfulness", "citation_accuracy",
                        "refusal_accuracy", "mrr"):
                totals[key] += float(row[key])
            for k in k_values:
                totals[f"recall@{k}"] += float(row[f"recall@{k}"])
                totals[f"hit@{k}"] += float(row[f"hit@{k}"])
                totals[f"ndcg@{k}"] += float(row[f"ndcg@{k}"])

            per_query.append(row)

        n = len(samples)
        return RAGEvaluationReport(
            metrics={name: value / n for name, value in totals.items()},
            sample_count=n,
            per_query=per_query,
        )
