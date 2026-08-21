"""Phase 7: evaluation metrics, runner, and the Banglish benchmark."""

from __future__ import annotations

import math

import pytest

from semfuse import SemFuse
from semfuse.evaluation import (
    EvalSample,
    RetrievalEvaluator,
    banglish_benchmark,
    hit_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_hit_at_k() -> None:
    assert hit_at_k({"a"}, ["b", "a", "c"], 1) == 0.0
    assert hit_at_k({"a"}, ["b", "a", "c"], 2) == 1.0
    assert hit_at_k(set(), ["a"], 3) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k({"a", "b"}, ["a", "x", "b"], 3) == 1.0
    assert recall_at_k({"a", "b"}, ["a", "x", "y"], 3) == 0.5
    # Duplicate retrieved ids are not double-counted.
    assert recall_at_k({"a", "b"}, ["a", "a", "a"], 3) == 0.5


def test_mrr() -> None:
    assert mrr({"a"}, ["a", "b"]) == 1.0
    assert mrr({"a"}, ["b", "a"]) == 0.5
    assert mrr({"a"}, ["b", "c"]) == 0.0


def test_ndcg_at_k_hand_computed() -> None:
    # One relevant doc at rank 2 of 2 retrieved: DCG = 1/log2(3), IDCG = 1.
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k({"a"}, ["b", "a"], 2) == pytest.approx(expected)
    assert ndcg_at_k({"a"}, ["a"], 1) == 1.0
    assert ndcg_at_k({"a"}, ["b"], 1) == 0.0


def test_evaluator_perfect_and_zero(db: SemFuse) -> None:
    db.add("alpha alpha alpha", document_id="alpha")
    db.add("beta beta beta", document_id="beta")
    samples = [
        EvalSample(query="alpha", relevant_document_ids=frozenset({"alpha"})),
        EvalSample(query="beta", relevant_document_ids=frozenset({"beta"})),
    ]
    report = RetrievalEvaluator(db).evaluate(samples, k_values=(1,))
    assert report.sample_count == 2
    assert report.metrics["hit@1"] == 1.0
    assert report.metrics["mrr"] == 1.0
    assert len(report.per_query) == 2


def test_evaluator_empty_dataset(db: SemFuse) -> None:
    report = RetrievalEvaluator(db).evaluate([])
    assert report.sample_count == 0
    assert report.metrics == {}


def test_banglish_benchmark_offline(db: SemFuse) -> None:
    """The built-in benchmark must score well even with the hashing provider —
    this is the runnable evidence that Banglish normalization bridges scripts."""
    docs, samples = banglish_benchmark()
    for doc_id, text in docs:
        db.add(text, document_id=doc_id)
    report = RetrievalEvaluator(db).evaluate(samples, k_values=(1, 3))
    assert report.sample_count == len(samples)
    assert report.metrics["hit@3"] >= 0.8
    assert report.metrics["mrr"] >= 0.5
