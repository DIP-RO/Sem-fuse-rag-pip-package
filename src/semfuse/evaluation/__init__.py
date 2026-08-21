"""semfuse.evaluation subpackage."""

from __future__ import annotations

from semfuse.evaluation.banglish import banglish_benchmark
from semfuse.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from semfuse.evaluation.runner import EvalSample, EvaluationReport, RetrievalEvaluator

__all__ = [
    "EvalSample",
    "EvaluationReport",
    "RetrievalEvaluator",
    "banglish_benchmark",
    "hit_at_k",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
