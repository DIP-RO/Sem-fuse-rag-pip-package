"""semfuse.evaluation subpackage."""

from __future__ import annotations

from semfuse.evaluation.ablation import (
    AblationConfig,
    AblationReport,
    AblationResult,
    AblationRunner,
    default_ablation_configs,
)
from semfuse.evaluation.banglish import banglish_benchmark
from semfuse.evaluation.baselines import BaselineReport, BaselineResult, BaselineRunner
from semfuse.evaluation.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from semfuse.evaluation.rag_metrics import (
    answer_accuracy,
    citation_accuracy,
    faithfulness,
    is_refusal,
    rag_score,
    refusal_accuracy,
)
from semfuse.evaluation.rag_runner import RAGEvalSample, RAGEvaluationReport, RAGEvaluator
from semfuse.evaluation.runner import EvalSample, EvaluationReport, RetrievalEvaluator

__all__ = [
    # Retrieval evaluation
    "EvalSample",
    "EvaluationReport",
    "RetrievalEvaluator",
    "banglish_benchmark",
    "hit_at_k",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    # RAG evaluation
    "RAGEvalSample",
    "RAGEvaluationReport",
    "RAGEvaluator",
    "answer_accuracy",
    "citation_accuracy",
    "faithfulness",
    "is_refusal",
    "rag_score",
    "refusal_accuracy",
    # Ablation experiments
    "AblationConfig",
    "AblationReport",
    "AblationResult",
    "AblationRunner",
    "default_ablation_configs",
    # Baseline comparison
    "BaselineReport",
    "BaselineResult",
    "BaselineRunner",
]
