"""Academic benchmarking example for SemFuse.

This script demonstrates how to use SemFuse's evaluation framework for
a research paper. It runs:

1. Retrieval evaluation (Hit@K, NDCG@K, MRR, Recall@K)
2. RAG evaluation (answer accuracy, faithfulness, citation accuracy, refusal accuracy)
3. Ablation experiments (with/without reranking, semantic-only, keyword-only, RRF, etc.)
4. Baseline comparison (SemFuse template vs SemFuse SLM vs raw SLM)

Usage::

    python examples/benchmarking.py

Output is a summary table suitable for inclusion in a paper.
"""

from __future__ import annotations

from semfuse import SemFuse
from semfuse.evaluation.ablation import AblationConfig, AblationRunner
from semfuse.evaluation.banglish import banglish_benchmark
from semfuse.evaluation.baselines import BaselineRunner
from semfuse.evaluation.rag_runner import RAGEvalSample, RAGEvaluator
from semfuse.evaluation.runner import EvalSample, RetrievalEvaluator


def main() -> None:
    # 1. Use the built-in Banglish benchmark dataset.
    documents, retrieval_samples = banglish_benchmark()

    # Build RAG samples (same queries, with expected answers).
    rag_samples = [
        RAGEvalSample(
            question="Bangladesh er rajdhani kothay?",
            expected_answer="ঢাকা",
            relevant_document_ids=frozenset({"capital"}),
        ),
        RAGEvalSample(
            question="desh er rajdhani ki?",
            expected_answer="ঢাকা",
            relevant_document_ids=frozenset({"capital"}),
        ),
        RAGEvalSample(
            question="bhorti porikkha kokhon hobe?",
            expected_answer="ডিসেম্বরে",
            relevant_document_ids=frozenset({"admission"}),
        ),
        RAGEvalSample(
            question="school er chuti ache ki?",
            expected_answer="ছুটি",
            relevant_document_ids=frozenset({"holiday"}),
        ),
        RAGEvalSample(
            question="ajke weather kemon ache?",
            expected_answer="ভালো",
            relevant_document_ids=frozenset({"weather"}),
        ),
        RAGEvalSample(
            question="manush ki khabar khay?",
            expected_answer="ভাত",
            relevant_document_ids=frozenset({"food"}),
        ),
        # Unanswerable — should refuse.
        RAGEvalSample(
            question="What is the distance to Andromeda?",
            expected_answer=None,
            relevant_document_ids=frozenset(),
            should_refuse=True,
        ),
    ]

    print("=" * 70)
    print("SemFuse Academic Benchmarking")
    print("=" * 70)

    # 2. Retrieval evaluation.
    print("\n--- Retrieval Evaluation ---")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        db = SemFuse(storage_path=td)
        for doc_id, text in documents:
            db.add(text, document_id=doc_id, source=doc_id)

        eval_samples = [
            EvalSample(query=s.query, relevant_document_ids=s.relevant_document_ids)
            for s in retrieval_samples
        ]
        retrieval_report = RetrievalEvaluator(db).evaluate(eval_samples, k_values=(1, 3, 5))
        print(retrieval_report)

    # 3. RAG evaluation.
    print("\n--- RAG Evaluation ---")
    with tempfile.TemporaryDirectory() as td:
        db = SemFuse(storage_path=td)
        for doc_id, text in documents:
            db.add(text, document_id=doc_id, source=doc_id)

        rag_report = RAGEvaluator(db).evaluate(rag_samples, k_values=(1, 3, 5))
        print(rag_report)
        print("\nPer-query breakdown:")
        for row in rag_report.per_query:
            print(f"  Q: {row['question']}")
            print(f"  A: {row['answer']}")
            print(f"  acc={row['answer_accuracy']:.2f} faith={row['faithfulness']:.2f} "
                  f"cite={row['citation_accuracy']:.2f} refuse={row['refusal_accuracy']:.2f}")

    # 4. Ablation experiments.
    print("\n--- Ablation Experiments ---")
    ablation_runner = AblationRunner(documents, rag_samples, k_values=(1, 3, 5))
    ablation_configs = [
        AblationConfig(name="baseline-hashing"),
        AblationConfig(name="with-lexical-rerank", reranker="lexical"),
        AblationConfig(name="semantic-only", search_mode="semantic"),
        AblationConfig(name="keyword-only", search_mode="keyword"),
        AblationConfig(name="hybrid-only", search_mode="hybrid"),
        AblationConfig(name="rrf-fusion", fusion_method="rrf"),
        AblationConfig(name="high-semantic", semantic_weight=0.9, keyword_weight=0.1),
        AblationConfig(name="high-keyword", semantic_weight=0.3, keyword_weight=0.7),
        # Confidence threshold ablations — does refusing weak matches help?
        AblationConfig(name="threshold-0.50", rag_confidence_threshold=0.50),
        AblationConfig(name="threshold-0.75", rag_confidence_threshold=0.75),
        AblationConfig(name="threshold-0.90", rag_confidence_threshold=0.90),
    ]
    ablation_report = ablation_runner.run_all(ablation_configs)
    print(ablation_report.summary())

    # 5. Baseline comparison.
    print("\n--- Baseline Comparison ---")
    baseline_runner = BaselineRunner(documents, rag_samples)
    baseline_report = baseline_runner.run_all()
    print(baseline_report.summary())

    print("\n" + "=" * 70)
    print("Done. Use these tables in your paper.")
    print("For JSON export: report.to_dict()")
    print("=" * 70)


if __name__ == "__main__":
    main()
