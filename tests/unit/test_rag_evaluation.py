"""Tests for RAG evaluation metrics: accuracy, faithfulness, citations, refusal."""

from __future__ import annotations

from semfuse.evaluation.rag_metrics import (
    answer_accuracy,
    citation_accuracy,
    faithfulness,
    is_refusal,
    rag_score,
    refusal_accuracy,
)

# ---------------------------------------------------------------------------
# Answer accuracy
# ---------------------------------------------------------------------------


def test_answer_accuracy_substring_bangla() -> None:
    assert answer_accuracy("ঢাকা [1]", "ঢাকা") == 1.0


def test_answer_accuracy_substring_english() -> None:
    assert answer_accuracy("Dhaka is the capital [1]", "Dhaka") == 1.0


def test_answer_accuracy_substring_miss() -> None:
    assert answer_accuracy("চট্টগ্রাম [1]", "ঢাকা") == 0.0


def test_answer_accuracy_case_insensitive() -> None:
    assert answer_accuracy("DHAKA [1]", "dhaka") == 1.0


def test_answer_accuracy_exact_match() -> None:
    # [1] is stripped, leaving "ঢাকা" which matches exactly.
    assert answer_accuracy("ঢাকা [1]", "ঢাকা", mode="exact") == 1.0
    assert answer_accuracy("ঢাকা বাংলাদেশের", "ঢাকা", mode="exact") == 0.0


def test_answer_accuracy_token_mode() -> None:
    # "Dhaka is the capital" contains all content tokens of "Dhaka capital"
    score = answer_accuracy("Dhaka is the capital [1]", "Dhaka capital", mode="token")
    assert score == 1.0


def test_answer_accuracy_token_partial() -> None:
    score = answer_accuracy("Dhaka [1]", "Dhaka capital city", mode="token")
    # 1 of 3 content tokens found
    assert 0.0 < score < 1.0


def test_answer_accuracy_empty() -> None:
    assert answer_accuracy("", "ঢাকা") == 0.0
    assert answer_accuracy("ঢাকা", "") == 0.0


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def test_faithfulness_grounded_bangla() -> None:
    evidence = ["ঢাকা বাংলাদেশের রাজধানী।"]
    assert faithfulness("ঢাকা বাংলাদেশের রাজধানী [1]", evidence) == 1.0


def test_faithfulness_grounded_english() -> None:
    evidence = ["Dhaka is the capital of Bangladesh."]
    assert faithfulness("Dhaka is the capital [1]", evidence) == 1.0


def test_faithfulness_hallucination() -> None:
    evidence = ["Dhaka is the capital of Bangladesh."]
    assert faithfulness("The moon is made of cheese [1]", evidence) == 0.0


def test_faithfulness_short_answer() -> None:
    evidence = ["ঢাকা বাংলাদেশের রাজধানী।"]
    assert faithfulness("ঢাকা [1]", evidence) == 1.0


def test_faithfulness_no_evidence() -> None:
    assert faithfulness("ঢাকা [1]", []) == 0.0


def test_faithfulness_empty_answer() -> None:
    assert faithfulness("", ["evidence"]) == 0.0


# ---------------------------------------------------------------------------
# Citation accuracy
# ---------------------------------------------------------------------------


def test_citation_accuracy_correct() -> None:
    evidence = ["ঢাকা বাংলাদেশের রাজধানী।", "পদ্মা একটি নদী।"]
    # [1] points to passage 1 which contains "ঢাকা" — correct.
    assert citation_accuracy("ঢাকা [1]", evidence) == 1.0


def test_citation_accuracy_out_of_range() -> None:
    evidence = ["ঢাকা বাংলাদেশের রাজধানী।"]
    # [5] doesn't exist — incorrect.
    assert citation_accuracy("ঢাকা [5]", evidence) == 0.0


def test_citation_accuracy_no_citations() -> None:
    evidence = ["ঢাকা বাংলাদেশের রাজধানী।"]
    assert citation_accuracy("ঢাকা", evidence) == 0.0


def test_citation_accuracy_multiple() -> None:
    evidence = ["ঢাকা বাংলাদেশের রাজধানী।", "পদ্মা একটি নদী।"]
    # [1] correct (ঢাকা in passage 1), [2] incorrect (ঢাকা not in passage 2)
    score = citation_accuracy("ঢাকা [1] [2]", evidence)
    assert score == 0.5


# ---------------------------------------------------------------------------
# Refusal accuracy
# ---------------------------------------------------------------------------


def test_is_refusal_english() -> None:
    assert is_refusal("I could not find relevant context.") is True
    assert is_refusal("I don't know the answer.") is True


def test_is_refusal_bangla() -> None:
    assert is_refusal("প্রাসঙ্গিক তথ্য নেই।") is True
    assert is_refusal("আমি জানি না।") is True


def test_is_not_refusal() -> None:
    assert is_refusal("ঢাকা বাংলাদেশের রাজধানী [1]") is False


def test_refusal_accuracy_should_refuse() -> None:
    assert refusal_accuracy("I could not find context.", should_refuse=True) == 1.0
    assert refusal_accuracy("ঢাকা [1]", should_refuse=True) == 0.0


def test_refusal_accuracy_should_not_refuse() -> None:
    assert refusal_accuracy("ঢাকা [1]", should_refuse=False) == 1.0
    assert refusal_accuracy("I don't know.", should_refuse=False) == 0.0


# ---------------------------------------------------------------------------
# Combined RAG score
# ---------------------------------------------------------------------------


def test_rag_score_normal_answer() -> None:
    score = rag_score(
        answer="ঢাকা [1]",
        expected="ঢাকা",
        evidence_passages=["ঢাকা বাংলাদেশের রাজধানী।"],
        should_refuse=False,
    )
    assert score["answer_accuracy"] == 1.0
    assert score["faithfulness"] == 1.0
    assert score["citation_accuracy"] == 1.0
    assert score["refusal_accuracy"] == 1.0


def test_rag_score_should_refuse() -> None:
    score = rag_score(
        answer="I could not find relevant context.",
        expected=None,
        evidence_passages=[],
        should_refuse=True,
    )
    assert score["answer_accuracy"] == 1.0
    assert score["refusal_accuracy"] == 1.0


def test_rag_score_hallucination() -> None:
    score = rag_score(
        answer="The moon is cheese [1]",
        expected="ঢাকা",
        evidence_passages=["ঢাকা বাংলাদেশের রাজধানী।"],
        should_refuse=False,
    )
    assert score["answer_accuracy"] == 0.0
    assert score["faithfulness"] == 0.0


# ---------------------------------------------------------------------------
# RAG evaluator (integration with SemFuse)
# ---------------------------------------------------------------------------


def test_rag_evaluator_runs() -> None:
    import tempfile

    from semfuse import SemFuse
    from semfuse.evaluation.rag_runner import RAGEvalSample, RAGEvaluator

    with tempfile.TemporaryDirectory() as td:
        db = SemFuse(storage_path=td)
        db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital_bn", source="capital_bn")
        db.add("পদ্মা একটি বড় নদী।", document_id="river_bn", source="river_bn")

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

        evaluator = RAGEvaluator(db)
        report = evaluator.evaluate(samples, k_values=(1, 3))

        assert report.sample_count == 2
        assert "answer_accuracy" in report.metrics
        assert "faithfulness" in report.metrics
        assert "citation_accuracy" in report.metrics
        assert "refusal_accuracy" in report.metrics
        assert "hit@1" in report.metrics
        assert "mrr" in report.metrics
        assert len(report.per_query) == 2


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------


def test_ablation_runner_runs() -> None:
    from semfuse.evaluation.ablation import AblationConfig, AblationRunner
    from semfuse.evaluation.rag_runner import RAGEvalSample

    documents = [
        ("capital_bn", "ঢাকা বাংলাদেশের রাজধানী।"),
        ("river_bn", "পদ্মা একটি বড় নদী।"),
    ]
    samples = [
        RAGEvalSample(
            question="বাংলাদেশের রাজধানী কী?",
            expected_answer="ঢাকা",
            relevant_document_ids=frozenset({"capital_bn"}),
        ),
    ]

    runner = AblationRunner(documents, samples, k_values=(1, 3))
    # Run just 2 configs to keep the test fast.
    configs = [
        AblationConfig(name="baseline"),
        AblationConfig(name="with-rerank", reranker="lexical"),
    ]
    report = runner.run_all(configs)
    assert len(report.results) == 2
    summary = report.summary()
    assert "baseline" in summary
    assert "with-rerank" in summary
    assert "answer_accuracy" in summary


def test_ablation_report_to_dict() -> None:
    from semfuse.evaluation.ablation import AblationConfig, AblationRunner
    from semfuse.evaluation.rag_runner import RAGEvalSample

    documents = [("d1", "ঢাকা বাংলাদেশের রাজধানী।")]
    samples = [
        RAGEvalSample(
            question="রাজধানী কী?",
            expected_answer="ঢাকা",
            relevant_document_ids=frozenset({"d1"}),
        ),
    ]

    runner = AblationRunner(documents, samples, k_values=(1,))
    report = runner.run_all([AblationConfig(name="test")])
    d = report.to_dict()
    assert "experiments" in d
    assert len(d["experiments"]) == 1
    assert d["experiments"][0]["name"] == "test"
