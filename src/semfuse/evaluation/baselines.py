"""Baseline comparison runner for RAG benchmarking.

Compares SemFuse's RAG pipeline against alternative approaches:

1. **Raw LLM** — ask the LLM directly without any retrieval (tests if the
   model already knows Bangla facts).
2. **SemFuse (template)** — extractive RAG with the template provider.
3. **SemFuse (SLM)** — generative RAG with the local SLM provider.
4. **SemFuse (OpenAI)** — generative RAG with OpenAI (if API key available).

Each baseline runs the same questions and is scored with the same RAG
metrics, so you can compare approaches fairly in a paper.

Usage::

    from semfuse.evaluation.baselines import BaselineRunner, BaselineResult
    from semfuse.evaluation.rag_runner import RAGEvalSample

    documents = [("capital_bn", "ঢাকা বাংলাদেশের রাজধানী।"), ...]
    samples = [RAGEvalSample(...), ...]

    runner = BaselineRunner(documents, samples)
    report = runner.run_all()
    print(report.summary())
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Protocol

from semfuse import SemFuse, SemFuseConfig
from semfuse.evaluation.rag_metrics import is_refusal, rag_score
from semfuse.evaluation.rag_runner import RAGEvalSample


@dataclass
class BaselineResult:
    """Result of a single baseline approach."""

    name: str
    description: str
    metrics: dict[str, float]
    per_query: list[dict[str, float | str | bool]] = field(default_factory=list)


@dataclass
class BaselineReport:
    """Aggregated results from all baselines."""

    results: list[BaselineResult] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable comparison table."""
        if not self.results:
            return "No baseline results."

        metric_names = list(self.results[0].metrics.keys())
        lines = []
        header = f"{'Baseline':<30} " + " ".join(f"{m:>14}" for m in metric_names)
        lines.append(header)
        lines.append("-" * len(header))

        for result in self.results:
            values = " ".join(
                f"{result.metrics.get(m, 0.0):>14.4f}" for m in metric_names
            )
            lines.append(f"{result.name:<30} {values}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Export as a dict for JSON serialization."""
        return {
            "baselines": [
                {
                    "name": r.name,
                    "description": r.description,
                    "metrics": r.metrics,
                    "per_query": r.per_query,
                }
                for r in self.results
            ]
        }


class _LLMProvider(Protocol):
    """Protocol for an LLM provider with a generate method."""

    def generate(self, prompt: str) -> str: ...


class BaselineRunner:
    """Runs baseline comparison experiments.

    Args:
        documents: List of (document_id, text) pairs to index.
        samples: Labeled RAG evaluation questions.
    """

    def __init__(
        self,
        documents: list[tuple[str, str]],
        samples: list[RAGEvalSample],
    ) -> None:
        self._documents = documents
        self._samples = samples

    def _run_raw_llm(self, provider: _LLMProvider, name: str, description: str) -> BaselineResult:
        """Run a raw LLM baseline (no retrieval, just the model)."""
        totals: dict[str, float] = {
            "answer_accuracy": 0.0,
            "faithfulness": 0.0,
            "citation_accuracy": 0.0,
            "refusal_accuracy": 0.0,
        }
        per_query: list[dict[str, float | str | bool]] = []

        for sample in self._samples:
            prompt = f"Question: {sample.question}\n\nAnswer:"
            answer = provider.generate(prompt).strip()

            if sample.should_refuse:
                score = {
                    "answer_accuracy": 1.0 if is_refusal(answer) else 0.0,
                    "faithfulness": 1.0,
                    "citation_accuracy": 1.0,
                    "refusal_accuracy": 1.0 if is_refusal(answer) else 0.0,
                }
            else:
                score = rag_score(
                    answer=answer,
                    expected=sample.expected_answer,
                    evidence_passages=[],  # No retrieval — can't check faithfulness.
                    should_refuse=False,
                )
                # Faithfulness is N/A for raw LLM (no evidence) — set to 0.
                score["faithfulness"] = 0.0
                score["citation_accuracy"] = 0.0  # No citations.

            row: dict[str, float | str | bool] = {
                "question": sample.question,
                "answer": answer[:100],
                "should_refuse": sample.should_refuse,
                **score,
            }
            per_query.append(row)
            for key in totals:
                totals[key] += score[key]

        n = len(self._samples)
        return BaselineResult(
            name=name,
            description=description,
            metrics={k: v / n for k, v in totals.items()},
            per_query=per_query,
        )

    def _run_semfuse(
        self,
        llm_provider: str,
        name: str,
        description: str,
        embedding_provider: str = "hashing",
    ) -> BaselineResult:
        """Run a SemFuse RAG baseline."""
        with tempfile.TemporaryDirectory() as td:
            cfg = SemFuseConfig(
                embedding_provider=embedding_provider,
                embedding_dimension=256 if embedding_provider == "hashing" else 384,
                storage_path=td,
                llm_provider=llm_provider,
            )
            db = SemFuse(config=cfg)
            for doc_id, text in self._documents:
                db.add(text, document_id=doc_id, source=doc_id)

            from semfuse.evaluation.rag_runner import RAGEvaluator

            evaluator = RAGEvaluator(db)
            report = evaluator.evaluate(self._samples, k_values=(1, 3, 5))

            # Extract only RAG metrics (not retrieval metrics).
            rag_metric_keys = (
                "answer_accuracy", "faithfulness",
                "citation_accuracy", "refusal_accuracy",
            )
            rag_metrics = {k: report.metrics.get(k, 0.0) for k in rag_metric_keys}

            return BaselineResult(
                name=name,
                description=description,
                metrics=rag_metrics,
                per_query=report.per_query,
            )

    def run_all(self) -> BaselineReport:
        """Run all available baselines and return a comparison report.

        Runs:
        1. SemFuse (template) — extractive RAG, always available
        2. SemFuse (SLM) — if llama-cpp-python installed
        3. Raw SLM — if llama-cpp-python installed (no retrieval)

        Returns:
            BaselineReport with all results.
        """
        report = BaselineReport()

        # 1. SemFuse template (always works, offline).
        report.results.append(
            self._run_semfuse(
                llm_provider="template",
                name="semfuse-template",
                description="SemFuse with extractive template RAG (offline, zero-dep)",
            )
        )

        # 2. SemFuse SLM (if available).
        try:
            from semfuse.rag.slm_provider import LocalSLMProvider  # noqa: F401

            report.results.append(
                self._run_semfuse(
                    llm_provider="slm",
                    name="semfuse-slm",
                    description="SemFuse with local SLM RAG (llama-cpp-python, ~450 MB)",
                )
            )

            # 3. Raw SLM (no retrieval).
            provider = LocalSLMProvider()
            report.results.append(
                self._run_raw_llm(
                    provider=provider,
                    name="raw-slm",
                    description="Raw Qwen2.5-0.5B without retrieval (tests model's Bangla knowledge)",
                )
            )
        except (ImportError, Exception) as exc:  # noqa: BLE001
            # SLM not installed or model load failed — skip SLM baselines.
            report.results.append(
                BaselineResult(
                    name="raw-slm",
                    description=f"Skipped (SLM unavailable: {exc})",
                    metrics={
                        "answer_accuracy": 0.0,
                        "faithfulness": 0.0,
                        "citation_accuracy": 0.0,
                        "refusal_accuracy": 0.0,
                    },
                )
            )

        return report
