"""Ablation experiment runner for RAG benchmarking.

Runs controlled experiments that toggle individual SemFuse features on/off
to measure their contribution to retrieval and RAG quality. Designed for
academic benchmarking with an assistant professor / research collaborator.

Each ablation creates a SemFuse instance with specific features enabled or
disabled, runs the same benchmark dataset, and returns a comparison report.

Usage::

    from semfuse.evaluation.ablation import AblationRunner, AblationConfig
    from semfuse.evaluation.rag_runner import RAGEvalSample

    samples = [
        RAGEvalSample(question="বাংলাদেশের রাজধানী কী?", expected_answer="ঢাকা",
                      relevant_document_ids=frozenset({"capital_bn"})),
        RAGEvalSample(question="Bangladesh er capital ki?", expected_answer="ঢাকা",
                      relevant_document_ids=frozenset({"capital_bn"})),
        ...
    ]
    documents = [("capital_bn", "ঢাকা বাংলাদেশের রাজধানী।"), ...]

    runner = AblationRunner(documents, samples)
    report = runner.run_all()
    print(report.summary())
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field

from semfuse import SemFuse, SemFuseConfig
from semfuse.core.enums import FusionMethod, SearchMode
from semfuse.evaluation.rag_runner import RAGEvalSample, RAGEvaluationReport, RAGEvaluator


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment.

    Attributes:
        name: Human-readable name for the experiment.
        embedding_provider: "hashing" or "local".
        reranker: None, "lexical", or "cross-encoder".
        llm_provider: "template", "slm", or "openai".
        search_mode: "auto", "semantic", "keyword", or "hybrid".
        fusion_method: "weighted" or "rrf".
        semantic_weight: Weight for semantic retriever.
        keyword_weight: Weight for keyword retriever.
    """

    name: str
    embedding_provider: str = "hashing"
    embedding_dimension: int = 256
    reranker: str | None = None
    llm_provider: str = "template"
    search_mode: str = "auto"
    fusion_method: str = "weighted"
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    rag_confidence_threshold: float = 0.0


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    config: AblationConfig
    report: RAGEvaluationReport


@dataclass
class AblationReport:
    """Aggregated results from all ablation experiments."""

    results: list[AblationResult] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable comparison table."""
        if not self.results:
            return "No ablation results."

        # Collect all metric names from the first result.
        metric_names = list(self.results[0].report.metrics.keys())

        # Build header.
        lines = []
        header = f"{'Experiment':<35} " + " ".join(f"{m:>14}" for m in metric_names)
        lines.append(header)
        lines.append("-" * len(header))

        for result in self.results:
            values = " ".join(
                f"{result.report.metrics.get(m, 0.0):>14.4f}" for m in metric_names
            )
            lines.append(f"{result.config.name:<35} {values}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Export as a dict for JSON serialization / paper tables."""
        return {
            "experiments": [
                {
                    "name": r.config.name,
                    "config": {
                        "embedding_provider": r.config.embedding_provider,
                        "reranker": r.config.reranker,
                        "llm_provider": r.config.llm_provider,
                        "search_mode": r.config.search_mode,
                        "fusion_method": r.config.fusion_method,
                        "semantic_weight": r.config.semantic_weight,
                        "keyword_weight": r.config.keyword_weight,
                    },
                    "metrics": r.report.metrics,
                    "sample_count": r.report.sample_count,
                    "per_query": r.report.per_query,
                }
                for r in self.results
            ]
        }


# ---------------------------------------------------------------------------
# Predefined ablation configs
# ---------------------------------------------------------------------------


def default_ablation_configs() -> list[AblationConfig]:
    """Return the standard set of ablation experiments.

    These cover the key dimensions a research paper would report:
    1. Baseline (hashing, no reranking, template RAG)
    2. With lexical reranking
    3. Semantic-only (no keyword)
    4. Keyword-only (no semantic)
    5. RRF fusion instead of weighted
    6. With real embeddings (if sentence-transformers available)
    7. With SLM RAG (if llama-cpp-python available)
    """
    return [
        AblationConfig(name="baseline-hashing"),
        AblationConfig(name="with-lexical-rerank", reranker="lexical"),
        AblationConfig(name="semantic-only", search_mode="semantic"),
        AblationConfig(name="keyword-only", search_mode="keyword"),
        AblationConfig(name="hybrid-only", search_mode="hybrid"),
        AblationConfig(name="rrf-fusion", fusion_method="rrf"),
        AblationConfig(name="high-semantic-weight", semantic_weight=0.9, keyword_weight=0.1),
        AblationConfig(name="high-keyword-weight", semantic_weight=0.3, keyword_weight=0.7),
        # Confidence threshold ablations — does refusing weak matches help?
        AblationConfig(name="threshold-0.50", rag_confidence_threshold=0.50),
        AblationConfig(name="threshold-0.75", rag_confidence_threshold=0.75),
        AblationConfig(name="threshold-0.90", rag_confidence_threshold=0.90),
    ]


class AblationRunner:
    """Runs ablation experiments over a benchmark dataset.

    Args:
        documents: List of (document_id, text) pairs to index.
        samples: Labeled RAG evaluation questions.
        k_values: K values for retrieval metrics.
    """

    def __init__(
        self,
        documents: list[tuple[str, str]],
        samples: list[RAGEvalSample],
        k_values: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        self._documents = documents
        self._samples = samples
        self._k_values = k_values

    def run_single(self, config: AblationConfig) -> AblationResult:
        """Run a single ablation experiment."""
        with tempfile.TemporaryDirectory() as td:
            cfg = SemFuseConfig(
                embedding_provider=config.embedding_provider,
                embedding_dimension=config.embedding_dimension,
                storage_path=td,
                reranker=config.reranker,
                llm_provider=config.llm_provider,
                search_mode=SearchMode(config.search_mode),
                fusion_method=FusionMethod(config.fusion_method),
                semantic_weight=config.semantic_weight,
                keyword_weight=config.keyword_weight,
                rag_confidence_threshold=config.rag_confidence_threshold,
            )
            db = SemFuse(config=cfg)
            # Index documents with their IDs as source.
            for doc_id, text in self._documents:
                db.add(text, document_id=doc_id, source=doc_id)

            evaluator = RAGEvaluator(db)
            report = evaluator.evaluate(self._samples, k_values=self._k_values)
            return AblationResult(config=config, report=report)

    def run_all(
        self, configs: list[AblationConfig] | None = None
    ) -> AblationReport:
        """Run all ablation experiments and return a comparison report.

        Args:
            configs: Ablation configs to run. If None, uses
                :func:`default_ablation_configs`.

        Returns:
            AblationReport with all results.
        """
        if configs is None:
            configs = default_ablation_configs()

        report = AblationReport()
        for config in configs:
            result = self.run_single(config)
            report.results.append(result)
        return report
