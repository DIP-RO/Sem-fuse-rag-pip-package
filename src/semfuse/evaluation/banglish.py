"""Built-in Banglish→Bangla retrieval benchmark.

A small, curated fixture: Bangla corpus documents paired with Banglish (and a
few English) queries whose answers live in specific documents. Used to measure
whether Banglish normalization actually improves retrieval — SemFuse does not
publish benchmark numbers that are not backed by runnable evaluations.
"""

from __future__ import annotations

from semfuse.evaluation.runner import EvalSample

# (document_id, text)
BENCHMARK_DOCUMENTS: tuple[tuple[str, str], ...] = (
    ("capital", "ঢাকা বাংলাদেশের রাজধানী।"),
    ("population", "বাংলাদেশের জনসংখ্যা প্রায় সতেরো কোটি।"),
    ("language", "বাংলাদেশের রাষ্ট্রভাষা বাংলা।"),
    ("food", "বাংলাদেশের মানুষের প্রধান খাবার ভাত ও মাছ।"),
    ("river", "পদ্মা বাংলাদেশের একটি বড় নদী।"),
    ("admission", "বিশ্ববিদ্যালয়ে ভর্তি পরীক্ষা ডিসেম্বরে অনুষ্ঠিত হবে।"),
    ("holiday", "আগামী সপ্তাহে স্কুল ছুটি আছে।"),
    ("weather", "আজ ঢাকায় আবহাওয়া খুব ভালো।"),
)

BENCHMARK_QUERIES: tuple[tuple[str, str], ...] = (
    # (banglish/english query, relevant document_id)
    ("Bangladesh er rajdhani kothay?", "capital"),
    ("desh er rajdhani ki?", "capital"),
    ("bhorti porikkha kokhon hobe?", "admission"),
    ("school er chuti ache ki?", "holiday"),
    ("ajke weather kemon ache?", "weather"),
    ("manush ki khabar khay?", "food"),
)


def banglish_benchmark() -> tuple[list[tuple[str, str]], list[EvalSample]]:
    """Return ``(documents, samples)`` for the built-in Banglish benchmark.

    ``documents`` is a list of ``(document_id, text)`` pairs to index (pass the
    id as ``document_id`` so metric matching works); ``samples`` are the
    labeled queries.
    """
    documents = list(BENCHMARK_DOCUMENTS)
    samples = [
        EvalSample(query=query, relevant_document_ids=frozenset({doc_id}))
        for query, doc_id in BENCHMARK_QUERIES
    ]
    return documents, samples
