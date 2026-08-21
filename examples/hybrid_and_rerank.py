"""Search modes (semantic / keyword / hybrid) and reranking."""

from __future__ import annotations

from semfuse import SemFuse


def main() -> None:
    db = SemFuse()

    db.add("ঢাকা বাংলাদেশের রাজধানী।")
    db.add("The Eiffel Tower is in Paris.")
    db.add("Photosynthesis converts light into chemical energy.")

    query = "Bangladesh er rajdhani kothay?"  # Banglish

    # The default mode is auto (= hybrid): semantic + BM25 keyword, fused.
    for mode in ("semantic", "keyword", "hybrid"):
        results = db.search(query, mode=mode, top_k=2)
        top = results[0].text if results else "(none)"
        print(f"{mode:>9}: {top}")

    # Rerank on demand (offline lexical reranker), or configure
    # SemFuse(reranker="cross-encoder") for model-based reranking.
    results = db.search(query, rerank=True, top_k=2)
    print(f" reranked: {results[0].text if results else '(none)'}")

    # See exactly how the query was processed.
    explanation = db.explain(query)
    print("detected:", explanation["detected_language"])
    print("normalized:", explanation["normalized_query"])


if __name__ == "__main__":
    main()
