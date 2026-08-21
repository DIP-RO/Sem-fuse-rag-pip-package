"""Basic SemFuse usage — the smallest possible example."""

from __future__ import annotations

from semfuse import SemFuse


def main() -> None:
    db = SemFuse()

    db.add("ঢাকা বাংলাদেশের রাজধানী।")
    db.add("Dhaka is the capital of Bangladesh.")
    db.add("The Eiffel Tower is in Paris.")

    for query in [
        "বাংলাদেশের রাজধানী কী?",
        "What is the capital of Bangladesh?",
        "Bangladesh er capital ki?",
    ]:
        print(f"\nQuery: {query}")
        for r in db.search(query, top_k=2):
            print(f"  {r.score:.4f}  {r.text}")


if __name__ == "__main__":
    main()
