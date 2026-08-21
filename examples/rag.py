"""RAG — answer questions from the index with citations.

The default LLM provider is "template": extractive, offline, no API key.
For generative answers install semfuse[rag] and pass
llm_provider="openai" (uses OPENAI_API_KEY).
"""

from __future__ import annotations

from semfuse import SemFuse


def main() -> None:
    db = SemFuse()

    db.add("ঢাকা বাংলাদেশের রাজধানী।", source="geography.txt")
    db.add("বিশ্ববিদ্যালয়ে ভর্তি পরীক্ষা ডিসেম্বরে অনুষ্ঠিত হবে।", source="admission.txt")
    db.add("The Eiffel Tower is in Paris.", source="travel.txt")

    for question in (
        "What is the capital of Bangladesh?",
        "bhorti porikkha kokhon hobe?",  # Banglish
    ):
        response = db.ask(question, top_k=3)
        print(f"\nQ: {question}")
        print(f"A: {response.answer}")
        for i, citation in enumerate(response.citations, start=1):
            print(f"   [{i}] ({citation.source}) {citation.text[:50]}")


if __name__ == "__main__":
    main()
