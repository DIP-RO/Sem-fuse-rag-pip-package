"""Phase 4: BM25 keyword retrieval."""

from __future__ import annotations

from semfuse import SemFuse
from semfuse.retrieval.keyword import KeywordRetriever, tokenize


def test_tokenize_multilingual() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]
    assert tokenize("ঢাকা বাংলাদেশের রাজধানী।") == ["ঢাকা", "বাংলাদেশের", "রাজধানী"]


def test_keyword_exact_term_wins(db: SemFuse) -> None:
    db.add("The quick brown fox jumps over the lazy dog.", document_id="fox")
    db.add("Stock markets rallied on Tuesday morning.", document_id="stocks")
    db.add("Photosynthesis converts light into chemical energy.", document_id="bio")
    results = db.search("photosynthesis energy", mode="keyword")
    assert results
    assert results[0].document_id == "bio"
    assert 0.0 < results[0].score <= 1.0  # normalized against the ideal BM25 score


def test_keyword_stopword_only_match_scores_low(db: SemFuse) -> None:
    """A document sharing only low-IDF stopwords must not be inflated to the
    top of the scale (this once outranked the true answer in hybrid mode)."""
    db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")
    db.add("The Eiffel Tower is in Paris.", document_id="eiffel")
    db.add("The quick brown fox is fast.", document_id="fox")
    results = db.search("What is the capital of Bangladesh?", mode="keyword")
    assert all(r.score < 0.5 for r in results)  # only "the"/"is"/"of" matched


def test_keyword_no_match_returns_empty(db: SemFuse) -> None:
    db.add("The quick brown fox.", document_id="fox")
    assert db.search("zzz qqq www", mode="keyword") == []


def test_keyword_rare_term_outranks_common(db: SemFuse) -> None:
    # "fox" appears once; "the" appears everywhere. IDF must favor the rare doc.
    db.add("the fox ran", document_id="fox")
    db.add("the cat sat on the mat", document_id="cat")
    db.add("the dog barked at the moon", document_id="dog")
    results = db.search("the fox", mode="keyword")
    assert results[0].document_id == "fox"


def test_keyword_respects_filter(db: SemFuse) -> None:
    db.add("CSE admission notice.", metadata={"department": "CSE"}, document_id="cse")
    db.add("EEE admission notice.", metadata={"department": "EEE"}, document_id="eee")
    results = db.search("admission notice", mode="keyword", filter={"department": "EEE"})
    assert results
    assert all(r.metadata["department"] == "EEE" for r in results)


def test_keyword_index_refreshes_on_mutation(db: SemFuse) -> None:
    db.add("alpha document", document_id="a")
    assert db.search("alpha", mode="keyword")
    db.add("beta document", document_id="b")
    results = db.search("beta", mode="keyword")
    assert results
    assert results[0].document_id == "b"
    db.clear()
    assert db.search("alpha", mode="keyword") == []


def test_keyword_retriever_empty_store(db: SemFuse) -> None:
    retriever = KeywordRetriever(db._store)
    assert retriever.retrieve("anything") == []


def test_keyword_matches_banglish_via_normalization(db: SemFuse) -> None:
    """Banglish query terms transliterate to Bangla and match lexically."""
    db.add("ঢাকা বাংলাদেশের রাজধানী।", document_id="capital")
    db.add("পদ্মা একটি বড় নদী।", document_id="river")
    results = db.search("desh er rajdhani ki?", mode="keyword")
    assert results
    assert results[0].document_id == "capital"
