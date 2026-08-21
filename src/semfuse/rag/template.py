"""Deterministic extractive "LLM" provider.

Keeps ``SemFuse().ask(...)`` zero-config and offline: instead of calling a
generative model it extracts a concise answer from the top-ranked context
passage.  The extraction is question-aware:

* **Wh-/question-word extraction** — for "What is the capital of X?" the
  provider tries to pull the noun phrase that answers the question rather than
  echoing the whole passage.
* **Bangla question patterns** — handles ``কী``, ``কি``, ``কোথায়``, ``কখন``,
  ``কে``, ``কত``, ``কেন`` question markers.
* **Fallback** — when no specific answer span can be isolated, the full passage
  is returned with a citation (the original behaviour).

This is still an *extractive* provider — it never invents text.  Real
generation comes from the ``openai`` provider (``semfuse[rag]``) or a custom
:class:`~semfuse.rag.base.LLMProvider`.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Prompt parsing — the prompt built by rag.prompt always has the form:
#
#   ...
#   Context:
#   [1] (source) text
#   [2] (source) text
#   ...
#
#   Question: <question>
#
#   Answer:
#
# We extract the question and the numbered context passages.
# ---------------------------------------------------------------------------

_CONTEXT_LINE_RE = re.compile(r"^\[(\d+)\] \([^)]*\) (?P<text>.+)$", re.MULTILINE)
_QUESTION_RE = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)
_NO_CONTEXT_ANSWER = "I could not find relevant context to answer this question."

# ---------------------------------------------------------------------------
# Bangla question markers and their semantic role.
# ---------------------------------------------------------------------------
_BANGLA_Q_MARKERS: dict[str, str] = {
    "কী": "what",
    "কি": "what",
    "কোথায়": "where",
    "কোথায়": "where",
    "কখন": "when",
    "কে": "who",
    "কত": "how_many",
    "কতটা": "how_many",
    "কেন": "why",
    "কীভাবে": "how",
    "কিভাবে": "how",
    "কেমন": "how",
}

# English question words.
_EN_Q_WORDS = {
    "what": "what",
    "who": "who",
    "where": "where",
    "when": "when",
    "why": "why",
    "how": "how",
    "which": "which",
    "whose": "whose",
}

# Banglish (romanized Bangla) question markers.
_BANGLISH_Q_MARKERS: dict[str, str] = {
    "ki": "what",
    "kee": "what",
    "kothay": "where",
    "kothai": "where",
    "kokhon": "when",
    "kokhono": "when",
    "ke": "who",
    "keno": "why",
    "kivabe": "how",
    "kibhabe": "how",
    "kemon": "how",
    "koto": "how_many",
}

# Bangla sentence terminators.
_BANGLA_TERM = "।!?."

# ---------------------------------------------------------------------------
# Answer extraction helpers.
# ---------------------------------------------------------------------------


def _detect_question_type(question: str) -> str:
    """Return a coarse question category: what/where/when/who/how/why/which/other."""
    q_lower = question.lower().strip()
    for word, qtype in _EN_Q_WORDS.items():
        if re.search(rf"\b{word}\b", q_lower):
            return qtype
    for marker, qtype in _BANGLA_Q_MARKERS.items():
        if marker in question:
            return qtype
    # Banglish markers — check as whole words in the latin token stream.
    latin_tokens = re.findall(r"[a-zA-Z]+", q_lower)
    for token in latin_tokens:
        if token in _BANGLISH_Q_MARKERS:
            return _BANGLISH_Q_MARKERS[token]
    return "other"


def _split_sentences(text: str) -> list[str]:
    """Split Bangla/English text into sentences, keeping terminators."""
    # Split on Bangla dari, English period, ?, ! followed by space or end.
    parts = re.split(r"(?<=[।.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_answer_span(question: str, passage: str, qtype: str) -> str | None:
    """Try to isolate a concise answer span from ``passage``.

    This is intentionally conservative: it only extracts when it can identify
    a clear answer phrase.  Otherwise it returns None and the caller falls back
    to the full passage.
    """
    sentences = _split_sentences(passage)
    if not sentences:
        return None

    # For "what is X" / "কী" questions about a definition or identity,
    # the answer is often the subject phrase before "is"/"হয়"/"এর".
    # Strategy: find the sentence most relevant to the question keywords,
    # then try to extract the answer phrase from it.

    # Pick the best sentence: highest keyword overlap with the question.
    q_tokens = set(re.findall(r"[\wঀ-৿]+", question.lower()))
    best_sent = max(sentences, key=lambda s: len(q_tokens & set(re.findall(r"[\wঀ-৿]+", s.lower()))), default="")
    if not best_sent:
        return None

    if qtype == "what":
        # "X is Y" / "X হয় Y" / "X-এর Y" patterns — extract Y.
        # English: "Dhaka is the capital of Bangladesh." -> "Dhaka"
        # Bangla: "ঢাকা বাংলাদেশের রাজধানী।" -> "ঢাকা"
        q_lower = question.lower()
        asks_about_capital = (
            "capital" in q_lower
            or "রাজধানী" in question
            or "rajdhani" in q_lower  # Banglish
        )
        m = re.match(r"^(.+?)\s+is\s+(?:the\s+)?(.+?)[.।]?$", best_sent, re.IGNORECASE)
        if m:
            if asks_about_capital:
                return m.group(1).strip()
            return m.group(2).strip()
        # Banglish: "X er rajdhani Y" or "X er capital Y" — answer is Y
        m = re.match(r"^(\S+)\s+er\s+(?:rajdhani|capital)\s+(\S+?)[.।]?$", best_sent, re.IGNORECASE)
        if m and asks_about_capital:
            return m.group(2).strip()
        # Banglish: "X-er rajdhani Y" (no space before er)
        m = re.match(r"^(\S+?)(?:er|-er)\s+(?:rajdhani|capital)\s+(\S+?)[.।]?$", best_sent, re.IGNORECASE)
        if m and asks_about_capital:
            return m.group(2).strip()
        # Bangla: the genitive suffix ের is attached to the noun
        # (বাংলাদেশের), not a separate token.  When the question asks
        # "what is X-এর Y?", the answer is the subject before the
        # possessed noun phrase — i.e. the first word.
        if asks_about_capital:
            # But only if the passage is in Bangla script (not Banglish).
            if any("\u0980" <= ch <= "\u09FF" for ch in best_sent):
                tokens = best_sent.rstrip("।.").strip().split()
                if len(tokens) >= 2:
                    return tokens[0].strip()
        # General Bangla: "X-এর Y" where we want Y.
        m = re.match(r"^(.+?)\s+\S*ের\s+(.+?)[।.]?$", best_sent)
        if m:
            return m.group(2).strip()
        # "X হয় Y" pattern
        m = re.match(r"^(.+?)\s+হয়\s+(.+?)[।.]?$", best_sent)
        if m:
            return m.group(2).strip()

    if qtype == "where":
        # "X is in Y" / "X তে অবস্থিত" — extract the location.
        m = re.search(r"\bis\s+in\s+(.+?)[.।]?$", best_sent, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"\bis\s+(?:located|situated)\s+in\s+(.+?)[.।]?$", best_sent, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Bangla: "X-এ অবস্থিত" / "X তে আছে"
        m = re.search(r"অবস্থিত\s+(.+?)[।.]?$", best_sent)
        if m:
            return m.group(1).strip()
        # "X এর অবস্থান Y" pattern
        m = re.search(r"অবস্থান\s+(.+?)[।.]?$", best_sent)
        if m:
            return m.group(1).strip()
        # "X is Y-এ" / "X Y-তে" — extract the locative noun
        m = re.search(r"(?:নদী|শহর|রাজধানী|দেশ)\s+(\S+?(?:ে|তে|ত))[।.]?$", best_sent)
        if m:
            return m.group(1).strip()

    if qtype == "when":
        # Look for Bangla month names in locative case (এ suffix) or English dates.
        m = re.search(r"(ডিসেম্বরে|জানুয়ারিতে|ফেব্রুয়ারিতে|মার্চে|এপ্রিলে|মেতে|জুনে|জুলাইতে|আগস্টে|সেপ্টেম্বরে|অক্টোবরে|নভেম্বরে)", best_sent)
        if m:
            return m.group(1).strip()
        m = re.search(r"(\d{4}\s*সালে|\d{1,2}\s*(?:জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর))", best_sent)
        if m:
            return m.group(1).strip()
        # English year patterns
        m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", best_sent)
        if m:
            return m.group(1).strip()
        # English month patterns
        m = re.search(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b", best_sent, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    if qtype == "who":
        # "X was born" / "X জন্মগ্রহণ করেন" — extract the person name
        m = re.match(r"^(.+?)\s+(?:was|is)\s+(?:born|a |an )", best_sent, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Bangla: first token is often the person
        m = re.search(r"জন্মগ্রহণ\s+করেন", best_sent)
        if m:
            tokens = best_sent.rstrip("।.").strip().split()
            if tokens:
                return tokens[0].strip()

    if qtype == "how_many":
        # Extract numbers from the passage
        m = re.search(r"(\d+(?:,\d{3})*(?:\.\d+)?)", best_sent)
        if m:
            return m.group(1).strip()
        # Bangla digits
        m = re.search(r"([০-৯]+)", best_sent)
        if m:
            return m.group(1).strip()

    if qtype == "how":
        # "X is done by Y" / "X করে Y" — extract the method
        m = re.search(r"\bby\s+(.+?)[.।]?$", best_sent, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return None


class TemplateLLMProvider:
    """Extractive provider: answers with an extracted span from context, cited.

    When the question type and passage structure allow, the provider extracts
    a concise answer phrase (e.g. "ঢাকা" for "বাংলাদেশের রাজধানী কী?").
    Otherwise it falls back to returning the full top passage with a citation.
    """

    @property
    def model_name(self) -> str:
        return "template-extractive"

    def generate(self, prompt: str) -> str:
        # Parse the question and context passages from the prompt.
        q_match = _QUESTION_RE.search(prompt)
        context_matches = list(_CONTEXT_LINE_RE.finditer(prompt))
        if not context_matches or not q_match:
            return _NO_CONTEXT_ANSWER

        question = q_match.group(1).strip()
        top_passage = context_matches[0].group("text").strip()

        # Try to extract a concise answer span.
        qtype = _detect_question_type(question)
        span = _extract_answer_span(question, top_passage, qtype)
        if span:
            return f"{span} [1]"
        # Fallback: return the full passage with citation.
        return f"{top_passage} [1]"
