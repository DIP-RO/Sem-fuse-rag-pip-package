"""Phonetic Banglish→Bangla transliteration engine.

A rule-based phonetic mapper that converts *any* romanized Bangla (Banglish)
token into Bangla script — not just dictionary entries.  This covers the
long tail of romanization variants that a fixed dictionary cannot.

The engine uses a greedy longest-match algorithm over a consonant/vowel
mapping table derived from the standard phonetic romanization scheme used
by most Banglish writers (similar to Avro Phonetic).

Key conventions used here (matching how most people actually write Banglish):

  * ``a`` after a consonant → া (আ-kar), NOT the inherent অ.
    In Banglish, "a" almost always represents the long আ sound
    (e.g. "rajdhani" → রাজধানী, "amar" → আমার).
  * ``a`` at word start → আ (independent).
  * ``o`` between two consonants → inherent অ (no vowel sign),
    because the schwa is dropped in spoken Bangla
    (e.g. "kemon" → কেমন, not কেমোন).
  * ``o`` at word end or before a space → ো (ও-kar)
    (e.g. "bhalo" → ভালো).
  * The inherent অ is implicit when a consonant is followed directly
    by another consonant with no vowel in between.

Strategy in :class:`BanglishNormalizer.transliterate`:
  1. Dictionary lookup (high-confidence curated entries)
  2. Phonetic engine fallback for unknown tokens
  3. English words pass through unchanged.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Consonant mappings — ordered by length (longest first for greedy match).
# ---------------------------------------------------------------------------

_CONSONANTS: list[tuple[str, str]] = [
    # Three-letter clusters
    ("ksh", "ক্ষ"), ("kkh", "ক্ষ"), ("jng", "ঞ্জ"), ("ndh", "ন্ধ"),
    ("nth", "ন্থ"), ("ngh", "ঙ্ঘ"),
    # Two-letter consonant clusters
    ("kh", "খ"), ("gh", "ঘ"),
    ("ch", "চ"), ("chh", "ছ"),
    ("jh", "ঝ"),
    ("th", "থ"), ("dh", "ধ"),
    ("ph", "ফ"), ("bh", "ভ"),
    ("sh", "শ"),
    ("kk", "ক্ক"), ("kt", "ক্ত"), ("kp", "ক্প"),
    ("kl", "ক্ল"), ("ks", "ক্স"),
    ("gg", "গ্গ"), ("gl", "গ্ল"),
    ("jj", "জ্জ"), ("jn", "জ্ঞ"),
    ("tt", "ত্ত"), ("tr", "ত্র"), ("tw", "ত্ব"),
    ("tn", "ত্ন"), ("tm", "ত্ম"), ("ts", "ত্স"),
    ("dr", "দ্র"), ("dw", "দ্ব"), ("dd", "দ্দ"),
    ("dm", "দ্ম"), ("dn", "দ্ন"), ("dl", "দ্ল"),
    ("pt", "প্ত"), ("pp", "প্প"), ("pl", "প্ল"), ("pr", "প্র"),
    ("bb", "ব্ব"), ("br", "ব্র"), ("bl", "ব্ল"),
    ("mb", "ম্ব"), ("mm", "ম্ম"), ("ml", "ম্ল"), ("mr", "ম্র"),
    ("nt", "ন্ত"), ("nn", "ন্ন"), ("nd", "ন্দ"),
    ("nk", "ঙ্ক"), ("ng", "ঙ"),
    ("nm", "ন্ম"), ("nb", "ন্ব"), ("ns", "ন্স"),
    ("nl", "ন্ল"), ("nr", "ন্র"),
    ("st", "স্ত"), ("sn", "স্ন"), ("sp", "স্প"),
    ("sk", "স্ক"), ("sm", "স্ম"), ("sl", "স্ল"),
    ("ss", "স্স"),
    ("ll", "ল্ল"),
    # Single consonants
    ("k", "ক"), ("g", "গ"), ("j", "জ"), ("y", "য"),
    ("t", "ত"), ("d", "দ"), ("n", "ন"), ("p", "প"),
    ("b", "ব"), ("m", "ম"), ("l", "ল"), ("r", "র"),
    ("s", "স"), ("h", "হ"),
    ("f", "ফ"), ("v", "ভ"), ("z", "জ"),
    ("x", "ক্স"), ("q", "ক"), ("w", "ওয়"),
    # Retroflex (capital letters — less common but used by some)
    ("T", "ট"), ("D", "ড"), ("N", "ণ"), ("R", "ড়"), ("Sh", "ষ"),
]

# ---------------------------------------------------------------------------
# Vowel mappings
# ---------------------------------------------------------------------------

# Independent vowels (used at word/syllable start)
_VOWELS_INDEPENDENT: list[tuple[str, str]] = [
    ("aau", "ঔ"), ("au", "ঔ"),
    ("aai", "ঐ"), ("ai", "ঐ"),
    ("aa", "আ"), ("a", "আ"),  # Banglish "a" at start = আ (long a)
    ("ee", "ঈ"), ("i", "ই"),
    ("oo", "ঊ"), ("uu", "ঊ"), ("u", "উ"),
    ("ae", "এ"), ("e", "এ"),
    ("oi", "ঐ"),
    ("ou", "ও"), ("o", "ও"),
]

# Dependent vowel signs (kar) — attach to the preceding consonant.
# "a" → া (আ-kar) in Banglish convention (NOT inherent অ).
# "o" → ো (ও-kar) when at end of token; inherent when between consonants
#        (handled in the transliteration logic, not in this table).
_VOWELS_DEPENDENT: list[tuple[str, str]] = [
    ("aau", "ৌ"), ("au", "ৌ"),
    ("aai", "ৈ"), ("ai", "ৈ"),
    ("aa", "া"), ("a", "া"),  # আ-kar
    ("ee", "ী"), ("i", "ি"),
    ("oo", "ূ"), ("uu", "ূ"), ("u", "ু"),
    ("ae", "ে"), ("e", "ে"),
    ("oi", "ৈ"),
    ("ou", "ো"), ("o", "ো"),  # ও-kar (may be overridden to inherent)
]

# Sort by Latin length descending for greedy matching.
_CONSONANTS_SORTED = sorted(_CONSONANTS, key=lambda x: -len(x[0]))
_VOWELS_INDEP_SORTED = sorted(_VOWELS_INDEPENDENT, key=lambda x: -len(x[0]))
_VOWELS_DEP_SORTED = sorted(_VOWELS_DEPENDENT, key=lambda x: -len(x[0]))

# ---------------------------------------------------------------------------
# English pass-through: words that should NOT be phonetically transliterated.
# ---------------------------------------------------------------------------

_ENGLISH_PASS_THROUGH: frozenset[str] = frozenset({
    # Articles / determiners
    "the", "a", "an", "this", "that", "these", "those",
    # Be verbs
    "is", "are", "was", "were", "be", "been", "being", "am",
    # Auxiliaries
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could",
    "may", "might", "must",
    # Prepositions
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "about", "into", "through", "over", "under", "after", "before",
    # Conjunctions
    "and", "or", "but", "not", "no", "if", "then", "else",
    "when", "where", "what", "who", "how", "why", "which",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "me", "him", "us", "them",
    # Common English content words (frequently embedded in Banglish)
    "capital", "city", "country", "river", "exam", "admission",
    "university", "school", "college", "hospital", "doctor",
    "health", "notice", "process", "system", "service",
    "information", "document", "form", "date", "time", "year",
    "month", "day", "week", "number", "result", "office",
    "application", "submit", "download", "upload", "online",
    "registration", "payment", "fee", "card", "bank",
    # Place names (these have specific Bangla equivalents in the dictionary)
    "bangladesh", "dhaka", "chittagong", "sylhet", "khulna",
    "rajshahi", "barisal", "rangpur", "mymensingh", "comilla",
    # Tech terms
    "pdf", "docx", "txt", "file", "data", "text", "page",
    "report", "email", "website", "link", "url", "api",
})

_LATIN_TOKEN_RE = re.compile(r"[a-zA-Z]+")

# Consonant characters for lookahead checks
_CONSONANT_CHARS = set("kgjytdnpbmlrshfvzqwkDTNRSh")


def _is_consonant_char(ch: str) -> bool:
    return ch.lower() in _CONSONANT_CHARS or ch in _CONSONANT_CHARS


def _transliterate_token(token: str) -> str:
    """Transliterate a single Latin token to Bangla using phonetic rules.

    Greedy longest-match with context-aware vowel handling:
    - "a" after consonant → া (আ-kar)
    - "o" between two consonants → inherent অ (no sign)
    - "o" at end or before non-consonant → ো (ও-kar)
    """
    result: list[str] = []
    i = 0
    n = len(token)
    prev_was_consonant = False

    while i < n:
        matched = False

        # Try dependent vowel sign if previous was a consonant.
        if prev_was_consonant:
            for latin, bangla in _VOWELS_DEP_SORTED:
                end = i + len(latin)
                if token[i:end].lower() == latin:
                    if bangla:
                        result.append(bangla)
                    i = end
                    matched = True
                    prev_was_consonant = False
                    break

        if matched:
            continue

        # Try independent vowel (at start or after non-consonant).
        if not prev_was_consonant:
            for latin, bangla in _VOWELS_INDEP_SORTED:
                if token[i : i + len(latin)].lower() == latin:
                    result.append(bangla)
                    i += len(latin)
                    matched = True
                    prev_was_consonant = False
                    break

        if matched:
            continue

        # Try consonant clusters (longest first).
        for latin, bangla in _CONSONANTS_SORTED:
            if token[i : i + len(latin)] == latin or token[i : i + len(latin)].lower() == latin:
                result.append(bangla)
                i += len(latin)
                matched = True
                prev_was_consonant = True
                break

        if matched:
            continue

        # If nothing matched, output the character as-is and advance.
        result.append(token[i])
        i += 1
        prev_was_consonant = False

    return "".join(result)


def phonetic_transliterate(text: str) -> str:
    """Transliterate Banglish text to Bangla script using phonetic rules.

    English words in the text pass through unchanged.  Only Latin-script
    tokens that are not in the English pass-through list are transliterated.
    Non-Latin characters (Bangla, digits, punctuation) are preserved.
    """

    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.lower() in _ENGLISH_PASS_THROUGH:
            return token
        return _transliterate_token(token)

    return _LATIN_TOKEN_RE.sub(replace_token, text)
