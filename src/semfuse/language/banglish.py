"""Banglish (Bengali written in Latin script) detection support and normalization.

Banglish has no standard orthography: the same Bangla word is romanized many
ways (``achhe``/``ache``, ``bhalo``/``valo``). This module provides:

* a curated marker lexicon of high-frequency romanized Bangla function words,
  used by the language detector to separate Banglish from plain English;
* :class:`BanglishNormalizer`, which canonicalizes spelling variants and can
  transliterate known tokens to Bangla script so that Banglish queries land
  closer to Bangla documents in both embedding and keyword space.

All processing is non-destructive: callers receive normalized copies and the
original text is always preserved (see ADR-0005).
"""

from __future__ import annotations

import re

_LATIN_TOKEN_RE = re.compile(r"[a-zA-Z]+")
# Collapse characters repeated 3+ times ("kiii" -> "ki", "bhaloooo" -> "bhalo").
_ELONGATION_RE = re.compile(r"(.)\1{2,}")

# Romanized Bangla function/common words that signal Banglish in Latin-only
# text. Deliberately excludes proper nouns ("dhaka", "bangladesh") and words
# that are common in English, so English prose does not trip the detector.
BANGLISH_MARKERS: frozenset[str] = frozenset(
    {
        "ami", "amar", "amake", "amra", "amader",
        "tumi", "tomar", "tomake", "tomra", "tomader",
        "apni", "apnar", "apnara", "se", "tar", "tara", "tader",
        "eta", "ota", "seta", "sheta", "egulo", "ogulo", "gulo",
        "ki", "keno", "kothay", "kothai", "kokhon", "kivabe", "kibhabe",
        "kemon", "kamon", "koto", "kar", "kake",
        "er", "ta", "ti", "ache", "achhe", "chilo", "chhilo", "nei", "nai",
        "na", "hobe", "hoye", "hoyeche", "hoyechhe", "hocche", "hochhe",
        "korbo", "korte", "kora", "koro", "korchi", "korche", "korlam",
        "bhalo", "valo", "kharap", "onek", "khub", "ekta", "ekhon", "akhon",
        "tokhon", "aj", "ajke", "kal", "kalke", "jonno", "jonne", "theke",
        "diye", "sathe", "shathe", "shate", "bolo", "bolen", "bole", "bolbo",
        "jabo", "jete", "jabe", "asbe", "ashbe", "eshe", "ese", "asche",
        "kintu", "ebong", "othoba", "jodi", "tahole", "tobe", "karon",
        "khabar", "khete", "lagbe", "lage", "chai", "chao", "dorkar",
        "khobor", "bhalobashi", "bhalobasha", "somoy", "kaj", "kotha",
        "rajdhani", "porikkha", "bhorti", "chhuti", "chuti",
    }
)

# Spelling-variant folding: alternate romanizations -> canonical form.
_VARIANTS: dict[str, str] = {
    "kee": "ki", "kii": "ki",
    "achhe": "ache", "achee": "ache",
    "chhilo": "chilo",
    "hoyechhe": "hoyeche",
    "hochhe": "hocche",
    "valo": "bhalo", "vhalo": "bhalo",
    "kamon": "kemon",
    "kano": "keno",
    "kothai": "kothay",
    "kibhabe": "kivabe",
    "akhon": "ekhon",
    "shathe": "sathe", "shate": "sathe",
    "ashbe": "asbe",
    "ese": "eshe",
    "jonne": "jonno",
    "onk": "onek",
    "amr": "amar",
    "tmr": "tomar",
    "chhuti": "chuti",
    "vorti": "bhorti",
}

# Canonical Banglish token -> Bangla script. Applied only to text already
# classified as Banglish, so English collisions ("ar", "na") are not a concern.
_TRANSLITERATIONS: dict[str, str] = {
    "ami": "আমি", "amar": "আমার", "amake": "আমাকে", "amra": "আমরা",
    "tumi": "তুমি", "tomar": "তোমার", "tomake": "তোমাকে",
    "apni": "আপনি", "apnar": "আপনার",
    "tara": "তারা", "eta": "এটা", "ota": "ওটা", "ekta": "একটা",
    "ki": "কি", "keno": "কেন", "kothay": "কোথায়", "kokhon": "কখন",
    "kivabe": "কীভাবে", "kemon": "কেমন", "koto": "কত",
    "er": "এর", "ache": "আছে", "chilo": "ছিল", "nei": "নেই", "nai": "নাই",
    "na": "না", "hobe": "হবে", "hoye": "হয়ে", "hoyeche": "হয়েছে",
    "hocche": "হচ্ছে", "korbo": "করব", "korte": "করতে", "kora": "করা",
    "korchi": "করছি", "korche": "করছে",
    "bhalo": "ভালো", "kharap": "খারাপ", "onek": "অনেক", "khub": "খুব",
    "ekhon": "এখন", "tokhon": "তখন", "aj": "আজ", "ajke": "আজকে",
    "kal": "কাল", "jonno": "জন্য", "theke": "থেকে", "diye": "দিয়ে",
    "sathe": "সাথে", "kintu": "কিন্তু", "ebong": "এবং", "jodi": "যদি",
    "tahole": "তাহলে", "karon": "কারণ",
    "khabar": "খাবার", "lagbe": "লাগবে", "lage": "লাগে", "chai": "চাই",
    "dorkar": "দরকার", "khobor": "খবর", "somoy": "সময়", "kaj": "কাজ",
    "kotha": "কথা", "bhalobashi": "ভালোবাসি",
    "rajdhani": "রাজধানী", "desh": "দেশ", "porikkha": "পরীক্ষা",
    "bhorti": "ভর্তি", "chuti": "ছুটি", "manush": "মানুষ", "pani": "পানি",
    "bari": "বাড়ি", "boi": "বই", "taka": "টাকা",
}


def banglish_marker_count(text: str) -> int:
    """Number of distinct Banglish marker words present in ``text``."""
    tokens = {t.lower() for t in _LATIN_TOKEN_RE.findall(text)}
    folded = {_VARIANTS.get(t, t) for t in tokens}
    return len((tokens | folded) & BANGLISH_MARKERS)


def looks_banglish(text: str) -> bool:
    """Heuristic: does Latin-script ``text`` read as romanized Bangla?

    True when at least two distinct marker words appear, or when a single
    marker makes up a large share of a very short text.
    """
    tokens = [t.lower() for t in _LATIN_TOKEN_RE.findall(text)]
    if not tokens:
        return False
    distinct = {_VARIANTS.get(t, t) for t in tokens} | set(tokens)
    markers = distinct & BANGLISH_MARKERS
    if len(markers) >= 2:
        return True
    marker_hits = sum(1 for t in tokens if t in BANGLISH_MARKERS or _VARIANTS.get(t, t) in BANGLISH_MARKERS)
    return marker_hits > 0 and marker_hits / len(tokens) >= 0.4


class BanglishNormalizer:
    """Canonicalizes Banglish spelling and optionally maps tokens to Bangla.

    ``normalize`` folds spelling variants to canonical romanizations so that
    ``achhe`` and ``ache`` embed and match identically. ``transliterate``
    additionally replaces known tokens with their Bangla-script equivalents,
    bridging Banglish queries to Bangla documents; unknown tokens (including
    embedded English words) pass through unchanged.
    """

    def normalize(self, text: str) -> str:
        def fold(match: re.Match[str]) -> str:
            token = _ELONGATION_RE.sub(r"\1", match.group(0).lower())
            return _VARIANTS.get(token, token)

        return _LATIN_TOKEN_RE.sub(fold, text)

    def transliterate(self, text: str) -> str:
        def to_bangla(match: re.Match[str]) -> str:
            token = _ELONGATION_RE.sub(r"\1", match.group(0).lower())
            token = _VARIANTS.get(token, token)
            return _TRANSLITERATIONS.get(token, token)

        return _LATIN_TOKEN_RE.sub(to_bangla, text)
