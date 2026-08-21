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
        # Pronouns
        "ami", "amar", "amake", "amra", "amader",
        "tumi", "tomar", "tomake", "tomra", "tomader",
        "apni", "apnar", "apnara", "apnake",
        "se", "she", "tar", "tara", "tader", "take",
        "ora", "oder", "otake",
        "eta", "ota", "seta", "sheta", "egulo", "ogulo", "gulo",
        "kichu", "kicchu", "kichhu",
        # Question words
        "ki", "keno", "kothay", "kothai", "kokhon", "kivabe", "kibhabe",
        "kemon", "kamon", "koto", "kar", "kake", "kona",
        # Common function words
        "er", "ta", "ti", "ache", "achhe", "chilo", "chhilo", "nei", "nai",
        "na", "hobe", "hoye", "hoyeche", "hoyechhe", "hocche", "hochhe",
        "holo", "hoilo", "hoy", "hoche",
        "korbo", "korte", "kora", "koro", "korchi", "korche", "korlam",
        "kore", "koreche", "korechhe", "korben",
        # Common verbs
        "bolo", "bolen", "bole", "bolbo", "bolchi", "bolche",
        "jabo", "jete", "jabe", "jachchi", "jacche", "gelam", "gechi",
        "asbe", "ashbe", "eshe", "ese", "asche", "aschi", "echilam",
        "dibo", "dite", "diyechi", "diyeche", "dao", "dan",
        "newa", "nite", "niyechi", "niyeche", "nilam", "nibo",
        "dekbo", "dekhte", "dekhi", "dekhe", "dekhechi", "dekhlam",
        "pabo", "paite", "peli", "peyechi", "peyeche",
        "bujhi", "bujhte", "bujhe", "bujhechi",
        "chai", "chao", "cchai", "chilam",
        # Adjectives / adverbs
        "bhalo", "valo", "kharap", "khrap", "onek", "onk", "khub",
        "ekta", "ekhon", "akhon", "tokhon", "tkhon",
        "aj", "ajke", "kal", "kalke", "somoy",
        # Postpositions / conjunctions
        "jonno", "jonne", "theke", "diye", "sathe", "shathe", "shate",
        "kintu", "ebong", "othoba", "jodi", "tahole", "tobe", "karon",
        "tai", "abar",
        # Common nouns
        "khabar", "khete", "lagbe", "lage", "dorkar",
        "khobor", "bhalobashi", "bhalobasha", "kaj", "kotha",
        "rajdhani", "porikkha", "porikkhar", "bhorti", "chhuti", "chuti",
        "manush", "manusher", "pani", "bari", "boi", "taka",
        "desh", "desher", "shahar", "nogor", "gram", "gramer",
        "shikkha", "shikkhok", "chhatra", "chhatro", "bidyaloy",
        "bishobidyaloy", "uni", "school", "college",
        "sorkar", "montrii", "montree", "podok", "podokkho",
        "potro", "notice", "suchona",
        "shastho", "daktar", "hospital", "oshudh",
        "krishi", "chash", "farmer", "din", "rat", "shokal", "belaka",
        # More common words
        "somvob", "somvobnoy", "parbo", "pari", "parte",
        "lagbo", "lagchi", "lagche",
        "hobe na", "hoyni", "hoini",
        "thakbe", "thakbo", "thakte",
        "korbe", "dibe", "nibe",
    }
)

# Spelling-variant folding: alternate romanizations -> canonical form.
_VARIANTS: dict[str, str] = {
    "kee": "ki", "kii": "ki",
    "achhe": "ache", "achee": "ache",
    "chhilo": "chilo",
    "hoyechhe": "hoyeche", "hoche": "hocche",
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
    "vorti": "bhorti", "vortir": "bhortir",
    "khrap": "kharap",
    "tkhon": "tokhon",
    "she": "se",
    "kicchu": "kichu", "kichhu": "kichu",
    "cchai": "chai",
    "hoilo": "holo",
    "nogor": "shahar",
    "chhatro": "chhatra",
    "montree": "montrii",
    "podokkho": "podok",
}

# Canonical Banglish token -> Bangla script. Applied only to text already
# classified as Banglish, so English collisions ("ar", "na") are not a concern.
_TRANSLITERATIONS: dict[str, str] = {
    # Pronouns
    "ami": "আমি", "amar": "আমার", "amake": "আমাকে", "amra": "আমরা", "amader": "আমাদের",
    "tumi": "তুমি", "tomar": "তোমার", "tomake": "তোমাকে", "tomra": "তোমরা", "tomader": "তোমাদের",
    "apni": "আপনি", "apnar": "আপনার", "apnara": "আপনারা", "apnake": "আপনাকে",
    "se": "সে", "tar": "তার", "tara": "তারা", "tader": "তাদের", "take": "তাকে",
    "ora": "ওরা", "oder": "ওদের",
    "eta": "এটা", "ota": "ওটা", "seta": "সেটা", "sheta": "সেটা",
    "ekta": "একটা", "egulo": "এগুলো", "ogulo": "ওগুলো", "gulo": "গুলো",
    "kichu": "কিছু",
    # Question words
    "ki": "কি", "keno": "কেন", "kothay": "কোথায়", "kokhon": "কখন",
    "kivabe": "কীভাবে", "kemon": "কেমন", "koto": "কত", "kona": "কোন",
    "kar": "কার", "kake": "কাকে",
    # Function words
    "er": "এর", "ta": "টা", "ti": "টি",
    "ache": "আছে", "chilo": "ছিল", "nei": "নেই", "nai": "নাই",
    "na": "না", "hobe": "হবে", "hoye": "হয়ে", "hoyeche": "হয়েছে",
    "hocche": "হচ্ছে", "holo": "হলো", "hoy": "হয়",
    "korbo": "করব", "korte": "করতে", "kora": "করা", "koro": "করো",
    "korchi": "করছি", "korche": "করছে", "korlam": "করলাম", "kore": "করে",
    "koreche": "করেছে", "korben": "করবেন",
    # Verbs
    "bolo": "বলো", "bolen": "বলেন", "bole": "বলে", "bolbo": "বলব",
    "bolchi": "বলছি", "bolche": "বলছে",
    "jabo": "যাব", "jete": "যেতে", "jabe": "যাবে", "gelam": "গেলাম",
    "gechi": "গেছি", "jachchi": "যাচ্ছি", "jacche": "যাচ্ছে",
    "asbe": "আসবে", "eshe": "এসে", "asche": "আসছে", "aschi": "আসছি",
    "dibo": "দিব", "dite": "দিতে", "diye": "দিয়ে", "dao": "দাও",
    "diyechi": "দিয়েছি", "diyeche": "দিয়েছে",
    "newa": "নেওয়া", "nite": "নিতে", "nilam": "নিলাম", "nibo": "নিব",
    "niyechi": "নিয়েছি", "niyeche": "নিয়েছে",
    "dekbo": "দেখব", "dekhte": "দেখতে", "dekhi": "দেখি", "dekhe": "দেখে",
    "dekhechi": "দেখেছি", "dekhlam": "দেখলাম",
    "pabo": "পাব", "peli": "পেলাম", "peyechi": "পেয়েছি", "peyeche": "পেয়েছে",
    "bujhi": "বুঝি", "bujhte": "বুঝতে", "bujhe": "বুঝে", "bujhechi": "বুঝেছি",
    "chai": "চাই", "chao": "চাও", "chilam": "ছিলাম",
    "parbo": "পারব", "pari": "পারি", "parte": "পারতে",
    "lagbe": "লাগবে", "lage": "লাগে", "lagchi": "লাগছি", "lagche": "লাগছে",
    "thakbe": "থাকবে", "thakbo": "থাকব", "thakte": "থাকতে",
    # Adjectives / adverbs
    "bhalo": "ভালো", "kharap": "খারাপ", "onek": "অনেক", "khub": "খুব",
    # Time
    "ekhon": "এখন", "tokhon": "তখন", "aj": "আজ", "ajke": "আজকে",
    "kal": "কাল", "kalke": "কালকে", "somoy": "সময়",
    "din": "দিন", "rat": "রাত", "shokal": "সকাল",
    # Postpositions / conjunctions
    "jonno": "জন্য", "theke": "থেকে", "sathe": "সাথে",
    "kintu": "কিন্তু", "ebong": "এবং", "othoba": "অথবা", "jodi": "যদি",
    "tahole": "তাহলে", "tobe": "তবে", "karon": "কারণ",
    "tai": "তাই", "abar": "আবার",
    # Common nouns
    "khabar": "খাবার", "khete": "খেতে",
    "dorkar": "দরকার", "khobor": "খবর",
    "kaj": "কাজ", "kotha": "কথা",
    "bhalobashi": "ভালোবাসি", "bhalobasha": "ভালোবাসা",
    "rajdhani": "রাজধানী", "porikkha": "পরীক্ষা", "porikkhar": "পরীক্ষার",
    "bhorti": "ভর্তি", "chuti": "ছুটি",
    "manush": "মানুষ", "manusher": "মানুষের", "pani": "পানি",
    "bari": "বাড়ি", "boi": "বই", "taka": "টাকা",
    "desh": "দেশ", "desher": "দেশের", "shahar": "শহর", "gram": "গ্রাম",
    "shikkha": "শিক্ষা", "shikkhok": "শিক্ষক",
    "chhatra": "ছাত্র", "bidyaloy": "বিদ্যালয়",
    "bishobidyaloy": "বিশ্ববিদ্যালয়",
    "sorkar": "সরকার", "podok": "পদক",
    "potro": "পত্র", "suchona": "সূচনা",
    "shastho": "স্বাস্থ্য", "daktar": "ডাক্তার", "oshudh": "ওষুধ",
    "krishi": "কৃষি", "chash": "চাষ",
    "somvob": "সম্ভব",
    # Places (common in queries)
    "dhaka": "ঢাকা", "bangladesh": "বাংলাদেশ",
    "chittagong": "চট্টগ্রাম", "khulna": "খুলনা", "rajshahi": "রাজশাহী",
    "sylhet": "সিলেট", "barisal": "বরিশাল", "rangpur": "রংপুর",
    "mymensingh": "ময়মনসিংহ", "comilla": "কুমিল্লা", "narayanganj": "নারায়ণগঞ্জ",
    "gazipur": "গাজীপুর", "coxsbazar": "কক্সবাজার",
    # Common content words
    "shadhinota": "স্বাধীনতা", "bangla": "বাংলা", "bhasha": "ভাষা",
    "nodi": "নদী", "padma": "পদ্মা", "meghna": "মেঘনা", "jamuna": "যমুনা",
    "podokkho": "পদক্ষেপ",
    "montree": "মন্ত্রী", "proshashok": "প্রশাসক",
    "university": "বিশ্ববিদ্যালয়", "school": "স্কুল",
    # Numbers (common in queries)
    "ek": "এক", "dui": "দুই", "tin": "তিন", "char": "চার",
    "pach": "পাঁচ", "chhoy": "ছয়", "sat": "সাত", "at": "আট",
    "noy": "নয়", "dosh": "দশ",
    # More common patterns
    "beshi": "বেশি", "kom": "কম", "shob": "সব", "shobcheye": "সবচেয়ে",
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
