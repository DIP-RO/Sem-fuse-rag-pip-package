# Banglish Support

Banglish — Bengali written in Latin script (`Bangladesh er rajdhani kothay?`)
— is a first-class concern in SemFuse, not an edge case. This document
describes how the pipeline handles it.

## The problem

Banglish has no standard orthography. The same Bangla word is romanized many
ways (`achhe` / `ache`, `bhalo` / `valo`, `kothay` / `kothai`), and Banglish
freely mixes untranslated English words (`admission`, `capital`). A generic
multilingual embedding model handles Banglish only partially, and lexical
(keyword) retrieval cannot bridge scripts at all: `rajdhani` shares no
characters with `রাজধানী`.

## The pipeline

Every query and document chunk flows through the language layer:

1. **Detection** (`detect_language`) — classifies text as `bn`, `en`,
   `banglish`, `mixed`, or `unknown`. Latin-only text is classified as
   Banglish when it contains enough words from a curated marker lexicon of
   high-frequency romanized Bangla function words (`er`, `ki`, `keno`,
   `ache`, `hobe`, ...). Proper nouns like `dhaka` and words common in
   English are deliberately excluded from the lexicon so English prose is
   never misclassified.

2. **Canonicalization** (`BanglishNormalizer.normalize`) — folds spelling
   variants to one canonical romanization (`achhe → ache`, `valo → bhalo`)
   and collapses elongations (`kiii → ki`), so variant spellings embed and
   match identically.

3. **Transliteration** (`BanglishNormalizer.transliterate`) — replaces known
   Banglish tokens with their Bangla-script equivalents
   (`rajdhani → রাজধানী`, `kothay → কোথায়`). Unknown tokens — embedded
   English words and proper nouns — pass through unchanged, producing text
   like `bangladesh এর রাজধানী কোথায়?` that is close to the Bangla documents
   in both embedding space and keyword space.

Transliteration is applied only to text detected as Banglish, and only to the
*normalized* representation: the original text is always preserved unchanged
(ADR-0005).

## Why this measurably helps

- **Semantic retrieval:** the multilingual model embeds the transliterated
  query much closer to Bangla documents than the raw romanization.
- **Keyword (BM25) retrieval:** without transliteration, Banglish→Bangla
  keyword matching is impossible; with it, `rajdhani` literally matches
  `রাজধানী`.
- **Offline proof:** the built-in Banglish benchmark
  (`semfuse.evaluation.banglish_benchmark`) scores `hit@3 ≥ 0.8` even with
  the *hashing* provider — which has no cross-script ability whatsoever — so
  the gain is attributable to normalization, not the model.

Run it yourself:

```python
from semfuse import SemFuse
from semfuse.evaluation import RetrievalEvaluator, banglish_benchmark

db = SemFuse(storage_path="./.semfuse-bench")
docs, samples = banglish_benchmark()
for doc_id, text in docs:
    db.add(text, document_id=doc_id)
print(RetrievalEvaluator(db).evaluate(samples, k_values=(1, 3)))
```

## Extending the lexicons

The marker lexicon, variant table, and transliteration table live in
`semfuse/language/banglish.py` (`BANGLISH_MARKERS`, `_VARIANTS`,
`_TRANSLITERATIONS`). They are ordinary dicts/sets — contributions that grow
coverage (with benchmark evidence) are welcome. The `BanglishNormalizer`
abstraction means Banglish processing can evolve independently of the
embedding model.

## Diagnostics

```python
db.explain("Bangladesh er rajdhani kothay?")
# {
#   "detected_language": "banglish",
#   "normalized_query": "bangladesh এর রাজধানী কোথায়?",
#   ...
# }
```
