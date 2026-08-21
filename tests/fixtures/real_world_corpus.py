"""Real-world test corpus: Bangla, English, Banglish, and mixed-language documents.

These fixtures model realistic retrieval scenarios that a developer in
Bangladesh would face: cross-language queries, mixed-script documents,
and questions about Bangla content asked in Banglish.
"""

from __future__ import annotations

# A diverse corpus of 15 documents spanning Bangla, English, and mixed text.
# Each entry is (document_id, text, metadata).
REAL_WORLD_CORPUS: list[tuple[str, str, dict[str, str]]] = [
    ("capital_bn", "ঢাকা বাংলাদেশের রাজধানী। এটি দেশের বৃহত্তম শহর।",
     {"topic": "geography", "language": "bn"}),
    ("capital_en", "Dhaka is the capital of Bangladesh. It is the largest city in the country.",
     {"topic": "geography", "language": "en"}),
    ("admission_bn", "বিশ্ববিদ্যালয়ে ভর্তি পরীক্ষা ডিসেম্বরে অনুষ্ঠিত হবে। আবেদনের শেষ তারিখ নভেম্বর।",
     {"topic": "education", "language": "bn"}),
    ("admission_en", "University admission exams are held in December. The application deadline is November.",
     {"topic": "education", "language": "en"}),
    ("river_bn", "পদ্মা বাংলাদেশের অন্যতম প্রধান নদী। এটি হিমালয় থেকে উৎপন্ন হয়ে বঙ্গোপসাগরে পড়ে।",
     {"topic": "geography", "language": "bn"}),
    ("river_en", "The Padma is one of the major rivers of Bangladesh. It originates in the Himalayas and flows into the Bay of Bengal.",
     {"topic": "geography", "language": "en"}),
    ("independence_bn", "বাংলাদেশ ১৯৭১ সালে স্বাধীনতা অর্জন করে। মুক্তিযুদ্ধে তিন মিলিয়ন মানুষ শহীদ হন।",
     {"topic": "history", "language": "bn"}),
    ("independence_en", "Bangladesh gained independence in 1971. Three million people were martyred in the liberation war.",
     {"topic": "history", "language": "en"}),
    ("language_bn", "বাংলা বাংলাদেশের রাষ্ট্রভাষা। ভাষা আন্দোলন ১৯৫২ সালে সংঘটিত হয়েছিল।",
     {"topic": "culture", "language": "bn"}),
    ("language_en", "Bengali is the state language of Bangladesh. The language movement took place in 1952.",
     {"topic": "culture", "language": "en"}),
    ("economy_bn", "বাংলাদেশের অর্থনীতি মূলত কৃষি ও তৈরি পোশাক শিল্পের উপর নির্ভরশীল।",
     {"topic": "economy", "language": "bn"}),
    ("economy_en", "Bangladesh's economy depends mainly on agriculture and the garment manufacturing industry.",
     {"topic": "economy", "language": "en"}),
    ("health_bn", "স্বাস্থ্য সচেতনতা গুরুত্বপূর্ণ। নিয়মিত ব্যায়াম ও পরিমিত খাদ্য সুস্থ জীবনের ভিত্তি।",
     {"topic": "health", "language": "bn"}),
    ("health_en", "Health awareness is important. Regular exercise and a balanced diet are the foundation of a healthy life.",
     {"topic": "health", "language": "en"}),
    ("mixed_1", "The admission notice says ভর্তি পরীক্ষা will be held in December.",
     {"topic": "education", "language": "mixed"}),
]

# Cross-language queries: (query, expected_top_document_id, topic)
# The expected_top is the document that SHOULD rank first for this query.
CROSS_LANGUAGE_QUERIES: list[tuple[str, str]] = [
    # Bangla queries -> should match Bangla docs
    ("বাংলাদেশের রাজধানী কী?", "capital_bn"),
    ("বাংলাদেশের প্রধান নদী কোনটি?", "river_bn"),
    ("বাংলাদেশ কবে স্বাধীন হয়?", "independence_bn"),
    # English queries -> should match English docs
    ("What is the capital of Bangladesh?", "capital_en"),
    ("Which is the major river of Bangladesh?", "river_en"),
    ("When did Bangladesh gain independence?", "independence_en"),
    # Banglish queries -> should match Bangla docs (via transliteration)
    ("Bangladesh er rajdhani ki?", "capital_bn"),
    ("Bangladesh er mukhti juddho kokhon holo?", "independence_bn"),
    ("bhorti porikkha kokhon hobe?", "admission_bn"),
    # Mixed queries
    ("admission exam December", "admission_en"),
]

# RAG question-answer pairs: (question, expected_answer_substring)
RAG_QA_PAIRS: list[tuple[str, str]] = [
    ("বাংলাদেশের রাজধানী কী?", "ঢাকা"),
    ("What is the capital of Bangladesh?", "Dhaka"),
    ("Bangladesh er rajdhani ki?", "ঢাকা"),
    ("Where does the Padma river flow into?", "Bengal"),
    ("bhorti porikkha kokhon hobe?", "ডিসেম্বরে"),
    ("When did Bangladesh gain independence?", "1971"),
]
