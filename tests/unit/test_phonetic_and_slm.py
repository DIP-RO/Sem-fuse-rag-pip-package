"""Tests for the phonetic transliteration engine and SLM provider."""

from __future__ import annotations

from semfuse.language.banglish import BanglishNormalizer
from semfuse.language.phonetic import _transliterate_token, phonetic_transliterate

# ---------------------------------------------------------------------------
# Phonetic engine — individual token tests
# ---------------------------------------------------------------------------


def test_phonetic_rajdhani() -> None:
    assert _transliterate_token("rajdhani") == "রাজধানি"


def test_phonetic_bhalo() -> None:
    assert _transliterate_token("bhalo") == "ভালো"


def test_phonetic_ami() -> None:
    assert _transliterate_token("ami") == "আমি"


def test_phonetic_tumi() -> None:
    assert _transliterate_token("tumi") == "তুমি"


def test_phonetic_desh() -> None:
    assert _transliterate_token("desh") == "দেশ"


def test_phonetic_shikkha() -> None:
    assert _transliterate_token("shikkha") == "শিক্ষা"


def test_phonetic_khub() -> None:
    assert _transliterate_token("khub") == "খুব"


def test_phonetic_bangladesh() -> None:
    result = _transliterate_token("bangladesh")
    # Phonetic fallback may produce বাঙলাদেশ (ঙ instead of ং);
    # the dictionary layer handles the correct বাংলাদেশ.
    assert "লাদেশ" in result


# ---------------------------------------------------------------------------
# Phonetic engine — full sentence tests
# ---------------------------------------------------------------------------


def test_phonetic_sentence_preserves_english() -> None:
    result = phonetic_transliterate("amra university te porikkha dibo")
    # "university" is in the English pass-through list.
    assert "university" in result
    # Banglish words should be transliterated.
    assert "আমরা" in result or "আম্রা" in result


def test_phonetic_sentence_preserves_punctuation() -> None:
    result = phonetic_transliterate("rajdhani kothay?")
    assert "?" in result
    assert "কোথায" in result or "কথায" in result


def test_phonetic_preserves_bangla_text() -> None:
    text = "ঢাকা is the capital"
    result = phonetic_transliterate(text)
    assert "ঢাকা" in result
    assert "capital" in result  # English pass-through


def test_phonetic_preserves_digits() -> None:
    result = phonetic_transliterate("1971 sal e")
    assert "1971" in result


# ---------------------------------------------------------------------------
# Two-layer transliteration: dictionary + phonetic fallback
# ---------------------------------------------------------------------------


def test_two_layer_dictionary_takes_priority() -> None:
    """Dictionary entries should be used over phonetic for known words."""
    norm = BanglishNormalizer()
    result = norm.transliterate("Bangladesh er rajdhani kothay?")
    # Dictionary entries: Bangladesh -> বাংলাদেশ, rajdhani -> রাজধানী
    assert "বাংলাদেশ" in result
    assert "রাজধানী" in result
    assert "কোথায়" in result or "কথায়" in result


def test_two_layer_phonetic_fallback_for_unknown() -> None:
    """Unknown words should be transliterated by the phonetic engine."""
    norm = BanglishNormalizer()
    result = norm.transliterate("amra notun kichu korbo")
    # "notun" and "kichu" are not in the dictionary but should get phonetic transliteration.
    # They should NOT remain in Latin script.
    assert "notun" not in result.lower()
    assert "kichu" not in result.lower()


def test_two_layer_english_passes_through() -> None:
    """English words not in the Bangla dictionary should pass through."""
    norm = BanglishNormalizer()
    # "hospital" is in the English pass-through list but also mapped in the
    # Bangla dictionary.  Use a word that's English-only.
    result = norm.transliterate("amra online e jabo")
    assert "online" in result


# ---------------------------------------------------------------------------
# SLM provider — lazy loading (no model download needed for tests)
# ---------------------------------------------------------------------------


def test_slm_provider_init_no_download() -> None:
    """Creating an SLM provider should not trigger a model download."""
    from semfuse.rag.slm_provider import LocalSLMProvider

    provider = LocalSLMProvider(model="Qwen/Qwen2.5-0.5B-Instruct")
    assert provider.model_name == "Qwen/Qwen2.5-0.5B-Instruct"
    # Model should not be loaded yet.
    assert provider._llm is None
    assert provider._tokenizer is None


def test_slm_factory_creates_provider() -> None:
    """The factory should create an SLM provider when llm_provider='slm'."""
    import tempfile

    from semfuse.core.config import DEFAULT_SLM_MODEL, SemFuseConfig
    from semfuse.rag.factory import create_llm_provider
    from semfuse.rag.slm_provider import LocalSLMProvider

    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(storage_path=td, llm_provider="slm", llm_model=DEFAULT_SLM_MODEL)
        provider = create_llm_provider(cfg)
        assert isinstance(provider, LocalSLMProvider)
        assert provider.model_name == DEFAULT_SLM_MODEL


def test_factory_unknown_provider_raises() -> None:
    """Unknown llm_provider should raise ConfigurationError."""
    import tempfile

    from semfuse.core.config import SemFuseConfig
    from semfuse.core.exceptions import ConfigurationError
    from semfuse.rag.factory import create_llm_provider

    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(storage_path=td, llm_provider="bogus")
        try:
            create_llm_provider(cfg)
            raise AssertionError("Should have raised")
        except ConfigurationError:
            pass


def test_factory_supports_slm_key() -> None:
    """Factory error message should mention 'slm' as a supported key."""
    import tempfile

    from semfuse.core.config import SemFuseConfig
    from semfuse.core.exceptions import ConfigurationError
    from semfuse.rag.factory import create_llm_provider

    with tempfile.TemporaryDirectory() as td:
        cfg = SemFuseConfig(storage_path=td, llm_provider="invalid")
        try:
            create_llm_provider(cfg)
        except ConfigurationError as e:
            assert "slm" in str(e)
