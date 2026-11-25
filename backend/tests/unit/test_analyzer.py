"""Unit tests for the deterministic Unicode-aware lexical analyzer."""

import pytest
from evorag.analyzer import TextAnalyzer, is_technical_identifier
from evorag.config import Sparse


def test_technical_identifiers_and_versions() -> None:
    analyzer = TextAnalyzer()
    assert analyzer.tokenize("HTTP 404") == ("http", "404")
    assert analyzer.tokenize("RTX 3050") == ("rtx", "3050")

    qwen = analyzer.tokenize("Qwen3-Embedding")
    assert "qwen3-embedding" in qwen and "qwen3" in qwen and "embedding" in qwen

    env_var = analyzer.tokenize("EVORAG_RUNTIME__BASE_URL")
    assert "evorag_runtime__base_url" in env_var
    assert all(part in env_var for part in ("evorag", "runtime", "base", "url"))

    err = analyzer.tokenize("vector_store_corrupted")
    assert "vector_store_corrupted" in err
    assert all(part in err for part in ("vector", "store", "corrupted"))

    py = analyzer.tokenize("Python 3.12")
    assert "python" in py and "3.12" in py

    route = analyzer.tokenize("/api/v1/retrieve")
    assert "/api/v1/retrieve" in route
    assert all(part in route for part in ("api", "v1", "retrieve"))


def test_unicode_normalization_case_folding_and_punctuation() -> None:
    analyzer = TextAnalyzer()
    # Fullwidth ASCII and decomposed accent normalize identically to standard form.
    assert analyzer.tokenize("ＨＴＴＰ ４０４ Café") == ("http", "404", "café")
    assert analyzer.tokenize("... --- !!!") == ()
    assert analyzer.tokenize("") == ()


def test_configurable_stopwords_compound_and_bounds() -> None:
    no_compound = TextAnalyzer(preserve_compound_identifiers=False, remove_stopwords=True)
    assert no_compound.tokenize("The Qwen3-Embedding is in the store") == (
        "qwen3",
        "embedding",
        "store",
    )

    bounded = TextAnalyzer(max_query_tokens=3)
    assert len(bounded.tokenize("one two three four five", for_query=True)) == 3

    from_cfg = TextAnalyzer.from_config(Sparse(analyzer_version="v1", max_query_tokens=5))
    assert from_cfg.version == "v1"
    analyzed = from_cfg.analyze("RTX 3050 RTX")
    assert analyzed.term_frequencies == {"rtx": 2, "3050": 1}
    assert analyzed.length == 3
    assert is_technical_identifier("3.12")
    assert not is_technical_identifier("python")

    with pytest.raises(ValueError, match="unsupported analyzer version"):
        TextAnalyzer(version="v2")
    with pytest.raises(ValueError, match="token limits"):
        TextAnalyzer(max_query_tokens=0)
