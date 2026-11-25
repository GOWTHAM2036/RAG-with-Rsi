from pathlib import Path

import pytest
from evorag.config import load_settings
from evorag.errors import ConfigurationError


def test_precedence(tmp_path: Path) -> None:
    config = tmp_path / "settings with spaces.toml"
    config.write_text("[chunking]\nchunk_size=512\noverlap=32\n", encoding="utf-8")
    settings = load_settings(
        environ={
            "EVORAG_CONFIG": str(config),
            "EVORAG_CHUNKING__OVERLAP": "0",
            "EVORAG_RSI__REQUIRE_HUMAN_APPROVAL": "true",
        }
    )
    assert (settings.chunking.chunk_size, settings.chunking.overlap) == (512, 0)
    assert settings.resources.max_loaded_llms == settings.runtime.max_concurrent_requests == 1
    assert settings.embeddings.dimensions == 1024
    assert not settings.rsi.automatic_promotion and not settings.reflection.enabled


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("CHUNKING__OVERLAP", "384"),
        ("EMBEDDINGS__DIMENSIONS", "0"),
        ("RUNTIME__MAX_OUTPUT_TOKENS", "4096"),
        ("RUNTIME__BASE_URL", "http://user:private@host/"),
        ("RUNTIME__BASE_URL", "ftp://host"),
        ("RUNTIME__GENERATION_MODEL", "  "),
        ("RUNTIME__TEMPERATURE", "nan"),
        ("RESOURCES__MAX_PARALLEL_GENERATIONS", "-1"),
        ("RETRIEVAL__FINAL_TOP_K", "41"),
        ("RSI__AUTOMATIC_PROMOTION", "true"),
        ("RSI__REQUIRE_HUMAN_APPROVAL", "false"),
        ("INVALID", "1"),
        ("RUNTIME__TYPO", "1"),
    ],
)
def test_invalid_override(key: str, value: str) -> None:
    with pytest.raises(ConfigurationError) as caught:
        load_settings(environ={"EVORAG_" + key: value})
    assert "private" not in str(caught.value)


def test_bad_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_settings(path, environ={})
    path.write_text("invalid file", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_settings(path, environ={})
    path.write_text("chunking=4", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="table"):
        load_settings(path, environ={"EVORAG_CHUNKING__OVERLAP": "1"})
