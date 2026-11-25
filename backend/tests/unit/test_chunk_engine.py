import pytest
from evorag.chunking import EstimatedTokens, iter_chunks
from evorag.config import Chunking, Ingestion
from evorag.errors import IngestionError
from evorag.parsing import Parsed, Section


def parsed_text(text: str) -> Parsed:
    return Parsed(
        text=text,
        sections=(
            Section(start=0, end=len(text), label="body", parsed_start=0, parsed_end=len(text)),
        ),
    )


def test_fixed_overlap_and_stable_ids() -> None:
    config = Chunking(strategy="fixed", chunk_size=3, overlap=1)
    source = parsed_text("xyz🐍" * 10)
    chunks = list(iter_chunks(source, "d", "v", config, Ingestion()))
    assert [(c.start, c.end) for c in chunks] == [(0, 12), (8, 20), (16, 28), (24, 36), (32, 40)]
    assert all(c.text == source.text[c.start : c.end] and c.estimated for c in chunks)
    assert chunks == list(iter_chunks(source, "d", "v", config, Ingestion()))
    assert (
        list(iter_chunks(source, "d", "v2", config, Ingestion()))[0].chunk_id != chunks[0].chunk_id
    )


@pytest.mark.parametrize(
    ("first", "text"),
    [
        ("aa\n\n", "aa\n\nbb cc dd ee ff"),
        ("aa\n", "aa\nbb cc dd ee ff"),
        ("aa. ", "aa. bb cc dd ee ff"),
        ("aa bb cc ", "aa bb cc longword"),
        ("abcdefghijkl", "abcdefghijklmnopqrst"),
    ],
)
def test_recursive_precedence(first: str, text: str) -> None:
    config = Chunking(strategy="recursive", chunk_size=3, overlap=0)
    chunks = list(iter_chunks(parsed_text(text), "d", "v", config, Ingestion()))
    assert chunks[0].text == first
    assert "".join(c.text for c in chunks) == text


def test_structure_and_limits() -> None:
    source = Parsed(
        text="A\nB",
        sections=(
            Section(start=0, end=1, label="/a", parsed_start=None, parsed_end=None),
            Section(start=2, end=3, label="/b", parsed_start=None, parsed_end=None),
        ),
    )
    cfg = Chunking(strategy="structure")
    chunks = list(iter_chunks(source, "d", "v", cfg, Ingestion()))
    assert [(c.text, c.sources[0].label) for c in chunks] == [("A", "/a"), ("B", "/b")]
    with pytest.raises(IngestionError, match="too_many_chunks"):
        list(iter_chunks(source, "d", "v", cfg, Ingestion(max_chunks_per_document=1)))


def test_high_overlap_finishes() -> None:
    source = parsed_text("a " * 128)
    chunks = list(iter_chunks(source, "d", "v", Chunking(chunk_size=3, overlap=2), Ingestion()))
    assert chunks[-1].end == len(source.text)
    assert all(a.start < b.start <= a.end < b.end for a, b in zip(chunks, chunks[1:], strict=False))
    assert (
        list(iter_chunks(parsed_text("猫"), "d", "v", Chunking(), Ingestion()))[0].token_count == 1
    )


def test_invalid_source_and_tokenizer_are_rejected() -> None:
    section = Section(start=1, end=2, label="a", parsed_start=None, parsed_end=None)
    for text, sections in [("aa", ()), ("aa", (section,)), ("a", (section,))]:
        with pytest.raises(IngestionError, match="source_span"):
            list(
                iter_chunks(Parsed(text=text, sections=sections), "d", "v", Chunking(), Ingestion())
            )

    class Broken(EstimatedTokens):
        def overlap_start(self, text: str, start: int, end: int, tokens: int) -> int:
            return start

    with pytest.raises(IngestionError, match="cannot_advance"):
        list(
            iter_chunks(
                parsed_text("abcde"),
                "d",
                "v",
                Chunking(chunk_size=1, overlap=0),
                Ingestion(),
                Broken(),
            )
        )


@pytest.mark.parametrize("suffix", ["", "last"])
def test_blank_windows_skip_to_next_content(suffix: str) -> None:
    text = "head" + " " * 5000 + suffix
    chunks = list(
        iter_chunks(
            parsed_text(text),
            "d",
            "v",
            Chunking(strategy="fixed", chunk_size=2, overlap=1),
            Ingestion(),
        )
    )
    assert [chunk.text.strip() for chunk in chunks] == (["head", "last"] if suffix else ["head"])
    assert all(text[c.start : c.end] == c.text for c in chunks)
