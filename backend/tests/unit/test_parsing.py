"""Unit tests for safe filename validation, bounded local reads, and parsing."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import pytest
from evorag.config import Ingestion
from evorag.errors import IngestionError
from evorag.parsing import (
    DocumentParser,
    Parsed,
    Section,
    parse_document,
    read_local,
    validate_filename,
)

_MIB = 1024 * 1024


def _limits(
    *,
    max_file_size_mb: int = 1,
    max_extracted_text_mb: int = 1,
    max_structure_items: int = 100,
    max_depth: int = 8,
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".markdown", ".csv", ".json"),
    whitespace: Literal["preserve", "trim_trailing"] = "preserve",
) -> Ingestion:
    return Ingestion(
        max_file_size_mb=max_file_size_mb,
        max_extracted_text_mb=max_extracted_text_mb,
        max_structure_items=max_structure_items,
        max_depth=max_depth,
        allowed_extensions=allowed_extensions,
        whitespace=whitespace,
    )


def _assert_valid_parsed(parsed: Parsed) -> None:
    assert parsed.text
    assert parsed.text.strip(" \t\n")
    assert parsed.sections
    prev_end = 0
    for section in parsed.sections:
        assert 0 <= section.start < section.end <= len(parsed.text)
        assert section.start >= prev_end
        assert parsed.text[prev_end : section.start].strip(" \t\n") == ""
        assert parsed.text[section.start : section.end].strip(" \t\n") != ""
        assert section.label
        if section.parsed_start is not None or section.parsed_end is not None:
            assert section.parsed_start is not None and section.parsed_end is not None
            assert 0 <= section.parsed_start < section.parsed_end
        prev_end = section.end
    assert parsed.text[prev_end:].strip(" \t\n") == ""


def _fake_stat(
    *,
    mode: int = stat.S_IFREG | 0o644,
    size: int = 4,
    dev: int = 1,
    ino: int = 100,
    attrs: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        st_mode=mode,
        st_size=size,
        st_dev=dev,
        st_ino=ino,
        st_file_attributes=attrs,
    )


def test_validate_filename_valid_and_nfc() -> None:
    limits = _limits(allowed_extensions=("txt", ".MD"))
    assert validate_filename("Report 2026.TXT", limits) == "Report 2026.TXT"
    assert validate_filename("e\u0301.txt", limits) == "\u00e9.txt"
    assert validate_filename("README.MD", limits) == "README.MD"
    assert validate_filename(".hidden.txt", limits) == ".hidden.txt"


@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "   ",
        " lead.txt",
        "trail.txt ",
        "traildot.txt.",
        ".",
        "..",
        "...",
        ".txt",
        "noext",
        "bad.exe",
        "sub/file.txt",
        "sub\\file.txt",
        "../secret.txt",
        "C:file.txt",
        "file.txt:Zone.Identifier",
        "file.txt::$DATA",
        "bad<name>.txt",
        "bad|name.txt",
        "bad?name.txt",
        "bad*name.txt",
        "bad\x00.txt",
        "bad\n.txt",
        "bad\x7f.txt",
        "bad\u0085.txt",
        "spoof\u202etxt.md",
        "CON.txt",
        "con",
        "nul.json",
        "PRN.csv",
        "AUX.md",
        "COM1.txt",
        "lpt9.txt",
        "COM\u00b9.txt",
        "CONIN$.txt",
        "CONOUT$.txt",
        "CLOCK$.txt",
        "CON .txt",
        "a" * 252 + ".txt",
        "bad\ud800.txt",
    ],
)
def test_validate_filename_rejects_unsafe(bad_name: str) -> None:
    with pytest.raises(IngestionError) as caught:
        validate_filename(bad_name, _limits())
    assert caught.value.status == 422
    assert "secret" not in str(caught.value)
    assert "Zone.Identifier" not in str(caught.value)


def test_validate_filename_non_string_and_empty_extensions() -> None:
    with pytest.raises(IngestionError):
        validate_filename(123, _limits())  # type: ignore[arg-type]
    with pytest.raises(IngestionError) as caught:
        validate_filename("notes.txt", _limits(allowed_extensions=("", "  ")))
    assert caught.value.code == "unsupported_extension"


def test_read_local_regular_and_missing(tmp_path: Path) -> None:
    limits = _limits()
    target = tmp_path / "notes.txt"
    payload = b"hello local\n"
    target.write_bytes(payload)
    assert read_local(tmp_path, "notes.txt", limits) == payload

    with pytest.raises(IngestionError) as missing:
        read_local(tmp_path, "missing.txt", limits)
    assert missing.value.status == 404
    assert "missing.txt" not in str(missing.value)


def test_read_local_root_and_size_bounds(tmp_path: Path) -> None:
    limits = _limits(max_file_size_mb=1)
    with pytest.raises(IngestionError):
        read_local(tmp_path / "absent_dir", "a.txt", limits)

    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_bytes(b"abc")
    with pytest.raises(IngestionError):
        read_local(not_a_dir, "a.txt", limits)

    subdir = tmp_path / "dir.txt"
    subdir.mkdir()
    with pytest.raises(IngestionError) as dir_err:
        read_local(tmp_path, "dir.txt", limits)
    assert dir_err.value.code == "not_regular_file"

    exact = tmp_path / "exact.txt"
    exact.write_bytes(b"a" * _MIB)
    assert len(read_local(tmp_path, "exact.txt", limits)) == _MIB

    oversized = tmp_path / "big.txt"
    oversized.write_bytes(b"a" * (_MIB + 1))
    with pytest.raises(IngestionError) as big_err:
        read_local(tmp_path, "big.txt", limits)
    assert big_err.value.status == 413


def test_read_local_rejects_symlinks_reparse_and_races(tmp_path: Path) -> None:
    limits = _limits()
    real_file = tmp_path / "real.txt"
    real_file.write_bytes(b"secret")
    link_file = tmp_path / "link.txt"
    try:
        link_file.symlink_to(real_file)
    except OSError:
        link_file.write_bytes(b"placeholder")
    else:
        with pytest.raises(IngestionError) as sym_err:
            read_local(tmp_path, "link.txt", limits)
        assert sym_err.value.code == "symlink_rejected"

    reparse_attr = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
    root_st = _fake_stat(mode=stat.S_IFDIR | 0o755, ino=1)
    reparse_st = _fake_stat(mode=stat.S_IFREG | 0o644, attrs=reparse_attr, ino=2)
    with patch("evorag.parsing.os.lstat", side_effect=[root_st, reparse_st]):
        with pytest.raises(IngestionError) as rep_err:
            read_local(tmp_path, "real.txt", limits)
        assert rep_err.value.code == "symlink_rejected"

    sym_st = _fake_stat(mode=stat.S_IFLNK | 0o777, ino=2)
    with patch("evorag.parsing.os.lstat", side_effect=[root_st, sym_st]):
        with pytest.raises(IngestionError) as lstat_sym:
            read_local(tmp_path, "real.txt", limits)
        assert lstat_sym.value.code == "symlink_rejected"

    fifo_st = _fake_stat(mode=stat.S_IFIFO | 0o600, ino=3)
    with patch("evorag.parsing.os.lstat", side_effect=[root_st, fifo_st]):
        with pytest.raises(IngestionError) as fifo_err:
            read_local(tmp_path, "real.txt", limits)
        assert fifo_err.value.code == "not_regular_file"


def test_read_local_fstat_identity_and_stream_overflow(tmp_path: Path) -> None:
    limits = _limits(max_file_size_mb=1)
    target = tmp_path / "data.txt"
    target.write_bytes(b"hello")
    real_lstat = os.lstat(target)
    swapped_fstat = _fake_stat(dev=real_lstat.st_dev, ino=real_lstat.st_ino + 1, size=5)

    with (
        patch("evorag.parsing.os.fstat", return_value=swapped_fstat),
        pytest.raises(IngestionError) as id_err,
    ):
        read_local(tmp_path, "data.txt", limits)
    assert id_err.value.code == "file_unreadable"

    huge_fstat = _fake_stat(dev=real_lstat.st_dev, ino=real_lstat.st_ino, size=_MIB + 1)
    with (
        patch("evorag.parsing.os.fstat", return_value=huge_fstat),
        pytest.raises(IngestionError) as fstat_big,
    ):
        read_local(tmp_path, "data.txt", limits)
    assert fstat_big.value.status == 413

    class _OverflowStream:
        def __init__(self, chunks: list[bytes]) -> None:
            self._iter: Iterator[bytes] = iter(chunks)

        def __enter__(self) -> _OverflowStream:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return next(self._iter, b"")

    small_lstat = _fake_stat(dev=real_lstat.st_dev, ino=real_lstat.st_ino, size=5)
    root_st = os.lstat(tmp_path)
    with (
        patch("evorag.parsing.os.lstat", side_effect=[root_st, small_lstat, small_lstat]),
        patch("evorag.parsing._open_regular_file", return_value=999),
        patch("evorag.parsing.os.fstat", return_value=small_lstat),
        patch(
            "evorag.parsing.os.fdopen",
            return_value=_OverflowStream([b"a" * _MIB, b"b"]),
        ),
        pytest.raises(IngestionError) as stream_big,
    ):
        read_local(tmp_path, "data.txt", limits)
    assert stream_big.value.status == 413


def test_parse_document_decoding_controls_and_limits() -> None:
    bom_payload = b"\xef\xbb\xbfCafe\xcc\x81\r\nLine two\rLine three"
    parsed = parse_document("note.txt", bom_payload, _limits())
    _assert_valid_parsed(parsed)
    assert parsed.text == "Café\nLine two\nLine three"
    assert parsed.sections == (
        Section(
            start=0,
            end=len(parsed.text),
            label="lines 1-3",
            parsed_start=0,
            parsed_end=len(bom_payload.decode("utf-8-sig")),
        ),
    )

    for bad_bytes in [b"", b"\xef\xbb\xbf", b"   \r\n\t ", b"\xff\xfe"]:
        with pytest.raises(IngestionError):
            parse_document("bad.txt", bad_bytes, _limits())

    for ctrl in [b"a\x00b", b"a\x08b", b"a\x1bb", b"a\x7fb", b"a\xc2\x85b"]:
        with pytest.raises(IngestionError) as ctrl_err:
            parse_document("ctrl.txt", ctrl, _limits())
        assert ctrl_err.value.code == "unsafe_control_characters"

    with pytest.raises(IngestionError) as size_err:
        parse_document("big.txt", b"x" * (_MIB + 1), _limits(max_file_size_mb=1))
    assert size_err.value.status == 413

    with pytest.raises(IngestionError) as fmt_err:
        parse_document("doc.pdf", b"%PDF-1.7", _limits(allowed_extensions=(".pdf",)))
    assert fmt_err.value.status == 415

    with pytest.raises(IngestionError):
        parse_document("doc.txt", "not-bytes", _limits())  # type: ignore[arg-type]

    bad_limits = SimpleNamespace(
        max_file_size_mb=1,
        max_extracted_text_mb=1,
        max_structure_items=10,
        max_depth=4,
        allowed_extensions=(".txt",),
        whitespace="collapse",
    )
    with pytest.raises(IngestionError) as lim_err:
        parse_document("doc.txt", b"hi", bad_limits)  # type: ignore[arg-type]
    assert lim_err.value.code == "invalid_limits"


def test_plain_text_whitespace_modes_and_bounds() -> None:
    raw = b"\n  alpha  \n  beta  \n\n  gamma  \n  \n"
    preserved = parse_document("a.txt", raw, _limits(whitespace="preserve"))
    _assert_valid_parsed(preserved)
    assert preserved.text == "\n  alpha  \n  beta  \n\n  gamma  \n  \n"
    assert [preserved.text[s.start : s.end] for s in preserved.sections] == [
        "  alpha  \n  beta  ",
        "  gamma  ",
    ]
    assert [s.label for s in preserved.sections] == ["lines 2-3", "lines 5-5"]

    trimmed = parse_document("a.txt", raw, _limits(whitespace="trim_trailing"))
    _assert_valid_parsed(trimmed)
    assert trimmed.text == "\n  alpha\n  beta\n\n  gamma"
    assert [trimmed.text[s.start : s.end] for s in trimmed.sections] == [
        "  alpha\n  beta",
        "  gamma",
    ]
    for sec in trimmed.sections:
        assert sec.parsed_start is not None and sec.parsed_end is not None
        source = preserved.text[sec.parsed_start : sec.parsed_end]
        assert (
            "\n".join(line.rstrip(" \t") for line in source.split("\n"))
            == trimmed.text[sec.start : sec.end]
        )

    many_blocks = "\n\n".join(f"p{i}" for i in range(6)).encode("utf-8")
    with pytest.raises(IngestionError) as struct_err:
        parse_document("many.txt", many_blocks, _limits(max_structure_items=5))
    assert struct_err.value.code == "too_many_structure_items"


def test_markdown_headings_fences_and_metadata_bounds() -> None:
    md = (
        b"Preamble text.\n\n"
        b"# Top Heading #\n"
        b"Body one  \n"
        b"## Sub ##\n"
        b"```python\n"
        b"# not a heading\n"
        b"```\n"
        b"~~~text\n"
        b"## also not a heading\n"
        b"~~~\n"
        b"###   \n"
        b"#hashtag is plain text\n"
        b"# Reset\n"
        b"Final line\n"
    )
    parsed = parse_document("doc.markdown", md, _limits(whitespace="trim_trailing"))
    _assert_valid_parsed(parsed)
    assert [s.label for s in parsed.sections] == [
        "(root)",
        "Top Heading",
        "Top Heading > Sub",
        "Top Heading > Sub > (untitled)",
        "Reset",
    ]
    assert "# not a heading" in parsed.text
    assert parsed.text[parsed.sections[1].start : parsed.sections[1].end] == (
        "# Top Heading #\nBody one"
    )

    huge_heading = ("# " + ("H" * 1000) + "\n## " + ("S" * 1000) + "\nBody\n").encode("utf-8")
    bounded = parse_document("bighead.md", huge_heading, _limits())
    _assert_valid_parsed(bounded)
    assert all(len(s.label) <= 256 for s in bounded.sections)

    # Adversarial repeated headings hitting metadata budget while raw text stays under limit.
    repeated = "# " + "A" * 128 + "\n" + "## Sub\nx\n" * 8500
    with pytest.raises(IngestionError) as meta_err:
        parse_document("repeat.md", repeated.encode("utf-8"), _limits(max_structure_items=20_000))
    assert meta_err.value.status == 413


def test_csv_parsing_and_bounds() -> None:
    csv_bytes = (
        b'name,role,notes\r\n , , \r\nAlice  ,"Lead\nEngineer  ",Core\r\nBob,,Researcher,Extra\r\n'
    )
    preserved = parse_document("team.csv", csv_bytes, _limits(whitespace="preserve"))
    _assert_valid_parsed(preserved)
    assert [s.label for s in preserved.sections] == ["row 1", "row 3", "row 4"]
    assert all(s.parsed_start is None and s.parsed_end is None for s in preserved.sections)
    assert "role: Lead\nEngineer  " in preserved.text
    assert "Bob | notes: Researcher | Extra" in preserved.text

    trimmed = parse_document("team.csv", csv_bytes, _limits(whitespace="trim_trailing"))
    _assert_valid_parsed(trimmed)
    assert "Alice  " not in trimmed.text
    assert "role: Lead\nEngineer" in trimmed.text

    too_many_cells = (",".join(["x"] * 11)).encode("utf-8")
    with pytest.raises(IngestionError) as cell_err:
        parse_document("wide.csv", too_many_cells, _limits(max_structure_items=10))
    assert cell_err.value.code == "too_many_structure_items"

    too_many_rows = ("\n".join(["a"] * 11)).encode("utf-8")
    with pytest.raises(IngestionError) as row_err:
        parse_document("tall.csv", too_many_rows, _limits(max_structure_items=10))
    assert row_err.value.code == "too_many_structure_items"

    blank_csv = b" , \n , \n"
    with pytest.raises(IngestionError) as empty_err:
        parse_document("empty.csv", blank_csv, _limits())
    assert empty_err.value.code == "empty_document"


def test_json_parsing_pointers_and_pre_loads_bounds() -> None:
    data = {
        "a/b": "hello  ",
        "tilde~key": [1, True, None, "  ", {"nested": "ok"}],
    }
    raw = json.dumps(data).encode("utf-8")
    parsed = parse_document("data.json", raw, _limits(whitespace="trim_trailing"))
    _assert_valid_parsed(parsed)
    assert [s.label for s in parsed.sections] == [
        "/a~1b",
        "/tilde~0key/0",
        "/tilde~0key/1",
        "/tilde~0key/4/nested",
    ]
    assert parsed.text == "hello\n1\ntrue\nok"
    assert all(s.parsed_start is None and s.parsed_end is None for s in parsed.sections)

    scalar = parse_document("scalar.json", b'"root value"', _limits())
    _assert_valid_parsed(scalar)
    assert scalar.sections == (Section(start=0, end=10, label="$"),)

    deep_json = ("[" * 5 + "1" + "]" * 5).encode("utf-8")
    with patch("evorag.parsing.json.loads", side_effect=AssertionError("must fail before loads")):
        with pytest.raises(IngestionError) as depth_err:
            parse_document("deep.json", deep_json, _limits(max_depth=4))
        assert depth_err.value.code == "max_depth_exceeded"

        wide_json = ("[" + ",".join(["1"] * 11) + "]").encode("utf-8")
        with pytest.raises(IngestionError) as count_err:
            parse_document("wide.json", wide_json, _limits(max_structure_items=10))
        assert count_err.value.code == "too_many_structure_items"

    for bad_json in [
        b"{}",
        b"[]",
        b"null",
        b'{"a": NaN}',
        b'{"a": Infinity}',
        b'{"a": "\\u001b"}',
        b'{"a": "unclosed',
        b'{"a": "trail\\',
        b'{"a": 1}}',
        b'{"a": 1',
        b'{"a": 1,}',
        ("{" + '"n": ' + ("1" * 5000) + "}").encode("utf-8"),
    ]:
        with pytest.raises(IngestionError):
            parse_document("bad.json", bad_json, _limits(max_structure_items=100))


def test_extensible_parser_protocol_and_validation() -> None:
    class _RstParser:
        extensions = (".rst",)

        def parse(self, source: str, limits: Ingestion) -> Parsed:
            _ = limits
            return Parsed(text=source, sections=(Section(start=0, end=len(source), label="body"),))

    registry: dict[str, DocumentParser] = {".rst": _RstParser()}
    parsed = parse_document(
        "guide.rst",
        b"Title",
        _limits(allowed_extensions=(".rst",)),
        parsers=registry,
    )
    assert parsed.text == "Title"
    assert parsed.sections == (Section(start=0, end=5, label="body"),)

    class _BrokenParser:
        extensions = (".rst",)

        def __init__(self, result: Parsed) -> None:
            self._result = result

        def parse(self, source: str, limits: Ingestion) -> Parsed:
            _ = (source, limits)
            return self._result

    bad_results = [
        Parsed(text="", sections=(Section(start=0, end=1, label="a"),)),
        Parsed(text="abc", sections=()),
        Parsed(
            text="ab",
            sections=(
                Section(start=0, end=1, label="a"),
                Section(start=1, end=2, label="b"),
                Section(start=1, end=2, label="c"),
            ),
        ),
        Parsed(
            text="axb",
            sections=(
                Section(start=0, end=1, label="a"),
                Section(start=2, end=3, label="b"),
            ),
        ),
        Parsed(text="ab", sections=(Section(start=0, end=2, label=""),)),
        Parsed(text="ab", sections=(Section(start=0, end=2, label="a", parsed_start=0),)),
        Parsed(
            text="ab",
            sections=(Section(start=0, end=2, label="a", parsed_start=2, parsed_end=1),),
        ),
        Parsed(text="ab tail", sections=(Section(start=0, end=2, label="a"),)),
    ]
    for bad in bad_results:
        with pytest.raises(IngestionError):
            parse_document(
                "bad.rst",
                b"ok",
                _limits(allowed_extensions=(".rst",)),
                parsers={".rst": _BrokenParser(bad)},
            )
