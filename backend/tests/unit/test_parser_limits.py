from pathlib import Path
from unittest.mock import patch

import pytest
from evorag.config import Ingestion
from evorag.errors import IngestionError
from evorag.parsing import parse_document, read_local


@pytest.mark.parametrize(
    ("name", "raw", "code"),
    [
        ("x.csv", b'a,b\n1,"unterminated', "invalid_csv"),
        ("x.json", b'{"a": 1, "a": 2}', "duplicate_json_key"),
        ("x.json", b"[1e9999]", "invalid_json"),
        ("x.json", b'["\\ud800"]', "invalid_unicode"),
        ("x.json", b'{"bad\\u0000key":"value"}', "invalid_json_key"),
        ("x.txt", b"%PDF-1.4", "binary_document"),
        ("x.txt", b"\n" * 9, "too_many_structure_items"),
        ("x.json", b'{"' + b"x" * 257 + b'":0}', "invalid_json_key"),
        ("x.json", b'{"' + b"x" * 140 + b'":{"' + b"y" * 140 + b'":0}}', "json_pointer_too_long"),
    ],
)
def test_reject_bad_input(name: str, raw: bytes, code: str) -> None:
    with pytest.raises(IngestionError) as error:
        parse_document(name, raw, Ingestion(max_structure_items=8))
    assert error.value.code == code


def test_source_offsets_preserve_original_decoded_positions() -> None:
    raw = "e\u0301\r\n\r\n🐍\rnext"
    doc = parse_document("x.txt", raw.encode(), Ingestion())
    assert doc.text == "é\n\n🐍\nnext"
    assert [(s.parsed_start, s.parsed_end, s.start, s.end) for s in doc.sections] == [
        (0, 2, 0, 1),
        (6, 12, 3, 9),
    ]


def test_local_read_error_sanitized(tmp_path: Path) -> None:
    with patch("evorag.parsing.os.lstat", side_effect=PermissionError("private/path")):
        with pytest.raises(IngestionError, match="file_unreadable") as error:
            read_local(tmp_path, "a.txt", Ingestion())
    assert "private" not in str(error.value)
