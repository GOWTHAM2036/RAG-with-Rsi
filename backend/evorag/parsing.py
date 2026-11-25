"""Safe filename validation, bounded local reads, and structural document parsing."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import stat
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol

from .config import Ingestion
from .errors import IngestionError

_MIB = 1024 * 1024
_READ_CHUNK_BYTES = 65_536
_MAX_FILENAME_BYTES = 255
_MAX_LABEL_PART_CHARS = 128
_MAX_LABEL_CHARS = 256
_MAX_CSV_HEADER_CHARS = 64
_REPARSE_ATTR = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)

_WINDOWS_FORBIDDEN_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
    | {f"{prefix}{sup}" for prefix in ("COM", "LPT") for sup in ("\u00b9", "\u00b2", "\u00b3")}
)
_BIDI_CONTROL_CHARS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)
_UNSAFE_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


@dataclass(frozen=True, slots=True)
class Section:
    """A nonempty span of normalized extracted text tied to its structural source."""

    start: int
    end: int
    label: str
    parsed_start: int | None = None
    parsed_end: int | None = None


@dataclass(frozen=True, slots=True)
class Parsed:
    """Normalized document text paired with disjoint ascending nonempty sections."""

    text: str
    sections: tuple[Section, ...]


class DocumentParser(Protocol):
    """Protocol for format-specific parsers operating on normalized UTF-8 text."""

    @property
    def extensions(self) -> tuple[str, ...]: ...

    def parse(self, source: str, limits: Ingestion) -> Parsed: ...


def _max_file_bytes(limits: Ingestion) -> int:
    value = int(limits.max_file_size_mb * _MIB)
    if value <= 0:
        raise IngestionError("Invalid ingestion file size limit", code="invalid_limits", status=422)
    return value


def _max_extracted_bytes(limits: Ingestion) -> int:
    value = int(limits.max_extracted_text_mb * _MIB)
    if value <= 0:
        raise IngestionError(
            "Invalid ingestion extracted text limit", code="invalid_limits", status=422
        )
    return value


def _validate_limits(limits: Ingestion) -> None:
    _max_file_bytes(limits)
    _max_extracted_bytes(limits)
    if limits.max_structure_items < 1 or limits.max_depth < 1:
        raise IngestionError(
            "Invalid ingestion structure limits", code="invalid_limits", status=422
        )
    if limits.whitespace not in ("preserve", "trim_trailing"):
        raise IngestionError(
            "Invalid whitespace normalization mode", code="invalid_limits", status=422
        )


def _allowed_extensions(limits: Ingestion) -> set[str]:
    result: set[str] = set()
    for raw in limits.allowed_extensions:
        cleaned = raw.strip().lower()
        if not cleaned:
            continue
        result.add(cleaned if cleaned.startswith(".") else f".{cleaned}")
    return result


def _is_reparse_point(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & _REPARSE_ATTR)


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _exceeds_utf8_budget(text: str, budget_bytes: int) -> bool:
    if len(text) > budget_bytes:
        return True
    if len(text) * 4 <= budget_bytes:
        return False
    return _utf8_len(text) > budget_bytes


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."


def _normalize_whitespace_lines(lines: list[str], mode: str) -> str:
    if mode == "trim_trailing":
        return "\n".join(line.rstrip(" \t") for line in lines)
    return "\n".join(lines)


def validate_filename(name: str, limits: Ingestion) -> str:
    """Validate and return an unchanged Unicode NFC basename with an allowed extension."""
    if not isinstance(name, str) or not name:
        raise IngestionError("Document filename is required", code="invalid_filename", status=422)
    for ch in name:
        codepoint = ord(ch)
        if codepoint < 0x20 or (0x7F <= codepoint <= 0x9F) or ch in _BIDI_CONTROL_CHARS:
            raise IngestionError(
                "Document filename contains unsafe characters", code="invalid_filename", status=422
            )
    try:
        nfc = unicodedata.normalize("NFC", name)
        encoded = nfc.encode("utf-8", errors="strict")
    except UnicodeError:
        raise IngestionError(
            "Document filename is not valid Unicode", code="invalid_filename", status=422
        ) from None
    if not nfc or nfc != nfc.strip() or nfc.endswith(".") or not nfc.strip("."):
        raise IngestionError(
            "Document filename format is invalid", code="invalid_filename", status=422
        )
    if len(encoded) > _MAX_FILENAME_BYTES:
        raise IngestionError("Document filename is too long", code="invalid_filename", status=422)
    if any(ch in _WINDOWS_FORBIDDEN_CHARS for ch in nfc):
        raise IngestionError(
            "Document filename contains forbidden path characters",
            code="invalid_filename",
            status=422,
        )
    device_stem = nfc.split(".", 1)[0].rstrip(" ").upper()
    if device_stem in _WINDOWS_RESERVED_DEVICES or PureWindowsPath(nfc).is_reserved():
        raise IngestionError(
            "Document filename uses a reserved device name", code="invalid_filename", status=422
        )
    suffix = PurePosixPath(nfc).suffix.lower()
    if not suffix or nfc == suffix:
        raise IngestionError(
            "Document filename must include a valid extension", code="invalid_filename", status=422
        )
    if suffix not in _allowed_extensions(limits):
        raise IngestionError(
            "Document extension is not allowed", code="unsupported_extension", status=422
        )
    return nfc


def _open_regular_file(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.open(path, flags | getattr(os, "O_NONBLOCK", 0))


def read_local(root: Path, name: str, limits: Ingestion) -> bytes:
    """Read from an operator-controlled directory; compare identities before reading bytes."""
    safe_name = validate_filename(name, limits)
    maximum = _max_file_bytes(limits)
    try:
        directory = os.lstat(root)
        if not stat.S_ISDIR(directory.st_mode) or _is_reparse_point(directory):
            raise IngestionError("invalid_root")
        target = root / safe_name
        before = os.lstat(target)
        if stat.S_ISLNK(before.st_mode) or _is_reparse_point(before):
            raise IngestionError("symlink_rejected")
        if not stat.S_ISREG(before.st_mode):
            raise IngestionError("not_regular_file")
        if before.st_size > maximum:
            raise IngestionError("file_too_large", 413)
        fd = _open_regular_file(target)
        # fdopen owns the descriptor on every subsequent exit.
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or _is_reparse_point(opened):
                raise IngestionError("not_regular_file")
            if not os.path.samestat(before, opened):
                raise IngestionError("file_unreadable")
            if opened.st_size > maximum:
                raise IngestionError("file_too_large", 413)
            data = bytearray()
            while block := stream.read(min(_READ_CHUNK_BYTES, maximum - len(data) + 1)):
                if len(data) + len(block) > maximum:
                    raise IngestionError("file_too_large", 413)
                data.extend(block)
            return bytes(data)
    except FileNotFoundError:
        raise IngestionError("file_not_found", 404) from None
    except OSError:
        raise IngestionError("file_unreadable") from None


def _split_lines_with_offsets(source: str) -> list[tuple[int, int, str, bool]]:
    """Return (start, end_no_nl, text, has_nl) per line in normalized LF source."""
    entries: list[tuple[int, int, str, bool]] = []
    pos = 0
    length = len(source)
    while pos <= length:
        nl = source.find("\n", pos)
        if nl == -1:
            if pos < length:
                entries.append((pos, length, source[pos:length], False))
            break
        entries.append((pos, nl, source[pos:nl], True))
        pos = nl + 1
        if pos == length:
            break
    return entries


def _assemble_line_sections(
    source: str,
    raw_blocks: list[tuple[str, list[tuple[int, int, str]]]],
    limits: Ingestion,
) -> Parsed:
    """Build normalized text and Section spans from labeled non-blank line groups."""
    if not raw_blocks:
        raise IngestionError(
            "Document contains no extractable text", code="empty_document", status=422
        )
    if len(raw_blocks) > limits.max_structure_items:
        raise IngestionError(
            "Document exceeds maximum structure items", code="too_many_structure_items", status=422
        )

    max_bytes = _max_extracted_bytes(limits)
    if limits.whitespace == "preserve":
        sections: list[Section] = []
        label_bytes = 0
        for label, group in raw_blocks:
            parsed_start = group[0][0]
            parsed_end = group[-1][1]
            label_bytes += _utf8_len(label)
            if label_bytes > max_bytes:
                raise IngestionError(
                    "Document metadata exceeds extracted budget",
                    code="extracted_text_too_large",
                    status=413,
                )
            sections.append(
                Section(
                    start=parsed_start,
                    end=parsed_end,
                    label=label,
                    parsed_start=parsed_start,
                    parsed_end=parsed_end,
                )
            )
        if _exceeds_utf8_budget(source, max_bytes):
            raise IngestionError(
                "Extracted text exceeds maximum allowed size",
                code="extracted_text_too_large",
                status=413,
            )
        return Parsed(text=source, sections=tuple(sections))

    # trim_trailing mode: preserve leading blank/gap line breaks cleanly while trimming line tails.
    parts: list[str] = []
    sections = []
    cursor = 0
    total_text_bytes = 0
    label_bytes = 0
    prev_parsed_end = 0
    for idx, (label, group) in enumerate(raw_blocks):
        parsed_start = group[0][0]
        parsed_end = group[-1][1]
        if idx == 0:
            leading = source[0:parsed_start]
            if leading:
                trimmed_leading = "\n".join(line.rstrip(" \t") for line in leading.split("\n"))
                if trimmed_leading:
                    parts.append(trimmed_leading)
                    cursor += len(trimmed_leading)
                    total_text_bytes += _utf8_len(trimmed_leading)
        else:
            gap = source[prev_parsed_end:parsed_start]
            trimmed_gap = "\n".join(line.rstrip(" \t") for line in gap.split("\n"))
            parts.append(trimmed_gap)
            cursor += len(trimmed_gap)
            total_text_bytes += _utf8_len(trimmed_gap)

        block_text = _normalize_whitespace_lines([text for _, _, text in group], "trim_trailing")
        start = cursor
        parts.append(block_text)
        cursor += len(block_text)
        end = cursor
        total_text_bytes += _utf8_len(block_text)
        label_bytes += _utf8_len(label)
        if total_text_bytes > max_bytes or label_bytes > max_bytes:
            raise IngestionError(
                "Extracted text exceeds maximum allowed size",
                code="extracted_text_too_large",
                status=413,
            )
        sections.append(
            Section(
                start=start,
                end=end,
                label=label,
                parsed_start=parsed_start,
                parsed_end=parsed_end,
            )
        )
        prev_parsed_end = parsed_end

    return Parsed(text="".join(parts), sections=tuple(sections))


class PlainTextParser:
    """Paragraph-block parser for plain text files using line-range labels."""

    extensions: tuple[str, ...] = (".txt",)

    def parse(self, source: str, limits: Ingestion) -> Parsed:
        lines = _split_lines_with_offsets(source)
        blocks: list[tuple[str, list[tuple[int, int, str]]]] = []
        current: list[tuple[int, int, str]] = []
        start_line = 0
        end_line = 0

        for line_num, (start, end, text, _) in enumerate(lines, start=1):
            if text.strip(" \t") == "":
                if current:
                    blocks.append((f"lines {start_line}-{end_line}", current))
                    if len(blocks) > limits.max_structure_items:
                        raise IngestionError(
                            "Document exceeds maximum structure items",
                            code="too_many_structure_items",
                            status=422,
                        )
                    current = []
                continue
            if not current:
                start_line = line_num
            end_line = line_num
            current.append((start, end, text))

        if current:
            blocks.append((f"lines {start_line}-{end_line}", current))
        return _assemble_line_sections(source, blocks, limits)


def _extract_atx_heading(line: str) -> tuple[int, str] | None:
    match = _ATX_HEADING_RE.match(line)
    if match is None:
        return None
    level = len(match.group(1))
    raw_title = (match.group(2) or "").strip(" \t")
    if raw_title and all(ch == "#" for ch in raw_title):
        raw_title = ""
    elif raw_title.endswith("#"):
        stripped = raw_title.rstrip("#")
        if stripped and stripped[-1] in (" ", "\t"):
            raw_title = stripped.rstrip(" \t")
    title = _truncate_text(raw_title or "(untitled)", _MAX_LABEL_PART_CHARS)
    return level, title


class MarkdownParser:
    """Heading-hierarchy parser for Markdown files respecting fenced code blocks."""

    extensions: tuple[str, ...] = (".md", ".markdown")

    def parse(self, source: str, limits: Ingestion) -> Parsed:
        lines = _split_lines_with_offsets(source)
        blocks: list[tuple[str, list[tuple[int, int, str]]]] = []
        heading_stack: list[tuple[int, str]] = []
        current_label = "(root)"
        current: list[tuple[int, int, str]] = []
        fence_char: str | None = None
        fence_len = 0

        def flush_current() -> None:
            nonlocal current
            # Trim leading/trailing blank lines from the section so gaps stay whitespace.
            first = 0
            while first < len(current) and not current[first][2].strip(" \t"):
                first += 1
            current = current[first:]
            while current and current[-1][2].strip(" \t") == "":
                current.pop()
            if current:
                blocks.append((current_label, current))
                if len(blocks) > limits.max_structure_items:
                    raise IngestionError(
                        "Document exceeds maximum structure items",
                        code="too_many_structure_items",
                        status=422,
                    )
                current = []

        for start, end, text, _ in lines:
            fence_match = _FENCE_RE.match(text)
            if fence_char is None:
                if fence_match is not None:
                    fence_marker = fence_match.group(1)
                    info = fence_match.group(2)
                    if not (fence_marker[0] == "`" and "`" in info):
                        fence_char = fence_marker[0]
                        fence_len = len(fence_marker)
                        current.append((start, end, text))
                        continue
                heading = _extract_atx_heading(text)
                if heading is not None:
                    flush_current()
                    level, title = heading
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, title))
                    path = " > ".join(item_title for _, item_title in heading_stack)
                    current_label = _truncate_text(path, _MAX_LABEL_CHARS)
                    current = [(start, end, text)]
                    continue
            else:
                if fence_match is not None:
                    marker = fence_match.group(1)
                    if (
                        marker[0] == fence_char
                        and len(marker) >= fence_len
                        and not fence_match.group(2).strip(" \t")
                    ):
                        fence_char = None
                        fence_len = 0

            current.append((start, end, text))

        flush_current()
        return _assemble_line_sections(source, blocks, limits)


class CsvParser:
    """Bounded logical-row parser for CSV documents."""

    extensions: tuple[str, ...] = (".csv",)

    def parse(self, source: str, limits: Ingestion) -> Parsed:
        max_bytes = _max_extracted_bytes(limits)
        # Bound materialization of one very wide CSV record before the C parser allocates it.
        if source.count(",") > limits.max_structure_items:
            raise IngestionError("too_many_structure_items")
        reader = csv.reader(io.StringIO(source, newline=""), strict=True)
        parts: list[str] = []
        sections: list[Section] = []
        cursor = 0
        total_text_bytes = 0
        label_bytes = 0
        raw_rows = 0
        total_cells = 0
        headers: list[str] | None = None

        try:
            for row in reader:
                raw_rows += 1
                total_cells += len(row)
                if (
                    raw_rows > limits.max_structure_items
                    or total_cells > limits.max_structure_items
                ):
                    raise IngestionError(
                        "CSV exceeds maximum structure items",
                        code="too_many_structure_items",
                        status=422,
                    )
                if not row or all(cell.strip(" \t\n") == "" for cell in row):
                    continue

                if limits.whitespace == "trim_trailing":
                    cleaned = [
                        _normalize_whitespace_lines(cell.split("\n"), "trim_trailing")
                        for cell in row
                    ]
                else:
                    cleaned = list(row)

                if headers is None:
                    headers = [
                        _truncate_text(c.strip(" \t\n"), _MAX_CSV_HEADER_CHARS) for c in cleaned
                    ]
                    row_text = " | ".join(c for c in cleaned if c.strip(" \t\n"))
                else:
                    formatted_cells: list[str] = []
                    for col_idx, cell in enumerate(cleaned):
                        if cell.strip(" \t\n") == "":
                            continue
                        header = headers[col_idx] if col_idx < len(headers) else ""
                        if header:
                            formatted_cells.append(f"{header}: {cell}")
                        else:
                            formatted_cells.append(cell)
                    row_text = " | ".join(formatted_cells)

                if limits.whitespace == "trim_trailing":
                    row_text = _normalize_whitespace_lines(row_text.split("\n"), "trim_trailing")

                if not row_text or row_text.strip(" \t\n") == "":
                    continue

                if len(sections) + 1 > limits.max_structure_items:
                    raise IngestionError(
                        "CSV exceeds maximum structure items",
                        code="too_many_structure_items",
                        status=422,
                    )
                if parts:
                    parts.append("\n")
                    cursor += 1
                    total_text_bytes += 1

                start = cursor
                parts.append(row_text)
                cursor += len(row_text)
                end = cursor
                total_text_bytes += _utf8_len(row_text)

                label = f"row {raw_rows}"
                label_bytes += _utf8_len(label)
                if total_text_bytes > max_bytes or label_bytes > max_bytes:
                    raise IngestionError(
                        "Extracted text exceeds maximum allowed size",
                        code="extracted_text_too_large",
                        status=413,
                    )
                sections.append(Section(start=start, end=end, label=label))
        except csv.Error:
            raise IngestionError("Malformed CSV document", code="invalid_csv", status=422) from None

        if not sections:
            raise IngestionError(
                "Document contains no extractable text", code="empty_document", status=422
            )
        return Parsed(text="".join(parts), sections=tuple(sections))


def _prescan_json(source: str, limits: Ingestion) -> None:
    """Enforce nesting depth and structural complexity BEFORE calling json.loads."""
    depth = items = 0
    quoted = escaped = False
    for char in source:
        if quoted:
            if escaped:
                escaped = False
            elif char == chr(92):
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            items += 1
            if depth > limits.max_depth:
                raise IngestionError("max_depth_exceeded")
        elif char in "]}":
            depth -= 1
            if depth < 0:
                raise IngestionError("invalid_json")
        elif char == ",":
            items += 1
        if items > limits.max_structure_items:
            raise IngestionError("too_many_structure_items")
    if quoted or depth:
        raise IngestionError("invalid_json")


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise IngestionError("duplicate_json_key")
    return result


def _reject_nonfinite_json(_token: str) -> object:
    raise IngestionError(
        "JSON contains non-finite numeric constant", code="invalid_json", status=422
    )


def _escape_json_pointer(segment: str) -> str:
    if len(segment) > _MAX_LABEL_CHARS or _UNSAFE_CONTROL_RE.search(segment):
        raise IngestionError("invalid_json_key")
    return segment.replace("~", "~0").replace("/", "~1")


def _format_json_scalar(value: object, limits: Ingestion) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise IngestionError("invalid_json")
        return str(value)
    if isinstance(value, str):
        if _UNSAFE_CONTROL_RE.search(value):
            raise IngestionError(
                "JSON string contains unsafe control characters",
                code="unsafe_control_characters",
                status=422,
            )
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")
        if not unicodedata.is_normalized("NFC", normalized):
            normalized = unicodedata.normalize("NFC", normalized)
        if limits.whitespace == "trim_trailing":
            normalized = _normalize_whitespace_lines(normalized.split("\n"), "trim_trailing")
        if normalized.strip(" \t\n") == "":
            return None
        return normalized
    return None


class JsonParser:
    """Depth-bounded iterative JSON parser producing RFC 6901 pointer sections."""

    extensions: tuple[str, ...] = (".json",)

    def parse(self, source: str, limits: Ingestion) -> Parsed:
        _prescan_json(source, limits)
        max_bytes = _max_extracted_bytes(limits)
        try:
            root = json.loads(
                source, parse_constant=_reject_nonfinite_json, object_pairs_hook=_unique_keys
            )
        except IngestionError:
            raise
        except (json.JSONDecodeError, ValueError, RecursionError):
            raise IngestionError(
                "Malformed JSON document", code="invalid_json", status=422
            ) from None

        parts: list[str] = []
        sections: list[Section] = []
        cursor = 0
        total_text_bytes = 0
        label_bytes = 0
        visited = 0

        # Explicit stack of (pointer_segments_tuple, depth, value)
        stack: list[tuple[tuple[str, ...], int, object]] = [((), 0, root)]
        while stack:
            path_parts, depth, value = stack.pop()
            visited += 1
            if visited > limits.max_structure_items * 2 + 1:
                raise IngestionError(
                    "JSON exceeds maximum structure items",
                    code="too_many_structure_items",
                    status=422,
                )
            if isinstance(value, dict):
                if depth + 1 > limits.max_depth:
                    raise IngestionError(
                        "JSON nesting exceeds maximum depth", code="max_depth_exceeded", status=422
                    )
                for key, child in reversed(list(value.items())):
                    seg = _escape_json_pointer(str(key))
                    stack.append(((*path_parts, seg), depth + 1, child))
                continue
            if isinstance(value, list):
                if depth + 1 > limits.max_depth:
                    raise IngestionError(
                        "JSON nesting exceeds maximum depth", code="max_depth_exceeded", status=422
                    )
                for idx in range(len(value) - 1, -1, -1):
                    stack.append(((*path_parts, str(idx)), depth + 1, value[idx]))
                continue

            rendered = _format_json_scalar(value, limits)
            if rendered is None:
                continue

            if len(sections) + 1 > limits.max_structure_items:
                raise IngestionError(
                    "JSON exceeds maximum structure items",
                    code="too_many_structure_items",
                    status=422,
                )
            label = "/" + "/".join(path_parts) if path_parts else "$"
            if len(label) > _MAX_LABEL_CHARS:
                raise IngestionError("json_pointer_too_long")

            if parts:
                parts.append("\n")
                cursor += 1
                total_text_bytes += 1
            start = cursor
            parts.append(rendered)
            cursor += len(rendered)
            end = cursor
            total_text_bytes += _utf8_len(rendered)
            label_bytes += _utf8_len(label)
            if total_text_bytes > max_bytes or label_bytes > max_bytes:
                raise IngestionError(
                    "Extracted text exceeds maximum allowed size",
                    code="extracted_text_too_large",
                    status=413,
                )
            sections.append(Section(start=start, end=end, label=label))

        if not sections:
            raise IngestionError(
                "Document contains no extractable text", code="empty_document", status=422
            )
        return Parsed(text="".join(parts), sections=tuple(sections))


def build_default_parsers() -> dict[str, DocumentParser]:
    """Create the default extension-to-parser mapping."""
    registry: dict[str, DocumentParser] = {}
    for parser in (PlainTextParser(), MarkdownParser(), CsvParser(), JsonParser()):
        for ext in parser.extensions:
            registry[ext.lower()] = parser
    return registry


DEFAULT_PARSERS: Mapping[str, DocumentParser] = build_default_parsers()


def _validate_parsed(parsed: Parsed, limits: Ingestion) -> Parsed:
    max_bytes = _max_extracted_bytes(limits)
    if not parsed.text or parsed.text.strip(" \t\n") == "":
        raise IngestionError(
            "Document contains no extractable text", code="empty_document", status=422
        )
    if _exceeds_utf8_budget(parsed.text, max_bytes):
        raise IngestionError(
            "Extracted text exceeds maximum allowed size",
            code="extracted_text_too_large",
            status=413,
        )
    if not parsed.sections:
        raise IngestionError(
            "Document contains no extractable sections", code="empty_document", status=422
        )
    if len(parsed.sections) > limits.max_structure_items:
        raise IngestionError(
            "Document exceeds maximum structure items", code="too_many_structure_items", status=422
        )

    prev_end = 0
    label_bytes = 0
    text_len = len(parsed.text)
    for section in parsed.sections:
        if section.start < prev_end or section.start >= section.end or section.end > text_len:
            raise IngestionError(
                "Parser produced invalid section spans", code="invalid_sections", status=422
            )
        gap = parsed.text[prev_end : section.start]
        if gap and gap.strip(" \t\n") != "":
            raise IngestionError(
                "Parser produced non-whitespace gap between sections",
                code="invalid_sections",
                status=422,
            )
        if not section.label:
            raise IngestionError(
                "Parser produced empty section label", code="invalid_sections", status=422
            )
        if (section.parsed_start is None) != (section.parsed_end is None):
            raise IngestionError(
                "Parser produced partial source offsets", code="invalid_sections", status=422
            )
        if section.parsed_start is not None and section.parsed_end is not None:
            if section.parsed_start < 0 or section.parsed_start >= section.parsed_end:
                raise IngestionError(
                    "Parser produced invalid source offsets", code="invalid_sections", status=422
                )
        label_bytes += _utf8_len(section.label)
        if label_bytes > max_bytes:
            raise IngestionError(
                "Document metadata exceeds extracted budget",
                code="extracted_text_too_large",
                status=413,
            )
        prev_end = section.end

    tail = parsed.text[prev_end:]
    if tail and tail.strip(" \t\n") != "":
        raise IngestionError(
            "Parser left unindexed non-whitespace trailing text",
            code="invalid_sections",
            status=422,
        )
    return parsed


def parse_document(
    name: str,
    payload: bytes,
    limits: Ingestion,
    *,
    parsers: Mapping[str, DocumentParser] | None = None,
) -> Parsed:
    """Validate, decode, normalize, and structurally parse a document payload."""
    _validate_limits(limits)
    safe_name = validate_filename(name, limits)
    if not isinstance(payload, bytes):
        raise IngestionError("Payload must be raw bytes", code="invalid_payload", status=422)
    max_bytes = _max_file_bytes(limits)
    if len(payload) == 0:
        raise IngestionError("Document is empty", code="empty_document", status=422)
    if len(payload) > max_bytes:
        raise IngestionError("File exceeds maximum allowed size", code="file_too_large", status=413)
    if payload.removeprefix(b"\xef\xbb\xbf").startswith(
        (b"%PDF-", b"PK\x03\x04", b"\x89PNG", b"\x7fELF")
    ):
        raise IngestionError("binary_document", 415)

    ext = PurePosixPath(safe_name).suffix.lower()
    active_parsers = DEFAULT_PARSERS if parsers is None else parsers
    parser = active_parsers.get(ext)
    if parser is None:
        raise IngestionError("Unsupported document format", code="unsupported_format", status=415)

    try:
        decoded = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        raise IngestionError(
            "Document is not valid UTF-8", code="invalid_encoding", status=422
        ) from None

    if not decoded:
        raise IngestionError("Document is empty", code="empty_document", status=422)
    if _UNSAFE_CONTROL_RE.search(decoded):
        raise IngestionError(
            "Document contains unsafe control characters",
            code="unsafe_control_characters",
            status=422,
        )

    if (
        decoded.count("\n") + decoded.count("\r") - decoded.count("\r\n")
        > limits.max_structure_items
    ):
        raise IngestionError("too_many_structure_items")
    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")
    if not unicodedata.is_normalized("NFC", normalized):
        normalized = unicodedata.normalize("NFC", normalized)

    try:
        parsed = _validate_parsed(parser.parse(normalized, limits), limits)
    except UnicodeEncodeError:
        raise IngestionError("invalid_unicode") from None
    if decoded != normalized and any(s.parsed_start is not None for s in parsed.sections):
        # Built-in TXT/Markdown blocks begin and end at physical line boundaries.
        offsets: dict[int, int] = {}
        cursor = 0
        for match in re.finditer(r"([^\r\n]*)(\r\n|\r|\n|$)", decoded):
            offsets[cursor] = match.start(1)
            cursor += len(unicodedata.normalize("NFC", match[1]))
            offsets[cursor] = match.end(1)
            cursor += bool(match[2])
        mapped = []
        for section in parsed.sections:
            start = offsets.get(section.parsed_start) if section.parsed_start is not None else None
            end = offsets.get(section.parsed_end) if section.parsed_end is not None else None
            mapped.append(
                replace(
                    section,
                    parsed_start=start if end is not None else None,
                    parsed_end=end if start is not None else None,
                )
            )
        return Parsed(parsed.text, tuple(mapped))
    return parsed
