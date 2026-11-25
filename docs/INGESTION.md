# Document ingestion

Start the service with the README command and one worker. Submit the file itself as the HTTP body, with a basename in X-Filename. The server treats input as data, reads incrementally with a 30-second upload deadline, and performs synchronous parsing and SQLite work in a worker thread. The resource permit covers upload, parsing and publication. Store the returned document ID for reads and job ID for status.

| Method and path under /api/v1 | Result |
| --- | --- |
| POST /documents | 201 for a new revision or 200 for a duplicate. The response includes job ID, document ID, revision and chunk count. |
| GET /documents | Logical sources ordered by document ID, with latest revision number |
| GET /documents/{id} | Metadata for one source |
| GET /documents/{id}/versions | Revision history in descending revision order |
| GET /documents/{id}/chunks | Ordered chunks for the latest revision, or for the optional positive version query parameter |
| GET /ingestion-jobs/{id} | Terminal outcome and sanitized failure code when applicable |

Collections accept limit (1–100, default 20) and offset (default 0). Errors contain an error code and optional job ID. Oversized data returns 413, missing records 404, invalid input 4xx and unavailable resources or storage 503. Parsing is synchronous with job completion. Close and reopen the application to read previously committed documents.

## Parsing rules

Strict UTF-8 accepts an optional initial BOM. Original bytes determine the content hash. Text uses NFC, LF line endings and preserved whitespace by default. The optional trim_trailing policy removes horizontal whitespace at line ends; text blocks drop trailing blank padding in that mode. Unsafe control characters and empty extracted content fail validation. Text has no intrinsic magic number: decodable printable data is valid plain text. Recognized binary signatures are rejected. Files other than the configured extensions are rejected.

| Format | Extraction and structure |
| --- | --- |
| TXT | Preserves text and paragraph boundaries. Sections identify source line ranges. |
| Markdown | Preserves markup. ATX headings establish the section hierarchy; fenced code keeps internal heading-like lines as text. Preamble has its own section. |
| CSV | Strict comma-separated quoting supports quoted commas and multiline cells. The first nonblank record supplies column labels. Ragged rows retain extra cells. Sections identify one-based logical record numbers. Blank rows and empty cells produce no extracted value. |
| JSON | Validates depth and structure before decoding. Ordered scalar values become text with RFC 6901 pointer labels. The top-level scalar label is $. Null and blank strings contribute no text. Duplicate keys, excessive pointer length, invalid Unicode and nonfinite numbers are rejected. |

TXT and Markdown map normalized section endpoints back to code-point offsets in decoded original text after BOM removal. Structured extraction supplies row or pointer labels. Full source byte offsets are unavailable. Markdown heading display labels are bounded to 256 characters; the document ID, revision, and normalized span identify the original section even if two headings share a display label.

## Local and batch use

```python
from pathlib import Path
from evorag.config import load_settings
from evorag.ingestion import IngestionService
from evorag.resources import ResourceManager

settings = load_settings()
engine = IngestionService(settings, ResourceManager(settings))
results = engine.ingest_batch(Path("."), ["README.md", "ROADMAP.md"])
for result in results:
    print(result.status)
```

The service also exposes ingest(name, bytes), ingest_file(root, name), and the async Phase 01 Ingestor.parse port. The repository provides document, documents, versions, chunks and job reads. Each batch item gets an independent transaction and error result. Do not feed an unbounded collection into the batch interface.

## Limits and extension points

Defaults are shown in configs/example.toml. The default maximum structured nesting is 32 and the maximum structure-item count is 50,000. The parser also guards line and comma counts before creating large intermediate structures. CSV fields observe the Python csv module's field-size limit. These conservative bounds can reject a quoted CSV file with many literal commas. Supply allowed_extensions in TOML as an array or in its environment override as a JSON array. A future format implements DocumentParser and registers an extension; enabling an extension alone does not install a parser.

Use a private operator-controlled directory for both trusted file inputs and SQLite. The HTTP API accepts only uploaded bytes. Logically distinct files with the same basename must be given distinct source labels. The service supports one local namespace and stores chunk text in plaintext. See [storage transactions](STORAGE.md), [chunk budgets](CHUNKING.md), and [measured results](PROJECT_STATE.md).
