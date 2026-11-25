# SQLite metadata storage

The operator chooses the database path, which defaults to data/evorag.sqlite3. Schema initialization is lazy. Each operation opens and closes its connection, enables foreign keys and uses a bounded busy timeout (5 seconds by default). Initialization requests WAL when configured and an exclusive writer reservation serializes schema creation. A filesystem must support SQLite locking. Preserve the WAL and shared-memory companions while the database is active; use SQLite's backup facility for a consistent snapshot.

| Table | Responsibility |
| --- | --- |
| schema_versions | Applied migration numbers and timestamps, paired with PRAGMA user_version |
| documents | Unique logical source basename and SHA-256 document ID |
| versions | Stable revision key, source hash, local revision number, parser, normalization and chunking fingerprint, creation time and chunk count |
| chunks | Unique chunk ID and ordered payload per revision, including text, source ranges and estimated token count |
| ingestion_jobs | Outcome, revision reference or sanitized error code, and timestamp |

The first revision of a logical source is 1. Each previously unseen combination of raw content and transformation settings creates the next revision. Re-ingestion of an earlier revision returns that revision and a new duplicate job record. The highest committed revision remains the default for reads. Matching bytes under a different basename retain independent lineage.

A BEGIN IMMEDIATE writer checks for duplicates again under the database lock. Source registration, version allocation, chunk streaming and the successful job form one transaction. A parsing, validation, storage or iterator failure rolls back provisional rows. After rollback, a rejected document may record a failed job in a separate transaction. Request validation, capacity rejection or unavailable storage can prevent job creation; the error then carries a null job ID.

Identifiers are full SHA-256 hex values. Document IDs and revision keys hash UTF-8 JSON arrays of tagged fields with fixed separators. The content hash is computed directly on submitted bytes, including any BOM. The design assumes SHA-256 collision resistance. Uniqueness constraints protect references and detect conflicting persisted keys. Human-readable version numbers describe insertion order within one database.

Schema version 1 is the only supported migration. Unknown schema versions fail closed. This phase stores private source metadata and chunk text in plaintext. Its tables contain no vectors.
