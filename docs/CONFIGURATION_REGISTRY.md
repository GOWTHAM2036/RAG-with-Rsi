# Configuration Registry

`configuration_versions` (`backend/evorag/rsi_store.py`) stores immutable `ConfigurationVersion` snapshots initialized with `v0_baseline`. Historical versions are never overwritten; lineage is preserved via `parent_version_id` and SHA-256 `fingerprint`.
