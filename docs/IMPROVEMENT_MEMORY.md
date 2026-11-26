# Improvement Memory

`RSIStore` (`backend/evorag/rsi_store.py`) persists `ImprovementMemoryEntry` records in SQLite WAL (`improvement_memory` table) keyed by `candidate_fingerprint` and `dataset_version`, preventing repeated evaluation of previously rejected configurations in the same experimental context.
