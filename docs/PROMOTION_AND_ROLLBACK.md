# Promotion and Rollback

- **Explicit Approval**: `approve_recommendation` binds approval to exact `candidate_fingerprint`, `baseline_version`, and `activation_scope`. Offline evaluations (`eligible_for_live_promotion = false`) cannot authorize `live_runtime` promotion.
- **Atomic Promotion & Rollback**: `promote_approved` and `rollback_to` atomically update `configuration_versions` and record `PromotionRecord` / `RollbackRecord` in SQLite WAL. Stale approvals are rejected if the active version changed.
