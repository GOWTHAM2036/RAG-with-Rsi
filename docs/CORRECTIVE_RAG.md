# Corrective RAG

`backend/evorag/correction_planner.py` maps verified findings to the least expensive allowed `CorrectionAction` (`accept_answer`, `remove_unsupported_claims`, `revise_answer`, `retrieve_more`, `reconcile_conflicting_evidence`, `return_partial_answer`, `abstain`).

- **Targeted Retrieval**: Preserves original trusted `MetadataFilter` and corpus revision scope.
- **Candidate Comparison (`compare_candidates`)**: Never auto-prefers the newest answer; rejects corrections that introduce invalid citations or regressions and retains admissible originals.
