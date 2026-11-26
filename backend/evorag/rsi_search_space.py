"""Trusted parameter search space and strict candidate validation for Phase 09 RSI."""

from typing import Any

from .rsi_types import (
    CandidateConfiguration,
    CandidateValidationResult,
    ParameterDefinition,
    SearchSpace,
)

TRUSTED_SEARCH_SPACE = SearchSpace(
    parameters=(
        ParameterDefinition(path="bm25.k1", param_type="float", min_value=0.5, max_value=2.5),
        ParameterDefinition(path="bm25.b", param_type="float", min_value=0.0, max_value=1.0),
        ParameterDefinition(
            path="fusion.dense_weight", param_type="float", min_value=0.25, max_value=2.0
        ),
        ParameterDefinition(
            path="fusion.sparse_weight", param_type="float", min_value=0.25, max_value=2.0
        ),
        ParameterDefinition(path="fusion.k", param_type="int", min_value=10, max_value=100),
        ParameterDefinition(
            path="retrieval.dense_top_k", param_type="int", min_value=5, max_value=40
        ),
        ParameterDefinition(
            path="retrieval.sparse_top_k", param_type="int", min_value=5, max_value=40
        ),
        ParameterDefinition(
            path="retrieval.final_top_k", param_type="int", min_value=3, max_value=8
        ),
        ParameterDefinition(
            path="adaptive_rag.default_strategy",
            param_type="enum",
            allowed_values=("baseline", "adaptive"),
        ),
        ParameterDefinition(path="query_rewriting.enabled", param_type="bool"),
        ParameterDefinition(path="query_decomposition.enabled", param_type="bool"),
        ParameterDefinition(path="reranking.enabled", param_type="bool"),
        ParameterDefinition(
            path="reflection.trigger_policy",
            param_type="enum",
            allowed_values=("on_failure", "always_complex", "explicit"),
        ),
        ParameterDefinition(
            path="reflection.max_cycles", param_type="int", min_value=0, max_value=1
        ),
        ParameterDefinition(
            path="reflection.max_additional_retrievals", param_type="int", min_value=0, max_value=2
        ),
        ParameterDefinition(
            path="retrieval_cache.max_entries", param_type="int", min_value=32, max_value=1024
        ),
        ParameterDefinition(
            path="retrieval_cache.ttl_seconds", param_type="float", min_value=30.0, max_value=1800.0
        ),
    )
)

_PARAMS_BY_PATH: dict[str, ParameterDefinition] = {
    p.path: p for p in TRUSTED_SEARCH_SPACE.parameters
}


def default_tunable_parameters() -> dict[str, Any]:
    return {
        "bm25.k1": 1.5,
        "bm25.b": 0.75,
        "fusion.dense_weight": 1.0,
        "fusion.sparse_weight": 1.0,
        "fusion.k": 60,
        "retrieval.dense_top_k": 20,
        "retrieval.sparse_top_k": 20,
        "retrieval.final_top_k": 5,
        "adaptive_rag.default_strategy": "baseline",
        "query_rewriting.enabled": True,
        "query_decomposition.enabled": True,
        "reranking.enabled": False,
        "reflection.trigger_policy": "on_failure",
        "reflection.max_cycles": 1,
        "reflection.max_additional_retrievals": 2,
        "retrieval_cache.max_entries": 256,
        "retrieval_cache.ttl_seconds": 300.0,
    }


def validate_candidate(
    candidate: CandidateConfiguration, *, allow_reindexing: bool = False
) -> CandidateValidationResult:
    """Validate candidate parameters against the trusted allowlist and cross-field rules."""
    errors: list[str] = []
    if candidate.requires_reindexing and not allow_reindexing:
        errors.append("reindexing_not_allowed")

    for path, val in candidate.changes.items():
        spec = _PARAMS_BY_PATH.get(path)
        if spec is None:
            errors.append(f"disallowed_parameter:{path}")
            continue
        if spec.requires_reindexing and not allow_reindexing:
            errors.append(f"reindexing_not_allowed:{path}")
        if spec.param_type == "bool":
            if not isinstance(val, bool):
                errors.append(f"invalid_type:{path}")
        elif spec.param_type == "int":
            if isinstance(val, bool) or not isinstance(val, int):
                errors.append(f"invalid_type:{path}")
            elif (spec.min_value is not None and val < spec.min_value) or (
                spec.max_value is not None and val > spec.max_value
            ):
                errors.append(f"out_of_range:{path}")
        elif spec.param_type == "float":
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                errors.append(f"invalid_type:{path}")
            elif (spec.min_value is not None and float(val) < spec.min_value) or (
                spec.max_value is not None and float(val) > spec.max_value
            ):
                errors.append(f"out_of_range:{path}")
        elif spec.param_type == "enum":
            if not isinstance(val, str) or val not in spec.allowed_values:
                errors.append(f"invalid_enum:{path}")

    p = candidate.parameters
    if int(p.get("retrieval.final_top_k", 5)) > int(p.get("retrieval.dense_top_k", 20)) + int(
        p.get("retrieval.sparse_top_k", 20)
    ):
        errors.append("final_top_k_exceeds_candidate_budget")

    return CandidateValidationResult(valid=not errors, errors=tuple(errors))
