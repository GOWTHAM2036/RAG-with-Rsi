"""Custom optimization algorithms: coordinate search, seeded random, local and Pareto."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from .eval_stats import is_pareto_dominated
from .rsi_search_space import TRUSTED_SEARCH_SPACE, validate_candidate
from .rsi_types import (
    CandidateConfiguration,
    CandidateRanking,
    CandidateScore,
    ImprovementHypothesis,
    OptimizationStrategy,
)


def generate_candidates(
    parent_params: dict[str, Any],
    hypotheses: Sequence[ImprovementHypothesis],
    *,
    strategy: OptimizationStrategy = "coordinate_search",
    max_candidates: int = 3,
    seed: int = 42,
    allow_reindexing: bool = False,
) -> tuple[CandidateConfiguration, ...]:
    """Generate and validate bounded candidate configurations from an immutable parent."""
    parent_fp = CandidateConfiguration.compute_fingerprint(parent_params)
    candidates: list[CandidateConfiguration] = []
    seen_fps: set[str] = {parent_fp}

    if strategy == "coordinate_search":
        for idx, hyp in enumerate(hypotheses):
            if len(candidates) >= max_candidates:
                break
            merged = {**parent_params, **hyp.proposed_changes}
            fp = CandidateConfiguration.compute_fingerprint(merged)
            if fp in seen_fps:
                continue
            cand = CandidateConfiguration(
                candidate_id=f"cand_{idx}_{fp[:8]}",
                parent_fingerprint=parent_fp,
                fingerprint=fp,
                hypothesis_id=hyp.hypothesis_id,
                changes=dict(hyp.proposed_changes),
                parameters=merged,
            )
            if validate_candidate(cand, allow_reindexing=allow_reindexing).valid:
                seen_fps.add(fp)
                candidates.append(cand)
    elif strategy == "seeded_random_search":
        rng = np.random.default_rng(seed)
        tunable = [p for p in TRUSTED_SEARCH_SPACE.parameters if not p.requires_reindexing]
        hyp_id = hypotheses[0].hypothesis_id if hypotheses else "hyp_random"
        for attempt in range(max_candidates * 3):
            if len(candidates) >= max_candidates or not tunable:
                break
            spec = tunable[int(rng.integers(0, len(tunable)))]
            val: Any
            if spec.param_type == "bool":
                val = bool(rng.integers(0, 2))
            elif spec.param_type == "enum" and spec.allowed_values:
                val = spec.allowed_values[int(rng.integers(0, len(spec.allowed_values)))]
            elif (
                spec.param_type == "int"
                and spec.min_value is not None
                and spec.max_value is not None
            ):
                val = int(rng.integers(int(spec.min_value), int(spec.max_value) + 1))
            elif spec.min_value is not None and spec.max_value is not None:
                val = round(float(rng.uniform(spec.min_value, spec.max_value)), 3)
            else:
                continue
            merged = {**parent_params, spec.path: val}
            fp = CandidateConfiguration.compute_fingerprint(merged)
            if fp in seen_fps:
                continue
            cand = CandidateConfiguration(
                candidate_id=f"cand_rnd_{attempt}_{fp[:8]}",
                parent_fingerprint=parent_fp,
                fingerprint=fp,
                hypothesis_id=hyp_id,
                changes={spec.path: val},
                parameters=merged,
            )
            if validate_candidate(cand, allow_reindexing=allow_reindexing).valid:
                seen_fps.add(fp)
                candidates.append(cand)
    else:
        # local_refinement / pareto_selection: step numerical parameters locally around parent.
        hyp_id = hypotheses[0].hypothesis_id if hypotheses else "hyp_local"
        for idx, (path, factor) in enumerate(
            (("fusion.sparse_weight", 1.1), ("fusion.dense_weight", 0.9), ("bm25.k1", 0.9))
        ):
            if len(candidates) >= max_candidates:
                break
            cur = float(parent_params.get(path, 1.0))
            new_val = round(cur * factor, 3)
            merged = {**parent_params, path: new_val}
            fp = CandidateConfiguration.compute_fingerprint(merged)
            if fp in seen_fps:
                continue
            cand = CandidateConfiguration(
                candidate_id=f"cand_loc_{idx}_{fp[:8]}",
                parent_fingerprint=parent_fp,
                fingerprint=fp,
                hypothesis_id=hyp_id,
                changes={path: new_val},
                parameters=merged,
            )
            if validate_candidate(cand, allow_reindexing=allow_reindexing).valid:
                seen_fps.add(fp)
                candidates.append(cand)

    return tuple(candidates)


def compute_composite_utility(
    *,
    retrieval_quality: float | None,
    answer_correctness: float | None,
    faithfulness: float | None,
    citation_quality: float | None,
) -> float | None:
    """Compute weighted composite utility only when required quality signals are present."""
    if retrieval_quality is None or citation_quality is None:
        return None
    ans = answer_correctness if answer_correctness is not None else retrieval_quality
    faith = faithfulness if faithfulness is not None else citation_quality
    return round(0.25 * retrieval_quality + 0.30 * ans + 0.30 * faith + 0.15 * citation_quality, 4)


def rank_candidates_pareto(scores: Sequence[CandidateScore]) -> CandidateRanking:
    """Identify Pareto non-dominated candidates across quality, latency and RAM."""
    marked: list[CandidateScore] = []
    objs = [
        (
            s.composite_utility if s.composite_utility is not None else (s.primary_metric or 0.0),
            -s.p95_latency_ms,
            -s.additional_ram_mib,
        )
        for s in scores
    ]
    for i, s in enumerate(scores):
        dominated = any(
            i != j and is_pareto_dominated(objs[i], objs[j]) for j in range(len(scores))
        )
        marked.append(s.model_copy(update={"pareto_optimal": not dominated}))
    marked.sort(
        key=lambda c: (
            not c.passed_gates,
            not c.pareto_optimal,
            -(c.composite_utility or c.primary_metric or 0.0),
            c.p95_latency_ms,
        )
    )
    return CandidateRanking(ranked=tuple(marked))
