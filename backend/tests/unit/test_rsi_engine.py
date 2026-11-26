"""Unit tests for Phase 09 RSI state machine, search space, optimizer and store."""

from pathlib import Path

import pytest
from evorag.config import RSIStorageSettings
from evorag.rsi_analysis import (
    analyze_failure_patterns,
    generate_hypotheses,
    prioritize_opportunities,
)
from evorag.rsi_optimizer import (
    compute_composite_utility,
    generate_candidates,
    rank_candidates_pareto,
)
from evorag.rsi_search_space import default_tunable_parameters, validate_candidate
from evorag.rsi_state import IllegalRSITransitionError, RSIStateMachine
from evorag.rsi_store import RSIStore
from evorag.rsi_types import CandidateConfiguration, CandidateScore, RSIBudget


def test_rsi_state_machine_and_budgets() -> None:
    sm = RSIStateMachine(RSIBudget(max_cycles=1, max_experiments=2))
    assert not sm.is_terminal()
    sm.transition("collecting_feedback", "start")
    sm.transition("analyzing_failures", "analyze")
    assert sm.can_start_cycle()
    sm.record_cycle()
    assert not sm.can_start_cycle()
    assert sm.can_run_experiment("fp1")
    sm.record_experiment("fp1")
    assert not sm.can_run_experiment("fp1")
    sm.transition("inconclusive", "done")
    assert sm.is_terminal()
    with pytest.raises(IllegalRSITransitionError):
        sm.transition("generating_candidates", "illegal")


def test_search_space_validation_and_reward_hacking_rejection() -> None:
    base = default_tunable_parameters()
    bad_param = CandidateConfiguration(
        candidate_id="c1",
        parent_fingerprint="p0",
        fingerprint="f1",
        hypothesis_id="h1",
        changes={"resources.minimum_free_ram_gb": 0.1},
        parameters={**base, "resources.minimum_free_ram_gb": 0.1},
    )
    res = validate_candidate(bad_param)
    assert not res.valid and "disallowed_parameter:resources.minimum_free_ram_gb" in res.errors

    out_of_range = CandidateConfiguration(
        candidate_id="c2",
        parent_fingerprint="p0",
        fingerprint="f2",
        hypothesis_id="h1",
        changes={"bm25.k1": 99.0},
        parameters={**base, "bm25.k1": 99.0},
    )
    assert not validate_candidate(out_of_range).valid

    reindex = CandidateConfiguration(
        candidate_id="c3",
        parent_fingerprint="p0",
        fingerprint="f3",
        hypothesis_id="h1",
        changes={"bm25.k1": 1.2},
        parameters={**base, "bm25.k1": 1.2},
        requires_reindexing=True,
    )
    assert "reindexing_not_allowed" in validate_candidate(reindex, allow_reindexing=False).errors

    # Cover int/float/bool/enum type, range, and cross-field validation errors.
    for bad_changes in (
        {"reranking.enabled": "yes"},
        {"fusion.k": "sixty"},
        {"fusion.k": 999},
        {"bm25.k1": True},
        {"adaptive_rag.default_strategy": "invalid_enum"},
        {"retrieval.final_top_k": 8, "retrieval.dense_top_k": 5, "retrieval.sparse_top_k": 2},
    ):
        cand = CandidateConfiguration(
            candidate_id="bad",
            parent_fingerprint="p0",
            fingerprint="f_bad",
            hypothesis_id="h1",
            changes=bad_changes,
            parameters={**base, **bad_changes},
        )
        assert not validate_candidate(cand).valid


def test_analysis_optimization_and_pareto_ranking() -> None:
    patterns = analyze_failure_patterns(
        {
            "missing_relevant_documents": 12,
            "incorrect_retrieval_ranking": 5,
            "incomplete_evidence_coverage": 4,
            "other_minor": 1,
        },
        minimum_samples=3,
    )
    assert prioritize_opportunities(()) == ()
    opps = prioritize_opportunities(patterns)
    hyps = generate_hypotheses(opps, max_hypotheses=4)
    assert len(hyps) == 3

    other_hyps = generate_hypotheses(
        prioritize_opportunities(
            analyze_failure_patterns(
                {"incomplete_evidence_coverage": 5, "context_budget_overflow": 4},
                minimum_samples=1,
            )
        ),
        max_hypotheses=3,
    )
    assert len(other_hyps) == 2

    base = default_tunable_parameters()
    c_coord = generate_candidates(base, hyps, strategy="coordinate_search", max_candidates=2)
    c_rand1 = generate_candidates(
        base, hyps, strategy="seeded_random_search", max_candidates=2, seed=42
    )
    c_rand2 = generate_candidates(
        base, hyps, strategy="seeded_random_search", max_candidates=2, seed=42
    )
    c_loc = generate_candidates(base, hyps, strategy="local_refinement", max_candidates=2)
    c_pareto = generate_candidates(base, hyps, strategy="pareto_selection", max_candidates=3)
    assert len(c_coord) == 2 and c_rand1 == c_rand2 and len(c_loc) == 2 and len(c_pareto) == 3

    util = compute_composite_utility(
        retrieval_quality=0.8, answer_correctness=0.8, faithfulness=0.9, citation_quality=1.0
    )
    assert util == pytest.approx(0.86)
    assert (
        compute_composite_utility(
            retrieval_quality=None, answer_correctness=0.8, faithfulness=0.9, citation_quality=1.0
        )
        is None
    )

    s1 = CandidateScore(
        candidate_id="a",
        fingerprint="fa",
        primary_metric=0.9,
        p95_latency_ms=10.0,
        passed_gates=True,
    )
    s2 = CandidateScore(
        candidate_id="b",
        fingerprint="fb",
        primary_metric=0.7,
        p95_latency_ms=20.0,
        passed_gates=True,
    )
    ranked = rank_candidates_pareto([s2, s1])
    assert ranked.ranked[0].candidate_id == "a" and ranked.ranked[0].pareto_optimal is True


def test_rsi_store_and_versions(tmp_path: Path) -> None:
    from evorag.rsi_cli import main as rsi_cli_main

    store = RSIStore(tmp_path / "rsi.sqlite3", RSIStorageSettings())
    active = store.active_version()
    assert active.version_id == "v0_baseline" and active.active is True
    assert rsi_cli_main(["status"]) == 0
    assert rsi_cli_main(["configurations"]) == 0
    assert rsi_cli_main(["recommendations"]) == 0
