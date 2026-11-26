"""Unit tests for Phase 07 dataset registry, metrics, stats, failure taxonomy and gates."""

from datetime import UTC, datetime

import pytest
from evorag.dataset_registry import DatasetRegistry, validate_dataset
from evorag.eval_cli import main as eval_cli_main
from evorag.eval_metrics import (
    compute_abstention_matrix,
    f1_at_k,
    ndcg_at_k_graded,
    score_answer_proxy,
    score_citations,
    score_faithfulness,
    token_precision_recall_f1,
)
from evorag.eval_stats import (
    bootstrap_ci,
    compare_runs,
    detect_regressions,
    is_pareto_dominated,
    paired_bootstrap_delta,
)
from evorag.eval_types import (
    AbstentionConfusionMatrix,
    DatasetVersion,
    EvaluationCase,
    EvaluationDataset,
    EvaluationRun,
    EvaluationRunResult,
    ExperimentConfiguration,
    GroundTruth,
    MetricResult,
    QualityGate,
)
from evorag.failure_analysis import classify_case_failures
from evorag.quality_gates import evaluate_gates
from evorag.rag_types import CitationReference, RAGResponse


def test_dataset_registry_and_contamination_detection() -> None:
    reg = DatasetRegistry()
    ds = reg.get("evorag_comprehensive_v1")
    assert len(ds.cases) == 108
    assert validate_dataset(ds) == ()
    assert len(reg.cases_for_split("evorag_comprehensive_v1", "dev")) == 54
    assert len(reg.cases_for_split("evorag_comprehensive_v1", "held_out")) == 54

    c1 = ds.cases[0]
    c_held = c1.model_copy(update={"case_id": "dup_id", "split": "held_out"})
    contaminated = EvaluationDataset(
        metadata=DatasetVersion(
            dataset_id="bad",
            version="1.0.0",
            fingerprint="0000000000000000",
            created_at="2026-11-26",
            known_limitations="test",
        ),
        cases=(c1, c1, c_held),
    )
    errs = validate_dataset(contaminated)
    assert any(e.startswith("duplicate_case_id") for e in errs)
    assert any(e.startswith("dev_heldout_overlap") for e in errs)
    assert "fingerprint_mismatch" in errs
    assert reg.register(contaminated) == errs
    with pytest.raises(KeyError):
        reg.get("missing-ds")


def test_metrics_and_abstention_confusion_matrix() -> None:
    assert f1_at_k(1.0, 0.5) == pytest.approx(2 / 3)
    assert f1_at_k(0.0, 0.0) == 0.0
    assert f1_at_k(0.5, None) is None

    grades = {"d1": 2, "d2": 1}
    assert ndcg_at_k_graded(["d1", "d2"], grades, 2) == pytest.approx(1.0)
    assert ndcg_at_k_graded(["d1"], {}, 2) is None

    p, r, f1 = token_precision_recall_f1("sqlite wal mode", "sqlite wal")
    assert p == pytest.approx(2 / 3) and r == pytest.approx(1.0) and f1 == pytest.approx(0.8)

    em, tf1, forb = score_answer_proxy(
        "The floor is 3 GiB.",
        GroundTruth(expected_numeric=3.0, numeric_tolerance=0.01),
    )
    assert em == 1.0 and tf1 == 1.0 and not forb

    cit = CitationReference(
        citation_number=1,
        evidence_id="E1",
        document_id="doc1",
        chunk_id="c1",
        excerpt="text",
    )
    cp, cr = score_citations([cit], frozenset({"arch.md"}), {"doc1": "arch.md"})
    assert cp == 1.0 and cr == 1.0

    faith = score_faithfulness(["supported", "uncertain", "contradicted"])
    assert faith.value == pytest.approx(0.5)
    assert score_faithfulness(["uncertain"]).value is None

    cm = compute_abstention_matrix(
        [(False, False, False), (True, True, False), (False, True, False), (False, False, True)]
    )
    assert cm.tp_answered == 1 and cm.tn_correct_abstain == 1
    assert cm.fn_false_abstain == 1 and cm.resource_rejections == 1


def test_bootstrap_pareto_failures_and_gates() -> None:
    ci1 = bootstrap_ci("f1", [0.8, 0.9, 1.0], seed=42)
    ci2 = bootstrap_ci("f1", [0.8, 0.9, 1.0], seed=42)
    assert ci1 == ci2 and ci1 is not None
    assert bootstrap_ci("empty", []) is None

    delta = paired_bootstrap_delta("f1", [0.1, 0.2, 0.3], [0.8, 0.9, 1.0], seed=42)
    assert delta.verdict == "improved" and delta.wins == 3
    assert is_pareto_dominated([0.5, 0.5], [0.6, 0.5]) is True

    case = EvaluationCase(
        case_id="c1",
        category="factual",
        split="dev",
        question="What?",
        ground_truth=GroundTruth(expect_abstain=False),
    )
    resp = RAGResponse(
        answer="Insufficient",
        status="insufficient_evidence",
        admission_outcome="insufficient_evidence",
        retrieval_mode="hybrid",
        effective_retrieval_mode="hybrid",
        trace_id="t1",
    )
    fails = classify_case_failures(case, resp, 0.0, 1.0, 0.0)
    assert any(f.category == "incorrect_abstention" for f in fails)
    assert any(f.category == "missing_relevant_documents" for f in fails)

    cfg = ExperimentConfiguration(dataset_id="evorag_comprehensive_v1", dataset_version="1.0.0")
    run_res = EvaluationRunResult(
        run=EvaluationRun(
            run_id="r1",
            status="completed",
            config_fingerprint=cfg.fingerprint(),
            dataset_id=cfg.dataset_id,
            dataset_version=cfg.dataset_version,
            split="dev",
            execution_mode="offline",
            total_cases=1,
            completed_cases=1,
            started_at=datetime.now(UTC),
        ),
        config=cfg,
        metrics=(
            MetricResult(name="citation_validity", value=1.0, sample_count=1),
            MetricResult(name="recall_at_k", value=0.5, sample_count=1),
            MetricResult(
                name="faithfulness", value=None, unavailable_reason="missing", sample_count=0
            ),
        ),
        abstention=AbstentionConfusionMatrix(),
    )
    gates = {g.gate_id: g.decision for g in evaluate_gates(run_res)}
    assert gates["citation_validity_floor"] == "pass"
    assert gates["recall_floor"] == "fail"
    assert gates["abstention_accuracy_floor"] == "inconclusive"
    assert gates["live_semantic_faithfulness"] == "blocked"

    reg_delta = paired_bootstrap_delta("f1", [0.9, 0.8, 0.9], [0.1, 0.2, 0.1], seed=42)
    assert reg_delta.verdict == "regressed"
    assert paired_bootstrap_delta("f1", [0.5], [0.6]).verdict == "inconclusive"

    comp = compare_runs(run_res, run_res)
    reg_rep = detect_regressions(run_res, run_res, comp)
    assert reg_rep.has_regression is False

    reg_gate = evaluate_gates(
        run_res,
        gates=[QualityGate(gate_id="reg", metric_name="recall_at_k", max_regression=0.05)],
        comparison=comp.model_copy(
            update={"deltas": (reg_delta.model_copy(update={"metric_name": "recall_at_k"}),)}
        ),
    )
    assert reg_gate[0].decision == "fail"

    for outcome_name in (
        "resource_unavailable",
        "invalid_model_output",
        "invalid_citation",
        "generation_failed",
    ):
        adm = "invalid_citations" if outcome_name == "invalid_citation" else outcome_name
        r_err = resp.model_copy(update={"admission_outcome": adm})
        assert classify_case_failures(case, r_err, 0.5, 0.5, 0.0)


def test_eval_cli_commands() -> None:
    assert eval_cli_main(["list-datasets"]) == 0
    assert eval_cli_main(["validate-datasets"]) == 0
    assert eval_cli_main(["show-run", "missing-run"]) == 1
