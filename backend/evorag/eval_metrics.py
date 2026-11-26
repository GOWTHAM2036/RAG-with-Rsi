"""Custom deterministic evaluation metrics for retrieval, answers, citations and abstention."""

import math
import re
from collections.abc import Sequence

from .eval_types import (
    AbstentionConfusionMatrix,
    FaithfulnessLabel,
    GroundTruth,
    MetricResult,
)
from .rag_types import CitationReference

_WORD_RE = re.compile(r"[a-z0-9_./:-]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def f1_at_k(precision: float, recall: float | None) -> float | None:
    if recall is None:
        return None
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def dcg_at_k_graded(retrieved_docs: Sequence[str], grades: dict[str, int], k: int) -> float:
    dcg = 0.0
    for rank, doc in enumerate(retrieved_docs[:k], start=1):
        rel = grades.get(doc, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1.0)
    return dcg


def ndcg_at_k_graded(retrieved_docs: Sequence[str], grades: dict[str, int], k: int) -> float | None:
    positives = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    if not positives:
        return None
    idcg = sum((2**rel - 1) / math.log2(rank + 1.0) for rank, rel in enumerate(positives, start=1))
    if idcg <= 0.0:
        return 0.0
    return float(dcg_at_k_graded(retrieved_docs, grades, k) / idcg)


def normalized_exact_match(predicted: str, expected: str) -> float:
    return 1.0 if " ".join(predicted.lower().split()) == " ".join(expected.lower().split()) else 0.0


def token_precision_recall_f1(predicted: str, expected: str) -> tuple[float, float, float]:
    p_toks = _tokens(predicted)
    e_toks = _tokens(expected)
    if not p_toks and not e_toks:
        return 1.0, 1.0, 1.0
    if not p_toks or not e_toks:
        return 0.0, 0.0, 0.0
    e_counts: dict[str, int] = {}
    for t in e_toks:
        e_counts[t] = e_counts.get(t, 0) + 1
    overlap = 0
    for t in p_toks:
        if e_counts.get(t, 0) > 0:
            overlap += 1
            e_counts[t] -= 1
    prec = overlap / len(p_toks)
    rec = overlap / len(e_toks)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
    return prec, rec, f1


def score_answer_proxy(
    predicted: str, truth: GroundTruth
) -> tuple[float | None, float | None, bool]:
    """Return (exact_match, token_f1, forbidden_violated). Lexical proxy only, not semantic."""
    if truth.expected_numeric is not None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", predicted)
        matched = any(
            abs(float(n) - truth.expected_numeric) <= truth.numeric_tolerance for n in nums
        )
        return (1.0 if matched else 0.0), (1.0 if matched else 0.0), False

    refs = list(truth.acceptable_answers)
    if truth.expected_answer:
        refs.insert(0, truth.expected_answer)
    if not refs:
        return None, None, False

    best_em = max(normalized_exact_match(predicted, r) for r in refs)
    best_f1 = max(token_precision_recall_f1(predicted, r)[2] for r in refs)
    lower_pred = predicted.lower()
    forbidden = any(f.lower() in lower_pred for f in truth.forbidden_facts)
    return best_em, best_f1, forbidden


def score_citations(
    citations: Sequence[CitationReference],
    expected_docs: frozenset[str],
    doc_id_to_name: dict[str, str],
) -> tuple[float | None, float | None]:
    """Return (citation_precision, citation_recall) against expected source documents."""
    if not citations and not expected_docs:
        return None, None
    cited_names = {doc_id_to_name.get(c.document_id, c.document_id) for c in citations}
    prec = (
        sum(1 for name in cited_names if name in expected_docs) / len(cited_names)
        if cited_names
        else 0.0
    )
    rec = (
        sum(1 for doc in expected_docs if doc in cited_names) / len(expected_docs)
        if expected_docs
        else None
    )
    return prec, rec


def score_faithfulness(labels: Sequence[FaithfulnessLabel]) -> MetricResult:
    """Compute supported / assessable claims while reporting uncertain claims separately."""
    if not labels:
        return MetricResult(
            name="faithfulness",
            value=None,
            unavailable_reason="no_faithfulness_labels",
            sample_count=0,
        )
    assessable = [lbl for lbl in labels if lbl != "uncertain"]
    if not assessable:
        return MetricResult(
            name="faithfulness",
            value=None,
            unavailable_reason="all_claims_uncertain",
            sample_count=len(labels),
        )
    supported = sum(1 for lbl in assessable if lbl == "supported")
    return MetricResult(
        name="faithfulness",
        value=supported / len(assessable),
        sample_count=len(assessable),
    )


def compute_abstention_matrix(
    outcomes: Sequence[tuple[bool, bool, bool]],
) -> AbstentionConfusionMatrix:
    """Compute confusion matrix from (expected_abstain, actual_abstain, resource_rejected)."""
    tp = fn = tn = fp = rej = 0
    for exp_abs, act_abs, res_rej in outcomes:
        if res_rej:
            rej += 1
            continue
        if exp_abs and act_abs:
            tn += 1
        elif exp_abs and not act_abs:
            fp += 1
        elif not exp_abs and act_abs:
            fn += 1
        else:
            tp += 1

    total = tp + fn + tn + fp
    answerable = tp + fn
    unanswerable = tn + fp
    predicted_abs = tn + fn
    return AbstentionConfusionMatrix(
        tp_answered=tp,
        fn_false_abstain=fn,
        tn_correct_abstain=tn,
        fp_false_answer=fp,
        resource_rejections=rej,
        accuracy=(tp + tn) / total if total else None,
        precision=tn / predicted_abs if predicted_abs else None,
        recall=tn / unanswerable if unanswerable else None,
        false_abstention_rate=fn / answerable if answerable else None,
        false_answer_rate=fp / unanswerable if unanswerable else None,
        answerable_coverage=tp / answerable if answerable else None,
    )
