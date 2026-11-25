"""Hand-calculated mathematical fixtures and edge-case tests for BM25."""

import math

import pytest
from evorag.bm25 import BM25Candidate, bm25_idf, score_bm25


def test_hand_calculated_bm25_fixture() -> None:
    # Corpus: N = 3 documents, lengths [4, 2, 6] => avgdl = 4.0
    # Term "alpha" (df = 1): IDF = ln(1 + (3 - 1 + 0.5) / (1 + 0.5)) = ln(1 + 2.5/1.5)
    # Term "common" (df = 3): IDF = ln(1 + (3 - 3 + 0.5) / (3 + 0.5)) = ln(1 + 0.5/3.5)
    N = 3
    avgdl = 4.0
    k1 = 1.5
    b = 0.75
    expected_idf_alpha = math.log(1.0 + 2.5 / 1.5)
    expected_idf_common = math.log(1.0 + 0.5 / 3.5)
    assert bm25_idf(N, 1) == pytest.approx(expected_idf_alpha, rel=1e-12)
    assert bm25_idf(N, 3) == pytest.approx(expected_idf_common, rel=1e-12)
    assert expected_idf_alpha > expected_idf_common > 0.0

    # Doc d1: length=4, tf(alpha)=2, tf(common)=1
    # length_norm = 1 - 0.75 + 0.75 * (4 / 4) = 1.0
    # alpha_score = idf_alpha * (2 * 2.5) / (2 + 1.5 * 1.0) = idf_alpha * 5.0 / 3.5
    # common_score = idf_common * (1 * 2.5) / (1 + 1.5 * 1.0) = idf_common * 2.5 / 2.5 = idf_common
    expected_d1 = expected_idf_alpha * (5.0 / 3.5) + expected_idf_common

    # Doc d2: length=2, tf(alpha)=0, tf(common)=1
    # length_norm = 1 - 0.75 + 0.75 * (2 / 4) = 0.625
    # common_score = idf_common * (1 * 2.5) / (1 + 1.5 * 0.625) = idf_common * 2.5 / 1.9375
    expected_d2 = expected_idf_common * (2.5 / (1.0 + 1.5 * 0.625))

    candidates = [
        BM25Candidate("d2", 2, {"common": 1, "beta": 1}),
        BM25Candidate("d1", 4, {"alpha": 2, "common": 1}),
        BM25Candidate("d3", 6, {"common": 1}),
    ]
    results = score_bm25(
        ["alpha", "common", "unknown"],
        candidates,
        corpus_size=N,
        avg_doc_length=avgdl,
        document_frequencies={"alpha": 1, "common": 3},
        k1=k1,
        b=b,
        top_k=10,
    )
    assert [r.chunk_id for r in results] == ["d1", "d2", "d3"]
    assert results[0].score == pytest.approx(expected_d1, rel=1e-12)
    assert results[1].score == pytest.approx(expected_d2, rel=1e-12)
    assert results[0].matched_terms == ("alpha", "common")


def test_bm25_edge_cases_and_tie_breaking() -> None:
    cands = [
        BM25Candidate("c_b", 3, {"rust": 1}),
        BM25Candidate("c_a", 3, {"rust": 1}),
        BM25Candidate("c_empty", 0, {"rust": 1}),
    ]
    # Identical scores tie-break deterministically by chunk_id ascending.
    tied = score_bm25(
        ["rust"],
        cands,
        corpus_size=2,
        avg_doc_length=3.0,
        document_frequencies={"rust": 2},
        top_k=5,
    )
    assert [r.chunk_id for r in tied] == ["c_a", "c_b"]
    assert tied[0].score == pytest.approx(tied[1].score)

    # Repeated query terms sum proportionally.
    single = score_bm25(
        ["rust"], cands, corpus_size=2, avg_doc_length=3.0, document_frequencies={"rust": 2}
    )
    repeated = score_bm25(
        ["rust", "rust"],
        cands,
        corpus_size=2,
        avg_doc_length=3.0,
        document_frequencies={"rust": 2},
    )
    assert repeated[0].score == pytest.approx(2.0 * single[0].score)

    # Empty corpus, empty query, unknown terms, zero avgdl.
    assert score_bm25([], cands, corpus_size=2, avg_doc_length=3.0, document_frequencies={}) == []
    assert (
        score_bm25(["rust"], [], corpus_size=0, avg_doc_length=0.0, document_frequencies={}) == []
    )
    assert (
        score_bm25(
            ["missing"],
            cands,
            corpus_size=2,
            avg_doc_length=3.0,
            document_frequencies={"rust": 2},
        )
        == []
    )
    assert bm25_idf(0, 1) == 0.0

    with pytest.raises(ValueError, match="top_k"):
        score_bm25(
            ["a"], cands, corpus_size=1, avg_doc_length=1.0, document_frequencies={}, top_k=0
        )
    with pytest.raises(ValueError, match="k1"):
        score_bm25(
            ["a"], cands, corpus_size=1, avg_doc_length=1.0, document_frequencies={}, k1=-1.0
        )
    with pytest.raises(ValueError, match="b"):
        score_bm25(["a"], cands, corpus_size=1, avg_doc_length=1.0, document_frequencies={}, b=1.5)
