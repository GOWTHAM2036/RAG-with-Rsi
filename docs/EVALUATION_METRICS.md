# Evaluation Metrics

Implemented from scratch in `backend/evorag/evaluation.py` and `backend/evorag/eval_metrics.py`:

- **Retrieval**: `precision_at_k`, `recall_at_k`, `hit_rate_at_k`, `f1_at_k`, `reciprocal_rank`, `ndcg_at_k` (binary and graded `dcg_at_k_graded` / `ndcg_at_k_graded`).
- **Answer Correctness Proxies**: `normalized_exact_match`, `token_precision_recall_f1`, numeric tolerance matching, and forbidden-fact detection. Lexical proxies are explicitly separated from semantic judgment.
- **Citation & Faithfulness**: `score_citations` (`citation_precision`, `citation_recall`) and `score_faithfulness` (`supported / assessable`, reporting `uncertain` separately).
- **Abstention Confusion Matrix**: `compute_abstention_matrix` tracks `tp_answered`, `fn_false_abstain`, `tn_correct_abstain`, `fp_false_answer`, and `resource_rejections` separately.
