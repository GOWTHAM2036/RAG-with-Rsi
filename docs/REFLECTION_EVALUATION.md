# Reflection Evaluation

`benchmarks/reflection_benchmark.py` defines an **80-case** reflection dataset (`40 dev`, `40 held_out`) across 8 categories (`correct_initial`, `unsupported_claim`, `incorrect_citation`, `incomplete_multipart`, `conflicting_evidence`, `unanswerable`, `adversarial_doc`, `correctable_retrieval`).

Metrics tracked via `ReflectionEvaluationResult`: `trigger_precision`, `trigger_recall`, `correction_success_rate`, and `regression_introduction_rate`.
