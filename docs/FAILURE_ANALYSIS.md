# Failure Analysis

`backend/evorag/failure_analysis.py` classifies failures across a 20-category structured taxonomy (`FailureCategory` in `eval_types.py`), explicitly marking each classification as `observed=True` or `observed=False` (inferred), and summarizes counts by category for future RSI diagnostics.
