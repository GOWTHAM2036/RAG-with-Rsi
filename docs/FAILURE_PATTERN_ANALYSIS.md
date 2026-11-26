# Failure Pattern Analysis

`analyze_failure_patterns` and `prioritize_opportunities` (`backend/evorag/rsi_analysis.py`) aggregate recurring failure classifications using `minimum_samples` thresholds and score opportunities via `Priority = normalized_freq * severity * expected_benefit * reliability / cost`. Heuristic priority scores are not calibrated probabilities.
