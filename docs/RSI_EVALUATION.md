# RSI Evaluation

RSI cycles reuse `ExperimentRunner`, `compare_runs`, and `evaluate_gates` over isolated `dev` splits (`max_dev_cases`), never mutating active runtime configuration or user document indexes.
