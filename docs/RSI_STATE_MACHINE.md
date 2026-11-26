# RSI State Machine

`RSIStateMachine` (`backend/evorag/rsi_state.py`) enforces legal transitions across 21 states (`idle -> collecting_feedback -> analyzing_failures -> generating_hypotheses -> generating_candidates -> validating_candidates -> running_experiments -> comparing_candidates -> selecting_finalist -> validating_finalist -> awaiting_approval -> promoting -> monitoring -> rolling_back -> {completed, rejected, inconclusive, blocked, failed, cancelled, budget_exhausted}`).
