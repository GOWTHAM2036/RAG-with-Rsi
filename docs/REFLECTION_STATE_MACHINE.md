# Reflection State Machine

`ReflectionStateMachine` (`backend/evorag/reflection_state.py`) enforces legal transitions across:

`initial_answer -> deterministic_verification -> [semantic_verification] -> [correction_planning -> [targeted_retrieval] -> [regeneration -> final_verification -> candidate_comparison]] -> {accepted, partially_accepted, abstained, failed, cancelled, budget_exhausted}`.

- Illegal transitions raise `IllegalTransitionError`.
- Repeated-action loop guard prevents repeating the same corrective action.
- Terminal states release all resource reservations.
