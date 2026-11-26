"""Deterministic evidence selection, context construction and token budget enforcement."""

from collections.abc import Sequence

from .config import ContextSettings
from .errors import IngestionError
from .rag_types import (
    ContextBudget,
    ContextPackage,
    ConversationMessage,
    EvidenceCandidate,
    SelectedEvidence,
)

# Conservative character-to-token approximation (4 chars/token + per-message framing overhead).
_CHARS_PER_TOKEN = 4
_EVIDENCE_WRAPPER_OVERHEAD = 24
_MESSAGE_OVERHEAD = 8


def estimate_tokens(text: str) -> int:
    """Return a conservative approximate token count (never exact provider tokenization)."""
    if not text:
        return 0
    return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)


def _truncate_text(text: str, max_tokens: int) -> tuple[str, bool]:
    if max_tokens <= 0 or not text:
        return "", bool(text)
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip(), True


def _is_redundant_overlap(
    candidate_text: str,
    candidate_doc: str,
    candidate_start: int,
    candidate_end: int,
    accepted: Sequence[tuple[str, int, int, str]],
) -> bool:
    norm = " ".join(candidate_text.split())
    for doc_id, start, end, text in accepted:
        if " ".join(text.split()) == norm:
            return True
        if doc_id != candidate_doc:
            continue
        overlap = max(0, min(candidate_end, end) - max(candidate_start, start))
        min_len = min(candidate_end - candidate_start, end - start)
        if min_len > 0 and overlap / min_len >= 0.85:
            return True
    return False


def select_evidence(
    candidates: Sequence[EvidenceCandidate], max_items: int
) -> tuple[SelectedEvidence, ...]:
    """Select deduplicated, non-redundant evidence candidates in rank order."""
    selected: list[SelectedEvidence] = []
    accepted_spans: list[tuple[str, int, int, str]] = []
    for cand in candidates:
        if len(selected) >= max_items:
            break
        chunk = cand.result.chunk
        if _is_redundant_overlap(
            chunk.text, chunk.document_id, chunk.start, chunk.end, accepted_spans
        ):
            continue
        eid = f"E{len(selected) + 1}"
        accepted_spans.append((chunk.document_id, chunk.start, chunk.end, chunk.text))
        selected.append(
            SelectedEvidence(
                evidence_id=eid,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                version=cand.version,
                source_id=cand.source_id,
                sources=cand.sources,
                text=chunk.text,
                original_rank=cand.result.rank,
                retrieval_mode=cand.retrieval_mode,
                score=cand.result.score,
                truncated=False,
                included_start=chunk.start,
                included_end=chunk.end,
            )
        )
    return tuple(selected)


def build_context_package(
    question: str,
    evidence: Sequence[SelectedEvidence],
    system_prompt: str,
    history: Sequence[ConversationMessage],
    settings: ContextSettings,
) -> ContextPackage:
    """Enforce context token budgets across system prompt, question, history and evidence."""
    truncations: list[str] = []

    sys_tokens = estimate_tokens(system_prompt)
    if sys_tokens > settings.system_token_budget:
        truncations.append("system_prompt_exceeded_budget")

    q_tokens = estimate_tokens(question)
    active_question = question
    if q_tokens > settings.question_token_budget:
        if settings.oversized_question_policy == "reject":
            raise IngestionError("question_exceeds_token_budget", 422)
        active_question, _ = _truncate_text(question, settings.question_token_budget)
        q_tokens = estimate_tokens(active_question)
        truncations.append("question_truncated")

    # Trim oldest conversation turns first so recent context and evidence are preserved.
    kept_history: list[ConversationMessage] = []
    hist_tokens = 0
    for msg in reversed(history):
        cost = estimate_tokens(msg.content) + _MESSAGE_OVERHEAD
        if hist_tokens + cost <= settings.history_token_budget:
            kept_history.append(msg)
            hist_tokens += cost
        else:
            truncations.append("history_trimmed")
    kept_history.reverse()

    # Reserve output, safety, system, question, and actual history before packing evidence.
    available_for_evidence = min(
        settings.evidence_token_budget,
        settings.max_context_tokens
        - settings.reserved_output_tokens
        - settings.safety_margin_tokens
        - max(sys_tokens, settings.system_token_budget)
        - max(q_tokens, 1)
        - hist_tokens,
    )
    available_for_evidence = max(0, available_for_evidence)

    packed: list[SelectedEvidence] = []
    used_evidence_tokens = 0
    for item in evidence:
        remaining = available_for_evidence - used_evidence_tokens
        if remaining <= _EVIDENCE_WRAPPER_OVERHEAD:
            truncations.append(f"evidence_omitted:{item.evidence_id}")
            continue
        text_budget = remaining - _EVIDENCE_WRAPPER_OVERHEAD
        item_tokens = estimate_tokens(item.text)
        if item_tokens <= text_budget:
            packed.append(item)
            used_evidence_tokens += item_tokens + _EVIDENCE_WRAPPER_OVERHEAD
        else:
            clipped, was_clipped = _truncate_text(item.text, text_budget)
            if not clipped:
                truncations.append(f"evidence_omitted:{item.evidence_id}")
                continue
            clipped_end = item.included_start + len(clipped)
            packed.append(
                item.model_copy(
                    update={
                        "text": clipped,
                        "truncated": was_clipped,
                        "included_end": clipped_end,
                    }
                )
            )
            used_evidence_tokens += estimate_tokens(clipped) + _EVIDENCE_WRAPPER_OVERHEAD
            truncations.append(f"evidence_truncated:{item.evidence_id}")

    total_input = sys_tokens + q_tokens + hist_tokens + used_evidence_tokens
    budget = ContextBudget(
        max_context_tokens=settings.max_context_tokens,
        system_token_budget=settings.system_token_budget,
        question_token_budget=settings.question_token_budget,
        history_token_budget=settings.history_token_budget,
        evidence_token_budget=settings.evidence_token_budget,
        reserved_output_tokens=settings.reserved_output_tokens,
        safety_margin_tokens=settings.safety_margin_tokens,
        estimated_system_tokens=sys_tokens,
        estimated_question_tokens=q_tokens,
        estimated_history_tokens=hist_tokens,
        estimated_evidence_tokens=used_evidence_tokens,
        total_estimated_input_tokens=total_input,
        truncations=tuple(truncations),
    )
    return ContextPackage(
        question=active_question,
        evidence=tuple(packed),
        budget=budget,
        history_messages=tuple(kept_history),
    )
