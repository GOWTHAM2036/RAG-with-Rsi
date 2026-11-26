"""Unit tests for bounded conversation context and SQLite WAL execution tracing."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from evorag.config import ConversationSettings, TracingSettings
from evorag.conversation import ConversationStore
from evorag.rag_types import CitationValidationResult, PipelineStageMetrics, RAGExecutionTrace
from evorag.tracing import TraceStore


def _trace(trace_id: str) -> RAGExecutionTrace:
    return RAGExecutionTrace(
        trace_id=trace_id,
        request_id="req-1",
        pipeline_version="0.5.0",
        prompt_template_version="1.0.0",
        retrieval_mode="hybrid",
        effective_retrieval_mode="hybrid",
        selected_evidence_ids=("E1",),
        stage_metrics=PipelineStageMetrics(
            retrieval_ms=1.0, context_ms=0.5, generation_ms=2.0, validation_ms=0.2, total_ms=3.7
        ),
        estimated_input_tokens=100,
        citation_validation=CitationValidationResult(valid=True),
        admission_outcome="accepted",
        created_at=datetime.now(UTC),
    )


def test_conversation_store_bounds_and_disabled() -> None:
    store = ConversationStore(ConversationSettings(enabled=True, max_history_messages=2))
    store.append_turn("c1", "q1", "a1")
    state = store.append_turn("c1", "q2", "a2")
    assert len(state.messages) == 2
    assert state.messages[0].content == "q2"
    assert state.persisted is False

    disabled = ConversationStore(ConversationSettings(enabled=False))
    assert disabled.append_turn("c1", "q1", "a1").messages == ()
    assert disabled.get("c1").messages == ()


def test_trace_store_retention_and_failure_isolation(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "meta.sqlite3", TracingSettings(enabled=True, max_records=2))
    store.save(_trace("t1"))
    store.save(_trace("t2"))
    store.save(_trace("t3"))
    assert store.count() == 2
    assert store.get("t1") is None
    assert store.get("t3") is not None

    with patch.object(store, "_connect", side_effect=sqlite3.OperationalError("disk full")):
        store.save(_trace("t4"))
