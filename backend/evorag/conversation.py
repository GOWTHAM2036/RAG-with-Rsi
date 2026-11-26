"""Lightweight bounded in-memory conversation context management."""

from datetime import UTC, datetime

from .config import ConversationSettings
from .rag_types import ConversationMessage, ConversationState


class ConversationStore:
    """Bounded per-conversation message buffer (non-persistent by default)."""

    def __init__(self, settings: ConversationSettings, max_conversations: int = 1024) -> None:
        self.settings = settings
        self.max_conversations = max_conversations
        self._sessions: dict[str, list[ConversationMessage]] = {}

    def get(self, conversation_id: str) -> ConversationState:
        if not self.settings.enabled or self.settings.max_history_messages == 0:
            return ConversationState(
                conversation_id=conversation_id,
                messages=(),
                persisted=self.settings.persist_history,
            )
        history = self._sessions.get(conversation_id, ())
        return ConversationState(
            conversation_id=conversation_id,
            messages=tuple(history),
            persisted=self.settings.persist_history,
        )

    def append_turn(self, conversation_id: str, question: str, answer: str) -> ConversationState:
        if not self.settings.enabled or self.settings.max_history_messages == 0:
            return ConversationState(
                conversation_id=conversation_id,
                messages=(),
                persisted=self.settings.persist_history,
            )
        if conversation_id not in self._sessions and len(self._sessions) >= self.max_conversations:
            oldest_key = next(iter(self._sessions))
            del self._sessions[oldest_key]
        now = datetime.now(UTC)
        buf = self._sessions.setdefault(conversation_id, [])
        buf.append(ConversationMessage(role="user", content=question, created_at=now))
        buf.append(ConversationMessage(role="assistant", content=answer, created_at=now))
        if len(buf) > self.settings.max_history_messages:
            del buf[: len(buf) - self.settings.max_history_messages]
        return ConversationState(
            conversation_id=conversation_id,
            messages=tuple(buf),
            persisted=self.settings.persist_history,
        )

    def count(self) -> int:
        return len(self._sessions)
