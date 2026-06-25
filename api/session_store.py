from datetime import datetime, timezone
from typing import Optional

from .models import ChatSession
from .rag_manager import condense_history

# session_id -> ChatSession; lives for the lifetime of the server process
_sessions: dict[str, ChatSession] = {}

# Monotonically increasing, never reused (even across deletes) -- this is what
# numbers default "New Chat (N)" titles.
_next_seq = 1

# Mirrors the chat UI's previous client-side rolling-history window: short
# chats stay fully verbatim, longer ones shrink toward a small floor so the
# context sent to the model stays bounded no matter how long a chat gets.
FULL_VERBATIM_TURNS = 5
MIN_HISTORY_WINDOW = 3


def _history_window(total_turns: int) -> int:
    if total_turns <= FULL_VERBATIM_TURNS:
        return total_turns
    return max(MIN_HISTORY_WINDOW, round((FULL_VERBATIM_TURNS**2) / total_turns))


def create_session(session_id: str) -> ChatSession:
    global _next_seq
    session = ChatSession(
        session_id=session_id, seq=_next_seq, created_at=datetime.now(timezone.utc)
    )
    _next_seq += 1
    _sessions[session_id] = session
    return session


def display_title(session: ChatSession) -> str:
    return session.name or f"New Chat ({session.seq})"


def rename_session(session: ChatSession, name: str) -> None:
    session.name = name.strip() or None


def get_session(session_id: str) -> Optional[ChatSession]:
    return _sessions.get(session_id)


def list_sessions() -> list[ChatSession]:
    return sorted(_sessions.values(), key=lambda s: s.created_at, reverse=True)


def delete_session(session_id: str) -> bool:
    return _sessions.pop(session_id, None) is not None


def add_turn(session: ChatSession, query: str, answer: str) -> None:
    session.turns.append({"query": query, "answer": answer})


async def condense_aged_out_turns(session: ChatSession) -> None:
    while len(session.turns) - session.summarized_count > _history_window(len(session.turns)):
        oldest = session.turns[session.summarized_count]
        session.summary = await condense_history(session.summary, [oldest])
        session.summarized_count += 1


def build_conversation_history(session: ChatSession) -> list[dict[str, str]]:
    history = []
    if session.summary:
        history.append(
            {"role": "user", "content": f"Earlier conversation summary: {session.summary}"}
        )
    for turn in session.turns[session.summarized_count :]:
        history.append({"role": "user", "content": turn["query"]})
        history.append({"role": "assistant", "content": turn["answer"]})
    return history
