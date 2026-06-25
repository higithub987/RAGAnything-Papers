import uuid

from fastapi import APIRouter, HTTPException

from .. import session_store
from ..models import ChatSessionDetail, ChatSessionSummary, RenameSessionRequest

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=ChatSessionSummary)
def create_session():
    session = session_store.create_session(str(uuid.uuid4()))
    return ChatSessionSummary(
        session_id=session.session_id,
        title=session_store.display_title(session),
        created_at=session.created_at,
    )


@router.get("", response_model=list[ChatSessionSummary])
def list_sessions():
    return [
        ChatSessionSummary(
            session_id=s.session_id,
            title=session_store.display_title(s),
            created_at=s.created_at,
        )
        for s in session_store.list_sessions()
    ]


@router.get("/{session_id}", response_model=ChatSessionDetail)
def get_session(session_id: str):
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return ChatSessionDetail(
        session_id=session.session_id,
        title=session_store.display_title(session),
        created_at=session.created_at,
        turns=session.turns,
    )


@router.patch("/{session_id}", response_model=ChatSessionSummary)
def rename_session(session_id: str, request: RenameSessionRequest):
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    session_store.rename_session(session, request.name)
    return ChatSessionSummary(
        session_id=session.session_id,
        title=session_store.display_title(session),
        created_at=session.created_at,
    )


@router.delete("/{session_id}")
def delete_session(session_id: str):
    if not session_store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"deleted": True}
