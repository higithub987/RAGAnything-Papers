from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentTask(BaseModel):
    task_id: str
    file_name: str
    status: TaskStatus
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    doc_id: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    conversation_history: list[dict[str, str]] = []
    session_id: Optional[str] = None


class ChatSession(BaseModel):
    session_id: str
    seq: int
    name: Optional[str] = None
    created_at: datetime
    turns: list[dict[str, str]] = []  # [{"query": ..., "answer": ...}], oldest-first
    summary: str = ""
    summarized_count: int = 0


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: datetime


class ChatSessionDetail(ChatSessionSummary):
    turns: list[dict[str, str]] = []


class RenameSessionRequest(BaseModel):
    name: str


class CondenseHistoryRequest(BaseModel):
    summary: str = ""
    turns: list[dict[str, str]]


class CondenseHistoryResponse(BaseModel):
    summary: str


class MultimodalItem(BaseModel):
    type: str  # "image" | "table" | "equation"
    # table
    table_data: Optional[str] = None
    table_caption: Optional[str] = None
    # equation
    latex: Optional[str] = None
    equation_caption: Optional[str] = None
    # image
    image_path: Optional[str] = None
    image_caption: Optional[str] = None


class MultimodalQueryRequest(BaseModel):
    query: str
    multimodal_content: list[MultimodalItem]
    mode: str = "hybrid"


class QueryResponse(BaseModel):
    answer: str
    query: str
    mode: str


class ErrorResponse(BaseModel):
    detail: str
