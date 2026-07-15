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
    stage: Optional[str] = None
    progress: Optional[float] = None
    progress_message: Optional[str] = None
    # Containers this document belongs to (display only). Seeded at upload with the
    # chosen container, then refreshed from the registry on read once doc_id exists.
    container_ids: list[str] = []


class Container(BaseModel):
    """A user-created grouping of documents ("container"/"database").

    Membership is many-to-many: a document (identified by its LightRAG doc_id)
    may belong to several containers. `id` is a generated uuid so a container can
    be renamed without breaking membership. The virtual "All" view (query with no
    container filter) is NOT a stored container -- the registry only holds real
    user-created ones.
    """

    id: str
    name: str
    created_at: datetime
    member_doc_ids: list[str] = []


class ContainerNameRequest(BaseModel):
    """Body for creating or renaming a container."""

    name: str


class AssignDocumentRequest(BaseModel):
    """Body for filing a document (by its LightRAG doc_id) into a container."""

    doc_id: str


class TopicDetail(BaseModel):
    name: str
    entity_type: str = ""
    description: str = ""


class RelationDetail(BaseModel):
    source: str
    target: str
    description: str = ""


class DocumentRelatedness(BaseModel):
    doc_id: str
    file_name: str
    related_doc_id: str
    related_file_name: str
    score: float
    shared_topics: list[TopicDetail] = []
    shared_relations: list[RelationDetail] = []


class DocumentTopics(BaseModel):
    doc_id: str
    file_name: str
    topics: list[TopicDetail] = []


class RelatednessOverrideRequest(BaseModel):
    doc_id: str
    related_doc_id: str
    score: float


class RelatednessOverridesResponse(BaseModel):
    # {"docA::docB": score, ...} -- committed connection strengths
    overrides: dict[str, float] = {}
    # {doc_id: multiplier, ...} -- derived per-document query boost (>= 1.0)
    doc_boost: dict[str, float] = {}


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    conversation_history: list[dict[str, str]] = []
    session_id: Optional[str] = None
    # Synthesis-model thinking control: "off" (fast, default), "on", or "auto"
    # (route by query complexity). See rag_manager.resolve_thinking.
    thinking: str = "off"
    # Cap on reasoning tokens when thinking is on; None = uncapped. See
    # rag_manager.resolve_thinking_budget.
    thinking_budget: Optional[int] = None


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
    # See QueryRequest.thinking / thinking_budget.
    thinking: str = "off"
    thinking_budget: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    query: str
    mode: str
    # True when auto mode retried with thinking-on + wider retrieval because the
    # fast first answer looked insufficient. Always False for on/off/stream paths.
    escalated: bool = False


class ErrorResponse(BaseModel):
    detail: str
