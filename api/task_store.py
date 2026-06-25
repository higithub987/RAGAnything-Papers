from datetime import datetime, timezone
from typing import Optional

from .models import DocumentTask, TaskStatus

# task_id -> DocumentTask; lives for the lifetime of the server process
_tasks: dict[str, DocumentTask] = {}


def create_task(task_id: str, file_name: str) -> DocumentTask:
    task = DocumentTask(
        task_id=task_id,
        file_name=file_name,
        status=TaskStatus.PROCESSING,
        created_at=datetime.now(timezone.utc),
    )
    _tasks[task_id] = task
    return task


def complete_task(task_id: str) -> None:
    if task_id in _tasks:
        _tasks[task_id].status = TaskStatus.COMPLETED
        _tasks[task_id].completed_at = datetime.now(timezone.utc)


def fail_task(task_id: str, error: str) -> None:
    if task_id in _tasks:
        _tasks[task_id].status = TaskStatus.FAILED
        _tasks[task_id].error = error
        _tasks[task_id].completed_at = datetime.now(timezone.utc)


def set_doc_id(task_id: str, doc_id: str) -> None:
    if task_id in _tasks:
        _tasks[task_id].doc_id = doc_id


def get_task(task_id: str) -> Optional[DocumentTask]:
    return _tasks.get(task_id)


def list_tasks() -> list[DocumentTask]:
    return list(_tasks.values())


def delete_task(task_id: str) -> bool:
    if task_id in _tasks:
        del _tasks[task_id]
        return True
    return False


def import_existing(task_id: str, file_name: str, status: TaskStatus,
                    created_at: datetime, completed_at: Optional[datetime],
                    error: Optional[str] = None) -> None:
    """Sync a document from LightRAG's doc_status into the task store.

    `task_id` here is the LightRAG doc_id. Upserts so changes to the backing
    doc_status/name data (e.g. a name backfill) take effect on the next call
    instead of only on the next clean process start. Skips entirely if this
    doc_id is already tracked under a different (live-upload) task_id.
    """
    existing = _tasks.get(task_id)
    if existing is not None:
        existing.file_name = file_name
        existing.status = status
        existing.error = error
        existing.completed_at = completed_at
        return

    if any(task.doc_id == task_id for task in _tasks.values()):
        return

    _tasks[task_id] = DocumentTask(
        task_id=task_id,
        file_name=file_name,
        status=status,
        error=error,
        created_at=created_at,
        completed_at=completed_at,
        doc_id=task_id,
    )
