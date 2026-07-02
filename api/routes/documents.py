import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from ..config import settings
from ..models import DocumentRelatedness, DocumentTask
from ..rag_manager import get_rag, load_existing_documents, process_document_task
from ..relatedness import compute_relatedness
from ..task_store import create_task, delete_task, get_task, list_tasks

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentTask, status_code=202)
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    upload_path = Path(settings.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    task_id = str(uuid.uuid4())
    dest = upload_path / f"{task_id}_{file.filename}"
    dest.write_bytes(await file.read())

    task = create_task(task_id, file.filename, on_disk_name=dest.name)
    background_tasks.add_task(process_document_task, task_id, str(dest))
    return task


@router.get("", response_model=list[DocumentTask])
def list_documents():
    load_existing_documents()
    return list_tasks()


@router.get("/relatedness", response_model=list[DocumentRelatedness])
async def get_relatedness():
    return await compute_relatedness()


@router.get("/{task_id}", response_model=DocumentTask)
def get_document_status(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")
    return task


@router.delete("/{task_id}", status_code=204)
async def delete_document(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    if task.doc_id is not None:
        result = await get_rag().lightrag.adelete_by_doc_id(task.doc_id)
        if result.status not in ("success", "not_found"):
            raise HTTPException(
                status_code=409,
                detail=f"Could not delete document {task.doc_id!r}: {result.message}",
            )

    delete_task(task_id)
