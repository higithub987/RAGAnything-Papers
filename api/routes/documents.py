import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from ..config import settings
from ..models import DocumentRelatedness, DocumentTask
from ..rag_manager import get_rag, load_existing_documents, process_document_task
from ..relatedness import compute_relatedness
from ..task_store import create_task, delete_task, get_task, list_tasks

router = APIRouter(prefix="/documents", tags=["documents"])


def _cleanup_upload_artifacts(doc_status: Optional[dict[str, Any]]) -> None:
    """Remove the raw upload and parser output tied to a deleted document.

    adelete_by_doc_id only clears LightRAG's own storages (chunks/entities/
    vectors/doc_status) -- it never touches settings.upload_dir or
    settings.output_dir, so without this those files leak on every delete.
    file_path (the on-disk upload basename) is read from doc_status before
    deletion; the output subfolder name is reconstructed the same way
    raganything's parser computed it (RAGAnythingPDFProcessor._create_unique_
    output_subdir: f"{stem}_{md5(resolved_absolute_path)[:8]}"), since that's
    the only place the mapping from upload path to output folder exists.
    """
    if not doc_status:
        return
    file_path = doc_status.get("file_path")
    if not file_path:
        return

    upload_file = (Path(settings.upload_dir) / file_path).resolve()
    upload_file.unlink(missing_ok=True)

    path_hash = hashlib.md5(str(upload_file).encode()).hexdigest()[:8]
    output_subdir = Path(settings.output_dir) / f"{upload_file.stem}_{path_hash}"
    if output_subdir.is_dir():
        shutil.rmtree(output_subdir, ignore_errors=True)


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
        lightrag = get_rag().lightrag
        doc_status = await lightrag.doc_status.get_by_id(task.doc_id)
        result = await lightrag.adelete_by_doc_id(task.doc_id)
        if result.status not in ("success", "not_found"):
            raise HTTPException(
                status_code=409,
                detail=f"Could not delete document {task.doc_id!r}: {result.message}",
            )
        _cleanup_upload_artifacts(doc_status)

    delete_task(task_id)
