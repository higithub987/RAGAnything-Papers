"""Document-lifecycle <-> container-registry glue tests (Step 2).

Verifies the wiring that connects uploads/deletes to the registry, without the
parse pipeline (Milvus/LLM):
  (a) record_task_containers writes a task's pending containers on doc_id resolution,
  (b) upload validates/seeds the container choice (parse monkeypatched, upload_dir tmp),
  (c) whole-database delete forgets membership (get_rag stubbed),
  (d) list enrichment maps registry membership onto tasks.
"""

from datetime import datetime, timezone

import pytest

pytest.importorskip("lightrag")  # importing rag_manager/documents pulls lightrag

from api import container_store, rag_manager, task_store  # noqa: E402
from api.config import settings  # noqa: E402
from api.container_store import ContainerRegistry  # noqa: E402
from api.models import DocumentTask, TaskStatus  # noqa: E402
from api.routes import documents as documents_module  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _use_registry(tmp_path, monkeypatch) -> ContainerRegistry:
    reg = ContainerRegistry(working_dir=str(tmp_path))
    monkeypatch.setattr(container_store, "_registry", reg)
    return reg


def _pending_task(container_ids) -> DocumentTask:
    return DocumentTask(
        task_id="t1",
        file_name="d.pdf",
        status=TaskStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        container_ids=container_ids,
    )


# --- (a) record_task_containers -----------------------------------------
def test_record_task_containers_writes_membership(tmp_path, monkeypatch):
    reg = _use_registry(tmp_path, monkeypatch)
    a = reg.create("A")
    b = reg.create("B")

    rag_manager.record_task_containers(_pending_task([a.id, b.id]), "doc-1")

    assert reg.containers_for_document("doc-1") == sorted([a.id, b.id])


def test_record_task_containers_all_is_noop(tmp_path, monkeypatch):
    reg = _use_registry(tmp_path, monkeypatch)
    rag_manager.record_task_containers(_pending_task([]), "doc-1")
    assert reg.containers_for_document("doc-1") == []


# --- (b) upload validates + seeds the container choice -------------------
def test_upload_seeds_and_validates_container(tmp_path, monkeypatch):
    reg = _use_registry(tmp_path, monkeypatch)
    cid = reg.create("Legal").id
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(documents_module, "process_document_task", _noop)

    app = FastAPI()
    app.include_router(documents_module.router)
    client = TestClient(app)
    upload = {"file": ("d.pdf", b"data", "application/pdf")}

    # A real container is seeded onto the task.
    resp = client.post("/documents", files=upload, data={"container_id": cid})
    assert resp.status_code == 202
    assert resp.json()["container_ids"] == [cid]

    # Unknown container -> 400.
    resp = client.post("/documents", files=upload, data={"container_id": "nope"})
    assert resp.status_code == 400

    # "all" -> the virtual All view: no specific container.
    resp = client.post("/documents", files=upload, data={"container_id": "all"})
    assert resp.status_code == 202
    assert resp.json()["container_ids"] == []

    # Omitted entirely -> also All.
    resp = client.post("/documents", files=upload)
    assert resp.status_code == 202
    assert resp.json()["container_ids"] == []


# --- (c) whole-database delete forgets membership -----------------------
def test_delete_forgets_membership(tmp_path, monkeypatch):
    reg = _use_registry(tmp_path, monkeypatch)
    cid = reg.create("Legal").id
    reg.add_document(cid, "doc-1")

    task_store.delete_task("tk1")
    task_store.create_task("tk1", "d.pdf")
    task_store.set_doc_id("tk1", "doc-1")

    class _Result:
        status = "success"
        message = ""

    class _DocStatus:
        async def get_by_id(self, doc_id):
            return {}  # falsy -> _cleanup_upload_artifacts is a no-op

    class _LightRAG:
        doc_status = _DocStatus()

        async def adelete_by_doc_id(self, doc_id):
            return _Result()

    class _Rag:
        lightrag = _LightRAG()

    monkeypatch.setattr(documents_module, "get_rag", lambda: _Rag())

    app = FastAPI()
    app.include_router(documents_module.router)
    client = TestClient(app)

    resp = client.delete("/documents/tk1")
    assert resp.status_code == 204
    assert reg.containers_for_document("doc-1") == []


# --- (d) list enrichment maps registry membership -----------------------
def test_enrich_containers_reflects_registry(tmp_path, monkeypatch):
    reg = _use_registry(tmp_path, monkeypatch)
    a = reg.create("A")
    reg.add_document(a.id, "doc-1")

    task_store.delete_task("tk-enrich")
    task_store.create_task("tk-enrich", "d.pdf")
    task_store.set_doc_id("tk-enrich", "doc-1")

    enriched = documents_module._enrich_containers(task_store.get_task("tk-enrich"))
    assert enriched.container_ids == [a.id]
