"""Route tests for the containers API (Step 2).

Hermetic: a minimal FastAPI app mounts only the containers router (so main.py's
rag-init lifespan never runs), and each test gets a fresh ContainerRegistry
pointed at tmp_path via monkeypatching the module singleton. No Milvus/LLM.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import container_store
from api.container_store import ContainerRegistry
from api.routes.containers import router as containers_router


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        container_store, "_registry", ContainerRegistry(working_dir=str(tmp_path))
    )
    app = FastAPI()
    app.include_router(containers_router)
    return TestClient(app)


def test_create_and_get(client):
    resp = client.post("/containers", json={"name": "Legal"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "Legal"
    assert body["member_doc_ids"] == []

    got = client.get(f"/containers/{body['id']}")
    assert got.status_code == 200
    assert got.json()["name"] == "Legal"


def test_list_is_creation_ordered(client):
    assert client.get("/containers").json() == []
    client.post("/containers", json={"name": "A"})
    client.post("/containers", json={"name": "B"})
    assert [c["name"] for c in client.get("/containers").json()] == ["A", "B"]


def test_get_missing_is_404(client):
    assert client.get("/containers/nope").status_code == 404


def test_rename(client):
    cid = client.post("/containers", json={"name": "Old"}).json()["id"]
    resp = client.patch(f"/containers/{cid}", json={"name": "New"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New"
    assert client.patch("/containers/nope", json={"name": "X"}).status_code == 404


def test_delete(client):
    cid = client.post("/containers", json={"name": "A"}).json()["id"]
    resp = client.delete(f"/containers/{cid}")
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}
    assert client.get(f"/containers/{cid}").status_code == 404
    assert client.delete(f"/containers/{cid}").status_code == 404


def test_add_and_remove_membership(client):
    cid = client.post("/containers", json={"name": "A"}).json()["id"]

    resp = client.post(f"/containers/{cid}/documents", json={"doc_id": "doc-1"})
    assert resp.status_code == 200
    assert resp.json()["member_doc_ids"] == ["doc-1"]

    # Idempotent: adding again does not duplicate.
    resp = client.post(f"/containers/{cid}/documents", json={"doc_id": "doc-1"})
    assert resp.json()["member_doc_ids"] == ["doc-1"]

    # Missing container -> 404.
    assert (
        client.post("/containers/nope/documents", json={"doc_id": "doc-1"}).status_code
        == 404
    )

    resp = client.delete(f"/containers/{cid}/documents/doc-1")
    assert resp.status_code == 200
    assert resp.json() == {"removed": True}
    assert client.get(f"/containers/{cid}").json()["member_doc_ids"] == []

    # Removing a non-member -> 404.
    assert client.delete(f"/containers/{cid}/documents/doc-1").status_code == 404
