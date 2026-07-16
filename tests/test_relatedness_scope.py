"""Tests for Step 5 relatedness/graph container scoping: the pure
`filter_relatedness_pairs` helper and the `/documents/relatedness` route wiring.

Hermetic: the helper tests build plain `DocumentRelatedness` + `ContainerScope`
objects; the route tests mount only the documents router with `compute_relatedness`
mocked and a real `ContainerRegistry` on a tmp working dir, so no Milvus/LLM/graph is
touched. End-to-end graph scoping is verified manually against the running stack.
"""

import pytest

# api.relatedness pulls rag_manager -> lightrag at import time; skip cleanly if absent.
pytest.importorskip("lightrag")

from api import container_store
from api.container_scope import ContainerScope
from api.container_store import ContainerRegistry
from api.models import DocumentRelatedness
from api.relatedness import filter_relatedness_pairs

ALL = ContainerScope(
    is_all=True, doc_ids=frozenset(), file_paths=frozenset(), chunk_filter_expr=None
)


def _scope(doc_ids) -> ContainerScope:
    docs = frozenset(doc_ids)
    return ContainerScope(
        is_all=False,
        doc_ids=docs,
        file_paths=frozenset(),
        chunk_filter_expr='full_doc_id in ["..."]',
    )


def _pair(a: str, b: str, score: float = 0.5) -> DocumentRelatedness:
    return DocumentRelatedness(
        doc_id=a,
        file_name=f"{a}.pdf",
        related_doc_id=b,
        related_file_name=f"{b}.pdf",
        score=score,
    )


# --- pure helper ---------------------------------------------------------------
def test_none_and_all_scope_passthrough():
    pairs = [_pair("a", "b"), _pair("a", "c")]
    assert filter_relatedness_pairs(pairs, None) == pairs
    assert filter_relatedness_pairs(pairs, ALL) == pairs


def test_strict_keeps_only_both_in_scope():
    pairs = [_pair("a", "b"), _pair("a", "c"), _pair("c", "d")]
    scope = _scope({"a", "b", "c"})
    out = filter_relatedness_pairs(pairs, scope)
    # a-b both in; a-c both in; c-d has d out-of-scope -> dropped.
    assert [(p.doc_id, p.related_doc_id) for p in out] == [("a", "b"), ("a", "c")]


def test_strict_drops_one_in_and_zero_in():
    pairs = [_pair("a", "x"), _pair("x", "y")]  # one endpoint in, then none in
    scope = _scope({"a", "b"})
    assert filter_relatedness_pairs(pairs, scope) == []


def test_empty_input():
    assert filter_relatedness_pairs([], _scope({"a"})) == []


# --- route wiring (compute mocked; real registry on tmp dir) -------------------
def _client(monkeypatch, tmp_path, pairs, calls=None):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routes import documents as documents_module

    reg = ContainerRegistry(working_dir=str(tmp_path))
    monkeypatch.setattr(container_store, "_registry", reg)

    async def fake_compute():
        if calls is not None:
            calls["compute"] = calls.get("compute", 0) + 1
        return pairs

    monkeypatch.setattr(documents_module, "compute_relatedness", fake_compute)
    app = FastAPI()
    app.include_router(documents_module.router)
    return TestClient(app), reg


def test_route_all_when_no_container_ids(monkeypatch, tmp_path):
    pairs = [_pair("a", "b"), _pair("c", "d")]
    client, _ = _client(monkeypatch, tmp_path, pairs)
    resp = client.get("/documents/relatedness")
    assert resp.status_code == 200
    assert len(resp.json()) == 2  # unscoped -> every pair


def test_route_scopes_to_single_container(monkeypatch, tmp_path):
    pairs = [_pair("a", "b"), _pair("a", "c"), _pair("c", "d")]
    client, reg = _client(monkeypatch, tmp_path, pairs)
    c = reg.create("Legal")
    reg.add_document(c.id, "a")
    reg.add_document(c.id, "b")
    resp = client.get("/documents/relatedness", params={"container_ids": [c.id]})
    assert resp.status_code == 200
    got = [(p["doc_id"], p["related_doc_id"]) for p in resp.json()]
    assert got == [("a", "b")]  # only the pair fully inside {a, b}


def test_route_union_across_containers(monkeypatch, tmp_path):
    pairs = [_pair("a", "b"), _pair("b", "c"), _pair("c", "d")]
    client, reg = _client(monkeypatch, tmp_path, pairs)
    c1 = reg.create("One")
    c2 = reg.create("Two")
    reg.add_document(c1.id, "a")
    reg.add_document(c1.id, "b")
    reg.add_document(c2.id, "c")
    resp = client.get(
        "/documents/relatedness", params={"container_ids": [c1.id, c2.id]}
    )
    got = [(p["doc_id"], p["related_doc_id"]) for p in resp.json()]
    # union {a, b, c}: a-b in, b-c in, c-d out (d not in the union).
    assert got == [("a", "b"), ("b", "c")]


def test_route_empty_container_short_circuits(monkeypatch, tmp_path):
    calls = {}
    client, reg = _client(monkeypatch, tmp_path, [_pair("a", "b")], calls=calls)
    empty = reg.create("Empty")
    resp = client.get("/documents/relatedness", params={"container_ids": [empty.id]})
    assert resp.status_code == 200
    assert resp.json() == []
    assert "compute" not in calls  # matches_nothing short-circuits before compute
