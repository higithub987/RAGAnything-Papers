"""Tests for container query scoping (Step 4): the pure filter helpers, the
monkeypatch installer, and the route wiring.

The helper/installer tests are hermetic (no Milvus/LLM; the installer runs against
a stub module). The route-wiring tests import the query router (which pulls
LightRAG) with `get_rag` mocked, so no real retrieval happens. End-to-end scoping
(does a scoped query actually return only in-scope content) is verified manually
against the running stack, per the plan.
"""

import asyncio
import types

import pytest

from api import container_store, lightrag_scope
from api.container_scope import GRAPH_FIELD_SEP, ContainerScope
from api.container_store import ContainerRegistry
from api.lightrag_scope import (
    query_scope,
    scope_chunks,
    scope_edges,
    scope_nodes,
)

ALL = ContainerScope(
    is_all=True, doc_ids=frozenset(), file_paths=frozenset(), chunk_filter_expr=None
)


def _scope(file_paths, doc_ids=("doc-x",)) -> ContainerScope:
    return ContainerScope(
        is_all=False,
        doc_ids=frozenset(doc_ids),
        file_paths=frozenset(file_paths),
        chunk_filter_expr='full_doc_id in ["doc-x"]',
    )


# --- pure filter helpers -------------------------------------------------------
def test_scope_chunks_keeps_in_scope():
    scope = _scope({"a.pdf"})
    chunks = [{"file_path": "a.pdf", "content": "x"}, {"file_path": "b.pdf"}]
    assert scope_chunks(chunks, scope) == [{"file_path": "a.pdf", "content": "x"}]


def test_helpers_passthrough_on_none_and_all():
    chunks = [{"file_path": "b.pdf"}]
    nodes = [{"entity_name": "E", "file_path": "b.pdf"}]
    for scope in (None, ALL):
        assert scope_chunks(chunks, scope) == chunks
        assert scope_nodes(nodes, scope) == nodes
        assert scope_edges(nodes, scope) == nodes


def test_scope_nodes_handles_multivalued_and_missing():
    scope = _scope({"a.pdf"})
    nodes = [
        {
            "entity_name": "E1",
            "file_path": f"b.pdf{GRAPH_FIELD_SEP}a.pdf",
        },  # merged, in
        {"entity_name": "E2", "file_path": "b.pdf"},  # out of scope
        {"entity_name": "E3", "file_path": None},  # missing provenance -> out
    ]
    assert [n["entity_name"] for n in scope_nodes(nodes, scope)] == ["E1"]


def test_scope_edges_filters_by_file_path():
    scope = _scope({"a.pdf"})
    edges = [
        {"src_id": "A", "tgt_id": "B", "file_path": "a.pdf"},
        {"src_id": "C", "tgt_id": "D", "file_path": "b.pdf"},
    ]
    assert scope_edges(edges, scope) == [
        {"src_id": "A", "tgt_id": "B", "file_path": "a.pdf"}
    ]


def test_helpers_empty_inputs():
    scope = _scope({"a.pdf"})
    assert scope_chunks([], scope) == []
    assert scope_nodes(None, scope) is None


# --- installer / wrapper -------------------------------------------------------
def _stub_operate() -> types.ModuleType:
    async def _get_vector_context(*a, **k):
        return [{"file_path": "a.pdf"}, {"file_path": "b.pdf"}]

    async def _get_node_data(*a, **k):
        return (
            [
                {"entity_name": "E", "file_path": "a.pdf"},
                {"entity_name": "F", "file_path": "b.pdf"},
            ],
            [{"src_id": "E", "tgt_id": "F", "file_path": "b.pdf"}],
        )

    async def _get_edge_data(*a, **k):
        return ([], [])

    m = types.ModuleType("fake_operate")
    m._get_vector_context = _get_vector_context
    m._get_node_data = _get_node_data
    m._get_edge_data = _get_edge_data
    return m


def test_patch_module_wraps_all_targets_and_is_idempotent():
    m = _stub_operate()
    wrapped = lightrag_scope._patch_module(m)
    assert set(wrapped) == set(lightrag_scope._TARGETS)
    assert m._get_vector_context.__container_scope_wrapped__ is True
    # Idempotent: a second pass does not re-wrap (same function object).
    fn = m._get_vector_context
    lightrag_scope._patch_module(m)
    assert m._get_vector_context is fn


def test_patch_module_skips_missing_target_without_raising():
    m = types.ModuleType("fake_operate_partial")

    async def _get_vector_context(*a, **k):
        return []

    m._get_vector_context = _get_vector_context
    # no _get_node_data / _get_edge_data present
    wrapped = lightrag_scope._patch_module(m)
    assert wrapped == ["_get_vector_context"]


def test_wrapped_chunk_fn_filters_via_contextvar():
    m = _stub_operate()
    lightrag_scope._patch_module(m)

    async def _call():
        query_scope.set(_scope({"a.pdf"}))
        return await m._get_vector_context()

    assert asyncio.run(_call()) == [{"file_path": "a.pdf"}]


def test_wrapped_node_fn_filters_both_lists():
    m = _stub_operate()
    lightrag_scope._patch_module(m)

    async def _call():
        query_scope.set(_scope({"a.pdf"}))
        return await m._get_node_data()

    nodes, relations = asyncio.run(_call())
    assert [n["entity_name"] for n in nodes] == ["E"]
    assert relations == []  # the only relation was file_path b.pdf -> dropped


def test_wrapped_fn_passthrough_when_no_scope():
    m = _stub_operate()
    lightrag_scope._patch_module(m)
    # No query_scope set -> default None -> unscoped passthrough.
    assert asyncio.run(m._get_vector_context()) == [
        {"file_path": "a.pdf"},
        {"file_path": "b.pdf"},
    ]


def test_wrapper_fails_open_on_bad_result_shape():
    # A transform error (e.g. unexpected result shape) must not crash the query.
    async def _orig(*a, **k):
        return "not-a-tuple"

    wrapped = lightrag_scope._make_wrapper(_orig, lightrag_scope._transform_node_data)

    async def _call():
        query_scope.set(_scope({"a.pdf"}))
        return await wrapped()

    assert asyncio.run(_call()) == "not-a-tuple"  # returned unscoped, no raise


# --- route wiring (get_rag mocked; LightRAG import guarded) ---------------------
def _query_client(monkeypatch, tmp_path, rag):
    pytest.importorskip("lightrag")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routes import query as query_module

    reg = ContainerRegistry(working_dir=str(tmp_path))
    monkeypatch.setattr(container_store, "_registry", reg)
    monkeypatch.setattr(query_module, "get_rag", lambda: rag)
    app = FastAPI()
    app.include_router(query_module.router)
    return TestClient(app), reg


def test_route_matches_nothing_short_circuits(monkeypatch, tmp_path):
    called = {"aquery": False}

    class _Rag:
        async def aquery(self, *a, **k):
            called["aquery"] = True
            return "SHOULD NOT RUN"

    client, reg = _query_client(monkeypatch, tmp_path, _Rag())
    empty = reg.create("Empty")

    resp = client.post("/query", json={"query": "hi", "container_ids": [empty.id]})
    assert resp.status_code == 200
    assert "no documents" in resp.json()["answer"].lower()
    assert called["aquery"] is False  # never hit the model


def test_route_sets_scope_for_selected_container(monkeypatch, tmp_path):
    captured = {}

    class _Rag:
        async def aquery(self, query, mode="hybrid", **k):
            captured["scope"] = query_scope.get()
            return "answer"

    client, reg = _query_client(monkeypatch, tmp_path, _Rag())
    c = reg.create("Legal")
    reg.add_document(c.id, "doc-a")

    resp = client.post("/query", json={"query": "hi", "container_ids": [c.id]})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "answer"
    scope = captured["scope"]
    assert scope is not None and scope.is_all is False
    assert "doc-a" in scope.doc_ids


def test_route_all_when_no_container_ids(monkeypatch, tmp_path):
    captured = {}

    class _Rag:
        async def aquery(self, query, mode="hybrid", **k):
            captured["scope"] = query_scope.get()
            return "answer"

    client, _ = _query_client(monkeypatch, tmp_path, _Rag())
    resp = client.post("/query", json={"query": "hi"})
    assert resp.status_code == 200
    assert captured["scope"].is_all is True
