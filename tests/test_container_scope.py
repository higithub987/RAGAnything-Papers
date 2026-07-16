"""Unit tests for the container query-scope resolver (Step 3, derive strategy).

Hermetic: build a ContainerRegistry pointed at tmp_path and write a tmp
kv_store_doc_status.json, then resolve_scope(..., registry=reg, working_dir=tmp).
No Milvus / LightRAG / LLM.
"""

import json

from api.container_scope import GRAPH_FIELD_SEP, resolve_scope
from api.container_store import ContainerRegistry


def _registry(tmp_path) -> ContainerRegistry:
    return ContainerRegistry(working_dir=str(tmp_path))


def _write_doc_status(tmp_path, mapping: dict) -> None:
    """mapping: {doc_id: file_path} -> a minimal doc_status JSON."""
    data = {doc_id: {"file_path": fp} for doc_id, fp in mapping.items()}
    (tmp_path / "kv_store_doc_status.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


def test_all_when_no_selection(tmp_path):
    reg = _registry(tmp_path)
    for selection in (None, [], [""]):
        scope = resolve_scope(selection, registry=reg, working_dir=str(tmp_path))
        assert scope.is_all is True
        assert scope.chunk_filter_expr is None
        assert scope.matches_nothing is False
        assert scope.file_path_in_scope("anything") is True
        assert scope.file_path_in_scope(None) is True


def test_single_container(tmp_path):
    reg = _registry(tmp_path)
    _write_doc_status(tmp_path, {"doc-a": "ua_a.pdf", "doc-b": "ub_b.pdf"})
    c = reg.create("Legal")
    reg.add_document(c.id, "doc-a")

    scope = resolve_scope([c.id], registry=reg, working_dir=str(tmp_path))
    assert scope.is_all is False
    assert scope.doc_ids == frozenset({"doc-a"})
    assert scope.file_paths == frozenset({"ua_a.pdf"})
    assert scope.chunk_filter_expr == 'full_doc_id in ["doc-a"]'
    assert scope.file_path_in_scope("ua_a.pdf") is True
    assert scope.file_path_in_scope("ub_b.pdf") is False


def test_multi_valued_file_path_in_scope(tmp_path):
    reg = _registry(tmp_path)
    _write_doc_status(tmp_path, {"doc-a": "ua_a.pdf"})
    c = reg.create("A")
    reg.add_document(c.id, "doc-a")

    scope = resolve_scope([c.id], registry=reg, working_dir=str(tmp_path))
    # A merged entity's file_path joins several sources with GRAPH_FIELD_SEP.
    assert scope.file_path_in_scope(f"ux_x.pdf{GRAPH_FIELD_SEP}ua_a.pdf") is True
    assert scope.file_path_in_scope(f"ux_x.pdf{GRAPH_FIELD_SEP}uy_y.pdf") is False


def test_multi_container_union(tmp_path):
    reg = _registry(tmp_path)
    _write_doc_status(
        tmp_path, {"doc-a": "ua_a.pdf", "doc-b": "ub_b.pdf", "doc-c": "uc_c.pdf"}
    )
    a = reg.create("A")
    b = reg.create("B")
    reg.add_document(a.id, "doc-a")
    reg.add_document(b.id, "doc-b")

    scope = resolve_scope([a.id, b.id], registry=reg, working_dir=str(tmp_path))
    assert scope.doc_ids == frozenset({"doc-a", "doc-b"})
    assert scope.file_paths == frozenset({"ua_a.pdf", "ub_b.pdf"})
    # Sorted for determinism.
    assert scope.chunk_filter_expr == 'full_doc_id in ["doc-a", "doc-b"]'


def test_empty_container_matches_nothing(tmp_path):
    reg = _registry(tmp_path)
    _write_doc_status(tmp_path, {"doc-a": "ua_a.pdf"})
    c = reg.create("Empty")

    scope = resolve_scope([c.id], registry=reg, working_dir=str(tmp_path))
    assert scope.is_all is False
    assert scope.matches_nothing is True
    assert scope.doc_ids == frozenset()
    assert scope.file_paths == frozenset()
    # A valid but never-matching filter (real doc ids look like "doc-...").
    assert "doc-a" not in scope.chunk_filter_expr
    assert scope.file_path_in_scope("ua_a.pdf") is False


def test_unknown_container_ignored(tmp_path):
    reg = _registry(tmp_path)
    _write_doc_status(tmp_path, {"doc-a": "ua_a.pdf"})
    scope = resolve_scope(["does-not-exist"], registry=reg, working_dir=str(tmp_path))
    assert scope.matches_nothing is True
    assert scope.doc_ids == frozenset()


def test_doc_missing_from_doc_status(tmp_path):
    # doc-a is in the registry but not (yet) in doc_status: it still filters chunks,
    # just contributes no file_path for graph pruning.
    reg = _registry(tmp_path)
    _write_doc_status(tmp_path, {})
    c = reg.create("A")
    reg.add_document(c.id, "doc-a")

    scope = resolve_scope([c.id], registry=reg, working_dir=str(tmp_path))
    assert scope.doc_ids == frozenset({"doc-a"})
    assert scope.chunk_filter_expr == 'full_doc_id in ["doc-a"]'
    assert scope.file_paths == frozenset()
    assert scope.file_path_in_scope("ua_a.pdf") is False


def test_missing_doc_status_file_tolerated(tmp_path):
    reg = _registry(tmp_path)
    c = reg.create("A")
    reg.add_document(c.id, "doc-a")
    # No doc_status file written at all.
    scope = resolve_scope([c.id], registry=reg, working_dir=str(tmp_path))
    assert scope.doc_ids == frozenset({"doc-a"})
    assert scope.file_paths == frozenset()


def test_corrupt_doc_status_tolerated(tmp_path):
    reg = _registry(tmp_path)
    (tmp_path / "kv_store_doc_status.json").write_text("{ not json", encoding="utf-8")
    c = reg.create("A")
    reg.add_document(c.id, "doc-a")

    scope = resolve_scope([c.id], registry=reg, working_dir=str(tmp_path))
    assert scope.doc_ids == frozenset({"doc-a"})
    assert scope.file_paths == frozenset()
