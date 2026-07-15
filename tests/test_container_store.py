"""Unit tests for the container registry (Part 1 of the container feature).

Covers CRUD, many-to-many membership, idempotency, partial/whole removal,
delete/forget, a persistence roundtrip across a fresh registry instance, and
tolerance of a missing/corrupt containers.json.

Fully hermetic: every test builds a ContainerRegistry pointed at pytest's
tmp_path, so nothing touches the real working_dir and there is no shared global
state between tests.
"""

from api.container_store import ContainerRegistry


def _registry(tmp_path) -> ContainerRegistry:
    return ContainerRegistry(working_dir=str(tmp_path))


def test_crud_roundtrip(tmp_path):
    reg = _registry(tmp_path)
    assert reg.list_containers() == []

    a = reg.create("Legal")
    b = reg.create("Research")
    assert a.id != b.id
    assert a.created_at is not None
    assert reg.get(a.id).name == "Legal"
    # Oldest-first == creation order.
    assert [c.name for c in reg.list_containers()] == ["Legal", "Research"]
    assert reg.get("does-not-exist") is None


def test_rename_preserves_membership(tmp_path):
    reg = _registry(tmp_path)
    c = reg.create("Old")
    reg.add_document(c.id, "doc-a")

    renamed = reg.rename(c.id, "New")
    assert renamed.name == "New"
    assert reg.documents_in_container(c.id) == ["doc-a"]
    assert reg.rename("missing", "x") is None


def test_many_to_many_membership(tmp_path):
    reg = _registry(tmp_path)
    a = reg.create("A")
    b = reg.create("B")
    reg.add_document(a.id, "doc-1")
    reg.add_document(b.id, "doc-1")

    assert reg.containers_for_document("doc-1") == sorted([a.id, b.id])
    assert reg.documents_in_container(a.id) == ["doc-1"]
    assert reg.documents_in_container(b.id) == ["doc-1"]


def test_add_is_idempotent_and_guards_missing_container(tmp_path):
    reg = _registry(tmp_path)
    a = reg.create("A")

    assert reg.add_document(a.id, "doc-1") is True
    assert reg.add_document(a.id, "doc-1") is True  # idempotent
    assert reg.documents_in_container(a.id) == ["doc-1"]  # no duplicate
    assert reg.add_document("missing", "doc-1") is False


def test_partial_remove_keeps_other_membership(tmp_path):
    reg = _registry(tmp_path)
    a = reg.create("A")
    b = reg.create("B")
    reg.add_document(a.id, "doc-1")
    reg.add_document(b.id, "doc-1")

    assert reg.remove_document(a.id, "doc-1") is True
    assert reg.containers_for_document("doc-1") == [b.id]
    assert reg.documents_in_container(a.id) == []
    assert reg.remove_document(a.id, "doc-1") is False  # already gone


def test_delete_container_drops_membership(tmp_path):
    reg = _registry(tmp_path)
    a = reg.create("A")
    b = reg.create("B")
    reg.add_document(a.id, "doc-1")
    reg.add_document(b.id, "doc-1")

    assert reg.delete(a.id) is True
    assert reg.get(a.id) is None
    assert reg.containers_for_document("doc-1") == [b.id]  # still in B
    assert reg.delete(a.id) is False  # already gone


def test_forget_document_clears_everywhere(tmp_path):
    reg = _registry(tmp_path)
    a = reg.create("A")
    b = reg.create("B")
    reg.add_document(a.id, "doc-1")
    reg.add_document(b.id, "doc-1")

    reg.forget_document("doc-1")
    assert reg.containers_for_document("doc-1") == []
    assert reg.documents_in_container(a.id) == []
    assert reg.documents_in_container(b.id) == []


def test_set_document_containers_replaces_membership(tmp_path):
    reg = _registry(tmp_path)
    a = reg.create("A")
    b = reg.create("B")
    c = reg.create("C")
    reg.add_document(a.id, "doc-1")

    reg.set_document_containers("doc-1", [b.id, c.id, "unknown"])
    assert reg.containers_for_document("doc-1") == sorted([b.id, c.id])
    assert reg.documents_in_container(a.id) == []  # moved out of A

    reg.set_document_containers("doc-1", [])  # clear
    assert reg.containers_for_document("doc-1") == []


def test_persistence_roundtrip(tmp_path):
    reg = _registry(tmp_path)
    a = reg.create("A")
    b = reg.create("B")
    reg.add_document(a.id, "doc-1")
    reg.add_document(a.id, "doc-2")
    reg.add_document(b.id, "doc-1")

    # A brand-new instance on the same dir must reload everything, including the
    # reverse index rebuilt from the persisted forward map.
    reloaded = ContainerRegistry(working_dir=str(tmp_path))
    assert {c.name for c in reloaded.list_containers()} == {"A", "B"}
    assert reloaded.documents_in_container(a.id) == ["doc-1", "doc-2"]
    assert reloaded.containers_for_document("doc-1") == sorted([a.id, b.id])
    assert reloaded.containers_for_document("doc-2") == [a.id]


def test_missing_file_is_empty(tmp_path):
    reg = ContainerRegistry(working_dir=str(tmp_path))
    assert reg.list_containers() == []
    assert reg.containers_for_document("doc-1") == []


def test_corrupt_file_is_tolerated(tmp_path):
    (tmp_path / "containers.json").write_text("{ not valid json", encoding="utf-8")

    reg = ContainerRegistry(working_dir=str(tmp_path))  # must not raise
    assert reg.list_containers() == []
    # ...and the registry is still usable after a corrupt load.
    c = reg.create("A")
    assert reg.get(c.id).name == "A"
