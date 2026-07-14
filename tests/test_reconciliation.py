import pytest

from raganything.base import DocStatus
from raganything.reconciliation import (
    find_orphaned_doc_ids,
    reconcile_orphaned_documents,
)


class FakeDeletionResult:
    def __init__(self, status="success", message="ok"):
        self.status = status
        self.message = message


class FakeDocStatusStorage:
    def __init__(self, stale_docs):
        self._stale_docs = stale_docs
        self.requested_statuses = None

    async def get_docs_by_statuses(self, statuses):
        self.requested_statuses = statuses
        return dict(self._stale_docs)


class FakeLightRAG:
    def __init__(self, stale_docs, delete_outcomes=None, delete_errors=None):
        self.doc_status = FakeDocStatusStorage(stale_docs)
        self.delete_calls = []
        self._delete_outcomes = delete_outcomes or {}
        self._delete_errors = delete_errors or {}

    async def adelete_by_doc_id(self, doc_id):
        self.delete_calls.append(doc_id)
        if doc_id in self._delete_errors:
            raise self._delete_errors[doc_id]
        return self._delete_outcomes.get(doc_id, FakeDeletionResult("success"))


@pytest.mark.asyncio
async def test_find_orphaned_doc_ids_filters_by_status():
    lightrag = FakeLightRAG({"doc-1": {"status": DocStatus.HANDLING}})

    orphans = await find_orphaned_doc_ids(lightrag)

    assert orphans == ["doc-1"]
    assert lightrag.doc_status.requested_statuses == [
        DocStatus.HANDLING,
        DocStatus.PROCESSING,
        DocStatus.PENDING,
    ]


@pytest.mark.asyncio
async def test_reconcile_orphaned_documents_calls_delete_for_each():
    lightrag = FakeLightRAG(
        {"doc-1": {"status": DocStatus.HANDLING}, "doc-2": {"status": DocStatus.PROCESSING}}
    )

    outcomes = await reconcile_orphaned_documents(lightrag)

    assert sorted(lightrag.delete_calls) == ["doc-1", "doc-2"]
    assert outcomes == {"doc-1": "success", "doc-2": "success"}


@pytest.mark.asyncio
async def test_reconcile_orphaned_documents_continues_after_one_failure():
    lightrag = FakeLightRAG(
        {"doc-1": {"status": DocStatus.HANDLING}, "doc-2": {"status": DocStatus.PROCESSING}},
        delete_errors={"doc-1": RuntimeError("boom")},
    )

    outcomes = await reconcile_orphaned_documents(lightrag)

    assert sorted(lightrag.delete_calls) == ["doc-1", "doc-2"]
    assert outcomes["doc-1"] == "error: boom"
    assert outcomes["doc-2"] == "success"


@pytest.mark.asyncio
async def test_reconcile_orphaned_documents_noop_when_nothing_stale():
    lightrag = FakeLightRAG({})

    outcomes = await reconcile_orphaned_documents(lightrag)

    assert outcomes == {}
    assert lightrag.delete_calls == []
